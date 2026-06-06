use crate::model::{LlamaModelConfig, LlamaWeights};
use crate::q8::{Q8DotKernelSelector, Q8_0Block};
use rayon::prelude::*;

pub struct LlamaKvCache {
    pub k_cache: Vec<f32>, // block_count * context_length * kv_width
    pub v_cache: Vec<f32>, // block_count * context_length * kv_width
    pub kv_width: usize,
    pub max_seq_len: usize,
}

impl LlamaKvCache {
    pub fn new(block_count: usize, max_seq_len: usize, kv_width: usize) -> Self {
        let size = block_count * max_seq_len * kv_width;
        Self {
            k_cache: vec![0.0; size],
            v_cache: vec![0.0; size],
            kv_width,
            max_seq_len,
        }
    }

    pub fn store_kv(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let layer_offset = layer * self.max_seq_len * self.kv_width;
        let pos_offset = pos * self.kv_width;
        let start = layer_offset + pos_offset;

        self.k_cache[start..start + self.kv_width].copy_from_slice(k);
        self.v_cache[start..start + self.kv_width].copy_from_slice(v);
    }

    pub fn get_k_cache(&self, layer: usize) -> &[f32] {
        let layer_offset = layer * self.max_seq_len * self.kv_width;
        &self.k_cache[layer_offset..layer_offset + self.max_seq_len * self.kv_width]
    }

    pub fn get_v_cache(&self, layer: usize) -> &[f32] {
        let layer_offset = layer * self.max_seq_len * self.kv_width;
        &self.v_cache[layer_offset..layer_offset + self.max_seq_len * self.kv_width]
    }
}

pub struct LlamaWorkspace {
    pub norm_x: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_output: Vec<f32>,
    pub attn_scores: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_gate_up: Vec<f32>,
    pub x_i8: Vec<i8>,
    pub x_scales: Vec<f32>,
    pub logits: Vec<f32>,
}

impl LlamaWorkspace {
    pub fn new(config: &LlamaModelConfig) -> Self {
        let max_size = config.vocab_size.max(config.feed_forward_length).max(config.embedding_length);
        Self {
            norm_x: vec![0.0; config.embedding_length],
            q: vec![0.0; config.embedding_length],
            k: vec![0.0; config.kv_width],
            v: vec![0.0; config.kv_width],
            attn_output: vec![0.0; config.embedding_length],
            attn_scores: vec![0.0; config.context_length],
            ffn_gate: vec![0.0; config.feed_forward_length],
            ffn_up: vec![0.0; config.feed_forward_length],
            ffn_gate_up: vec![0.0; config.feed_forward_length],
            x_i8: vec![0; max_size],
            x_scales: vec![0.0; max_size / 32 + 1],
            logits: vec![0.0; config.vocab_size],
        }
    }
}

pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], epsilon: f32) {
    let mut sum = 0.0_f32;
    for &val in x {
        sum += val * val;
    }
    let scale = 1.0_f32 / (sum / x.len() as f32 + epsilon).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

pub fn apply_rope(
    data: &mut [f32],
    pos: usize,
    head_count: usize,
    head_dim: usize,
    rope_dim: usize,
    freq_base: f32,
    scaling_factor: Option<f32>,
    original_context_len: Option<f32>,
    low_freq_factor: Option<f32>,
    high_freq_factor: Option<f32>,
) {
    for head in 0..head_count {
        let head_start = head * head_dim;
        for i in 0..(rope_dim / 2) {
            let dim0 = head_start + (i * 2);
            let dim1 = dim0 + 1;

            let mut theta = freq_base.powf(-((i * 2) as f32) / rope_dim as f32);

            // Apply Llama 3 / 3.2 scaled RoPE if config is present
            if let (Some(factor), Some(orig_len), Some(low), Some(high)) =
                (scaling_factor, original_context_len, low_freq_factor, high_freq_factor)
            {
                let wavelength = (2.0 * std::f32::consts::PI) / theta;
                let low_freq_wavelength = orig_len / low;
                let high_freq_wavelength = orig_len / high;
                if wavelength >= high_freq_wavelength {
                    if wavelength > low_freq_wavelength {
                        theta /= factor;
                    } else {
                        let smooth = (orig_len / wavelength - low) / (high - low);
                        theta = ((1.0 - smooth) * theta / factor) + (smooth * theta);
                    }
                }
            }

            let angle = pos as f32 * theta;
            let (sin, cos) = angle.sin_cos();

            let x0 = data[dim0];
            let x1 = data[dim1];

            data[dim0] = x0 * cos - x1 * sin;
            data[dim1] = x0 * sin + x1 * cos;
        }
    }
}

pub fn quantize_f32_to_q8_0(x: &[f32], x_i8: &mut [i8], x_scales: &mut [f32]) {
    let num_blocks = x.len() / 32;
    for b in 0..num_blocks {
        let chunk = &x[b * 32..(b + 1) * 32];
        let mut max_abs = 0.0_f32;
        for &val in chunk {
            let abs = val.abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }
        let scale = max_abs / 127.0;
        x_scales[b] = scale;
        if scale > 0.0 {
            let inv_scale = 1.0 / scale;
            for i in 0..32 {
                let q = (chunk[i] * inv_scale).round();
                x_i8[b * 32 + i] = q.max(-127.0).min(127.0) as i8;
            }
        } else {
            for i in 0..32 {
                x_i8[b * 32 + i] = 0;
            }
        }
    }
}

pub fn matmul_q8_0(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q8_0Block],
    _rows: usize,
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    let blocks_per_row = cols / 32;
    match selector.selected {
        crate::q8::Q8DotKernel::Scalar => {
            out.par_iter_mut().enumerate().for_each(|(r, out_val)| {
                let mut sum = 0.0_f32;
                let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
                for b in 0..blocks_per_row {
                    let w_block = &w_row[b];
                    let x_block_vals = &x_i8[b * 32..(b + 1) * 32];
                    let dot_val = crate::q8::dot_i8_scalar(w_block.values(), x_block_vals);
                    sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
                }
                *out_val = sum;
            });
        }
        crate::q8::Q8DotKernel::Neon => {
            out.par_iter_mut().enumerate().for_each(|(r, out_val)| {
                let mut sum = 0.0_f32;
                let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
                for b in 0..blocks_per_row {
                    let w_block = &w_row[b];
                    let x_block_vals = &x_i8[b * 32..(b + 1) * 32];
                    let dot_val = crate::q8::dot_i8_neon(w_block.values(), x_block_vals);
                    sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
                }
                *out_val = sum;
            });
        }
        crate::q8::Q8DotKernel::Sdot => {
            out.par_iter_mut().enumerate().for_each(|(r, out_val)| {
                let mut sum = 0.0_f32;
                let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
                for b in 0..blocks_per_row {
                    let w_block = &w_row[b];
                    let x_block_vals = &x_i8[b * 32..(b + 1) * 32];
                    let dot_val = crate::q8::dot_i8_sdot(w_block.values(), x_block_vals);
                    sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
                }
                *out_val = sum;
            });
        }
    }
}

pub fn matmul_f32(out: &mut [f32], x: &[f32], w: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let mut sum = 0.0_f32;
        let w_row = &w[r * cols..(r + 1) * cols];
        for c in 0..cols {
            sum += x[c] * w_row[c];
        }
        out[r] = sum;
    }
}

pub fn silu(x: &mut [f32]) {
    for val in x.iter_mut() {
        let sigmoid = 1.0 / (1.0 + (-*val).exp());
        *val = *val * sigmoid;
    }
}

pub fn softmax(x: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &val in x.iter() {
        if val > max_val {
            max_val = val;
        }
    }

    let mut sum = 0.0;
    for val in x.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }

    let inv_sum = 1.0 / sum;
    for val in x.iter_mut() {
        *val *= inv_sum;
    }
}

/// Embedding lookup for `token_id`. Produces the initial hidden-state vector.
///
/// Lifted verbatim from the head of `forward_pass`. Only `node0` (or a single-node run)
/// calls this; the resulting `Vec<f32>` is what travels between pipeline stages.
pub fn embed(token_id: usize, config: &LlamaModelConfig, weights: &LlamaWeights) -> Vec<f32> {
    let emb_start = token_id * config.embedding_length;
    let mut x = vec![0.0; config.embedding_length];
    let table = match weights {
        LlamaWeights::Q8_0(w) => &w.token_embeddings,
        LlamaWeights::Q4_0(w) => &w.token_embeddings,
    };
    x.copy_from_slice(&table[emb_start..emb_start + config.embedding_length]);
    x
}

/// Run the transformer layers in the range `[start_layer, end_layer)` in place on `x`.
///
/// IMPORTANT (pipeline sharding): `weights.layers` and `cache` are addressed with a
/// **local** index `0..(end_layer - start_layer)`. On a sharded node the weights vector
/// holds only this node's layers and the cache is sized for only this node's layers, so
/// local indexing is correct. For a whole-model run (`start_layer == 0`,
/// `end_layer == block_count`) local index == global index, so this is byte-identical to
/// the original monolithic loop.
///
/// Every reduction here keeps the exact summation order of the original `forward_pass`.
pub fn forward_layers(
    x: &mut [f32],
    start_layer: usize,
    end_layer: usize,
    pos: usize,
    config: &LlamaModelConfig,
    weights: &LlamaWeights,
    cache: &mut LlamaKvCache,
    ws: &mut LlamaWorkspace,
    selector_q8: Q8DotKernelSelector,
    selector_q4: crate::q4::Q4DotKernelSelector,
    rope_scaling_factor: Option<f32>,
    rope_scaling_original_context_length: Option<f32>,
    rope_scaling_low_freq_factor: Option<f32>,
    rope_scaling_high_freq_factor: Option<f32>,
) {
    let n_local = end_layer - start_layer;
    let mut residual = vec![0.0f32; config.embedding_length];

    match weights {
        LlamaWeights::Q8_0(w) => {
            for local in 0..n_local {
                let layer = &w.layers[local];

                // --- Attention block ---
                residual.copy_from_slice(x);

                rms_norm(&mut ws.norm_x, x, &layer.attention_norm, config.rms_norm_epsilon);
                quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);

                matmul_q8_0(&mut ws.q, &ws.x_i8, &ws.x_scales, &layer.wq, config.embedding_length, config.embedding_length, selector_q8);
                matmul_q8_0(&mut ws.k, &ws.x_i8, &ws.x_scales, &layer.wk, config.kv_width, config.embedding_length, selector_q8);
                matmul_q8_0(&mut ws.v, &ws.x_i8, &ws.x_scales, &layer.wav, config.kv_width, config.embedding_length, selector_q8);

                apply_rope(
                    &mut ws.q,
                    pos,
                    config.attention_head_count,
                    config.head_dim,
                    config.head_dim,
                    config.rope_freq_base,
                    rope_scaling_factor,
                    rope_scaling_original_context_length,
                    rope_scaling_low_freq_factor,
                    rope_scaling_high_freq_factor,
                );
                apply_rope(
                    &mut ws.k,
                    pos,
                    config.attention_head_count_kv,
                    config.head_dim,
                    config.head_dim,
                    config.rope_freq_base,
                    rope_scaling_factor,
                    rope_scaling_original_context_length,
                    rope_scaling_low_freq_factor,
                    rope_scaling_high_freq_factor,
                );

                cache.store_kv(local, pos, &ws.k, &ws.v);
                let k_cache = cache.get_k_cache(local);
                let v_cache = cache.get_v_cache(local);

                let scale = 1.0 / (config.head_dim as f32).sqrt();

                ws.attn_output.fill(0.0);
                let kv_mul = config.attention_head_count / config.attention_head_count_kv;

                for h in 0..config.attention_head_count {
                    let kv_h = h / kv_mul;
                    let q_head = &ws.q[h * config.head_dim..(h + 1) * config.head_dim];

                    for p in 0..=pos {
                        let k_head = &k_cache[p * cache.kv_width + kv_h * config.head_dim .. p * cache.kv_width + (kv_h + 1) * config.head_dim];

                        let mut score = 0.0;
                        for i in 0..config.head_dim {
                            score += q_head[i] * k_head[i];
                        }
                        ws.attn_scores[p] = score * scale;
                    }

                    softmax(&mut ws.attn_scores[0..=pos]);

                    let out_head = &mut ws.attn_output[h * config.head_dim..(h + 1) * config.head_dim];
                    for p in 0..=pos {
                        let v_head = &v_cache[p * cache.kv_width + kv_h * config.head_dim .. p * cache.kv_width + (kv_h + 1) * config.head_dim];
                        let weight = ws.attn_scores[p];
                        for i in 0..config.head_dim {
                            out_head[i] += weight * v_head[i];
                        }
                    }
                }

                quantize_f32_to_q8_0(&ws.attn_output, &mut ws.x_i8, &mut ws.x_scales);
                matmul_q8_0(x, &ws.x_i8, &ws.x_scales, &layer.wo, config.embedding_length, config.embedding_length, selector_q8);

                for i in 0..config.embedding_length {
                    x[i] += residual[i];
                }

                // --- FFN block ---
                residual.copy_from_slice(x);

                rms_norm(&mut ws.norm_x, x, &layer.ffn_norm, config.rms_norm_epsilon);
                quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);

                matmul_q8_0(&mut ws.ffn_gate, &ws.x_i8, &ws.x_scales, &layer.w1, config.feed_forward_length, config.embedding_length, selector_q8);
                matmul_q8_0(&mut ws.ffn_up, &ws.x_i8, &ws.x_scales, &layer.w3, config.feed_forward_length, config.embedding_length, selector_q8);

                silu(&mut ws.ffn_gate);

                for i in 0..config.feed_forward_length {
                    ws.ffn_gate_up[i] = ws.ffn_gate[i] * ws.ffn_up[i];
                }

                quantize_f32_to_q8_0(&ws.ffn_gate_up, &mut ws.x_i8, &mut ws.x_scales);
                matmul_q8_0(x, &ws.x_i8, &ws.x_scales, &layer.w2, config.embedding_length, config.feed_forward_length, selector_q8);

                for i in 0..config.embedding_length {
                    x[i] += residual[i];
                }
            }
        }
        LlamaWeights::Q4_0(w) => {
            for local in 0..n_local {
                let layer = &w.layers[local];

                // --- Attention block ---
                residual.copy_from_slice(x);

                rms_norm(&mut ws.norm_x, x, &layer.attention_norm, config.rms_norm_epsilon);
                quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);

                crate::q4::matmul_q4_0(&mut ws.q, &ws.x_i8, &ws.x_scales, &layer.wq, config.embedding_length, config.embedding_length, selector_q4);
                crate::q4::matmul_q4_0(&mut ws.k, &ws.x_i8, &ws.x_scales, &layer.wk, config.kv_width, config.embedding_length, selector_q4);
                crate::q4::matmul_q4_0(&mut ws.v, &ws.x_i8, &ws.x_scales, &layer.wav, config.kv_width, config.embedding_length, selector_q4);

                apply_rope(
                    &mut ws.q,
                    pos,
                    config.attention_head_count,
                    config.head_dim,
                    config.head_dim,
                    config.rope_freq_base,
                    rope_scaling_factor,
                    rope_scaling_original_context_length,
                    rope_scaling_low_freq_factor,
                    rope_scaling_high_freq_factor,
                );
                apply_rope(
                    &mut ws.k,
                    pos,
                    config.attention_head_count_kv,
                    config.head_dim,
                    config.head_dim,
                    config.rope_freq_base,
                    rope_scaling_factor,
                    rope_scaling_original_context_length,
                    rope_scaling_low_freq_factor,
                    rope_scaling_high_freq_factor,
                );

                cache.store_kv(local, pos, &ws.k, &ws.v);
                let k_cache = cache.get_k_cache(local);
                let v_cache = cache.get_v_cache(local);

                let scale = 1.0 / (config.head_dim as f32).sqrt();

                ws.attn_output.fill(0.0);
                let kv_mul = config.attention_head_count / config.attention_head_count_kv;

                for h in 0..config.attention_head_count {
                    let kv_h = h / kv_mul;
                    let q_head = &ws.q[h * config.head_dim..(h + 1) * config.head_dim];

                    for p in 0..=pos {
                        let k_head = &k_cache[p * cache.kv_width + kv_h * config.head_dim .. p * cache.kv_width + (kv_h + 1) * config.head_dim];

                        let mut score = 0.0;
                        for i in 0..config.head_dim {
                            score += q_head[i] * k_head[i];
                        }
                        ws.attn_scores[p] = score * scale;
                    }

                    softmax(&mut ws.attn_scores[0..=pos]);

                    let out_head = &mut ws.attn_output[h * config.head_dim..(h + 1) * config.head_dim];
                    for p in 0..=pos {
                        let v_head = &v_cache[p * cache.kv_width + kv_h * config.head_dim .. p * cache.kv_width + (kv_h + 1) * config.head_dim];
                        let weight = ws.attn_scores[p];
                        for i in 0..config.head_dim {
                            out_head[i] += weight * v_head[i];
                        }
                    }
                }

                quantize_f32_to_q8_0(&ws.attn_output, &mut ws.x_i8, &mut ws.x_scales);
                crate::q4::matmul_q4_0(x, &ws.x_i8, &ws.x_scales, &layer.wo, config.embedding_length, config.embedding_length, selector_q4);

                for i in 0..config.embedding_length {
                    x[i] += residual[i];
                }

                // --- FFN block ---
                residual.copy_from_slice(x);

                rms_norm(&mut ws.norm_x, x, &layer.ffn_norm, config.rms_norm_epsilon);
                quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);

                crate::q4::matmul_q4_0(&mut ws.ffn_gate, &ws.x_i8, &ws.x_scales, &layer.w1, config.feed_forward_length, config.embedding_length, selector_q4);
                crate::q4::matmul_q4_0(&mut ws.ffn_up, &ws.x_i8, &ws.x_scales, &layer.w3, config.feed_forward_length, config.embedding_length, selector_q4);

                silu(&mut ws.ffn_gate);

                for i in 0..config.feed_forward_length {
                    ws.ffn_gate_up[i] = ws.ffn_gate[i] * ws.ffn_up[i];
                }

                quantize_f32_to_q8_0(&ws.ffn_gate_up, &mut ws.x_i8, &mut ws.x_scales);
                crate::q4::matmul_q4_0(x, &ws.x_i8, &ws.x_scales, &layer.w2, config.embedding_length, config.feed_forward_length, selector_q4);

                for i in 0..config.embedding_length {
                    x[i] += residual[i];
                }
            }
        }
    }
}

/// Final RMSNorm + output projection -> logits. Runs only on the tail node (or as the
/// last step of a single-node run). Returns a borrow of `ws.logits`.
pub fn finalize<'a>(
    x: &[f32],
    config: &LlamaModelConfig,
    weights: &LlamaWeights,
    ws: &'a mut LlamaWorkspace,
    selector_q8: Q8DotKernelSelector,
    selector_q4: crate::q4::Q4DotKernelSelector,
) -> &'a [f32] {
    match weights {
        LlamaWeights::Q8_0(w) => {
            rms_norm(&mut ws.norm_x, x, &w.output_norm, config.rms_norm_epsilon);
            if let Some(out_proj) = &w.output_projection {
                quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);
                matmul_q8_0(&mut ws.logits, &ws.x_i8, &ws.x_scales, out_proj, config.vocab_size, config.embedding_length, selector_q8);
            } else {
                matmul_f32(&mut ws.logits, &ws.norm_x, &w.token_embeddings, config.vocab_size, config.embedding_length);
            }
        }
        LlamaWeights::Q4_0(w) => {
            rms_norm(&mut ws.norm_x, x, &w.output_norm, config.rms_norm_epsilon);
            if let Some(out_proj) = &w.output_projection {
                quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);
                crate::q4::matmul_q4_0(&mut ws.logits, &ws.x_i8, &ws.x_scales, out_proj, config.vocab_size, config.embedding_length, selector_q4);
            } else {
                matmul_f32(&mut ws.logits, &ws.norm_x, &w.token_embeddings, config.vocab_size, config.embedding_length);
            }
        }
    }
    &ws.logits
}

/// Single-node forward pass: `embed` + all layers + `finalize`. Thin wrapper preserved so
/// existing callers are byte-identical to the pre-refactor implementation.
pub fn forward_pass<'a>(
    token_id: usize,
    pos: usize,
    config: &LlamaModelConfig,
    weights: &LlamaWeights,
    cache: &mut LlamaKvCache,
    ws: &'a mut LlamaWorkspace,
    selector_q8: Q8DotKernelSelector,
    selector_q4: crate::q4::Q4DotKernelSelector,
    rope_scaling_factor: Option<f32>,
    rope_scaling_original_context_length: Option<f32>,
    rope_scaling_low_freq_factor: Option<f32>,
    rope_scaling_high_freq_factor: Option<f32>,
) -> &'a [f32] {
    let mut x = embed(token_id, config, weights);
    forward_layers(
        &mut x,
        0,
        config.block_count,
        pos,
        config,
        weights,
        cache,
        ws,
        selector_q8,
        selector_q4,
        rope_scaling_factor,
        rope_scaling_original_context_length,
        rope_scaling_low_freq_factor,
        rope_scaling_high_freq_factor,
    );
    finalize(&x, config, weights, ws, selector_q8, selector_q4)
}

pub fn sample_logits(logits: &[f32], temperature: f32) -> usize {
    if temperature <= 0.0 {
        // Greedy sampling: argmax
        let mut max_val = logits[0];
        let mut max_idx = 0;
        for (i, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        max_idx
    } else {
        // Temperature sampling
        let mut scaled_probs = vec![0.0_f32; logits.len()];
        let mut max_logit = logits[0];
        for &val in logits.iter() {
            if val > max_logit {
                max_logit = val;
            }
        }

        let mut sum = 0.0;
        for (i, &val) in logits.iter().enumerate() {
            let p = ((val - max_logit) / temperature).exp();
            scaled_probs[i] = p;
            sum += p;
        }

        // Draw sample
        let mut r = rand_simple() * sum;
        for (i, &p) in scaled_probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
        logits.len() - 1
    }
}

// Simple deterministic pseudo-random number generator for zero-dependency sampling
static mut RNG_STATE: u64 = 0x123456789abcdef0;
fn rand_simple() -> f32 {
    unsafe {
        RNG_STATE = RNG_STATE.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((RNG_STATE >> 32) as u32) as f32;
        x / (u32::MAX as f32)
    }
}
