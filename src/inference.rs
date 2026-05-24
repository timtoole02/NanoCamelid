use crate::model::{LlamaModelConfig, LlamaWeights};
use crate::q8::{Q8_0Block, Q8DotKernel, Q8DotKernelSelector};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, Default)]
pub struct RopeScaling {
    pub factor: Option<f32>,
    pub original_context_length: Option<f32>,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct LlamaRuntimeOptions {
    pub q8_selector: Q8DotKernelSelector,
    pub rope_scaling: RopeScaling,
}

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
        let max_size = config
            .vocab_size
            .max(config.feed_forward_length)
            .max(config.embedding_length);
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
    rope_scaling: RopeScaling,
) {
    for head in 0..head_count {
        let head_start = head * head_dim;
        for i in 0..(rope_dim / 2) {
            let dim0 = head_start + (i * 2);
            let dim1 = dim0 + 1;

            let mut theta = freq_base.powf(-((i * 2) as f32) / rope_dim as f32);

            // Apply Llama 3 / 3.2 scaled RoPE if config is present
            if let (Some(factor), Some(orig_len), Some(low), Some(high)) = (
                rope_scaling.factor,
                rope_scaling.original_context_length,
                rope_scaling.low_freq_factor,
                rope_scaling.high_freq_factor,
            ) {
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
                x_i8[b * 32 + i] = q.clamp(-127.0, 127.0) as i8;
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
        Q8DotKernel::Scalar => {
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
        Q8DotKernel::Neon => {
            out.par_iter_mut().enumerate().for_each(|(r, out_val)| {
                let mut sum = 0.0_f32;
                let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
                for b in 0..blocks_per_row {
                    let w_block = &w_row[b];
                    let x_block_vals = &x_i8[b * 32..(b + 1) * 32];
                    let dot_val = crate::q8::dot_i8_neon_selected(w_block.values(), x_block_vals);
                    sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
                }
                *out_val = sum;
            });
        }
        Q8DotKernel::Sdot => {
            out.par_iter_mut().enumerate().for_each(|(r, out_val)| {
                let mut sum = 0.0_f32;
                let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
                for b in 0..blocks_per_row {
                    let w_block = &w_row[b];
                    let x_block_vals = &x_i8[b * 32..(b + 1) * 32];
                    let dot_val = crate::q8::dot_i8_sdot_selected(w_block.values(), x_block_vals);
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
        *val *= sigmoid;
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

pub fn forward_pass<'a>(
    token_id: usize,
    pos: usize,
    config: &LlamaModelConfig,
    weights: &LlamaWeights,
    cache: &mut LlamaKvCache,
    ws: &'a mut LlamaWorkspace,
    options: LlamaRuntimeOptions,
) -> &'a [f32] {
    // 1. Embedding lookup
    let emb_start = token_id * config.embedding_length;
    let mut x = vec![0.0; config.embedding_length];
    x.copy_from_slice(&weights.token_embeddings[emb_start..emb_start + config.embedding_length]);

    // 2. Transformer layers
    for layer_idx in 0..config.block_count {
        let layer = &weights.layers[layer_idx];

        // --- Attention block ---
        // Save residual
        let mut residual = x.clone();

        // RMSNorm
        rms_norm(
            &mut ws.norm_x,
            &x,
            &layer.attention_norm,
            config.rms_norm_epsilon,
        );

        // Quantize normalized activations for Q8 matmul
        quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);

        // Q, K, V Projections
        matmul_q8_0(
            &mut ws.q,
            &ws.x_i8,
            &ws.x_scales,
            &layer.wq,
            config.embedding_length,
            config.embedding_length,
            options.q8_selector,
        );
        matmul_q8_0(
            &mut ws.k,
            &ws.x_i8,
            &ws.x_scales,
            &layer.wk,
            config.kv_width,
            config.embedding_length,
            options.q8_selector,
        );
        matmul_q8_0(
            &mut ws.v,
            &ws.x_i8,
            &ws.x_scales,
            &layer.wav,
            config.kv_width,
            config.embedding_length,
            options.q8_selector,
        );

        // Apply RoPE
        apply_rope(
            &mut ws.q,
            pos,
            config.attention_head_count,
            config.head_dim,
            config.rope_dimension_count,
            config.rope_freq_base,
            options.rope_scaling,
        );

        apply_rope(
            &mut ws.k,
            pos,
            config.attention_head_count_kv,
            config.head_dim,
            config.rope_dimension_count,
            config.rope_freq_base,
            options.rope_scaling,
        );

        // Store to KV Cache
        cache.store_kv(layer_idx, pos, &ws.k, &ws.v);

        // Retrieve K, V history
        let k_cache = cache.get_k_cache(layer_idx);
        let v_cache = cache.get_v_cache(layer_idx);

        // Score scale
        let scale = 1.0 / (config.head_dim as f32).sqrt();

        // Compute attention outputs per head
        ws.attn_output.fill(0.0);
        let kv_mul = config.attention_head_count / config.attention_head_count_kv;

        for h in 0..config.attention_head_count {
            let kv_h = h / kv_mul;
            let q_head = &ws.q[h * config.head_dim..(h + 1) * config.head_dim];

            // Compute scores against all positions p = 0..=pos
            for p in 0..=pos {
                let k_head = &k_cache[p * cache.kv_width + kv_h * config.head_dim
                    ..p * cache.kv_width + (kv_h + 1) * config.head_dim];

                let mut score = 0.0;
                for i in 0..config.head_dim {
                    score += q_head[i] * k_head[i];
                }
                ws.attn_scores[p] = score * scale;
            }

            // Softmax
            softmax(&mut ws.attn_scores[0..=pos]);

            // Weighted sum of V
            let out_head = &mut ws.attn_output[h * config.head_dim..(h + 1) * config.head_dim];
            for p in 0..=pos {
                let v_head = &v_cache[p * cache.kv_width + kv_h * config.head_dim
                    ..p * cache.kv_width + (kv_h + 1) * config.head_dim];
                let weight = ws.attn_scores[p];
                for i in 0..config.head_dim {
                    out_head[i] += weight * v_head[i];
                }
            }
        }

        // Projection O (wo)
        quantize_f32_to_q8_0(&ws.attn_output, &mut ws.x_i8, &mut ws.x_scales);
        matmul_q8_0(
            &mut x,
            &ws.x_i8,
            &ws.x_scales,
            &layer.wo,
            config.embedding_length,
            config.embedding_length,
            options.q8_selector,
        );

        // Residual addition
        for i in 0..config.embedding_length {
            x[i] += residual[i];
        }

        // --- FFN block ---
        residual.copy_from_slice(&x);

        // RMSNorm
        rms_norm(&mut ws.norm_x, &x, &layer.ffn_norm, config.rms_norm_epsilon);

        // Quantize
        quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);

        // Gate (w1) & Up (w3) matmuls
        matmul_q8_0(
            &mut ws.ffn_gate,
            &ws.x_i8,
            &ws.x_scales,
            &layer.w1,
            config.feed_forward_length,
            config.embedding_length,
            options.q8_selector,
        );
        matmul_q8_0(
            &mut ws.ffn_up,
            &ws.x_i8,
            &ws.x_scales,
            &layer.w3,
            config.feed_forward_length,
            config.embedding_length,
            options.q8_selector,
        );

        // SiLU activation on Gate
        silu(&mut ws.ffn_gate);

        // Element-wise product of Gate and Up
        for i in 0..config.feed_forward_length {
            ws.ffn_gate_up[i] = ws.ffn_gate[i] * ws.ffn_up[i];
        }

        // Down projection (w2)
        quantize_f32_to_q8_0(&ws.ffn_gate_up, &mut ws.x_i8, &mut ws.x_scales);
        matmul_q8_0(
            &mut x,
            &ws.x_i8,
            &ws.x_scales,
            &layer.w2,
            config.embedding_length,
            config.feed_forward_length,
            options.q8_selector,
        );

        // Residual addition
        for i in 0..config.embedding_length {
            x[i] += residual[i];
        }
    }

    // 3. Final RMSNorm
    rms_norm(
        &mut ws.norm_x,
        &x,
        &weights.output_norm,
        config.rms_norm_epsilon,
    );

    // 4. Logits Projection
    if let Some(out_proj) = &weights.output_projection {
        quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);
        matmul_q8_0(
            &mut ws.logits,
            &ws.x_i8,
            &ws.x_scales,
            out_proj,
            config.vocab_size,
            config.embedding_length,
            options.q8_selector,
        );
    } else {
        // Tied embeddings (multiply norm_x by token_embeddings transposed)
        matmul_f32(
            &mut ws.logits,
            &ws.norm_x,
            &weights.token_embeddings,
            config.vocab_size,
            config.embedding_length,
        );
    }

    &ws.logits
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

#[cfg(test)]
mod tests {
    use super::{RopeScaling, apply_rope, matmul_q8_0};
    use crate::q8::{Q8_0Block, Q8_BLOCK_SIZE, Q8DotKernel, Q8DotKernelSelector};

    fn selector(selected: Q8DotKernel) -> Q8DotKernelSelector {
        Q8DotKernelSelector {
            requested: Some(selected),
            selected,
            fallback_reason: None,
        }
    }

    fn q8_block(scale_bits: u16, seed: i16) -> Q8_0Block {
        let mut values = [0_i8; Q8_BLOCK_SIZE];
        for (idx, value) in values.iter_mut().enumerate() {
            *value = ((seed + idx as i16 * 7) % 63 - 31) as i8;
        }
        Q8_0Block::from_parts(scale_bits, values)
    }

    #[test]
    fn matmul_q8_selected_kernels_match_scalar_reference() {
        let rows = 3;
        let cols = 64;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales = [0.25, 0.5];
        let weights = [
            q8_block(0x3800, 1),
            q8_block(0x3c00, 2),
            q8_block(0x4000, 3),
            q8_block(0x4200, 4),
            q8_block(0x4400, 5),
            q8_block(0x4600, 6),
        ];

        let mut scalar = vec![0.0; rows];
        matmul_q8_0(
            &mut scalar,
            &x_i8,
            &x_scales,
            &weights,
            rows,
            cols,
            selector(Q8DotKernel::Scalar),
        );

        for kernel in [Q8DotKernel::Neon, Q8DotKernel::Sdot] {
            let mut candidate = vec![0.0; rows];
            matmul_q8_0(
                &mut candidate,
                &x_i8,
                &x_scales,
                &weights,
                rows,
                cols,
                selector(kernel),
            );
            assert_eq!(candidate, scalar, "{kernel:?} matmul diverged");
        }
    }

    #[test]
    fn apply_rope_respects_partial_rope_dimension_count() {
        let mut data = vec![1.0, 0.0, 10.0, 20.0];
        apply_rope(&mut data, 1, 1, 4, 2, 10000.0, RopeScaling::default());

        assert!((data[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((data[1] - 1.0_f32.sin()).abs() < 1e-6);
        assert_eq!(data[2], 10.0);
        assert_eq!(data[3], 20.0);
    }
}
