//! Tensor-parallel sharding (Phase 4 MVP): dense models, greedy decode.
//!
//! Megatron-style, GQA-aware: attention is sharded by KV-head groups (each
//! shard holds its head group's full Q/K/V row slices, its own KV-cache
//! slice, and the matching wo column slice); the MLP is column-parallel for
//! gate/up and row-parallel for down. Norms and the embedding/head stay
//! replicated. Two partial-sum reductions per layer per token — in-process
//! here (the split-parity harness); a wire transport only earns a socket
//! after `cluster_tp_split_smoke` passes.
//!
//! Shards are sliced from row-major Q8_0/Q4_0 matrices at quant-block
//! granularity, so the harness must load weights with the 1x4 swizzle
//! disabled (`NANOCAMELID_Q4_SWIZZLE_1X4=0`, `NANOCAMELID_Q8_SWIZZLE_1X4=0`).

use crate::inference::{
    self, AttentionInput, LlamaKvCache, LlamaRuntimeOptions, LlamaWorkspace,
};
use crate::model::{LlamaFfnWeights, LlamaLayerWeights, LlamaModelConfig, QuantizedMatrix};

pub struct TpShard {
    pub config: LlamaModelConfig,
    pub layers: Vec<LlamaLayerWeights>,
    pub cache: LlamaKvCache,
    pub ws: LlamaWorkspace,
}

/// Scratch buffers shared across shards for one token.
pub struct TpRuntime {
    pub residual: Vec<f32>,
    pub norm_x: Vec<f32>,
}

impl TpRuntime {
    pub fn new(config: &LlamaModelConfig) -> Self {
        Self {
            residual: vec![0.0; config.embedding_length],
            norm_x: vec![0.0; config.embedding_length],
        }
    }
}

fn blocks_per_row(cols: usize) -> usize {
    cols / 32
}

/// Row slice [r0, r1) of a row-major quantized matrix.
fn slice_rows(
    matrix: &QuantizedMatrix,
    cols: usize,
    r0: usize,
    r1: usize,
) -> Result<QuantizedMatrix, String> {
    let bpr = blocks_per_row(cols);
    match matrix {
        QuantizedMatrix::Q8_0(blocks) => Ok(QuantizedMatrix::Q8_0(
            blocks[r0 * bpr..r1 * bpr].to_vec(),
        )),
        QuantizedMatrix::Q4_0(blocks) => Ok(QuantizedMatrix::Q4_0(
            blocks[r0 * bpr..r1 * bpr].to_vec(),
        )),
        _ => Err(
            "tensor parallelism requires row-major Q8_0/Q4_0 matrices (load with the 1x4 swizzle disabled)"
                .to_owned(),
        ),
    }
}

/// Column slice [c0, c1) of a row-major quantized matrix; c0/c1 must sit on
/// 32-value block boundaries.
fn slice_cols(
    matrix: &QuantizedMatrix,
    rows: usize,
    cols: usize,
    c0: usize,
    c1: usize,
) -> Result<QuantizedMatrix, String> {
    if c0 % 32 != 0 || c1 % 32 != 0 {
        return Err(format!("column slice {c0}..{c1} not block-aligned"));
    }
    let bpr = blocks_per_row(cols);
    let b0 = c0 / 32;
    let b1 = c1 / 32;
    match matrix {
        QuantizedMatrix::Q8_0(blocks) => {
            let mut out = Vec::with_capacity(rows * (b1 - b0));
            for r in 0..rows {
                out.extend_from_slice(&blocks[r * bpr + b0..r * bpr + b1]);
            }
            Ok(QuantizedMatrix::Q8_0(out))
        }
        QuantizedMatrix::Q4_0(blocks) => {
            let mut out = Vec::with_capacity(rows * (b1 - b0));
            for r in 0..rows {
                out.extend_from_slice(&blocks[r * bpr + b0..r * bpr + b1]);
            }
            Ok(QuantizedMatrix::Q4_0(out))
        }
        _ => Err(
            "tensor parallelism requires row-major Q8_0/Q4_0 matrices (load with the 1x4 swizzle disabled)"
                .to_owned(),
        ),
    }
}

fn slice_bias(bias: &Option<Vec<f32>>, r0: usize, r1: usize) -> Option<Vec<f32>> {
    bias.as_ref().map(|values| values[r0..r1].to_vec())
}

/// Build `shard_count` tensor-parallel shards from full dense weights.
pub fn build_tp_shards(
    config: &LlamaModelConfig,
    layers: &[LlamaLayerWeights],
    shard_count: usize,
) -> Result<Vec<TpShard>, String> {
    if shard_count < 2 {
        return Err("tensor parallelism needs at least 2 shards".to_owned());
    }
    if config.attention_head_count % shard_count != 0
        || config.attention_head_count_kv % shard_count != 0
    {
        return Err(format!(
            "head counts {}q/{}kv not divisible by {shard_count} shards",
            config.attention_head_count, config.attention_head_count_kv
        ));
    }
    if config.feed_forward_length % (shard_count * 32) != 0 {
        return Err(format!(
            "feed_forward_length {} not block-divisible by {shard_count} shards",
            config.feed_forward_length
        ));
    }
    if config.expert_count != 0 {
        return Err("tensor parallelism MVP covers dense models only".to_owned());
    }

    let q_heads = config.attention_head_count / shard_count;
    let kv_heads = config.attention_head_count_kv / shard_count;
    let head_dim = config.head_dim;
    let ffn = config.feed_forward_length / shard_count;
    let emb = config.embedding_length;

    let mut shards = Vec::with_capacity(shard_count);
    for s in 0..shard_count {
        let mut shard_config = config.clone();
        shard_config.attention_head_count = q_heads;
        shard_config.attention_head_count_kv = kv_heads;
        shard_config.attention_output_width = q_heads * head_dim;
        shard_config.kv_width = kv_heads * head_dim;
        shard_config.feed_forward_length = ffn;

        let q0 = s * q_heads * head_dim;
        let q1 = (s + 1) * q_heads * head_dim;
        let k0 = s * kv_heads * head_dim;
        let k1 = (s + 1) * kv_heads * head_dim;
        let f0 = s * ffn;
        let f1 = (s + 1) * ffn;

        let mut shard_layers = Vec::with_capacity(layers.len());
        for layer in layers {
            let LlamaFfnWeights::Dense { w1, w3, w2 } = &layer.ffn else {
                return Err("tensor parallelism MVP covers dense models only".to_owned());
            };
            shard_layers.push(LlamaLayerWeights {
                attention_norm: layer.attention_norm.clone(),
                wq: slice_rows(&layer.wq, emb, q0, q1)?,
                wk: slice_rows(&layer.wk, emb, k0, k1)?,
                wav: slice_rows(&layer.wav, emb, k0, k1)?,
                wq_bias: slice_bias(&layer.wq_bias, q0, q1),
                wk_bias: slice_bias(&layer.wk_bias, k0, k1),
                wav_bias: slice_bias(&layer.wav_bias, k0, k1),
                // QK-norm weights are per-head_dim and apply identically to
                // every head, so each shard keeps the full weight.
                wq_norm: layer.wq_norm.clone(),
                wk_norm: layer.wk_norm.clone(),
                wo: slice_cols(&layer.wo, emb, config.attention_output_width, q0, q1)?,
                ffn_norm: layer.ffn_norm.clone(),
                ffn: LlamaFfnWeights::Dense {
                    w1: slice_rows(w1, emb, f0, f1)?,
                    w3: slice_rows(w3, emb, f0, f1)?,
                    w2: slice_cols(w2, emb, config.feed_forward_length, f0, f1)?,
                },
            });
        }

        let cache = LlamaKvCache::new(
            shard_config.block_count,
            shard_config.context_length,
            shard_config.kv_width,
        );
        let ws = LlamaWorkspace::new(&shard_config);
        shards.push(TpShard {
            config: shard_config,
            layers: shard_layers,
            cache,
            ws,
        });
    }
    Ok(shards)
}

/// Which of the two per-layer reductions a reducer callback is serving.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReducePhase {
    Attention,
    Ffn,
}

/// One tensor-parallel decode step over `hidden` (updated in place) with
/// in-process reductions: the local shards ARE all the shards.
pub fn tp_forward_token(
    hidden: &mut [f32],
    shards: &mut [TpShard],
    rt: &mut TpRuntime,
    pos: usize,
    options: LlamaRuntimeOptions,
) -> Result<(), String> {
    tp_forward_token_reduced(hidden, shards, rt, pos, options, |_, _, _| Ok(()))
}

/// One tensor-parallel decode step where the local shards hold only part of
/// the model: after the local partial sums are accumulated into the buffer,
/// `reduce` is called with (partial, layer_idx, phase) and must leave the
/// GLOBAL sum in the buffer (e.g. by exchanging partials over TCP). The
/// summation order across shards must be fixed cluster-wide so every node
/// computes bit-identical activations.
pub fn tp_forward_token_reduced<R>(
    hidden: &mut [f32],
    shards: &mut [TpShard],
    rt: &mut TpRuntime,
    pos: usize,
    options: LlamaRuntimeOptions,
    mut reduce: R,
) -> Result<(), String>
where
    R: FnMut(&mut [f32], usize, ReducePhase) -> Result<(), String>,
{
    let block_count = shards[0].config.block_count;
    let emb = shards[0].config.embedding_length;
    let epsilon = shards[0].config.rms_norm_epsilon;

    for layer_idx in 0..block_count {
        // --- Attention ---
        rt.residual.copy_from_slice(hidden);
        inference::rms_norm(
            &mut rt.norm_x,
            hidden,
            &shards[0].layers[layer_idx].attention_norm,
            epsilon,
        );

        hidden.fill(0.0);
        for shard in shards.iter_mut() {
            let layer = &shard.layers[layer_idx];
            let cfg = &shard.config;
            inference::quantize_f32_to_q8_0(&rt.norm_x, &mut shard.ws.x_i8, &mut shard.ws.x_scales);
            inference::matmul_quantized(
                &mut shard.ws.q,
                &shard.ws.x_i8,
                &shard.ws.x_scales,
                &layer.wq,
                cfg.attention_output_width,
                emb,
                options.q8_selector,
            );
            if let Some(bias) = &layer.wq_bias {
                inference::add_bias(&mut shard.ws.q, bias);
            }
            inference::matmul_quantized(
                &mut shard.ws.k,
                &shard.ws.x_i8,
                &shard.ws.x_scales,
                &layer.wk,
                cfg.kv_width,
                emb,
                options.q8_selector,
            );
            if let Some(bias) = &layer.wk_bias {
                inference::add_bias(&mut shard.ws.k, bias);
            }
            inference::matmul_quantized(
                &mut shard.ws.v,
                &shard.ws.x_i8,
                &shard.ws.x_scales,
                &layer.wav,
                cfg.kv_width,
                emb,
                options.q8_selector,
            );
            if let Some(bias) = &layer.wav_bias {
                inference::add_bias(&mut shard.ws.v, bias);
            }

            inference::apply_rope(
                &mut shard.ws.q,
                pos,
                cfg.attention_head_count,
                cfg.head_dim,
                cfg.rope_dimension_count,
                cfg.rope_freq_base,
                options.rope_scaling,
                cfg.rope_style,
            );
            inference::apply_rope(
                &mut shard.ws.k,
                pos,
                cfg.attention_head_count_kv,
                cfg.head_dim,
                cfg.rope_dimension_count,
                cfg.rope_freq_base,
                options.rope_scaling,
                cfg.rope_style,
            );

            shard.cache.store_kv(layer_idx, pos, &shard.ws.k, &shard.ws.v);
            let scale = 1.0 / (cfg.head_dim as f32).sqrt();
            shard.ws.attn_output.fill(0.0);
            inference::apply_attention_heads(
                &mut shard.ws.attn_output,
                &mut shard.ws.attn_scores,
                AttentionInput {
                    q: &shard.ws.q,
                    k_cache: shard.cache.get_k_cache(layer_idx),
                    v_cache: shard.cache.get_v_cache(layer_idx),
                    pos,
                    head_count: cfg.attention_head_count,
                    kv_head_count: cfg.attention_head_count_kv,
                    head_dim: cfg.head_dim,
                    cache_kv_width: cfg.kv_width,
                    context_length: cfg.context_length,
                    scale,
                },
                false,
            );

            // Row-parallel wo: this shard's columns produce a full-width
            // partial that the reduction below sums.
            inference::quantize_f32_to_q8_0(
                &shard.ws.attn_output,
                &mut shard.ws.x_i8,
                &mut shard.ws.x_scales,
            );
            inference::matmul_quantized(
                &mut shard.ws.hidden,
                &shard.ws.x_i8,
                &shard.ws.x_scales,
                &layer.wo,
                emb,
                cfg.attention_output_width,
                options.q8_selector,
            );
            // all-reduce #1 (in-process)
            for (accum, &partial) in hidden.iter_mut().zip(shard.ws.hidden.iter()) {
                *accum += partial;
            }
        }
        reduce(hidden, layer_idx, ReducePhase::Attention)?;
        for (value, &residual) in hidden.iter_mut().zip(rt.residual.iter()) {
            *value += residual;
        }

        // --- MLP ---
        rt.residual.copy_from_slice(hidden);
        inference::rms_norm(
            &mut rt.norm_x,
            hidden,
            &shards[0].layers[layer_idx].ffn_norm,
            epsilon,
        );

        hidden.fill(0.0);
        for shard in shards.iter_mut() {
            let layer = &shard.layers[layer_idx];
            let cfg = &shard.config;
            let LlamaFfnWeights::Dense { w1, w3, w2 } = &layer.ffn else {
                return Err("tensor parallelism MVP covers dense models only".to_owned());
            };
            inference::quantize_f32_to_q8_0(&rt.norm_x, &mut shard.ws.x_i8, &mut shard.ws.x_scales);
            inference::matmul_quantized(
                &mut shard.ws.ffn_gate,
                &shard.ws.x_i8,
                &shard.ws.x_scales,
                w1,
                cfg.feed_forward_length,
                emb,
                options.q8_selector,
            );
            inference::matmul_quantized(
                &mut shard.ws.ffn_up,
                &shard.ws.x_i8,
                &shard.ws.x_scales,
                w3,
                cfg.feed_forward_length,
                emb,
                options.q8_selector,
            );
            inference::fused_silu_mul(&mut shard.ws.ffn_gate_up, &shard.ws.ffn_gate, &shard.ws.ffn_up);
            inference::quantize_f32_to_q8_0(
                &shard.ws.ffn_gate_up,
                &mut shard.ws.x_i8,
                &mut shard.ws.x_scales,
            );
            inference::matmul_quantized(
                &mut shard.ws.hidden,
                &shard.ws.x_i8,
                &shard.ws.x_scales,
                w2,
                emb,
                cfg.feed_forward_length,
                options.q8_selector,
            );
            // all-reduce #2 (in-process)
            for (accum, &partial) in hidden.iter_mut().zip(shard.ws.hidden.iter()) {
                *accum += partial;
            }
        }
        reduce(hidden, layer_idx, ReducePhase::Ffn)?;
        for (value, &residual) in hidden.iter_mut().zip(rt.residual.iter()) {
            *value += residual;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// P4.3: fast-kernel shards and a sharded LM head.

/// Convert a shard's row-major Q8_0/Q4_0 matrices to the swizzled 1x4 layout
/// so the fast SDOT kernels apply. Row-major matrices whose row counts are
/// not 4-aligned are left as they are. Parity runs should skip this: the
/// swizzled kernels are a different (faster) kernel class than the pinned
/// scalar reference.
pub fn swizzle_shards(shards: &mut [TpShard]) {
    for shard in shards.iter_mut() {
        let cfg_attn_out = shard.config.attention_output_width;
        let cfg_kv = shard.config.kv_width;
        let cfg_ffn = shard.config.feed_forward_length;
        let emb = shard.config.embedding_length;
        for layer in shard.layers.iter_mut() {
            swizzle_matrix(&mut layer.wq, cfg_attn_out, emb);
            swizzle_matrix(&mut layer.wk, cfg_kv, emb);
            swizzle_matrix(&mut layer.wav, cfg_kv, emb);
            swizzle_matrix(&mut layer.wo, emb, cfg_attn_out);
            if let LlamaFfnWeights::Dense { w1, w3, w2 } = &mut layer.ffn {
                swizzle_matrix(w1, cfg_ffn, emb);
                swizzle_matrix(w3, cfg_ffn, emb);
                swizzle_matrix(w2, emb, cfg_ffn);
            }
        }
    }
}

fn swizzle_matrix(matrix: &mut QuantizedMatrix, rows: usize, cols: usize) {
    if !rows.is_multiple_of(4) || !cols.is_multiple_of(32) {
        return;
    }
    let bpr = cols / 32;
    let replacement = match matrix {
        QuantizedMatrix::Q8_0(blocks) if blocks.len() == rows * bpr => {
            QuantizedMatrix::Q8_0Swizzled1x4(crate::model::Q8_0Swizzled1x4Matrix {
                swizzled_1x4: crate::q8::swizzle_q8_0_1x4(blocks, rows, bpr),
                page_aligned_1x4: None,
                rows,
                cols,
            })
        }
        QuantizedMatrix::Q4_0(blocks) if blocks.len() == rows * bpr => {
            QuantizedMatrix::Q4_0Swizzled1x4(crate::model::Q4_0Swizzled1x4Matrix {
                swizzled_1x4: crate::q8::swizzle_q4_0_1x4(blocks, rows, bpr),
                page_aligned_1x4: None,
                rows,
                cols,
            })
        }
        _ => return,
    };
    *matrix = replacement;
}

/// A row-parallel slice of the LM head: rows [row_start, row_start+rows) of
/// the vocab, quantized to the Q8-swizzled layout (the production tied-head
/// path).
pub enum TpHeadMatrix {
    Quantized(QuantizedMatrix),
    /// Dequantized rows for parity runs on tied-head models: reproduces the
    /// reference's f32 head fold bit-exactly.
    F32(Vec<f32>),
}

pub struct TpHeadShard {
    pub matrix: TpHeadMatrix,
    pub row_start: usize,
    pub rows: usize,
    pub logits: Vec<f32>,
}

/// Split a tied f32 embedding table into `shard_count` Q8-swizzled head
/// slices. Row boundaries stay 4-aligned; the last shard absorbs the
/// remainder.
pub fn build_tp_head_shards(
    embeddings: &[f32],
    vocab: usize,
    emb: usize,
    shard_count: usize,
) -> Result<Vec<TpHeadShard>, String> {
    if embeddings.len() != vocab * emb || !emb.is_multiple_of(32) {
        return Err("head shard: embedding table shape mismatch".to_owned());
    }
    let base = (vocab / shard_count) & !3; // 4-aligned slice
    let mut shards = Vec::with_capacity(shard_count);
    let mut start = 0usize;
    for s in 0..shard_count {
        let rows = if s + 1 == shard_count { vocab - start } else { base };
        let slice = &embeddings[start * emb..(start + rows) * emb];
        let row_major = crate::q8::quantize_f32_matrix_to_q8_0_blocks(slice, rows, emb);
        let matrix = if rows.is_multiple_of(4) {
            QuantizedMatrix::Q8_0Swizzled1x4(crate::model::Q8_0Swizzled1x4Matrix {
                swizzled_1x4: crate::q8::swizzle_q8_0_1x4(&row_major, rows, emb / 32),
                page_aligned_1x4: None,
                rows,
                cols: emb,
            })
        } else {
            QuantizedMatrix::Q8_0(row_major)
        };
        shards.push(TpHeadShard {
            matrix: TpHeadMatrix::Quantized(matrix),
            row_start: start,
            rows,
            logits: vec![0.0; rows],
        });
        start += rows;
    }
    Ok(shards)
}

/// Compute this node's head slice over the (already reduced) hidden state
/// and return (global_token_id, logit) of the local argmax. The caller
/// merges across nodes with strict-greater comparison and lowest row_start
/// winning ties, which reproduces the full-scan first-max rule.
pub fn head_shard_argmax(
    head: &mut TpHeadShard,
    hidden: &[f32],
    output_norm: &[f32],
    rt: &mut TpRuntime,
    ws: &mut LlamaWorkspace,
    epsilon: f32,
    options: LlamaRuntimeOptions,
) -> (u32, f32) {
    inference::rms_norm(&mut rt.norm_x, hidden, output_norm, epsilon);
    match &head.matrix {
        TpHeadMatrix::Quantized(matrix) => {
            inference::quantize_f32_to_q8_0(&rt.norm_x, &mut ws.x_i8, &mut ws.x_scales);
            inference::matmul_quantized(
                &mut head.logits,
                &ws.x_i8,
                &ws.x_scales,
                matrix,
                head.rows,
                rt.norm_x.len(),
                options.q8_selector,
            );
        }
        TpHeadMatrix::F32(rows_f32) => {
            inference::matmul_f32(
                &mut head.logits,
                &rt.norm_x,
                rows_f32,
                head.rows,
                rt.norm_x.len(),
            );
        }
    }
    let local = inference::sample_logits(&head.logits, 0.0);
    ((head.row_start + local) as u32, head.logits[local])
}

// ---------------------------------------------------------------------------
// Weighted uneven shards, loaded shard-direct from the GGUF (peak memory =
// the shard itself). Shares are per-shard KV-head counts (e.g. [2, 3, 3] on
// 8 KV heads gives a half-speed node the small slice). Q attention heads and
// the FFN width split proportionally; norms are replicated; nothing reads a
// tensor wider than its slice.

use crate::gguf::{GgufFile, GgufTensorDescriptor, GgufTensorType};
use memmap2::Mmap;
use std::path::Path;

pub struct TpShardGeometry {
    pub shard_idx: usize,
    pub q_start: usize,
    pub q_heads: usize,
    pub kv_start: usize,
    pub kv_heads: usize,
    pub ffn_start: usize,
    pub ffn_len: usize,
    pub head_row_start: usize,
    pub head_rows: usize,
}

/// Validate shares against the model config and compute shard geometry.
pub fn shard_geometry(
    config: &LlamaModelConfig,
    shares: &[usize],
    shard_idx: usize,
) -> Result<TpShardGeometry, String> {
    let kv_total = config.attention_head_count_kv;
    if shares.is_empty() || shard_idx >= shares.len() {
        return Err("bad shard index".to_owned());
    }
    if shares.iter().sum::<usize>() != kv_total {
        return Err(format!(
            "shares {shares:?} must sum to {kv_total} kv heads"
        ));
    }
    if shares.iter().any(|&s| s == 0) {
        return Err("every shard needs at least one kv head".to_owned());
    }
    let q_per_kv = config.attention_head_count / kv_total;
    for &s in shares {
        if (config.feed_forward_length * s) % (kv_total * 32) != 0 {
            return Err(format!(
                "ffn {} not 32-block divisible for share {s}/{kv_total}",
                config.feed_forward_length
            ));
        }
    }
    let kv_start: usize = shares[..shard_idx].iter().sum();
    let ffn_unit = config.feed_forward_length / kv_total;
    // Head rows: proportional to shares, 4-aligned, remainder to the last.
    let vocab = config.vocab_size;
    let mut head_starts = Vec::with_capacity(shares.len() + 1);
    let mut acc = 0usize;
    for &s in shares {
        head_starts.push(acc);
        acc += (vocab * s / kv_total) & !3;
    }
    head_starts.push(vocab);
    let head_row_start = head_starts[shard_idx];
    let head_rows = if shard_idx + 1 == shares.len() {
        vocab - head_row_start
    } else {
        head_starts[shard_idx + 1] - head_row_start
    };
    Ok(TpShardGeometry {
        shard_idx,
        q_start: kv_start * q_per_kv,
        q_heads: shares[shard_idx] * q_per_kv,
        kv_start,
        kv_heads: shares[shard_idx],
        ffn_start: kv_start * ffn_unit,
        ffn_len: shares[shard_idx] * ffn_unit,
        head_row_start,
        head_rows,
    })
}

fn find_tensor<'a>(gguf: &'a GgufFile, name: &str) -> Result<&'a GgufTensorDescriptor, String> {
    gguf.tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("tensor {name} not found"))
}

fn tensor_all_bytes<'a>(mmap: &'a Mmap, desc: &GgufTensorDescriptor) -> Result<&'a [u8], String> {
    let start = desc.absolute_offset as usize;
    let end = start + desc.n_bytes as usize;
    mmap.get(start..end)
        .ok_or_else(|| format!("tensor {} out of file bounds", desc.name))
}

fn decode_block_run(
    bytes: &[u8],
    tensor_type: GgufTensorType,
    name: &str,
) -> Result<QuantizedMatrix, String> {
    match tensor_type {
        GgufTensorType::Q4_0 => crate::q8::decode_q4_0_blocks(bytes)
            .map(QuantizedMatrix::Q4_0)
            .map_err(|e| format!("{name}: {e}")),
        GgufTensorType::Q8_0 => crate::q8::decode_q8_0_blocks(bytes)
            .map(QuantizedMatrix::Q8_0)
            .map_err(|e| format!("{name}: {e}")),
        GgufTensorType::Q6K => crate::q8::decode_q6_k_blocks(bytes)
            .map(QuantizedMatrix::Q6K)
            .map_err(|e| format!("{name}: {e}")),
        other => Err(format!(
            "{name}: shard-direct loading supports Q4_0/Q8_0/Q6_K, got {other:?}"
        )),
    }
}

/// Load rows [r0, r1) of a row-major quantized 2D tensor straight from the
/// file: a contiguous block-byte run, decoded without touching other rows.
fn load_rows_direct(
    mmap: &Mmap,
    gguf: &GgufFile,
    name: &str,
    cols: usize,
    r0: usize,
    r1: usize,
) -> Result<QuantizedMatrix, String> {
    let desc = find_tensor(gguf, name)?;
    let (block_vals, block_bytes) = desc
        .tensor_type
        .layout()
        .ok_or_else(|| format!("{name}: no layout"))?;
    let block_vals = block_vals as usize;
    if cols % block_vals != 0 {
        return Err(format!("{name}: cols {cols} not divisible by block {block_vals}"));
    }
    let bpr = cols / block_vals;
    let bytes = tensor_all_bytes(mmap, desc)?;
    let row_bytes = bpr * block_bytes as usize;
    decode_block_run(
        &bytes[r0 * row_bytes..r1 * row_bytes],
        desc.tensor_type,
        name,
    )
}

/// Load columns [c0, c1) of every row of a row-major quantized 2D tensor:
/// a per-row gather of a block-byte range.
fn load_cols_direct(
    mmap: &Mmap,
    gguf: &GgufFile,
    name: &str,
    rows: usize,
    cols: usize,
    c0: usize,
    c1: usize,
) -> Result<QuantizedMatrix, String> {
    let desc = find_tensor(gguf, name)?;
    let (block_vals, block_bytes) = desc
        .tensor_type
        .layout()
        .ok_or_else(|| format!("{name}: no layout"))?;
    if block_vals != 32 || c0 % 32 != 0 || c1 % 32 != 0 {
        return Err(format!("{name}: column slice not block-aligned"));
    }
    let bpr = cols / 32;
    let bb = block_bytes as usize;
    let bytes = tensor_all_bytes(mmap, desc)?;
    let (b0, b1) = (c0 / 32, c1 / 32);
    let mut gathered = Vec::with_capacity(rows * (b1 - b0) * bb);
    for r in 0..rows {
        let start = (r * bpr + b0) * bb;
        gathered.extend_from_slice(&bytes[start..start + (b1 - b0) * bb]);
    }
    decode_block_run(&gathered, desc.tensor_type, name)
}

fn load_norm(mmap: &Mmap, gguf: &GgufFile, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let desc = find_tensor(gguf, name)?;
    let bytes = tensor_all_bytes(mmap, desc)?;
    match desc.tensor_type {
        GgufTensorType::F32 => {
            if bytes.len() != len * 4 {
                return Err(format!("{name}: length mismatch"));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        GgufTensorType::F16 => {
            if bytes.len() != len * 2 {
                return Err(format!("{name}: length mismatch"));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| half_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect())
        }
        other => Err(format!("{name}: expected F32/F16 norm, got {other:?}")),
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, _) => {
            // subnormal
            let mut e = 127 - 15 + 1;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m as u32 & 0x3ff) << 13)
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, _) => (sign << 31) | 0x7fc0_0000,
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(f)
}

/// A shard-direct TP node: its layer slices, its head slice, and the
/// replicated output norm. Workers never hold the embedding table (the
/// master ships the embedded hidden state with each token).
pub struct TpNodeShard {
    pub shard: TpShard,
    pub geometry: TpShardGeometry,
    pub output_norm: Vec<f32>,
    pub head: TpHeadShard,
}

/// Build one shard directly from the GGUF. Peak memory is the shard plus a
/// single tensor slice. The head slice comes from `output.weight` when the
/// model is untied, otherwise from the same rows of the embedding table;
/// both end up Q8-swizzled unless `fast` is false (parity runs keep the
/// row-major class... the head stays as loaded either way; parity compares
/// like against like).
pub fn load_tp_shard_direct(
    path: &Path,
    config: &LlamaModelConfig,
    gguf: &GgufFile,
    shares: &[usize],
    shard_idx: usize,
    fast: bool,
) -> Result<TpNodeShard, String> {
    if config.expert_count != 0 {
        return Err("tensor parallelism covers dense models only".to_owned());
    }
    // Fail closed: the wire TP shard loader does not yet carry per-head
    // QK-norm, so it must refuse Qwen3/Gemma3 rather than silently drop it.
    if gguf.tensors.iter().any(|t| t.name.ends_with(".attn_q_norm.weight")) {
        return Err(
            "wire tensor parallelism does not yet support QK-norm architectures (qwen3/gemma3)"
                .to_owned(),
        );
    }
    let geo = shard_geometry(config, shares, shard_idx)?;
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| e.to_string())? };
    let emb = config.embedding_length;
    let head_dim = config.head_dim;

    let q0 = geo.q_start * head_dim;
    let q1 = (geo.q_start + geo.q_heads) * head_dim;
    let k0 = geo.kv_start * head_dim;
    let k1 = (geo.kv_start + geo.kv_heads) * head_dim;
    let f0 = geo.ffn_start;
    let f1 = geo.ffn_start + geo.ffn_len;

    let mut layers = Vec::with_capacity(config.block_count);
    for i in 0..config.block_count {
        layers.push(LlamaLayerWeights {
            attention_norm: load_norm(&mmap, gguf, &format!("blk.{i}.attn_norm.weight"), emb)?,
            wq: load_rows_direct(&mmap, gguf, &format!("blk.{i}.attn_q.weight"), emb, q0, q1)?,
            wk: load_rows_direct(&mmap, gguf, &format!("blk.{i}.attn_k.weight"), emb, k0, k1)?,
            wav: load_rows_direct(&mmap, gguf, &format!("blk.{i}.attn_v.weight"), emb, k0, k1)?,
            wq_bias: None,
            wk_bias: None,
            wav_bias: None,
            // QK-norm archs are rejected above; wire TP is llama-family only.
            wq_norm: None,
            wk_norm: None,
            wo: load_cols_direct(
                &mmap,
                gguf,
                &format!("blk.{i}.attn_output.weight"),
                emb,
                config.attention_output_width,
                q0,
                q1,
            )?,
            ffn_norm: load_norm(&mmap, gguf, &format!("blk.{i}.ffn_norm.weight"), emb)?,
            ffn: LlamaFfnWeights::Dense {
                w1: load_rows_direct(&mmap, gguf, &format!("blk.{i}.ffn_gate.weight"), emb, f0, f1)?,
                w3: load_rows_direct(&mmap, gguf, &format!("blk.{i}.ffn_up.weight"), emb, f0, f1)?,
                w2: load_cols_direct(
                    &mmap,
                    gguf,
                    &format!("blk.{i}.ffn_down.weight"),
                    emb,
                    config.feed_forward_length,
                    f0,
                    f1,
                )?,
            },
        });
    }

    let mut shard_config = config.clone();
    shard_config.attention_head_count = geo.q_heads;
    shard_config.attention_head_count_kv = geo.kv_heads;
    shard_config.attention_output_width = geo.q_heads * head_dim;
    shard_config.kv_width = geo.kv_heads * head_dim;
    shard_config.feed_forward_length = geo.ffn_len;

    let cache = LlamaKvCache::new(
        shard_config.block_count,
        shard_config.context_length,
        shard_config.kv_width,
    );
    let ws = LlamaWorkspace::new(&shard_config);
    let mut shard = TpShard {
        config: shard_config,
        layers,
        cache,
        ws,
    };
    if fast {
        swizzle_shards(std::slice::from_mut(&mut shard));
    }

    let output_norm = load_norm(&mmap, gguf, "output_norm.weight", emb)?;

    // Head slice: untied models slice output.weight rows; tied models slice
    // the embedding table rows and re-quantize to the fast head layout.
    let untied = gguf.tensors.iter().any(|t| t.name == "output.weight");
    let head_matrix = if untied {
        // Untied: head rows are complete rows of output.weight, so the
        // sliced matmul is bit-identical to the reference full scan.
        let mut m = load_rows_direct(
            &mmap,
            gguf,
            "output.weight",
            emb,
            geo.head_row_start,
            geo.head_row_start + geo.head_rows,
        )?;
        if fast {
            swizzle_matrix(&mut m, geo.head_rows, emb);
        }
        TpHeadMatrix::Quantized(m)
    } else {
        // Tied head: our embedding rows, dequantized. Parity runs keep them
        // f32 (matches the reference fold); fast runs requantize to the
        // production Q8-swizzled head layout.
        let table = load_rows_direct(
            &mmap,
            gguf,
            "token_embd.weight",
            emb,
            geo.head_row_start,
            geo.head_row_start + geo.head_rows,
        )?;
        let f32_rows = dequantize_matrix_f32(&table, geo.head_rows, emb)?;
        if fast {
            let row_major =
                crate::q8::quantize_f32_matrix_to_q8_0_blocks(&f32_rows, geo.head_rows, emb);
            if geo.head_rows.is_multiple_of(4) {
                TpHeadMatrix::Quantized(QuantizedMatrix::Q8_0Swizzled1x4(
                    crate::model::Q8_0Swizzled1x4Matrix {
                        swizzled_1x4: crate::q8::swizzle_q8_0_1x4(&row_major, geo.head_rows, emb / 32),
                        page_aligned_1x4: None,
                        rows: geo.head_rows,
                        cols: emb,
                    },
                ))
            } else {
                TpHeadMatrix::Quantized(QuantizedMatrix::Q8_0(row_major))
            }
        } else {
            TpHeadMatrix::F32(f32_rows)
        }
    };
    let head = TpHeadShard {
        matrix: head_matrix,
        row_start: geo.head_row_start,
        rows: geo.head_rows,
        logits: vec![0.0; geo.head_rows],
    };

    Ok(TpNodeShard {
        shard,
        geometry: geo,
        output_norm,
        head,
    })
}

fn dequantize_matrix_f32(
    matrix: &QuantizedMatrix,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    let mut out = vec![0.0_f32; rows * cols];
    match matrix {
        QuantizedMatrix::Q8_0(blocks) => {
            for (i, block) in blocks.iter().enumerate() {
                let scale = block.scale_f32();
                for (o, &v) in out[i * 32..(i + 1) * 32].iter_mut().zip(block.values()) {
                    *o = scale * v as f32;
                }
            }
        }
        QuantizedMatrix::Q4_0(blocks) => {
            for (i, block) in blocks.iter().enumerate() {
                let scale = block.scale_f32();
                let values = block.unpack_values();
                for (o, &v) in out[i * 32..(i + 1) * 32].iter_mut().zip(values.iter()) {
                    *o = scale * v as f32;
                }
            }
        }
        QuantizedMatrix::Q6K(blocks) => {
            let mut buf = [0.0_f32; 256];
            for (i, block) in blocks.iter().enumerate() {
                block.dequantize(&mut buf);
                out[i * 256..(i + 1) * 256].copy_from_slice(&buf);
            }
        }
        _ => return Err("dequantize: unsupported matrix class".to_owned()),
    }
    Ok(out)
}

/// Dequantize a full embedding table to f32 (master-side lookup table).
pub fn load_embeddings_f32(
    path: &Path,
    gguf: &GgufFile,
    vocab: usize,
    emb: usize,
) -> Result<Vec<f32>, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| e.to_string())? };
    let desc = find_tensor(gguf, "token_embd.weight")?;
    match desc.tensor_type {
        GgufTensorType::F32 | GgufTensorType::F16 => {
            load_norm(&mmap, gguf, "token_embd.weight", vocab * emb)
        }
        GgufTensorType::Q4_0 | GgufTensorType::Q8_0 | GgufTensorType::Q6K => {
            let matrix = load_rows_direct(&mmap, gguf, "token_embd.weight", emb, 0, vocab)?;
            dequantize_matrix_f32(&matrix, vocab, emb)
        }
        other => Err(format!("token_embd: unsupported type {other:?}")),
    }
}
