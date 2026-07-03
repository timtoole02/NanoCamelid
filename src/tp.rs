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

/// One tensor-parallel decode step over `hidden` (updated in place). Every
/// shard computes every layer on its slice; the two per-layer reductions are
/// in-process sums here.
pub fn tp_forward_token(
    hidden: &mut [f32],
    shards: &mut [TpShard],
    rt: &mut TpRuntime,
    pos: usize,
    options: LlamaRuntimeOptions,
) -> Result<(), String> {
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
        for (value, &residual) in hidden.iter_mut().zip(rt.residual.iter()) {
            *value += residual;
        }
    }
    Ok(())
}
