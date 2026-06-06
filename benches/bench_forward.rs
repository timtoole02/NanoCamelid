// Criterion benchmark: one decode forward pass on a tiny synthetic Q8_0 model.
//
// Exercises the full hot path (RMSNorm, quantize, Q8 matmuls, RoPE, attention, FFN) so the
// before/after effect of SDOT + multi-thread + LTO shows up end-to-end. Run with:
// `cargo bench --bench bench_forward`.
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};

use nanocamelid::inference::{forward_pass, LlamaKvCache, LlamaWorkspace};
use nanocamelid::model::{LlamaLayerWeightsQ8, LlamaModelConfig, LlamaWeights, LlamaWeightsQ8};
use nanocamelid::q4::Q4DotKernelSelector;
use nanocamelid::q8::{Q8DotKernelSelector, Q8_0Block, Q8_BLOCK_SIZE};

fn make_blocks(count: usize, seed: u64) -> Vec<Q8_0Block> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..count)
        .map(|_| {
            let mut values = [0i8; Q8_BLOCK_SIZE];
            for v in values.iter_mut() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *v = ((state >> 33) as i8).clamp(-127, 127);
            }
            Q8_0Block::from_parts(0x3C00, values) // f16 scale = 1.0
        })
        .collect()
}

fn make_w(rows: usize, cols: usize, seed: u64) -> Vec<Q8_0Block> {
    make_blocks(rows * cols / Q8_BLOCK_SIZE, seed)
}

fn tiny_config() -> LlamaModelConfig {
    let embedding_length = 128;
    let attention_head_count = 4;
    let head_dim = embedding_length / attention_head_count;
    LlamaModelConfig {
        context_length: 64,
        embedding_length,
        block_count: 4,
        feed_forward_length: 256,
        attention_head_count,
        attention_head_count_kv: attention_head_count,
        rope_freq_base: 10000.0,
        rms_norm_epsilon: 1e-5,
        vocab_size: 256,
        head_dim,
        kv_width: attention_head_count * head_dim,
    }
}

fn build_weights(config: &LlamaModelConfig) -> LlamaWeights {
    let emb = config.embedding_length;
    let ff = config.feed_forward_length;
    let kv = config.kv_width;
    let layers = (0..config.block_count)
        .map(|i| {
            let s = i as u64 + 1;
            LlamaLayerWeightsQ8 {
                attention_norm: vec![1.0; emb],
                wq: make_w(emb, emb, s * 10 + 1),
                wk: make_w(kv, emb, s * 10 + 2),
                wav: make_w(kv, emb, s * 10 + 3),
                wo: make_w(emb, emb, s * 10 + 4),
                ffn_norm: vec![1.0; emb],
                w1: make_w(ff, emb, s * 10 + 5),
                w3: make_w(ff, emb, s * 10 + 6),
                w2: make_w(emb, ff, s * 10 + 7),
            }
        })
        .collect();

    LlamaWeights::Q8_0(LlamaWeightsQ8 {
        token_embeddings: vec![0.02; config.vocab_size * emb],
        output_norm: vec![1.0; emb],
        output_projection: Some(make_w(config.vocab_size, emb, 999)),
        layers,
    })
}

fn bench(c: &mut Criterion) {
    let config = tiny_config();
    let weights = build_weights(&config);
    let selector_q8 = Q8DotKernelSelector::from_env_or_auto();
    let selector_q4 = Q4DotKernelSelector::from_env_or_auto();

    let mut cache = LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut ws = LlamaWorkspace::new(&config);

    c.bench_function("forward_pass_tiny_q8", |b| {
        b.iter(|| {
            let logits = forward_pass(
                1, 0, &config, &weights, &mut cache, &mut ws, selector_q8, selector_q4, None, None,
                None, None,
            );
            black_box(logits[0]);
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
