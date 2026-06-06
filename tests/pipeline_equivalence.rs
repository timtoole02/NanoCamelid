//! Bit-identical equivalence proof for the pipeline-parallel split.
//!
//! Builds a tiny synthetic Q8_0 model, runs the original single-node `forward_pass` as the
//! reference, then runs the same sequence through three layer shards (mirroring exactly what
//! `LlamaWeights::load_range` produces per node: locally-indexed layers, embeddings only on
//! the head shard, output tensors only on the tail shard), with the hidden state round-tripped
//! through the little-endian wire encoding between shards.
//!
//! Logits must match the reference **bit for bit** (`f32::to_bits`), and greedy token
//! sequences must be identical — across the prompt AND multiple generated steps, so the
//! sharded KV caches are exercised at several positions.

use nanocamelid::distributed::config::auto_split;
use nanocamelid::distributed::frame::{read_message, write_message, Message};
use nanocamelid::inference::{
    embed, finalize, forward_layers, forward_pass, sample_logits, LlamaKvCache, LlamaWorkspace,
};
use nanocamelid::model::{LlamaLayerWeightsQ8, LlamaModelConfig, LlamaWeights, LlamaWeightsQ8};
use nanocamelid::q4::Q4DotKernelSelector;
use nanocamelid::q8::{Q8DotKernelSelector, Q8_0Block, Q8_BLOCK_SIZE};

fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn make_blocks(count: usize, seed: u64) -> Vec<Q8_0Block> {
    let mut state = seed;
    // A few exact f16 scales for variety: 0.5, 1.0, 1.5
    const SCALES: [u16; 3] = [0x3800, 0x3C00, 0x3E00];
    (0..count)
        .map(|_| {
            let scale_bits = SCALES[(lcg(&mut state) % 3) as usize];
            let mut values = [0i8; Q8_BLOCK_SIZE];
            for v in values.iter_mut() {
                *v = ((lcg(&mut state) >> 33) as i8).clamp(-127, 127);
            }
            Q8_0Block::from_parts(scale_bits, values)
        })
        .collect()
}

fn make_w(rows: usize, cols: usize, seed: u64) -> Vec<Q8_0Block> {
    make_blocks(rows * cols / Q8_BLOCK_SIZE, seed)
}

fn make_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| ((lcg(&mut state) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
        .collect()
}

fn tiny_config() -> LlamaModelConfig {
    let embedding_length = 64;
    let attention_head_count = 2;
    let head_dim = embedding_length / attention_head_count;
    LlamaModelConfig {
        context_length: 32,
        embedding_length,
        block_count: 5,
        feed_forward_length: 128,
        attention_head_count,
        attention_head_count_kv: attention_head_count,
        rope_freq_base: 10000.0,
        rms_norm_epsilon: 1e-5,
        vocab_size: 96,
        head_dim,
        kv_width: attention_head_count * head_dim,
    }
}

/// Deterministic per-layer weights: shard builds must produce values identical to the full
/// model's, exactly as two nodes reading the same GGUF file would.
fn build_layer(config: &LlamaModelConfig, layer_idx: usize) -> LlamaLayerWeightsQ8 {
    let emb = config.embedding_length;
    let ff = config.feed_forward_length;
    let kv = config.kv_width;
    let s = (layer_idx as u64 + 1) * 1000;
    LlamaLayerWeightsQ8 {
        attention_norm: make_f32(emb, s + 1).iter().map(|v| 1.0 + v * 0.1).collect(),
        wq: make_w(emb, emb, s + 2),
        wk: make_w(kv, emb, s + 3),
        wav: make_w(kv, emb, s + 4),
        wo: make_w(emb, emb, s + 5),
        ffn_norm: make_f32(emb, s + 6).iter().map(|v| 1.0 + v * 0.1).collect(),
        w1: make_w(ff, emb, s + 7),
        w3: make_w(ff, emb, s + 8),
        w2: make_w(emb, ff, s + 9),
    }
}

fn token_embeddings(config: &LlamaModelConfig) -> Vec<f32> {
    make_f32(config.vocab_size * config.embedding_length, 77)
}

fn output_norm(config: &LlamaModelConfig) -> Vec<f32> {
    make_f32(config.embedding_length, 88).iter().map(|v| 1.0 + v * 0.1).collect()
}

fn output_projection(config: &LlamaModelConfig) -> Vec<Q8_0Block> {
    make_w(config.vocab_size, config.embedding_length, 99)
}

fn build_full(config: &LlamaModelConfig) -> LlamaWeights {
    LlamaWeights::Q8_0(LlamaWeightsQ8 {
        token_embeddings: token_embeddings(config),
        output_norm: output_norm(config),
        output_projection: Some(output_projection(config)),
        layers: (0..config.block_count).map(|i| build_layer(config, i)).collect(),
    })
}

/// Mirror of what `LlamaWeights::load_range(.., start, end, need_embeddings, need_output)`
/// materializes for a pipeline node.
fn build_shard(
    config: &LlamaModelConfig,
    start: usize,
    end: usize,
    need_embeddings: bool,
    need_output: bool,
) -> LlamaWeights {
    LlamaWeights::Q8_0(LlamaWeightsQ8 {
        token_embeddings: if need_embeddings { token_embeddings(config) } else { Vec::new() },
        output_norm: if need_output { output_norm(config) } else { Vec::new() },
        output_projection: if need_output { Some(output_projection(config)) } else { None },
        layers: (start..end).map(|i| build_layer(config, i)).collect(),
    })
}

/// Simulate the TCP hop: hidden state serialized to LE bytes and back, like frame.rs does.
fn wire_roundtrip(x: Vec<f32>) -> Vec<f32> {
    x.iter().map(|v| f32::from_le_bytes(v.to_le_bytes())).collect()
}

const PROMPT: [usize; 5] = [3, 1, 4, 15, 9];
const GEN_STEPS: usize = 6;

#[test]
fn pipeline_split_is_bit_identical_to_single_node() {
    let config = tiny_config();
    let sel8 = Q8DotKernelSelector::from_env_or_auto();
    let sel4 = Q4DotKernelSelector::from_env_or_auto();

    // ---- Reference: single-node forward_pass over prompt + greedy continuation ----
    let full = build_full(&config);
    let mut ref_cache = LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut ref_ws = LlamaWorkspace::new(&config);
    let mut ref_logits: Vec<Vec<f32>> = Vec::new();
    let mut ref_tokens: Vec<usize> = Vec::new();

    let mut pos = 0;
    for &t in &PROMPT {
        let l = forward_pass(t, pos, &config, &full, &mut ref_cache, &mut ref_ws, sel8, sel4, None, None, None, None);
        ref_logits.push(l.to_vec());
        pos += 1;
    }
    for _ in 0..GEN_STEPS {
        let next = sample_logits(ref_logits.last().unwrap(), 0.0);
        ref_tokens.push(next);
        let l = forward_pass(next, pos, &config, &full, &mut ref_cache, &mut ref_ws, sel8, sel4, None, None, None, None);
        ref_logits.push(l.to_vec());
        pos += 1;
    }

    // ---- Pipeline: 3 shards, hidden state wire-round-tripped between them ----
    let ranges = auto_split(config.block_count, 3);
    assert_eq!(ranges, vec![(0, 2), (2, 4), (4, 5)]);

    let shard0 = build_shard(&config, ranges[0].0, ranges[0].1, true, false);
    let shard1 = build_shard(&config, ranges[1].0, ranges[1].1, false, false);
    let shard2 = build_shard(&config, ranges[2].0, ranges[2].1, false, true);

    let mut caches: Vec<LlamaKvCache> = ranges
        .iter()
        .map(|(s, e)| LlamaKvCache::new(e - s, config.context_length, config.kv_width))
        .collect();
    let mut ws: Vec<LlamaWorkspace> = (0..3).map(|_| LlamaWorkspace::new(&config)).collect();

    let mut step = |token: usize, pos: usize, caches: &mut Vec<LlamaKvCache>, ws: &mut Vec<LlamaWorkspace>| -> Vec<f32> {
        let mut x = embed(token, &config, &shard0);
        forward_layers(&mut x, ranges[0].0, ranges[0].1, pos, &config, &shard0, &mut caches[0], &mut ws[0], sel8, sel4, None, None, None, None);
        let mut x = wire_roundtrip(x);
        forward_layers(&mut x, ranges[1].0, ranges[1].1, pos, &config, &shard1, &mut caches[1], &mut ws[1], sel8, sel4, None, None, None, None);
        let mut x = wire_roundtrip(x);
        forward_layers(&mut x, ranges[2].0, ranges[2].1, pos, &config, &shard2, &mut caches[2], &mut ws[2], sel8, sel4, None, None, None, None);
        finalize(&x, &config, &shard2, &mut ws[2], sel8, sel4).to_vec()
    };

    let mut pipe_logits: Vec<Vec<f32>> = Vec::new();
    let mut pipe_tokens: Vec<usize> = Vec::new();

    let mut pos = 0;
    for &t in &PROMPT {
        pipe_logits.push(step(t, pos, &mut caches, &mut ws));
        pos += 1;
    }
    for _ in 0..GEN_STEPS {
        let next = sample_logits(pipe_logits.last().unwrap(), 0.0);
        pipe_tokens.push(next);
        pipe_logits.push(step(next, pos, &mut caches, &mut ws));
        pos += 1;
    }

    // ---- Bit-exact comparison ----
    assert_eq!(ref_tokens, pipe_tokens, "greedy token streams diverged");
    assert_eq!(ref_logits.len(), pipe_logits.len());
    for (pos, (r, p)) in ref_logits.iter().zip(&pipe_logits).enumerate() {
        assert_eq!(r.len(), p.len());
        for (i, (a, b)) in r.iter().zip(p).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "logit mismatch at pos {pos} index {i}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn forward_layers_empty_range_is_noop() {
    let config = tiny_config();
    let sel8 = Q8DotKernelSelector::from_env_or_auto();
    let sel4 = Q4DotKernelSelector::from_env_or_auto();
    let shard = build_shard(&config, 0, 0, true, false);
    let mut cache = LlamaKvCache::new(0, config.context_length, config.kv_width);
    let mut ws = LlamaWorkspace::new(&config);

    let x_before = embed(3, &config, &shard);
    let mut x = x_before.clone();
    forward_layers(&mut x, 2, 2, 0, &config, &shard, &mut cache, &mut ws, sel8, sel4, None, None, None, None);
    for (a, b) in x_before.iter().zip(&x) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[tokio::test]
async fn frame_roundtrip_is_bit_exact() {
    let (mut a, mut b) = tokio::io::duplex(1 << 20);
    let hidden: Vec<f32> = vec![
        0.1,
        -0.0,
        0.0,
        f32::MIN_POSITIVE,
        12345.678,
        -1e-30,
        f32::MAX,
        f32::MIN,
    ];
    let msg = Message::Forward { request_id: 42, pos: 7, hidden: hidden.clone() };
    write_message(&mut a, &msg).await.unwrap();
    match read_message(&mut b).await.unwrap() {
        Message::Forward { request_id, pos, hidden: got } => {
            assert_eq!(request_id, 42);
            assert_eq!(pos, 7);
            assert_eq!(got.len(), hidden.len());
            for (x, y) in got.iter().zip(&hidden) {
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
        other => panic!("unexpected message type {}", other.type_byte()),
    }

    // Logits reply too.
    let logits: Vec<f32> = (0..97).map(|i| (i as f32) * 0.37 - 18.0).collect();
    let msg = Message::Logits { request_id: 42, logits: logits.clone() };
    write_message(&mut b, &msg).await.unwrap();
    match read_message(&mut a).await.unwrap() {
        Message::Logits { logits: got, .. } => {
            for (x, y) in got.iter().zip(&logits) {
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
        other => panic!("unexpected message type {}", other.type_byte()),
    }
}
