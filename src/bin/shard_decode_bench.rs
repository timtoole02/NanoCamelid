// Reproduce the cluster final-role slowdown outside the pipeline: load a
// distributed layer shard, run single-token decode steps over it, and report
// ms/layer plus major-fault counts. Mirrors cluster_tcp_smoke's runtime
// options (scalar-pinned Q8 kernel) so numbers are comparable.
// Usage: shard_decode_bench <model.gguf> <start_layer> <end_layer> [tokens]

use nanocamelid::{gguf, inference, model, q8};
use std::path::Path;
use std::time::Instant;

fn majflt() -> u64 {
    std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .nth(11)
                .and_then(|v| v.parse::<u64>().ok())
        })
        .unwrap_or(0)
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("missing model path")?;
    let start: usize = args
        .next()
        .and_then(|v| v.parse().ok())
        .ok_or("missing start_layer")?;
    let end: usize = args
        .next()
        .and_then(|v| v.parse().ok())
        .ok_or("missing end_layer")?;
    let tokens: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(16);
    let with_logits = args.next().as_deref() == Some("--with-logits");

    let gguf = gguf::read_file(Path::new(&path)).map_err(|e| e.to_string())?;
    let mut config = model::LlamaModelConfig::from_gguf(&gguf).map_err(|e| e.to_string())?;
    if config.context_length > 512 {
        config.context_length = 512;
    }

    let load_started = Instant::now();
    let shard =
        model::LlamaWeights::load_distributed(Path::new(&path), &config, &gguf, start, end)?;
    let load_secs = load_started.elapsed().as_secs_f64();

    let options = inference::LlamaRuntimeOptions {
        q8_selector: q8::Q8DotKernelSelector {
            requested: Some(q8::Q8DotKernel::Scalar),
            selected: q8::Q8DotKernel::Scalar,
            fallback_reason: None,
        },
        rope_scaling: inference::RopeScaling::default(),
        compute_logits: true,
    };
    let mut cache =
        inference::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut ws = inference::LlamaWorkspace::new(&config);
    for (i, v) in ws.hidden.iter_mut().enumerate() {
        *v = ((i % 89) as f32 - 44.0) / 44.0;
    }

    // Warmup token
    inference::run_layer_range(
        start,
        &shard.layers,
        0,
        &config,
        &mut cache,
        &mut ws,
        options,
    );

    // Mirror the final worker's steady state when asked: layers, then the
    // untied-head logits pass, per token.
    let output_norm = shard
        .output_norm
        .clone()
        .unwrap_or_else(|| vec![1.0; config.embedding_length]);
    let empty_embeddings: Vec<f32> = Vec::new();

    let faults_before = majflt();
    let started = Instant::now();
    let mut layers_ms = 0.0_f64;
    let mut logits_ms = 0.0_f64;
    for pos in 1..=tokens {
        let t0 = Instant::now();
        inference::run_layer_range(
            start,
            &shard.layers,
            pos,
            &config,
            &mut cache,
            &mut ws,
            options,
        );
        layers_ms += t0.elapsed().as_secs_f64() * 1000.0;
        if with_logits {
            let t1 = Instant::now();
            inference::compute_logits_from_hidden(
                &config,
                &empty_embeddings,
                &output_norm,
                shard.output_projection.as_ref(),
                &mut ws,
                options,
            );
            logits_ms += t1.elapsed().as_secs_f64() * 1000.0;
        }
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let faults_after = majflt();

    let (spin, rayon_lock, rayon_nopool) = inference::pool_dispatch_counts();
    println!("pool_dispatch: spin {spin} rayon_lock {rayon_lock} rayon_nopool {rayon_nopool}");
    for (stage, stats) in inference::trace_snapshot().into_iter().take(16) {
        let total_ms = stats.total.as_secs_f64() * 1000.0;
        println!(
            "trace: {stage} calls {} total_ms {:.1} avg_ms {:.4}",
            stats.calls,
            total_ms,
            total_ms / stats.calls.max(1) as f64
        );
    }
    let per_token = elapsed_ms / tokens as f64;
    let per_layer = layers_ms / tokens as f64 / (end - start) as f64;
    println!(
        "json: {{\"benchmark\":\"shard-decode\",\"range\":\"{start}..{end}\",\"tokens\":{tokens},\"with_logits\":{with_logits},\"load_secs\":{load_secs:.1},\"ms_per_token\":{per_token:.1},\"ms_per_layer\":{per_layer:.2},\"logits_avg_ms\":{:.2},\"majflt_delta\":{}}}",
        logits_ms / tokens as f64,
        faults_after - faults_before
    );
    Ok(())
}
