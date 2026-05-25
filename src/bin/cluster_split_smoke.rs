use std::{env, path::Path, process::ExitCode};

use nanocamelid::{
    gguf, inference,
    model::{self, DistributedLlamaWeights},
    q8,
};

const CLUSTER_CONTEXT_LIMIT_ENV: &str = "NANOCAMELID_CLUSTER_CONTEXT_LIMIT";
const DEFAULT_CONTEXT_LIMIT: usize = 4;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("cluster split smoke failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let Some(model_path) = args.next() else {
        print_usage();
        return Err("missing model path".to_owned());
    };
    let token_id = parse_optional_usize(args.next(), 1, "token_id")?;
    let requested_split_layer = parse_optional_usize(args.next(), 0, "split_layer")?;

    let model_path = Path::new(&model_path);
    let gguf = gguf::read_file(model_path).map_err(|err| format!("failed to read GGUF: {err}"))?;
    let mut config = model::LlamaModelConfig::from_gguf(&gguf)
        .map_err(|err| format!("failed to parse model config: {err}"))?;
    apply_context_limit(&mut config)?;
    if token_id >= config.vocab_size {
        return Err(format!(
            "token_id {token_id} exceeds vocab size {}",
            config.vocab_size
        ));
    }

    let split_layer = if requested_split_layer == 0 {
        config.block_count / 2
    } else {
        requested_split_layer
    };
    if split_layer == 0 || split_layer >= config.block_count {
        return Err(format!(
            "split_layer must be in 1..{}, got {split_layer}",
            config.block_count
        ));
    }

    println!("NanoCamelid cluster split smoke");
    println!("model: {}", model_path.display());
    println!("token_id: {token_id}");
    println!("context_length: {}", config.context_length);
    println!("layers: {}", config.block_count);
    println!("split_layer: {split_layer}");
    println!("loading full model and two partial distributed views...");

    let full_weights = model::LlamaWeights::load(model_path, &config, &gguf)
        .map_err(|err| format!("failed to load full weights: {err}"))?;
    let node0 = model::LlamaWeights::load_distributed(model_path, &config, &gguf, 0, split_layer)
        .map_err(|err| format!("failed to load node0 partial weights: {err}"))?;
    let node1 = model::LlamaWeights::load_distributed(
        model_path,
        &config,
        &gguf,
        split_layer,
        config.block_count,
    )
    .map_err(|err| format!("failed to load node1 partial weights: {err}"))?;

    let options = inference::LlamaRuntimeOptions {
        q8_selector: q8::Q8DotKernelSelector {
            requested: Some(q8::Q8DotKernel::Scalar),
            selected: q8::Q8DotKernel::Scalar,
            fallback_reason: None,
        },
        rope_scaling: inference::RopeScaling::default(),
        compute_logits: true,
    };

    let mut full_cache =
        inference::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut full_ws = inference::LlamaWorkspace::new(&config);
    inference::forward_pass(
        token_id,
        0,
        &config,
        &full_weights,
        &mut full_cache,
        &mut full_ws,
        options,
    );

    let mut node0_cache =
        inference::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut node0_ws = inference::LlamaWorkspace::new(&config);
    let token_embeddings = node0
        .token_embeddings
        .as_ref()
        .ok_or_else(|| "node0 did not load token embeddings".to_owned())?;
    inference::embed_token(token_id, &config, token_embeddings, &mut node0_ws);
    run_distributed_range(&node0, 0, &config, &mut node0_cache, &mut node0_ws, options)?;

    let mut node1_cache =
        inference::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut node1_ws = inference::LlamaWorkspace::new(&config);
    node1_ws.hidden.copy_from_slice(&node0_ws.hidden);
    run_distributed_range(&node1, 0, &config, &mut node1_cache, &mut node1_ws, options)?;

    let output_norm = node1
        .output_norm
        .as_ref()
        .ok_or_else(|| "node1 did not load output norm".to_owned())?;
    inference::compute_logits_from_hidden(
        &config,
        &full_weights.token_embeddings,
        output_norm,
        node1.output_projection.as_ref(),
        &mut node1_ws,
        options,
    );

    let hidden_delta = max_abs_delta(&full_ws.hidden, &node1_ws.hidden);
    let logit_delta = max_abs_delta(&full_ws.logits, &node1_ws.logits);
    let full_next = inference::sample_logits(&full_ws.logits, 0.0);
    let split_next = inference::sample_logits(&node1_ws.logits, 0.0);

    println!("hidden_max_abs_delta: {hidden_delta:.8}");
    println!("logit_max_abs_delta: {logit_delta:.8}");
    println!("full_next_token: {full_next}");
    println!("split_next_token: {split_next}");

    if hidden_delta > 0.0001 || logit_delta > 0.0001 || full_next != split_next {
        return Err("split execution diverged from full execution".to_owned());
    }

    println!("result: PASS");
    Ok(())
}

fn run_distributed_range(
    weights: &DistributedLlamaWeights,
    pos: usize,
    config: &model::LlamaModelConfig,
    cache: &mut inference::LlamaKvCache,
    ws: &mut inference::LlamaWorkspace,
    options: inference::LlamaRuntimeOptions,
) -> Result<(), String> {
    if weights.layers.is_empty() {
        return Err("distributed layer range is empty".to_owned());
    }
    inference::run_layer_range(
        weights.layer_start,
        &weights.layers,
        pos,
        config,
        cache,
        ws,
        options,
    );
    Ok(())
}

fn apply_context_limit(config: &mut model::LlamaModelConfig) -> Result<(), String> {
    let limit = env::var(CLUSTER_CONTEXT_LIMIT_ENV)
        .ok()
        .map(|value| parse_positive_usize(&value, CLUSTER_CONTEXT_LIMIT_ENV))
        .transpose()?
        .unwrap_or(DEFAULT_CONTEXT_LIMIT);
    if limit < config.context_length {
        config.context_length = limit;
    }
    Ok(())
}

fn parse_optional_usize(
    value: Option<String>,
    default: usize,
    name: &'static str,
) -> Result<usize, String> {
    value
        .as_deref()
        .map(|value| parse_positive_usize(value, name))
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn parse_positive_usize(value: &str, name: &'static str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{name} must be a positive integer, got {value:?}"))?;
    if parsed == 0 {
        return Err(format!("{name} must be greater than zero"));
    }
    Ok(parsed)
}

fn max_abs_delta(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}

fn print_usage() {
    println!("Usage:");
    println!(
        "  cargo run --release --bin cluster_split_smoke -- <model.gguf> [token_id] [split_layer]"
    );
    println!();
    println!("Environment:");
    println!("  {CLUSTER_CONTEXT_LIMIT_ENV}=<tokens>   Default {DEFAULT_CONTEXT_LIMIT}");
}
