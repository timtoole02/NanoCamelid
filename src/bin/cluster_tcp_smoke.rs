use std::{
    env, io,
    net::{TcpListener, TcpStream},
    path::Path,
    process::ExitCode,
    time::{Duration, Instant},
};

use nanocamelid::{
    cluster, gguf, inference,
    model::{self, DistributedLlamaWeights},
    q8,
};

const CLUSTER_CONTEXT_LIMIT_ENV: &str = "NANOCAMELID_CLUSTER_CONTEXT_LIMIT";
const DEFAULT_CONTEXT_LIMIT: usize = 4;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("cluster TCP smoke failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("worker") => {
            let Some(model_path) = args.next() else {
                print_usage();
                return Err("missing worker model path".to_owned());
            };
            let bind_addr = args.next().unwrap_or_else(|| "127.0.0.1:5005".to_owned());
            let requested_split_layer = parse_optional_usize(args.next(), 0, "split_layer")?;
            run_worker(Path::new(&model_path), &bind_addr, requested_split_layer)
        }
        Some("master") => {
            let Some(model_path) = args.next() else {
                print_usage();
                return Err("missing master model path".to_owned());
            };
            let worker_addr = args.next().unwrap_or_else(|| "127.0.0.1:5005".to_owned());
            let token_id = parse_optional_usize(args.next(), 1, "token_id")?;
            let requested_split_layer = parse_optional_usize(args.next(), 0, "split_layer")?;
            let max_tokens = parse_optional_usize(args.next(), 1, "max_tokens")?;
            run_master(
                Path::new(&model_path),
                &worker_addr,
                token_id,
                requested_split_layer,
                max_tokens,
            )
        }
        Some("master-unchecked") => {
            let Some(model_path) = args.next() else {
                print_usage();
                return Err("missing master model path".to_owned());
            };
            let worker_addr = args.next().unwrap_or_else(|| "127.0.0.1:5005".to_owned());
            let token_id = parse_optional_usize(args.next(), 1, "token_id")?;
            let requested_split_layer = parse_optional_usize(args.next(), 0, "split_layer")?;
            let max_tokens = parse_optional_usize(args.next(), 1, "max_tokens")?;
            run_master_unchecked(
                Path::new(&model_path),
                &worker_addr,
                token_id,
                requested_split_layer,
                max_tokens,
            )
        }
        _ => {
            print_usage();
            Err("missing mode".to_owned())
        }
    }
}

fn run_worker(
    model_path: &Path,
    bind_addr: &str,
    requested_split_layer: usize,
) -> Result<(), String> {
    let loaded = load_cluster_model(model_path, requested_split_layer)?;
    let node1 = model::LlamaWeights::load_distributed(
        model_path,
        &loaded.config,
        &loaded.gguf,
        loaded.split_layer,
        loaded.config.block_count,
    )
    .map_err(|err| format!("failed to load worker partial weights: {err}"))?;
    let output_norm = node1
        .output_norm
        .as_ref()
        .ok_or_else(|| "worker did not load output norm".to_owned())?;
    let output_projection = node1.output_projection.as_ref();
    let output_token_embeddings = match output_projection {
        Some(_) => &[][..],
        None => node1
            .token_embeddings
            .as_ref()
            .ok_or_else(|| "worker did not load tied token embeddings".to_owned())?
            .as_slice(),
    };

    let listener =
        TcpListener::bind(bind_addr).map_err(|err| format!("failed to bind {bind_addr}: {err}"))?;
    println!("NanoCamelid cluster TCP worker");
    println!("model: {}", model_path.display());
    println!("bind_addr: {bind_addr}");
    println!("layers: {}", loaded.config.block_count);
    println!("split_layer: {}", loaded.split_layer);
    println!("waiting for one master connection...");

    let (mut stream, peer_addr) = listener
        .accept()
        .map_err(|err| format!("failed to accept master connection: {err}"))?;
    stream
        .set_nodelay(true)
        .map_err(|err| format!("failed to set TCP_NODELAY: {err}"))?;
    println!("master_connected: {peer_addr}");

    let options = runtime_options();
    let mut cache = inference::LlamaKvCache::new(
        loaded.config.block_count,
        loaded.config.context_length,
        loaded.config.kv_width,
    );
    let mut ws = inference::LlamaWorkspace::new(&loaded.config);
    let mut activations = Vec::new();
    let mut decoded_tokens = Vec::new();
    let mut worker_compute_total = Duration::ZERO;
    loop {
        let header = match cluster::recv_activation_packet(&mut stream, &mut activations) {
            Ok(header) => header,
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(format!("failed to receive activation packet: {err}")),
        };
        if activations.len() != loaded.config.embedding_length {
            return Err(format!(
                "received {} activations, expected {}",
                activations.len(),
                loaded.config.embedding_length
            ));
        }
        if header.pos as usize >= loaded.config.context_length {
            return Err(format!(
                "received pos {} outside context length {}",
                header.pos, loaded.config.context_length
            ));
        }

        ws.hidden.copy_from_slice(&activations);
        let compute_start = Instant::now();
        run_distributed_range(
            &node1,
            header.pos as usize,
            &loaded.config,
            &mut cache,
            &mut ws,
            options,
        )?;
        inference::compute_logits_from_hidden(
            &loaded.config,
            output_token_embeddings,
            output_norm,
            output_projection,
            &mut ws,
            options,
        );
        worker_compute_total += compute_start.elapsed();
        let next_token = inference::sample_logits(&ws.logits, 0.0);
        cluster::send_token_feedback(&mut stream, next_token as u32, false)
            .map_err(|err| format!("failed to send token feedback: {err}"))?;

        println!("received_pos: {}", header.pos);
        println!("received_seq_len: {}", header.seq_len);
        println!("received_float_count: {}", header.float_count);
        println!("worker_next_token: {next_token}");
        decoded_tokens.push(next_token as u32);
    }

    println!("worker_tokens: {}", decoded_tokens.len());
    println!("worker_generated_tokens: {decoded_tokens:?}");
    println!(
        "worker_compute_total_ms: {:.3}",
        worker_compute_total.as_secs_f64() * 1000.0
    );
    if !decoded_tokens.is_empty() {
        println!(
            "worker_compute_avg_ms: {:.3}",
            worker_compute_total.as_secs_f64() * 1000.0 / decoded_tokens.len() as f64
        );
    }
    println!("result: WORKER_DONE");
    Ok(())
}

fn run_master(
    model_path: &Path,
    worker_addr: &str,
    token_id: usize,
    requested_split_layer: usize,
    max_tokens: usize,
) -> Result<(), String> {
    run_master_session(
        model_path,
        worker_addr,
        token_id,
        requested_split_layer,
        max_tokens,
        true,
    )
}

fn run_master_unchecked(
    model_path: &Path,
    worker_addr: &str,
    token_id: usize,
    requested_split_layer: usize,
    max_tokens: usize,
) -> Result<(), String> {
    run_master_session(
        model_path,
        worker_addr,
        token_id,
        requested_split_layer,
        max_tokens,
        false,
    )
}

fn run_master_session(
    model_path: &Path,
    worker_addr: &str,
    token_id: usize,
    requested_split_layer: usize,
    max_tokens: usize,
    check_full_forward: bool,
) -> Result<(), String> {
    let loaded = load_cluster_model(model_path, requested_split_layer)?;
    if token_id >= loaded.config.vocab_size {
        return Err(format!(
            "token_id {token_id} exceeds vocab size {}",
            loaded.config.vocab_size
        ));
    }
    let full_weights = if check_full_forward {
        Some(
            model::LlamaWeights::load(model_path, &loaded.config, &loaded.gguf)
                .map_err(|err| format!("failed to load full weights: {err}"))?,
        )
    } else {
        None
    };
    let node0 = model::LlamaWeights::load_distributed(
        model_path,
        &loaded.config,
        &loaded.gguf,
        0,
        loaded.split_layer,
    )
    .map_err(|err| format!("failed to load master partial weights: {err}"))?;
    let token_embeddings = node0
        .token_embeddings
        .as_ref()
        .ok_or_else(|| "master did not load token embeddings".to_owned())?;

    println!("NanoCamelid cluster TCP master");
    println!("model: {}", model_path.display());
    println!("worker_addr: {worker_addr}");
    println!("token_id: {token_id}");
    println!("max_tokens: {max_tokens}");
    println!("layers: {}", loaded.config.block_count);
    println!("split_layer: {}", loaded.split_layer);
    println!("full_forward_check: {check_full_forward}");

    let options = runtime_options();
    let mut full_cache = inference::LlamaKvCache::new(
        loaded.config.block_count,
        loaded.config.context_length,
        loaded.config.kv_width,
    );
    let mut full_ws = inference::LlamaWorkspace::new(&loaded.config);
    let mut node0_cache = inference::LlamaKvCache::new(
        loaded.config.block_count,
        loaded.config.context_length,
        loaded.config.kv_width,
    );
    let mut node0_ws = inference::LlamaWorkspace::new(&loaded.config);
    let mut stream = TcpStream::connect(worker_addr)
        .map_err(|err| format!("failed to connect to worker {worker_addr}: {err}"))?;
    stream
        .set_nodelay(true)
        .map_err(|err| format!("failed to set TCP_NODELAY: {err}"))?;
    let mut current_token = token_id;
    let mut generated_tokens = Vec::new();
    let mut full_forward_total = Duration::ZERO;
    let mut master_stage_total = Duration::ZERO;
    let mut tcp_round_trip_total = Duration::ZERO;
    let decode_limit = max_tokens.min(loaded.config.context_length);

    for pos in 0..decode_limit {
        let full_next = if let Some(full_weights) = &full_weights {
            let full_start = Instant::now();
            inference::forward_pass(
                current_token,
                pos,
                &loaded.config,
                full_weights,
                &mut full_cache,
                &mut full_ws,
                options,
            );
            full_forward_total += full_start.elapsed();
            Some(inference::sample_logits(&full_ws.logits, 0.0))
        } else {
            None
        };

        let master_start = Instant::now();
        inference::embed_token(
            current_token,
            &loaded.config,
            token_embeddings,
            &mut node0_ws,
        );
        run_distributed_range(
            &node0,
            pos,
            &loaded.config,
            &mut node0_cache,
            &mut node0_ws,
            options,
        )?;
        master_stage_total += master_start.elapsed();

        let round_trip_start = Instant::now();
        cluster::send_activation_packet(
            &mut stream,
            pos as u32,
            (pos + 1) as u32,
            &node0_ws.hidden,
        )
        .map_err(|err| format!("failed to send activation packet: {err}"))?;
        let feedback = cluster::recv_token_feedback(&mut stream)
            .map_err(|err| format!("failed to receive token feedback: {err}"))?;
        let round_trip = round_trip_start.elapsed();
        tcp_round_trip_total += round_trip;

        if let Some(full_next) = full_next {
            println!(
                "token[{pos}]: input={current_token} full_next={full_next} worker_next={} round_trip_ms={:.3}",
                feedback.token_id,
                round_trip.as_secs_f64() * 1000.0
            );

            if full_next != feedback.token_id as usize {
                return Err(format!(
                    "worker token {} did not match full forward token {full_next} at generated index {pos}",
                    feedback.token_id
                ));
            }
        } else {
            println!(
                "token[{pos}]: input={current_token} worker_next={} round_trip_ms={:.3}",
                feedback.token_id,
                round_trip.as_secs_f64() * 1000.0
            );
        }

        generated_tokens.push(feedback.token_id);
        current_token = feedback.token_id as usize;
        if feedback.is_finished {
            break;
        }
    }

    println!(
        "activation_floats_sent_each_token: {}",
        node0_ws.hidden.len()
    );
    println!(
        "activation_payload_kb_each_token: {:.2}",
        std::mem::size_of_val(node0_ws.hidden.as_slice()) as f64 / 1024.0
    );
    if check_full_forward {
        println!(
            "full_forward_total_ms: {:.3}",
            full_forward_total.as_secs_f64() * 1000.0
        );
    }
    println!(
        "cluster_master_stage_total_ms: {:.3}",
        master_stage_total.as_secs_f64() * 1000.0
    );
    println!(
        "cluster_tcp_round_trip_total_ms: {:.3}",
        tcp_round_trip_total.as_secs_f64() * 1000.0
    );
    println!(
        "cluster_total_measured_ms: {:.3}",
        (master_stage_total + tcp_round_trip_total).as_secs_f64() * 1000.0
    );
    if !generated_tokens.is_empty() {
        let token_count = generated_tokens.len() as f64;
        if check_full_forward {
            println!(
                "full_forward_avg_ms: {:.3}",
                full_forward_total.as_secs_f64() * 1000.0 / token_count
            );
        }
        println!(
            "cluster_total_measured_avg_ms: {:.3}",
            (master_stage_total + tcp_round_trip_total).as_secs_f64() * 1000.0 / token_count
        );
    }
    println!("generated_tokens: {generated_tokens:?}");

    if check_full_forward {
        println!("result: PASS");
    } else {
        println!("result: PASS_UNCHECKED");
    }
    Ok(())
}

struct LoadedClusterModel {
    gguf: gguf::GgufFile,
    config: model::LlamaModelConfig,
    split_layer: usize,
}

fn load_cluster_model(
    model_path: &Path,
    requested_split_layer: usize,
) -> Result<LoadedClusterModel, String> {
    let gguf = gguf::read_file(model_path).map_err(|err| format!("failed to read GGUF: {err}"))?;
    let mut config = model::LlamaModelConfig::from_gguf(&gguf)
        .map_err(|err| format!("failed to parse model config: {err}"))?;
    apply_context_limit(&mut config)?;
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
    Ok(LoadedClusterModel {
        gguf,
        config,
        split_layer,
    })
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

fn runtime_options() -> inference::LlamaRuntimeOptions {
    inference::LlamaRuntimeOptions {
        q8_selector: q8::Q8DotKernelSelector {
            requested: Some(q8::Q8DotKernel::Scalar),
            selected: q8::Q8DotKernel::Scalar,
            fallback_reason: None,
        },
        rope_scaling: inference::RopeScaling::default(),
        compute_logits: true,
    }
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

fn print_usage() {
    println!("Usage:");
    println!(
        "  cargo run --release --bin cluster_tcp_smoke -- worker <model.gguf> [bind_addr] [split_layer]"
    );
    println!(
        "  cargo run --release --bin cluster_tcp_smoke -- master <model.gguf> [worker_addr] [token_id] [split_layer] [max_tokens]"
    );
    println!(
        "  cargo run --release --bin cluster_tcp_smoke -- master-unchecked <model.gguf> [worker_addr] [token_id] [split_layer] [max_tokens]"
    );
    println!();
    println!("Environment:");
    println!("  {CLUSTER_CONTEXT_LIMIT_ENV}=<tokens>   Default {DEFAULT_CONTEXT_LIMIT}");
}
