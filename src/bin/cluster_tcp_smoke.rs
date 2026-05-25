use std::{
    env, io,
    io::Write,
    net::{TcpListener, TcpStream},
    path::Path,
    process::ExitCode,
    time::{Duration, Instant},
};

use nanocamelid::{
    cluster, gguf, inference,
    model::{self, DistributedLlamaWeights},
    q8, tokenizer,
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
        Some("master-generate") => {
            let Some(model_path) = args.next() else {
                print_usage();
                return Err("missing master model path".to_owned());
            };
            let worker_addr = args.next().unwrap_or_else(|| "127.0.0.1:5005".to_owned());
            let Some(prompt) = args.next() else {
                print_usage();
                return Err("missing prompt".to_owned());
            };
            let requested_split_layer = parse_optional_usize(args.next(), 0, "split_layer")?;
            let max_tokens = parse_optional_usize(args.next(), 16, "max_tokens")?;
            run_master_generate(
                Path::new(&model_path),
                &worker_addr,
                &prompt,
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
    let mut batch_ws =
        inference::LlamaBatchWorkspace::new(&loaded.config, loaded.config.context_length);
    let mut activations = Vec::new();
    let mut decoded_tokens = Vec::new();
    let mut worker_compute_total = Duration::ZERO;
    loop {
        let header = match cluster::recv_activation_packet(&mut stream, &mut activations) {
            Ok(header) => header,
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(format!("failed to receive activation packet: {err}")),
        };
        let seq_len = header.seq_len as usize;
        let pos = header.pos as usize;
        let expected_floats = seq_len * loaded.config.embedding_length;
        if activations.len() != expected_floats {
            return Err(format!(
                "received {} activations, expected {}",
                activations.len(),
                expected_floats
            ));
        }
        if seq_len == 0 {
            return Err("received empty activation sequence".to_owned());
        }
        if pos + seq_len > loaded.config.context_length {
            return Err(format!(
                "received pos {} seq_len {} outside context length {}",
                header.pos, header.seq_len, loaded.config.context_length
            ));
        }

        let compute_start = Instant::now();
        if seq_len > 1 {
            batch_ws.hidden[..expected_floats].copy_from_slice(&activations);
            inference::run_layer_range_batch(
                node1.layer_start,
                &node1.layers,
                seq_len,
                pos,
                &loaded.config,
                &mut cache,
                &mut batch_ws,
                options,
            );
            let last_hidden_start = (seq_len - 1) * loaded.config.embedding_length;
            ws.hidden.copy_from_slice(
                &batch_ws.hidden
                    [last_hidden_start..last_hidden_start + loaded.config.embedding_length],
            );
        } else {
            ws.hidden.copy_from_slice(&activations);
            run_distributed_range(&node1, pos, &loaded.config, &mut cache, &mut ws, options)?;
        }
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

fn run_master_generate(
    model_path: &Path,
    worker_addr: &str,
    prompt: &str,
    requested_split_layer: usize,
    max_tokens: usize,
) -> Result<(), String> {
    let loaded = load_cluster_model(model_path, requested_split_layer)?;
    let tokenizer = tokenizer::Tokenizer::from_gguf(&loaded.gguf)
        .map_err(|err| format!("failed to load tokenizer: {err}"))?;
    let prompt_tokens = tokenizer
        .encode(prompt, true, true)
        .map_err(|err| format!("failed to tokenize prompt: {err}"))?;
    if prompt_tokens.is_empty() {
        return Err("prompt tokenized to an empty sequence".to_owned());
    }
    if prompt_tokens.len() + max_tokens > loaded.config.context_length {
        return Err(format!(
            "prompt has {} tokens and max_tokens is {max_tokens}, exceeding context length {}",
            prompt_tokens.len(),
            loaded.config.context_length
        ));
    }

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

    println!("NanoCamelid cluster TCP master generate");
    println!("model: {}", model_path.display());
    println!("worker_addr: {worker_addr}");
    println!("prompt: {prompt:?}");
    println!("prompt_tokens: {prompt_tokens:?}");
    println!("max_tokens: {max_tokens}");
    println!("layers: {}", loaded.config.block_count);
    println!("split_layer: {}", loaded.split_layer);
    println!("full_forward_check: false");

    let options = runtime_options();
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

    let started_prefill = Instant::now();
    let prompt_len = prompt_tokens.len();
    let mut batch_ws = inference::LlamaBatchWorkspace::new(&loaded.config, prompt_len);
    for (token_idx, &token) in prompt_tokens.iter().enumerate() {
        let emb_start = token as usize * loaded.config.embedding_length;
        let hidden_start = token_idx * loaded.config.embedding_length;
        batch_ws.hidden[hidden_start..hidden_start + loaded.config.embedding_length]
            .copy_from_slice(
                &token_embeddings[emb_start..emb_start + loaded.config.embedding_length],
            );
    }
    inference::run_layer_range_batch(
        0,
        &node0.layers,
        prompt_len,
        0,
        &loaded.config,
        &mut node0_cache,
        &mut batch_ws,
        options,
    );
    cluster::send_activation_packet(
        &mut stream,
        0,
        prompt_len as u32,
        &batch_ws.hidden[..prompt_len * loaded.config.embedding_length],
    )
    .map_err(|err| format!("failed to send batched prefill activations: {err}"))?;
    let last_feedback = cluster::recv_token_feedback(&mut stream)
        .map_err(|err| format!("failed to receive batched prefill token feedback: {err}"))?;

    println!(
        "prompt_ingest_seconds: {:.3}",
        started_prefill.elapsed().as_secs_f64()
    );
    println!("\nGenerating response:\n");

    let mut generated_tokens = Vec::new();
    let mut last_printed_len = 0;
    let mut next_token = last_feedback.token_id as usize;
    let mut pos = prompt_tokens.len();
    let started_generation = Instant::now();
    let mut master_stage_total = Duration::ZERO;
    let mut tcp_round_trip_total = Duration::ZERO;

    while generated_tokens.len() < max_tokens {
        if is_stop_token(&tokenizer, next_token as u32) || pos >= loaded.config.context_length {
            break;
        }

        generated_tokens.push(next_token as u32);
        stream_generated_text(&tokenizer, &generated_tokens, &mut last_printed_len)?;

        if generated_tokens.len() >= max_tokens {
            break;
        }

        let master_start = Instant::now();
        run_master_half_token(
            next_token,
            pos,
            MasterHalfState {
                config: &loaded.config,
                node0: &node0,
                token_embeddings,
                cache: &mut node0_cache,
                ws: &mut node0_ws,
                options,
            },
        )?;
        master_stage_total += master_start.elapsed();

        let round_trip_start = Instant::now();
        cluster::send_activation_packet(
            &mut stream,
            pos as u32,
            (pos + 1) as u32,
            &node0_ws.hidden,
        )
        .map_err(|err| format!("failed to send decode activation packet: {err}"))?;
        let feedback = cluster::recv_token_feedback(&mut stream)
            .map_err(|err| format!("failed to receive decode token feedback: {err}"))?;
        tcp_round_trip_total += round_trip_start.elapsed();

        next_token = feedback.token_id as usize;
        pos += 1;
    }

    let elapsed = started_generation.elapsed().as_secs_f64();
    println!();
    println!();
    println!("generated_tokens: {generated_tokens:?}");
    println!("generated_token_count: {}", generated_tokens.len());
    println!("generation_seconds: {elapsed:.3}");
    if !generated_tokens.is_empty() {
        println!(
            "cluster_tokens_per_sec: {:.3}",
            generated_tokens.len() as f64 / elapsed
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
    println!("result: PASS_GENERATE_UNCHECKED");
    Ok(())
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

struct MasterHalfState<'a> {
    config: &'a model::LlamaModelConfig,
    node0: &'a DistributedLlamaWeights,
    token_embeddings: &'a [f32],
    cache: &'a mut inference::LlamaKvCache,
    ws: &'a mut inference::LlamaWorkspace,
    options: inference::LlamaRuntimeOptions,
}

fn run_master_half_token(
    token_id: usize,
    pos: usize,
    state: MasterHalfState<'_>,
) -> Result<(), String> {
    inference::embed_token(token_id, state.config, state.token_embeddings, state.ws);
    run_distributed_range(
        state.node0,
        pos,
        state.config,
        state.cache,
        state.ws,
        state.options,
    )
}

fn stream_generated_text(
    tokenizer: &tokenizer::Tokenizer,
    generated_tokens: &[u32],
    last_printed_len: &mut usize,
) -> Result<(), String> {
    let full_text = tokenizer
        .decode(generated_tokens, true)
        .map_err(|err| format!("failed to decode generated tokens: {err}"))?;
    if full_text.len() > *last_printed_len {
        print!("{}", &full_text[*last_printed_len..]);
        io::stdout()
            .flush()
            .map_err(|err| format!("failed to flush stdout: {err}"))?;
        *last_printed_len = full_text.len();
    }
    Ok(())
}

fn is_stop_token(tokenizer: &tokenizer::Tokenizer, token_id: u32) -> bool {
    Some(token_id) == tokenizer.special.eos
        || Some(token_id) == tokenizer.special.eot
        || Some(token_id) == tokenizer.special.eom
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
    println!(
        "  cargo run --release --bin cluster_tcp_smoke -- master-generate <model.gguf> [worker_addr] <prompt> [split_layer] [max_tokens]"
    );
    println!();
    println!("Environment:");
    println!("  {CLUSTER_CONTEXT_LIMIT_ENV}=<tokens>   Default {DEFAULT_CONTEXT_LIMIT}");
}
