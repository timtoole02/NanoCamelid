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
        Some("middle-worker") => {
            let Some(model_path) = args.next() else {
                print_usage();
                return Err("missing middle worker model path".to_owned());
            };
            let bind_addr = args.next().unwrap_or_else(|| "127.0.0.1:5006".to_owned());
            let next_addr = args.next().unwrap_or_else(|| "127.0.0.1:5005".to_owned());
            let start_layer = parse_optional_usize(args.next(), 0, "start_layer")?;
            let end_layer = parse_optional_usize(args.next(), 0, "end_layer")?;
            run_middle_worker(
                Path::new(&model_path),
                &bind_addr,
                &next_addr,
                start_layer,
                end_layer,
            )
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
                false,
            )
        }
        Some("master-chat") | Some("master-chat-generate") => {
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
                true,
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

    let local_hello = cluster_hello(
        cluster::ClusterNodeRole::FinalWorker,
        &loaded.config,
        node1.layer_start,
        node1.layer_end,
    )?;
    let upstream_hello = cluster::recv_cluster_hello(&mut stream)
        .map_err(|err| format!("failed to receive cluster hello from upstream: {err}"))?;
    validate_peer_hello(
        &local_hello,
        &upstream_hello,
        &[
            cluster::ClusterNodeRole::Master,
            cluster::ClusterNodeRole::Middle,
        ],
        None,
        Some(node1.layer_start),
    )?;
    cluster::send_cluster_hello(&mut stream, local_hello)
        .map_err(|err| format!("failed to send cluster hello to upstream: {err}"))?;
    print_cluster_peer("upstream", &upstream_hello);

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
    let mut upstream_wait_samples = StageSamples::default();
    let mut compute_samples = StageSamples::default();
    let mut layers_samples = StageSamples::default();
    let mut logits_samples = StageSamples::default();
    let mut feedback_send_samples = StageSamples::default();
    loop {
        let upstream_wait_start = Instant::now();
        let header = match cluster::recv_activation_packet(&mut stream, &mut activations) {
            Ok(header) => header,
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(format!("failed to receive activation packet: {err}")),
        };
        let upstream_wait_elapsed = upstream_wait_start.elapsed();
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
        let layers_elapsed = compute_start.elapsed();
        let logits_start = Instant::now();
        inference::compute_logits_from_hidden(
            &loaded.config,
            output_token_embeddings,
            output_norm,
            output_projection,
            &mut ws,
            options,
        );
        let logits_elapsed = logits_start.elapsed();
        let compute_elapsed = compute_start.elapsed();
        worker_compute_total += compute_elapsed;
        let next_token = inference::sample_logits(&ws.logits, 0.0);
        let feedback_send_start = Instant::now();
        cluster::send_token_feedback(&mut stream, next_token as u32, false)
            .map_err(|err| format!("failed to send token feedback: {err}"))?;
        if seq_len == 1 {
            upstream_wait_samples.push(upstream_wait_elapsed);
            compute_samples.push(compute_elapsed);
            layers_samples.push(layers_elapsed);
            logits_samples.push(logits_elapsed);
            feedback_send_samples.push(feedback_send_start.elapsed());
        }

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
    print_worker_trace_summary();
    println!(
        "json: {{\"benchmark\":\"cluster-decode-breakdown\",\"role\":\"final\",\"model\":{:?},\"layer_range\":\"{}..{}\",{},{},{},{},{}}}",
        model_path.display().to_string(),
        node1.layer_start,
        node1.layer_end,
        upstream_wait_samples.json_fields("upstream_wait"),
        compute_samples.json_fields("compute"),
        layers_samples.json_fields("layers"),
        logits_samples.json_fields("logits"),
        feedback_send_samples.json_fields("feedback_send"),
    );
    println!("result: WORKER_DONE");
    Ok(())
}

fn run_middle_worker(
    model_path: &Path,
    bind_addr: &str,
    next_addr: &str,
    start_layer: usize,
    end_layer: usize,
) -> Result<(), String> {
    let gguf = gguf::read_file(model_path).map_err(|err| format!("failed to read GGUF: {err}"))?;
    let mut config = model::LlamaModelConfig::from_gguf(&gguf)
        .map_err(|err| format!("failed to parse model config: {err}"))?;
    apply_context_limit(&mut config)?;
    if start_layer == 0 || end_layer == 0 {
        return Err("middle-worker requires explicit start_layer and end_layer".to_owned());
    }
    let node =
        model::LlamaWeights::load_distributed(model_path, &config, &gguf, start_layer, end_layer)
            .map_err(|err| format!("failed to load middle partial weights: {err}"))?;

    let listener =
        TcpListener::bind(bind_addr).map_err(|err| format!("failed to bind {bind_addr}: {err}"))?;
    println!("NanoCamelid cluster TCP middle worker");
    println!("model: {}", model_path.display());
    println!("bind_addr: {bind_addr}");
    println!("next_addr: {next_addr}");
    println!("layers: {}", config.block_count);
    println!("layer_range: {}..{}", node.layer_start, node.layer_end);
    println!("waiting for upstream master connection...");

    let mut downstream = TcpStream::connect(next_addr)
        .map_err(|err| format!("failed to connect to downstream worker {next_addr}: {err}"))?;
    downstream
        .set_nodelay(true)
        .map_err(|err| format!("failed to set downstream TCP_NODELAY: {err}"))?;
    println!("downstream_connected: {next_addr}");
    let local_hello = cluster_hello(
        cluster::ClusterNodeRole::Middle,
        &config,
        node.layer_start,
        node.layer_end,
    )?;
    cluster::send_cluster_hello(&mut downstream, local_hello)
        .map_err(|err| format!("failed to send cluster hello to downstream worker: {err}"))?;
    let downstream_hello = cluster::recv_cluster_hello(&mut downstream)
        .map_err(|err| format!("failed to receive cluster hello from downstream worker: {err}"))?;
    validate_peer_hello(
        &local_hello,
        &downstream_hello,
        &[cluster::ClusterNodeRole::FinalWorker],
        Some(node.layer_end),
        Some(config.block_count),
    )?;
    print_cluster_peer("downstream", &downstream_hello);

    let (mut upstream, peer_addr) = listener
        .accept()
        .map_err(|err| format!("failed to accept upstream connection: {err}"))?;
    upstream
        .set_nodelay(true)
        .map_err(|err| format!("failed to set upstream TCP_NODELAY: {err}"))?;
    println!("upstream_connected: {peer_addr}");
    let upstream_hello = cluster::recv_cluster_hello(&mut upstream)
        .map_err(|err| format!("failed to receive cluster hello from upstream: {err}"))?;
    validate_peer_hello(
        &local_hello,
        &upstream_hello,
        &[cluster::ClusterNodeRole::Master],
        Some(0),
        Some(node.layer_start),
    )?;
    cluster::send_cluster_hello(&mut upstream, local_hello)
        .map_err(|err| format!("failed to send cluster hello to upstream: {err}"))?;
    print_cluster_peer("upstream", &upstream_hello);

    let options = runtime_options();
    let mut cache =
        inference::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut ws = inference::LlamaWorkspace::new(&config);
    let mut batch_ws = inference::LlamaBatchWorkspace::new(&config, config.context_length);
    let mut activations = Vec::new();
    let mut feedback_tokens = Vec::new();
    let mut middle_compute_total = Duration::ZERO;
    let mut downstream_round_trip_total = Duration::ZERO;
    let mut upstream_wait_samples = StageSamples::default();
    let mut compute_samples = StageSamples::default();
    let mut downstream_send_samples = StageSamples::default();
    let mut downstream_wait_samples = StageSamples::default();

    loop {
        let upstream_wait_start = Instant::now();
        let header = match cluster::recv_activation_packet(&mut upstream, &mut activations) {
            Ok(header) => header,
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => {
                return Err(format!(
                    "failed to receive upstream activation packet: {err}"
                ));
            }
        };
        let upstream_wait_elapsed = upstream_wait_start.elapsed();
        let seq_len = header.seq_len as usize;
        let pos = header.pos as usize;
        let expected_floats = seq_len * config.embedding_length;
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
        if pos + seq_len > config.context_length {
            return Err(format!(
                "received pos {} seq_len {} outside context length {}",
                header.pos, header.seq_len, config.context_length
            ));
        }

        let compute_start = Instant::now();
        let outgoing = if seq_len > 1 {
            batch_ws.hidden[..expected_floats].copy_from_slice(&activations);
            inference::run_layer_range_batch(
                node.layer_start,
                &node.layers,
                seq_len,
                pos,
                &config,
                &mut cache,
                &mut batch_ws,
                options,
            );
            &batch_ws.hidden[..expected_floats]
        } else {
            ws.hidden.copy_from_slice(&activations);
            run_distributed_range(&node, pos, &config, &mut cache, &mut ws, options)?;
            &ws.hidden[..]
        };
        let compute_elapsed = compute_start.elapsed();
        middle_compute_total += compute_elapsed;

        let downstream_send_start = Instant::now();
        cluster::send_activation_packet(&mut downstream, header.pos, header.seq_len, outgoing)
            .map_err(|err| format!("failed to send downstream activation packet: {err}"))?;
        let downstream_send_elapsed = downstream_send_start.elapsed();
        let downstream_wait_start = Instant::now();
        let feedback = cluster::recv_token_feedback(&mut downstream)
            .map_err(|err| format!("failed to receive downstream token feedback: {err}"))?;
        let downstream_wait_elapsed = downstream_wait_start.elapsed();
        downstream_round_trip_total += downstream_send_elapsed + downstream_wait_elapsed;
        if seq_len == 1 {
            upstream_wait_samples.push(upstream_wait_elapsed);
            compute_samples.push(compute_elapsed);
            downstream_send_samples.push(downstream_send_elapsed);
            downstream_wait_samples.push(downstream_wait_elapsed);
        }
        cluster::send_token_feedback(&mut upstream, feedback.token_id, feedback.is_finished)
            .map_err(|err| format!("failed to send upstream token feedback: {err}"))?;

        println!("received_pos: {}", header.pos);
        println!("received_seq_len: {}", header.seq_len);
        println!("received_float_count: {}", header.float_count);
        println!("middle_feedback_token: {}", feedback.token_id);
        feedback_tokens.push(feedback.token_id);
    }

    println!("middle_tokens: {}", feedback_tokens.len());
    println!("middle_feedback_tokens: {feedback_tokens:?}");
    println!(
        "middle_compute_total_ms: {:.3}",
        middle_compute_total.as_secs_f64() * 1000.0
    );
    println!(
        "middle_downstream_round_trip_total_ms: {:.3}",
        downstream_round_trip_total.as_secs_f64() * 1000.0
    );
    if !feedback_tokens.is_empty() {
        let token_count = feedback_tokens.len() as f64;
        println!(
            "middle_compute_avg_ms: {:.3}",
            middle_compute_total.as_secs_f64() * 1000.0 / token_count
        );
        println!(
            "middle_downstream_round_trip_avg_ms: {:.3}",
            downstream_round_trip_total.as_secs_f64() * 1000.0 / token_count
        );
    }
    print_worker_trace_summary();
    println!(
        "json: {{\"benchmark\":\"cluster-decode-breakdown\",\"role\":\"middle\",\"model\":{:?},\"layer_range\":\"{}..{}\",{},{},{},{}}}",
        model_path.display().to_string(),
        node.layer_start,
        node.layer_end,
        upstream_wait_samples.json_fields("upstream_wait"),
        compute_samples.json_fields("compute"),
        downstream_send_samples.json_fields("downstream_send"),
        downstream_wait_samples.json_fields("downstream_wait"),
    );
    println!("result: MIDDLE_WORKER_DONE");
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
    chat_prompt: bool,
) -> Result<(), String> {
    let loaded = load_cluster_model(model_path, requested_split_layer)?;
    let tokenizer = tokenizer::Tokenizer::from_gguf(&loaded.gguf)
        .map_err(|err| format!("failed to load tokenizer: {err}"))?;
    let rendered_chat = if chat_prompt {
        Some(tokenizer.render_chat_prompt(&[tokenizer::ChatMessage {
            role: "user",
            content: prompt,
        }]))
    } else {
        None
    };
    let prompt_text = rendered_chat
        .as_ref()
        .map(|rendered| rendered.text.as_str())
        .unwrap_or(prompt);
    let add_special = rendered_chat
        .as_ref()
        .map(|rendered| rendered.add_special)
        .unwrap_or(true);
    let parse_special = rendered_chat
        .as_ref()
        .map(|rendered| rendered.parse_special)
        .unwrap_or(true);
    let prompt_tokens = tokenizer
        .encode(prompt_text, add_special, parse_special)
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
    println!("prompt_mode: {}", if chat_prompt { "chat" } else { "raw" });
    println!("prompt: {prompt:?}");
    if let Some(rendered) = rendered_chat.as_ref() {
        println!("chat_renderer: {}", rendered.renderer);
        println!(
            "chat_template_format: {}",
            tokenizer.chat_template_format().unwrap_or("none")
        );
        println!("rendered_prompt: {:?}", rendered.text);
    }
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
    let local_hello = cluster_hello(
        cluster::ClusterNodeRole::Master,
        &loaded.config,
        0,
        loaded.split_layer,
    )?;
    cluster::send_cluster_hello(&mut stream, local_hello)
        .map_err(|err| format!("failed to send cluster hello to worker: {err}"))?;
    let worker_hello = cluster::recv_cluster_hello(&mut stream)
        .map_err(|err| format!("failed to receive cluster hello from worker: {err}"))?;
    validate_peer_hello(
        &local_hello,
        &worker_hello,
        &[
            cluster::ClusterNodeRole::Middle,
            cluster::ClusterNodeRole::FinalWorker,
        ],
        Some(loaded.split_layer),
        None,
    )?;
    print_cluster_peer("worker", &worker_hello);

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
    let mut compute_samples = StageSamples::default();
    let mut send_samples = StageSamples::default();
    let mut recv_wait_samples = StageSamples::default();

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
        let master_elapsed = master_start.elapsed();
        master_stage_total += master_elapsed;
        compute_samples.push(master_elapsed);

        let send_start = Instant::now();
        cluster::send_activation_packet(&mut stream, pos as u32, 1, &node0_ws.hidden)
            .map_err(|err| format!("failed to send decode activation packet: {err}"))?;
        let send_elapsed = send_start.elapsed();
        let recv_start = Instant::now();
        let feedback = cluster::recv_token_feedback(&mut stream)
            .map_err(|err| format!("failed to receive decode token feedback: {err}"))?;
        let recv_elapsed = recv_start.elapsed();
        tcp_round_trip_total += send_elapsed + recv_elapsed;
        send_samples.push(send_elapsed);
        recv_wait_samples.push(recv_elapsed);

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
    println!(
        "json: {{\"benchmark\":\"cluster-decode-breakdown\",\"role\":\"master\",\"model\":{:?},\"generated_tokens\":{},\"generation_seconds\":{:.3},\"decode_tokens_per_sec\":{:.3},{},{},{}}}",
        model_path.display().to_string(),
        generated_tokens.len(),
        elapsed,
        if elapsed > 0.0 {
            generated_tokens.len() as f64 / elapsed
        } else {
            0.0
        },
        compute_samples.json_fields("compute"),
        send_samples.json_fields("send"),
        recv_wait_samples.json_fields("recv_wait"),
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
    let local_hello = cluster_hello(
        cluster::ClusterNodeRole::Master,
        &loaded.config,
        0,
        loaded.split_layer,
    )?;
    cluster::send_cluster_hello(&mut stream, local_hello)
        .map_err(|err| format!("failed to send cluster hello to worker: {err}"))?;
    let worker_hello = cluster::recv_cluster_hello(&mut stream)
        .map_err(|err| format!("failed to receive cluster hello from worker: {err}"))?;
    validate_peer_hello(
        &local_hello,
        &worker_hello,
        &[
            cluster::ClusterNodeRole::Middle,
            cluster::ClusterNodeRole::FinalWorker,
        ],
        Some(loaded.split_layer),
        None,
    )?;
    print_cluster_peer("worker", &worker_hello);
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
        cluster::send_activation_packet(&mut stream, pos as u32, 1, &node0_ws.hidden)
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

// Stage-level trace summary (populated only when NANOCAMELID_TRACE=1).
fn print_worker_trace_summary() {
    for (stage, stats) in inference::trace_snapshot().into_iter().take(24) {
        let total_ms = stats.total.as_secs_f64() * 1000.0;
        println!(
            "trace: {stage} calls {} total_ms {:.1} avg_ms {:.4}",
            stats.calls,
            total_ms,
            total_ms / stats.calls.max(1) as f64
        );
    }
}

// Per-token stage timings for the Phase 0 decode breakdown. Samples are
// decode-step-only (seq_len == 1); the batched prefill packet is excluded so
// the summary reflects steady-state decode.
#[derive(Default)]
struct StageSamples {
    ms: Vec<f64>,
}

impl StageSamples {
    fn push(&mut self, elapsed: Duration) {
        self.ms.push(elapsed.as_secs_f64() * 1000.0);
    }

    fn json_fields(&self, label: &str) -> String {
        if self.ms.is_empty() {
            return format!("\"{label}_samples\":0");
        }
        let mut sorted = self.ms.clone();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let count = sorted.len();
        let avg = sorted.iter().sum::<f64>() / count as f64;
        let p50 = sorted[count / 2];
        let p95 = sorted[((count as f64 * 0.95) as usize).min(count - 1)];
        format!(
            "\"{label}_samples\":{count},\"{label}_avg_ms\":{avg:.3},\"{label}_p50_ms\":{p50:.3},\"{label}_p95_ms\":{p95:.3}"
        )
    }
}

struct LoadedClusterModel {
    gguf: gguf::GgufFile,
    config: model::LlamaModelConfig,
    split_layer: usize,
}

fn cluster_hello(
    role: cluster::ClusterNodeRole,
    config: &model::LlamaModelConfig,
    layer_start: usize,
    layer_end: usize,
) -> Result<cluster::ClusterHello, String> {
    let to_u32 = |label: &str, value: usize| {
        u32::try_from(value).map_err(|_| format!("{label} {value} exceeds u32 wire range"))
    };
    Ok(cluster::ClusterHello {
        role,
        block_count: to_u32("block_count", config.block_count)?,
        embedding_length: to_u32("embedding_length", config.embedding_length)?,
        context_length: to_u32("context_length", config.context_length)?,
        kv_width: to_u32("kv_width", config.kv_width)?,
        vocab_size: to_u32("vocab_size", config.vocab_size)?,
        layer_start: to_u32("layer_start", layer_start)?,
        layer_end: to_u32("layer_end", layer_end)?,
        expert_count: to_u32("expert_count", config.expert_count)?,
        expert_used_count: to_u32("expert_used_count", config.expert_used_count)?,
    })
}

fn validate_peer_hello(
    local: &cluster::ClusterHello,
    peer: &cluster::ClusterHello,
    allowed_roles: &[cluster::ClusterNodeRole],
    expected_layer_start: Option<usize>,
    expected_layer_end: Option<usize>,
) -> Result<(), String> {
    if !allowed_roles.contains(&peer.role) {
        return Err(format!(
            "cluster peer role {:?} is not allowed here; expected one of {:?}",
            peer.role, allowed_roles
        ));
    }

    let shape_pairs = [
        ("block_count", local.block_count, peer.block_count),
        (
            "embedding_length",
            local.embedding_length,
            peer.embedding_length,
        ),
        ("context_length", local.context_length, peer.context_length),
        ("kv_width", local.kv_width, peer.kv_width),
        ("vocab_size", local.vocab_size, peer.vocab_size),
        ("expert_count", local.expert_count, peer.expert_count),
        (
            "expert_used_count",
            local.expert_used_count,
            peer.expert_used_count,
        ),
    ];
    for (label, expected, actual) in shape_pairs {
        if expected != actual {
            return Err(format!(
                "cluster peer {label} mismatch: local {expected}, peer {actual}"
            ));
        }
    }

    if let Some(expected) = expected_layer_start
        && peer.layer_start != expected as u32
    {
        return Err(format!(
            "cluster peer layer_start mismatch: expected {expected}, got {}",
            peer.layer_start
        ));
    }
    if let Some(expected) = expected_layer_end
        && peer.layer_end != expected as u32
    {
        return Err(format!(
            "cluster peer layer_end mismatch: expected {expected}, got {}",
            peer.layer_end
        ));
    }
    if peer.layer_start >= peer.layer_end || peer.layer_end > peer.block_count {
        return Err(format!(
            "cluster peer has invalid layer range {}..{} for {} layers",
            peer.layer_start, peer.layer_end, peer.block_count
        ));
    }

    Ok(())
}

fn print_cluster_peer(label: &str, hello: &cluster::ClusterHello) {
    println!(
        "{label}_cluster_peer: role={:?} layers={}..{} blocks={} hidden={} ctx={} kv={} vocab={} experts={}/{}",
        hello.role,
        hello.layer_start,
        hello.layer_end,
        hello.block_count,
        hello.embedding_length,
        hello.context_length,
        hello.kv_width,
        hello.vocab_size,
        hello.expert_used_count,
        hello.expert_count
    );
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
        "  cargo run --release --bin cluster_tcp_smoke -- middle-worker <model.gguf> [bind_addr] [next_addr] <start_layer> <end_layer>"
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
    println!(
        "  cargo run --release --bin cluster_tcp_smoke -- master-chat <model.gguf> [worker_addr] <prompt> [split_layer] [max_tokens]"
    );
    println!();
    println!("Environment:");
    println!("  {CLUSTER_CONTEXT_LIMIT_ENV}=<tokens>   Default {DEFAULT_CONTEXT_LIMIT}");
}
