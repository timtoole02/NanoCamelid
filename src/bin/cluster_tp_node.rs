// Phase 4 P4.2: two-node tensor parallelism over TCP.
//
// Both nodes load the same GGUF, build both TP shards locally, and keep only
// their own (shard 0 = master, shard 1 = worker). Every token: the master
// sends (pos, token id); both nodes run every layer on their shard; at each
// of the two per-layer reduction points the worker sends its partial and the
// master returns the global sum (fixed order: shard0 + shard1, so both sides
// hold bit-identical activations). Embeddings are replicated; the head runs
// on the master only.
//
//   cluster_tp_node worker      <model.gguf> <bind_addr>
//   cluster_tp_node master-chat <model.gguf> <worker_addr> "<prompt>" [max_tokens]
//   cluster_tp_node reference   <model.gguf> "<prompt>" [max_tokens]
//
// NANOCAMELID_TP_PARITY=1 pins the scalar Q8 kernel (use for parity A/Bs
// against `reference`, which honors the same flag). Weights load with the
// 1x4 swizzle disabled on every mode (shards slice row-major blocks), so
// `reference` here — not `nanocamelid chat` — is the like-for-like baseline.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use nanocamelid::{gguf, inference, model, q8, tokenizer, tp};

const HELLO_MAGIC: u32 = 0x5450_4E31; // "TPN1"
const TOKEN_MAGIC: u32 = 0x5450_544B; // "TPTK"
const PARTIAL_MAGIC: u32 = 0x5450_5041; // "TPPA"
const SUM_MAGIC: u32 = 0x5450_5355; // "TPSU"
const ARGMAX_MAGIC: u32 = 0x5450_414D; // "TPAM"
const SHUTDOWN_POS: u32 = u32::MAX;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            println!("result: FAIL");
            ExitCode::FAILURE
        }
    }
}

fn rope_scaling_from_gguf(gguf: &gguf::GgufFile) -> inference::RopeScaling {
    let prefix = gguf
        .metadata_string("general.architecture")
        .and_then(model::metadata_prefix_for_arch)
        .unwrap_or("llama");
    inference::RopeScaling {
        factor: gguf.metadata_f32(&format!("{prefix}.rope.scaling.factor")),
        original_context_length: gguf
            .metadata_u32(&format!("{prefix}.rope.scaling.original_context_length"))
            .map(|value| value as f32),
        low_freq_factor: gguf.metadata_f32(&format!("{prefix}.rope.scaling.low_freq_factor")),
        high_freq_factor: gguf.metadata_f32(&format!("{prefix}.rope.scaling.high_freq_factor")),
    }
}

fn runtime_options(gguf: &gguf::GgufFile) -> inference::LlamaRuntimeOptions {
    let parity = std::env::var("NANOCAMELID_TP_PARITY").is_ok_and(|v| v == "1");
    let selector = if parity {
        q8::Q8DotKernelSelector {
            requested: Some(q8::Q8DotKernel::Scalar),
            selected: q8::Q8DotKernel::Scalar,
            fallback_reason: None,
        }
    } else {
        q8::Q8DotKernelSelector::from_env()
    };
    inference::LlamaRuntimeOptions {
        q8_selector: selector,
        rope_scaling: rope_scaling_from_gguf(gguf),
        compute_logits: true,
    }
}

struct LoadedModel {
    config: model::LlamaModelConfig,
    weights: model::LlamaWeights,
    tokenizer: tokenizer::Tokenizer,
    options: inference::LlamaRuntimeOptions,
}

fn load(model_path: &str) -> Result<LoadedModel, String> {
    // Shards slice row-major blocks; must be set before any load.
    unsafe {
        std::env::set_var("NANOCAMELID_Q4_SWIZZLE_1X4", "0");
        std::env::set_var("NANOCAMELID_Q8_SWIZZLE_1X4", "0");
    }
    let path = Path::new(model_path);
    let gguf = gguf::read_file(path).map_err(|e| e.to_string())?;
    let mut config = model::LlamaModelConfig::from_gguf(&gguf).map_err(|e| e.to_string())?;
    if config.context_length > 512 {
        config.context_length = 512;
    }
    let tok = tokenizer::Tokenizer::from_gguf(&gguf).map_err(|e| e.to_string())?;
    let options = runtime_options(&gguf);
    println!("Loading weights: {model_path}");
    let weights = model::LlamaWeights::load(path, &config, &gguf)?;
    Ok(LoadedModel {
        config,
        weights,
        tokenizer: tok,
        options,
    })
}

fn write_msg<W: Write>(w: &mut W, magic: u32, a: u32, b: u32, payload: &[f32]) -> Result<(), String> {
    let mut msg = Vec::with_capacity(12 + payload.len() * 4);
    msg.extend_from_slice(&magic.to_le_bytes());
    msg.extend_from_slice(&a.to_le_bytes());
    msg.extend_from_slice(&b.to_le_bytes());
    for v in payload {
        msg.extend_from_slice(&v.to_le_bytes());
    }
    w.write_all(&msg).map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())
}

fn read_msg<R: Read>(
    r: &mut R,
    expect_magic: u32,
    payload: &mut [f32],
) -> Result<(u32, u32), String> {
    let mut header = [0_u8; 12];
    r.read_exact(&mut header).map_err(|e| e.to_string())?;
    let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
    if magic != expect_magic {
        return Err(format!(
            "expected magic 0x{expect_magic:08X}, got 0x{magic:08X}"
        ));
    }
    let a = u32::from_le_bytes(header[4..8].try_into().unwrap());
    let b = u32::from_le_bytes(header[8..12].try_into().unwrap());
    if !payload.is_empty() {
        let mut bytes = vec![0_u8; payload.len() * 4];
        r.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        for (v, chunk) in payload.iter_mut().zip(bytes.chunks_exact(4)) {
            *v = f32::from_le_bytes(chunk.try_into().unwrap());
        }
    }
    Ok((a, b))
}

fn reduce_tag(layer_idx: usize, phase: tp::ReducePhase) -> u32 {
    ((layer_idx as u32) << 1) | matches!(phase, tp::ReducePhase::Ffn) as u32
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("worker") => {
            let model_path = args.next().ok_or("missing model path")?;
            let bind = args.next().unwrap_or_else(|| "0.0.0.0:5915".to_owned());
            run_worker(&model_path, &bind)
        }
        Some("master-chat") => {
            let model_path = args.next().ok_or("missing model path")?;
            let worker = args.next().ok_or("missing worker addr")?;
            let prompt = args.next().ok_or("missing prompt")?;
            let max_tokens: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(64);
            run_master(&model_path, &worker, &prompt, max_tokens)
        }
        Some("reference") => {
            let model_path = args.next().ok_or("missing model path")?;
            let prompt = args.next().ok_or("missing prompt")?;
            let max_tokens: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(64);
            run_reference(&model_path, &prompt, max_tokens)
        }
        _ => Err("usage: cluster_tp_node worker|master-chat|reference ...".to_owned()),
    }
}

fn encode_chat(
    loaded: &LoadedModel,
    prompt: &str,
) -> Result<Vec<u32>, String> {
    let rendered = loaded.tokenizer.render_chat_prompt(&[tokenizer::ChatMessage {
        role: "user",
        content: prompt,
    }]);
    loaded
        .tokenizer
        .encode(&rendered.text, rendered.add_special, rendered.parse_special)
        .map_err(|e| e.to_string())
}

fn run_reference(model_path: &str, prompt: &str, max_tokens: usize) -> Result<(), String> {
    let loaded = load(model_path)?;
    let prompt_tokens = encode_chat(&loaded, prompt)?;
    let mut cache = inference::LlamaKvCache::new(
        loaded.config.block_count,
        loaded.config.context_length,
        loaded.config.kv_width,
    );
    let mut ws = inference::LlamaWorkspace::new(&loaded.config);
    let mut generated = Vec::new();
    let mut pos = 0usize;
    let started = Instant::now();
    let mut decode_started = None;
    for (i, &t) in prompt_tokens.iter().enumerate() {
        inference::forward_pass(
            t as usize,
            pos,
            &loaded.config,
            &loaded.weights,
            &mut cache,
            &mut ws,
            loaded.options,
        );
        pos += 1;
        if i + 1 == prompt_tokens.len() {
            decode_started = Some(Instant::now());
        }
    }
    let mut next = inference::sample_logits(&ws.logits, 0.0) as u32;
    while generated.len() < max_tokens && pos < loaded.config.context_length {
        generated.push(next);
        inference::forward_pass(
            next as usize,
            pos,
            &loaded.config,
            &loaded.weights,
            &mut cache,
            &mut ws,
            loaded.options,
        );
        pos += 1;
        next = inference::sample_logits(&ws.logits, 0.0) as u32;
    }
    let decode_secs = decode_started
        .map(|s| s.elapsed().as_secs_f64())
        .unwrap_or_default();
    println!("generated_tokens: {generated:?}");
    println!(
        "generated_text: {:?}",
        loaded.tokenizer.decode(&generated, true).unwrap_or_default()
    );
    println!(
        "json: {{\"benchmark\":\"tp-reference\",\"prompt_tokens\":{},\"generated\":{},\"total_seconds\":{:.3},\"decode_tokens_per_sec\":{:.3}}}",
        prompt_tokens.len(),
        generated.len(),
        started.elapsed().as_secs_f64(),
        if decode_secs > 0.0 { generated.len() as f64 / decode_secs } else { 0.0 },
    );
    println!("result: PASS");
    Ok(())
}

fn run_worker(model_path: &str, bind: &str) -> Result<(), String> {
    let loaded = load(model_path)?;
    let parity = std::env::var("NANOCAMELID_TP_PARITY").is_ok_and(|v| v == "1");
    let mut shards = tp::build_tp_shards(&loaded.config, &loaded.weights.layers, 2)?;
    let mut shard = shards.remove(1);
    drop(shards);
    let emb = loaded.config.embedding_length;
    // Fast mode: swizzled kernels + a row-parallel slice of the LM head.
    let mut head = if parity {
        None
    } else {
        tp::swizzle_shards(std::slice::from_mut(&mut shard));
        let mut heads = tp::build_tp_head_shards(
            &loaded.weights.token_embeddings,
            loaded.config.vocab_size,
            emb,
            2,
        )?;
        Some(heads.remove(1))
    };
    let mut head_ws = inference::LlamaWorkspace::new(&loaded.config);

    let listener = TcpListener::bind(bind).map_err(|e| e.to_string())?;
    println!("tp worker (shard 1) listening on {bind}");
    let (mut stream, peer) = listener.accept().map_err(|e| e.to_string())?;
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    println!("master connected: {peer}");
    write_msg(
        &mut stream,
        HELLO_MAGIC,
        loaded.config.block_count as u32,
        emb as u32,
        &[],
    )?;

    let mut rt = tp::TpRuntime::new(&loaded.config);
    let mut hidden = vec![0.0_f32; emb];
    let mut compute_total = Duration::ZERO;
    let mut wait_total = Duration::ZERO;
    let mut tokens = 0u64;

    loop {
        let wait_start = Instant::now();
        let (pos, token) = read_msg(&mut stream, TOKEN_MAGIC, &mut [])?;
        wait_total += wait_start.elapsed();
        if pos == SHUTDOWN_POS {
            break;
        }
        tokens += 1;
        let compute_start = Instant::now();
        let emb_start = token as usize * emb;
        hidden.copy_from_slice(&loaded.weights.token_embeddings[emb_start..emb_start + emb]);
        let stream_cell = std::cell::RefCell::new(&mut stream);
        tp::tp_forward_token_reduced(
            &mut hidden,
            std::slice::from_mut(&mut shard),
            &mut rt,
            pos as usize,
            loaded.options,
            |partial, layer_idx, phase| {
                let mut s = stream_cell.borrow_mut();
                write_msg(*s, PARTIAL_MAGIC, reduce_tag(layer_idx, phase), 0, partial)?;
                let (tag, _) = read_msg(*s, SUM_MAGIC, partial)?;
                if tag != reduce_tag(layer_idx, phase) {
                    return Err(format!("reduce tag mismatch at layer {layer_idx}"));
                }
                Ok(())
            },
        )?;
        if let Some(head) = head.as_mut() {
            let (idx, logit) = tp::head_shard_argmax(
                head,
                &hidden,
                &loaded.weights.output_norm,
                &mut rt,
                &mut head_ws,
                loaded.config.rms_norm_epsilon,
                loaded.options,
            );
            write_msg(&mut stream, ARGMAX_MAGIC, idx, logit.to_bits(), &[])?;
        }
        compute_total += compute_start.elapsed();
    }

    println!(
        "json: {{\"benchmark\":\"tp-node\",\"role\":\"worker\",\"tokens\":{tokens},\"compute_total_ms\":{:.1},\"wait_total_ms\":{:.1}}}",
        compute_total.as_secs_f64() * 1000.0,
        wait_total.as_secs_f64() * 1000.0,
    );
    println!("result: WORKER_DONE");
    Ok(())
}

fn run_master(
    model_path: &str,
    worker_addr: &str,
    prompt: &str,
    max_tokens: usize,
) -> Result<(), String> {
    let loaded = load(model_path)?;
    let prompt_tokens = encode_chat(&loaded, prompt)?;
    let parity = std::env::var("NANOCAMELID_TP_PARITY").is_ok_and(|v| v == "1");
    let mut shards = tp::build_tp_shards(&loaded.config, &loaded.weights.layers, 2)?;
    shards.truncate(1);
    let mut shard = shards.remove(0);
    let emb = loaded.config.embedding_length;
    let mut head = if parity {
        None
    } else {
        tp::swizzle_shards(std::slice::from_mut(&mut shard));
        let mut heads = tp::build_tp_head_shards(
            &loaded.weights.token_embeddings,
            loaded.config.vocab_size,
            emb,
            2,
        )?;
        heads.truncate(1);
        Some(heads.remove(0))
    };

    let mut stream = TcpStream::connect(worker_addr).map_err(|e| e.to_string())?;
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    let (blocks, wemb) = read_msg(&mut stream, HELLO_MAGIC, &mut [])?;
    if blocks as usize != loaded.config.block_count || wemb as usize != emb {
        return Err("worker/master model shape mismatch".to_owned());
    }
    println!("worker hello ok: {blocks} layers, emb {wemb}");

    let mut rt = tp::TpRuntime::new(&loaded.config);
    let mut ws = inference::LlamaWorkspace::new(&loaded.config);
    let mut remote = vec![0.0_f32; emb];
    let mut generated = Vec::new();
    let mut compute_total = Duration::ZERO;
    let mut sync_total = Duration::ZERO;
    let mut logits_total = Duration::ZERO;
    let started = Instant::now();
    let mut decode_started = None;
    let mut pos = 0usize;
    let mut token = *prompt_tokens.first().ok_or("empty prompt")?;
    let total_prompt = prompt_tokens.len();
    let mut step = 0usize;

    loop {
        // Lockstep: announce the token, then both sides compute.
        write_msg(&mut stream, TOKEN_MAGIC, pos as u32, token, &[])?;
        let compute_start = Instant::now();
        let emb_start = token as usize * emb;
        ws.hidden.copy_from_slice(&loaded.weights.token_embeddings[emb_start..emb_start + emb]);
        {
            let stream_cell = std::cell::RefCell::new(&mut stream);
            let remote_cell = std::cell::RefCell::new(&mut remote);
            let sync_cell = std::cell::RefCell::new(&mut sync_total);
            tp::tp_forward_token_reduced(
                &mut ws.hidden,
                std::slice::from_mut(&mut shard),
                &mut rt,
                pos,
                loaded.options,
                |partial, layer_idx, phase| {
                    let mut s = stream_cell.borrow_mut();
                    let mut r = remote_cell.borrow_mut();
                    let sync_start = Instant::now();
                    let (tag, _) = read_msg(*s, PARTIAL_MAGIC, *r)?;
                    if tag != reduce_tag(layer_idx, phase) {
                        return Err(format!("reduce tag mismatch at layer {layer_idx}"));
                    }
                    // Fixed order: shard0 partial + shard1 partial.
                    for (local, &rem) in partial.iter_mut().zip(r.iter()) {
                        *local += rem;
                    }
                    write_msg(*s, SUM_MAGIC, tag, 0, partial)?;
                    **sync_cell.borrow_mut() += sync_start.elapsed();
                    Ok(())
                },
            )?;
        }
        compute_total += compute_start.elapsed();

        let logits_start = Instant::now();
        let next = if let Some(head) = head.as_mut() {
            // Row-parallel head: local slice argmax merged with the
            // worker's; ties go to the lower row range (master), which
            // reproduces the full-scan first-max rule.
            let local_hidden = ws.hidden.clone();
            let (local_idx, local_logit) = tp::head_shard_argmax(
                head,
                &local_hidden,
                &loaded.weights.output_norm,
                &mut rt,
                &mut ws,
                loaded.config.rms_norm_epsilon,
                loaded.options,
            );
            let (remote_idx, remote_bits) = read_msg(&mut stream, ARGMAX_MAGIC, &mut [])?;
            let remote_logit = f32::from_bits(remote_bits);
            if remote_logit > local_logit { remote_idx } else { local_idx }
        } else {
            inference::compute_logits_from_hidden(
                &loaded.config,
                &loaded.weights.token_embeddings,
                &loaded.weights.output_norm,
                loaded.weights.output_projection.as_ref(),
                &mut ws,
                loaded.options,
            );
            inference::sample_logits(&ws.logits, 0.0) as u32
        };
        logits_total += logits_start.elapsed();

        pos += 1;
        step += 1;
        if step < total_prompt {
            token = prompt_tokens[step];
            continue;
        }
        if step == total_prompt {
            decode_started = Some(Instant::now());
        }
        generated.push(next);
        token = next;
        if generated.len() >= max_tokens || pos >= loaded.config.context_length {
            break;
        }
    }
    write_msg(&mut stream, TOKEN_MAGIC, SHUTDOWN_POS, 0, &[])?;

    let decode_secs = decode_started
        .map(|s| s.elapsed().as_secs_f64())
        .unwrap_or_default();
    let steps = step as f64;
    println!("generated_tokens: {generated:?}");
    println!(
        "generated_text: {:?}",
        loaded.tokenizer.decode(&generated, true).unwrap_or_default()
    );
    println!(
        "json: {{\"benchmark\":\"tp-node\",\"role\":\"master\",\"shards\":2,\"prompt_tokens\":{},\"generated\":{},\"total_seconds\":{:.3},\"decode_tokens_per_sec\":{:.3},\"local_compute_avg_ms\":{:.2},\"sync_avg_ms\":{:.2},\"logits_avg_ms\":{:.2}}}",
        total_prompt,
        generated.len(),
        started.elapsed().as_secs_f64(),
        if decode_secs > 0.0 { generated.len() as f64 / decode_secs } else { 0.0 },
        (compute_total - sync_total).as_secs_f64() * 1000.0 / steps,
        sync_total.as_secs_f64() * 1000.0 / steps,
        logits_total.as_secs_f64() * 1000.0 / steps,
    );
    println!("result: PASS");
    Ok(())
}
