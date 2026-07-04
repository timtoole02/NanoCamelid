// N-node tensor parallelism over TCP (weighted, shard-direct).
//
// Shares are per-shard KV-head counts (e.g. "2,3,3" on an 8-KV-head model
// gives a half-speed node the small slice). The master is shard 0; workers
// are shards 1..N in address order. Every node loads ONLY its slice straight
// from the GGUF (peak memory = the shard); the master additionally holds the
// f32 embedding table and ships the embedded hidden state with each token,
// so workers never touch embeddings. Per layer, two reductions: each worker
// sends its partial, the master sums in fixed shard order and broadcasts.
// The LM head is row-parallel: every node reports its slice argmax and the
// master merges (ties resolve to the lowest row range = full-scan rule).
//
//   cluster_tp_node worker      <model.gguf> <bind_addr> <shard_idx> <shares>
//   cluster_tp_node master-chat <model.gguf> <worker1,worker2,...> <shares> "<prompt>" [max_tokens]
//   cluster_tp_node reference   <model.gguf> "<prompt>" [max_tokens]
//
// NANOCAMELID_TP_PARITY=1 pins the scalar Q8 kernel and keeps shards in the
// row-major class; `reference` honors the same flag.

use std::cell::RefCell;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use nanocamelid::{gguf, inference, model, q8, tokenizer, tp};

const HELLO_MAGIC: u32 = 0x5450_4E32; // "TPN2"
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

fn parity_mode() -> bool {
    std::env::var("NANOCAMELID_TP_PARITY").is_ok_and(|v| v == "1")
}

fn runtime_options(gguf: &gguf::GgufFile) -> inference::LlamaRuntimeOptions {
    let selector = if parity_mode() {
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

fn parse_shares(csv: &str) -> Result<Vec<usize>, String> {
    csv.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|_| format!("bad share {part:?}"))
        })
        .collect()
}

fn write_msg<W: Write>(
    w: &mut W,
    magic: u32,
    a: u32,
    b: u32,
    payload: &[f32],
) -> Result<(), String> {
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

struct LoadedBase {
    config: model::LlamaModelConfig,
    gguf: gguf::GgufFile,
    tokenizer: tokenizer::Tokenizer,
    options: inference::LlamaRuntimeOptions,
}

fn load_base(model_path: &str) -> Result<LoadedBase, String> {
    // Shard slicing and the reference both work on row-major blocks.
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
    Ok(LoadedBase {
        config,
        gguf,
        tokenizer: tok,
        options,
    })
}

fn encode_chat(base: &LoadedBase, prompt: &str) -> Result<Vec<u32>, String> {
    let rendered = base
        .tokenizer
        .render_chat_prompt(&[tokenizer::ChatMessage {
            role: "user",
            content: prompt,
        }]);
    base.tokenizer
        .encode(&rendered.text, rendered.add_special, rendered.parse_special)
        .map_err(|e| e.to_string())
}

fn is_stop(base: &LoadedBase, token: u32) -> bool {
    Some(token) == base.tokenizer.special.eos
        || Some(token) == base.tokenizer.special.eot
        || Some(token) == base.tokenizer.special.eom
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("worker") => {
            let model_path = args.next().ok_or("missing model path")?;
            let bind = args.next().ok_or("missing bind addr")?;
            let shard_idx: usize = args
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or("missing shard idx")?;
            let shares = parse_shares(&args.next().ok_or("missing shares")?)?;
            run_worker(&model_path, &bind, shard_idx, &shares)
        }
        Some("master-chat") => {
            let model_path = args.next().ok_or("missing model path")?;
            let workers: Vec<String> = args
                .next()
                .ok_or("missing worker list")?
                .split(',')
                .map(|s| s.trim().to_owned())
                .collect();
            let shares = parse_shares(&args.next().ok_or("missing shares")?)?;
            let prompt = args.next().ok_or("missing prompt")?;
            let max_tokens: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(64);
            run_master(&model_path, &workers, &shares, &prompt, max_tokens)
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

fn run_reference(model_path: &str, prompt: &str, max_tokens: usize) -> Result<(), String> {
    let base = load_base(model_path)?;
    let prompt_tokens = encode_chat(&base, prompt)?;
    println!("Loading full weights (reference)...");
    let weights = model::LlamaWeights::load(Path::new(model_path), &base.config, &base.gguf)?;
    let mut cache = inference::LlamaKvCache::new(
        base.config.block_count,
        base.config.context_length,
        base.config.kv_width,
    );
    let mut ws = inference::LlamaWorkspace::new(&base.config);
    let mut generated = Vec::new();
    let mut pos = 0usize;
    let mut decode_started = None;
    for (i, &t) in prompt_tokens.iter().enumerate() {
        inference::forward_pass(
            t as usize,
            pos,
            &base.config,
            &weights,
            &mut cache,
            &mut ws,
            base.options,
        );
        pos += 1;
        if i + 1 == prompt_tokens.len() {
            decode_started = Some(Instant::now());
        }
    }
    let mut next = inference::sample_logits(&ws.logits, 0.0) as u32;
    while generated.len() < max_tokens && pos < base.config.context_length {
        if is_stop(&base, next) {
            break;
        }
        generated.push(next);
        inference::forward_pass(
            next as usize,
            pos,
            &base.config,
            &weights,
            &mut cache,
            &mut ws,
            base.options,
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
        base.tokenizer.decode(&generated, true).unwrap_or_default()
    );
    println!(
        "json: {{\"benchmark\":\"tp-reference\",\"prompt_tokens\":{},\"generated\":{},\"decode_tokens_per_sec\":{:.3}}}",
        prompt_tokens.len(),
        generated.len(),
        if decode_secs > 0.0 {
            generated.len() as f64 / decode_secs
        } else {
            0.0
        },
    );
    println!("result: PASS");
    Ok(())
}

fn run_worker(
    model_path: &str,
    bind: &str,
    shard_idx: usize,
    shares: &[usize],
) -> Result<(), String> {
    let base = load_base(model_path)?;
    println!("Loading shard {shard_idx} of {shares:?} (direct)...");
    let mut node = tp::load_tp_shard_direct(
        Path::new(model_path),
        &base.config,
        &base.gguf,
        shares,
        shard_idx,
        !parity_mode(),
    )?;
    let emb = base.config.embedding_length;

    let listener = TcpListener::bind(bind).map_err(|e| e.to_string())?;
    println!("tp worker (shard {shard_idx}) listening on {bind}");
    let (mut stream, peer) = listener.accept().map_err(|e| e.to_string())?;
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    println!("master connected: {peer}");
    write_msg(
        &mut stream,
        HELLO_MAGIC,
        shard_idx as u32,
        base.config.block_count as u32,
        &[],
    )?;

    let mut rt = tp::TpRuntime::new(&base.config);
    let mut hidden = vec![0.0_f32; emb];
    let mut head_ws = inference::LlamaWorkspace::new(&base.config);
    let mut compute_total = Duration::ZERO;
    let mut wait_total = Duration::ZERO;
    let mut tokens = 0u64;

    loop {
        let wait_start = Instant::now();
        let (pos, _token) = read_msg(&mut stream, TOKEN_MAGIC, &mut hidden)?;
        wait_total += wait_start.elapsed();
        if pos == SHUTDOWN_POS {
            break;
        }
        tokens += 1;
        let compute_start = Instant::now();
        let stream_cell = RefCell::new(&mut stream);
        tp::tp_forward_token_reduced(
            &mut hidden,
            std::slice::from_mut(&mut node.shard),
            &mut rt,
            pos as usize,
            base.options,
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
        let (idx, logit) = tp::head_shard_argmax(
            &mut node.head,
            &hidden,
            &node.output_norm,
            &mut rt,
            &mut head_ws,
            base.config.rms_norm_epsilon,
            base.options,
        );
        write_msg(&mut stream, ARGMAX_MAGIC, idx, logit.to_bits(), &[])?;
        compute_total += compute_start.elapsed();
    }

    println!(
        "json: {{\"benchmark\":\"tp-node\",\"role\":\"worker\",\"shard\":{shard_idx},\"tokens\":{tokens},\"compute_total_ms\":{:.1},\"wait_total_ms\":{:.1}}}",
        compute_total.as_secs_f64() * 1000.0,
        wait_total.as_secs_f64() * 1000.0,
    );
    println!("result: WORKER_DONE");
    Ok(())
}

fn run_master(
    model_path: &str,
    worker_addrs: &[String],
    shares: &[usize],
    prompt: &str,
    max_tokens: usize,
) -> Result<(), String> {
    let base = load_base(model_path)?;
    if shares.len() != worker_addrs.len() + 1 {
        return Err(format!(
            "{} shares for master + {} workers",
            shares.len(),
            worker_addrs.len()
        ));
    }
    let prompt_tokens = encode_chat(&base, prompt)?;
    println!("Loading shard 0 of {shares:?} (direct)...");
    let mut node = tp::load_tp_shard_direct(
        Path::new(model_path),
        &base.config,
        &base.gguf,
        shares,
        0,
        !parity_mode(),
    )?;
    let emb = base.config.embedding_length;
    println!("Loading embedding table...");
    let embeddings = tp::load_embeddings_f32(
        Path::new(model_path),
        &base.gguf,
        base.config.vocab_size,
        emb,
    )?;

    let mut streams = Vec::with_capacity(worker_addrs.len());
    for (i, addr) in worker_addrs.iter().enumerate() {
        let mut stream = TcpStream::connect(addr).map_err(|e| format!("{addr}: {e}"))?;
        stream.set_nodelay(true).map_err(|e| e.to_string())?;
        let (idx, blocks) = read_msg(&mut stream, HELLO_MAGIC, &mut [])?;
        if idx as usize != i + 1 || blocks as usize != base.config.block_count {
            return Err(format!(
                "worker {addr} hello mismatch (shard {idx}, {blocks} layers)"
            ));
        }
        println!("worker {addr} ok (shard {idx})");
        streams.push(stream);
    }

    let mut rt = tp::TpRuntime::new(&base.config);
    let mut ws = inference::LlamaWorkspace::new(&base.config);
    let mut head_ws = inference::LlamaWorkspace::new(&base.config);
    let mut remote = vec![0.0_f32; emb];
    let mut generated = Vec::new();
    let mut compute_total = Duration::ZERO;
    let mut sync_total = Duration::ZERO;
    let mut logits_total = Duration::ZERO;
    let mut decode_started = None;
    let mut pos = 0usize;
    let mut token = *prompt_tokens.first().ok_or("empty prompt")?;
    let total_prompt = prompt_tokens.len();
    let mut step = 0usize;

    loop {
        let compute_start = Instant::now();
        let emb_start = token as usize * emb;
        ws.hidden
            .copy_from_slice(&embeddings[emb_start..emb_start + emb]);
        // Ship the embedded hidden with the token so workers skip embeddings.
        for stream in streams.iter_mut() {
            write_msg(stream, TOKEN_MAGIC, pos as u32, token, &ws.hidden)?;
        }
        {
            let streams_cell = RefCell::new(&mut streams);
            let remote_cell = RefCell::new(&mut remote);
            let sync_cell = RefCell::new(&mut sync_total);
            tp::tp_forward_token_reduced(
                &mut ws.hidden,
                std::slice::from_mut(&mut node.shard),
                &mut rt,
                pos,
                base.options,
                |partial, layer_idx, phase| {
                    let mut ss = streams_cell.borrow_mut();
                    let mut r = remote_cell.borrow_mut();
                    let sync_start = Instant::now();
                    let tag = reduce_tag(layer_idx, phase);
                    // Fixed order: shard0 + shard1 + shard2 ...
                    for stream in ss.iter_mut() {
                        let (got, _) = read_msg(stream, PARTIAL_MAGIC, *r)?;
                        if got != tag {
                            return Err(format!("reduce tag mismatch at layer {layer_idx}"));
                        }
                        for (local, &rem) in partial.iter_mut().zip(r.iter()) {
                            *local += rem;
                        }
                    }
                    for stream in ss.iter_mut() {
                        write_msg(stream, SUM_MAGIC, tag, 0, partial)?;
                    }
                    **sync_cell.borrow_mut() += sync_start.elapsed();
                    Ok(())
                },
            )?;
        }
        compute_total += compute_start.elapsed();

        let logits_start = Instant::now();
        let (mut best_idx, mut best_logit) = tp::head_shard_argmax(
            &mut node.head,
            &ws.hidden,
            &node.output_norm,
            &mut rt,
            &mut head_ws,
            base.config.rms_norm_epsilon,
            base.options,
        );
        for stream in streams.iter_mut() {
            let (idx, bits) = read_msg(stream, ARGMAX_MAGIC, &mut [])?;
            let logit = f32::from_bits(bits);
            // Strict greater: ties resolve to the lowest row range.
            if logit > best_logit {
                best_logit = logit;
                best_idx = idx;
            }
        }
        let next = best_idx;
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
        if is_stop(&base, next) {
            break;
        }
        generated.push(next);
        token = next;
        if generated.len() >= max_tokens || pos >= base.config.context_length {
            break;
        }
    }
    let zeros = vec![0.0_f32; emb];
    for stream in streams.iter_mut() {
        write_msg(stream, TOKEN_MAGIC, SHUTDOWN_POS, 0, &zeros)?;
    }

    let decode_secs = decode_started
        .map(|s| s.elapsed().as_secs_f64())
        .unwrap_or_default();
    let steps = step as f64;
    println!("generated_tokens: {generated:?}");
    println!(
        "generated_text: {:?}",
        base.tokenizer.decode(&generated, true).unwrap_or_default()
    );
    println!(
        "json: {{\"benchmark\":\"tp-node\",\"role\":\"master\",\"shards\":{},\"shares\":{shares:?},\"prompt_tokens\":{},\"generated\":{},\"decode_tokens_per_sec\":{:.3},\"local_compute_avg_ms\":{:.2},\"sync_avg_ms\":{:.2},\"logits_avg_ms\":{:.2}}}",
        shares.len(),
        total_prompt,
        generated.len(),
        if decode_secs > 0.0 {
            generated.len() as f64 / decode_secs
        } else {
            0.0
        },
        (compute_total - sync_total).as_secs_f64() * 1000.0 / steps,
        sync_total.as_secs_f64() * 1000.0 / steps,
        logits_total.as_secs_f64() * 1000.0 / steps,
    );
    println!("result: PASS");
    Ok(())
}
