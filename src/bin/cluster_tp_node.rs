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

include!(concat!(env!("OUT_DIR"), "/webui_assets.rs"));

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

fn normalized_model_id(filename: &str) -> String {
    let stem = filename.strip_suffix(".gguf").unwrap_or(filename).to_lowercase();
    let mut out = String::new();
    let mut last_us = false;
    for c in stem.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            last_us = false;
        } else if !last_us {
            out.push('_');
            last_us = true;
        }
    }
    out.trim_matches('_').to_owned()
}

fn write_json_response(client: &mut TcpStream, status: &str, body: &str) {
    let _ = write!(
        client,
        "HTTP/1.1 {status}
Content-Type: application/json
Access-Control-Allow-Origin: *
Content-Length: {}
Connection: close

{body}",
        body.len()
    );
}

/// Host identity for the cluster topology page: (cpu_model, threads, platform).
fn host_specs() -> (String, usize, String) {
    let platform = std::fs::read_to_string("/proc/device-tree/model")
        .map(|s| s.trim_end_matches('\0').trim().to_owned())
        .unwrap_or_else(|_| format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH));
    let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name") || l.starts_with("Model"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_owned())
        })
        .unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (cpu_model, threads, platform)
}

fn health_json(model_file: &str) -> String {
    format!(
        "{{\"ok\":true,\"status\":\"ok\",\"engine\":\"nanocamelid\",\"active_model_id\":\"{id}\",\"loaded_now\":true,\"generation_ready\":true,\"model_dir\":\"cluster\",\"model_ready\":true}}",
        id = json_escape(model_file)
    )
}

fn capabilities_json(model_file: &str, role_note: &str) -> String {
    let (cpu_model, threads, platform) = host_specs();
    format!(
        "{{\"engine\":\"nanocamelid\",\"cpu_model\":\"{cpu}\",\"thread_count\":{threads},\"platform_label\":\"{plat}\",\"selected_backend\":\"neon_sdot\",\"model_compatibility\":[{{\"id\":\"{nid}\",\"family\":\"llama_bpe_decoder\",\"quantization\":\"q4_0\",\"status\":\"supported\",\"evidence\":\"three-Pi tensor-parallel cluster chat\",\"notes\":\"{note}\"}}],\"api_features\":[],\"planned_model_families\":[]}}",
        cpu = json_escape(&cpu_model),
        plat = json_escape(&platform),
        nid = normalized_model_id(model_file),
        note = json_escape(role_note),
    )
}

/// Best-effort LAN address of this node (the interface that reaches `peer`).
fn local_ip_toward(peer: &str) -> String {
    std::net::UdpSocket::bind("0.0.0.0:0")
        .ok()
        .and_then(|s| {
            s.connect(peer).ok()?;
            s.local_addr().ok()
        })
        .map(|a| a.ip().to_string())
        .unwrap_or_else(|| "127.0.0.1".to_owned())
}

struct TopoNode {
    id: &'static str,
    name: String,
    host: String,
    port: u16,
    roles: &'static str,
    command: String,
    x: i32,
    y: i32,
}

fn cluster_import_json(
    model_path: &str,
    model_file: &str,
    worker_addrs: &[String],
    shares: &[usize],
    serve_port: u16,
) -> String {
    let master_ip = local_ip_toward(
        worker_addrs.first().map(String::as_str).unwrap_or("8.8.8.8:80"),
    );
    let mut nodes = vec![TopoNode {
        id: "camelid-tp-master",
        name: format!("master · shard 0 (share {})", shares.first().copied().unwrap_or(0)),
        host: master_ip.clone(),
        port: serve_port,
        roles: "\"coordinator\",\"gateway\",\"model_host\"",
        command: format!(
            "cluster_tp_node master-serve {model_path} {} {} {serve_port}",
            worker_addrs.join(","),
            shares.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
        ),
        x: 120,
        y: 200,
    }];
    for (i, addr) in worker_addrs.iter().enumerate() {
        let host = addr.split(':').next().unwrap_or(addr).to_owned();
        let ids: &'static str = match i {
            0 => "camelid-tp-worker-1",
            _ => "camelid-tp-worker-2",
        };
        nodes.push(TopoNode {
            id: ids,
            name: format!(
                "worker · shard {} (share {})",
                i + 1,
                shares.get(i + 1).copied().unwrap_or(0)
            ),
            host,
            port: 8181,
            roles: "\"worker\",\"model_host\"",
            command: format!(
                "cluster_tp_node worker {model_path} 0.0.0.0:5921 {} {}",
                i + 1,
                shares.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
            ),
            x: 460,
            y: 80 + (i as i32) * 240,
        });
    }
    let node_json: Vec<String> = nodes
        .iter()
        .map(|n| {
            format!(
                "{{\"id\":\"{id}\",\"display_name\":\"{name}\",\"node_type\":\"linux\",\"hostname\":\"{host}\",\"ip_address\":\"{host}\",\"port\":{port},\"connection_method\":\"manual\",\"roles\":[{roles}],\"os\":\"linux\",\"arch\":\"aarch64\",\"cpu_cores\":4,\"ram_gb\":16,\"model_paths\":[\"{mp}\"],\"worker_command\":\"{cmd}\",\"layout_x\":{x},\"layout_y\":{y},\"notes\":\"{model} tensor-parallel\"}}",
                id = n.id,
                name = json_escape(&n.name),
                host = json_escape(&n.host),
                port = n.port,
                roles = n.roles,
                mp = json_escape(model_path),
                cmd = json_escape(&n.command),
                x = n.x,
                y = n.y,
                model = json_escape(model_file),
            )
        })
        .collect();
    let mut connections = Vec::new();
    for i in 0..worker_addrs.len() {
        connections.push(format!(
            "{{\"source_node_id\":\"camelid-tp-master\",\"target_node_id\":\"camelid-tp-worker-{}\",\"label\":\"TP all-reduce\"}}",
            i + 1
        ));
    }
    format!(
        "[{{\"import_id\":\"tp-{nid}-{sh}\",\"nodes\":[{nodes}],\"connections\":[{conns}]}}]",
        nid = normalized_model_id(model_file),
        sh = shares.iter().map(usize::to_string).collect::<Vec<_>>().join("-"),
        nodes = node_json.join(","),
        conns = connections.join(","),
    )
}

/// Tiny read-only status endpoint every cluster node exposes so the web
/// UI's topology page can see it as online with real specs. Runs on its own
/// thread; serves /v1/health and /api/capabilities with permissive CORS.
fn spawn_status_listener(port: u16, model_file: String, role_note: String) {
    std::thread::spawn(move || {
        let Ok(listener) = TcpListener::bind(("0.0.0.0", port)) else {
            eprintln!("status listener: port {port} unavailable");
            return;
        };
        println!("status listener on :{port}");
        for client in listener.incoming() {
            let Ok(mut client) = client else { continue };
            let mut buf = [0_u8; 2048];
            let Ok(n) = client.read(&mut buf) else { continue };
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let first = req.lines().next().unwrap_or("");
            let body = if first.starts_with("GET /v1/health") {
                health_json(&model_file)
            } else if first.starts_with("GET /api/capabilities") {
                capabilities_json(&model_file, &role_note)
            } else if first.starts_with("OPTIONS") {
                let _ = write!(
                    client,
                    "HTTP/1.1 204 No Content
Access-Control-Allow-Origin: *
Access-Control-Allow-Headers: *
Connection: close

"
                );
                continue;
            } else {
                String::from("[]")
            };
            let _ = write!(
                client,
                "HTTP/1.1 200 OK
Content-Type: application/json
Access-Control-Allow-Origin: *
Content-Length: {}
Connection: close

{}",
                body.len(),
                body
            );
        }
    });
}

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
        Some("master-serve") => {
            let model_path = args.next().ok_or("missing model path")?;
            let workers: Vec<String> = args
                .next()
                .ok_or("missing worker list")?
                .split(',')
                .map(|s| s.trim().to_owned())
                .collect();
            let shares = parse_shares(&args.next().ok_or("missing shares")?)?;
            let port: u16 = args.next().and_then(|v| v.parse().ok()).unwrap_or(8090);
            run_master_serve(&model_path, &workers, &shares, port)
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

    let status_model = Path::new(model_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| model_path.to_owned());
    spawn_status_listener(
        8181,
        status_model,
        format!("TP worker shard {shard_idx} of {shares:?}"),
    );

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

// ---------------------------------------------------------------------------
// master-serve: a minimal single-session HTTP chat front end over the TP
// cluster. GET / returns an embedded chat page; POST /chat {"prompt": "..."}
// streams the reply as chunked plaintext while the cluster decodes. One
// request at a time; each request is a fresh single-turn chat (positions
// restart at 0, which is safe: every attended KV row is rewritten first).

fn http_chunk<W: Write>(w: &mut W, data: &[u8]) -> Result<(), String> {
    write!(w, "{:x}\r\n", data.len()).map_err(|e| e.to_string())?;
    w.write_all(data).map_err(|e| e.to_string())?;
    w.write_all(b"\r\n").map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())
}

const CHAT_PAGE: &str = r#"<!doctype html><meta charset="utf-8">
<title>NanoCamelid TP cluster</title>
<style>body{font-family:system-ui;max-width:720px;margin:2rem auto;padding:0 1rem}
textarea{width:100%;height:4rem;font-size:1rem}button{font-size:1rem;padding:.4rem 1.2rem;margin:.5rem 0}
#out{white-space:pre-wrap;border:1px solid #ccc;border-radius:8px;padding:1rem;min-height:6rem;font-size:1.05rem}
#meta{color:#666;font-size:.85rem}</style>
<h2>Llama 3 70B — three Raspberry Pi 5s, tensor parallel</h2>
<textarea id="p">Explain in one sentence why the sky appears blue during the day.</textarea><br>
<button id="go">Generate</button> <span id="meta"></span>
<div id="out"></div>
<script>
const go=document.getElementById('go'),out=document.getElementById('out'),meta=document.getElementById('meta');
go.onclick=async()=>{go.disabled=true;out.textContent='';meta.textContent='prompt ingest (about 1.5s per prompt token)...';
const t0=Date.now();let n=0;
try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:document.getElementById('p').value,max_tokens:96})});
const rd=r.body.getReader(),dec=new TextDecoder();
while(true){const{done,value}=await rd.read();if(done)break;const s=dec.decode(value);out.textContent+=s;n+=1;
meta.textContent=`streaming — ${((Date.now()-t0)/1000).toFixed(0)}s elapsed`;}
meta.textContent=`done in ${((Date.now()-t0)/1000).toFixed(0)}s`;}catch(e){meta.textContent='error: '+e;}
go.disabled=false;};
</script>"#;

#[allow(clippy::too_many_lines)]
fn run_master_serve(
    model_path: &str,
    worker_addrs: &[String],
    shares: &[usize],
    port: u16,
) -> Result<(), String> {
    let base = load_base(model_path)?;
    if shares.len() != worker_addrs.len() + 1 {
        return Err("share count must be workers + 1".to_owned());
    }
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
            return Err(format!("worker {addr} hello mismatch"));
        }
        println!("worker {addr} ok (shard {idx})");
        streams.push(stream);
    }

    let mut rt = tp::TpRuntime::new(&base.config);
    let mut ws = inference::LlamaWorkspace::new(&base.config);
    let mut head_ws = inference::LlamaWorkspace::new(&base.config);
    let mut remote = vec![0.0_f32; emb];

    let listener = TcpListener::bind(("0.0.0.0", port)).map_err(|e| e.to_string())?;
    println!("master-serve listening on http://0.0.0.0:{port}/ (single session)");
    println!("result: SERVE_READY");

    for client in listener.incoming() {
        let Ok(mut client) = client else { continue };
        let _ = client.set_nodelay(true);
        // Minimal HTTP request parse: header block, then Content-Length body.
        let mut buf = Vec::new();
        let mut tmp = [0_u8; 4096];
        let header_end = loop {
            let Ok(n) = client.read(&mut tmp) else { break None };
            if n == 0 {
                break None;
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(idx) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break Some(idx + 4);
            }
            if buf.len() > 65536 {
                break None;
            }
        };
        let Some(header_end) = header_end else { continue };
        let header_text = String::from_utf8_lossy(&buf[..header_end]).to_string();
        let first_line = header_text.lines().next().unwrap_or("").to_owned();

        let mut parts = first_line.split_whitespace();
        let method = parts.next().unwrap_or("");
        let path = parts.next().unwrap_or("/").split('?').next().unwrap_or("/");
        let model_file = Path::new(model_path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| model_path.to_owned());

        if method == "OPTIONS" {
            let _ = write!(client, "HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n");
            continue;
        }
        if method == "GET" {
            match path {
                "/cluster-import.json" => {
                    write_json_response(
                        &mut client,
                        "200 OK",
                        &cluster_import_json(model_path, &model_file, worker_addrs, shares, port),
                    );
                    continue;
                }
                "/__camelid/cluster/discover" => {
                    let import = cluster_import_json(model_path, &model_file, worker_addrs, shares, port);
                    write_json_response(
                        &mut client,
                        "200 OK",
                        &format!("{{\"available\":true,\"devices\":{}}}",
                            // devices = the import's node list
                            import
                                .split("\"nodes\":")
                                .nth(1)
                                .and_then(|s| s.split(",\"connections\"").next())
                                .unwrap_or("[]")),
                    );
                    continue;
                }
                "/v1/health" => {
                    write_json_response(&mut client, "200 OK", &health_json(&model_file));
                    continue;
                }
                "/api/capabilities" => {
                    write_json_response(
                        &mut client,
                        "200 OK",
                        &capabilities_json(&model_file, "TP master (coordinator + web UI)"),
                    );
                    continue;
                }
                "/v1/models" => {
                    write_json_response(&mut client, "200 OK", &format!(
                        "{{\"object\":\"list\",\"data\":[{{\"id\":\"{id}\",\"object\":\"model\",\"created\":0,\"owned_by\":\"nanocamelid\"}}]}}",
                        id = json_escape(&model_file)));
                    continue;
                }
                "/api/models/current" => {
                    write_json_response(&mut client, "200 OK", &format!(
                        "{{\"id\":\"{id}\",\"name\":\"Llama 3 70B Instruct (3-Pi cluster)\",\"filename\":\"{id}\",\"model_path\":\"{p}\",\"path\":\"{p}\",\"gguf\":{{\"metadata\":{{\"general.file_type\":2}}}},\"quant\":\"Q4_0\",\"runtime_model_name\":\"{id}\",\"loaded_now\":true,\"generation_ready\":true}}",
                        id = json_escape(&model_file), p = json_escape(model_path)));
                    continue;
                }
                "/api/models/local" => {
                    write_json_response(&mut client, "200 OK", &format!(
                        "{{\"models\":[{{\"id\":\"{id}\",\"filename\":\"{id}\",\"runtime_model_name\":\"{id}\",\"path\":\"{p}\",\"model_path\":\"{p}\",\"bytes\":0,\"quant\":\"Q4_0\",\"family\":\"llama\",\"status\":\"ready\"}}]}}",
                        id = json_escape(&model_file), p = json_escape(model_path)));
                    continue;
                }
                p if p.starts_with("/api/") || p.starts_with("/metrics") => {
                    write_json_response(&mut client, "200 OK", "[]");
                    continue;
                }
                _ => {
                    // Static webui assets with SPA index fallback.
                    let rel = path.trim_start_matches('/');
                    let hit = WEBUI_ASSETS
                        .iter()
                        .find(|&&(p, _, _)| p == rel)
                        .or_else(|| WEBUI_ASSETS.iter().find(|&&(p, _, _)| p == "index.html"));
                    if let Some(&(_, mime, data)) = hit {
                        let _ = write!(
                            client,
                            "HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            data.len()
                        );
                        let _ = client.write_all(data);
                    } else {
                        let body = CHAT_PAGE.as_bytes();
                        let _ = write!(
                            client,
                            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = client.write_all(body);
                    }
                    continue;
                }
            }
        }
        if method == "POST" && path == "/__camelid/cluster/probe" {
            let content_length = header_text
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    k.eq_ignore_ascii_case("content-length")
                        .then(|| v.trim().parse::<usize>().ok())?
                })
                .unwrap_or(0)
                .min(4096);
            let mut body = buf[header_end..].to_vec();
            while body.len() < content_length {
                let Ok(n) = client.read(&mut tmp) else { break };
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&tmp[..n]);
            }
            let text = String::from_utf8_lossy(&body).to_string();
            let host = extract_json_string(&text, "host").unwrap_or_default();
            let probe_port = extract_json_usize(&text, "port").unwrap_or(8181) as u16;
            let started = Instant::now();
            let reachable = std::net::TcpStream::connect_timeout(
                &format!("{host}:{probe_port}")
                    .parse()
                    .unwrap_or_else(|_| "127.0.0.1:1".parse().unwrap()),
                Duration::from_millis(1500),
            )
            .is_ok();
            let latency = started.elapsed().as_secs_f64() * 1000.0;
            write_json_response(
                &mut client,
                "200 OK",
                &format!("{{\"reachable\":{reachable},\"latencyMs\":{latency:.1}}}"),
            );
            continue;
        }
        let is_completion = method == "POST"
            && (path == "/v1/chat/completions" || path == "/v1/completions");
        if !(is_completion || (method == "POST" && path == "/chat")) {
            let _ = write!(
                client,
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            continue;
        }

        let content_length = header_text
            .lines()
            .find_map(|l| {
                let (k, v) = l.split_once(':')?;
                k.eq_ignore_ascii_case("content-length")
                    .then(|| v.trim().parse::<usize>().ok())?
            })
            .unwrap_or(0)
            .min(65536);
        let mut body = buf[header_end..].to_vec();
        while body.len() < content_length {
            let Ok(n) = client.read(&mut tmp) else { break };
            if n == 0 {
                break;
            }
            body.extend_from_slice(&tmp[..n]);
        }
        let body_text = String::from_utf8_lossy(&body).to_string();
        // Tolerant JSON-ish field extraction (no JSON dep in this crate).
        let prompt = if is_completion {
            extract_last_content(&body_text)
                .or_else(|| extract_json_string(&body_text, "prompt"))
                .unwrap_or_else(|| "Say hello in one sentence.".to_owned())
        } else {
            extract_json_string(&body_text, "prompt")
                .unwrap_or_else(|| "Say hello in one sentence.".to_owned())
        };
        // The cluster decodes ~0.7 tok/s; cap replies so the UI never waits
        // more than a couple of minutes.
        let max_tokens = extract_json_usize(&body_text, "max_tokens").unwrap_or(96).min(96);

        if is_completion {
            let mut collected = String::new();
            let result = serve_completion_collect(
                &base,
                &mut node,
                &embeddings,
                &mut streams,
                &mut rt,
                &mut ws,
                &mut head_ws,
                &mut remote,
                &prompt,
                max_tokens,
                &mut collected,
            );
            match result {
                Ok((pt, ct)) => {
                    let json = format!(
                        "{{\"id\":\"chatcmpl-nano-tp\",\"object\":\"chat.completion\",\"created\":0,\"model\":\"{model}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{content}\"}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":{pt},\"completion_tokens\":{ct},\"total_tokens\":{tt}}}}}",
                        model = json_escape(&model_file),
                        content = json_escape(&collected),
                        tt = pt + ct
                    );
                    write_json_response(&mut client, "200 OK", &json);
                }
                Err(err) => {
                    eprintln!("completion failed: {err}");
                    write_json_response(
                        &mut client,
                        "500 Internal Server Error",
                        &format!("{{\"error\":\"{}\"}}", json_escape(&err)),
                    );
                    return Err(err);
                }
            }
            continue;
        }

        let _ = write!(
            client,
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
        );

        match serve_one_request(
            &base,
            &mut node,
            &embeddings,
            &mut streams,
            &mut rt,
            &mut ws,
            &mut head_ws,
            &mut remote,
            &mut client,
            &prompt,
            max_tokens,
        ) {
            Ok(tokens_per_sec) => {
                let _ = http_chunk(
                    &mut client,
                    format!("\n\n[{tokens_per_sec:.2} tok/s across 3 Pis]").as_bytes(),
                );
            }
            Err(err) => {
                eprintln!("request failed: {err}");
                let _ = http_chunk(&mut client, format!("\n[error: {err}]").as_bytes());
                let _ = http_chunk(&mut client, b"");
                return Err(err);
            }
        }
        let _ = http_chunk(&mut client, b"");
    }
    Ok(())
}

fn extract_json_string(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let start = body.find(&pat)? + pat.len();
    let rest = &body[start..];
    let open = rest.find('"')? + 1;
    let mut out = String::new();
    let mut chars = rest[open..].chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                other => out.push(other),
            },
            other => out.push(other),
        }
    }
    None
}

/// Last "content" string in the request body (the newest chat message).
fn extract_last_content(body: &str) -> Option<String> {
    let mut last = None;
    let mut search = 0usize;
    while let Some(rel) = body[search..].find("\"content\"") {
        let abs = search + rel + "\"content\"".len();
        if let Some(s) = extract_json_string(&body[abs - "\"content\"".len()..], "content") {
            last = Some(s);
        }
        search = abs;
    }
    last
}

#[allow(clippy::too_many_arguments)]
fn serve_completion_collect(
    base: &LoadedBase,
    node: &mut tp::TpNodeShard,
    embeddings: &[f32],
    streams: &mut Vec<TcpStream>,
    rt: &mut tp::TpRuntime,
    ws: &mut inference::LlamaWorkspace,
    head_ws: &mut inference::LlamaWorkspace,
    remote: &mut Vec<f32>,
    prompt: &str,
    max_tokens: usize,
    collected: &mut String,
) -> Result<(usize, usize), String> {
    let emb = base.config.embedding_length;
    let prompt_tokens = encode_chat(base, prompt)?;
    let budget = base.config.context_length.saturating_sub(prompt_tokens.len() + 2);
    let max_tokens = max_tokens.min(budget);
    let mut generated: Vec<u32> = Vec::new();
    let mut pos = 0usize;
    let mut token = *prompt_tokens.first().ok_or("empty prompt")?;
    let total_prompt = prompt_tokens.len();
    let mut step = 0usize;
    loop {
        let emb_start = token as usize * emb;
        ws.hidden.copy_from_slice(&embeddings[emb_start..emb_start + emb]);
        for stream in streams.iter_mut() {
            write_msg(stream, TOKEN_MAGIC, pos as u32, token, &ws.hidden)?;
        }
        {
            let streams_cell = RefCell::new(&mut *streams);
            let remote_cell = RefCell::new(&mut *remote);
            tp::tp_forward_token_reduced(
                &mut ws.hidden,
                std::slice::from_mut(&mut node.shard),
                rt,
                pos,
                base.options,
                |partial, layer_idx, phase| {
                    let mut ss = streams_cell.borrow_mut();
                    let mut r = remote_cell.borrow_mut();
                    let tag = reduce_tag(layer_idx, phase);
                    for stream in ss.iter_mut() {
                        let (got, _) = read_msg(stream, PARTIAL_MAGIC, *r)?;
                        if got != tag {
                            return Err("reduce tag mismatch".to_owned());
                        }
                        for (local, &rem) in partial.iter_mut().zip(r.iter()) {
                            *local += rem;
                        }
                    }
                    for stream in ss.iter_mut() {
                        write_msg(stream, SUM_MAGIC, tag, 0, partial)?;
                    }
                    Ok(())
                },
            )?;
        }
        let (mut best_idx, mut best_logit) = tp::head_shard_argmax(
            &mut node.head,
            &ws.hidden,
            &node.output_norm,
            rt,
            head_ws,
            base.config.rms_norm_epsilon,
            base.options,
        );
        for stream in streams.iter_mut() {
            let (idx, bits) = read_msg(stream, ARGMAX_MAGIC, &mut [])?;
            let logit = f32::from_bits(bits);
            if logit > best_logit {
                best_logit = logit;
                best_idx = idx;
            }
        }
        pos += 1;
        step += 1;
        if step < total_prompt {
            token = prompt_tokens[step];
            continue;
        }
        if is_stop(base, best_idx) {
            break;
        }
        generated.push(best_idx);
        token = best_idx;
        if generated.len() >= max_tokens || pos >= base.config.context_length {
            break;
        }
    }
    *collected = base.tokenizer.decode(&generated, true).unwrap_or_default();
    println!("served completion: {} prompt, {} generated", total_prompt, generated.len());
    Ok((total_prompt, generated.len()))
}

fn extract_json_usize(body: &str, key: &str) -> Option<usize> {
    let pat = format!("\"{key}\"");
    let start = body.find(&pat)? + pat.len();
    let digits: String = body[start..]
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

#[allow(clippy::too_many_arguments)]
fn serve_one_request(
    base: &LoadedBase,
    node: &mut tp::TpNodeShard,
    embeddings: &[f32],
    streams: &mut Vec<TcpStream>,
    rt: &mut tp::TpRuntime,
    ws: &mut inference::LlamaWorkspace,
    head_ws: &mut inference::LlamaWorkspace,
    remote: &mut Vec<f32>,
    client: &mut TcpStream,
    prompt: &str,
    max_tokens: usize,
) -> Result<f64, String> {
    let emb = base.config.embedding_length;
    let prompt_tokens = encode_chat(base, prompt)?;
    let budget = base.config.context_length.saturating_sub(prompt_tokens.len() + 2);
    let max_tokens = max_tokens.min(budget);
    let mut generated: Vec<u32> = Vec::new();
    let mut printed = 0usize;
    let mut pos = 0usize;
    let mut token = *prompt_tokens.first().ok_or("empty prompt")?;
    let total_prompt = prompt_tokens.len();
    let mut step = 0usize;
    let mut decode_started: Option<Instant> = None;

    loop {
        let emb_start = token as usize * emb;
        ws.hidden
            .copy_from_slice(&embeddings[emb_start..emb_start + emb]);
        for stream in streams.iter_mut() {
            write_msg(stream, TOKEN_MAGIC, pos as u32, token, &ws.hidden)?;
        }
        {
            let streams_cell = RefCell::new(&mut *streams);
            let remote_cell = RefCell::new(&mut *remote);
            tp::tp_forward_token_reduced(
                &mut ws.hidden,
                std::slice::from_mut(&mut node.shard),
                rt,
                pos,
                base.options,
                |partial, layer_idx, phase| {
                    let mut ss = streams_cell.borrow_mut();
                    let mut r = remote_cell.borrow_mut();
                    let tag = reduce_tag(layer_idx, phase);
                    for stream in ss.iter_mut() {
                        let (got, _) = read_msg(stream, PARTIAL_MAGIC, *r)?;
                        if got != tag {
                            return Err("reduce tag mismatch".to_owned());
                        }
                        for (local, &rem) in partial.iter_mut().zip(r.iter()) {
                            *local += rem;
                        }
                    }
                    for stream in ss.iter_mut() {
                        write_msg(stream, SUM_MAGIC, tag, 0, partial)?;
                    }
                    Ok(())
                },
            )?;
        }
        let (mut best_idx, mut best_logit) = tp::head_shard_argmax(
            &mut node.head,
            &ws.hidden,
            &node.output_norm,
            rt,
            head_ws,
            base.config.rms_norm_epsilon,
            base.options,
        );
        for stream in streams.iter_mut() {
            let (idx, bits) = read_msg(stream, ARGMAX_MAGIC, &mut [])?;
            let logit = f32::from_bits(bits);
            if logit > best_logit {
                best_logit = logit;
                best_idx = idx;
            }
        }

        pos += 1;
        step += 1;
        if step < total_prompt {
            token = prompt_tokens[step];
            continue;
        }
        if step == total_prompt {
            decode_started = Some(Instant::now());
        }
        if is_stop(base, best_idx) {
            break;
        }
        generated.push(best_idx);
        // Stream newly stable decoded text to the browser.
        if let Ok(text) = base.tokenizer.decode(&generated, true)
            && text.len() > printed
        {
            let _ = http_chunk(client, text[printed..].as_bytes());
            printed = text.len();
        }
        token = best_idx;
        if generated.len() >= max_tokens || pos >= base.config.context_length {
            break;
        }
    }
    let secs = decode_started
        .map(|s| s.elapsed().as_secs_f64())
        .unwrap_or(1.0);
    println!(
        "served: {} prompt tokens, {} generated, {:.3} tok/s",
        total_prompt,
        generated.len(),
        generated.len() as f64 / secs
    );
    Ok(generated.len() as f64 / secs)
}
