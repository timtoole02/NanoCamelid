use std::{
    collections::BTreeSet,
    env, fs,
    hint::black_box,
    io::{self, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    time::{Duration, Instant},
};

use nanocamelid::{gguf, inference, model, q8, speculative, tokenizer};

const DEFAULT_MODEL_GGUF_ENV: &str = "NANOCAMELID_MODEL_GGUF";
const MODEL_DIR_ENV: &str = "NANOCAMELID_MODEL_DIR";
const SMOKE_MODEL_GGUF_ENV: &str = "NANOCAMELID_SMOKE_GGUF";
const WORKSPACE_ENV: &str = "NANOCAMELID_WORKSPACE";
const RAYON_THREADS_ENV: &str = "NANOCAMELID_RAYON_THREADS";
const WORKER_CORES_ENV: &str = "NANOCAMELID_WORKER_CORES";
const PREFILL_BATCH_ENV: &str = "NANOCAMELID_PREFILL_BATCH";
const CONTEXT_LIMIT_ENV: &str = "NANOCAMELID_CONTEXT_LIMIT";
const ROPE_CACHE_ENV: &str = inference::ROPE_CACHE_ENV;
const TRACE_ENV: &str = inference::TRACE_ENV;
const READY_CHAT_ENV: &str = "NANOCAMELID_READY_CHAT";
const SMOKE_KIND_ENV: &str = "NANOCAMELID_SMOKE_KIND";
const SMOKE_PROMPT_ENV: &str = "NANOCAMELID_SMOKE_PROMPT";
const SMOKE_TOKENS_ENV: &str = "NANOCAMELID_SMOKE_TOKENS";
const CONTEXT_PACKS_ENV: &str = "NANOCAMELID_CONTEXT_PACKS";
const READY_SMOKE_KIND_ENV: &str = "NANOCAMELID_READY_SMOKE_KIND";
const READY_SMOKE_PROMPT_ENV: &str = "NANOCAMELID_READY_SMOKE_PROMPT";
const READY_SMOKE_TOKENS_ENV: &str = "NANOCAMELID_READY_SMOKE_TOKENS";
const READY_PROMPT_ENV: &str = "NANOCAMELID_READY_PROMPT";
const READY_TOKENS_ENV: &str = "NANOCAMELID_READY_TOKENS";
const READY_TEMP_ENV: &str = "NANOCAMELID_READY_TEMP";
const API_KEY_ENV: &str = "NANOCAMELID_API_KEY";
const SERVE_MAX_REQUEST_BYTES_ENV: &str = "NANOCAMELID_MAX_REQUEST_BYTES";
const SERVE_MAX_INPUT_TOKENS_ENV: &str = "NANOCAMELID_MAX_INPUT_TOKENS";
const SERVE_MAX_OUTPUT_TOKENS_ENV: &str = "NANOCAMELID_MAX_OUTPUT_TOKENS";
const DEFAULT_PI_WORKSPACE: &str = "/mnt/nanocamelid";
const DEFAULT_API_HOST: &str = "127.0.0.1";
const DEFAULT_API_PORT: u16 = 8080;
const DEFAULT_SERVE_MAX_REQUEST_BYTES: usize = 65_536;
const DEFAULT_SERVE_MAX_INPUT_TOKENS: usize = 2048;
const DEFAULT_SERVE_MAX_OUTPUT_TOKENS: usize = 256;
const LLAMA32_1B_Q4_MODEL: &str = "Llama-3.2-1B-Instruct-Q4_0.gguf";
const LLAMA32_1B_Q8_MODEL: &str = "Llama-3.2-1B-Instruct-Q8_0.gguf";
const LLAMA32_3B_Q4_MODEL: &str = "Llama-3.2-3B-Instruct-Q4_0.gguf";
const DEFAULT_1B_SMOKE_PROMPT: &str = "Say hello in one sentence.";
const DEFAULT_1B_SMOKE_TOKENS: usize = 8;
const PREFILL_PROMPT_ENV: &str = "NANOCAMELID_PREFILL_PROMPT";
const PREFILL_TOKENS_ENV: &str = "NANOCAMELID_PREFILL_TOKENS";
const PREFILL_TEMP_ENV: &str = "NANOCAMELID_PREFILL_TEMP";
const PREFILL_BATCHES_ENV: &str = "NANOCAMELID_PREFILL_BATCHES";
const DEFAULT_1B_PREFILL_PROMPT: &str =
    "Explain one practical Raspberry Pi inference bottleneck in two short sentences.";
const DEFAULT_1B_PREFILL_TOKENS: usize = 2;
const DEFAULT_1B_PREFILL_TEMP: &str = "0.0";
const DEFAULT_1B_PREFILL_BATCHES: &str = "1,16,32,64";
const DEFAULT_1B_CONTEXT_PACKS: &str = "512,1024,2048,4096,8192";
const PERFORMANCE_GOVERNOR_COMMAND: &str =
    "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor";
const DEFAULT_RAYON_THREADS: usize = 4;
const DEFAULT_Q4_PREFILL_PROMPT_LEN: usize = 128;
const DEFAULT_Q4_PREFILL_BATCH: usize = 16;
const DEFAULT_Q4_PREFILL_RUNS: usize = 5;
const Q4_PREFILL_ROWS: usize = 3_584;
const Q4_PREFILL_COLS: usize = 3_584;

fn main() -> ExitCode {
    setup_thread_pool();

    let args = env::args().skip(1).collect::<Vec<_>>();

    if args.len() == 1 && matches!(args[0].as_str(), "-V" | "--version") {
        println!("nanocamelid {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }

    if let Some(help_topic) = help_topic_for_args(&args) {
        print_help(help_topic);
        return ExitCode::SUCCESS;
    }

    match args.first().map(String::as_str) {
        Some("help") => {
            let topic = args.get(1).map(String::as_str).unwrap_or_default();
            eprintln!("unknown help topic: {topic}");
            print_help(HelpTopic::TopLevel);
            ExitCode::from(2)
        }
        Some("model") => {
            if args.get(1).is_some_and(|arg| is_help_flag(arg)) {
                print_help(HelpTopic::Model);
                return ExitCode::SUCCESS;
            }

            match args.get(1).map(String::as_str) {
                Some(alias) if is_llama32_1b_alias(alias) => {
                    match parse_model_1b_args(&args[2..]) {
                        Ok(parsed) => run_model_1b_audit(parsed),
                        Err(err) => {
                            eprintln!("{err}");
                            print_help(HelpTopic::Model);
                            ExitCode::from(2)
                        }
                    }
                }
                Some(other) => {
                    eprintln!("unknown model audit target: {other}");
                    print_help(HelpTopic::Model);
                    ExitCode::from(2)
                }
                None => {
                    eprintln!("missing model audit target");
                    print_help(HelpTopic::Model);
                    ExitCode::from(2)
                }
            }
        }
        Some("models") => {
            if args.get(1).is_some_and(|arg| is_help_flag(arg)) {
                print_help(HelpTopic::Models);
                return ExitCode::SUCCESS;
            }

            match parse_models_args(&args[1..]) {
                Ok(parsed) => run_models(parsed),
                Err(err) => {
                    eprintln!("{err}");
                    print_help(HelpTopic::Models);
                    ExitCode::from(2)
                }
            }
        }
        Some("doctor") => {
            if args.get(1).is_some_and(|arg| is_help_flag(arg)) {
                print_help(HelpTopic::Doctor);
                return ExitCode::SUCCESS;
            }

            match parse_doctor_args(&args[1..]) {
                Ok(parsed) => run_doctor(parsed),
                Err(err) => {
                    eprintln!("{err}");
                    print_help(HelpTopic::Doctor);
                    ExitCode::from(2)
                }
            }
        }
        Some("serve") => {
            if args.get(1).is_some_and(|arg| is_help_flag(arg)) {
                print_help(HelpTopic::Serve);
                return ExitCode::SUCCESS;
            }

            match parse_serve_args(&args[1..]) {
                Ok(parsed) => run_serve(parsed),
                Err(err) => {
                    eprintln!("{err}");
                    print_help(HelpTopic::Serve);
                    ExitCode::from(2)
                }
            }
        }
        Some("probe") => {
            if args.get(1).is_some_and(|arg| is_help_flag(arg)) {
                print_help(HelpTopic::Probe);
                return ExitCode::SUCCESS;
            }
            print_probe();
            ExitCode::SUCCESS
        }
        Some("inspect") => {
            if args.get(1).is_some_and(|arg| is_help_flag(arg)) {
                print_help(HelpTopic::Inspect);
                return ExitCode::SUCCESS;
            }
            match parse_inspect_args(&args[1..]) {
                Ok(parsed) => run_inspect(parsed),
                Err(err) => {
                    eprintln!("{err}");
                    print_help(HelpTopic::Inspect);
                    ExitCode::from(2)
                }
            }
        }
        Some("generate") => {
            if args.get(1).is_some_and(|arg| is_help_flag(arg)) {
                print_help(HelpTopic::Generate);
                return ExitCode::SUCCESS;
            }

            match parse_generate_args(&args[1..]) {
                Ok(parsed) => {
                    if parsed.dry_run {
                        print_generation_dry_run("generate", &parsed)
                    } else {
                        run_generation_command(&parsed, |parsed| {
                            run_generation(
                                Path::new(&parsed.model_path),
                                &parsed.prompt,
                                parsed.temp,
                                parsed.max_tokens,
                                parsed.model_source,
                                parsed.audit_1b_shape,
                            )
                        })
                    }
                }
                Err(err) => {
                    eprintln!("{err}");
                    print_help(HelpTopic::Generate);
                    ExitCode::from(2)
                }
            }
        }
        Some("chat") => {
            if args.get(1).is_some_and(|arg| is_help_flag(arg)) {
                print_help(HelpTopic::Chat);
                return ExitCode::SUCCESS;
            }

            match parse_generate_args(&args[1..]) {
                Ok(parsed) => {
                    if parsed.dry_run {
                        print_generation_dry_run("chat", &parsed)
                    } else {
                        run_generation_command(&parsed, |parsed| {
                            run_chat(
                                Path::new(&parsed.model_path),
                                &parsed.prompt,
                                parsed.temp,
                                parsed.max_tokens,
                                parsed.model_source,
                                parsed.audit_1b_shape,
                            )
                        })
                    }
                }
                Err(err) => {
                    eprintln!("{err}");
                    print_help(HelpTopic::Chat);
                    ExitCode::from(2)
                }
            }
        }
        Some("tui") => {
            if args.get(1).is_some_and(|arg| is_help_flag(arg)) {
                print_help(HelpTopic::Tui);
                return ExitCode::SUCCESS;
            }

            match parse_tui_args(&args[1..]) {
                Ok(parsed) => {
                    if parsed.dry_run {
                        print_tui_dry_run(&parsed)
                    } else {
                        run_tui_command(&parsed)
                    }
                }
                Err(err) => {
                    eprintln!("{err}");
                    print_help(HelpTopic::Tui);
                    ExitCode::from(2)
                }
            }
        }
        Some("ready") => {
            if args.get(1).is_some_and(|arg| is_help_flag(arg)) {
                print_help(HelpTopic::Ready);
                return ExitCode::SUCCESS;
            }

            match args.get(1).map(String::as_str) {
                Some("1b" | "llama32-1b" | "llama-3.2-1b") => {
                    match parse_ready_1b_args(&args[2..]) {
                        Ok(parsed) => run_ready_1b(parsed),
                        Err(err) => {
                            eprintln!("{err}");
                            print_help(HelpTopic::Ready);
                            ExitCode::from(2)
                        }
                    }
                }
                Some(other) => {
                    eprintln!("unknown readiness target: {other}");
                    print_help(HelpTopic::Ready);
                    ExitCode::from(2)
                }
                None => {
                    eprintln!("missing readiness target");
                    print_help(HelpTopic::Ready);
                    ExitCode::from(2)
                }
            }
        }
        Some("evidence") => {
            if args.get(1).is_some_and(|arg| is_help_flag(arg)) {
                print_help(HelpTopic::Evidence);
                return ExitCode::SUCCESS;
            }

            match args.get(1).map(String::as_str) {
                Some(alias) if is_llama32_1b_alias(alias) => {
                    match parse_evidence_1b_args(&args[2..]) {
                        Ok(parsed) if parsed.dry_run => print_evidence_1b_dry_run(&parsed),
                        Ok(parsed) => run_evidence_1b(parsed),
                        Err(err) => {
                            eprintln!("{err}");
                            print_help(HelpTopic::Evidence);
                            ExitCode::from(2)
                        }
                    }
                }
                Some(other) => {
                    eprintln!("unknown evidence target: {other}");
                    print_help(HelpTopic::Evidence);
                    ExitCode::from(2)
                }
                None => {
                    eprintln!("missing evidence target");
                    print_help(HelpTopic::Evidence);
                    ExitCode::from(2)
                }
            }
        }
        Some("bench") => {
            if args.get(1).is_some_and(|arg| is_help_flag(arg)) {
                print_help(HelpTopic::Bench);
                return ExitCode::SUCCESS;
            }

            match args.get(1).map(String::as_str) {
                Some("q8-dot") => match parse_bench_q8_dot_args(&args[2..]) {
                    Ok(parsed) => bench_q8_dot(parsed.iterations, parsed.runs),
                    Err(err) => {
                        eprintln!("{err}");
                        print_help(HelpTopic::Bench);
                        ExitCode::from(2)
                    }
                },
                Some("q4-layout") => match parse_bench_q4_layout_args(&args[2..]) {
                    Ok(parsed) => bench_q4_layout(parsed.rows, parsed.cols, parsed.runs),
                    Err(err) => {
                        eprintln!("{err}");
                        print_help(HelpTopic::Bench);
                        ExitCode::from(2)
                    }
                },
                Some("q4-prefill") => match parse_bench_q4_prefill_args(&args[2..]) {
                    Ok(parsed) => {
                        bench_q4_prefill(parsed.prompt_len, parsed.batch_size, parsed.runs)
                    }
                    Err(err) => {
                        eprintln!("{err}");
                        print_help(HelpTopic::Bench);
                        ExitCode::from(2)
                    }
                },
                Some(alias) if is_llama32_1b_alias(alias) => {
                    match parse_bench_1b_args(&args[2..]) {
                        Ok(parsed) if parsed.dry_run => print_bench_1b_dry_run(&parsed),
                        Ok(parsed) => run_bench_1b_prefill(parsed),
                        Err(err) => {
                            eprintln!("{err}");
                            print_help(HelpTopic::Bench);
                            ExitCode::from(2)
                        }
                    }
                }
                Some(other) => {
                    eprintln!("unknown benchmark: {other}");
                    print_help(HelpTopic::Bench);
                    ExitCode::from(2)
                }
                None => {
                    eprintln!("missing benchmark name");
                    print_help(HelpTopic::Bench);
                    ExitCode::from(2)
                }
            }
        }
        Some("smoke") => {
            if args.get(1).is_some_and(|arg| is_help_flag(arg)) {
                print_help(HelpTopic::Smoke);
                return ExitCode::SUCCESS;
            }

            match args.get(1).map(String::as_str) {
                Some("q8-model") => match parse_smoke_args(&args[2..]) {
                    Ok(parsed) => smoke_q8_model(
                        Path::new(&parsed.model_path),
                        &parsed.prompt,
                        parsed.max_tokens,
                    ),
                    Err(err) => {
                        eprintln!("{err}");
                        print_help(HelpTopic::Smoke);
                        ExitCode::from(2)
                    }
                },
                Some("q8-chat") => match parse_smoke_args(&args[2..]) {
                    Ok(parsed) => smoke_q8_chat(
                        Path::new(&parsed.model_path),
                        &parsed.prompt,
                        parsed.max_tokens,
                    ),
                    Err(err) => {
                        eprintln!("{err}");
                        print_help(HelpTopic::Smoke);
                        ExitCode::from(2)
                    }
                },
                Some("1b" | "llama32-1b" | "llama-3.2-1b") => {
                    match parse_smoke_1b_args(&args[2..]) {
                        Ok(parsed) => {
                            let model_path = Path::new(&parsed.model_path);
                            if parsed.dry_run {
                                return print_smoke_dry_run(
                                    "NanoCamelid Llama 3.2 1B smoke dry run",
                                    "1b",
                                    model_path,
                                    &parsed,
                                );
                            }
                            run_smoke_1b_gate(model_path, &parsed)
                        }
                        Err(err) => {
                            eprintln!("{err}");
                            print_help(HelpTopic::Smoke);
                            ExitCode::from(2)
                        }
                    }
                }
                Some("3b" | "llama32-3b" | "llama-3.2-3b") => {
                    match parse_smoke_3b_args(&args[2..]) {
                        Ok(parsed) => {
                            let model_path = Path::new(&parsed.model_path);
                            if parsed.dry_run {
                                return print_smoke_dry_run(
                                    "NanoCamelid Llama 3.2 3B smoke dry run",
                                    "3b",
                                    model_path,
                                    &parsed,
                                );
                            }
                            if !model_path.is_file() {
                                eprintln!("{}", llama32_3b_model_not_found_message(model_path));
                                return ExitCode::from(2);
                            }
                            match parsed.kind {
                                SmokeKind::Q8Model => {
                                    smoke_q8_model(model_path, &parsed.prompt, parsed.max_tokens)
                                }
                                SmokeKind::Q8Chat => {
                                    smoke_q8_chat(model_path, &parsed.prompt, parsed.max_tokens)
                                }
                            }
                        }
                        Err(err) => {
                            eprintln!("{err}");
                            print_help(HelpTopic::Smoke);
                            ExitCode::from(2)
                        }
                    }
                }
                Some(other) => {
                    eprintln!("unknown smoke: {other}");
                    print_help(HelpTopic::Smoke);
                    ExitCode::from(2)
                }
                None => {
                    eprintln!("missing smoke name");
                    print_help(HelpTopic::Smoke);
                    ExitCode::from(2)
                }
            }
        }
        Some(other) if is_help_flag(other) => {
            print_help(HelpTopic::TopLevel);
            ExitCode::SUCCESS
        }
        None => {
            print_help(HelpTopic::TopLevel);
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown command: {other}");
            print_help(HelpTopic::TopLevel);
            ExitCode::from(2)
        }
    }
}

fn print_runtime_trace_summary() {
    let rows = inference::trace_snapshot();
    if rows.is_empty() {
        return;
    }
    println!("\nRuntime trace:");
    for (stage, stats) in rows.into_iter().take(24) {
        let total_ms = stats.total.as_secs_f64() * 1000.0;
        let avg_ms = total_ms / stats.calls.max(1) as f64;
        println!(
            "  {stage:<22} calls {:>6} total {:>10.3} ms avg {:>8.4} ms",
            stats.calls, total_ms, avg_ms
        );
    }
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn pthread_set_qos_class_self_np(
        qos_class: u32,
        relative_priority: std::os::raw::c_int,
    ) -> std::os::raw::c_int;
}

fn setup_thread_pool() {
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let worker_core_indices =
        worker_core_indices_from_env().or_else(isolated_cpu_indices_from_sysfs);
    let worker_core_ids = worker_core_indices
        .as_ref()
        .map(|indices| {
            indices
                .iter()
                .filter_map(|&idx| core_ids.get(idx).copied())
                .collect::<Vec<_>>()
        })
        .filter(|ids| !ids.is_empty())
        .unwrap_or_else(|| core_ids.clone());
    let default_threads = worker_core_ids.len().clamp(1, DEFAULT_RAYON_THREADS);
    let thread_count = env::var(RAYON_THREADS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default_threads);

    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .start_handler(move |thread_idx| {
            if let Some(core_id) = worker_core_ids.get(thread_idx % worker_core_ids.len().max(1)) {
                core_affinity::set_for_current(*core_id);
            }
            #[cfg(target_os = "macos")]
            unsafe {
                pthread_set_qos_class_self_np(0x21, 0); // QOS_CLASS_USER_INTERACTIVE (forces P-cores)
            }
        })
        .build_global();
}

fn worker_core_indices_from_env() -> Option<Vec<usize>> {
    env::var(WORKER_CORES_ENV)
        .ok()
        .and_then(|value| parse_cpu_list(&value))
}

fn isolated_cpu_indices_from_sysfs() -> Option<Vec<usize>> {
    fs::read_to_string("/sys/devices/system/cpu/isolated")
        .ok()
        .and_then(|value| parse_cpu_list(&value))
}

fn parse_cpu_list(value: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in value.trim().split(',').filter(|part| !part.is_empty()) {
        if let Some((start, end)) = part.split_once('-') {
            let start = start.trim().parse::<usize>().ok()?;
            let end = end.trim().parse::<usize>().ok()?;
            if start > end {
                return None;
            }
            cpus.extend(start..=end);
        } else {
            cpus.push(part.trim().parse::<usize>().ok()?);
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    (!cpus.is_empty()).then_some(cpus)
}

fn prefill_batch_size() -> usize {
    prefill_batch_size_from_env().unwrap_or(DEFAULT_Q4_PREFILL_BATCH)
}

fn prefill_batch_size_from_env() -> Result<usize, &'static str> {
    prefill_batch_size_from_env_value(env::var(PREFILL_BATCH_ENV).ok())
}

fn prefill_batch_size_from_env_value(value: Option<String>) -> Result<usize, &'static str> {
    match value {
        Some(value) => value
            .parse::<usize>()
            .ok()
            .filter(|&value| value > 0)
            .ok_or("NANOCAMELID_PREFILL_BATCH must be a positive integer"),
        None => Ok(DEFAULT_Q4_PREFILL_BATCH),
    }
}

fn ready_chat_enabled() -> Result<bool, &'static str> {
    env::var(READY_CHAT_ENV)
        .ok()
        .map(|value| ready_chat_enabled_from_env_value(&value))
        .unwrap_or(Ok(true))
}

fn ready_chat_enabled_from_env_value(value: &str) -> Result<bool, &'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err("NANOCAMELID_READY_CHAT must be 0, 1, false, true, no, yes, off, or on"),
    }
}

fn ready_chat_prompt(smoke_prompt: &str) -> String {
    ready_chat_prompt_from_env_value(env::var(READY_PROMPT_ENV).ok(), smoke_prompt)
}

fn ready_chat_prompt_from_env_value(value: Option<String>, smoke_prompt: &str) -> String {
    value
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| smoke_prompt.to_owned())
}

fn ready_chat_tokens(smoke_tokens: usize) -> Result<usize, &'static str> {
    ready_chat_tokens_from_env_value(env::var(READY_TOKENS_ENV).ok(), smoke_tokens)
}

fn ready_chat_tokens_from_env_value(
    value: Option<String>,
    smoke_tokens: usize,
) -> Result<usize, &'static str> {
    parse_optional_positive_usize(
        value.as_ref(),
        "ready direct chat env token count must be a positive integer",
    )
    .map(|value| value.unwrap_or(smoke_tokens))
}

fn ready_chat_temp() -> Result<f32, &'static str> {
    ready_chat_temp_from_env_value(env::var(READY_TEMP_ENV).ok())
}

fn ready_chat_temp_from_env_value(value: Option<String>) -> Result<f32, &'static str> {
    match value {
        Some(value) => parse_non_negative_f32(
            &value,
            "ready direct chat env temperature must be a non-negative number",
        ),
        None => Ok(0.0),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HelpTopic {
    TopLevel,
    Model,
    Models,
    Doctor,
    Serve,
    Probe,
    Inspect,
    Generate,
    Chat,
    Tui,
    Ready,
    Evidence,
    Bench,
    Smoke,
}

fn help_topic_for_args(args: &[String]) -> Option<HelpTopic> {
    match args.first().map(String::as_str) {
        Some("-h" | "--help") | Some("help") if args.len() == 1 => Some(HelpTopic::TopLevel),
        Some("help") => help_topic_named(args.get(1).map(String::as_str).unwrap_or_default()),
        Some("models")
            if args
                .get(1)
                .is_some_and(|value| matches!(value.as_str(), "list" | "scan" | "inspect"))
                && args.get(2).is_some_and(|value| is_help_flag(value)) =>
        {
            Some(HelpTopic::Models)
        }
        Some("model")
            if args.get(1).is_some_and(|value| is_llama32_1b_alias(value))
                && args.get(2).is_some_and(|value| is_help_flag(value)) =>
        {
            Some(HelpTopic::Model)
        }
        Some("inspect")
            if args
                .get(1)
                .is_some_and(|value| is_llama32_1b_alias(value) || is_llama32_3b_alias(value))
                && args.get(2).is_some_and(|value| is_help_flag(value)) =>
        {
            Some(HelpTopic::Inspect)
        }
        Some("generate")
            if args
                .get(1)
                .is_some_and(|value| is_llama32_1b_alias(value) || is_llama32_3b_alias(value))
                && args.get(2).is_some_and(|value| is_help_flag(value)) =>
        {
            Some(HelpTopic::Generate)
        }
        Some("chat")
            if args
                .get(1)
                .is_some_and(|value| is_llama32_1b_alias(value) || is_llama32_3b_alias(value))
                && args.get(2).is_some_and(|value| is_help_flag(value)) =>
        {
            Some(HelpTopic::Chat)
        }
        Some("tui")
            if args
                .get(1)
                .is_some_and(|value| is_llama32_1b_alias(value) || is_llama32_3b_alias(value))
                && args.get(2).is_some_and(|value| is_help_flag(value)) =>
        {
            Some(HelpTopic::Tui)
        }
        Some("ready")
            if args.get(1).is_some_and(|value| is_llama32_1b_alias(value))
                && args.get(2).is_some_and(|value| is_help_flag(value)) =>
        {
            Some(HelpTopic::Ready)
        }
        Some("evidence")
            if args.get(1).is_some_and(|value| is_llama32_1b_alias(value))
                && args.get(2).is_some_and(|value| is_help_flag(value)) =>
        {
            Some(HelpTopic::Evidence)
        }
        Some("bench")
            if args.get(1).is_some_and(|value| {
                is_llama32_1b_alias(value)
                    || matches!(value.as_str(), "q8-dot" | "q4-layout" | "q4-prefill")
            }) && args.get(2).is_some_and(|value| is_help_flag(value)) =>
        {
            Some(HelpTopic::Bench)
        }
        Some("smoke")
            if args.get(1).is_some_and(|value| {
                is_llama32_1b_alias(value)
                    || is_llama32_3b_alias(value)
                    || matches!(value.as_str(), "q8-model" | "q8-chat")
            }) && args.get(2).is_some_and(|value| is_help_flag(value)) =>
        {
            Some(HelpTopic::Smoke)
        }
        _ => None,
    }
}

fn help_topic_named(name: &str) -> Option<HelpTopic> {
    match name {
        "model" => Some(HelpTopic::Model),
        "models" => Some(HelpTopic::Models),
        "doctor" => Some(HelpTopic::Doctor),
        "serve" => Some(HelpTopic::Serve),
        "probe" => Some(HelpTopic::Probe),
        "inspect" => Some(HelpTopic::Inspect),
        "generate" => Some(HelpTopic::Generate),
        "chat" => Some(HelpTopic::Chat),
        "tui" => Some(HelpTopic::Tui),
        "ready" => Some(HelpTopic::Ready),
        "evidence" => Some(HelpTopic::Evidence),
        "bench" => Some(HelpTopic::Bench),
        "smoke" => Some(HelpTopic::Smoke),
        _ => None,
    }
}

fn is_help_flag(value: &str) -> bool {
    matches!(value, "-h" | "--help")
}

fn print_help(topic: HelpTopic) {
    match topic {
        HelpTopic::TopLevel => print_usage(),
        HelpTopic::Model => print_model_usage(),
        HelpTopic::Models => print_models_usage(),
        HelpTopic::Doctor => print_doctor_usage(),
        HelpTopic::Serve => print_serve_usage(),
        HelpTopic::Probe => print_probe_usage(),
        HelpTopic::Inspect => print_inspect_usage(),
        HelpTopic::Generate => print_generate_usage(),
        HelpTopic::Chat => print_chat_usage(),
        HelpTopic::Tui => print_tui_usage(),
        HelpTopic::Ready => print_ready_usage(),
        HelpTopic::Evidence => print_evidence_usage(),
        HelpTopic::Bench => print_bench_usage(),
        HelpTopic::Smoke => print_smoke_usage(),
    }
}

fn print_usage() {
    println!("NanoCamelid");
    println!();
    println!("Usage:");
    println!("  nanocamelid <command> [args]");
    println!("  nanocamelid --version");
    println!();
    println!("Commands:");
    println!(
        "  probe                                     Print host CPU and runtime feature information"
    );
    println!(
        "  doctor [--json] [--dry-run]              Check install, host, and model directory readiness"
    );
    println!("  serve [--host <addr>] [--port <port>]    Run the local HTTP API server");
    println!("  model 1b [model.gguf] [--q4|--q8] [--dry-run]");
    println!(
        "                                            Audit the default Llama 3.2 1B model path"
    );
    println!("  models list [--dir <path>] [--json]      List GGUF files in the model directory");
    println!("  models scan [--dir <path>] [--json]      Recursively classify GGUF files");
    println!(
        "  models inspect <model.gguf|1b|3b>         Inspect a model using the stable models namespace"
    );
    println!(
        "  inspect <model.gguf>                      Inspect GGUF metadata and tensor layouts"
    );
    println!("  inspect 1b [model.gguf] [--q4|--q8] [--dry-run]");
    println!("                                            Strictly inspect the Llama 3.2 1B path");
    println!("  inspect 3b [--dry-run]                    Inspect the default Llama 3.2 3B path");
    println!("  generate <model.gguf> <prompt> [temp] [max_tokens] [--dry-run]");
    println!("  generate 1b <prompt> [temp] [max_tokens] [--dry-run]");
    println!("  generate 3b <prompt> [temp] [max_tokens] [--dry-run]");
    println!(
        "                                            Generate text from prompt on Raspberry Pi 5"
    );
    println!("  chat <model.gguf> <prompt> [temp] [max_tokens] [--dry-run]");
    println!("  chat 1b <prompt> [temp] [max_tokens] [--dry-run]");
    println!("  chat 3b <prompt> [temp] [max_tokens] [--dry-run]");
    println!(
        "                                            Render a single-turn chat prompt before generation"
    );
    println!("  tui <model.gguf> [temp] [max_tokens]");
    println!("  tui 1b [temp] [max_tokens]");
    println!("  tui 3b [temp] [max_tokens]");
    println!("                                            Open an interactive terminal chat");
    println!(
        "  ready 1b [model.gguf] [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--q4|--q8] [--no-chat|--smoke-only|--chat|--dry-run]"
    );
    println!(
        "                                            Run inspect, smoke, and direct chat gates for 1B"
    );
    println!("  evidence 1b [model.gguf] [--q4|--q8] [--dry-run]");
    println!("                                            Run the bounded 1B evidence bundle");
    println!("  bench q8-dot [iterations] [runs]          Benchmark Q8 dot product kernels");
    println!(
        "  bench 1b [model.gguf] [prompt] [max_tokens] [temp] [batches] [--q4|--q8] [--dry-run]"
    );
    println!("                                            Run the Pi 1B prefill sweep");
    println!("  smoke q8-model <model.gguf> [prompt] [max_tokens]");
    println!(
        "                                            Compare scalar vs selected Q8 model logits and greedy generation"
    );
    println!("  smoke q8-chat <model.gguf> [prompt] [max_tokens]");
    println!(
        "                                            Compare scalar vs selected Q8 model logits through the tokenizer chat template"
    );
    println!(
        "  smoke 1b [model.gguf] [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--dry-run]"
    );
    println!("                                            Run the default Llama 3.2 1B smoke path");
    println!("  smoke 3b [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--dry-run]");
    println!("                                            Run the default Llama 3.2 3B smoke path");
    println!("  help [command]                            Show top-level or subcommand help");
    println!("  --version                                 Print the NanoCamelid release version");
    println!();
    println!("Run `nanocamelid help <command>` or `nanocamelid <command> --help` for details.");
}

fn print_doctor_usage() {
    println!("NanoCamelid doctor");
    println!();
    println!("Usage:");
    println!("  nanocamelid doctor [--json] [--dry-run]");
    println!();
    println!(
        "Run a non-loading product preflight for the binary version, host CPU summary, default model directory, and known 1B/3B model defaults."
    );
    println!();
    println!("Options:");
    println!("  --json                                   Print a machine-readable status line");
    println!(
        "  --dry-run                                Print the checks without reading model files"
    );
    println!();
    println!("Env:");
    println!("  {MODEL_DIR_ENV:<38} Override the model directory");
    println!(
        "  {WORKSPACE_ENV:<38} Pi workspace used when {MODEL_DIR_ENV} is unset; default {DEFAULT_PI_WORKSPACE}"
    );
    println!(
        "  {DEFAULT_MODEL_GGUF_ENV:<38} Optional explicit default GGUF path for model commands"
    );
}

fn print_serve_usage() {
    println!("NanoCamelid serve");
    println!();
    println!("Usage:");
    println!(
        "  nanocamelid serve [--host <addr>] [--port <port>] [--model-dir <path>] [--api-key <token>] [--max-request-bytes <count>] [--max-input-tokens <count>] [--max-output-tokens <count>] [--dry-run]"
    );
    println!();
    println!(
        "Run the local HTTP API server. The default bind address is {DEFAULT_API_HOST}:{DEFAULT_API_PORT}."
    );
    println!();
    println!("Endpoints:");
    println!("  GET  /health");
    println!("  GET  /v1/models");
    println!("  POST /v1/completions");
    println!("  POST /v1/chat/completions");
    println!("  GET  /metrics");
    println!();
    println!(
        "Completion endpoints accept a model id from /v1/models, a model filename/stem, 1b/3b aliases, or an explicit .gguf path."
    );
    println!("/v1/chat/completions renders supported tokenizer chat templates before generation.");
    println!();
    println!("Options:");
    println!("  --host <addr>                            Bind address, default {DEFAULT_API_HOST}");
    println!("  --port <port>                            Bind port, default {DEFAULT_API_PORT}");
    println!("  --model-dir <path>                       Model directory for /v1/models");
    println!("  --api-key <token>                        Require Authorization: Bearer <token>");
    println!(
        "  --max-request-bytes <count>              HTTP request byte cap, default {DEFAULT_SERVE_MAX_REQUEST_BYTES}"
    );
    println!(
        "  --max-input-tokens <count>               Request input token cap, default {DEFAULT_SERVE_MAX_INPUT_TOKENS}"
    );
    println!(
        "  --max-output-tokens <count>              Response token cap, default {DEFAULT_SERVE_MAX_OUTPUT_TOKENS}"
    );
    println!(
        "  --dry-run                                Print the server plan without binding a socket"
    );
    println!();
    println!("Env:");
    println!("  {MODEL_DIR_ENV:<38} Default model directory for /v1/models");
    println!("  {WORKSPACE_ENV:<38} Pi workspace used when {MODEL_DIR_ENV} is unset");
    println!("  {API_KEY_ENV:<38} Require bearer-token auth when set");
    println!("  {SERVE_MAX_REQUEST_BYTES_ENV:<38} HTTP request byte cap");
    println!("  {SERVE_MAX_INPUT_TOKENS_ENV:<38} Request input token cap");
    println!("  {SERVE_MAX_OUTPUT_TOKENS_ENV:<38} Response token cap");
}

fn print_models_usage() {
    println!("NanoCamelid models");
    println!();
    println!("Usage:");
    println!("  nanocamelid models list [--dir <path>] [--json] [--dry-run]");
    println!("  nanocamelid models scan [--dir <path>] [--json] [--dry-run]");
    println!("  nanocamelid models inspect <model.gguf|1b|3b> [--q4|--q8|--dry-run]");
    println!();
    println!("Args:");
    println!(
        "  list                                     List GGUF files directly under the model directory"
    );
    println!(
        "  scan                                     Recursively find GGUF files and classify target/quantization hints"
    );
    println!(
        "  inspect                                  Alias namespace for `nanocamelid inspect`"
    );
    println!();
    println!("Options:");
    println!("  --dir <path>                             Model directory for list or scan");
    println!("  --json                                   Print machine-readable summary lines");
    println!(
        "  --dry-run                                Print the resolved plan without reading model files"
    );
    println!("  --q4, --q8                               Forwarded to `models inspect 1b`");
    println!();
    println!("Env:");
    println!("  {MODEL_DIR_ENV:<38} Override the model directory");
    println!(
        "  {WORKSPACE_ENV:<38} Pi workspace used when {MODEL_DIR_ENV} is unset; default {DEFAULT_PI_WORKSPACE}"
    );
    println!("  {DEFAULT_MODEL_GGUF_ENV:<38} Default GGUF path for inspect");
    println!("  {SMOKE_MODEL_GGUF_ENV:<38} Alias-specific inspect override");
}

fn print_probe_usage() {
    println!("NanoCamelid probe");
    println!();
    println!("Usage:");
    println!("  nanocamelid probe");
    println!();
    println!(
        "Print host CPU model, feature flags, cpufreq telemetry, isolated CPUs, and runtime SIMD detection."
    );
}

fn print_model_usage() {
    println!("NanoCamelid model");
    println!();
    println!("Usage:");
    println!("  nanocamelid model 1b [model.gguf] [--q4|--q8] [--dry-run]");
    println!("  nanocamelid model llama32-1b [model.gguf] [--q4|--q8] [--dry-run]");
    println!();
    println!(
        "Audit the Llama 3.2 1B model selection plan and verify that the selected GGUF exists."
    );
    println!();
    println!("Options:");
    println!("  --q4                                     Select the Pi-local Q4_0 default row");
    println!("  --q8                                     Select the Pi-local Q8_0 default row");
    println!(
        "  --dry-run                                Print the audit without failing when the selected model is missing"
    );
    println!();
    println!("Env:");
    println!("  {SMOKE_MODEL_GGUF_ENV:<38} Override the 1B model audit GGUF path");
    println!("  {DEFAULT_MODEL_GGUF_ENV:<38} Shared default GGUF path for inspect/generate/smoke");
    println!("  {WORKSPACE_ENV:<38} Pi workspace for 1B defaults; default {DEFAULT_PI_WORKSPACE}");
    println!();
    println!(
        "Explicit model paths override --q4/--q8. Without a path or quant flag, `1b` prefers the Pi-local Q4_0 Llama 3.2 1B GGUF, then falls back to Q8_0."
    );
}

fn print_inspect_usage() {
    println!("NanoCamelid inspect");
    println!();
    println!("Usage:");
    println!("  nanocamelid inspect <model.gguf> [--dry-run]");
    println!("  nanocamelid inspect 1b [model.gguf] [--q4|--q8] [--dry-run]");
    println!("  nanocamelid inspect 3b [--dry-run]         inspect the default Llama 3.2 3B path");
    println!("  nanocamelid inspect                        with NANOCAMELID_MODEL_GGUF set");
    println!();
    println!(
        "Inspect GGUF metadata, runtime-ready LLaMA config, tokenizer support, and the first tensor layouts."
    );
    println!("Use --dry-run to print the resolved inspect command without reading the GGUF.");
    println!("For `inspect 1b`, explicit model paths override --q4/--q8 and env defaults.");
    println!();
    println!("Env:");
    println!(
        "  {DEFAULT_MODEL_GGUF_ENV}                    Default GGUF path for inspect and generate"
    );
    println!("  {SMOKE_MODEL_GGUF_ENV}                    Override the 1b/3b inspect aliases");
    println!(
        "  {WORKSPACE_ENV}                         Pi workspace for the 1b/3b aliases; default {DEFAULT_PI_WORKSPACE}"
    );
}

fn print_generate_usage() {
    println!("NanoCamelid generate");
    println!();
    println!("Usage:");
    println!("  nanocamelid generate <model.gguf> <prompt> [temp] [max_tokens] [--dry-run]");
    println!("  nanocamelid generate 1b <prompt> [temp] [max_tokens] [--dry-run]");
    println!("  nanocamelid generate 3b <prompt> [temp] [max_tokens] [--dry-run]");
    println!(
        "  nanocamelid generate <prompt> [temp] [max_tokens] [--dry-run]   with NANOCAMELID_MODEL_GGUF set"
    );
    println!();
    println!("Args:");
    println!("  <model.gguf>                              Path to the GGUF model file");
    println!(
        "  <prompt>                                  Prompt text to prefill before generation"
    );
    println!("  [temp]                                    Sampling temperature, default 0.0");
    println!("  [max_tokens]                              Maximum tokens to generate, default 128");
    println!();
    println!("Options:");
    println!(
        "  --dry-run                                Print the resolved generation plan without loading the model"
    );
    println!();
    println!("Env:");
    println!(
        "  {SMOKE_MODEL_GGUF_ENV}                    Override the 1b/3b aliases before {DEFAULT_MODEL_GGUF_ENV}"
    );
    println!(
        "  {DEFAULT_MODEL_GGUF_ENV}                    Default GGUF path for inspect and generate"
    );
    println!(
        "  {WORKSPACE_ENV}                         Pi workspace for the 1b/3b aliases; default {DEFAULT_PI_WORKSPACE}"
    );
    println!(
        "  {PREFILL_BATCH_ENV}                         Prefill prompt token batch size; default {DEFAULT_Q4_PREFILL_BATCH}, set 1 for single-token prefill"
    );
    println!(
        "  {CONTEXT_LIMIT_ENV}                         Optional runtime context cap for short long-context smoke runs"
    );
    println!(
        "  {TRACE_ENV}                            Set to 1 to print stage-level inference timings"
    );
    println!("  {ROPE_CACHE_ENV}                       Set to 0 to disable RoPE angle caching");
    println!();
    println!(
        "When {DEFAULT_MODEL_GGUF_ENV} is set, the first positional argument is treated as the prompt unless it looks like a .gguf path or a 1b/3b alias."
    );
}

fn print_chat_usage() {
    println!("NanoCamelid chat");
    println!();
    println!("Usage:");
    println!("  nanocamelid chat <model.gguf> <prompt> [temp] [max_tokens] [--dry-run]");
    println!("  nanocamelid chat 1b <prompt> [temp] [max_tokens] [--dry-run]");
    println!("  nanocamelid chat 3b <prompt> [temp] [max_tokens] [--dry-run]");
    println!(
        "  nanocamelid chat <prompt> [temp] [max_tokens] [--dry-run]   with NANOCAMELID_MODEL_GGUF set"
    );
    println!();
    println!("Args:");
    println!("  <model.gguf>                              Path to the GGUF model file");
    println!(
        "  <prompt>                                  User message content for a single-turn chat request"
    );
    println!("  [temp]                                    Sampling temperature, default 0.0");
    println!("  [max_tokens]                              Maximum tokens to generate, default 128");
    println!();
    println!("Options:");
    println!(
        "  --dry-run                                Print the resolved chat plan without loading the model"
    );
    println!();
    println!("Env:");
    println!(
        "  {SMOKE_MODEL_GGUF_ENV}                    Override the 1b/3b aliases before {DEFAULT_MODEL_GGUF_ENV}"
    );
    println!(
        "  {DEFAULT_MODEL_GGUF_ENV}                    Default GGUF path for inspect, generate, and chat"
    );
    println!(
        "  {WORKSPACE_ENV}                         Pi workspace for the 1b/3b aliases; default {DEFAULT_PI_WORKSPACE}"
    );
    println!(
        "  {PREFILL_BATCH_ENV}                         Prefill prompt token batch size; default {DEFAULT_Q4_PREFILL_BATCH}, set 1 for single-token prefill"
    );
    println!(
        "  {CONTEXT_LIMIT_ENV}                         Optional runtime context cap for short long-context smoke runs"
    );
    println!(
        "  {TRACE_ENV}                            Set to 1 to print stage-level inference timings"
    );
    println!("  {ROPE_CACHE_ENV}                       Set to 0 to disable RoPE angle caching");
    println!();
    println!(
        "Chat uses recognized tokenizer chat templates when present, including the Llama 3 instruct header/eot format."
    );
}

fn print_tui_usage() {
    println!("NanoCamelid tui");
    println!();
    println!("Usage:");
    println!("  nanocamelid tui <model.gguf> [temp] [max_tokens]");
    println!("  nanocamelid tui 1b [temp] [max_tokens] [--dry-run]");
    println!("  nanocamelid tui 3b [temp] [max_tokens] [--dry-run]");
    println!("  nanocamelid tui [temp] [max_tokens] [--dry-run]   with NANOCAMELID_MODEL_GGUF set");
    println!();
    println!("Args:");
    println!("  <model.gguf>                              Path to the GGUF model file");
    println!("  [temp]                                    Sampling temperature, default 0.0");
    println!(
        "  [max_tokens]                              Maximum tokens per assistant turn, default 128"
    );
    println!();
    println!("Options:");
    println!(
        "  --dry-run                                Print the resolved TUI launch plan without loading the model"
    );
    println!();
    println!("Env:");
    println!(
        "  {SMOKE_MODEL_GGUF_ENV}                    Override the 1b/3b aliases before {DEFAULT_MODEL_GGUF_ENV}"
    );
    println!(
        "  {DEFAULT_MODEL_GGUF_ENV}                    Default GGUF path for inspect, generate, chat, and tui"
    );
    println!(
        "  {WORKSPACE_ENV}                         Pi workspace for the 1b/3b aliases; default {DEFAULT_PI_WORKSPACE}"
    );
    println!(
        "  {RAYON_THREADS_ENV}                         Rayon worker count; defaults to up to 4 pinned workers"
    );
    println!(
        "  {WORKER_CORES_ENV}                          Comma/range CPU list for pinned Rayon workers, e.g. 1,2,3"
    );
    println!(
        "  {PREFILL_BATCH_ENV}                         Prefill prompt token batch size; default {DEFAULT_Q4_PREFILL_BATCH}, set 1 for single-token prefill"
    );
    println!(
        "  {CONTEXT_LIMIT_ENV}                         Optional runtime context cap for short long-context smoke runs"
    );
    println!(
        "  {TRACE_ENV}                            Set to 1 to print stage-level inference timings after each turn"
    );
    println!("  {ROPE_CACHE_ENV}                       Set to 0 to disable RoPE angle caching");
    println!();
    println!(
        "Commands inside the TUI: /help, /model <path>, /models, /temp <value>, /tokens <count>, /system <prompt>, /status, /history, /save <path>, /clear, /exit"
    );
}

fn print_ready_usage() {
    println!("NanoCamelid ready");
    println!();
    println!("Usage:");
    println!(
        "  nanocamelid ready 1b [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--q4|--q8] [--no-chat|--smoke-only|--chat|--dry-run]"
    );
    println!(
        "  nanocamelid ready 1b <model.gguf> [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--q4|--q8] [--no-chat|--smoke-only|--chat|--dry-run]"
    );
    println!();
    println!(
        "Run the Llama 3.2 1B readiness gate: audit shape, inspect metadata, smoke scalar-vs-selected logits, then run one direct chat turn."
    );
    println!();
    println!("Options:");
    println!("  --q4                                     Select the Pi-local Q4_0 default row");
    println!("  --q8                                     Select the Pi-local Q8_0 default row");
    println!(
        "  --no-chat, --smoke-only                  Stop after audit, inspect, and smoke; positionals override the smoke prompt"
    );
    println!("  --chat                                   Force the direct chat turn");
    println!(
        "  --dry-run                                Print the resolved readiness plan without loading the model"
    );
    println!(
        "  [prompt] [max_tokens]                    Override the final direct chat turn unless direct chat is disabled"
    );
    println!();
    println!("Env:");
    println!("  {SMOKE_MODEL_GGUF_ENV:<38} Override the 1B readiness GGUF path");
    println!("  {DEFAULT_MODEL_GGUF_ENV:<38} Shared default GGUF path for inspect/generate/smoke");
    println!(
        "  {WORKSPACE_ENV:<38} Pi workspace for the 1B default; default {DEFAULT_PI_WORKSPACE}"
    );
    println!(
        "  {CONTEXT_LIMIT_ENV:<38} Optional runtime context cap for short long-context smoke runs"
    );
    println!(
        "  {PREFILL_BATCH_ENV:<38} Prefill prompt token batch size; default {DEFAULT_Q4_PREFILL_BATCH}"
    );
    println!("  {READY_SMOKE_KIND_ENV:<38} Smoke kind, default chat");
    println!("  {READY_SMOKE_PROMPT_ENV:<38} Smoke prompt");
    println!("  {READY_SMOKE_TOKENS_ENV:<38} Smoke generated token count");
    println!(
        "  {READY_CHAT_ENV:<38} Set to 0/false/no/off to stop after audit, inspect, and smoke"
    );
    println!("  {READY_PROMPT_ENV:<38} Direct chat prompt after smoke");
    println!("  {READY_TOKENS_ENV:<38} Direct chat generated token count");
    println!("  {READY_TEMP_ENV:<38} Direct chat temperature, default 0.0");
    println!("Explicit model paths override --q4/--q8 and env defaults.");
}

fn print_evidence_usage() {
    println!("NanoCamelid evidence");
    println!();
    println!("Usage:");
    println!("  nanocamelid evidence 1b [model.gguf] [--q4|--q8] [--dry-run]");
    println!("  nanocamelid evidence llama32-1b [model.gguf] [--q4|--q8] [--dry-run]");
    println!();
    println!(
        "Run the bounded Llama 3.2 1B evidence bundle: readiness without final chat, context-pack smoke, and prefill batch sweep."
    );
    println!();
    println!("Options:");
    println!("  --q4                                     Select the Pi-local Q4_0 default row");
    println!("  --q8                                     Select the Pi-local Q8_0 default row");
    println!(
        "  --dry-run                                Print the resolved evidence plan without loading the model"
    );
    println!();
    println!("Env:");
    println!("  {SMOKE_MODEL_GGUF_ENV:<38} Override the 1B evidence GGUF path");
    println!("  {DEFAULT_MODEL_GGUF_ENV:<38} Shared 1B evidence GGUF override");
    println!(
        "  {WORKSPACE_ENV:<38} Pi workspace for the 1B default; default {DEFAULT_PI_WORKSPACE}"
    );
    println!("  {SMOKE_KIND_ENV:<38} Smoke kind, default chat");
    println!("  {SMOKE_PROMPT_ENV:<38} Smoke prompt");
    println!("  {SMOKE_TOKENS_ENV:<38} Smoke generated token count");
    println!("  {CONTEXT_LIMIT_ENV:<38} Optional runtime context cap for readiness and sweeps");
    println!("  {CONTEXT_PACKS_ENV:<38} Context caps, default {DEFAULT_1B_CONTEXT_PACKS}");
    println!("  {PREFILL_PROMPT_ENV:<38} 1B prefill sweep prompt");
    println!("  {PREFILL_TOKENS_ENV:<38} 1B generated token count");
    println!("  {PREFILL_TEMP_ENV:<38} 1B sweep temperature");
    println!("  {PREFILL_BATCHES_ENV:<38} 1B prefill batch list");
    println!("Explicit model paths override --q4/--q8 and env defaults.");
}

fn print_bench_usage() {
    println!("NanoCamelid bench");
    println!();
    println!("Usage:");
    println!("  nanocamelid bench q8-dot [iterations] [runs]");
    println!("  nanocamelid bench q4-layout [rows] [cols] [runs]");
    println!("  nanocamelid bench q4-prefill [prompt_len] [batch_size] [runs]");
    println!(
        "  nanocamelid bench 1b [model.gguf] [prompt] [max_tokens] [temp] [batches] [--q4|--q8] [--dry-run]"
    );
    println!();
    println!("Args:");
    println!(
        "  q8-dot [iterations]                      Blocks per run, default {}",
        q8::DEFAULT_DOT_BENCH_ITERATIONS
    );
    println!(
        "  q4-layout [rows]                         Synthetic matrix rows, default {}",
        q8::DEFAULT_Q4_LAYOUT_BENCH_ROWS
    );
    println!(
        "  q4-layout [cols]                         Synthetic matrix cols, default {}",
        q8::DEFAULT_Q4_LAYOUT_BENCH_COLS
    );
    println!(
        "  [runs]                                    Repeated timing samples, default {}",
        q8::DEFAULT_DOT_BENCH_RUNS
    );
    println!(
        "  q4-prefill [prompt_len]                  Synthetic prompt length, default {}",
        DEFAULT_Q4_PREFILL_PROMPT_LEN
    );
    println!(
        "  q4-prefill [batch_size]                  Synthetic prefill batch size, default {}",
        DEFAULT_Q4_PREFILL_BATCH
    );
    println!(
        "  q4-prefill [runs]                        Repeated timing samples, default {}",
        DEFAULT_Q4_PREFILL_RUNS
    );
    println!("  1b [model.gguf]                          Llama 3.2 1B GGUF path");
    println!(
        "  1b [prompt]                              Prompt for the Pi prefill sweep, default {DEFAULT_1B_PREFILL_PROMPT:?}"
    );
    println!(
        "  1b [max_tokens]                          Generated token count, default {DEFAULT_1B_PREFILL_TOKENS}"
    );
    println!(
        "  1b [temp]                                Temperature, default {DEFAULT_1B_PREFILL_TEMP}"
    );
    println!(
        "  1b [batches]                             Comma-separated prefill batches, default {DEFAULT_1B_PREFILL_BATCHES}"
    );
    println!();
    println!("Options:");
    println!("  --q4                                     Select the Pi-local Q4_0 default row");
    println!("  --q8                                     Select the Pi-local Q8_0 default row");
    println!(
        "  --dry-run                                Print the 1B prefill sweep plan without loading the model"
    );
    println!();
    println!("Env:");
    println!("  NANOCAMELID_Q8_DOT_KERNEL                 Force scalar, neon, or sdot selection");
    println!(
        "  NANOCAMELID_Q8_DOT_SDOT                   Enable SDOT candidate benchmarking when supported"
    );
    println!("  NANOCAMELID_Q6K_SDOT                      Enable experimental Q6_K SDOT matmuls");
    println!(
        "  NANOCAMELID_ATTENTION_HEAD_PARALLEL       Enable experimental head-parallel attention"
    );
    println!("  NANOCAMELID_KV_CACHE_F16                  Store KV cache entries as f16");
    println!(
        "  {TRACE_ENV}                            Set to 1 to print stage-level inference timings"
    );
    println!("  {ROPE_CACHE_ENV}                       Set to 0 to disable RoPE angle caching");
    println!(
        "  {RAYON_THREADS_ENV}                         Rayon worker count; defaults to up to 4 pinned workers"
    );
    println!(
        "  {WORKER_CORES_ENV}                          Comma/range CPU list for pinned Rayon workers, e.g. 1,2,3"
    );
    println!("  {SMOKE_MODEL_GGUF_ENV:<38} Smoke-specific 1B benchmark GGUF override");
    println!("  {DEFAULT_MODEL_GGUF_ENV:<38} Shared 1B benchmark GGUF override");
    println!(
        "  {WORKSPACE_ENV:<38} Pi workspace for the 1B default; default {DEFAULT_PI_WORKSPACE}"
    );
    println!("  {PREFILL_PROMPT_ENV:<38} 1B prefill sweep prompt");
    println!("  {PREFILL_TOKENS_ENV:<38} 1B generated token count");
    println!("  {PREFILL_TEMP_ENV:<38} 1B sweep temperature");
    println!("  {PREFILL_BATCHES_ENV:<38} 1B prefill batch list");
    println!("  {CONTEXT_LIMIT_ENV:<38} Optional runtime context cap for the 1B sweep");
}

fn print_smoke_usage() {
    println!("NanoCamelid smoke");
    println!();
    println!("Usage:");
    println!("  nanocamelid smoke q8-model <model.gguf> [prompt] [max_tokens]");
    println!("  nanocamelid smoke q8-chat <model.gguf> [prompt] [max_tokens]");
    println!(
        "  nanocamelid smoke 1b [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--q4|--q8] [--dry-run]"
    );
    println!(
        "  nanocamelid smoke 3b [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--dry-run]"
    );
    println!("  nanocamelid smoke q8-model [prompt] [max_tokens]  with NANOCAMELID_SMOKE_GGUF set");
    println!("  nanocamelid smoke q8-chat [prompt] [max_tokens]   with NANOCAMELID_SMOKE_GGUF set");
    println!(
        "  nanocamelid smoke 1b <model.gguf> [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--q4|--q8] [--dry-run]"
    );
    println!(
        "  nanocamelid smoke 3b <model.gguf> [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--dry-run]"
    );
    println!();
    println!("Args:");
    println!("  <model.gguf>                              Path to the GGUF model file");
    println!("  q8-* [prompt]                             Prompt text, default \"Hello\"");
    println!(
        "  q8-* [max_tokens]                         Greedy tokens to generate after parity, default 1"
    );
    println!(
        "  1b/3b [prompt]                            Prompt text, default {DEFAULT_1B_SMOKE_PROMPT:?}"
    );
    println!(
        "  1b/3b [max_tokens]                        Greedy tokens to generate after parity, default {DEFAULT_1B_SMOKE_TOKENS}"
    );
    println!();
    println!("Env:");
    println!("  {SMOKE_MODEL_GGUF_ENV}                    Default GGUF path for smoke validation");
    println!(
        "  {DEFAULT_MODEL_GGUF_ENV}                    Shared default GGUF path for inspect/generate/smoke"
    );
    println!("  {SMOKE_KIND_ENV}                    Default smoke 1b/3b kind; default chat");
    println!("  {SMOKE_PROMPT_ENV}                  Default smoke 1b/3b prompt");
    println!("  {SMOKE_TOKENS_ENV}                  Default smoke 1b/3b generated token count");
    println!(
        "  {WORKSPACE_ENV}                         Pi workspace for smoke 1b/3b defaults; default {DEFAULT_PI_WORKSPACE}"
    );
    println!(
        "  NANOCAMELID_Q8_DOT_KERNEL                 Force scalar, neon, or sdot kernel selection"
    );
    println!(
        "  {PREFILL_BATCH_ENV}                         Prefill prompt token batch size; default {DEFAULT_Q4_PREFILL_BATCH}, set 1 for single-token prefill"
    );
    println!(
        "  {CONTEXT_LIMIT_ENV}                         Optional runtime context cap for short long-context smoke runs"
    );
    println!();
    println!("Options:");
    println!(
        "  --dry-run                                Print the resolved 1b/3b smoke plan without loading the model"
    );
    println!("  --q4                                     Select the Pi-local 1B Q4_0 default row");
    println!("  --q8                                     Select the Pi-local 1B Q8_0 default row");
    println!();
    println!(
        "When {SMOKE_MODEL_GGUF_ENV} or {DEFAULT_MODEL_GGUF_ENV} is set, the first positional argument is treated as the prompt unless it looks like a .gguf path."
    );
    println!();
    println!(
        "`q8-model` tokenizes the prompt directly. `q8-chat` renders a single-turn user message through the model tokenizer chat template before parity/generation."
    );
    println!(
        "`1b` defaults to chat, prompt {DEFAULT_1B_SMOKE_PROMPT:?}, and {DEFAULT_1B_SMOKE_TOKENS} tokens. It prefers the Pi-local Q4_0 Llama 3.2 1B GGUF, then Q8_0. Use --q4 or --q8 to force one default row. The legacy q8-chat and q8-model aliases are still accepted."
    );
    println!(
        "`3b` defaults to chat, prompt {DEFAULT_1B_SMOKE_PROMPT:?}, and {DEFAULT_1B_SMOKE_TOKENS} tokens. It expects the Pi-local Q4_0 Llama 3.2 3B GGUF."
    );
}

#[derive(Debug, PartialEq)]
struct GenerateArgs {
    model_path: String,
    model_source: &'static str,
    prompt: String,
    temp: f32,
    max_tokens: usize,
    dry_run: bool,
    audit_1b_shape: bool,
}

#[derive(Debug, PartialEq)]
struct TuiArgs {
    model_path: String,
    model_source: &'static str,
    temp: f32,
    max_tokens: usize,
    dry_run: bool,
    audit_1b_shape: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct SmokeQ8ModelArgs {
    model_path: String,
    prompt: String,
    max_tokens: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct Model1BAuditArgs {
    workspace: String,
    q4_model_path: String,
    q8_model_path: String,
    model_path: String,
    model_source: &'static str,
    dry_run: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmokeKind {
    Q8Model,
    Q8Chat,
}

impl SmokeKind {
    fn from_arg(value: &str) -> Option<Self> {
        match value {
            "model" | "q8-model" => Some(Self::Q8Model),
            "chat" | "q8-chat" => Some(Self::Q8Chat),
            _ => None,
        }
    }

    fn looks_like_arg(value: &str) -> bool {
        matches!(value, "chat" | "model") || value.starts_with("q8-")
    }

    fn label(self) -> &'static str {
        match self {
            Self::Q8Model => "model",
            Self::Q8Chat => "chat",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SmokeDefaults {
    kind: SmokeKind,
    prompt: String,
    max_tokens: usize,
}

impl Default for SmokeDefaults {
    fn default() -> Self {
        Self {
            kind: SmokeKind::Q8Chat,
            prompt: DEFAULT_1B_SMOKE_PROMPT.to_owned(),
            max_tokens: DEFAULT_1B_SMOKE_TOKENS,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Smoke1BArgs {
    kind: SmokeKind,
    model_path: String,
    model_source: &'static str,
    prompt: String,
    max_tokens: usize,
    dry_run: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct Ready1BArgs {
    smoke: Smoke1BArgs,
    chat_enabled_override: Option<bool>,
    chat_prompt_override: Option<String>,
    chat_tokens_override: Option<usize>,
    dry_run: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct InspectArgs {
    model_path: String,
    model_source: &'static str,
    target: Option<InspectTarget>,
    dry_run: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InspectTarget {
    Llama32_1B,
    Llama32_3B,
}

#[derive(Debug, PartialEq, Eq)]
struct BenchQ8DotArgs {
    iterations: usize,
    runs: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct BenchQ4LayoutArgs {
    rows: usize,
    cols: usize,
    runs: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct BenchQ4PrefillArgs {
    prompt_len: usize,
    batch_size: usize,
    runs: usize,
}

#[derive(Debug, PartialEq)]
struct Bench1BArgs {
    workspace: String,
    q4_model_path: String,
    q8_model_path: String,
    model_path: String,
    model_source: &'static str,
    prompt: String,
    max_tokens: usize,
    temp: String,
    batches: Vec<usize>,
    dry_run: bool,
}

#[derive(Debug, PartialEq)]
struct Evidence1BArgs {
    workspace: String,
    q4_model_path: String,
    q8_model_path: String,
    model_path: String,
    model_source: &'static str,
    smoke: SmokeDefaults,
    prefill_batch: usize,
    context_packs: Vec<usize>,
    prefill: Bench1BArgs,
    dry_run: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct DoctorArgs {
    json: bool,
    dry_run: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct ServeArgs {
    host: String,
    port: u16,
    model_dir: String,
    api_key: Option<String>,
    max_request_bytes: usize,
    max_input_tokens: usize,
    max_output_tokens: usize,
    dry_run: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct ModelsArgs {
    action: ModelsAction,
    json: bool,
    dry_run: bool,
}

#[derive(Debug, PartialEq, Eq)]
enum ModelsAction {
    List { dir: String },
    Scan { dir: String },
    Inspect(InspectArgs),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelEntry {
    path: PathBuf,
    bytes: u64,
    target: Option<&'static str>,
    quantization: Option<&'static str>,
}

fn parse_doctor_args(args: &[String]) -> Result<DoctorArgs, &'static str> {
    let mut json = false;
    let mut dry_run = false;
    for arg in args {
        match arg.as_str() {
            "--json" => json = true,
            "--dry-run" => dry_run = true,
            arg if arg.starts_with('-') => return Err("unknown doctor option"),
            _ => return Err("unexpected doctor argument"),
        }
    }
    Ok(DoctorArgs { json, dry_run })
}

fn parse_serve_args(args: &[String]) -> Result<ServeArgs, &'static str> {
    parse_serve_args_with_defaults(
        args,
        default_model_dir(),
        env::var(API_KEY_ENV).ok(),
        env::var(SERVE_MAX_REQUEST_BYTES_ENV).ok(),
        env::var(SERVE_MAX_INPUT_TOKENS_ENV).ok(),
        env::var(SERVE_MAX_OUTPUT_TOKENS_ENV).ok(),
    )
}

fn parse_serve_args_with_defaults(
    args: &[String],
    default_model_dir: String,
    default_api_key: Option<String>,
    default_max_request_bytes: Option<String>,
    default_max_input_tokens: Option<String>,
    default_max_output_tokens: Option<String>,
) -> Result<ServeArgs, &'static str> {
    let mut host = DEFAULT_API_HOST.to_owned();
    let mut port = DEFAULT_API_PORT;
    let mut model_dir = default_model_dir;
    let mut api_key = default_api_key.filter(|value| !value.is_empty());
    let mut max_request_bytes = parse_optional_positive_usize(
        default_max_request_bytes.as_ref(),
        "NANOCAMELID_MAX_REQUEST_BYTES must be a positive integer",
    )?
    .unwrap_or(DEFAULT_SERVE_MAX_REQUEST_BYTES);
    let mut max_input_tokens = parse_optional_positive_usize(
        default_max_input_tokens.as_ref(),
        "NANOCAMELID_MAX_INPUT_TOKENS must be a positive integer",
    )?
    .unwrap_or(DEFAULT_SERVE_MAX_INPUT_TOKENS);
    let mut max_output_tokens = parse_optional_positive_usize(
        default_max_output_tokens.as_ref(),
        "NANOCAMELID_MAX_OUTPUT_TOKENS must be a positive integer",
    )?
    .unwrap_or(DEFAULT_SERVE_MAX_OUTPUT_TOKENS);
    let mut dry_run = false;
    let mut idx = 0;

    while idx < args.len() {
        match args[idx].as_str() {
            "--dry-run" => dry_run = true,
            "--host" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err("serve --host requires a bind address");
                };
                host = value.clone();
            }
            arg if arg.starts_with("--host=") => {
                host = arg.trim_start_matches("--host=").to_owned();
            }
            "--port" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err("serve --port requires a port");
                };
                port = parse_port(value)?;
            }
            arg if arg.starts_with("--port=") => {
                port = parse_port(arg.trim_start_matches("--port="))?;
            }
            "--model-dir" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err("serve --model-dir requires a path");
                };
                model_dir = value.clone();
            }
            arg if arg.starts_with("--model-dir=") => {
                model_dir = arg.trim_start_matches("--model-dir=").to_owned();
            }
            "--api-key" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err("serve --api-key requires a token");
                };
                if value.is_empty() {
                    return Err("serve --api-key requires a non-empty token");
                }
                api_key = Some(value.clone());
            }
            arg if arg.starts_with("--api-key=") => {
                let value = arg.trim_start_matches("--api-key=");
                if value.is_empty() {
                    return Err("serve --api-key requires a non-empty token");
                }
                api_key = Some(value.to_owned());
            }
            "--max-request-bytes" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err("serve --max-request-bytes requires a count");
                };
                max_request_bytes = parse_positive_usize(
                    value,
                    "serve --max-request-bytes must be a positive integer",
                )?;
            }
            arg if arg.starts_with("--max-request-bytes=") => {
                max_request_bytes = parse_positive_usize(
                    arg.trim_start_matches("--max-request-bytes="),
                    "serve --max-request-bytes must be a positive integer",
                )?;
            }
            "--max-input-tokens" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err("serve --max-input-tokens requires a count");
                };
                max_input_tokens = parse_positive_usize(
                    value,
                    "serve --max-input-tokens must be a positive integer",
                )?;
            }
            arg if arg.starts_with("--max-input-tokens=") => {
                max_input_tokens = parse_positive_usize(
                    arg.trim_start_matches("--max-input-tokens="),
                    "serve --max-input-tokens must be a positive integer",
                )?;
            }
            "--max-output-tokens" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err("serve --max-output-tokens requires a count");
                };
                max_output_tokens = parse_positive_usize(
                    value,
                    "serve --max-output-tokens must be a positive integer",
                )?;
            }
            arg if arg.starts_with("--max-output-tokens=") => {
                max_output_tokens = parse_positive_usize(
                    arg.trim_start_matches("--max-output-tokens="),
                    "serve --max-output-tokens must be a positive integer",
                )?;
            }
            arg if arg.starts_with('-') => return Err("unknown serve option"),
            _ => return Err("unexpected serve argument"),
        }
        idx += 1;
    }

    if host.is_empty() {
        return Err("serve --host requires a non-empty bind address");
    }

    Ok(ServeArgs {
        host,
        port,
        model_dir,
        api_key,
        max_request_bytes,
        max_input_tokens,
        max_output_tokens,
        dry_run,
    })
}

fn parse_models_args(args: &[String]) -> Result<ModelsArgs, &'static str> {
    let Some(command) = args.first().map(String::as_str) else {
        return Err("missing models command; expected list, scan, or inspect");
    };

    match command {
        "list" | "scan" => {
            let mut dir = None;
            let mut json = false;
            let mut dry_run = false;
            let mut idx = 1;
            while idx < args.len() {
                match args[idx].as_str() {
                    "--json" => json = true,
                    "--dry-run" => dry_run = true,
                    "--dir" => {
                        idx += 1;
                        let Some(value) = args.get(idx) else {
                            return Err("models --dir requires a path");
                        };
                        dir = Some(value.clone());
                    }
                    arg if arg.starts_with("--dir=") => {
                        dir = Some(arg.trim_start_matches("--dir=").to_owned());
                    }
                    arg if arg.starts_with('-') => return Err("unknown models option"),
                    _ => return Err("unexpected models argument"),
                }
                idx += 1;
            }
            let dir = dir.unwrap_or_else(default_model_dir);
            let action = if command == "list" {
                ModelsAction::List { dir }
            } else {
                ModelsAction::Scan { dir }
            };
            Ok(ModelsArgs {
                action,
                json,
                dry_run,
            })
        }
        "inspect" => {
            let inspect = parse_inspect_args(&args[1..])?;
            Ok(ModelsArgs {
                action: ModelsAction::Inspect(inspect),
                json: false,
                dry_run: false,
            })
        }
        _ => Err("unknown models command; expected list, scan, or inspect"),
    }
}

fn parse_generate_args(args: &[String]) -> Result<GenerateArgs, &'static str> {
    let (workspace, q4_exists) = llama32_1b_workspace_defaults();
    parse_generate_args_with_env_and_alias_env_and_workspace(
        args,
        default_model_path_from_env(),
        model_1b_env_path_and_source(),
        &workspace,
        q4_exists,
    )
}

fn parse_tui_args(args: &[String]) -> Result<TuiArgs, &'static str> {
    let (workspace, q4_exists) = llama32_1b_workspace_defaults();
    parse_tui_args_with_env_and_alias_env_and_workspace(
        args,
        default_model_path_from_env(),
        model_1b_env_path_and_source(),
        &workspace,
        q4_exists,
    )
}

fn parse_inspect_args(args: &[String]) -> Result<InspectArgs, &'static str> {
    let (workspace, q4_exists) = llama32_1b_workspace_defaults();
    parse_inspect_args_with_env(
        args,
        default_model_path_and_source_from_env(),
        smoke_model_path_and_source_from_env(),
        &workspace,
        q4_exists,
    )
}

fn parse_inspect_args_with_env(
    args: &[String],
    default_env_model_path: Option<(String, &'static str)>,
    alias_env_model_path: Option<(String, &'static str)>,
    workspace: &str,
    q4_exists: bool,
) -> Result<InspectArgs, &'static str> {
    let mut dry_run = false;
    let mut selected_quant = None;
    let mut positionals = Vec::with_capacity(args.len());
    for arg in args {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
            "--q4" => {
                if selected_quant.replace(LLAMA32_1B_Q4_MODEL).is_some() {
                    return Err("inspect 1B accepts only one quantization selector");
                }
            }
            "--q8" => {
                if selected_quant.replace(LLAMA32_1B_Q8_MODEL).is_some() {
                    return Err("inspect 1B accepts only one quantization selector");
                }
            }
            arg if arg.starts_with('-') => return Err("unknown inspect option"),
            _ => positionals.push(arg.clone()),
        }
    }

    if selected_quant.is_some()
        && positionals
            .first()
            .is_none_or(|value| !is_llama32_1b_alias(value))
    {
        return Err("inspect --q4/--q8 requires the 1B alias");
    }

    if positionals.len() > 1 {
        if positionals
            .first()
            .is_some_and(|value| is_llama32_1b_alias(value))
        {
            reject_extra_positionals(&positionals[1..], 1, "unexpected extra inspect 1B argument")?;
        } else {
            return Err("unexpected extra inspect argument");
        }
    }

    let Some(first) = positionals.first() else {
        let Some((model_path, model_source)) = default_env_model_path else {
            return Err("missing GGUF path; pass one or set NANOCAMELID_MODEL_GGUF");
        };
        return Ok(InspectArgs {
            model_path,
            model_source,
            target: None,
            dry_run,
        });
    };

    if is_llama32_1b_alias(first) {
        let model_arg = positionals.get(1);
        if let Some(model_path) = model_arg
            && !looks_like_gguf_path(model_path)
        {
            return Err("inspect 1B model argument must be a .gguf path");
        }
        let (model_path, model_source) = if let Some(model_path) = model_arg {
            (model_path.clone(), "explicit argument")
        } else if let Some(model_name) = selected_quant {
            (
                llama32_1b_model_path(workspace, model_name),
                if model_name == LLAMA32_1B_Q4_MODEL {
                    "workspace Q4_0 requested"
                } else {
                    "workspace Q8_0 requested"
                },
            )
        } else {
            resolve_llama32_1b_model_path_and_source(alias_env_model_path, workspace, q4_exists)?
        };
        return Ok(InspectArgs {
            model_path,
            model_source,
            target: Some(InspectTarget::Llama32_1B),
            dry_run,
        });
    }

    if is_llama32_3b_alias(first) {
        let (model_path, model_source) =
            resolve_llama32_3b_model_path_and_source(alias_env_model_path, workspace)?;
        return Ok(InspectArgs {
            model_path,
            model_source,
            target: Some(InspectTarget::Llama32_3B),
            dry_run,
        });
    }

    Ok(InspectArgs {
        model_path: first.clone(),
        model_source: "explicit argument",
        target: None,
        dry_run,
    })
}

#[cfg(test)]
fn parse_tui_args_with_env(
    args: &[String],
    env_model_path: Option<String>,
) -> Result<TuiArgs, &'static str> {
    parse_tui_args_with_env_and_workspace(args, env_model_path, DEFAULT_PI_WORKSPACE, false)
}

#[cfg(test)]
fn parse_tui_args_with_env_and_workspace(
    args: &[String],
    env_model_path: Option<String>,
    workspace: &str,
    q4_exists: bool,
) -> Result<TuiArgs, &'static str> {
    parse_tui_args_with_env_and_alias_env_and_workspace(
        args,
        env_model_path.clone(),
        env_model_path.map(|path| (path, DEFAULT_MODEL_GGUF_ENV)),
        workspace,
        q4_exists,
    )
}

fn parse_tui_args_with_env_and_alias_env_and_workspace(
    args: &[String],
    env_model_path: Option<String>,
    alias_env_model_path: Option<(String, &'static str)>,
    workspace: &str,
    q4_exists: bool,
) -> Result<TuiArgs, &'static str> {
    let env_model_path = env_model_path.map(|path| (path, DEFAULT_MODEL_GGUF_ENV));
    let mut dry_run = false;
    let mut positionals = Vec::with_capacity(args.len());
    for arg in args {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
            _ => positionals.push(arg.clone()),
        }
    }

    let (model_path, model_source, option_idx, audit_1b_shape) = parse_model_path_position(
        &positionals,
        env_model_path,
        alias_env_model_path,
        workspace,
        q4_exists,
        ModelPathPositionErrors {
            env_path: "model env path must be a .gguf path",
            alias_env_path: "model alias env path must be a .gguf path",
            missing: "missing GGUF model path; pass one or set NANOCAMELID_MODEL_GGUF",
            alias_path: "model alias argument must not be a path; use `tui <model.gguf>` for explicit models",
        },
    )?;

    let temp = positionals
        .get(option_idx)
        .map(|value| parse_non_negative_f32(value, "tui temp must be a non-negative number"))
        .transpose()?
        .unwrap_or(0.0);
    let max_tokens = parse_optional_positive_usize(
        positionals.get(option_idx + 1),
        "tui max_tokens must be a positive integer",
    )?
    .unwrap_or(128);
    reject_extra_positionals(
        &positionals,
        option_idx + 2,
        "unexpected extra tui argument",
    )?;

    Ok(TuiArgs {
        model_path,
        model_source,
        temp,
        max_tokens,
        dry_run,
        audit_1b_shape,
    })
}

#[cfg(test)]
fn parse_generate_args_with_env(
    args: &[String],
    env_model_path: Option<String>,
) -> Result<GenerateArgs, &'static str> {
    parse_generate_args_with_env_and_workspace(args, env_model_path, DEFAULT_PI_WORKSPACE, false)
}

#[cfg(test)]
fn parse_generate_args_with_env_and_workspace(
    args: &[String],
    env_model_path: Option<String>,
    workspace: &str,
    q4_exists: bool,
) -> Result<GenerateArgs, &'static str> {
    parse_generate_args_with_env_and_alias_env_and_workspace(
        args,
        env_model_path.clone(),
        env_model_path.map(|path| (path, DEFAULT_MODEL_GGUF_ENV)),
        workspace,
        q4_exists,
    )
}

fn parse_generate_args_with_env_and_alias_env_and_workspace(
    args: &[String],
    env_model_path: Option<String>,
    alias_env_model_path: Option<(String, &'static str)>,
    workspace: &str,
    q4_exists: bool,
) -> Result<GenerateArgs, &'static str> {
    let env_model_path = env_model_path.map(|path| (path, DEFAULT_MODEL_GGUF_ENV));
    let mut dry_run = false;
    let mut positionals = Vec::with_capacity(args.len());
    for arg in args {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
            _ => positionals.push(arg.clone()),
        }
    }

    let (model_path, model_source, prompt_idx, audit_1b_shape) = parse_model_path_position(
        &positionals,
        env_model_path,
        alias_env_model_path,
        workspace,
        q4_exists,
        ModelPathPositionErrors {
            env_path: "model env path must be a .gguf path",
            alias_env_path: "model alias env path must be a .gguf path",
            missing: "missing GGUF model path; pass one or set NANOCAMELID_MODEL_GGUF",
            alias_path: "model alias prompt must not look like a model path; omit the alias when passing an explicit model",
        },
    )?;

    let prompt = match positionals.get(prompt_idx) {
        Some(prompt) => prompt.clone(),
        None if dry_run => "<prompt>".to_owned(),
        None => {
            return Err(
                "missing prompt; pass one after the GGUF path or set NANOCAMELID_MODEL_GGUF and pass the prompt first",
            );
        }
    };
    let temp = positionals
        .get(prompt_idx + 1)
        .map(|value| parse_non_negative_f32(value, "generate temp must be a non-negative number"))
        .transpose()?
        .unwrap_or(0.0);
    let max_tokens = parse_optional_positive_usize(
        positionals.get(prompt_idx + 2),
        "generate max_tokens must be a positive integer",
    )?
    .unwrap_or(128);
    reject_extra_positionals(
        &positionals,
        prompt_idx + 3,
        "unexpected extra generate argument",
    )?;

    Ok(GenerateArgs {
        model_path,
        model_source,
        prompt,
        temp,
        max_tokens,
        dry_run,
        audit_1b_shape,
    })
}

fn parse_model_path_position(
    args: &[String],
    env_model_path: Option<(String, &'static str)>,
    alias_env_model_path: Option<(String, &'static str)>,
    workspace: &str,
    q4_exists: bool,
    errors: ModelPathPositionErrors,
) -> Result<(String, &'static str, usize, bool), &'static str> {
    let first_looks_like_model = args
        .first()
        .is_some_and(|value| looks_like_gguf_path(value));
    let first_is_1b_alias = args.first().is_some_and(|value| is_llama32_1b_alias(value));
    let first_is_3b_alias = args.first().is_some_and(|value| is_llama32_3b_alias(value));
    if (first_is_1b_alias || first_is_3b_alias)
        && args
            .get(1)
            .is_some_and(|value| looks_like_model_path_argument(value))
    {
        return Err(errors.alias_path);
    }

    match (args.first(), env_model_path) {
        (Some(path), _) if first_looks_like_model => {
            Ok((path.clone(), "explicit argument", 1, false))
        }
        (Some(_), _) if first_is_1b_alias => {
            if let Some((path, source)) = alias_env_model_path {
                if !looks_like_gguf_path(&path) {
                    return Err(errors.alias_env_path);
                }
                Ok((path, source, 1, true))
            } else {
                Ok((
                    resolve_llama32_1b_model_path_with_workspace(None, workspace, q4_exists),
                    if q4_exists {
                        "workspace Q4_0 default"
                    } else {
                        "workspace Q8_0 fallback"
                    },
                    1,
                    true,
                ))
            }
        }
        (Some(_), _) if first_is_3b_alias => {
            if let Some((path, source)) = alias_env_model_path {
                if !looks_like_gguf_path(&path) {
                    return Err(errors.alias_env_path);
                }
                Ok((path, source, 1, false))
            } else {
                Ok((
                    resolve_llama32_3b_model_path_with_workspace(None, workspace),
                    "workspace 3B Q4_0 default",
                    1,
                    false,
                ))
            }
        }
        (Some(path), None) => Ok((path.clone(), "explicit argument", 1, false)),
        (_, Some((path, source))) => {
            if !looks_like_gguf_path(&path) {
                return Err(errors.env_path);
            }
            Ok((path, source, 0, false))
        }
        (None, None) => Err(errors.missing),
    }
}

#[derive(Clone, Copy)]
struct ModelPathPositionErrors {
    env_path: &'static str,
    alias_env_path: &'static str,
    missing: &'static str,
    alias_path: &'static str,
}

fn llama32_1b_workspace_defaults() -> (String, bool) {
    let workspace = env::var(WORKSPACE_ENV).unwrap_or_else(|_| DEFAULT_PI_WORKSPACE.to_owned());
    let q4_path = llama32_1b_model_path(&workspace, LLAMA32_1B_Q4_MODEL);
    let q4_exists = Path::new(&q4_path).is_file();
    (workspace, q4_exists)
}

fn parse_model_1b_args(args: &[String]) -> Result<Model1BAuditArgs, &'static str> {
    let workspace = env::var(WORKSPACE_ENV).unwrap_or_else(|_| DEFAULT_PI_WORKSPACE.to_owned());
    let q4_path = llama32_1b_model_path(&workspace, LLAMA32_1B_Q4_MODEL);
    parse_model_1b_args_with_env(
        args,
        model_1b_env_path_and_source(),
        &workspace,
        Path::new(&q4_path).is_file(),
    )
}

#[cfg(test)]
fn parse_model_1b_args_with_path(
    args: &[String],
    env_model_path: Option<String>,
    workspace: &str,
    q4_exists: bool,
) -> Result<Model1BAuditArgs, &'static str> {
    parse_model_1b_args_with_env(
        args,
        env_model_path.map(|path| (path, DEFAULT_MODEL_GGUF_ENV)),
        workspace,
        q4_exists,
    )
}

fn parse_model_1b_args_with_env(
    args: &[String],
    env_model_path: Option<(String, &'static str)>,
    workspace: &str,
    q4_exists: bool,
) -> Result<Model1BAuditArgs, &'static str> {
    let env_model_path = require_env_gguf_path(
        env_model_path,
        "1B model audit env path must be a .gguf path",
    )?;
    let mut dry_run = false;
    let mut selected_quant = None;
    let mut positionals = Vec::with_capacity(args.len());

    for arg in args {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
            "--q4" => {
                if selected_quant.replace(LLAMA32_1B_Q4_MODEL).is_some() {
                    return Err("1B model audit accepts only one quantization selector");
                }
            }
            "--q8" => {
                if selected_quant.replace(LLAMA32_1B_Q8_MODEL).is_some() {
                    return Err("1B model audit accepts only one quantization selector");
                }
            }
            arg if arg.starts_with('-') => return Err("unknown 1B model audit option"),
            _ => positionals.push(arg.clone()),
        }
    }

    reject_extra_positionals(&positionals, 1, "unexpected extra 1B model audit argument")?;

    if let Some(model_path) = positionals.first()
        && !looks_like_gguf_path(model_path)
    {
        return Err("1B model audit argument must be a .gguf path");
    }

    let q4_model_path = llama32_1b_model_path(workspace, LLAMA32_1B_Q4_MODEL);
    let q8_model_path = llama32_1b_model_path(workspace, LLAMA32_1B_Q8_MODEL);
    let (model_path, model_source) = if let Some(model_path) = positionals.first() {
        (model_path.clone(), "explicit argument")
    } else if let Some(model_name) = selected_quant {
        (
            llama32_1b_model_path(workspace, model_name),
            if model_name == LLAMA32_1B_Q4_MODEL {
                "workspace Q4_0 requested"
            } else {
                "workspace Q8_0 requested"
            },
        )
    } else if let Some((model_path, source)) = env_model_path {
        (model_path, source)
    } else if q4_exists {
        (q4_model_path.clone(), "workspace Q4_0 default")
    } else {
        (q8_model_path.clone(), "workspace Q8_0 fallback")
    };

    Ok(Model1BAuditArgs {
        workspace: workspace.to_owned(),
        q4_model_path,
        q8_model_path,
        model_path,
        model_source,
        dry_run,
    })
}

fn parse_smoke_args(args: &[String]) -> Result<SmokeQ8ModelArgs, &'static str> {
    parse_smoke_args_with_env(args, smoke_model_path_from_env())
}

fn parse_smoke_1b_args(args: &[String]) -> Result<Smoke1BArgs, &'static str> {
    let workspace = env::var(WORKSPACE_ENV).unwrap_or_else(|_| DEFAULT_PI_WORKSPACE.to_owned());
    let q4_path = llama32_1b_model_path(&workspace, LLAMA32_1B_Q4_MODEL);
    let q4_exists = Path::new(&q4_path).is_file();
    let defaults = smoke_defaults_from_env()?;
    parse_smoke_1b_args_with_env_and_defaults(
        args,
        smoke_model_path_and_source_from_env(),
        &workspace,
        q4_exists,
        defaults,
    )
}

fn parse_ready_1b_args(args: &[String]) -> Result<Ready1BArgs, &'static str> {
    let workspace = env::var(WORKSPACE_ENV).unwrap_or_else(|_| DEFAULT_PI_WORKSPACE.to_owned());
    let q4_path = llama32_1b_model_path(&workspace, LLAMA32_1B_Q4_MODEL);
    let q4_exists = Path::new(&q4_path).is_file();
    let smoke_defaults = ready_smoke_defaults_from_env()?;
    let chat_enabled_default = ready_chat_enabled_default_for_args(args)?;
    parse_ready_1b_args_with_env_and_smoke_defaults_and_chat_default(
        args,
        smoke_model_path_and_source_from_env(),
        &workspace,
        q4_exists,
        smoke_defaults,
        chat_enabled_default,
    )
}

fn ready_chat_enabled_default_for_args(args: &[String]) -> Result<bool, &'static str> {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--no-chat" | "--smoke-only" | "--chat"))
    {
        Ok(true)
    } else {
        ready_chat_enabled()
    }
}

#[cfg(test)]
fn parse_ready_1b_args_with_env_and_smoke_defaults(
    args: &[String],
    env_model_path: Option<String>,
    workspace: &str,
    q4_exists: bool,
    smoke_defaults: SmokeDefaults,
) -> Result<Ready1BArgs, &'static str> {
    parse_ready_1b_args_with_env_and_smoke_defaults_and_chat_default(
        args,
        env_model_path.map(|path| (path, DEFAULT_MODEL_GGUF_ENV)),
        workspace,
        q4_exists,
        smoke_defaults,
        true,
    )
}

fn parse_ready_1b_args_with_env_and_smoke_defaults_and_chat_default(
    args: &[String],
    env_model_path: Option<(String, &'static str)>,
    workspace: &str,
    q4_exists: bool,
    smoke_defaults: SmokeDefaults,
    chat_enabled_default: bool,
) -> Result<Ready1BArgs, &'static str> {
    parse_ready_1b_args_inner(
        args,
        env_model_path,
        workspace,
        q4_exists,
        smoke_defaults,
        chat_enabled_default,
    )
}

fn parse_ready_1b_args_inner(
    args: &[String],
    env_model_path: Option<(String, &'static str)>,
    workspace: &str,
    q4_exists: bool,
    smoke_defaults: SmokeDefaults,
    chat_enabled_default: bool,
) -> Result<Ready1BArgs, &'static str> {
    let env_model_path = require_env_gguf_path(
        env_model_path,
        "ready 1B env model path must be a .gguf path",
    )?;
    let mut chat_enabled_override = None;
    let mut dry_run = false;
    let mut selected_quant = None;
    let mut smoke_args = Vec::with_capacity(args.len());

    for arg in args {
        match arg.as_str() {
            "--no-chat" | "--smoke-only" => chat_enabled_override = Some(false),
            "--chat" => chat_enabled_override = Some(true),
            "--dry-run" => dry_run = true,
            "--q4" => {
                if selected_quant.replace(LLAMA32_1B_Q4_MODEL).is_some() {
                    return Err("ready 1B accepts only one quantization selector");
                }
            }
            "--q8" => {
                if selected_quant.replace(LLAMA32_1B_Q8_MODEL).is_some() {
                    return Err("ready 1B accepts only one quantization selector");
                }
            }
            _ => smoke_args.push(arg.clone()),
        }
    }
    reject_path_like_non_gguf_first_arg(
        &smoke_args,
        "ready 1B model argument must be a .gguf path",
    )?;

    let first_looks_like_model = smoke_args
        .first()
        .is_some_and(|value| looks_like_gguf_path(value));
    let (model_path, model_source, mut option_idx) = match (smoke_args.first(), env_model_path) {
        (Some(path), _) if first_looks_like_model => (path.clone(), "explicit argument", 1),
        (_, _) if selected_quant.is_some() => {
            let model_name = selected_quant.expect("selected quant should be present");
            (
                llama32_1b_model_path(workspace, model_name),
                llama32_1b_requested_model_source(model_name),
                0,
            )
        }
        (_, Some((path, source))) => (path, source, 0),
        _ => (
            default_llama32_1b_model_path(workspace, q4_exists),
            if q4_exists {
                "workspace Q4_0 default"
            } else {
                "workspace Q8_0 fallback"
            },
            0,
        ),
    };

    let kind = match smoke_args.get(option_idx).map(String::as_str) {
        Some(value) if SmokeKind::looks_like_arg(value) => {
            option_idx += 1;
            SmokeKind::from_arg(value)
                .ok_or("unknown 1B smoke kind; expected chat, model, q8-chat, or q8-model")?
        }
        _ => smoke_defaults.kind,
    };

    let positional_prompt = smoke_args.get(option_idx).cloned();
    let positional_tokens = parse_optional_positive_usize(
        smoke_args.get(option_idx + 1),
        "ready 1B token count must be a positive integer",
    )?;
    reject_extra_positionals(
        &smoke_args,
        option_idx + 2,
        "unexpected extra ready 1B argument",
    )?;
    let smoke_only = !chat_enabled_override.unwrap_or(chat_enabled_default);

    let (smoke_prompt, smoke_tokens, chat_prompt_override, chat_tokens_override) = if smoke_only {
        (
            positional_prompt.unwrap_or_else(|| smoke_defaults.prompt.clone()),
            positional_tokens.unwrap_or(smoke_defaults.max_tokens),
            None,
            None,
        )
    } else {
        (
            smoke_defaults.prompt,
            smoke_defaults.max_tokens,
            positional_prompt,
            positional_tokens,
        )
    };

    let smoke = Smoke1BArgs {
        kind,
        model_path,
        model_source,
        prompt: smoke_prompt,
        max_tokens: smoke_tokens,
        dry_run: false,
    };
    Ok(Ready1BArgs {
        smoke,
        chat_enabled_override,
        chat_prompt_override,
        chat_tokens_override,
        dry_run,
    })
}

fn parse_smoke_3b_args(args: &[String]) -> Result<Smoke1BArgs, &'static str> {
    let workspace = env::var(WORKSPACE_ENV).unwrap_or_else(|_| DEFAULT_PI_WORKSPACE.to_owned());
    let defaults = smoke_defaults_from_env()?;
    parse_smoke_3b_args_with_env_and_defaults(
        args,
        smoke_model_path_and_source_from_env(),
        &workspace,
        defaults,
    )
}

#[cfg(test)]
fn parse_ready_1b_args_with_env(
    args: &[String],
    env_model_path: Option<String>,
    workspace: &str,
    q4_exists: bool,
) -> Result<Ready1BArgs, &'static str> {
    parse_ready_1b_args_with_env_and_smoke_defaults(
        args,
        env_model_path,
        workspace,
        q4_exists,
        SmokeDefaults::default(),
    )
}

#[cfg(test)]
fn parse_smoke_1b_args_with_env(
    args: &[String],
    env_model_path: Option<String>,
    workspace: &str,
    q4_exists: bool,
) -> Result<Smoke1BArgs, &'static str> {
    parse_smoke_1b_args_with_env_and_defaults(
        args,
        env_model_path.map(|path| (path, DEFAULT_MODEL_GGUF_ENV)),
        workspace,
        q4_exists,
        SmokeDefaults::default(),
    )
}

fn parse_smoke_1b_args_with_env_and_defaults(
    args: &[String],
    env_model_path: Option<(String, &'static str)>,
    workspace: &str,
    q4_exists: bool,
    defaults: SmokeDefaults,
) -> Result<Smoke1BArgs, &'static str> {
    let env_model_path = require_env_gguf_path(
        env_model_path,
        "1B smoke env model path must be a .gguf path",
    )?;
    let mut dry_run = false;
    let mut selected_quant = None;
    let mut positionals = Vec::with_capacity(args.len());
    for arg in args {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
            "--q4" => {
                if selected_quant.replace(LLAMA32_1B_Q4_MODEL).is_some() {
                    return Err("1B smoke accepts only one quantization selector");
                }
            }
            "--q8" => {
                if selected_quant.replace(LLAMA32_1B_Q8_MODEL).is_some() {
                    return Err("1B smoke accepts only one quantization selector");
                }
            }
            arg if arg.starts_with("--") => return Err("unknown 1B smoke option"),
            _ => positionals.push(arg.clone()),
        }
    }
    reject_path_like_non_gguf_first_arg(
        &positionals,
        "1B smoke model argument must be a .gguf path",
    )?;

    let first_looks_like_model = positionals
        .first()
        .is_some_and(|value| looks_like_gguf_path(value));
    let (model_path, model_source, mut option_idx) = match (positionals.first(), env_model_path) {
        (Some(path), _) if first_looks_like_model => (path.clone(), "explicit argument", 1),
        (_, _) if selected_quant.is_some() => {
            let model_name = selected_quant.expect("selected quant should be present");
            (
                llama32_1b_model_path(workspace, model_name),
                llama32_1b_requested_model_source(model_name),
                0,
            )
        }
        (_, Some((path, source))) => (path, source, 0),
        _ => (
            default_llama32_1b_model_path(workspace, q4_exists),
            if q4_exists {
                "workspace Q4_0 default"
            } else {
                "workspace Q8_0 fallback"
            },
            0,
        ),
    };

    let kind = match positionals.get(option_idx).map(String::as_str) {
        Some(value) if SmokeKind::looks_like_arg(value) => {
            option_idx += 1;
            SmokeKind::from_arg(value)
                .ok_or("unknown 1B smoke kind; expected chat, model, q8-chat, or q8-model")?
        }
        _ => defaults.kind,
    };
    let prompt = positionals
        .get(option_idx)
        .cloned()
        .unwrap_or_else(|| defaults.prompt.clone());
    let max_tokens = parse_optional_positive_usize(
        positionals.get(option_idx + 1),
        "1B smoke max_tokens must be a positive integer",
    )?
    .unwrap_or(defaults.max_tokens);
    reject_extra_positionals(
        &positionals,
        option_idx + 2,
        "unexpected extra 1B smoke argument",
    )?;

    Ok(Smoke1BArgs {
        kind,
        model_path,
        model_source,
        prompt,
        max_tokens,
        dry_run,
    })
}

fn smoke_defaults_from_env() -> Result<SmokeDefaults, &'static str> {
    smoke_defaults_from_values(
        SmokeDefaults::default(),
        env::var(SMOKE_KIND_ENV).ok(),
        env::var(SMOKE_PROMPT_ENV).ok(),
        env::var(SMOKE_TOKENS_ENV).ok(),
        "unknown smoke kind env; expected chat, model, q8-chat, or q8-model",
        "smoke env max_tokens must be a positive integer",
    )
}

fn ready_smoke_defaults_from_env() -> Result<SmokeDefaults, &'static str> {
    let defaults = smoke_defaults_from_env()?;
    smoke_defaults_from_values(
        defaults,
        env::var(READY_SMOKE_KIND_ENV).ok(),
        env::var(READY_SMOKE_PROMPT_ENV).ok(),
        env::var(READY_SMOKE_TOKENS_ENV).ok(),
        "unknown ready smoke kind env; expected chat, model, q8-chat, or q8-model",
        "ready smoke env max_tokens must be a positive integer",
    )
}

fn smoke_defaults_from_values(
    mut defaults: SmokeDefaults,
    kind: Option<String>,
    prompt: Option<String>,
    max_tokens: Option<String>,
    kind_error: &'static str,
    token_error: &'static str,
) -> Result<SmokeDefaults, &'static str> {
    if let Some(value) = kind {
        defaults.kind = SmokeKind::from_arg(value.trim()).ok_or(kind_error)?;
    }
    if let Some(value) = prompt.filter(|value| !value.is_empty()) {
        defaults.prompt = value;
    }
    if let Some(value) = parse_optional_positive_usize(max_tokens.as_ref(), token_error)? {
        defaults.max_tokens = value;
    }
    Ok(defaults)
}

#[cfg(test)]
fn parse_smoke_3b_args_with_env(
    args: &[String],
    env_model_path: Option<String>,
    workspace: &str,
) -> Result<Smoke1BArgs, &'static str> {
    parse_smoke_3b_args_with_env_and_defaults(
        args,
        env_model_path.map(|path| (path, DEFAULT_MODEL_GGUF_ENV)),
        workspace,
        SmokeDefaults::default(),
    )
}

fn parse_smoke_3b_args_with_env_and_defaults(
    args: &[String],
    env_model_path: Option<(String, &'static str)>,
    workspace: &str,
    defaults: SmokeDefaults,
) -> Result<Smoke1BArgs, &'static str> {
    let env_model_path = require_env_gguf_path(
        env_model_path,
        "3B smoke env model path must be a .gguf path",
    )?;
    let mut dry_run = false;
    let mut positionals = Vec::with_capacity(args.len());
    for arg in args {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
            _ => positionals.push(arg.clone()),
        }
    }
    reject_path_like_non_gguf_first_arg(
        &positionals,
        "3B smoke model argument must be a .gguf path",
    )?;

    let first_looks_like_model = positionals
        .first()
        .is_some_and(|value| looks_like_gguf_path(value));
    let (model_path, model_source, mut option_idx) = match (positionals.first(), env_model_path) {
        (Some(path), _) if first_looks_like_model => (path.clone(), "explicit argument", 1),
        (_, Some((path, source))) => (path, source, 0),
        _ => (
            default_llama32_3b_model_path(workspace),
            "workspace 3B Q4_0 default",
            0,
        ),
    };

    let kind = match positionals.get(option_idx).map(String::as_str) {
        Some(value) if SmokeKind::looks_like_arg(value) => {
            option_idx += 1;
            SmokeKind::from_arg(value)
                .ok_or("unknown 3B smoke kind; expected chat, model, q8-chat, or q8-model")?
        }
        _ => defaults.kind,
    };
    let prompt = positionals
        .get(option_idx)
        .cloned()
        .unwrap_or_else(|| defaults.prompt.clone());
    let max_tokens = parse_optional_positive_usize(
        positionals.get(option_idx + 1),
        "3B smoke max_tokens must be a positive integer",
    )?
    .unwrap_or(defaults.max_tokens);
    reject_extra_positionals(
        &positionals,
        option_idx + 2,
        "unexpected extra 3B smoke argument",
    )?;

    Ok(Smoke1BArgs {
        kind,
        model_path,
        model_source,
        prompt,
        max_tokens,
        dry_run,
    })
}

fn parse_smoke_args_with_env(
    args: &[String],
    env_model_path: Option<String>,
) -> Result<SmokeQ8ModelArgs, &'static str> {
    let first_looks_like_model = args
        .first()
        .is_some_and(|value| looks_like_gguf_path(value));

    let (model_path, prompt_idx) = match (args.first(), env_model_path) {
        (Some(path), _) if first_looks_like_model => (path.clone(), 1),
        (Some(path), None) => (path.clone(), 1),
        (_, Some(path)) => (path, 0),
        (None, None) => {
            return Err(
                "missing GGUF model path; pass one or set NANOCAMELID_SMOKE_GGUF or NANOCAMELID_MODEL_GGUF",
            );
        }
    };

    let prompt = args
        .get(prompt_idx)
        .cloned()
        .unwrap_or_else(|| "Hello".to_owned());
    let max_tokens = parse_optional_positive_usize(
        args.get(prompt_idx + 1),
        "smoke max_tokens must be a positive integer",
    )?
    .unwrap_or(1);
    reject_extra_positionals(args, prompt_idx + 2, "unexpected extra smoke argument")?;

    Ok(SmokeQ8ModelArgs {
        model_path,
        prompt,
        max_tokens,
    })
}

fn parse_optional_positive_usize(
    value: Option<&String>,
    error: &'static str,
) -> Result<Option<usize>, &'static str> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value.parse::<usize>() {
        Ok(parsed) if parsed > 0 => Ok(Some(parsed)),
        _ => Err(error),
    }
}

fn parse_positive_usize(value: &str, error: &'static str) -> Result<usize, &'static str> {
    match value.parse::<usize>() {
        Ok(parsed) if parsed > 0 => Ok(parsed),
        _ => Err(error),
    }
}

fn parse_port(value: &str) -> Result<u16, &'static str> {
    match value.parse::<u16>() {
        Ok(parsed) if parsed > 0 => Ok(parsed),
        _ => Err("serve --port must be an integer from 1 to 65535"),
    }
}

fn parse_non_negative_f32(value: &str, error: &'static str) -> Result<f32, &'static str> {
    match value.parse::<f32>() {
        Ok(parsed) if parsed.is_finite() && parsed >= 0.0 => Ok(parsed),
        _ => Err(error),
    }
}

fn reject_extra_positionals(
    args: &[String],
    first_extra_idx: usize,
    error: &'static str,
) -> Result<(), &'static str> {
    if args.get(first_extra_idx).is_some() {
        Err(error)
    } else {
        Ok(())
    }
}

fn require_env_gguf_path(
    env_model_path: Option<(String, &'static str)>,
    error: &'static str,
) -> Result<Option<(String, &'static str)>, &'static str> {
    if let Some((path, _source)) = &env_model_path
        && !looks_like_gguf_path(path)
    {
        return Err(error);
    }
    Ok(env_model_path)
}

fn parse_bench_q8_dot_args(args: &[String]) -> Result<BenchQ8DotArgs, &'static str> {
    let iterations = parse_optional_positive_usize(
        args.first(),
        "q8-dot iterations must be a positive integer",
    )?
    .unwrap_or(q8::DEFAULT_DOT_BENCH_ITERATIONS);
    let runs =
        parse_optional_positive_usize(args.get(1), "q8-dot runs must be a positive integer")?
            .unwrap_or(q8::DEFAULT_DOT_BENCH_RUNS);
    reject_extra_positionals(args, 2, "unexpected extra q8-dot benchmark argument")?;

    Ok(BenchQ8DotArgs { iterations, runs })
}

fn parse_bench_q4_layout_args(args: &[String]) -> Result<BenchQ4LayoutArgs, &'static str> {
    let rows =
        parse_optional_positive_usize(args.first(), "q4-layout rows must be a positive integer")?
            .unwrap_or(q8::DEFAULT_Q4_LAYOUT_BENCH_ROWS);
    let cols =
        parse_optional_positive_usize(args.get(1), "q4-layout cols must be a positive integer")?
            .unwrap_or(q8::DEFAULT_Q4_LAYOUT_BENCH_COLS);
    let runs =
        parse_optional_positive_usize(args.get(2), "q4-layout runs must be a positive integer")?
            .unwrap_or(q8::DEFAULT_DOT_BENCH_RUNS);
    reject_extra_positionals(args, 3, "unexpected extra q4-layout benchmark argument")?;

    Ok(BenchQ4LayoutArgs { rows, cols, runs })
}

fn parse_bench_q4_prefill_args(args: &[String]) -> Result<BenchQ4PrefillArgs, &'static str> {
    let prompt_len = parse_optional_positive_usize(
        args.first(),
        "q4-prefill prompt_len must be a positive integer",
    )?
    .unwrap_or(DEFAULT_Q4_PREFILL_PROMPT_LEN);
    let batch_size = parse_optional_positive_usize(
        args.get(1),
        "q4-prefill batch_size must be a positive integer",
    )?
    .unwrap_or(DEFAULT_Q4_PREFILL_BATCH);
    let runs =
        parse_optional_positive_usize(args.get(2), "q4-prefill runs must be a positive integer")?
            .unwrap_or(DEFAULT_Q4_PREFILL_RUNS);
    reject_extra_positionals(args, 3, "unexpected extra q4-prefill benchmark argument")?;

    Ok(BenchQ4PrefillArgs {
        prompt_len,
        batch_size,
        runs,
    })
}

fn parse_bench_1b_args(args: &[String]) -> Result<Bench1BArgs, &'static str> {
    let workspace = env::var(WORKSPACE_ENV).unwrap_or_else(|_| DEFAULT_PI_WORKSPACE.to_owned());
    let q4_path = llama32_1b_model_path(&workspace, LLAMA32_1B_Q4_MODEL);
    parse_bench_1b_args_with_env(
        args,
        smoke_model_path_and_source_from_env(),
        &workspace,
        Path::new(&q4_path).is_file(),
    )
}

fn parse_evidence_1b_args(args: &[String]) -> Result<Evidence1BArgs, &'static str> {
    let workspace = env::var(WORKSPACE_ENV).unwrap_or_else(|_| DEFAULT_PI_WORKSPACE.to_owned());
    let q4_path = llama32_1b_model_path(&workspace, LLAMA32_1B_Q4_MODEL);
    parse_evidence_1b_args_with_env(
        args,
        smoke_model_path_and_source_from_env(),
        &workspace,
        Path::new(&q4_path).is_file(),
    )
}

#[cfg(test)]
fn parse_evidence_1b_args_with_path(
    args: &[String],
    env_model_path: Option<String>,
    workspace: &str,
    q4_exists: bool,
) -> Result<Evidence1BArgs, &'static str> {
    parse_evidence_1b_args_with_env(
        args,
        env_model_path.map(|path| (path, DEFAULT_MODEL_GGUF_ENV)),
        workspace,
        q4_exists,
    )
}

fn parse_evidence_1b_args_with_env(
    args: &[String],
    env_model_path: Option<(String, &'static str)>,
    workspace: &str,
    q4_exists: bool,
) -> Result<Evidence1BArgs, &'static str> {
    let env_model_path = require_env_gguf_path(
        env_model_path,
        "1B evidence env model path must be a .gguf path",
    )?;
    let mut dry_run = false;
    let mut selected_quant = None;
    let mut positionals = Vec::with_capacity(args.len());
    for arg in args {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
            "--q4" => {
                if selected_quant.replace(LLAMA32_1B_Q4_MODEL).is_some() {
                    return Err("1B evidence accepts only one quantization selector");
                }
            }
            "--q8" => {
                if selected_quant.replace(LLAMA32_1B_Q8_MODEL).is_some() {
                    return Err("1B evidence accepts only one quantization selector");
                }
            }
            arg if arg.starts_with('-') => return Err("unknown 1B evidence option"),
            _ => positionals.push(arg.clone()),
        }
    }
    reject_extra_positionals(&positionals, 1, "unexpected extra 1B evidence argument")?;
    reject_path_like_non_gguf_first_arg(
        &positionals,
        "1B evidence model argument must be a .gguf path",
    )?;

    let q4_model_path = llama32_1b_model_path(workspace, LLAMA32_1B_Q4_MODEL);
    let q8_model_path = llama32_1b_model_path(workspace, LLAMA32_1B_Q8_MODEL);
    let (model_path, model_source) = if let Some(model_path) = positionals.first() {
        (model_path.clone(), "explicit argument")
    } else if let Some(model_name) = selected_quant {
        (
            llama32_1b_model_path(workspace, model_name),
            llama32_1b_requested_model_source(model_name),
        )
    } else if let Some((model_path, source)) = env_model_path {
        (model_path, source)
    } else if q4_exists {
        (q4_model_path.clone(), "workspace Q4_0 default")
    } else {
        (q8_model_path.clone(), "workspace Q8_0 fallback")
    };

    let smoke = smoke_defaults_from_env()?;
    let context_packs_raw = env::var(CONTEXT_PACKS_ENV)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_1B_CONTEXT_PACKS.to_owned());
    let context_packs = parse_context_packs(&context_packs_raw)?;
    let prefill_prompt = env::var(PREFILL_PROMPT_ENV)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_1B_PREFILL_PROMPT.to_owned());
    let prefill_tokens = match env::var(PREFILL_TOKENS_ENV)
        .ok()
        .filter(|value| !value.is_empty())
    {
        Some(value) => parse_optional_positive_usize(
            Some(&value),
            "1B prefill benchmark max_tokens must be a positive integer",
        )?
        .expect("env value is present"),
        None => DEFAULT_1B_PREFILL_TOKENS,
    };
    let prefill_temp = env::var(PREFILL_TEMP_ENV)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_1B_PREFILL_TEMP.to_owned());
    parse_non_negative_f32(
        &prefill_temp,
        "1B prefill benchmark temp must be a non-negative number",
    )?;
    let prefill_batches_raw = env::var(PREFILL_BATCHES_ENV)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_1B_PREFILL_BATCHES.to_owned());
    let prefill_batches = parse_prefill_batches(&prefill_batches_raw)?;
    let prefill_batch = prefill_batch_size_from_env()?;

    let prefill = Bench1BArgs {
        workspace: workspace.to_owned(),
        q4_model_path: q4_model_path.clone(),
        q8_model_path: q8_model_path.clone(),
        model_path: model_path.clone(),
        model_source,
        prompt: prefill_prompt,
        max_tokens: prefill_tokens,
        temp: prefill_temp,
        batches: prefill_batches,
        dry_run,
    };

    Ok(Evidence1BArgs {
        workspace: workspace.to_owned(),
        q4_model_path,
        q8_model_path,
        model_path,
        model_source,
        smoke,
        prefill_batch,
        context_packs,
        prefill,
        dry_run,
    })
}

#[cfg(test)]
fn parse_bench_1b_args_with_path(
    args: &[String],
    env_model_path: Option<String>,
    workspace: &str,
    q4_exists: bool,
) -> Result<Bench1BArgs, &'static str> {
    parse_bench_1b_args_with_env(
        args,
        env_model_path.map(|path| (path, DEFAULT_MODEL_GGUF_ENV)),
        workspace,
        q4_exists,
    )
}

fn parse_bench_1b_args_with_env(
    args: &[String],
    env_model_path: Option<(String, &'static str)>,
    workspace: &str,
    q4_exists: bool,
) -> Result<Bench1BArgs, &'static str> {
    let env_model_path = require_env_gguf_path(
        env_model_path,
        "1B benchmark env model path must be a .gguf path",
    )?;
    let mut dry_run = false;
    let mut selected_quant = None;
    let mut positionals = Vec::with_capacity(args.len());
    for arg in args {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
            "--q4" => {
                if selected_quant.replace(LLAMA32_1B_Q4_MODEL).is_some() {
                    return Err("1B prefill benchmark accepts only one quantization selector");
                }
            }
            "--q8" => {
                if selected_quant.replace(LLAMA32_1B_Q8_MODEL).is_some() {
                    return Err("1B prefill benchmark accepts only one quantization selector");
                }
            }
            arg if arg.starts_with("--") => return Err("unknown 1B prefill benchmark option"),
            _ => positionals.push(arg.clone()),
        }
    }
    reject_path_like_non_gguf_first_arg(
        &positionals,
        "1B prefill benchmark model argument must be a .gguf path",
    )?;

    let first_looks_like_model = positionals
        .first()
        .is_some_and(|value| looks_like_gguf_path(value));
    let q4_model_path = llama32_1b_model_path(workspace, LLAMA32_1B_Q4_MODEL);
    let q8_model_path = llama32_1b_model_path(workspace, LLAMA32_1B_Q8_MODEL);
    let (model_path, model_source, option_idx) = match (positionals.first(), env_model_path) {
        (Some(path), _) if first_looks_like_model => (path.clone(), "explicit argument", 1),
        (_, _) if selected_quant.is_some() => {
            let model_name = selected_quant.expect("selected quant should be present");
            (
                llama32_1b_model_path(workspace, model_name),
                llama32_1b_requested_model_source(model_name),
                0,
            )
        }
        (_, Some((path, source))) => (path, source, 0),
        _ if q4_exists => (q4_model_path.clone(), "workspace Q4_0 default", 0),
        _ => (q8_model_path.clone(), "workspace Q8_0 fallback", 0),
    };

    let prompt = positionals
        .get(option_idx)
        .cloned()
        .or_else(|| {
            env::var(PREFILL_PROMPT_ENV)
                .ok()
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| DEFAULT_1B_PREFILL_PROMPT.to_owned());
    let max_tokens = match positionals.get(option_idx + 1) {
        Some(value) => parse_optional_positive_usize(
            Some(value),
            "1B prefill benchmark max_tokens must be a positive integer",
        )?
        .expect("positional value is present"),
        None => match env::var(PREFILL_TOKENS_ENV)
            .ok()
            .filter(|value| !value.is_empty())
        {
            Some(value) => parse_optional_positive_usize(
                Some(&value),
                "1B prefill benchmark max_tokens must be a positive integer",
            )?
            .expect("env value is present"),
            None => DEFAULT_1B_PREFILL_TOKENS,
        },
    };
    let temp = positionals
        .get(option_idx + 2)
        .cloned()
        .or_else(|| {
            env::var(PREFILL_TEMP_ENV)
                .ok()
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| DEFAULT_1B_PREFILL_TEMP.to_owned());
    parse_non_negative_f32(
        &temp,
        "1B prefill benchmark temp must be a non-negative number",
    )?;
    let batches_raw = positionals
        .get(option_idx + 3)
        .cloned()
        .or_else(|| {
            env::var(PREFILL_BATCHES_ENV)
                .ok()
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| DEFAULT_1B_PREFILL_BATCHES.to_owned());
    let batches = parse_prefill_batches(&batches_raw)?;
    reject_extra_positionals(
        &positionals,
        option_idx + 4,
        "unexpected extra 1B prefill benchmark argument",
    )?;

    Ok(Bench1BArgs {
        workspace: workspace.to_owned(),
        q4_model_path,
        q8_model_path,
        model_path,
        model_source,
        prompt,
        max_tokens,
        temp,
        batches,
        dry_run,
    })
}

fn parse_prefill_batches(value: &str) -> Result<Vec<usize>, &'static str> {
    let batches = value
        .split(',')
        .map(|part| part.trim())
        .map(|part| {
            if part.is_empty() {
                Err("1B prefill benchmark batches must be positive integers")
            } else {
                Ok(part)
            }
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flat_map(|part| part.split_whitespace())
        .map(|part| {
            part.parse::<usize>()
                .ok()
                .filter(|&batch| batch > 0)
                .ok_or("1B prefill benchmark batches must be positive integers")
        })
        .collect::<Result<Vec<_>, _>>()?;

    if batches.is_empty() {
        Err("1B prefill benchmark batches must include at least one positive integer")
    } else if has_duplicate_usize(&batches) {
        Err("1B prefill benchmark batches must be unique")
    } else {
        Ok(batches)
    }
}

fn parse_context_packs(value: &str) -> Result<Vec<usize>, &'static str> {
    let packs = value
        .split(',')
        .map(|part| part.trim())
        .map(|part| {
            if part.is_empty() {
                Err("1B evidence context packs must be positive integers")
            } else {
                Ok(part)
            }
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flat_map(|part| part.split_whitespace())
        .map(|part| {
            part.parse::<usize>()
                .ok()
                .filter(|&pack| pack > 0)
                .ok_or("1B evidence context packs must be positive integers")
        })
        .collect::<Result<Vec<_>, _>>()?;

    if packs.is_empty() {
        Err("1B evidence context packs must include at least one positive integer")
    } else if has_duplicate_usize(&packs) {
        Err("1B evidence context packs must be unique")
    } else {
        Ok(packs)
    }
}

fn has_duplicate_usize(values: &[usize]) -> bool {
    values
        .iter()
        .enumerate()
        .any(|(idx, value)| values[..idx].contains(value))
}

fn looks_like_gguf_path(value: &str) -> bool {
    value
        .trim_end_matches('/')
        .to_ascii_lowercase()
        .ends_with(".gguf")
}

fn reject_path_like_non_gguf_first_arg(
    args: &[String],
    error: &'static str,
) -> Result<(), &'static str> {
    if args
        .first()
        .is_some_and(|value| looks_like_non_gguf_model_path(value))
    {
        Err(error)
    } else {
        Ok(())
    }
}

fn looks_like_model_path_argument(value: &str) -> bool {
    looks_like_gguf_path(value) || looks_like_non_gguf_model_path(value)
}

fn looks_like_non_gguf_model_path(value: &str) -> bool {
    let value = value.trim_end_matches('/');
    !looks_like_gguf_path(value)
        && (value.contains('/') || value.contains('\\') || value.starts_with('~'))
}

fn is_llama32_1b_alias(value: &str) -> bool {
    matches!(value, "1b" | "llama32-1b" | "llama-3.2-1b")
}

fn is_llama32_3b_alias(value: &str) -> bool {
    matches!(value, "3b" | "llama32-3b" | "llama-3.2-3b")
}

fn resolve_llama32_1b_model_path_with_workspace(
    env_model_path: Option<String>,
    workspace: &str,
    q4_exists: bool,
) -> String {
    env_model_path.unwrap_or_else(|| default_llama32_1b_model_path(workspace, q4_exists))
}

fn resolve_llama32_1b_model_path_and_source(
    env_model_path: Option<(String, &'static str)>,
    workspace: &str,
    q4_exists: bool,
) -> Result<(String, &'static str), &'static str> {
    match env_model_path {
        Some((path, source)) => {
            if !looks_like_gguf_path(&path) {
                return Err("inspect alias env model path must be a .gguf path");
            }
            Ok((path, source))
        }
        None if q4_exists => Ok((
            llama32_1b_model_path(workspace, LLAMA32_1B_Q4_MODEL),
            "workspace Q4_0 default",
        )),
        None => Ok((
            llama32_1b_model_path(workspace, LLAMA32_1B_Q8_MODEL),
            "workspace Q8_0 fallback",
        )),
    }
}

fn default_llama32_1b_model_path(workspace: &str, q4_exists: bool) -> String {
    let model_name = if q4_exists {
        LLAMA32_1B_Q4_MODEL
    } else {
        LLAMA32_1B_Q8_MODEL
    };
    llama32_1b_model_path(workspace, model_name)
}

fn llama32_1b_requested_model_source(model_name: &str) -> &'static str {
    if model_name == LLAMA32_1B_Q4_MODEL {
        "workspace Q4_0 requested"
    } else {
        "workspace Q8_0 requested"
    }
}

fn resolve_llama32_3b_model_path_with_workspace(
    env_model_path: Option<String>,
    workspace: &str,
) -> String {
    env_model_path.unwrap_or_else(|| default_llama32_3b_model_path(workspace))
}

fn resolve_llama32_3b_model_path_and_source(
    env_model_path: Option<(String, &'static str)>,
    workspace: &str,
) -> Result<(String, &'static str), &'static str> {
    match env_model_path {
        Some((path, source)) => {
            if !looks_like_gguf_path(&path) {
                return Err("inspect alias env model path must be a .gguf path");
            }
            Ok((path, source))
        }
        None => Ok((
            default_llama32_3b_model_path(workspace),
            "workspace Q4_0 default",
        )),
    }
}

fn default_llama32_3b_model_path(workspace: &str) -> String {
    llama32_1b_model_path(workspace, LLAMA32_3B_Q4_MODEL)
}

fn llama32_1b_model_path(workspace: &str, model_name: &str) -> String {
    Path::new(workspace)
        .join("models")
        .join(model_name)
        .to_string_lossy()
        .into_owned()
}

fn default_model_dir() -> String {
    env::var(MODEL_DIR_ENV).unwrap_or_else(|_| {
        let workspace = env::var(WORKSPACE_ENV).unwrap_or_else(|_| DEFAULT_PI_WORKSPACE.to_owned());
        Path::new(&workspace)
            .join("models")
            .to_string_lossy()
            .into_owned()
    })
}

fn default_model_path_from_env() -> Option<String> {
    env::var(DEFAULT_MODEL_GGUF_ENV).ok()
}

fn default_model_path_and_source_from_env() -> Option<(String, &'static str)> {
    env::var(DEFAULT_MODEL_GGUF_ENV)
        .ok()
        .map(|path| (path, DEFAULT_MODEL_GGUF_ENV))
}

fn smoke_model_path_from_env() -> Option<String> {
    env::var(SMOKE_MODEL_GGUF_ENV)
        .ok()
        .or_else(default_model_path_from_env)
}

fn smoke_model_path_and_source_from_env() -> Option<(String, &'static str)> {
    env::var(SMOKE_MODEL_GGUF_ENV)
        .ok()
        .map(|path| (path, SMOKE_MODEL_GGUF_ENV))
        .or_else(|| {
            env::var(DEFAULT_MODEL_GGUF_ENV)
                .ok()
                .map(|path| (path, DEFAULT_MODEL_GGUF_ENV))
        })
}

fn model_1b_env_path_and_source() -> Option<(String, &'static str)> {
    env::var(SMOKE_MODEL_GGUF_ENV)
        .ok()
        .map(|path| (path, SMOKE_MODEL_GGUF_ENV))
        .or_else(|| {
            env::var(DEFAULT_MODEL_GGUF_ENV)
                .ok()
                .map(|path| (path, DEFAULT_MODEL_GGUF_ENV))
        })
}

fn llama32_1b_model_not_found_message(model_path: &Path) -> String {
    format!(
        "1B model not found: {}\nSet {SMOKE_MODEL_GGUF_ENV} or {DEFAULT_MODEL_GGUF_ENV}, pass an explicit .gguf path, or place {LLAMA32_1B_Q4_MODEL} or {LLAMA32_1B_Q8_MODEL} under ${{{WORKSPACE_ENV}:-{DEFAULT_PI_WORKSPACE}}}/models.",
        model_path.display()
    )
}

fn llama32_3b_model_not_found_message(model_path: &Path) -> String {
    format!(
        "3B model not found: {}\nSet {SMOKE_MODEL_GGUF_ENV} or {DEFAULT_MODEL_GGUF_ENV}, pass an explicit .gguf path, or place {LLAMA32_3B_Q4_MODEL} under ${{{WORKSPACE_ENV}:-{DEFAULT_PI_WORKSPACE}}}/models.",
        model_path.display()
    )
}

fn model_dir_not_found_message(dir: &Path) -> String {
    format!(
        "models directory not found: {}\nSet {MODEL_DIR_ENV}, set {WORKSPACE_ENV}, pass --dir <path>, or place GGUF files under ${{{WORKSPACE_ENV}:-{DEFAULT_PI_WORKSPACE}}}/models.",
        dir.display()
    )
}

fn run_doctor(parsed: DoctorArgs) -> ExitCode {
    let model_dir = PathBuf::from(default_model_dir());
    let model_dir_exists = model_dir.is_dir();
    let model_count = if parsed.dry_run {
        None
    } else if model_dir_exists {
        scan_model_dir(&model_dir, false)
            .ok()
            .map(|entries| entries.len())
    } else {
        Some(0)
    };
    let (workspace, q4_exists) = llama32_1b_workspace_defaults();
    let q4_model = llama32_1b_model_path(&workspace, LLAMA32_1B_Q4_MODEL);
    let q8_model = llama32_1b_model_path(&workspace, LLAMA32_1B_Q8_MODEL);
    let three_b_model = default_llama32_3b_model_path(&workspace);
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let device_tree_model = fs::read_to_string("/proc/device-tree/model").unwrap_or_default();
    let cpu_model = cpu_model(&cpuinfo)
        .or_else(|| device_model(&device_tree_model))
        .unwrap_or("unknown");
    let status = if model_dir_exists { "ok" } else { "warn" };

    println!("NanoCamelid doctor");
    println!("status: {status}");
    println!("version: {}", env!("CARGO_PKG_VERSION"));
    println!("host: {} {}", env::consts::OS, env::consts::ARCH);
    println!("cpu_model: {cpu_model}");
    println!("model_dir: {}", model_dir.display());
    println!("model_dir_exists: {model_dir_exists}");
    if let Some(model_count) = model_count {
        println!("model_count: {model_count}");
    } else {
        println!("model_count: not_checked");
    }
    println!("workspace: {workspace}");
    println!("default_1b_q4: {q4_model}");
    println!("default_1b_q4_exists: {}", Path::new(&q4_model).is_file());
    println!("default_1b_q8: {q8_model}");
    println!("default_1b_q8_exists: {}", Path::new(&q8_model).is_file());
    println!("default_3b_q4: {three_b_model}");
    println!(
        "default_3b_q4_exists: {}",
        Path::new(&three_b_model).is_file()
    );
    println!(
        "default_1b_selected: {}",
        default_llama32_1b_model_path(&workspace, q4_exists)
    );
    println!(
        "{DEFAULT_MODEL_GGUF_ENV}_set: {}",
        env::var(DEFAULT_MODEL_GGUF_ENV).is_ok()
    );
    println!(
        "{SMOKE_MODEL_GGUF_ENV}_set: {}",
        env::var(SMOKE_MODEL_GGUF_ENV).is_ok()
    );
    println!("{MODEL_DIR_ENV}_set: {}", env::var(MODEL_DIR_ENV).is_ok());
    if !model_dir_exists {
        println!(
            "next_action: create {}, set {MODEL_DIR_ENV}, or run `nanocamelid models list --dir <path>`.",
            model_dir.display()
        );
    }
    if parsed.json {
        println!(
            "json: {{\"command\":\"doctor\",\"status\":\"{}\",\"version\":{},\"host\":{},\"arch\":{},\"model_dir\":{},\"model_dir_exists\":{},\"model_count\":{}}}",
            status,
            json_string(env!("CARGO_PKG_VERSION")),
            json_string(env::consts::OS),
            json_string(env::consts::ARCH),
            json_string(&model_dir.display().to_string()),
            model_dir_exists,
            model_count
                .map(|count| count.to_string())
                .unwrap_or_else(|| "null".to_owned())
        );
    }
    ExitCode::SUCCESS
}

fn run_serve(parsed: ServeArgs) -> ExitCode {
    if parsed.dry_run {
        print_serve_dry_run(&parsed);
        return ExitCode::SUCCESS;
    }

    let addr = format!("{}:{}", parsed.host, parsed.port);
    let listener = match TcpListener::bind(&addr) {
        Ok(listener) => listener,
        Err(err) => {
            eprintln!("serve bind failed for {addr}: {err}");
            eprintln!(
                "next_action: pass --host {DEFAULT_API_HOST}, choose a free --port, or stop the process already using the port."
            );
            return ExitCode::FAILURE;
        }
    };

    println!("NanoCamelid serve");
    println!("listen: http://{addr}");
    println!("model_dir: {}", parsed.model_dir);
    println!("api_key_required: {}", parsed.api_key.is_some());
    println!("max_request_bytes: {}", parsed.max_request_bytes);
    println!("max_input_tokens: {}", parsed.max_input_tokens);
    println!("max_output_tokens: {}", parsed.max_output_tokens);
    println!("endpoints: /health /v1/models /v1/completions /v1/chat/completions /metrics");

    let started = Instant::now();
    let mut request_count = 0usize;
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                request_count = request_count.saturating_add(1);
                if let Err(err) = handle_serve_connection(
                    &mut stream,
                    &parsed,
                    request_count,
                    started.elapsed().as_secs_f64(),
                ) {
                    eprintln!("serve request failed: {err}");
                }
            }
            Err(err) => eprintln!("serve accept failed: {err}"),
        }
    }

    ExitCode::SUCCESS
}

fn print_serve_dry_run(parsed: &ServeArgs) {
    println!("NanoCamelid serve dry run");
    println!("listen: http://{}:{}", parsed.host, parsed.port);
    println!("model_dir: {}", parsed.model_dir);
    println!("api_key_required: {}", parsed.api_key.is_some());
    println!("max_request_bytes: {}", parsed.max_request_bytes);
    println!("max_input_tokens: {}", parsed.max_input_tokens);
    println!("max_output_tokens: {}", parsed.max_output_tokens);
    println!("endpoints: /health /v1/models /v1/completions /v1/chat/completions /metrics");
    println!(
        "serve_command: {}",
        shell_command(&[
            "nanocamelid",
            "serve",
            "--host",
            &parsed.host,
            "--port",
            &parsed.port.to_string(),
            "--model-dir",
            &parsed.model_dir,
            "--max-request-bytes",
            &parsed.max_request_bytes.to_string(),
            "--max-input-tokens",
            &parsed.max_input_tokens.to_string(),
            "--max-output-tokens",
            &parsed.max_output_tokens.to_string(),
        ])
    );
}

#[derive(Debug, PartialEq, Eq)]
struct HttpRequest {
    method: String,
    path: String,
    authorization: Option<String>,
    body: String,
}

#[derive(Debug, PartialEq)]
struct ValidatedApiCompletionRequest {
    model: String,
    prompts: Vec<String>,
    input_tokens: usize,
    requested_output_tokens: usize,
    temperature: f32,
}

#[derive(Debug, PartialEq, Eq)]
struct ApiChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, PartialEq)]
struct ValidatedApiChatCompletionRequest {
    model: String,
    messages: Vec<ApiChatMessage>,
    input_tokens: usize,
    requested_output_tokens: usize,
    temperature: f32,
}

#[derive(Debug, PartialEq)]
struct ApiCompletionChoice {
    index: usize,
    text: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    finish_reason: &'static str,
}

#[derive(Clone, Copy)]
struct ApiGenerationRuntime<'a> {
    config: &'a model::LlamaModelConfig,
    weights: &'a model::LlamaWeights,
    tokenizer: &'a tokenizer::Tokenizer,
    runtime_options: inference::LlamaRuntimeOptions,
}

#[derive(Debug, PartialEq, Eq)]
struct ServeApiError {
    status: u16,
    error_type: &'static str,
    code: &'static str,
    message: &'static str,
}

fn handle_serve_connection(
    stream: &mut TcpStream,
    parsed: &ServeArgs,
    request_count: usize,
    uptime_seconds: f64,
) -> io::Result<()> {
    let Some(request_text) = read_http_request_text(stream, parsed.max_request_bytes)? else {
        return Ok(());
    };
    let request_text = match request_text {
        Ok(request_text) => request_text,
        Err(err) => {
            return write_json_error(stream, err.status, err.error_type, err.code, err.message);
        }
    };

    let Some(request) = parse_http_request(&request_text) else {
        return write_json_error(
            stream,
            400,
            "invalid_request_error",
            "bad_request",
            "Request line is missing or invalid.",
        );
    };

    if let Some(api_key) = &parsed.api_key {
        let expected = format!("Bearer {api_key}");
        if request.authorization.as_deref() != Some(expected.as_str()) {
            return write_json_error(
                stream,
                401,
                "authentication_error",
                "unauthorized",
                "Missing or invalid bearer token.",
            );
        }
    }

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => write_json_response(
            stream,
            200,
            &format!(
                "{{\"status\":\"ok\",\"version\":{},\"model_dir\":{},\"api_key_required\":{}}}",
                json_string(env!("CARGO_PKG_VERSION")),
                json_string(&parsed.model_dir),
                parsed.api_key.is_some()
            ),
        ),
        ("GET", "/v1/models") => write_json_response(stream, 200, &serve_models_json(parsed)),
        ("GET", "/metrics") => write_text_response(
            stream,
            200,
            &serve_metrics_text(request_count, uptime_seconds, parsed),
            "text/plain; charset=utf-8",
        ),
        ("POST", "/v1/completions") => {
            match validate_api_completion_request(&request.body, parsed)
                .and_then(|request| serve_completion_response_json(&request, parsed))
            {
                Ok(response) => write_json_response(stream, 200, &response),
                Err(err) => {
                    write_json_error(stream, err.status, err.error_type, err.code, err.message)
                }
            }
        }
        ("POST", "/v1/chat/completions") => {
            match validate_api_chat_completion_request(&request.body, parsed)
                .and_then(|request| serve_chat_completion_response_json(&request, parsed))
            {
                Ok(response) => write_json_response(stream, 200, &response),
                Err(err) => {
                    write_json_error(stream, err.status, err.error_type, err.code, err.message)
                }
            }
        }
        ("GET", "/v1/completions") | ("GET", "/v1/chat/completions") => write_json_error(
            stream,
            405,
            "invalid_request_error",
            "method_not_allowed",
            "Use POST for completion endpoints.",
        ),
        _ => write_json_error(
            stream,
            404,
            "invalid_request_error",
            "not_found",
            "Unknown NanoCamelid API endpoint.",
        ),
    }
}

fn read_http_request_text(
    stream: &mut TcpStream,
    max_request_bytes: usize,
) -> io::Result<Option<Result<String, ServeApiError>>> {
    let mut buffer = vec![0u8; max_request_bytes.clamp(1, 16 * 1024)];
    let bytes_read = stream.read(&mut buffer)?;
    if bytes_read == 0 {
        return Ok(None);
    }

    let mut request_bytes = buffer[..bytes_read].to_vec();
    if request_bytes.len() > max_request_bytes {
        return Ok(Some(Err(request_too_large_error())));
    }

    let Some(header_end) = http_header_end(&request_bytes) else {
        return Ok(Some(Ok(
            String::from_utf8_lossy(&request_bytes).into_owned()
        )));
    };
    let headers = String::from_utf8_lossy(&request_bytes[..header_end]);
    let content_length = match parse_content_length(&headers) {
        Ok(content_length) => content_length,
        Err(err) => return Ok(Some(Err(err))),
    };
    let Some(content_length) = content_length else {
        return Ok(Some(Ok(
            String::from_utf8_lossy(&request_bytes).into_owned()
        )));
    };
    let target_len = header_end.saturating_add(content_length);
    if target_len > max_request_bytes {
        return Ok(Some(Err(request_too_large_error())));
    }
    while request_bytes.len() < target_len {
        let remaining = target_len - request_bytes.len();
        let mut chunk = vec![0u8; remaining.min(16 * 1024)];
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        request_bytes.extend_from_slice(&chunk[..read]);
    }
    request_bytes.truncate(target_len.min(request_bytes.len()));
    Ok(Some(Ok(
        String::from_utf8_lossy(&request_bytes).into_owned()
    )))
}

fn http_header_end(request_bytes: &[u8]) -> Option<usize> {
    request_bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|idx| idx + 4)
        .or_else(|| {
            request_bytes
                .windows(2)
                .position(|window| window == b"\n\n")
                .map(|idx| idx + 2)
        })
}

fn parse_content_length(headers: &str) -> Result<Option<usize>, ServeApiError> {
    headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then_some(value.trim())
        })
        .map(|value| {
            value.parse::<usize>().map_err(|_| {
                serve_api_error(
                    400,
                    "invalid_request_error",
                    "invalid_content_length",
                    "Content-Length must be a non-negative integer.",
                )
            })
        })
        .transpose()
}

fn request_too_large_error() -> ServeApiError {
    serve_api_error(
        413,
        "invalid_request_error",
        "request_too_large",
        "HTTP request exceeds the configured max request byte cap.",
    )
}

fn parse_http_request(request_text: &str) -> Option<HttpRequest> {
    let (headers, body) = request_text
        .split_once("\r\n\r\n")
        .or_else(|| request_text.split_once("\n\n"))
        .unwrap_or((request_text, ""));
    let mut lines = headers.lines();
    let mut request_line = lines.next()?.split_whitespace();
    let method = request_line.next()?.to_owned();
    let path = request_line.next()?.split('?').next()?.to_owned();
    request_line.next()?;
    let authorization = lines.find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case("authorization")
            .then(|| value.trim().to_owned())
    });

    Some(HttpRequest {
        method,
        path,
        authorization,
        body: body.to_owned(),
    })
}

fn validate_api_completion_request(
    body: &str,
    parsed: &ServeArgs,
) -> Result<ValidatedApiCompletionRequest, ServeApiError> {
    if body.trim().is_empty() {
        return Err(serve_api_error(
            400,
            "invalid_request_error",
            "missing_body",
            "Request body must be a JSON object.",
        ));
    }
    if !looks_like_json_object(body) {
        return Err(invalid_json_error());
    }
    let model = required_json_string_field(body, "model", "missing_model")?;
    let prompts = required_completion_prompts(body)?;
    let input_tokens = prompts
        .iter()
        .map(|prompt| estimate_request_tokens(prompt))
        .sum::<usize>();
    let requested_output_tokens = json_usize_field(body, "max_tokens")
        .map_err(|_| {
            serve_api_error(
                400,
                "invalid_request_error",
                "invalid_max_tokens",
                "max_tokens must be a positive integer.",
            )
        })?
        .unwrap_or(parsed.max_output_tokens);
    let temperature = json_f32_field(body, "temperature")
        .map_err(|_| {
            serve_api_error(
                400,
                "invalid_request_error",
                "invalid_temperature",
                "temperature must be a non-negative number.",
            )
        })?
        .unwrap_or(0.0);
    validate_api_output_cap(requested_output_tokens, parsed)?;
    Ok(ValidatedApiCompletionRequest {
        model,
        prompts,
        input_tokens,
        requested_output_tokens,
        temperature,
    })
}

fn validate_api_chat_completion_request(
    body: &str,
    parsed: &ServeArgs,
) -> Result<ValidatedApiChatCompletionRequest, ServeApiError> {
    if body.trim().is_empty() {
        return Err(serve_api_error(
            400,
            "invalid_request_error",
            "missing_body",
            "Request body must be a JSON object.",
        ));
    }
    if !looks_like_json_object(body) {
        return Err(invalid_json_error());
    }
    let model = required_json_string_field(body, "model", "missing_model")?;
    let messages = required_chat_messages(body)?;
    let input_tokens = messages
        .iter()
        .map(|message| estimate_request_tokens(&message.content))
        .sum::<usize>();
    let requested_output_tokens = json_usize_field(body, "max_tokens")
        .map_err(|_| {
            serve_api_error(
                400,
                "invalid_request_error",
                "invalid_max_tokens",
                "max_tokens must be a positive integer.",
            )
        })?
        .unwrap_or(parsed.max_output_tokens);
    let temperature = json_f32_field(body, "temperature")
        .map_err(|_| {
            serve_api_error(
                400,
                "invalid_request_error",
                "invalid_temperature",
                "temperature must be a non-negative number.",
            )
        })?
        .unwrap_or(0.0);
    validate_api_output_cap(requested_output_tokens, parsed)?;
    Ok(ValidatedApiChatCompletionRequest {
        model,
        messages,
        input_tokens,
        requested_output_tokens,
        temperature,
    })
}

fn required_json_string_field(
    body: &str,
    field: &str,
    missing_code: &'static str,
) -> Result<String, ServeApiError> {
    json_string_field(body, field)
        .map_err(|_| invalid_json_error())?
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            serve_api_error(
                400,
                "invalid_request_error",
                missing_code,
                "Request JSON is missing a required non-empty field.",
            )
        })
}

fn required_completion_prompts(body: &str) -> Result<Vec<String>, ServeApiError> {
    let Some(value_start) = json_field_value_start(body, "prompt") else {
        return Err(serve_api_error(
            400,
            "invalid_request_error",
            "missing_prompt",
            "Completion requests require a non-empty prompt string or string array.",
        ));
    };
    let prompts = json_string_or_string_array_at(body, value_start).ok_or_else(|| {
        serve_api_error(
            400,
            "invalid_request_error",
            "invalid_prompt",
            "prompt must be a non-empty string or array of non-empty strings.",
        )
    })?;
    if prompts.iter().any(|prompt| prompt.trim().is_empty()) {
        return Err(serve_api_error(
            400,
            "invalid_request_error",
            "invalid_prompt",
            "prompt must be a non-empty string or array of non-empty strings.",
        ));
    }
    Ok(prompts)
}

fn required_chat_messages(body: &str) -> Result<Vec<ApiChatMessage>, ServeApiError> {
    let Some(messages_start) = json_field_value_start(body, "messages") else {
        return Err(serve_api_error(
            400,
            "invalid_request_error",
            "missing_messages",
            "Chat completion requests require a non-empty messages array.",
        ));
    };
    let messages = parse_json_chat_messages_at(body, messages_start).ok_or_else(|| {
        serve_api_error(
            400,
            "invalid_request_error",
            "invalid_messages",
            "messages must contain role/content objects with non-empty string content fields.",
        )
    })?;
    if messages.iter().any(|message| {
        !matches!(message.role.as_str(), "system" | "user" | "assistant")
            || message.content.trim().is_empty()
    }) {
        return Err(serve_api_error(
            400,
            "invalid_request_error",
            "invalid_messages",
            "messages must use system, user, or assistant roles and non-empty string content.",
        ));
    }
    Ok(messages)
}

fn validate_api_input_token_cap(
    input_tokens: usize,
    parsed: &ServeArgs,
) -> Result<(), ServeApiError> {
    if input_tokens > parsed.max_input_tokens {
        return Err(serve_api_error(
            400,
            "invalid_request_error",
            "input_tokens_exceeded",
            "Request input exceeds the configured max input token cap.",
        ));
    }
    Ok(())
}

fn validate_api_output_cap(
    requested_output_tokens: usize,
    parsed: &ServeArgs,
) -> Result<(), ServeApiError> {
    if requested_output_tokens == 0 || requested_output_tokens > parsed.max_output_tokens {
        return Err(serve_api_error(
            400,
            "invalid_request_error",
            "output_tokens_exceeded",
            "Requested max_tokens exceeds the configured max output token cap.",
        ));
    }
    Ok(())
}

fn estimate_request_tokens(text: &str) -> usize {
    let words = text.split_whitespace().count();
    words.max(text.chars().count().div_ceil(4)).max(1)
}

fn serve_api_error(
    status: u16,
    error_type: &'static str,
    code: &'static str,
    message: &'static str,
) -> ServeApiError {
    ServeApiError {
        status,
        error_type,
        code,
        message,
    }
}

fn invalid_json_error() -> ServeApiError {
    serve_api_error(
        400,
        "invalid_request_error",
        "invalid_json",
        "Request body must be valid JSON for the supported fields.",
    )
}

fn looks_like_json_object(body: &str) -> bool {
    let body = body.trim();
    body.starts_with('{') && body.ends_with('}')
}

fn json_string_field(body: &str, field: &str) -> Result<Option<String>, ()> {
    let Some(value_start) = json_field_value_start(body, field) else {
        return Ok(None);
    };
    parse_json_string_at(body, value_start)
        .map(|(value, _)| Some(value))
        .ok_or(())
}

fn parse_json_chat_messages_at(body: &str, value_start: usize) -> Option<Vec<ApiChatMessage>> {
    let mut idx = skip_json_whitespace(body, value_start);
    if body[idx..].chars().next()? != '[' {
        return None;
    }
    idx += 1;

    let mut messages = Vec::new();
    loop {
        idx = skip_json_whitespace(body, idx);
        match body[idx..].chars().next()? {
            ']' => return (!messages.is_empty()).then_some(messages),
            '{' => {
                let end_idx = json_container_end(body, idx, '{', '}')?;
                let object = &body[idx..end_idx];
                let role = json_string_field(object, "role").ok().flatten()?;
                let content = json_string_field(object, "content").ok().flatten()?;
                messages.push(ApiChatMessage { role, content });
                idx = skip_json_whitespace(body, end_idx);
                match body[idx..].chars().next()? {
                    ',' => idx += 1,
                    ']' => return Some(messages),
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
}

fn skip_json_whitespace(input: &str, start_idx: usize) -> usize {
    start_idx
        + input[start_idx..]
            .chars()
            .take_while(|ch| ch.is_whitespace())
            .map(char::len_utf8)
            .sum::<usize>()
}

fn json_container_end(input: &str, start_idx: usize, open: char, close: char) -> Option<usize> {
    let mut chars = input[start_idx..].char_indices();
    if chars.next()?.1 != open {
        return None;
    }
    let mut depth = 1usize;
    let mut in_string = false;
    let mut escaping = false;

    for (relative_idx, ch) in chars {
        if in_string {
            if escaping {
                escaping = false;
            } else if ch == '\\' {
                escaping = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            ch if ch == open => depth += 1,
            ch if ch == close => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(start_idx + relative_idx + ch.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

fn json_usize_field(body: &str, field: &str) -> Result<Option<usize>, ()> {
    let Some(value_start) = json_field_value_start(body, field) else {
        return Ok(None);
    };
    let value = body[value_start..].trim_start();
    let end = value
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(value.len());
    if end == 0 {
        return Err(());
    }
    let parsed = value[..end].parse::<usize>().map_err(|_| ())?;
    (parsed > 0).then_some(Some(parsed)).ok_or(())
}

fn json_f32_field(body: &str, field: &str) -> Result<Option<f32>, ()> {
    let Some(value_start) = json_field_value_start(body, field) else {
        return Ok(None);
    };
    let value = body[value_start..].trim_start();
    let end = value
        .find(|ch: char| !(ch.is_ascii_digit() || matches!(ch, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(value.len());
    if end == 0 {
        return Err(());
    }
    let parsed = value[..end].parse::<f32>().map_err(|_| ())?;
    (parsed.is_finite() && parsed >= 0.0)
        .then_some(Some(parsed))
        .ok_or(())
}

fn json_string_or_string_array_at(body: &str, value_start: usize) -> Option<Vec<String>> {
    let value = body[value_start..].trim_start();
    if value.starts_with('"') {
        return parse_json_string_at(value, 0).map(|(prompt, _)| vec![prompt]);
    }
    if !value.starts_with('[') {
        return None;
    }
    let mut prompts = Vec::new();
    let mut idx = 1;
    loop {
        idx += value[idx..]
            .chars()
            .take_while(|ch| ch.is_whitespace())
            .count();
        match value[idx..].chars().next()? {
            ']' => return (!prompts.is_empty()).then_some(prompts),
            '"' => {
                let (prompt, end_idx) = parse_json_string_at(value, idx)?;
                prompts.push(prompt);
                idx = end_idx;
                idx += value[idx..]
                    .chars()
                    .take_while(|ch| ch.is_whitespace())
                    .count();
                match value[idx..].chars().next()? {
                    ',' => idx += 1,
                    ']' => return Some(prompts),
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
}

fn json_field_value_start(body: &str, field: &str) -> Option<usize> {
    let marker = format!("\"{field}\"");
    let marker_idx = body.find(&marker)?;
    json_field_value_start_from(body, marker_idx, &marker)
}

fn json_field_value_start_from(body: &str, marker_idx: usize, marker: &str) -> Option<usize> {
    let mut idx = marker_idx + marker.len();
    idx += body[idx..]
        .chars()
        .take_while(|ch| ch.is_whitespace())
        .count();
    if body[idx..].chars().next()? != ':' {
        return None;
    }
    idx += 1;
    idx += body[idx..]
        .chars()
        .take_while(|ch| ch.is_whitespace())
        .count();
    Some(idx)
}

fn parse_json_string_at(input: &str, start_idx: usize) -> Option<(String, usize)> {
    let mut chars = input[start_idx..].char_indices();
    if chars.next()?.1 != '"' {
        return None;
    }
    let mut out = String::new();
    while let Some((relative_idx, ch)) = chars.next() {
        match ch {
            '"' => return Some((out, start_idx + relative_idx + ch.len_utf8())),
            '\\' => {
                let (_, escaped) = chars.next()?;
                match escaped {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'b' => out.push('\u{0008}'),
                    'f' => out.push('\u{000c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => return None,
                    _ => return None,
                }
            }
            ch if ch.is_control() => return None,
            ch => out.push(ch),
        }
    }
    None
}

fn serve_models_json(parsed: &ServeArgs) -> String {
    let dir = Path::new(&parsed.model_dir);
    let entries = scan_model_dir(dir, false).unwrap_or_default();
    let data = entries
        .iter()
        .map(|entry| {
            let id = entry
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown.gguf");
            format!(
                "{{\"id\":{},\"object\":\"model\",\"path\":{},\"bytes\":{},\"target\":{},\"quantization\":{}}}",
                json_string(id),
                json_string(&entry.path.display().to_string()),
                entry.bytes,
                json_optional_string(entry.target),
                json_optional_string(entry.quantization)
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "{{\"object\":\"list\",\"model_dir\":{},\"model_dir_exists\":{},\"data\":[{}]}}",
        json_string(&parsed.model_dir),
        dir.is_dir(),
        data
    )
}

fn serve_completion_response_json(
    request: &ValidatedApiCompletionRequest,
    parsed: &ServeArgs,
) -> Result<String, ServeApiError> {
    let model_path = resolve_api_model_path(&request.model, parsed)?;
    let choices = generate_api_completion_choices_with_cap(
        &model_path,
        &request.prompts,
        request.temperature,
        request.requested_output_tokens,
        parsed,
    )?;
    Ok(api_completion_response_json(&request.model, &choices))
}

fn serve_chat_completion_response_json(
    request: &ValidatedApiChatCompletionRequest,
    parsed: &ServeArgs,
) -> Result<String, ServeApiError> {
    let model_path = resolve_api_model_path(&request.model, parsed)?;
    let choices = generate_api_chat_completion_choices_with_cap(
        &model_path,
        &request.messages,
        request.temperature,
        request.requested_output_tokens,
        parsed,
    )?;
    Ok(api_chat_completion_response_json(&request.model, &choices))
}

fn api_generation_failed_error(message: &'static str) -> ServeApiError {
    serve_api_error(500, "server_error", "generation_failed", message)
}

fn resolve_api_model_path(model_id: &str, parsed: &ServeArgs) -> Result<PathBuf, ServeApiError> {
    let explicit = Path::new(model_id);
    if looks_like_model_path_argument(model_id) {
        return explicit
            .is_file()
            .then(|| explicit.to_path_buf())
            .ok_or_else(|| {
                serve_api_error(
                    404,
                    "invalid_request_error",
                    "model_not_found",
                    "Requested model path does not exist or is not a file.",
                )
            });
    }

    if is_llama32_1b_alias(model_id) {
        for filename in [LLAMA32_1B_Q4_MODEL, LLAMA32_1B_Q8_MODEL] {
            let path = Path::new(&parsed.model_dir).join(filename);
            if path.is_file() {
                return Ok(path);
            }
        }
    } else if is_llama32_3b_alias(model_id) {
        let path = Path::new(&parsed.model_dir).join(LLAMA32_3B_Q4_MODEL);
        if path.is_file() {
            return Ok(path);
        }
    }

    let entries = scan_model_dir(Path::new(&parsed.model_dir), false).unwrap_or_default();
    entries
        .into_iter()
        .find(|entry| api_model_entry_matches(&entry.path, model_id))
        .map(|entry| entry.path)
        .ok_or_else(|| {
            serve_api_error(
                404,
                "invalid_request_error",
                "model_not_found",
                "Requested model was not found in the configured model directory.",
            )
        })
}

fn api_model_entry_matches(path: &Path, model_id: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|filename| filename == model_id)
        || path
            .file_stem()
            .and_then(|name| name.to_str())
            .is_some_and(|stem| stem == model_id)
}

fn generate_api_completion_choices_with_cap(
    model_path: &Path,
    prompts: &[String],
    temp: f32,
    max_tokens: usize,
    parsed: &ServeArgs,
) -> Result<Vec<ApiCompletionChoice>, ServeApiError> {
    prefill_batch_size_from_env().map_err(|_| {
        api_generation_failed_error(
            "Model generation failed. Run nanocamelid inspect or nanocamelid generate for the same model and prompt.",
        )
    })?;

    let generation_failed = || {
        api_generation_failed_error(
            "Model generation failed. Run nanocamelid inspect or nanocamelid generate for the same model and prompt.",
        )
    };
    let gguf = gguf::read_file(model_path).map_err(|_| generation_failed())?;
    let mut config = model::LlamaModelConfig::from_gguf(&gguf).map_err(|_| generation_failed())?;
    apply_context_limit(&mut config).map_err(|_| generation_failed())?;
    let tokenizer = tokenizer::Tokenizer::from_gguf(&gguf).map_err(|_| generation_failed())?;
    let prompt_tokens = prompts
        .iter()
        .map(|prompt| tokenizer.encode(prompt, true, true))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| generation_failed())?;
    let input_tokens = prompt_tokens.iter().map(Vec::len).sum::<usize>();
    validate_api_input_token_cap(input_tokens, parsed)?;
    let weights =
        model::LlamaWeights::load(model_path, &config, &gguf).map_err(|_| generation_failed())?;
    let runtime_options = runtime_options_from_gguf(&gguf, q8::Q8DotKernelSelector::from_env());

    prompt_tokens
        .into_iter()
        .enumerate()
        .map(|(index, prompt_tokens)| {
            let runtime = ApiGenerationRuntime {
                config: &config,
                weights: &weights,
                tokenizer: &tokenizer,
                runtime_options,
            };
            generate_api_completion_choice_from_tokens(
                index,
                prompt_tokens,
                temp,
                max_tokens,
                runtime,
            )
            .map_err(|_| generation_failed())
        })
        .collect()
}

fn generate_api_chat_completion_choices_with_cap(
    model_path: &Path,
    messages: &[ApiChatMessage],
    temp: f32,
    max_tokens: usize,
    parsed: &ServeArgs,
) -> Result<Vec<ApiCompletionChoice>, ServeApiError> {
    prefill_batch_size_from_env().map_err(|_| {
        api_generation_failed_error(
            "Chat completion generation failed. Run nanocamelid inspect or nanocamelid chat for the same model and messages.",
        )
    })?;

    let generation_failed = || {
        api_generation_failed_error(
            "Chat completion generation failed. Run nanocamelid inspect or nanocamelid chat for the same model and messages.",
        )
    };
    let gguf = gguf::read_file(model_path).map_err(|_| generation_failed())?;
    let mut config = model::LlamaModelConfig::from_gguf(&gguf).map_err(|_| generation_failed())?;
    apply_context_limit(&mut config).map_err(|_| generation_failed())?;
    let tokenizer = tokenizer::Tokenizer::from_gguf(&gguf).map_err(|_| generation_failed())?;
    let rendered_messages = messages
        .iter()
        .map(|message| tokenizer::ChatMessage {
            role: message.role.as_str(),
            content: message.content.as_str(),
        })
        .collect::<Vec<_>>();
    let rendered = tokenizer.render_chat_prompt(&rendered_messages);
    let prompt_tokens = tokenizer
        .encode(&rendered.text, rendered.add_special, rendered.parse_special)
        .map_err(|_| generation_failed())?;
    validate_api_input_token_cap(prompt_tokens.len(), parsed)?;
    let weights =
        model::LlamaWeights::load(model_path, &config, &gguf).map_err(|_| generation_failed())?;
    let runtime_options = runtime_options_from_gguf(&gguf, q8::Q8DotKernelSelector::from_env());
    let runtime = ApiGenerationRuntime {
        config: &config,
        weights: &weights,
        tokenizer: &tokenizer,
        runtime_options,
    };
    generate_api_completion_choice_from_tokens(0, prompt_tokens, temp, max_tokens, runtime)
        .map(|choice| vec![choice])
        .map_err(|_| generation_failed())
}

fn generate_api_completion_choice_from_tokens(
    index: usize,
    prompt_tokens: Vec<u32>,
    temp: f32,
    max_tokens: usize,
    runtime: ApiGenerationRuntime<'_>,
) -> Result<ApiCompletionChoice, String> {
    if prompt_tokens.is_empty() {
        return Err("prompt tokenized to an empty sequence".to_owned());
    }
    validate_generation_budget(
        prompt_tokens.len(),
        max_tokens,
        runtime.config.context_length,
    )?;

    let mut cache = inference::LlamaKvCache::new(
        runtime.config.block_count,
        runtime.config.context_length,
        runtime.config.kv_width,
    );
    let mut ws = inference::LlamaWorkspace::new(runtime.config);
    let mut batch_ws = inference::LlamaBatchWorkspace::new(runtime.config, prefill_batch_size());
    let mut pos = 0usize;

    if let Some((&last_token, prefix_tokens)) = prompt_tokens.split_last() {
        let mut context_tokens = Vec::with_capacity(prompt_tokens.len());
        prefill_tokens(
            prefix_tokens,
            runtime.config,
            runtime.weights,
            PrefillTokenState {
                cache: &mut cache,
                ws: &mut ws,
                batch_ws: Some(&mut batch_ws),
                context_tokens: &mut context_tokens,
                pos: &mut pos,
            },
            runtime.runtime_options,
        );
        inference::forward_pass(
            last_token as usize,
            pos,
            runtime.config,
            runtime.weights,
            &mut cache,
            &mut ws,
            runtime.runtime_options,
        );
        pos += 1;
    }

    let mut generated_tokens = Vec::new();
    let mut finish_reason = "length";
    loop {
        let next_token = inference::sample_logits(&ws.logits, temp);

        if is_generation_stop_token(&runtime.tokenizer.special, next_token as u32) {
            finish_reason = "stop";
            break;
        }
        if pos >= runtime.config.context_length || generated_tokens.len() >= max_tokens {
            break;
        }

        generated_tokens.push(next_token as u32);
        if generated_tokens.len() >= max_tokens {
            break;
        }

        inference::forward_pass(
            next_token,
            pos,
            runtime.config,
            runtime.weights,
            &mut cache,
            &mut ws,
            runtime.runtime_options,
        );
        pos += 1;
    }

    Ok(ApiCompletionChoice {
        index,
        text: runtime.tokenizer.decode(&generated_tokens, true)?,
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated_tokens.len(),
        finish_reason,
    })
}

fn api_completion_response_json(model: &str, choices: &[ApiCompletionChoice]) -> String {
    let choices_json = choices
        .iter()
        .map(|choice| {
            format!(
                "{{\"index\":{},\"text\":{},\"finish_reason\":{}}}",
                choice.index,
                json_string(&choice.text),
                json_string(choice.finish_reason)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let prompt_tokens = choices
        .iter()
        .map(|choice| choice.prompt_tokens)
        .sum::<usize>();
    let completion_tokens = choices
        .iter()
        .map(|choice| choice.generated_tokens)
        .sum::<usize>();
    format!(
        "{{\"id\":\"cmpl-nanocamelid\",\"object\":\"text_completion\",\"model\":{},\"choices\":[{}],\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}}}",
        json_string(model),
        choices_json,
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens
    )
}

fn api_chat_completion_response_json(model: &str, choices: &[ApiCompletionChoice]) -> String {
    let choices_json = choices
        .iter()
        .map(|choice| {
            format!(
                "{{\"index\":{},\"message\":{{\"role\":\"assistant\",\"content\":{}}},\"finish_reason\":{}}}",
                choice.index,
                json_string(&choice.text),
                json_string(choice.finish_reason)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let prompt_tokens = choices
        .iter()
        .map(|choice| choice.prompt_tokens)
        .sum::<usize>();
    let completion_tokens = choices
        .iter()
        .map(|choice| choice.generated_tokens)
        .sum::<usize>();
    format!(
        "{{\"id\":\"chatcmpl-nanocamelid\",\"object\":\"chat.completion\",\"model\":{},\"choices\":[{}],\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}}}",
        json_string(model),
        choices_json,
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens
    )
}

fn serve_metrics_text(request_count: usize, uptime_seconds: f64, parsed: &ServeArgs) -> String {
    format!(
        "nanocamelid_requests_total {}\nnanocamelid_uptime_seconds {:.3}\nnanocamelid_max_request_bytes {}\nnanocamelid_max_input_tokens {}\nnanocamelid_max_output_tokens {}\n",
        request_count,
        uptime_seconds,
        parsed.max_request_bytes,
        parsed.max_input_tokens,
        parsed.max_output_tokens
    )
}

fn write_json_error(
    stream: &mut TcpStream,
    status: u16,
    error_type: &str,
    code: &str,
    message: &str,
) -> io::Result<()> {
    write_json_response(
        stream,
        status,
        &format!(
            "{{\"error\":{{\"message\":{},\"type\":{},\"code\":{}}}}}",
            json_string(message),
            json_string(error_type),
            json_string(code)
        ),
    )
}

fn write_json_response(stream: &mut TcpStream, status: u16, body: &str) -> io::Result<()> {
    write_text_response(stream, status, body, "application/json; charset=utf-8")
}

fn write_text_response(
    stream: &mut TcpStream,
    status: u16,
    body: &str,
    content_type: &str,
) -> io::Result<()> {
    let status_text = http_status_text(status);
    write!(
        stream,
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn http_status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        413 => "Payload Too Large",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        _ => "Internal Server Error",
    }
}

fn run_models(parsed: ModelsArgs) -> ExitCode {
    match parsed.action {
        ModelsAction::List { dir } => run_models_list(&dir, false, parsed.json, parsed.dry_run),
        ModelsAction::Scan { dir } => run_models_list(&dir, true, parsed.json, parsed.dry_run),
        ModelsAction::Inspect(inspect) => run_inspect(inspect),
    }
}

fn run_models_list(dir: &str, recursive: bool, json: bool, dry_run: bool) -> ExitCode {
    let dir = Path::new(dir);
    let command = if recursive {
        "models scan"
    } else {
        "models list"
    };
    println!("NanoCamelid {command}");
    println!("model_dir: {}", dir.display());
    println!("recursive: {recursive}");

    if dry_run {
        println!("dry_run: true");
        println!(
            "{}_command: {}",
            if recursive { "scan" } else { "list" },
            shell_command(&[
                "nanocamelid",
                "models",
                if recursive { "scan" } else { "list" },
                "--dir",
                &dir.display().to_string(),
            ])
        );
        if json {
            println!(
                "json_on_success: {{\"command\":\"{}\",\"status\":\"ok\",\"model_dir\":{},\"recursive\":{},\"count\":null}}",
                command,
                json_string(&dir.display().to_string()),
                recursive
            );
        }
        return ExitCode::SUCCESS;
    }

    let entries = match scan_model_dir(dir, recursive) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            eprintln!("{}", model_dir_not_found_message(dir));
            return ExitCode::from(2);
        }
        Err(err) => {
            eprintln!("models scan failed for {}: {err}", dir.display());
            return ExitCode::FAILURE;
        }
    };

    println!("count: {}", entries.len());
    if entries.is_empty() {
        println!("  no GGUF files found");
    }
    for entry in &entries {
        println!("  {}", entry.path.display());
        println!("    bytes: {}", entry.bytes);
        println!("    target: {}", entry.target.unwrap_or("unknown"));
        println!(
            "    quantization: {}",
            entry.quantization.unwrap_or("unknown")
        );
        println!(
            "    inspect_command: {}",
            shell_command(&[
                "nanocamelid",
                "models",
                "inspect",
                &entry.path.display().to_string(),
            ])
        );
        if json {
            println!(
                "json_model: {{\"path\":{},\"bytes\":{},\"target\":{},\"quantization\":{}}}",
                json_string(&entry.path.display().to_string()),
                entry.bytes,
                json_optional_string(entry.target),
                json_optional_string(entry.quantization)
            );
        }
    }
    if json {
        println!(
            "json: {{\"command\":\"{}\",\"status\":\"ok\",\"model_dir\":{},\"recursive\":{},\"count\":{}}}",
            command,
            json_string(&dir.display().to_string()),
            recursive,
            entries.len()
        );
    }
    ExitCode::SUCCESS
}

fn scan_model_dir(dir: &Path, recursive: bool) -> io::Result<Vec<ModelEntry>> {
    let mut entries = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(current_dir) = pending.pop() {
        for entry in fs::read_dir(&current_dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() && recursive {
                pending.push(path);
                continue;
            }
            if !file_type.is_file() || !looks_like_gguf_path(&path.to_string_lossy()) {
                continue;
            }
            let bytes = entry.metadata().map(|metadata| metadata.len()).unwrap_or(0);
            entries.push(ModelEntry {
                target: classify_model_target(&path),
                quantization: classify_model_quantization(&path),
                path,
                bytes,
            });
        }
    }
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(entries)
}

fn classify_model_target(path: &Path) -> Option<&'static str> {
    let filename = path.file_name()?.to_string_lossy().to_ascii_lowercase();
    if filename.contains("llama-3.2-1b") {
        Some("llama32-1b")
    } else if filename.contains("llama-3.2-3b") {
        Some("llama32-3b")
    } else if filename.contains("qwen3") {
        Some("qwen3")
    } else if filename.contains("qwen2.5") || filename.contains("qwen2") {
        Some("qwen2")
    } else if filename.contains("gemma-3") {
        Some("gemma3")
    } else if filename.contains("smollm3") {
        Some("smollm3")
    } else if filename.contains("smollm2") {
        Some("smollm2")
    } else if filename.contains("mistral") {
        Some("mistral")
    } else {
        None
    }
}

fn classify_model_quantization(path: &Path) -> Option<&'static str> {
    let filename = path.file_name()?.to_string_lossy().to_ascii_lowercase();
    [
        ("iq4_nl", "IQ4_NL"),
        ("q8_0", "Q8_0"),
        ("q8_k", "Q8_K"),
        ("q6_k", "Q6_K"),
        ("q5_k", "Q5_K"),
        ("q5_1", "Q5_1"),
        ("q5_0", "Q5_0"),
        ("q4_k", "Q4_K"),
        ("q4_1", "Q4_1"),
        ("q4_0", "Q4_0"),
        ("q3_k", "Q3_K"),
        ("q2_k", "Q2_K"),
    ]
    .iter()
    .find_map(|(needle, label)| filename.contains(needle).then_some(*label))
}

fn print_probe() {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let device_tree_model = fs::read_to_string("/proc/device-tree/model").unwrap_or_default();
    let model = device_model(&device_tree_model)
        .or_else(|| cpu_model(&cpuinfo))
        .unwrap_or("unknown");
    let features = cpu_features(&cpuinfo);
    let core_count = core_affinity::get_core_ids()
        .map(|cores| cores.len())
        .unwrap_or(0);
    let max_freq_khz = read_trimmed("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    let governor = read_trimmed("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    let isolated_cpus = read_trimmed("/sys/devices/system/cpu/isolated");
    let worker_cores = worker_core_indices_from_env()
        .or_else(|| isolated_cpus.as_deref().and_then(parse_cpu_list));

    println!("NanoCamelid host probe");
    println!("arch: {}", env::consts::ARCH);
    println!("os: {}", env::consts::OS);
    println!("cpu_model: {model}");
    println!("logical_cores: {core_count}");
    println!("cpu_features: {}", features.unwrap_or("unknown"));
    println!(
        "cpuinfo_max_freq_khz: {}",
        max_freq_khz.as_deref().unwrap_or("unknown")
    );
    println!(
        "scaling_governor: {}",
        governor.as_deref().unwrap_or("unknown")
    );
    if let Some(recommendation) = cpu_governor_recommendation(governor.as_deref()) {
        println!("governor_recommendation: {recommendation}");
    }
    println!(
        "isolated_cpus: {}",
        isolated_cpus
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or("none")
    );
    println!(
        "rayon_worker_cores: {}",
        worker_cores
            .as_deref()
            .map(format_cpu_list)
            .unwrap_or_else(|| "default".to_owned())
    );
    println!("runtime_neon: {}", runtime_neon());
    println!("runtime_dotprod: {}", runtime_dotprod());
}

fn read_trimmed(path: &str) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn cpu_governor_recommendation(governor: Option<&str>) -> Option<&'static str> {
    matches!(governor, Some("ondemand")).then_some(PERFORMANCE_GOVERNOR_COMMAND)
}

fn format_cpu_list(cpus: &[usize]) -> String {
    cpus.iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn bench_q8_dot(iterations: usize, runs: usize) -> ExitCode {
    if iterations == 0 {
        eprintln!("iterations must be greater than zero");
        return ExitCode::from(2);
    }
    if runs == 0 {
        eprintln!("runs must be greater than zero");
        return ExitCode::from(2);
    }

    let report = q8::bench_dot_runs(iterations, runs);

    println!("NanoCamelid Q8 dot benchmark");
    println!("iterations: {}", report.iterations);
    println!("runs: {}", report.runs);
    println!("blocks_per_iteration: {}", report.blocks_per_iteration);
    println!("elements_per_iteration: {}", report.elements_per_iteration);
    println!(
        "kernel_selector_requested: {}",
        report
            .kernel_selector
            .requested
            .map(q8::Q8DotKernel::name)
            .unwrap_or("default")
    );
    println!(
        "kernel_selector_selected: {}",
        report.kernel_selector.selected.name()
    );
    if let Some(reason) = report.kernel_selector.fallback_reason {
        println!("kernel_selector_fallback: {reason}");
    }
    println!("selected_checksum: {}", report.selected.checksum);
    println!(
        "selected_total_ms: {:.3}",
        report.selected.total_elapsed().as_secs_f64() * 1000.0
    );
    println!("scalar_checksum: {}", report.scalar.checksum);
    println!(
        "scalar_total_ms: {:.3}",
        report.scalar.total_elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "scalar_min_ns_per_block: {:.2}",
        report.scalar_min_ns_per_block()
    );
    println!(
        "scalar_median_ns_per_block: {:.2}",
        report.scalar_median_ns_per_block()
    );
    if report.selected.checksum != report.scalar.checksum {
        eprintln!("selected kernel checksum mismatch");
        return ExitCode::FAILURE;
    }

    match &report.neon {
        Some(neon) => {
            println!("neon_available: true");
            println!("dotprod_feature_detected: {}", q8::dotprod_available());
            println!(
                "sdot_candidate_requested: {}",
                q8::sdot_candidate_requested()
            );
            println!("sdot_candidate_enabled: {}", q8::sdot_candidate_enabled());
            println!("neon_checksum: {}", neon.checksum);
            println!(
                "neon_total_ms: {:.3}",
                neon.total_elapsed().as_secs_f64() * 1000.0
            );
            println!(
                "neon_min_ns_per_block: {:.2}",
                report.neon_min_ns_per_block().unwrap_or_default()
            );
            println!(
                "neon_median_ns_per_block: {:.2}",
                report.neon_median_ns_per_block().unwrap_or_default()
            );
            println!(
                "neon_min_speedup: {:.2}x",
                report.neon_min_speedup().unwrap_or_default()
            );
            println!(
                "neon_median_speedup: {:.2}x",
                report.neon_median_speedup().unwrap_or_default()
            );
            if neon.checksum != report.scalar.checksum {
                eprintln!("neon checksum mismatch");
                return ExitCode::FAILURE;
            }
        }
        None => {
            println!("neon_available: false");
            println!("dotprod_feature_detected: {}", q8::dotprod_available());
            println!(
                "sdot_candidate_requested: {}",
                q8::sdot_candidate_requested()
            );
            println!("sdot_candidate_enabled: {}", q8::sdot_candidate_enabled());
        }
    }

    if let Some(sdot) = &report.sdot {
        println!("sdot_checksum: {}", sdot.checksum);
        println!(
            "sdot_total_ms: {:.3}",
            sdot.total_elapsed().as_secs_f64() * 1000.0
        );
        println!(
            "sdot_min_ns_per_block: {:.2}",
            report.sdot_min_ns_per_block().unwrap_or_default()
        );
        println!(
            "sdot_median_ns_per_block: {:.2}",
            report.sdot_median_ns_per_block().unwrap_or_default()
        );
        println!(
            "sdot_min_speedup: {:.2}x",
            report.sdot_min_speedup().unwrap_or_default()
        );
        println!(
            "sdot_median_speedup: {:.2}x",
            report.sdot_median_speedup().unwrap_or_default()
        );
        if report.neon.is_some() {
            println!(
                "sdot_vs_neon_min_speedup: {:.2}x",
                report.sdot_vs_neon_min_speedup().unwrap_or_default()
            );
            println!(
                "sdot_vs_neon_median_speedup: {:.2}x",
                report.sdot_vs_neon_median_speedup().unwrap_or_default()
            );
        }
        if sdot.checksum != report.scalar.checksum {
            eprintln!("sdot checksum mismatch");
            return ExitCode::FAILURE;
        }
    }

    println!("json: {}", q8_dot_json(&report));

    ExitCode::SUCCESS
}

fn bench_q4_layout(rows: usize, cols: usize, runs: usize) -> ExitCode {
    let report = match q8::bench_q4_1x4_layout_runs(rows, cols, runs) {
        Ok(report) => report,
        Err(err) => {
            eprintln!("Q4 layout benchmark failed: {err}");
            return ExitCode::from(2);
        }
    };

    println!("NanoCamelid Q4 1x4 layout benchmark");
    println!("rows: {}", report.rows);
    println!("cols: {}", report.cols);
    println!("runs: {}", report.runs);
    println!("blocks_per_row: {}", report.blocks_per_row);
    println!(
        "dotprod_feature_detected: {}",
        report.dotprod_feature_detected
    );
    println!("row_major_checksum: {}", report.row_major.checksum);
    println!("row_major_total_ms: {:.3}", report.row_major_total_ms());
    println!("swizzled_checksum: {}", report.swizzled_1x4.checksum);
    println!("swizzled_total_ms: {:.3}", report.swizzled_total_ms());
    println!("swizzled_speedup: {:.3}x", report.swizzled_speedup());
    println!(
        "aligned_swizzled_checksum: {}",
        report.aligned_swizzled_1x4.checksum
    );
    println!(
        "aligned_swizzled_total_ms: {:.3}",
        report.aligned_swizzled_total_ms()
    );
    println!(
        "aligned_vs_swizzled_speedup: {:.3}x",
        report.aligned_vs_swizzled_speedup()
    );
    println!(
        "json: {{\"benchmark\":\"q4-layout\",\"rows\":{},\"cols\":{},\"runs\":{},\"blocks_per_row\":{},\"dotprod_feature_detected\":{},\"row_major\":{{\"checksum\":{},\"run_ms\":{}}},\"swizzled_1x4\":{{\"checksum\":{},\"run_ms\":{}}},\"aligned_swizzled_1x4\":{{\"checksum\":{},\"run_ms\":{}}},\"swizzled_speedup\":{:.6},\"aligned_vs_swizzled_speedup\":{:.6}}}",
        report.rows,
        report.cols,
        report.runs,
        report.blocks_per_row,
        report.dotprod_feature_detected,
        report.row_major.checksum,
        duration_ms_json(&report.row_major.elapsed_runs),
        report.swizzled_1x4.checksum,
        duration_ms_json(&report.swizzled_1x4.elapsed_runs),
        report.aligned_swizzled_1x4.checksum,
        duration_ms_json(&report.aligned_swizzled_1x4.elapsed_runs),
        report.swizzled_speedup(),
        report.aligned_vs_swizzled_speedup()
    );

    if report.row_major.checksum != report.swizzled_1x4.checksum {
        eprintln!("swizzled layout checksum mismatch");
        return ExitCode::FAILURE;
    }
    if report.row_major.checksum != report.aligned_swizzled_1x4.checksum {
        eprintln!("aligned swizzled layout checksum mismatch");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn bench_q4_prefill(prompt_len: usize, batch_size: usize, runs: usize) -> ExitCode {
    if prompt_len == 0 {
        eprintln!("prompt_len must be greater than zero");
        return ExitCode::from(2);
    }
    if batch_size == 0 {
        eprintln!("batch_size must be greater than zero");
        return ExitCode::from(2);
    }
    if runs == 0 {
        eprintln!("runs must be greater than zero");
        return ExitCode::from(2);
    }

    let rows = Q4_PREFILL_ROWS;
    let cols = Q4_PREFILL_COLS;
    let blocks_per_row = cols / q8::Q8_BLOCK_SIZE;
    let q8_blocks_per_token = cols / q8::Q8_BLOCK_SIZE;
    let selector = q8::Q8DotKernelSelector::from_env();
    let row_major = synthetic_q4_blocks(rows, blocks_per_row);
    let swizzled_1x4 = swizzle_q4_0_1x4(&row_major, rows, blocks_per_row);
    let matrix = model::QuantizedMatrix::Q4_0Swizzled1x4(model::Q4_0Swizzled1x4Matrix {
        swizzled_1x4,
        page_aligned_1x4: None,
        rows,
        cols,
    });
    let mut x_i8 = vec![0_i8; batch_size * cols];
    let mut x_scales = vec![0.0_f32; batch_size * q8_blocks_per_token];
    let mut out = vec![0.0_f32; batch_size * rows];

    let elapsed_runs = (0..runs)
        .map(|run_idx| {
            fill_synthetic_q8_activations(
                &mut x_i8,
                &mut x_scales,
                batch_size,
                cols,
                run_idx as u32,
            );
            out.fill(0.0);

            let started = std::time::Instant::now();
            for start in (0..prompt_len).step_by(batch_size) {
                let current_batch = (prompt_len - start).min(batch_size);
                inference::matmul_quantized_batch(
                    &mut out[..current_batch * rows],
                    &x_i8[..current_batch * cols],
                    &x_scales[..current_batch * q8_blocks_per_token],
                    &matrix,
                    inference::BatchMatmulShape {
                        batch_size: current_batch,
                        rows,
                        cols,
                    },
                    selector,
                );
            }
            let elapsed = started.elapsed();
            black_box(q4_prefill_checksum(&out));
            elapsed
        })
        .collect::<Vec<_>>();

    let median = median_duration(elapsed_runs.clone());
    let median_ms = median.as_secs_f64() * 1000.0;
    let median_ms_per_token = median_ms / prompt_len as f64;
    let median_prompt_tokens_per_sec = if median_ms > 0.0 {
        prompt_len as f64 / (median_ms / 1000.0)
    } else {
        0.0
    };
    let min_ms = elapsed_runs
        .iter()
        .min()
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .unwrap_or_default();
    let max_prompt_tokens_per_sec = if min_ms > 0.0 {
        prompt_len as f64 / (min_ms / 1000.0)
    } else {
        0.0
    };

    println!("NanoCamelid Q4 prefill benchmark");
    println!("prompt_len: {prompt_len}");
    println!("batch_size: {batch_size}");
    println!("runs: {runs}");
    println!("rows: {rows}");
    println!("cols: {cols}");
    println!("kernel_selector_selected: {}", selector.selected.name());
    if let Some(reason) = selector.fallback_reason {
        println!("kernel_selector_fallback: {reason}");
    }
    println!("median_ms: {median_ms:.3}");
    println!("median_ms_per_token: {median_ms_per_token:.3}");
    println!("median_prompt_tokens_per_sec: {median_prompt_tokens_per_sec:.3}");
    println!("min_ms: {min_ms:.3}");
    println!("max_prompt_tokens_per_sec: {max_prompt_tokens_per_sec:.3}");
    println!(
        "json: {{\"benchmark\":\"q4-prefill\",\"prompt_len\":{},\"batch_size\":{},\"runs\":{},\"rows\":{},\"cols\":{},\"kernel\":\"{}\",\"run_ms\":{},\"median_ms\":{:.6},\"median_ms_per_token\":{:.6},\"median_prompt_tokens_per_sec\":{:.6},\"min_ms\":{:.6},\"max_prompt_tokens_per_sec\":{:.6}}}",
        prompt_len,
        batch_size,
        runs,
        rows,
        cols,
        selector.selected.name(),
        duration_ms_json(&elapsed_runs),
        median_ms,
        median_ms_per_token,
        median_prompt_tokens_per_sec,
        min_ms,
        max_prompt_tokens_per_sec
    );

    ExitCode::SUCCESS
}

fn print_bench_1b_dry_run(parsed: &Bench1BArgs) -> ExitCode {
    if let Err(err) = validate_context_limit_env() {
        eprintln!("{err}");
        return ExitCode::from(2);
    }

    let model_path = Path::new(&parsed.model_path);
    let batches = parsed
        .batches
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    println!("NanoCamelid Llama 3.2 1B prefill sweep dry run");
    println!("workspace: {}", parsed.workspace);
    println!("q4_model: {}", parsed.q4_model_path);
    println!("q4_exists: {}", Path::new(&parsed.q4_model_path).is_file());
    println!("q8_model: {}", parsed.q8_model_path);
    println!("q8_exists: {}", Path::new(&parsed.q8_model_path).is_file());
    println!("selected_source: {}", parsed.model_source);
    println!("model: {}", model_path.display());
    println!("model_exists: {}", model_path.is_file());
    println!(
        "quantization: {}",
        llama32_1b_quantization_for_path(model_path)
    );
    println!("prompt: {}", parsed.prompt);
    println!("max_tokens: {}", parsed.max_tokens);
    println!("temp: {}", parsed.temp);
    println!("context_limit: {}", context_limit_plan_value());
    println!("probe: enabled");
    println!("shape_audit: enabled");
    println!("smoke_gate: enabled");
    println!("batches: {batches}");
    println!("status_on_success: prefill_bench_1b_status: ok");
    println!(
        "json_on_success: {}",
        prefill_bench_1b_status_json(parsed, &context_limit_plan_value())
    );
    println!(
        "probe_command: {}",
        shell_command(&["nanocamelid", "probe"])
    );
    println!(
        "model_command: {}",
        shell_command(&[
            "nanocamelid",
            "model",
            "1b",
            &model_path.display().to_string()
        ])
    );
    println!(
        "inspect_command: {}",
        shell_command(&[
            "nanocamelid",
            "inspect",
            "1b",
            &model_path.display().to_string(),
        ])
    );
    println!(
        "smoke_command: {}",
        prefill_bench_1b_smoke_command(parsed, context_limit_env_value().as_deref())
    );
    for batch in &parsed.batches {
        println!(
            "batch_{batch}_command: {}",
            prefill_bench_1b_batch_command(parsed, *batch, context_limit_env_value().as_deref())
        );
    }
    ExitCode::SUCCESS
}

fn prefill_bench_1b_batch_command(
    parsed: &Bench1BArgs,
    batch: usize,
    context_limit: Option<&str>,
) -> String {
    let model = Path::new(&parsed.model_path).display().to_string();
    let max_tokens = parsed.max_tokens.to_string();
    let args = [
        "nanocamelid",
        "chat",
        &model,
        &parsed.prompt,
        &parsed.temp,
        &max_tokens,
    ];
    let env = prefill_bench_1b_batch_env(batch, context_limit);

    shell_command_with_owned_env(&args, &env)
}

fn prefill_bench_1b_smoke_command(parsed: &Bench1BArgs, context_limit: Option<&str>) -> String {
    let model = Path::new(&parsed.model_path).display().to_string();
    let max_tokens = parsed.max_tokens.to_string();
    let args = [
        "nanocamelid",
        "smoke",
        "1b",
        &model,
        "chat",
        &parsed.prompt,
        &max_tokens,
    ];
    let env = prefill_bench_1b_smoke_env(context_limit);

    shell_command_with_owned_env(&args, &env)
}

fn prefill_bench_1b_smoke_env(context_limit: Option<&str>) -> Vec<(&'static str, String)> {
    let mut env = Vec::with_capacity(3);
    if let Some(context_limit) = context_limit {
        env.push((CONTEXT_LIMIT_ENV, context_limit.to_owned()));
    }
    env.push(("NANOCAMELID_Q8_DOT_SDOT", "1".to_owned()));
    env.push(("NANOCAMELID_Q8_DOT_KERNEL", "sdot".to_owned()));
    env
}

fn prefill_bench_1b_batch_env(
    batch: usize,
    context_limit: Option<&str>,
) -> Vec<(&'static str, String)> {
    let mut env = Vec::with_capacity(4);
    if let Some(context_limit) = context_limit {
        env.push((CONTEXT_LIMIT_ENV, context_limit.to_owned()));
    }
    env.push(("NANOCAMELID_Q8_DOT_SDOT", "1".to_owned()));
    env.push(("NANOCAMELID_Q8_DOT_KERNEL", "sdot".to_owned()));
    env.push((PREFILL_BATCH_ENV, batch.to_string()));
    env
}

fn prefill_bench_1b_status_json(parsed: &Bench1BArgs, context_limit: &str) -> String {
    prefill_bench_1b_status_json_with_results(parsed, context_limit, None, None, None)
}

fn prefill_bench_1b_status_json_with_results(
    parsed: &Bench1BArgs,
    context_limit: &str,
    best_prefill: Option<(usize, f64)>,
    best_prefill_prompt_tokens_per_sec: Option<f64>,
    best_decode: Option<(usize, f64)>,
) -> String {
    let batches = parsed
        .batches
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let best_prefill_batch = best_prefill.map(|(batch, _)| batch);
    let best_prefill_sec = best_prefill.map(|(_, seconds)| seconds);
    let best_decode_batch = best_decode.map(|(batch, _)| batch);
    let best_tokens_per_sec = best_decode.map(|(_, tokens_per_sec)| tokens_per_sec);
    format!(
        "{{\"benchmark\":\"llama32-1b-prefill\",\"target\":\"llama32-1b\",\"status\":\"ok\",\"model\":{},\"selected_source\":{},\"quantization\":{},\"probe\":true,\"shape\":\"llama32_1b\",\"shape_ready\":true,\"context_limit\":{},\"prompt\":{},\"max_tokens\":{},\"temp\":{},\"batches\":[{}],\"best_prefill_batch\":{},\"best_prefill_sec\":{},\"best_prefill_prompt_tokens_per_sec\":{},\"best_decode_batch\":{},\"best_tokens_per_sec\":{}}}",
        json_string(&parsed.model_path),
        json_string(parsed.model_source),
        json_string(llama32_1b_quantization_for_path(Path::new(
            &parsed.model_path
        ))),
        json_string(context_limit),
        json_string(&parsed.prompt),
        parsed.max_tokens,
        parsed.temp,
        batches,
        json_optional_usize(best_prefill_batch),
        json_optional_f64(best_prefill_sec),
        json_optional_f64(best_prefill_prompt_tokens_per_sec),
        json_optional_usize(best_decode_batch),
        json_optional_f64(best_tokens_per_sec),
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PrefillBenchBatchMetrics {
    prompt_tokens: Option<usize>,
    prefill_sec: Option<f64>,
    generated_tokens: Option<usize>,
    generation_sec: Option<f64>,
    tokens_per_sec: Option<f64>,
}

fn run_bench_1b_prefill(parsed: Bench1BArgs) -> ExitCode {
    if let Err(err) = validate_context_limit_env() {
        eprintln!("Failed to apply context limit: {err}");
        return ExitCode::from(2);
    }
    if !Path::new(&parsed.model_path).is_file() {
        eprintln!(
            "{}",
            llama32_1b_model_not_found_message(Path::new(&parsed.model_path))
        );
        eprintln!(
            "Run `nanocamelid bench 1b --dry-run` to print the resolved preflight and batch commands without loading a model."
        );
        return ExitCode::from(2);
    }

    println!("NanoCamelid Llama 3.2 1B prefill sweep");
    println!("workspace: {}", parsed.workspace);
    println!("q4_model: {}", parsed.q4_model_path);
    println!("q4_exists: {}", Path::new(&parsed.q4_model_path).is_file());
    println!("q8_model: {}", parsed.q8_model_path);
    println!("q8_exists: {}", Path::new(&parsed.q8_model_path).is_file());
    println!("selected_source: {}", parsed.model_source);
    println!("model: {}", parsed.model_path);
    println!("prompt: {}", parsed.prompt);
    println!("max_tokens: {}", parsed.max_tokens);
    println!("temp: {}", parsed.temp);
    println!("context_limit: {}", context_limit_plan_value());
    println!("probe: enabled");
    println!("shape_audit: enabled");
    println!("smoke_gate: enabled");
    println!(
        "batches: {}",
        parsed
            .batches
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    );

    println!("==> Probing host fast-path support");
    print_probe();

    println!("==> Auditing 1B model shape: {}", parsed.model_path);
    let audit_code = run_model_1b_audit(Model1BAuditArgs {
        workspace: parsed.workspace.clone(),
        q4_model_path: parsed.q4_model_path.clone(),
        q8_model_path: parsed.q8_model_path.clone(),
        model_path: parsed.model_path.clone(),
        model_source: parsed.model_source,
        dry_run: false,
    });
    if audit_code != ExitCode::SUCCESS {
        return audit_code;
    }

    println!("==> Inspecting 1B model: {}", parsed.model_path);
    let inspect_code = inspect_gguf(
        Path::new(&parsed.model_path),
        true,
        Some(parsed.model_source),
    );
    if inspect_code != ExitCode::SUCCESS {
        return inspect_code;
    }

    println!("==> Running 1B chat smoke gate");
    match run_prefill_bench_1b_smoke(&parsed) {
        Ok(0) => {}
        Ok(status) => return ExitCode::from(status as u8),
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    }

    let mut best_prefill: Option<(usize, f64)> = None;
    let mut best_prefill_prompt_tokens_per_sec: Option<f64> = None;
    let mut best_decode: Option<(usize, f64)> = None;

    for batch in &parsed.batches {
        println!();
        println!("==> Running with {PREFILL_BATCH_ENV}={batch}");
        match run_prefill_bench_1b_batch(&parsed, *batch) {
            Ok((status, output, metrics)) => {
                print!("{output}");
                print_prefill_bench_1b_batch_json(*batch, status, metrics);
                if status != 0 {
                    return ExitCode::from(status as u8);
                }
                if let Some(prefill_sec) = metrics.prefill_sec
                    && best_prefill.is_none_or(|(_, best_sec)| prefill_sec < best_sec)
                {
                    best_prefill = Some((*batch, prefill_sec));
                    best_prefill_prompt_tokens_per_sec = prefill_prompt_tokens_per_sec(metrics);
                }
                if let Some(tokens_per_sec) = metrics.tokens_per_sec
                    && best_decode
                        .is_none_or(|(_, best_tokens_per_sec)| tokens_per_sec > best_tokens_per_sec)
                {
                    best_decode = Some((*batch, tokens_per_sec));
                }
            }
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        }
    }

    println!("prefill_bench_1b_status: ok");
    println!(
        "json: {}",
        prefill_bench_1b_result_json(
            &parsed,
            &context_limit_plan_value(),
            best_prefill,
            best_prefill_prompt_tokens_per_sec,
            best_decode
        )
    );
    ExitCode::SUCCESS
}

fn run_prefill_bench_1b_smoke(parsed: &Bench1BArgs) -> Result<i32, String> {
    let current_exe =
        env::current_exe().map_err(|err| format!("failed to resolve current executable: {err}"))?;
    let mut command = Command::new(current_exe);
    command
        .arg("smoke")
        .arg("1b")
        .arg(&parsed.model_path)
        .arg("chat")
        .arg(&parsed.prompt)
        .arg(parsed.max_tokens.to_string());
    for (key, value) in prefill_bench_1b_smoke_env(context_limit_env_value().as_deref()) {
        command.env(key, value);
    }
    let output = command
        .output()
        .map_err(|err| format!("failed to run 1B prefill smoke gate: {err}"))?;
    io::stderr().write_all(&output.stderr).ok();
    print!("{}", String::from_utf8_lossy(&output.stdout));
    Ok(output.status.code().unwrap_or(1))
}

fn run_prefill_bench_1b_batch(
    parsed: &Bench1BArgs,
    batch: usize,
) -> Result<(i32, String, PrefillBenchBatchMetrics), String> {
    let current_exe =
        env::current_exe().map_err(|err| format!("failed to resolve current executable: {err}"))?;
    let mut command = Command::new(current_exe);
    command
        .arg("chat")
        .arg(&parsed.model_path)
        .arg(&parsed.prompt)
        .arg(&parsed.temp)
        .arg(parsed.max_tokens.to_string());
    for (key, value) in prefill_bench_1b_batch_env(batch, context_limit_env_value().as_deref()) {
        command.env(key, value);
    }
    let output = command
        .output()
        .map_err(|err| format!("failed to run 1B prefill batch {batch}: {err}"))?;
    io::stderr().write_all(&output.stderr).ok();
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let status = output.status.code().unwrap_or(1);
    let metrics = parse_prefill_bench_1b_batch_metrics(&stdout);
    Ok((status, stdout, metrics))
}

fn parse_prefill_bench_1b_batch_metrics(output: &str) -> PrefillBenchBatchMetrics {
    let mut metrics = PrefillBenchBatchMetrics::default();
    for line in output.lines() {
        if let Some(rest) = line.strip_prefix("Prompt ingested in ")
            && let Some((seconds, _)) = rest.split_once("s with prefill batch ")
        {
            metrics.prefill_sec = seconds.parse::<f64>().ok();
        } else if let Some(rest) = line.strip_prefix("Generated ")
            && let Some((tokens, rest)) = rest.split_once(" tokens in ")
            && let Some((seconds, rest)) = rest.split_once("s (")
            && let Some((tokens_per_sec, _)) = rest.split_once(" tokens/sec)")
        {
            metrics.generated_tokens = tokens.parse::<usize>().ok();
            metrics.generation_sec = seconds.parse::<f64>().ok();
            metrics.tokens_per_sec = tokens_per_sec.parse::<f64>().ok();
        } else if let Some(json) = line.strip_prefix("json: ") {
            metrics.prompt_tokens =
                parse_json_usize_field(json, "prompt_tokens").or(metrics.prompt_tokens);
            metrics.prefill_sec = parse_json_f64_field(json, "prefill_sec").or(metrics.prefill_sec);
            metrics.generated_tokens =
                parse_json_usize_field(json, "generated_tokens").or(metrics.generated_tokens);
            metrics.generation_sec =
                parse_json_f64_field(json, "generation_sec").or(metrics.generation_sec);
            metrics.tokens_per_sec =
                parse_json_f64_field(json, "tokens_per_sec").or(metrics.tokens_per_sec);
        }
    }
    metrics
}

fn parse_json_usize_field(json: &str, field: &str) -> Option<usize> {
    parse_json_number_field(json, field)?.parse::<usize>().ok()
}

fn parse_json_f64_field(json: &str, field: &str) -> Option<f64> {
    parse_json_number_field(json, field)?.parse::<f64>().ok()
}

fn parse_json_number_field<'a>(json: &'a str, field: &str) -> Option<&'a str> {
    let marker = format!("\"{field}\":");
    let rest = json.split_once(&marker)?.1;
    let value = rest.trim_start();
    let end = value
        .find(|ch: char| !(ch.is_ascii_digit() || matches!(ch, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(value.len());
    (end > 0).then_some(&value[..end])
}

fn print_prefill_bench_1b_batch_json(
    batch: usize,
    exit_status: i32,
    metrics: PrefillBenchBatchMetrics,
) {
    let status = if exit_status == 0 { "ok" } else { "failed" };
    println!(
        "json: {{\"benchmark\":\"llama32-1b-prefill\",\"batch_size\":{},\"status\":\"{}\",\"exit_status\":{},\"prompt_tokens\":{},\"prefill_sec\":{},\"prompt_tokens_per_sec\":{},\"generated_tokens\":{},\"generation_sec\":{},\"tokens_per_sec\":{}}}",
        batch,
        status,
        exit_status,
        json_optional_usize(metrics.prompt_tokens),
        json_optional_f64(metrics.prefill_sec),
        json_optional_f64(prefill_prompt_tokens_per_sec(metrics)),
        json_optional_usize(metrics.generated_tokens),
        json_optional_f64(metrics.generation_sec),
        json_optional_f64(metrics.tokens_per_sec),
    );
}

fn prefill_prompt_tokens_per_sec(metrics: PrefillBenchBatchMetrics) -> Option<f64> {
    let prompt_tokens = metrics.prompt_tokens?;
    let prefill_sec = metrics.prefill_sec?;
    (prefill_sec > 0.0).then_some(prompt_tokens as f64 / prefill_sec)
}

fn prefill_bench_1b_result_json(
    parsed: &Bench1BArgs,
    context_limit: &str,
    best_prefill: Option<(usize, f64)>,
    best_prefill_prompt_tokens_per_sec: Option<f64>,
    best_decode: Option<(usize, f64)>,
) -> String {
    prefill_bench_1b_status_json_with_results(
        parsed,
        context_limit,
        best_prefill,
        best_prefill_prompt_tokens_per_sec,
        best_decode,
    )
}

fn json_optional_f64(value: Option<f64>) -> String {
    value.map(json_f64).unwrap_or_else(|| "null".to_owned())
}

fn json_optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_owned())
}

fn synthetic_q4_blocks(rows: usize, blocks_per_row: usize) -> Vec<q8::Q4_0Block> {
    let mut weights = Vec::with_capacity(rows * blocks_per_row);
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let scale_bits = 0x3800 + ((row + block) % 8) as u16;
            let values = core::array::from_fn(|idx| {
                let low = ((row + block + idx) % 16) as u8;
                let high = ((row.wrapping_mul(3) + block + idx * 5) % 16) as u8;
                low | (high << 4)
            });
            weights.push(q8::Q4_0Block::from_parts(scale_bits, values));
        }
    }
    weights
}

fn swizzle_q4_0_1x4(
    row_major: &[q8::Q4_0Block],
    rows: usize,
    blocks_per_row: usize,
) -> Vec<q8::Q4_0Block> {
    debug_assert_eq!(row_major.len(), rows * blocks_per_row);
    debug_assert!(rows.is_multiple_of(4));

    let mut swizzled = Vec::with_capacity(row_major.len());
    for row_base in (0..rows).step_by(4) {
        for block in 0..blocks_per_row {
            swizzled.push(row_major[row_base * blocks_per_row + block]);
            swizzled.push(row_major[(row_base + 1) * blocks_per_row + block]);
            swizzled.push(row_major[(row_base + 2) * blocks_per_row + block]);
            swizzled.push(row_major[(row_base + 3) * blocks_per_row + block]);
        }
    }
    swizzled
}

fn fill_synthetic_q8_activations(
    x_i8: &mut [i8],
    x_scales: &mut [f32],
    batch_size: usize,
    cols: usize,
    salt: u32,
) {
    let blocks_per_token = cols / q8::Q8_BLOCK_SIZE;
    for token_idx in 0..batch_size {
        let token_salt = salt.wrapping_add(token_idx as u32 * 31);
        let x_start = token_idx * cols;
        for (idx, value) in x_i8[x_start..x_start + cols].iter_mut().enumerate() {
            *value = ((idx as u32 * 17 + token_salt) % 127) as i8 - 63;
        }
        let scale_start = token_idx * blocks_per_token;
        for (idx, scale) in x_scales[scale_start..scale_start + blocks_per_token]
            .iter_mut()
            .enumerate()
        {
            *scale = 0.015625 * (1 + ((idx + token_idx + salt as usize) % 7)) as f32;
        }
    }
}

fn q4_prefill_checksum(out: &[f32]) -> f64 {
    out.iter()
        .enumerate()
        .map(|(idx, value)| *value as f64 * (1 + idx % 17) as f64)
        .sum()
}

fn median_duration(mut durations: Vec<Duration>) -> Duration {
    durations.sort_unstable();
    durations[durations.len() / 2]
}

fn q8_dot_json(report: &q8::DotBenchmarkReport) -> String {
    let scalar_runs = duration_ms_json(&report.scalar.elapsed_runs);
    let neon_json = report.neon.as_ref().map(|neon| {
        format!(
            ",\"neon\":{{\"checksum\":{},\"run_ms\":{},\"min_ns_per_block\":{:.6},\"median_ns_per_block\":{:.6},\"min_speedup\":{:.6},\"median_speedup\":{:.6}}}",
            neon.checksum,
            duration_ms_json(&neon.elapsed_runs),
            report.neon_min_ns_per_block().unwrap_or_default(),
            report.neon_median_ns_per_block().unwrap_or_default(),
            report.neon_min_speedup().unwrap_or_default(),
            report.neon_median_speedup().unwrap_or_default()
        )
    }).unwrap_or_default();
    let sdot_json = report
        .sdot
        .as_ref()
        .map(|sdot| {
            format!(
                ",\"sdot\":{{\"checksum\":{},\"run_ms\":{},\"min_ns_per_block\":{:.6},\"median_ns_per_block\":{:.6},\"min_speedup\":{:.6},\"median_speedup\":{:.6},\"vs_neon_min_speedup\":{:.6},\"vs_neon_median_speedup\":{:.6}}}",
                sdot.checksum,
                duration_ms_json(&sdot.elapsed_runs),
                report.sdot_min_ns_per_block().unwrap_or_default(),
                report.sdot_median_ns_per_block().unwrap_or_default(),
                report.sdot_min_speedup().unwrap_or_default(),
                report.sdot_median_speedup().unwrap_or_default(),
                report.sdot_vs_neon_min_speedup().unwrap_or_default(),
                report.sdot_vs_neon_median_speedup().unwrap_or_default()
            )
        })
        .unwrap_or_default();
    let kernel_json = kernel_selector_json(report);
    let suffix_json = format!("{kernel_json}{neon_json}{sdot_json}");

    format!(
        "{{\"benchmark\":\"q8-dot\",\"iterations\":{},\"runs\":{},\"blocks_per_iteration\":{},\"elements_per_iteration\":{},\"selected\":{{\"checksum\":{},\"run_ms\":{}}},\"scalar\":{{\"checksum\":{},\"run_ms\":{},\"min_ns_per_block\":{:.6},\"median_ns_per_block\":{:.6}}},\"neon_available\":{},\"dotprod_feature_detected\":{},\"sdot_candidate_requested\":{},\"sdot_candidate_enabled\":{}{}}}",
        report.iterations,
        report.runs,
        report.blocks_per_iteration,
        report.elements_per_iteration,
        report.selected.checksum,
        duration_ms_json(&report.selected.elapsed_runs),
        report.scalar.checksum,
        scalar_runs,
        report.scalar_min_ns_per_block(),
        report.scalar_median_ns_per_block(),
        report.neon.is_some(),
        q8::dotprod_available(),
        q8::sdot_candidate_requested(),
        q8::sdot_candidate_enabled(),
        suffix_json
    )
}

fn kernel_selector_json(report: &q8::DotBenchmarkReport) -> String {
    format!(
        ",\"kernel_selector\":{{\"requested\":{},\"selected\":\"{}\",\"fallback_reason\":{}}}",
        report
            .kernel_selector
            .requested
            .map(|kernel| format!("\"{}\"", kernel.name()))
            .unwrap_or_else(|| "null".to_string()),
        report.kernel_selector.selected.name(),
        report
            .kernel_selector
            .fallback_reason
            .map(|reason| format!("\"{reason}\""))
            .unwrap_or_else(|| "null".to_string())
    )
}

fn duration_ms_json(durations: &[std::time::Duration]) -> String {
    let values = durations
        .iter()
        .map(|duration| format!("{:.6}", duration.as_secs_f64() * 1000.0))
        .collect::<Vec<_>>();
    format!("[{}]", values.join(","))
}

fn print_tui_dry_run(parsed: &TuiArgs) -> ExitCode {
    if let Err(err) = validate_context_limit_env() {
        eprintln!("{err}");
        return ExitCode::from(2);
    }
    let prefill_batch = match prefill_batch_size_from_env() {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(2);
        }
    };

    let model_path = Path::new(&parsed.model_path);
    println!("NanoCamelid TUI launch dry run");
    println!("selected_source: {}", parsed.model_source);
    println!("model: {}", model_path.display());
    println!("model_exists: {}", model_path.is_file());
    println!("temp: {}", parsed.temp);
    println!("max_tokens: {}", parsed.max_tokens);
    println!("context_limit: {}", context_limit_plan_value());
    println!("prefill_batch: {prefill_batch}");
    print_direct_1b_audit_plan(model_path, parsed.audit_1b_shape);
    println!(
        "tui_command: {}",
        context_limited_shell_command(&[
            "nanocamelid",
            "tui",
            &model_path.display().to_string(),
            &parsed.temp.to_string(),
            &parsed.max_tokens.to_string(),
        ])
    );
    ExitCode::SUCCESS
}

fn print_generation_dry_run(command: &str, parsed: &GenerateArgs) -> ExitCode {
    if let Err(err) = validate_context_limit_env() {
        eprintln!("{err}");
        return ExitCode::from(2);
    }
    let prefill_batch = match prefill_batch_size_from_env() {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(2);
        }
    };

    let model_path = Path::new(&parsed.model_path);
    println!("NanoCamelid {command} dry run");
    println!("selected_source: {}", parsed.model_source);
    println!("model: {}", model_path.display());
    println!("model_exists: {}", model_path.is_file());
    println!("prompt: {}", parsed.prompt);
    println!("temp: {}", parsed.temp);
    println!("max_tokens: {}", parsed.max_tokens);
    println!("context_limit: {}", context_limit_plan_value());
    println!("prefill_batch: {prefill_batch}");
    print_direct_1b_audit_plan(model_path, parsed.audit_1b_shape);
    println!(
        "{command}_command: {}",
        context_limited_shell_command(&[
            "nanocamelid",
            command,
            &model_path.display().to_string(),
            &parsed.prompt,
            &parsed.temp.to_string(),
            &parsed.max_tokens.to_string(),
        ])
    );
    ExitCode::SUCCESS
}

fn print_direct_1b_audit_plan(model_path: &Path, audit_1b_shape: bool) {
    if audit_1b_shape {
        println!("shape_audit: enabled");
        println!(
            "model_command: {}",
            shell_command(&[
                "nanocamelid",
                "model",
                "1b",
                &model_path.display().to_string(),
            ])
        );
    } else {
        println!("shape_audit: not_applicable");
    }
}

fn run_inspect(parsed: InspectArgs) -> ExitCode {
    let model_path = Path::new(&parsed.model_path);

    if parsed.dry_run {
        println!("NanoCamelid inspect dry run");
        match parsed.target {
            Some(InspectTarget::Llama32_1B) => {
                let workspace =
                    env::var(WORKSPACE_ENV).unwrap_or_else(|_| DEFAULT_PI_WORKSPACE.to_owned());
                let q4_model_path = llama32_1b_model_path(&workspace, LLAMA32_1B_Q4_MODEL);
                let q8_model_path = llama32_1b_model_path(&workspace, LLAMA32_1B_Q8_MODEL);
                println!("workspace: {workspace}");
                println!("q4_model: {q4_model_path}");
                println!("q4_exists: {}", Path::new(&q4_model_path).is_file());
                println!("q8_model: {q8_model_path}");
                println!("q8_exists: {}", Path::new(&q8_model_path).is_file());
            }
            Some(InspectTarget::Llama32_3B) => {
                let workspace =
                    env::var(WORKSPACE_ENV).unwrap_or_else(|_| DEFAULT_PI_WORKSPACE.to_owned());
                let q4_model_path = llama32_1b_model_path(&workspace, LLAMA32_3B_Q4_MODEL);
                println!("workspace: {workspace}");
                println!("q4_model: {q4_model_path}");
                println!("q4_exists: {}", Path::new(&q4_model_path).is_file());
            }
            None => {}
        }
        println!("selected_source: {}", parsed.model_source);
        println!("model: {}", model_path.display());
        println!("model_exists: {}", model_path.is_file());
        if parsed.target == Some(InspectTarget::Llama32_1B) {
            println!("shape_audit: enabled");
            println!("status_on_success: inspect_1b_status: ok");
            println!(
                "json_on_success: {}",
                inspect_1b_status_json(model_path, parsed.model_source)
            );
        }
        println!(
            "inspect_command: {}",
            match parsed.target {
                Some(InspectTarget::Llama32_1B) => shell_command(&[
                    "nanocamelid",
                    "inspect",
                    "1b",
                    &model_path.display().to_string(),
                ]),
                _ => shell_command(&["nanocamelid", "inspect", &model_path.display().to_string()]),
            }
        );
        return ExitCode::SUCCESS;
    }

    match parsed.target {
        Some(InspectTarget::Llama32_1B) if !model_path.is_file() => {
            eprintln!("{}", llama32_1b_model_not_found_message(model_path));
            ExitCode::from(2)
        }
        Some(InspectTarget::Llama32_3B) if !model_path.is_file() => {
            eprintln!("{}", llama32_3b_model_not_found_message(model_path));
            ExitCode::from(2)
        }
        _ => inspect_gguf(
            model_path,
            parsed.target == Some(InspectTarget::Llama32_1B),
            (parsed.target == Some(InspectTarget::Llama32_1B)).then_some(parsed.model_source),
        ),
    }
}

fn inspect_gguf(
    path: &Path,
    strict_llama32_1b_shape: bool,
    llama32_1b_model_source: Option<&str>,
) -> ExitCode {
    match gguf::read_file(path) {
        Ok(file) => {
            let summary = gguf::summarize(&file);
            let runtime = inspect_runtime_summary(&file);
            println!("NanoCamelid GGUF inspect");
            println!("path: {}", path.display());
            println!("version: {}", summary.version);
            println!("tensor_count: {}", summary.tensor_count);
            println!("metadata_count: {}", summary.metadata_count);
            println!("alignment: {}", summary.alignment);
            println!("data_start_offset: {}", summary.data_start_offset);

            println!();
            println!("runtime:");
            println!(
                "  readiness: {}",
                if runtime.ready {
                    "ready"
                } else {
                    "unsupported"
                }
            );
            println!("  tied_output: {}", runtime.tied_output);
            println!(
                "  tensor_type_support: {}",
                if runtime.unsupported_tensor_types.is_empty() {
                    "ok"
                } else {
                    "unsupported"
                }
            );
            if !runtime.supported_tensor_types.is_empty() {
                println!(
                    "  supported_tensor_types: {}",
                    runtime.supported_tensor_types.join(",")
                );
            }
            if !runtime.unsupported_tensor_types.is_empty() {
                println!(
                    "  unsupported_tensor_types: {}",
                    runtime.unsupported_tensor_types.join(",")
                );
            }

            match &runtime.model_config {
                Ok(config) => {
                    println!("  architecture: {}", config.architecture);
                    println!("  vocab_size: {}", config.vocab_size);
                    println!("  context_length: {}", config.context_length);
                    println!("  embedding_length: {}", config.embedding_length);
                    println!("  block_count: {}", config.block_count);
                    println!("  attention_heads: {}", config.attention_head_count);
                    println!("  attention_kv_heads: {}", config.attention_head_count_kv);
                    println!("  head_dim: {}", config.head_dim);
                    println!("  kv_width: {}", config.kv_width);
                    println!("  rope_dimension_count: {}", config.rope_dimension_count);
                    println!("  rope_freq_base: {}", config.rope_freq_base);
                    println!("  rms_norm_epsilon: {}", config.rms_norm_epsilon);
                }
                Err(err) => println!("  config_error: {err}"),
            }

            match &runtime.tensor_layouts {
                Ok(()) => println!("  tensor_layouts: ok"),
                Err(err) => println!("  tensor_layout_error: {err}"),
            }

            match &runtime.tokenizer {
                Ok(tokenizer) => {
                    println!("  tokenizer_model: {}", tokenizer.model.as_summary_model());
                    println!(
                        "  tokenizer_chat_template: {}",
                        tokenizer.chat_template_present
                    );
                    println!(
                        "  tokenizer_chat_template_format: {}",
                        tokenizer.chat_template_format.unwrap_or("none")
                    );
                    println!("  tokenizer_add_bos: {}", tokenizer.add_bos);
                    println!("  tokenizer_add_eos: {}", tokenizer.add_eos);
                    println!(
                        "  tokenizer_add_space_prefix: {}",
                        tokenizer.add_space_prefix
                    );
                    println!(
                        "  tokenizer_remove_extra_whitespaces: {}",
                        tokenizer.remove_extra_whitespaces
                    );
                    println!("  tokenizer_bos: {:?}", tokenizer.bos);
                    println!("  tokenizer_eos: {:?}", tokenizer.eos);
                    println!("  tokenizer_eot: {:?}", tokenizer.eot);
                    println!("  tokenizer_eom: {:?}", tokenizer.eom);
                }
                Err(err) => println!("  tokenizer_error: {err}"),
            }

            let shape = llama32_1b_shape_audit(&file);
            println!(
                "  llama32_1b_shape: {}",
                if shape.ready { "ok" } else { "mismatch" }
            );
            if !shape.mismatches.is_empty() {
                println!("  llama32_1b_mismatches: {}", shape.mismatches.join("; "));
            }

            let rope_prefix = runtime
                .model_config
                .as_ref()
                .map(|config| config.metadata_prefix.as_str())
                .unwrap_or("llama");
            if let Some(factor) = file.metadata_f32(&format!("{rope_prefix}.rope.scaling.factor")) {
                println!("  rope_scaling_factor: {factor}");
            }
            if let Some(original_context_length) = file.metadata_u32(&format!(
                "{rope_prefix}.rope.scaling.original_context_length"
            )) {
                println!("  rope_scaling_original_context_length: {original_context_length}");
            }
            if let Some(low_freq_factor) =
                file.metadata_f32(&format!("{rope_prefix}.rope.scaling.low_freq_factor"))
            {
                println!("  rope_scaling_low_freq_factor: {low_freq_factor}");
            }
            if let Some(high_freq_factor) =
                file.metadata_f32(&format!("{rope_prefix}.rope.scaling.high_freq_factor"))
            {
                println!("  rope_scaling_high_freq_factor: {high_freq_factor}");
            }

            if !summary.important_metadata.is_empty() {
                println!();
                println!("metadata:");
                for entry in &summary.important_metadata {
                    println!("  {}: {}", entry.key, entry.value);
                }
            }

            if !summary.tensor_types.is_empty() {
                println!();
                println!("tensor_types:");
                for entry in &summary.tensor_types {
                    println!("  {}: {}", entry.name, entry.count);
                }
            }

            if !summary.tensors.is_empty() {
                println!();
                println!("tensors:");
                for tensor in summary.tensors.iter().take(8) {
                    println!(
                        "  {} dims={:?} type={} rel_offset={} abs_offset={} bytes={}",
                        tensor.name,
                        tensor.dimensions,
                        tensor.tensor_type.name(),
                        tensor.relative_offset,
                        tensor.absolute_offset,
                        tensor.n_bytes
                    );
                }
                if summary.tensors.len() > 8 {
                    println!("  ... {} more", summary.tensors.len() - 8);
                }
            }

            if strict_llama32_1b_shape && !shape.ready {
                eprintln!(
                    "inspect 1b failed strict shape audit: {}",
                    shape.mismatches.join("; ")
                );
                return ExitCode::from(2);
            }
            if strict_llama32_1b_shape {
                println!("inspect_1b_status: ok");
                println!(
                    "json: {}",
                    inspect_1b_status_json(
                        path,
                        llama32_1b_model_source.unwrap_or("explicit argument")
                    )
                );
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("inspect failed: {err}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct InspectRuntimeSummary {
    ready: bool,
    tied_output: bool,
    supported_tensor_types: Vec<String>,
    unsupported_tensor_types: Vec<String>,
    model_config: Result<model::LlamaModelConfig, String>,
    tensor_layouts: Result<(), String>,
    tokenizer: Result<InspectTokenizerSummary, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InspectTokenizerSummary {
    model: tokenizer::TokenizerModel,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    remove_extra_whitespaces: bool,
    chat_template_present: bool,
    chat_template_format: Option<&'static str>,
    bos: Option<u32>,
    eos: Option<u32>,
    eot: Option<u32>,
    eom: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelShapeAudit {
    label: &'static str,
    ready: bool,
    mismatches: Vec<String>,
}

fn llama32_1b_shape_audit(gguf: &gguf::GgufFile) -> ModelShapeAudit {
    let mut mismatches = Vec::new();
    let config = match model::LlamaModelConfig::from_gguf(gguf) {
        Ok(config) => {
            check_shape_value(
                &mut mismatches,
                "architecture",
                "llama",
                config.architecture.as_str(),
            );
            check_shape_value(
                &mut mismatches,
                "context_length",
                131_072,
                config.context_length,
            );
            check_shape_value(
                &mut mismatches,
                "embedding_length",
                2_048,
                config.embedding_length,
            );
            check_shape_value(&mut mismatches, "block_count", 16, config.block_count);
            check_shape_value(
                &mut mismatches,
                "feed_forward_length",
                8_192,
                config.feed_forward_length,
            );
            check_shape_value(
                &mut mismatches,
                "attention_head_count",
                32,
                config.attention_head_count,
            );
            check_shape_value(
                &mut mismatches,
                "attention_head_count_kv",
                8,
                config.attention_head_count_kv,
            );
            check_shape_value(
                &mut mismatches,
                "rope_dimension_count",
                64,
                config.rope_dimension_count,
            );
            check_shape_value(&mut mismatches, "vocab_size", 128_256, config.vocab_size);
            if (config.rope_freq_base - 500_000.0).abs() > f32::EPSILON {
                mismatches.push(format!(
                    "rope_freq_base expected 500000 got {}",
                    config.rope_freq_base
                ));
            }
            Some(config)
        }
        Err(err) => {
            mismatches.push(format!("config_error {err}"));
            None
        }
    };
    if let Some(config) = &config {
        let embedding = config.embedding_length as u64;
        let vocab = config.vocab_size as u64;

        check_tensor_dimensions(
            &mut mismatches,
            gguf,
            "token_embd.weight",
            &[embedding, vocab],
        );
        check_tensor_dimensions(&mut mismatches, gguf, "output_norm.weight", &[embedding]);
        check_optional_tensor_matrix_dimensions(
            &mut mismatches,
            gguf,
            "output.weight",
            embedding,
            vocab,
        );
        check_llama32_1b_block_tensors(&mut mismatches, gguf, config);
    }
    match tokenizer::Tokenizer::from_gguf(gguf) {
        Ok(tokenizer) => check_shape_value(
            &mut mismatches,
            "tokenizer_chat_template_format",
            "llama3_instruct",
            tokenizer.chat_template_format().unwrap_or("none"),
        ),
        Err(err) => mismatches.push(format!("tokenizer_error {err}")),
    }

    ModelShapeAudit {
        label: "llama32_1b",
        ready: mismatches.is_empty(),
        mismatches,
    }
}

fn check_llama32_1b_block_tensors(
    mismatches: &mut Vec<String>,
    gguf: &gguf::GgufFile,
    config: &model::LlamaModelConfig,
) {
    let embedding = config.embedding_length as u64;
    let attention_output = config.attention_output_width as u64;
    let kv_width = config.kv_width as u64;
    let feed_forward = config.feed_forward_length as u64;

    for layer_idx in 0..config.block_count {
        check_tensor_dimensions(
            mismatches,
            gguf,
            &format!("blk.{layer_idx}.attn_norm.weight"),
            &[embedding],
        );
        check_tensor_dimensions(
            mismatches,
            gguf,
            &format!("blk.{layer_idx}.attn_q.weight"),
            &[embedding, attention_output],
        );
        check_tensor_dimensions(
            mismatches,
            gguf,
            &format!("blk.{layer_idx}.attn_k.weight"),
            &[embedding, kv_width],
        );
        check_tensor_dimensions(
            mismatches,
            gguf,
            &format!("blk.{layer_idx}.attn_v.weight"),
            &[embedding, kv_width],
        );
        check_tensor_dimensions(
            mismatches,
            gguf,
            &format!("blk.{layer_idx}.attn_output.weight"),
            &[attention_output, embedding],
        );
        check_tensor_dimensions(
            mismatches,
            gguf,
            &format!("blk.{layer_idx}.ffn_norm.weight"),
            &[embedding],
        );
        check_tensor_dimensions(
            mismatches,
            gguf,
            &format!("blk.{layer_idx}.ffn_gate.weight"),
            &[embedding, feed_forward],
        );
        check_tensor_dimensions(
            mismatches,
            gguf,
            &format!("blk.{layer_idx}.ffn_up.weight"),
            &[embedding, feed_forward],
        );
        check_tensor_dimensions(
            mismatches,
            gguf,
            &format!("blk.{layer_idx}.ffn_down.weight"),
            &[feed_forward, embedding],
        );
    }
}

fn check_tensor_dimensions(
    mismatches: &mut Vec<String>,
    gguf: &gguf::GgufFile,
    name: &str,
    expected: &[u64],
) {
    match gguf.tensors.iter().find(|tensor| tensor.name == name) {
        Some(tensor) if tensor.dimensions == expected => {}
        Some(tensor) => mismatches.push(format!(
            "{name} dims expected {:?} got {:?}",
            expected, tensor.dimensions
        )),
        None => mismatches.push(format!("{name} missing")),
    }
}

fn check_optional_tensor_matrix_dimensions(
    mismatches: &mut Vec<String>,
    gguf: &gguf::GgufFile,
    name: &str,
    input_width: u64,
    output_width: u64,
) {
    let Some(tensor) = gguf.tensors.iter().find(|tensor| tensor.name == name) else {
        return;
    };
    let direct = [input_width, output_width];
    let transposed = [output_width, input_width];
    if tensor.dimensions != direct && tensor.dimensions != transposed {
        mismatches.push(format!(
            "{name} dims expected {:?} or {:?} got {:?}",
            direct, transposed, tensor.dimensions
        ));
    }
}

fn check_shape_value<T>(mismatches: &mut Vec<String>, name: &str, expected: T, actual: T)
where
    T: std::fmt::Display + PartialEq,
{
    if actual != expected {
        mismatches.push(format!("{name} expected {expected} got {actual}"));
    }
}

fn inspect_runtime_summary(gguf: &gguf::GgufFile) -> InspectRuntimeSummary {
    let model_config = model::LlamaModelConfig::from_gguf(gguf);
    let tensor_layouts = model_config
        .as_ref()
        .map_err(std::clone::Clone::clone)
        .and_then(|config| model::validate_model_tensors(gguf, config));
    let tokenizer =
        tokenizer::Tokenizer::from_gguf(gguf).map(|tokenizer| InspectTokenizerSummary {
            model: tokenizer.model,
            add_bos: tokenizer.config.add_bos,
            add_eos: tokenizer.config.add_eos,
            add_space_prefix: tokenizer.config.add_space_prefix,
            remove_extra_whitespaces: tokenizer.config.remove_extra_whitespaces,
            chat_template_present: tokenizer.chat_template.is_some(),
            chat_template_format: tokenizer.chat_template_format(),
            bos: tokenizer.special.bos,
            eos: tokenizer.special.eos,
            eot: tokenizer.special.eot,
            eom: tokenizer.special.eom,
        });
    let tied_output = !gguf
        .tensors
        .iter()
        .any(|tensor| tensor.name == "output.weight");
    let (supported_tensor_types, unsupported_tensor_types) = inspect_tensor_type_support(gguf);
    let ready = model_config.is_ok()
        && tensor_layouts.is_ok()
        && tokenizer.is_ok()
        && unsupported_tensor_types.is_empty();

    InspectRuntimeSummary {
        ready,
        tied_output,
        supported_tensor_types,
        unsupported_tensor_types,
        model_config,
        tensor_layouts,
        tokenizer,
    }
}

fn inspect_tensor_type_support(gguf: &gguf::GgufFile) -> (Vec<String>, Vec<String>) {
    let mut supported = BTreeSet::new();
    let mut unsupported = BTreeSet::new();

    for tensor in &gguf.tensors {
        let name = tensor.tensor_type.name().to_owned();
        if runtime_supports_tensor_type(tensor.tensor_type) {
            supported.insert(name);
        } else {
            unsupported.insert(name);
        }
    }

    (
        supported.into_iter().collect(),
        unsupported.into_iter().collect(),
    )
}

fn runtime_supports_tensor_type(tensor_type: gguf::GgufTensorType) -> bool {
    matches!(
        tensor_type,
        gguf::GgufTensorType::F32
            | gguf::GgufTensorType::F16
            | gguf::GgufTensorType::Q8_0
            | gguf::GgufTensorType::Q4_0
            | gguf::GgufTensorType::Q4_1
            | gguf::GgufTensorType::Q5_0
            | gguf::GgufTensorType::Q5_1
            | gguf::GgufTensorType::Q2K
            | gguf::GgufTensorType::Q3K
            | gguf::GgufTensorType::Q4K
            | gguf::GgufTensorType::Q5K
            | gguf::GgufTensorType::Q6K
            | gguf::GgufTensorType::Q8K
            | gguf::GgufTensorType::IQ4NL
    )
}

fn smoke_q8_model(model_path: &Path, prompt: &str, max_tokens: usize) -> ExitCode {
    match run_q8_model_smoke(model_path, prompt, max_tokens) {
        Ok(report) => {
            print_q8_smoke_report("NanoCamelid Q8 model smoke", model_path, &report);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("Q8 model smoke failed: {err}");
            ExitCode::FAILURE
        }
    }
}

fn smoke_q8_chat(model_path: &Path, prompt: &str, max_tokens: usize) -> ExitCode {
    match run_q8_chat_smoke(model_path, prompt, max_tokens) {
        Ok(report) => {
            print_q8_smoke_report("NanoCamelid Q8 chat smoke", model_path, &report);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("Q8 chat smoke failed: {err}");
            ExitCode::FAILURE
        }
    }
}

fn print_smoke_dry_run(
    title: &str,
    target: &str,
    model_path: &Path,
    parsed: &Smoke1BArgs,
) -> ExitCode {
    if let Err(err) = validate_context_limit_env() {
        eprintln!("{err}");
        return ExitCode::from(2);
    }
    let prefill_batch = match prefill_batch_size_from_env() {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(2);
        }
    };

    println!("{title}");
    if target == "1b" {
        let workspace = env::var(WORKSPACE_ENV).unwrap_or_else(|_| DEFAULT_PI_WORKSPACE.to_owned());
        let q4_model_path = llama32_1b_model_path(&workspace, LLAMA32_1B_Q4_MODEL);
        let q8_model_path = llama32_1b_model_path(&workspace, LLAMA32_1B_Q8_MODEL);

        println!("workspace: {workspace}");
        println!("q4_model: {q4_model_path}");
        println!("q4_exists: {}", Path::new(&q4_model_path).is_file());
        println!("q8_model: {q8_model_path}");
        println!("q8_exists: {}", Path::new(&q8_model_path).is_file());
    }
    println!("selected_source: {}", parsed.model_source);
    println!("model: {}", model_path.display());
    println!("model_exists: {}", model_path.is_file());
    if target == "1b" {
        println!(
            "quantization: {}",
            llama32_1b_quantization_for_path(model_path)
        );
    }
    println!("context_limit: {}", context_limit_plan_value());
    if target == "1b" {
        println!("shape_audit: enabled");
    }
    println!("smoke_kind: {}", parsed.kind.label());
    println!("smoke_prompt: {}", parsed.prompt);
    println!("smoke_tokens: {}", parsed.max_tokens);
    println!("prefill_batch: {prefill_batch}");
    if target == "1b" {
        println!("status_on_success: smoke_1b_status: ok");
        println!(
            "json_on_success: {}",
            smoke_1b_status_json(
                model_path,
                parsed,
                &context_limit_plan_value(),
                prefill_batch,
            )
        );
        println!(
            "model_command: {}",
            shell_command(&[
                "nanocamelid",
                "model",
                "1b",
                &model_path.display().to_string()
            ])
        );
    }
    println!(
        "smoke_command: {}",
        smoke_plan_command(target, model_path, parsed)
    );
    ExitCode::SUCCESS
}

fn run_model_1b_audit(parsed: Model1BAuditArgs) -> ExitCode {
    let model_path = Path::new(&parsed.model_path);
    println!("NanoCamelid Llama 3.2 1B model audit");
    println!("workspace: {}", parsed.workspace);
    println!("q4_model: {}", parsed.q4_model_path);
    println!("q4_exists: {}", Path::new(&parsed.q4_model_path).is_file());
    println!("q8_model: {}", parsed.q8_model_path);
    println!("q8_exists: {}", Path::new(&parsed.q8_model_path).is_file());
    println!("selected_source: {}", parsed.model_source);
    println!("selected_model: {}", parsed.model_path);
    println!("selected_exists: {}", model_path.is_file());
    println!(
        "quantization: {}",
        llama32_1b_quantization_for_path(model_path)
    );

    if parsed.dry_run {
        let model_arg = model_path.display().to_string();
        let smoke_tokens = DEFAULT_1B_SMOKE_TOKENS.to_string();
        println!("shape_audit: enabled");
        println!("status_on_success: model_1b_status: ok");
        println!(
            "json_on_success: {}",
            model_1b_status_json(model_path, parsed.model_source)
        );
        println!(
            "model_command: {}",
            shell_command(&["nanocamelid", "model", "1b", &model_arg])
        );
        println!(
            "inspect_command: {}",
            shell_command(&["nanocamelid", "inspect", "1b", &model_arg])
        );
        println!(
            "smoke_command: {}",
            shell_command(&[
                "nanocamelid",
                "smoke",
                "1b",
                &model_arg,
                "chat",
                DEFAULT_1B_SMOKE_PROMPT,
                &smoke_tokens,
            ])
        );
        println!(
            "ready_command: {}",
            shell_command(&["nanocamelid", "ready", "1b", &model_arg])
        );
        println!(
            "evidence_command: {}",
            shell_command(&["nanocamelid", "evidence", "1b", &model_arg])
        );
        return ExitCode::SUCCESS;
    }

    if !model_path.is_file() {
        eprintln!("{}", llama32_1b_model_not_found_message(model_path));
        return ExitCode::from(2);
    }

    let audit_code = audit_llama32_1b_model_shape(model_path);
    if audit_code == ExitCode::SUCCESS {
        println!("model_1b_status: ok");
        println!(
            "json: {}",
            model_1b_status_json(model_path, parsed.model_source)
        );
    }
    audit_code
}

fn audit_llama32_1b_model_shape(model_path: &Path) -> ExitCode {
    match gguf::read_file(model_path) {
        Ok(file) => {
            let shape = llama32_1b_shape_audit(&file);
            println!("shape_check: {}", shape.label);
            println!("shape_ready: {}", shape.ready);
            if !shape.mismatches.is_empty() {
                println!("shape_mismatches: {}", shape.mismatches.join("; "));
                return ExitCode::from(2);
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("1B model audit failed: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run_ready_1b(parsed: Ready1BArgs) -> ExitCode {
    let prefill_batch = match validate_runtime_generation_env() {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(2);
        }
    };
    let smoke = parsed.smoke;
    let model_path = Path::new(&smoke.model_path);
    let chat_enabled = match parsed.chat_enabled_override {
        Some(enabled) => enabled,
        None => match ready_chat_enabled() {
            Ok(enabled) => enabled,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(2);
            }
        },
    };
    let chat_prompt = parsed
        .chat_prompt_override
        .clone()
        .unwrap_or_else(|| ready_chat_prompt(&smoke.prompt));
    let (chat_tokens, chat_temp) = if chat_enabled {
        let chat_tokens = match parsed.chat_tokens_override {
            Some(tokens) => tokens,
            None => match ready_chat_tokens(smoke.max_tokens) {
                Ok(tokens) => tokens,
                Err(err) => {
                    eprintln!("{err}");
                    return ExitCode::from(2);
                }
            },
        };
        let chat_temp = match ready_chat_temp() {
            Ok(temp) => temp,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(2);
            }
        };
        (Some(chat_tokens), Some(chat_temp))
    } else {
        (None, None)
    };

    if parsed.dry_run {
        if let Err(err) = validate_context_limit_env() {
            eprintln!("{err}");
            return ExitCode::from(2);
        }

        let workspace = env::var(WORKSPACE_ENV).unwrap_or_else(|_| DEFAULT_PI_WORKSPACE.to_owned());
        let q4_model_path = llama32_1b_model_path(&workspace, LLAMA32_1B_Q4_MODEL);
        let q8_model_path = llama32_1b_model_path(&workspace, LLAMA32_1B_Q8_MODEL);

        println!("NanoCamelid Llama 3.2 1B readiness dry run");
        println!("workspace: {workspace}");
        println!("q4_model: {q4_model_path}");
        println!("q4_exists: {}", Path::new(&q4_model_path).is_file());
        println!("q8_model: {q8_model_path}");
        println!("q8_exists: {}", Path::new(&q8_model_path).is_file());
        println!("selected_source: {}", smoke.model_source);
        println!("model: {}", model_path.display());
        println!("model_exists: {}", model_path.is_file());
        println!(
            "quantization: {}",
            llama32_1b_quantization_for_path(model_path)
        );
        println!("context_limit: {}", context_limit_plan_value());
        println!("shape_audit: enabled");
        println!("smoke_kind: {}", smoke.kind.label());
        println!("smoke_prompt: {}", smoke.prompt);
        println!("smoke_tokens: {}", smoke.max_tokens);
        println!("prefill_batch: {prefill_batch}");
        println!(
            "direct_chat: {}",
            if chat_enabled { "enabled" } else { "disabled" }
        );
        println!("status_on_success: ready_1b_status: ok");
        println!(
            "json_on_success: {}",
            ready_1b_status_json(
                model_path,
                &smoke,
                ReadyDirectChatStatus {
                    enabled: chat_enabled,
                    prompt: chat_enabled.then_some(chat_prompt.as_str()),
                    tokens: chat_tokens,
                    temp: chat_temp,
                },
                &context_limit_plan_value(),
                prefill_batch,
            )
        );
        println!(
            "probe_command: {}",
            shell_command(&["nanocamelid", "probe"])
        );
        println!(
            "model_command: {}",
            shell_command(&[
                "nanocamelid",
                "model",
                "1b",
                &model_path.display().to_string()
            ])
        );
        println!(
            "inspect_command: {}",
            shell_command(&[
                "nanocamelid",
                "inspect",
                "1b",
                &model_path.display().to_string(),
            ])
        );
        println!(
            "smoke_command: {}",
            smoke_plan_command("1b", model_path, &smoke)
        );
        if chat_enabled {
            let chat_tokens = chat_tokens.expect("chat tokens should be parsed when chat is on");
            let chat_temp = chat_temp.expect("chat temp should be parsed when chat is on");
            println!("chat_prompt: {chat_prompt}");
            println!("chat_temp: {chat_temp}");
            println!("chat_tokens: {chat_tokens}");
            println!(
                "chat_command: {}",
                context_limited_shell_command(&[
                    "nanocamelid",
                    "chat",
                    &model_path.display().to_string(),
                    &chat_prompt,
                    &chat_temp.to_string(),
                    &chat_tokens.to_string(),
                ])
            );
        } else {
            println!("chat_command: skipped");
        }
        return ExitCode::SUCCESS;
    }

    if !model_path.is_file() {
        eprintln!("{}", llama32_1b_model_not_found_message(model_path));
        return ExitCode::from(2);
    }

    println!("NanoCamelid Llama 3.2 1B readiness");
    println!("==> Probing host fast-path support");
    print_probe();
    println!("==> Auditing 1B model shape: {}", model_path.display());
    let model_audit_code = audit_llama32_1b_model_shape(model_path);
    if model_audit_code != ExitCode::SUCCESS {
        return model_audit_code;
    }
    println!("==> Inspecting 1B model: {}", model_path.display());
    let inspect_code = inspect_gguf(model_path, false, None);
    if inspect_code != ExitCode::SUCCESS {
        return inspect_code;
    }

    println!("==> Running 1B {} smoke gate", smoke.kind.label());
    let smoke_code = match smoke.kind {
        SmokeKind::Q8Model => smoke_q8_model(model_path, &smoke.prompt, smoke.max_tokens),
        SmokeKind::Q8Chat => smoke_q8_chat(model_path, &smoke.prompt, smoke.max_tokens),
    };
    if smoke_code != ExitCode::SUCCESS {
        return smoke_code;
    }

    if !chat_enabled {
        let reason = if parsed.chat_enabled_override == Some(false) {
            "--no-chat"
        } else {
            READY_CHAT_ENV
        };
        println!("==> Skipping direct 1B chat turn; {reason}");
        println!("ready_1b_status: ok");
        println!(
            "json: {}",
            ready_1b_status_json(
                model_path,
                &smoke,
                ReadyDirectChatStatus::disabled(),
                &context_limit_plan_value(),
                prefill_batch,
            )
        );
        return ExitCode::SUCCESS;
    }

    println!("==> Running direct 1B chat turn");
    let chat_code = run_chat(
        model_path,
        &chat_prompt,
        chat_temp.expect("chat temp should be parsed when chat is on"),
        chat_tokens.expect("chat tokens should be parsed when chat is on"),
        smoke.model_source,
        true,
    );
    if chat_code == ExitCode::SUCCESS {
        println!("ready_1b_status: ok");
        println!(
            "json: {}",
            ready_1b_status_json(
                model_path,
                &smoke,
                ReadyDirectChatStatus {
                    enabled: true,
                    prompt: Some(chat_prompt.as_str()),
                    tokens: chat_tokens,
                    temp: chat_temp,
                },
                &context_limit_plan_value(),
                prefill_batch,
            )
        );
    }
    chat_code
}

fn print_evidence_1b_dry_run(parsed: &Evidence1BArgs) -> ExitCode {
    if let Err(err) = validate_runtime_generation_env() {
        eprintln!("{err}");
        return ExitCode::from(2);
    }

    let model_path = Path::new(&parsed.model_path);
    println!("NanoCamelid Llama 3.2 1B evidence dry run");
    println!("workspace: {}", parsed.workspace);
    println!("q4_model: {}", parsed.q4_model_path);
    println!("q4_exists: {}", Path::new(&parsed.q4_model_path).is_file());
    println!("q8_model: {}", parsed.q8_model_path);
    println!("q8_exists: {}", Path::new(&parsed.q8_model_path).is_file());
    println!("selected_source: {}", parsed.model_source);
    println!("model: {}", model_path.display());
    println!("model_exists: {}", model_path.is_file());
    println!(
        "quantization: {}",
        llama32_1b_quantization_for_path(model_path)
    );
    println!("shape_audit: enabled");
    println!("context_limit: {}", context_limit_plan_value());
    println!("smoke_kind: {}", parsed.smoke.kind.label());
    println!("smoke_prompt: {}", parsed.smoke.prompt);
    println!("smoke_tokens: {}", parsed.smoke.max_tokens);
    println!("prefill_batch: {}", parsed.prefill_batch);
    println!(
        "context_pack_caps: {}",
        join_usize_values(&parsed.context_packs, " ")
    );
    println!("prefill_prompt: {}", parsed.prefill.prompt);
    println!("prefill_tokens: {}", parsed.prefill.max_tokens);
    println!("prefill_temp: {}", parsed.prefill.temp);
    println!(
        "prefill_batches: {}",
        join_usize_values(&parsed.prefill.batches, " ")
    );
    println!("status_on_success: evidence_1b_status: ok");
    println!("json_on_success: {}", evidence_1b_status_json(parsed));
    println!("model_command: {}", evidence_model_command(parsed));
    println!(
        "ready_command: {}",
        evidence_ready_no_chat_command(parsed, context_limit_env_value().as_deref())
    );
    for cap in &parsed.context_packs {
        println!(
            "context_{cap}_command: {}",
            evidence_context_pack_command(parsed, *cap)
        );
    }
    println!(
        "prefill_bench_command: {}",
        evidence_prefill_bench_command(parsed, context_limit_env_value().as_deref())
    );
    ExitCode::SUCCESS
}

fn run_evidence_1b(parsed: Evidence1BArgs) -> ExitCode {
    if let Err(err) = validate_runtime_generation_env() {
        eprintln!("{err}");
        return ExitCode::from(2);
    }
    if !Path::new(&parsed.model_path).is_file() {
        eprintln!(
            "{}",
            llama32_1b_model_not_found_message(Path::new(&parsed.model_path))
        );
        eprintln!(
            "Run `nanocamelid evidence 1b --dry-run` to print the resolved evidence commands without loading a model."
        );
        return ExitCode::from(2);
    }

    println!("NanoCamelid Llama 3.2 1B evidence bundle");
    println!("workspace: {}", parsed.workspace);
    println!("selected_source: {}", parsed.model_source);
    println!("model: {}", parsed.model_path);
    println!(
        "quantization: {}",
        llama32_1b_quantization_for_path(Path::new(&parsed.model_path))
    );
    println!("shape_audit: enabled");
    println!("context_limit: {}", context_limit_plan_value());
    println!("smoke_kind: {}", parsed.smoke.kind.label());
    println!("smoke_prompt: {}", parsed.smoke.prompt);
    println!("smoke_tokens: {}", parsed.smoke.max_tokens);
    println!("prefill_batch: {}", parsed.prefill_batch);
    println!(
        "context_pack_caps: {}",
        join_usize_values(&parsed.context_packs, " ")
    );
    println!("prefill_prompt: {}", parsed.prefill.prompt);
    println!("prefill_tokens: {}", parsed.prefill.max_tokens);
    println!("prefill_temp: {}", parsed.prefill.temp);
    println!(
        "prefill_batches: {}",
        join_usize_values(&parsed.prefill.batches, " ")
    );

    println!("==> Auditing selected 1B model");
    let model_code = run_model_1b_audit(Model1BAuditArgs {
        workspace: parsed.workspace.clone(),
        q4_model_path: parsed.q4_model_path.clone(),
        q8_model_path: parsed.q8_model_path.clone(),
        model_path: parsed.model_path.clone(),
        model_source: parsed.model_source,
        dry_run: false,
    });
    if model_code != ExitCode::SUCCESS {
        return model_code;
    }

    println!("==> Running readiness gate without final direct chat");
    let ready_code = run_ready_1b(Ready1BArgs {
        smoke: Smoke1BArgs {
            kind: parsed.smoke.kind,
            model_path: parsed.model_path.clone(),
            model_source: parsed.model_source,
            prompt: parsed.smoke.prompt.clone(),
            max_tokens: parsed.smoke.max_tokens,
            dry_run: false,
        },
        chat_enabled_override: Some(false),
        chat_prompt_override: None,
        chat_tokens_override: None,
        dry_run: false,
    });
    if ready_code != ExitCode::SUCCESS {
        return ready_code;
    }

    println!("==> Running context-pack smoke gate");
    for cap in &parsed.context_packs {
        println!();
        println!("==> Running with {CONTEXT_LIMIT_ENV}={cap}");
        match run_evidence_context_pack_smoke(&parsed, *cap) {
            Ok(0) => {}
            Ok(status) => return ExitCode::from(status as u8),
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        }
    }

    println!("==> Running prefill batch sweep");
    let final_json = evidence_1b_status_json(&parsed);
    let prefill_code = run_bench_1b_prefill(parsed.prefill);
    if prefill_code == ExitCode::SUCCESS {
        println!("evidence_1b_status: ok");
        println!("json: {final_json}");
    }
    prefill_code
}

fn run_evidence_context_pack_smoke(parsed: &Evidence1BArgs, cap: usize) -> Result<i32, String> {
    let current_exe =
        env::current_exe().map_err(|err| format!("failed to resolve current executable: {err}"))?;
    let mut command = Command::new(current_exe);
    command
        .arg("smoke")
        .arg("1b")
        .arg(&parsed.model_path)
        .arg(parsed.smoke.kind.label())
        .arg(&parsed.smoke.prompt)
        .arg(parsed.smoke.max_tokens.to_string())
        .env(CONTEXT_LIMIT_ENV, cap.to_string());
    let output = command
        .output()
        .map_err(|err| format!("failed to run 1B context-pack smoke for cap {cap}: {err}"))?;
    io::stderr().write_all(&output.stderr).ok();
    print!("{}", String::from_utf8_lossy(&output.stdout));
    Ok(output.status.code().unwrap_or(1))
}

fn evidence_model_command(parsed: &Evidence1BArgs) -> String {
    shell_command(&["nanocamelid", "model", "1b", &parsed.model_path])
}

fn evidence_ready_no_chat_command(parsed: &Evidence1BArgs, context_limit: Option<&str>) -> String {
    let max_tokens = parsed.smoke.max_tokens.to_string();
    let args = [
        "nanocamelid",
        "ready",
        "1b",
        &parsed.model_path,
        parsed.smoke.kind.label(),
        &parsed.smoke.prompt,
        &max_tokens,
        "--no-chat",
    ];
    shell_command_with_optional_runtime_env(
        &args,
        context_limit,
        prefill_batch_env_value().as_deref(),
    )
}

fn evidence_context_pack_command(parsed: &Evidence1BArgs, cap: usize) -> String {
    let cap = cap.to_string();
    let max_tokens = parsed.smoke.max_tokens.to_string();
    let args = [
        "nanocamelid",
        "smoke",
        "1b",
        &parsed.model_path,
        parsed.smoke.kind.label(),
        &parsed.smoke.prompt,
        &max_tokens,
    ];
    shell_command_with_optional_runtime_env(&args, Some(&cap), prefill_batch_env_value().as_deref())
}

fn evidence_prefill_bench_command(parsed: &Evidence1BArgs, context_limit: Option<&str>) -> String {
    evidence_prefill_bench_command_with_env(
        parsed,
        context_limit,
        prefill_batch_env_value().as_deref(),
    )
}

fn evidence_prefill_bench_command_with_env(
    parsed: &Evidence1BArgs,
    context_limit: Option<&str>,
    prefill_batch: Option<&str>,
) -> String {
    let max_tokens = parsed.prefill.max_tokens.to_string();
    let batches = join_usize_values(&parsed.prefill.batches, ",");
    let args = [
        "nanocamelid",
        "bench",
        "1b",
        &parsed.model_path,
        &parsed.prefill.prompt,
        &max_tokens,
        &parsed.prefill.temp,
        &batches,
    ];
    shell_command_with_optional_runtime_env(&args, context_limit, prefill_batch)
}

fn evidence_1b_status_json(parsed: &Evidence1BArgs) -> String {
    format!(
        "{{\"target\":\"llama32-1b\",\"status\":\"ok\",\"model\":{},\"selected_source\":{},\"quantization\":{},\"shape\":\"llama32_1b\",\"shape_ready\":true,\"context_limit\":{},\"ready_no_chat\":true,\"context_pack\":true,\"prefill_bench\":true,\"smoke_prompt\":{},\"smoke_kind\":\"{}\",\"smoke_tokens\":{},\"prefill_batch\":{},\"context_pack_caps\":[{}],\"prefill_prompt\":{},\"prefill_tokens\":{},\"prefill_temp\":{},\"prefill_batches\":[{}]}}",
        json_string(&parsed.model_path),
        json_string(parsed.model_source),
        json_string(llama32_1b_quantization_for_path(Path::new(
            &parsed.model_path
        ))),
        json_string(&context_limit_plan_value()),
        json_string(&parsed.smoke.prompt),
        parsed.smoke.kind.label(),
        parsed.smoke.max_tokens,
        parsed.prefill_batch,
        join_usize_values(&parsed.context_packs, ","),
        json_string(&parsed.prefill.prompt),
        parsed.prefill.max_tokens,
        parsed.prefill.temp,
        join_usize_values(&parsed.prefill.batches, ","),
    )
}

fn join_usize_values(values: &[usize], separator: &str) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(separator)
}

#[derive(Clone, Copy)]
struct ReadyDirectChatStatus<'a> {
    enabled: bool,
    prompt: Option<&'a str>,
    tokens: Option<usize>,
    temp: Option<f32>,
}

impl ReadyDirectChatStatus<'_> {
    fn disabled() -> Self {
        Self {
            enabled: false,
            prompt: None,
            tokens: None,
            temp: None,
        }
    }
}

fn ready_1b_status_json(
    model_path: &Path,
    smoke: &Smoke1BArgs,
    direct_chat: ReadyDirectChatStatus<'_>,
    context_limit: &str,
    prefill_batch: usize,
) -> String {
    let chat_prompt = if direct_chat.enabled {
        json_optional_string(direct_chat.prompt)
    } else {
        "null".to_owned()
    };
    let chat_tokens = direct_chat
        .tokens
        .map(|tokens| tokens.to_string())
        .unwrap_or_else(|| "null".to_owned());
    let chat_temp = if direct_chat.enabled {
        direct_chat
            .temp
            .map(|temp| temp.to_string())
            .unwrap_or_else(|| "null".to_owned())
    } else {
        "null".to_owned()
    };
    format!(
        "{{\"target\":\"llama32-1b\",\"status\":\"ok\",\"model\":{},\"selected_source\":{},\"quantization\":{},\"probe\":true,\"shape\":\"llama32_1b\",\"shape_ready\":true,\"context_limit\":{},\"smoke_prompt\":{},\"smoke_kind\":\"{}\",\"smoke_tokens\":{},\"prefill_batch\":{},\"direct_chat\":{},\"chat_prompt\":{},\"chat_tokens\":{},\"chat_temp\":{}}}",
        json_string(&model_path.display().to_string()),
        json_string(smoke.model_source),
        json_string(llama32_1b_quantization_for_path(model_path)),
        json_string(context_limit),
        json_string(&smoke.prompt),
        smoke.kind.label(),
        smoke.max_tokens,
        prefill_batch,
        direct_chat.enabled,
        chat_prompt,
        chat_tokens,
        chat_temp
    )
}

fn model_1b_status_json(model_path: &Path, model_source: &str) -> String {
    format!(
        "{{\"target\":\"llama32-1b\",\"status\":\"ok\",\"model\":{},\"selected_source\":{},\"quantization\":{},\"shape\":\"llama32_1b\",\"shape_ready\":true}}",
        json_string(&model_path.display().to_string()),
        json_string(model_source),
        json_string(llama32_1b_quantization_for_path(model_path)),
    )
}

fn inspect_1b_status_json(model_path: &Path, model_source: &str) -> String {
    format!(
        "{{\"target\":\"llama32-1b\",\"command\":\"inspect\",\"status\":\"ok\",\"model\":{},\"selected_source\":{},\"quantization\":{},\"shape\":\"llama32_1b\",\"shape_ready\":true}}",
        json_string(&model_path.display().to_string()),
        json_string(model_source),
        json_string(llama32_1b_quantization_for_path(model_path)),
    )
}

fn smoke_1b_status_json(
    model_path: &Path,
    smoke: &Smoke1BArgs,
    context_limit: &str,
    prefill_batch: usize,
) -> String {
    format!(
        "{{\"target\":\"llama32-1b\",\"status\":\"ok\",\"model\":{},\"selected_source\":{},\"quantization\":{},\"shape\":\"llama32_1b\",\"shape_ready\":true,\"context_limit\":{},\"smoke_prompt\":{},\"smoke_kind\":\"{}\",\"smoke_tokens\":{},\"prefill_batch\":{}}}",
        json_string(&model_path.display().to_string()),
        json_string(smoke.model_source),
        json_string(llama32_1b_quantization_for_path(model_path)),
        json_string(context_limit),
        json_string(&smoke.prompt),
        smoke.kind.label(),
        smoke.max_tokens,
        prefill_batch,
    )
}

fn llama32_1b_quantization_for_path(model_path: &Path) -> &'static str {
    match model_path.file_name().and_then(|name| name.to_str()) {
        Some(LLAMA32_1B_Q4_MODEL) => "q4_0",
        Some(LLAMA32_1B_Q8_MODEL) => "q8_0",
        _ => "unknown",
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn json_optional_string(value: Option<&str>) -> String {
    value.map(json_string).unwrap_or_else(|| "null".to_owned())
}

fn smoke_plan_command(target: &str, model_path: &Path, parsed: &Smoke1BArgs) -> String {
    let context_limit = context_limit_env_value();
    let prefill_batch = prefill_batch_env_value();
    smoke_plan_command_with_env(
        target,
        model_path,
        parsed,
        context_limit.as_deref(),
        prefill_batch.as_deref(),
    )
}

#[cfg(test)]
fn smoke_plan_command_with_context(
    target: &str,
    model_path: &Path,
    parsed: &Smoke1BArgs,
    context_limit: Option<&str>,
) -> String {
    smoke_plan_command_with_env(target, model_path, parsed, context_limit, None)
}

fn smoke_plan_command_with_env(
    target: &str,
    model_path: &Path,
    parsed: &Smoke1BArgs,
    context_limit: Option<&str>,
    prefill_batch: Option<&str>,
) -> String {
    shell_command_with_optional_runtime_env(
        &[
            "nanocamelid",
            "smoke",
            target,
            &model_path.display().to_string(),
            parsed.kind.label(),
            &parsed.prompt,
            &parsed.max_tokens.to_string(),
        ],
        context_limit,
        prefill_batch,
    )
}

fn run_smoke_1b_gate(model_path: &Path, parsed: &Smoke1BArgs) -> ExitCode {
    let prefill_batch = match validate_runtime_generation_env() {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(2);
        }
    };

    if !model_path.is_file() {
        eprintln!("{}", llama32_1b_model_not_found_message(model_path));
        return ExitCode::from(2);
    }

    println!("==> Auditing 1B model shape: {}", model_path.display());
    let model_audit_code = audit_llama32_1b_model_shape(model_path);
    if model_audit_code != ExitCode::SUCCESS {
        return model_audit_code;
    }

    println!("==> Running 1B {} smoke gate", parsed.kind.label());
    let smoke_code = match parsed.kind {
        SmokeKind::Q8Model => smoke_q8_model(model_path, &parsed.prompt, parsed.max_tokens),
        SmokeKind::Q8Chat => smoke_q8_chat(model_path, &parsed.prompt, parsed.max_tokens),
    };
    if smoke_code == ExitCode::SUCCESS {
        println!("smoke_1b_status: ok");
        println!(
            "json: {}",
            smoke_1b_status_json(
                model_path,
                parsed,
                &context_limit_plan_value(),
                prefill_batch
            )
        );
    }
    smoke_code
}

fn context_limit_plan_value() -> String {
    context_limit_env_value().unwrap_or_else(|| "unset".to_owned())
}

fn validate_context_limit_env() -> Result<(), String> {
    parse_context_limit_env().map(|_| ())
}

fn validate_runtime_generation_env() -> Result<usize, String> {
    validate_context_limit_env()?;
    prefill_batch_size_from_env().map_err(str::to_owned)
}

fn shell_command(args: &[&str]) -> String {
    args.iter()
        .map(|arg| shell_quote_arg(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
fn shell_command_with_env(args: &[&str], env: &[(&str, &str)]) -> String {
    let mut parts = env
        .iter()
        .map(|(key, value)| format!("{key}={}", shell_quote_arg(value)))
        .collect::<Vec<_>>();
    parts.push(shell_command(args));
    parts.join(" ")
}

fn shell_command_with_owned_env(args: &[&str], env: &[(&str, String)]) -> String {
    let mut parts = env
        .iter()
        .map(|(key, value)| format!("{key}={}", shell_quote_arg(value)))
        .collect::<Vec<_>>();
    parts.push(shell_command(args));
    parts.join(" ")
}

fn context_limited_shell_command(args: &[&str]) -> String {
    let context_limit = context_limit_env_value();
    let prefill_batch = prefill_batch_env_value();
    shell_command_with_optional_runtime_env(
        args,
        context_limit.as_deref(),
        prefill_batch.as_deref(),
    )
}

fn shell_command_with_optional_runtime_env(
    args: &[&str],
    context_limit: Option<&str>,
    prefill_batch: Option<&str>,
) -> String {
    let mut env = Vec::with_capacity(2);
    if let Some(context_limit) = context_limit {
        env.push((CONTEXT_LIMIT_ENV, context_limit.to_owned()));
    }
    if let Some(prefill_batch) = prefill_batch {
        env.push((PREFILL_BATCH_ENV, prefill_batch.to_owned()));
    }

    if env.is_empty() {
        shell_command(args)
    } else {
        shell_command_with_owned_env(args, &env)
    }
}

fn context_limit_env_value() -> Option<String> {
    env::var(CONTEXT_LIMIT_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn prefill_batch_env_value() -> Option<String> {
    env::var(PREFILL_BATCH_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn shell_quote_arg(arg: &str) -> String {
    if !arg.is_empty()
        && arg
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'.' | b'_' | b'-'))
    {
        return arg.to_owned();
    }

    format!("'{}'", arg.replace('\'', "'\\''"))
}

#[derive(Debug)]
struct Q8ModelSmokeReport {
    prompt_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    generated_text: String,
    max_logit_delta: f32,
    kernel_selector: q8::Q8DotKernelSelector,
    chat_renderer: Option<&'static str>,
    chat_template_format: Option<&'static str>,
}

#[derive(Debug)]
struct Q8SmokePromptPlan {
    prompt_tokens: Vec<u32>,
    chat_renderer: Option<&'static str>,
    chat_template_format: Option<&'static str>,
}

fn run_q8_model_smoke(
    model_path: &Path,
    prompt: &str,
    max_tokens: usize,
) -> Result<Q8ModelSmokeReport, String> {
    run_q8_smoke_with_prompt_plan(model_path, max_tokens, |tokenizer| {
        let prompt_tokens = tokenizer
            .encode(prompt, true, true)
            .map_err(|err| format!("failed to tokenize prompt: {err}"))?;
        Ok(Q8SmokePromptPlan {
            prompt_tokens,
            chat_renderer: None,
            chat_template_format: None,
        })
    })
}

fn run_q8_chat_smoke(
    model_path: &Path,
    prompt: &str,
    max_tokens: usize,
) -> Result<Q8ModelSmokeReport, String> {
    run_q8_smoke_with_prompt_plan(model_path, max_tokens, |tokenizer| {
        let rendered = tokenizer.render_chat_prompt(&[tokenizer::ChatMessage {
            role: "user",
            content: prompt,
        }]);
        let prompt_tokens = tokenizer
            .encode(&rendered.text, rendered.add_special, rendered.parse_special)
            .map_err(|err| format!("failed to tokenize chat prompt: {err}"))?;
        Ok(Q8SmokePromptPlan {
            prompt_tokens,
            chat_renderer: Some(rendered.renderer),
            chat_template_format: tokenizer.chat_template_format(),
        })
    })
}

fn run_q8_smoke_with_prompt_plan<F>(
    model_path: &Path,
    max_tokens: usize,
    prompt_builder: F,
) -> Result<Q8ModelSmokeReport, String>
where
    F: FnOnce(&tokenizer::Tokenizer) -> Result<Q8SmokePromptPlan, String>,
{
    let gguf = gguf::read_file(model_path).map_err(|err| format!("failed to read GGUF: {err}"))?;
    let mut config = model::LlamaModelConfig::from_gguf(&gguf)
        .map_err(|err| format!("failed to parse config: {err}"))?;
    apply_context_limit(&mut config)?;
    let prefill_batch = prefill_batch_size_from_env().map_err(str::to_owned)?;
    let tokenizer = tokenizer::Tokenizer::from_gguf(&gguf)
        .map_err(|err| format!("failed to load tokenizer: {err}"))?;
    let weights = model::LlamaWeights::load(model_path, &config, &gguf)
        .map_err(|err| format!("failed to load weights: {err}"))?;
    let prompt_plan = prompt_builder(&tokenizer)?;
    let prompt_tokens = prompt_plan.prompt_tokens;
    if prompt_tokens.is_empty() {
        return Err("prompt tokenized to an empty sequence".to_owned());
    }
    validate_generation_budget(prompt_tokens.len(), max_tokens, config.context_length)?;

    let selected = q8::Q8DotKernelSelector::from_env();
    let scalar = q8::Q8DotKernelSelector {
        requested: Some(q8::Q8DotKernel::Scalar),
        selected: q8::Q8DotKernel::Scalar,
        fallback_reason: None,
    };
    let rope_scaling = rope_scaling_from_gguf(&gguf);
    let scalar_options = inference::LlamaRuntimeOptions {
        q8_selector: scalar,
        rope_scaling,
        compute_logits: true,
    };
    let selected_options = inference::LlamaRuntimeOptions {
        q8_selector: selected,
        rope_scaling,
        compute_logits: true,
    };

    let mut scalar_cache =
        inference::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut selected_cache =
        inference::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut scalar_ws = inference::LlamaWorkspace::new(&config);
    let mut selected_ws = inference::LlamaWorkspace::new(&config);
    let mut selected_batch_ws = inference::LlamaBatchWorkspace::new(&config, prefill_batch);
    let mut max_logit_delta = 0.0_f32;
    let mut pos = 0;

    for &token in &prompt_tokens {
        inference::forward_pass(
            token as usize,
            pos,
            &config,
            &weights,
            &mut scalar_cache,
            &mut scalar_ws,
            scalar_options,
        );
        pos += 1;
    }

    let mut selected_pos = 0;
    let mut selected_context_tokens = Vec::with_capacity(prompt_tokens.len());
    if let Some((&last_token, prefix_tokens)) = prompt_tokens.split_last() {
        prefill_tokens(
            prefix_tokens,
            &config,
            &weights,
            PrefillTokenState {
                cache: &mut selected_cache,
                ws: &mut selected_ws,
                batch_ws: Some(&mut selected_batch_ws),
                context_tokens: &mut selected_context_tokens,
                pos: &mut selected_pos,
            },
            selected_options,
        );
        inference::forward_pass(
            last_token as usize,
            selected_pos,
            &config,
            &weights,
            &mut selected_cache,
            &mut selected_ws,
            selected_options,
        );
        selected_context_tokens.push(last_token);
        selected_pos += 1;
    }
    max_logit_delta = max_logit_delta.max(max_abs_delta(&scalar_ws.logits, &selected_ws.logits));
    debug_assert_eq!(pos, selected_pos);

    let mut generated_tokens = Vec::new();
    for _ in 0..max_tokens {
        let scalar_next = inference::sample_logits(&scalar_ws.logits, 0.0);
        let selected_next = inference::sample_logits(&selected_ws.logits, 0.0);
        if scalar_next != selected_next {
            return Err(format!(
                "greedy token mismatch at generated index {}: scalar={scalar_next}, selected={selected_next}",
                generated_tokens.len()
            ));
        }
        if is_generation_stop_token(&tokenizer.special, scalar_next as u32)
            || pos >= config.context_length
        {
            break;
        }

        generated_tokens.push(scalar_next as u32);
        inference::forward_pass(
            scalar_next,
            pos,
            &config,
            &weights,
            &mut scalar_cache,
            &mut scalar_ws,
            scalar_options,
        );
        inference::forward_pass(
            selected_next,
            pos,
            &config,
            &weights,
            &mut selected_cache,
            &mut selected_ws,
            selected_options,
        );
        max_logit_delta =
            max_logit_delta.max(max_abs_delta(&scalar_ws.logits, &selected_ws.logits));
        pos += 1;
    }

    if max_logit_delta > 0.0001 {
        return Err(format!(
            "scalar vs selected max logit delta {max_logit_delta:.8} exceeds tolerance"
        ));
    }

    let generated_text = tokenizer
        .decode(&generated_tokens, true)
        .map_err(|err| format!("failed to decode generated tokens: {err}"))?;

    Ok(Q8ModelSmokeReport {
        prompt_tokens,
        generated_tokens,
        generated_text,
        max_logit_delta,
        kernel_selector: selected,
        chat_renderer: prompt_plan.chat_renderer,
        chat_template_format: prompt_plan.chat_template_format,
    })
}

fn print_q8_smoke_report(title: &str, model_path: &Path, report: &Q8ModelSmokeReport) {
    println!("{title}");
    println!("path: {}", model_path.display());
    if let Some(renderer) = report.chat_renderer {
        println!("chat_renderer: {renderer}");
        println!(
            "chat_template_format: {}",
            report.chat_template_format.unwrap_or("none")
        );
    }
    println!("prompt_tokens: {:?}", report.prompt_tokens);
    println!(
        "kernel_selector_requested: {}",
        report
            .kernel_selector
            .requested
            .map(q8::Q8DotKernel::name)
            .unwrap_or("default")
    );
    println!(
        "kernel_selector_selected: {}",
        report.kernel_selector.selected.name()
    );
    if let Some(reason) = report.kernel_selector.fallback_reason {
        println!("kernel_selector_fallback: {reason}");
    }
    println!("max_logit_delta: {:.8}", report.max_logit_delta);
    println!("generated_tokens: {:?}", report.generated_tokens);
    println!("generated_text: {:?}", report.generated_text);
    println!("status: ok");
}

fn is_generation_stop_token(special: &tokenizer::SpecialTokens, token_id: u32) -> bool {
    special.eog.contains(&token_id)
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

fn max_abs_delta(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}

fn validate_generation_budget(
    prompt_token_count: usize,
    requested_generation_tokens: usize,
    context_length: usize,
) -> Result<(), String> {
    if prompt_token_count > context_length {
        return Err(format!(
            "prompt requires {prompt_token_count} tokens but model context length is {context_length}"
        ));
    }

    let remaining_tokens = context_length - prompt_token_count;
    if requested_generation_tokens > remaining_tokens {
        return Err(format!(
            "prompt uses {prompt_token_count} of {context_length} context tokens; requested {requested_generation_tokens} generation tokens but only {remaining_tokens} remain"
        ));
    }

    Ok(())
}

fn apply_context_limit(config: &mut model::LlamaModelConfig) -> Result<(), String> {
    let Some(limit) = parse_context_limit_env()? else {
        return Ok(());
    };
    config.context_length = config.context_length.min(limit);
    Ok(())
}

fn parse_context_limit_env() -> Result<Option<usize>, String> {
    let Some(raw_limit) = env::var(CONTEXT_LIMIT_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(None);
    };

    let limit = raw_limit
        .parse::<usize>()
        .map_err(|_| format!("{CONTEXT_LIMIT_ENV} must be a positive integer: {raw_limit}"))?;
    if limit == 0 {
        return Err(format!("{CONTEXT_LIMIT_ENV} must be greater than zero"));
    }
    Ok(Some(limit))
}

fn cpu_model(cpuinfo: &str) -> Option<&str> {
    cpuinfo.lines().find_map(|line| {
        value_after_colon(line, "Hardware").or_else(|| value_after_colon(line, "model name"))
    })
}

fn cpu_features(cpuinfo: &str) -> Option<&str> {
    cpuinfo.lines().find_map(|line| {
        value_after_colon(line, "Features").or_else(|| value_after_colon(line, "flags"))
    })
}

fn value_after_colon<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let (candidate, value) = line.split_once(':')?;
    (candidate.trim() == key).then(|| value.trim())
}

fn device_model(model: &str) -> Option<&str> {
    let trimmed = model.trim_matches(char::from(0)).trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

#[derive(Debug)]
struct GenerationPromptPlan {
    text: String,
    add_special: bool,
    parse_special: bool,
    renderer: Option<&'static str>,
    template_format: Option<&'static str>,
}

fn run_generation_command<F>(parsed: &GenerateArgs, run: F) -> ExitCode
where
    F: FnOnce(&GenerateArgs) -> ExitCode,
{
    if let Err(err) = validate_runtime_generation_env() {
        eprintln!("{err}");
        return ExitCode::from(2);
    }
    if let Some(exit_code) = audit_direct_1b_launch(&parsed.model_path, parsed.audit_1b_shape) {
        return exit_code;
    }
    run(parsed)
}

fn run_tui_command(parsed: &TuiArgs) -> ExitCode {
    if let Err(err) = validate_runtime_generation_env() {
        eprintln!("{err}");
        return ExitCode::from(2);
    }
    if let Some(exit_code) = audit_direct_1b_launch(&parsed.model_path, parsed.audit_1b_shape) {
        return exit_code;
    }
    run_chat_tui(
        Path::new(&parsed.model_path),
        parsed.temp,
        parsed.max_tokens,
    )
}

fn audit_direct_1b_launch(model_path: &str, audit_1b_shape: bool) -> Option<ExitCode> {
    if !audit_1b_shape {
        return None;
    }

    let model_path = Path::new(model_path);
    if !model_path.is_file() {
        eprintln!("{}", llama32_1b_model_not_found_message(model_path));
        return Some(ExitCode::from(2));
    }

    println!("==> Auditing 1B model shape: {}", model_path.display());
    let audit_code = audit_llama32_1b_model_shape(model_path);
    (audit_code != ExitCode::SUCCESS).then_some(audit_code)
}

fn run_generation(
    model_path: &Path,
    prompt: &str,
    temp: f32,
    max_tokens: usize,
    model_source: &str,
    audit_1b_shape: bool,
) -> ExitCode {
    run_generation_with_prompt_builder(
        "generate",
        model_path,
        temp,
        max_tokens,
        |_tokenizer| {
            Ok(GenerationPromptPlan {
                text: prompt.to_owned(),
                add_special: true,
                parse_special: true,
                renderer: None,
                template_format: None,
            })
        },
        model_source,
        audit_1b_shape,
    )
}

fn run_chat(
    model_path: &Path,
    prompt: &str,
    temp: f32,
    max_tokens: usize,
    model_source: &str,
    audit_1b_shape: bool,
) -> ExitCode {
    run_generation_with_prompt_builder(
        "chat",
        model_path,
        temp,
        max_tokens,
        |tokenizer| {
            let rendered = tokenizer.render_chat_prompt(&[tokenizer::ChatMessage {
                role: "user",
                content: prompt,
            }]);
            Ok(GenerationPromptPlan {
                text: rendered.text,
                add_special: rendered.add_special,
                parse_special: rendered.parse_special,
                renderer: Some(rendered.renderer),
                template_format: tokenizer.chat_template_format(),
            })
        },
        model_source,
        audit_1b_shape,
    )
}

#[derive(Debug, Clone)]
struct ChatTurn {
    role: String,
    content: String,
}

#[derive(Debug)]
struct ChatTurnReport {
    input_tokens: usize,
    output_tokens: usize,
    ttft_secs: Option<f64>,
    elapsed_secs: f64,
}

struct ChatSession {
    cache: inference::LlamaKvCache,
    ws: inference::LlamaWorkspace,
    batch_ws: inference::LlamaBatchWorkspace,
    context_tokens: Vec<u32>,
    pos: usize,
}

#[derive(Debug, Clone)]
struct TuiSettings {
    temp: f32,
    max_tokens: usize,
    system_prompt: Option<String>,
}

#[derive(Debug, PartialEq)]
enum TuiCommand {
    Help,
    ModelShow,
    ModelLoad(String),
    Models,
    Clear,
    Status,
    History,
    Save(String),
    TempShow,
    TempSet(f32),
    TokensShow,
    TokensSet(usize),
    SystemShow,
    SystemSet(String),
    SystemClear,
    Trim(usize),
    Exit,
    Unknown(String),
}

impl ChatSession {
    fn new(config: &model::LlamaModelConfig) -> Self {
        Self {
            cache: inference::LlamaKvCache::new(
                config.block_count,
                config.context_length,
                config.kv_width,
            ),
            ws: inference::LlamaWorkspace::new(config),
            batch_ws: inference::LlamaBatchWorkspace::new(config, prefill_batch_size()),
            context_tokens: Vec::new(),
            pos: 0,
        }
    }

    fn reset(&mut self, config: &model::LlamaModelConfig) {
        *self = Self::new(config);
    }
}

struct ChatGenerationEnv<'a> {
    config: &'a model::LlamaModelConfig,
    weights: &'a model::LlamaWeights,
    tokenizer: &'a tokenizer::Tokenizer,
    runtime_options: inference::LlamaRuntimeOptions,
    draft: Option<&'a mut speculative::SpeculativeContext>,
}

struct TuiLoadedModel {
    model_path: PathBuf,
    config: model::LlamaModelConfig,
    weights: model::LlamaWeights,
    tokenizer: tokenizer::Tokenizer,
    runtime_options: inference::LlamaRuntimeOptions,
    model_name: String,
    renderer: String,
    load_secs: f64,
    draft: Option<speculative::SpeculativeContext>,
}

fn load_tui_model(
    model_path: &Path,
    selector: q8::Q8DotKernelSelector,
) -> Result<TuiLoadedModel, String> {
    println!("Loading GGUF file: {}...", model_path.display());
    let gguf = gguf::read_file(model_path).map_err(|e| format!("failed to read GGUF: {e}"))?;

    let mut config = model::LlamaModelConfig::from_gguf(&gguf)
        .map_err(|e| format!("failed to parse config: {e}"))?;
    apply_context_limit(&mut config)?;
    let tokenizer = tokenizer::Tokenizer::from_gguf(&gguf)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;

    println!("Loading model weights into memory...");
    let started_load = std::time::Instant::now();
    let weights = model::LlamaWeights::load(model_path, &config, &gguf)
        .map_err(|e| format!("failed to load weights: {e}"))?;

    let runtime_options = runtime_options_from_gguf(&gguf, selector);
    let model_name = gguf
        .metadata_string("general.name")
        .map(str::to_owned)
        .or_else(|| {
            model_path
                .file_name()
                .map(|name| name.to_string_lossy().into())
        })
        .unwrap_or_else(|| model_path.display().to_string());
    let renderer = tokenizer.render_chat_prompt(&[]).renderer.to_owned();

    let draft = if let Ok(draft_path_str) = std::env::var("NANOCAMELID_DRAFT_GGUF") {
        if !draft_path_str.is_empty() {
            let draft_path = Path::new(&draft_path_str);
            println!("Loading draft GGUF file: {}...", draft_path.display());
            let draft_ctx = speculative::SpeculativeContext::load(draft_path, runtime_options)?;
            if draft_ctx.config.vocab_size != config.vocab_size {
                return Err(format!(
                    "Vocabulary size mismatch: Target has {}, Draft has {}",
                    config.vocab_size, draft_ctx.config.vocab_size
                ));
            }
            Some(draft_ctx)
        } else {
            None
        }
    } else {
        None
    };

    Ok(TuiLoadedModel {
        model_path: model_path.to_path_buf(),
        config,
        weights,
        tokenizer,
        runtime_options,
        model_name,
        renderer,
        load_secs: started_load.elapsed().as_secs_f64(),
        draft,
    })
}

fn run_chat_tui(model_path: &Path, temp: f32, max_tokens: usize) -> ExitCode {
    if let Err(err) = prefill_batch_size_from_env() {
        eprintln!("{err}");
        return ExitCode::from(2);
    }

    let selector = q8::Q8DotKernelSelector::from_env();
    let governor_recommendation = cpu_governor_recommendation(
        read_trimmed("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").as_deref(),
    );
    let mut loaded = match load_tui_model(model_path, selector) {
        Ok(loaded) => loaded,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    };
    let mut history = Vec::<ChatTurn>::new();
    let mut session = ChatSession::new(&loaded.config);
    let mut total_in = 0usize;
    let mut total_out = 0usize;
    let mut settings = TuiSettings {
        temp,
        max_tokens,
        system_prompt: None,
    };

    print_tui_banner(TuiBanner {
        model_name: &loaded.model_name,
        model_path: &loaded.model_path,
        config: &loaded.config,
        kernel: selector.selected.name(),
        renderer: &loaded.renderer,
        settings: &settings,
        load_secs: loaded.load_secs,
        governor_recommendation,
    });

    let stdin = io::stdin();
    loop {
        print!("{}nano>{} ", ansi::INPUT, ansi::RESET);
        if io::stdout().flush().is_err() {
            return ExitCode::FAILURE;
        }

        let mut input = String::new();
        match stdin.read_line(&mut input) {
            Ok(0) => break,
            Ok(_) => {}
            Err(err) => {
                eprintln!("Failed to read input: {err}");
                return ExitCode::FAILURE;
            }
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input.starts_with('/') {
            match parse_tui_command(input) {
                TuiCommand::Exit => break,
                TuiCommand::Help => {
                    print_tui_commands();
                }
                TuiCommand::ModelShow => {
                    println!(
                        "{}current model{} {}",
                        ansi::LABEL,
                        ansi::RESET,
                        loaded.model_path.display()
                    );
                }
                TuiCommand::ModelLoad(next_model) => {
                    let next_model_path = PathBuf::from(next_model);
                    match load_tui_model(&next_model_path, selector) {
                        Ok(next_loaded) => {
                            loaded = next_loaded;
                            history.clear();
                            session = ChatSession::new(&loaded.config);
                            total_in = 0;
                            total_out = 0;
                            print_tui_banner(TuiBanner {
                                model_name: &loaded.model_name,
                                model_path: &loaded.model_path,
                                config: &loaded.config,
                                kernel: selector.selected.name(),
                                renderer: &loaded.renderer,
                                settings: &settings,
                                load_secs: loaded.load_secs,
                                governor_recommendation,
                            });
                            println!(
                                "{}model switched; conversation reset{}",
                                ansi::DIM,
                                ansi::RESET
                            );
                        }
                        Err(err) => {
                            eprintln!(
                                "{}model switch failed; keeping current model:{} {err}",
                                ansi::ERROR,
                                ansi::RESET
                            );
                        }
                    }
                }
                TuiCommand::Models => print_tui_models(&loaded.model_path),
                TuiCommand::Clear => {
                    history.clear();
                    session.reset(&loaded.config);
                    total_in = 0;
                    total_out = 0;
                    println!("{}conversation cleared{}", ansi::DIM, ansi::RESET);
                    print_tui_status(TuiStatus {
                        model_name: &loaded.model_name,
                        kernel: selector.selected.name(),
                        settings: &settings,
                        context_tokens: session.context_tokens.len(),
                        turns: history.len(),
                        input_tokens: 0,
                        output_tokens: 0,
                        total_in,
                        total_out,
                        ttft_secs: None,
                        elapsed_secs: 0.0,
                    });
                }
                TuiCommand::Status => print_tui_status(TuiStatus {
                    model_name: &loaded.model_name,
                    kernel: selector.selected.name(),
                    settings: &settings,
                    context_tokens: session.context_tokens.len(),
                    turns: history.len(),
                    input_tokens: 0,
                    output_tokens: 0,
                    total_in,
                    total_out,
                    ttft_secs: None,
                    elapsed_secs: 0.0,
                }),
                TuiCommand::History => print_tui_history(&history),
                TuiCommand::Save(path) => match save_tui_transcript(&path, &settings, &history) {
                    Ok(()) => println!("{}saved{} {}", ansi::LABEL, ansi::RESET, path),
                    Err(err) => eprintln!("{}save failed:{} {err}", ansi::ERROR, ansi::RESET),
                },
                TuiCommand::TempShow => {
                    println!(
                        "{}temperature{} {:.2}",
                        ansi::LABEL,
                        ansi::RESET,
                        settings.temp
                    );
                }
                TuiCommand::TempSet(next_temp) => {
                    settings.temp = next_temp;
                    println!(
                        "{}temperature{} {:.2}",
                        ansi::LABEL,
                        ansi::RESET,
                        settings.temp
                    );
                }
                TuiCommand::TokensShow => {
                    println!(
                        "{}max output tokens{} {}",
                        ansi::LABEL,
                        ansi::RESET,
                        settings.max_tokens
                    );
                }
                TuiCommand::TokensSet(next_tokens) => {
                    settings.max_tokens = next_tokens;
                    println!(
                        "{}max output tokens{} {}",
                        ansi::LABEL,
                        ansi::RESET,
                        settings.max_tokens
                    );
                }
                TuiCommand::SystemShow => {
                    print_tui_system_prompt(settings.system_prompt.as_deref())
                }
                TuiCommand::SystemSet(prompt) => {
                    settings.system_prompt = Some(prompt);
                    history.clear();
                    session.reset(&loaded.config);
                    total_in = 0;
                    total_out = 0;
                    println!(
                        "{}system prompt set; conversation reset{}",
                        ansi::DIM,
                        ansi::RESET
                    );
                }
                TuiCommand::SystemClear => {
                    settings.system_prompt = None;
                    history.clear();
                    session.reset(&loaded.config);
                    total_in = 0;
                    total_out = 0;
                    println!(
                        "{}system prompt cleared; conversation reset{}",
                        ansi::DIM,
                        ansi::RESET
                    );
                }
                TuiCommand::Trim(keep_turns) => {
                    trim_tui_history(&mut history, keep_turns);
                    session.reset(&loaded.config);
                    println!(
                        "{}history trimmed to {} turns; KV cache will rebuild on next prompt{}",
                        ansi::DIM,
                        history.len(),
                        ansi::RESET
                    );
                }
                TuiCommand::Unknown(command) => {
                    eprintln!(
                        "{}unknown command:{} {command}. Try /help.",
                        ansi::ERROR,
                        ansi::RESET
                    );
                }
            }
            continue;
        }

        history.push(ChatTurn {
            role: "user".to_owned(),
            content: input.to_owned(),
        });

        print!("{}assistant>{} ", ansi::ASSISTANT, ansi::RESET);
        if io::stdout().flush().is_err() {
            return ExitCode::FAILURE;
        }

        inference::trace_reset();
        let report = match generate_chat_turn(
            &tui_prompt_history(settings.system_prompt.as_deref(), &history),
            &mut session,
            ChatGenerationEnv {
                config: &loaded.config,
                weights: &loaded.weights,
                tokenizer: &loaded.tokenizer,
                runtime_options: loaded.runtime_options,
                draft: loaded.draft.as_mut(),
            },
            settings.temp,
            settings.max_tokens,
        ) {
            Ok((assistant_text, report)) => {
                history.push(ChatTurn {
                    role: "assistant".to_owned(),
                    content: assistant_text,
                });
                report
            }
            Err(err) => {
                history.pop();
                eprintln!("\n{}generation failed:{} {err}", ansi::ERROR, ansi::RESET);
                continue;
            }
        };
        print_runtime_trace_summary();

        total_in += report.input_tokens;
        total_out += report.output_tokens;
        print_tui_status(TuiStatus {
            model_name: &loaded.model_name,
            kernel: selector.selected.name(),
            settings: &settings,
            context_tokens: session.context_tokens.len(),
            turns: history.len(),
            input_tokens: report.input_tokens,
            output_tokens: report.output_tokens,
            total_in,
            total_out,
            ttft_secs: report.ttft_secs,
            elapsed_secs: report.elapsed_secs,
        });
    }

    println!("{}disconnected{}", ansi::DIM, ansi::RESET);
    ExitCode::SUCCESS
}

fn generate_chat_turn(
    history: &[ChatTurn],
    session: &mut ChatSession,
    mut env: ChatGenerationEnv<'_>,
    temp: f32,
    max_tokens: usize,
) -> Result<(String, ChatTurnReport), String> {
    let messages = history
        .iter()
        .map(|turn| tokenizer::ChatMessage {
            role: turn.role.as_str(),
            content: turn.content.as_str(),
        })
        .collect::<Vec<_>>();
    let rendered = env.tokenizer.render_chat_prompt(&messages);
    let prompt_tokens =
        env.tokenizer
            .encode(&rendered.text, rendered.add_special, rendered.parse_special)?;
    if prompt_tokens.is_empty() {
        return Err("prompt tokenized to an empty sequence".to_owned());
    }
    validate_generation_budget(prompt_tokens.len(), max_tokens, env.config.context_length)?;

    let shared_prefix = shared_token_prefix_len(&session.context_tokens, &prompt_tokens);
    if shared_prefix < session.context_tokens.len() {
        session.context_tokens.truncate(shared_prefix);
        session.pos = shared_prefix;
    }
    let input_tokens = prompt_tokens.len().saturating_sub(shared_prefix);
    let new_prompt_tokens = &prompt_tokens[shared_prefix..];
    let started_turn = std::time::Instant::now();

    let mut draft_pos = session.pos;
    if let Some(ref mut draft_ctx) = env.draft
        && let Some((&last_token, prefix_tokens)) = new_prompt_tokens.split_last()
    {
        let mut dummy_context = Vec::new();
        let mut d_pos = shared_prefix;
        prefill_tokens(
            prefix_tokens,
            &draft_ctx.config,
            &draft_ctx.weights,
            PrefillTokenState {
                cache: &mut draft_ctx.cache,
                ws: &mut draft_ctx.ws,
                batch_ws: Some(&mut draft_ctx.batch_ws),
                context_tokens: &mut dummy_context,
                pos: &mut d_pos,
            },
            env.runtime_options,
        );
        inference::forward_pass(
            last_token as usize,
            d_pos,
            &draft_ctx.config,
            &draft_ctx.weights,
            &mut draft_ctx.cache,
            &mut draft_ctx.ws,
            env.runtime_options,
        );
        draft_pos = d_pos + 1;
    }

    if let Some((&last_token, prefix_tokens)) = new_prompt_tokens.split_last() {
        prefill_tokens(
            prefix_tokens,
            env.config,
            env.weights,
            PrefillTokenState {
                cache: &mut session.cache,
                ws: &mut session.ws,
                batch_ws: Some(&mut session.batch_ws),
                context_tokens: &mut session.context_tokens,
                pos: &mut session.pos,
            },
            env.runtime_options,
        );
        inference::forward_pass(
            last_token as usize,
            session.pos,
            env.config,
            env.weights,
            &mut session.cache,
            &mut session.ws,
            env.runtime_options,
        );
        session.context_tokens.push(last_token);
        session.pos += 1;
    }

    let mut generated_tokens = Vec::new();
    let mut last_printed_len = 0usize;
    let mut ttft_secs = None;

    if let Some(ref mut draft_ctx) = env.draft {
        let draft_k = std::env::var("NANOCAMELID_DRAFT_K")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .unwrap_or(4);
        let mut total_drafted = 0;
        let mut total_accepted = 0;

        loop {
            if session.pos >= env.config.context_length || generated_tokens.len() >= max_tokens {
                break;
            }

            let current_len = generated_tokens.len();
            let step_budget = max_tokens - current_len;
            let current_k = draft_k.min(step_budget);

            let is_stop = |token: u32| is_generation_stop_token(&env.tokenizer.special, token);

            let (step_tokens, step_stats) = speculative::speculative_decoding_step(
                &mut speculative::SpeculativeTarget {
                    config: env.config,
                    weights: env.weights,
                    cache: &mut session.cache,
                    ws: &mut session.ws,
                    batch_ws: &mut session.batch_ws,
                    pos: &mut session.pos,
                    context_tokens: &mut session.context_tokens,
                    runtime_options: env.runtime_options,
                },
                draft_ctx,
                &mut draft_pos,
                temp,
                current_k,
                is_stop,
            )?;

            total_drafted += step_stats.drafted;
            total_accepted += step_stats.accepted;

            if step_tokens.is_empty() {
                break;
            }

            for &token in &step_tokens {
                generated_tokens.push(token);
                ttft_secs.get_or_insert_with(|| started_turn.elapsed().as_secs_f64());
                if is_generation_stop_token(&env.tokenizer.special, token) {
                    break;
                }
            }

            if let Ok(full_text) = env.tokenizer.decode(&generated_tokens, true)
                && full_text.len() > last_printed_len
            {
                print!("{}", &full_text[last_printed_len..]);
                io::stdout()
                    .flush()
                    .map_err(|err| format!("failed to flush stdout: {err}"))?;
                last_printed_len = full_text.len();
            }

            if generated_tokens
                .iter()
                .any(|&t| is_generation_stop_token(&env.tokenizer.special, t))
            {
                break;
            }
        }

        println!();
        println!(
            "{}Speculation: {:.1}% acceptance rate ({}/{} tokens accepted/drafted){}",
            ansi::DIM,
            if total_drafted > 0 {
                (total_accepted as f32 / total_drafted as f32) * 100.0
            } else {
                0.0
            },
            total_accepted,
            total_drafted,
            ansi::RESET
        );
    } else {
        loop {
            let next_token = inference::sample_logits(&session.ws.logits, temp);
            if is_generation_stop_token(&env.tokenizer.special, next_token as u32)
                || session.pos >= env.config.context_length
                || generated_tokens.len() >= max_tokens
            {
                break;
            }

            generated_tokens.push(next_token as u32);
            ttft_secs.get_or_insert_with(|| started_turn.elapsed().as_secs_f64());
            if let Ok(full_text) = env.tokenizer.decode(&generated_tokens, true)
                && full_text.len() > last_printed_len
            {
                print!("{}", &full_text[last_printed_len..]);
                io::stdout()
                    .flush()
                    .map_err(|err| format!("failed to flush stdout: {err}"))?;
                last_printed_len = full_text.len();
            }

            if generated_tokens.len() >= max_tokens {
                inference::prefill_pass(
                    next_token,
                    session.pos,
                    env.config,
                    env.weights,
                    &mut session.cache,
                    &mut session.ws,
                    env.runtime_options,
                );
                session.context_tokens.push(next_token as u32);
                session.pos += 1;
                break;
            }

            inference::forward_pass(
                next_token,
                session.pos,
                env.config,
                env.weights,
                &mut session.cache,
                &mut session.ws,
                env.runtime_options,
            );
            session.context_tokens.push(next_token as u32);
            session.pos += 1;
        }
    }

    let assistant_text = env.tokenizer.decode(&generated_tokens, true)?;
    println!();
    Ok((
        assistant_text,
        ChatTurnReport {
            input_tokens,
            output_tokens: generated_tokens.len(),
            ttft_secs,
            elapsed_secs: started_turn.elapsed().as_secs_f64(),
        },
    ))
}

struct PrefillTokenState<'a> {
    cache: &'a mut inference::LlamaKvCache,
    ws: &'a mut inference::LlamaWorkspace,
    batch_ws: Option<&'a mut inference::LlamaBatchWorkspace>,
    context_tokens: &'a mut Vec<u32>,
    pos: &'a mut usize,
}

fn prefill_tokens(
    tokens: &[u32],
    config: &model::LlamaModelConfig,
    weights: &model::LlamaWeights,
    mut state: PrefillTokenState<'_>,
    runtime_options: inference::LlamaRuntimeOptions,
) {
    let batch_size = state
        .batch_ws
        .as_ref()
        .map(|ws| ws.max_batch)
        .unwrap_or_else(prefill_batch_size);
    if batch_size <= 1 {
        for &token in tokens {
            inference::prefill_pass(
                token as usize,
                *state.pos,
                config,
                weights,
                state.cache,
                state.ws,
                runtime_options,
            );
            state.context_tokens.push(token);
            *state.pos += 1;
        }
        return;
    }

    for chunk in tokens.chunks(batch_size) {
        if chunk.len() == 1 {
            let token = chunk[0];
            inference::prefill_pass(
                token as usize,
                *state.pos,
                config,
                weights,
                state.cache,
                state.ws,
                runtime_options,
            );
        } else if let Some(batch_ws) = state.batch_ws.as_deref_mut() {
            inference::prefill_pass_batch(
                chunk,
                *state.pos,
                config,
                weights,
                state.cache,
                batch_ws,
                runtime_options,
            );
        } else {
            for &token in chunk {
                inference::prefill_pass(
                    token as usize,
                    *state.pos,
                    config,
                    weights,
                    state.cache,
                    state.ws,
                    runtime_options,
                );
                state.context_tokens.push(token);
                *state.pos += 1;
            }
            continue;
        }
        state.context_tokens.extend_from_slice(chunk);
        *state.pos += chunk.len();
    }
}

fn shared_token_prefix_len(lhs: &[u32], rhs: &[u32]) -> usize {
    lhs.iter()
        .zip(rhs)
        .take_while(|(left, right)| left == right)
        .count()
}

fn run_generation_with_prompt_builder<F>(
    command: &str,
    model_path: &Path,
    temp: f32,
    max_tokens: usize,
    prompt_builder: F,
    model_source: &str,
    audit_1b_shape: bool,
) -> ExitCode
where
    F: FnOnce(&tokenizer::Tokenizer) -> Result<GenerationPromptPlan, String>,
{
    if let Err(err) = prefill_batch_size_from_env() {
        eprintln!("{err}");
        return ExitCode::from(2);
    }

    println!("Loading GGUF file: {}...", model_path.display());
    let gguf = match gguf::read_file(model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to read GGUF: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut config = match model::LlamaModelConfig::from_gguf(&gguf) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to parse config: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(err) = apply_context_limit(&mut config) {
        eprintln!("Failed to apply context limit: {err}");
        return ExitCode::FAILURE;
    }
    println!("Architecture: {}", config.architecture);
    println!("Vocab size: {}", config.vocab_size);
    println!("Layers: {}", config.block_count);
    println!("Embedding width: {}", config.embedding_length);
    println!("Attention heads: {}", config.attention_head_count);

    let tokenizer = match tokenizer::Tokenizer::from_gguf(&gguf) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to load tokenizer: {e}");
            return ExitCode::FAILURE;
        }
    };

    let prompt_plan = match prompt_builder(&tokenizer) {
        Ok(plan) => plan,
        Err(err) => {
            eprintln!("Failed to prepare prompt: {err}");
            return ExitCode::FAILURE;
        }
    };

    println!("Loading model weights into memory...");
    let started_load = std::time::Instant::now();
    let weights = match model::LlamaWeights::load(model_path, &config, &gguf) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to load weights: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "Weights loaded in {:.2}s",
        started_load.elapsed().as_secs_f64()
    );

    let prompt_tokens = match tokenizer.encode(
        &prompt_plan.text,
        prompt_plan.add_special,
        prompt_plan.parse_special,
    ) {
        Ok(tokens) => tokens,
        Err(e) => {
            eprintln!("Failed to tokenize prompt: {e}");
            return ExitCode::FAILURE;
        }
    };
    if prompt_tokens.is_empty() {
        eprintln!("Prompt tokenized to an empty sequence");
        return ExitCode::FAILURE;
    }
    if let Err(err) =
        validate_generation_budget(prompt_tokens.len(), max_tokens, config.context_length)
    {
        eprintln!("Prompt exceeds model context: {err}");
        return ExitCode::FAILURE;
    }
    if let Some(renderer) = prompt_plan.renderer {
        println!("Chat renderer: {renderer}");
        println!(
            "Chat template format: {}",
            prompt_plan.template_format.unwrap_or("none")
        );
    }
    println!("Prompt tokens: {:?}", prompt_tokens);

    let mut cache =
        inference::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut ws = inference::LlamaWorkspace::new(&config);
    let mut batch_ws = inference::LlamaBatchWorkspace::new(&config, prefill_batch_size());
    let selector = q8::Q8DotKernelSelector::from_env();
    let runtime_options = runtime_options_from_gguf(&gguf, selector);

    let mut draft = if let Ok(draft_path_str) = std::env::var("NANOCAMELID_DRAFT_GGUF") {
        if !draft_path_str.is_empty() {
            let draft_path = Path::new(&draft_path_str);
            println!("Loading draft GGUF file: {}...", draft_path.display());
            let draft_ctx = match speculative::SpeculativeContext::load(draft_path, runtime_options)
            {
                Ok(ctx) => ctx,
                Err(err) => {
                    eprintln!("failed to load draft GGUF: {err}");
                    return ExitCode::FAILURE;
                }
            };
            if draft_ctx.config.vocab_size != config.vocab_size {
                eprintln!(
                    "Vocabulary size mismatch: Target has {}, Draft has {}",
                    config.vocab_size, draft_ctx.config.vocab_size
                );
                return ExitCode::FAILURE;
            }
            Some(draft_ctx)
        } else {
            None
        }
    } else {
        None
    };

    inference::trace_reset();

    println!("Selected dot-product kernel: {}", selector.selected.name());
    println!("\nGenerating response:\n");

    let mut pos = 0;
    let started_prefill = std::time::Instant::now();

    // Prefill draft model if active
    let mut draft_pos = 0;
    if let Some(ref mut draft_ctx) = draft
        && let Some((&last_token, prefix_tokens)) = prompt_tokens.split_last()
    {
        let mut dummy_context = Vec::new();
        prefill_tokens(
            prefix_tokens,
            &draft_ctx.config,
            &draft_ctx.weights,
            PrefillTokenState {
                cache: &mut draft_ctx.cache,
                ws: &mut draft_ctx.ws,
                batch_ws: Some(&mut draft_ctx.batch_ws),
                context_tokens: &mut dummy_context,
                pos: &mut draft_pos,
            },
            runtime_options,
        );
        inference::forward_pass(
            last_token as usize,
            draft_pos,
            &draft_ctx.config,
            &draft_ctx.weights,
            &mut draft_ctx.cache,
            &mut draft_ctx.ws,
            runtime_options,
        );
        draft_pos += 1;
    }

    // Decode prompt tokens (prefill path)
    if let Some((&last_token, prefix_tokens)) = prompt_tokens.split_last() {
        let mut context_tokens = Vec::with_capacity(prompt_tokens.len());
        prefill_tokens(
            prefix_tokens,
            &config,
            &weights,
            PrefillTokenState {
                cache: &mut cache,
                ws: &mut ws,
                batch_ws: Some(&mut batch_ws),
                context_tokens: &mut context_tokens,
                pos: &mut pos,
            },
            runtime_options,
        );
        inference::forward_pass(
            last_token as usize,
            pos,
            &config,
            &weights,
            &mut cache,
            &mut ws,
            runtime_options,
        );
        pos += 1;
    }
    let prefill_batch = batch_ws.max_batch;
    let prefill_sec = started_prefill.elapsed().as_secs_f64();
    println!(
        "Prompt ingested in {:.2}s with prefill batch {}",
        prefill_sec, prefill_batch
    );

    // Now generate the next tokens
    let mut generated_tokens = Vec::new();
    let mut last_printed_len = 0;
    let start_gen = std::time::Instant::now();

    if let Some(ref mut draft_ctx) = draft {
        let draft_k = std::env::var("NANOCAMELID_DRAFT_K")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .unwrap_or(4);
        let mut total_drafted = 0;
        let mut total_accepted = 0;
        let mut context_tokens = prompt_tokens.clone();

        loop {
            if pos >= config.context_length || generated_tokens.len() >= max_tokens {
                break;
            }

            let current_len = generated_tokens.len();
            let step_budget = max_tokens - current_len;
            let current_k = draft_k.min(step_budget);

            let is_stop = |token: u32| is_generation_stop_token(&tokenizer.special, token);

            let (step_tokens, step_stats) = match speculative::speculative_decoding_step(
                &mut speculative::SpeculativeTarget {
                    config: &config,
                    weights: &weights,
                    cache: &mut cache,
                    ws: &mut ws,
                    batch_ws: &mut batch_ws,
                    pos: &mut pos,
                    context_tokens: &mut context_tokens,
                    runtime_options,
                },
                draft_ctx,
                &mut draft_pos,
                temp,
                current_k,
                is_stop,
            ) {
                Ok(result) => result,
                Err(err) => {
                    eprintln!("speculative decoding failed: {err}");
                    return ExitCode::FAILURE;
                }
            };

            total_drafted += step_stats.drafted;
            total_accepted += step_stats.accepted;

            if step_tokens.is_empty() {
                break;
            }

            for &token in &step_tokens {
                generated_tokens.push(token);
                if is_generation_stop_token(&tokenizer.special, token) {
                    break;
                }
            }

            if let Ok(full_text) = tokenizer.decode(&generated_tokens, true)
                && full_text.len() > last_printed_len
            {
                print!("{}", &full_text[last_printed_len..]);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                last_printed_len = full_text.len();
            }

            if generated_tokens
                .iter()
                .any(|&t| is_generation_stop_token(&tokenizer.special, t))
            {
                break;
            }
        }

        let elapsed = start_gen.elapsed().as_secs_f64();
        println!(
            "\n\nGenerated {} tokens in {:.2}s ({:.2} tokens/sec)",
            generated_tokens.len(),
            elapsed,
            generated_tokens.len() as f64 / elapsed
        );
        println!(
            "{}Speculation: {:.1}% acceptance rate ({}/{} tokens accepted/drafted){}",
            ansi::DIM,
            if total_drafted > 0 {
                (total_accepted as f32 / total_drafted as f32) * 100.0
            } else {
                0.0
            },
            total_accepted,
            total_drafted,
            ansi::RESET
        );
    } else {
        loop {
            let next_token = inference::sample_logits(&ws.logits, temp);

            if is_generation_stop_token(&tokenizer.special, next_token as u32)
                || pos >= config.context_length
                || generated_tokens.len() >= max_tokens
            {
                break;
            }

            generated_tokens.push(next_token as u32);
            if let Ok(full_text) = tokenizer.decode(&generated_tokens, true)
                && full_text.len() > last_printed_len
            {
                print!("{}", &full_text[last_printed_len..]);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                last_printed_len = full_text.len();
            }

            if generated_tokens.len() >= max_tokens {
                break;
            }

            inference::forward_pass(
                next_token,
                pos,
                &config,
                &weights,
                &mut cache,
                &mut ws,
                runtime_options,
            );

            pos += 1;
        }

        let elapsed = start_gen.elapsed().as_secs_f64();
        println!(
            "\n\nGenerated {} tokens in {:.2}s ({:.2} tokens/sec)",
            generated_tokens.len(),
            elapsed,
            generated_tokens.len() as f64 / elapsed
        );
    }
    let elapsed = start_gen.elapsed().as_secs_f64();
    println!("generation_status: ok");
    println!(
        "json: {}",
        generation_status_json(GenerationStatusJson {
            command,
            model_path,
            model_source,
            audit_1b_shape,
            architecture: &config.architecture,
            renderer: prompt_plan.renderer,
            template_format: prompt_plan.template_format,
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated_tokens.len(),
            prefill_batch,
            prefill_sec,
            generation_sec: elapsed,
        })
    );
    print_runtime_trace_summary();

    ExitCode::SUCCESS
}

struct GenerationStatusJson<'a> {
    command: &'a str,
    model_path: &'a Path,
    model_source: &'a str,
    audit_1b_shape: bool,
    architecture: &'a str,
    renderer: Option<&'a str>,
    template_format: Option<&'a str>,
    prompt_tokens: usize,
    generated_tokens: usize,
    prefill_batch: usize,
    prefill_sec: f64,
    generation_sec: f64,
}

fn generation_status_json(status: GenerationStatusJson<'_>) -> String {
    let tokens_per_sec = if status.generation_sec > 0.0 {
        Some(status.generated_tokens as f64 / status.generation_sec)
    } else {
        None
    };
    let target = if status.audit_1b_shape {
        json_string("llama32-1b")
    } else {
        "null".to_owned()
    };
    let shape = if status.audit_1b_shape {
        json_string("llama32_1b")
    } else {
        "null".to_owned()
    };
    let shape_ready = if status.audit_1b_shape {
        "true"
    } else {
        "null"
    };
    format!(
        "{{\"command\":{},\"status\":\"ok\",\"model\":{},\"selected_source\":{},\"target\":{},\"shape\":{},\"shape_ready\":{},\"architecture\":{},\"renderer\":{},\"template_format\":{},\"prompt_tokens\":{},\"generated_tokens\":{},\"prefill_batch\":{},\"prefill_sec\":{},\"generation_sec\":{},\"tokens_per_sec\":{}}}",
        json_string(status.command),
        json_string(&status.model_path.display().to_string()),
        json_string(status.model_source),
        target,
        shape,
        shape_ready,
        json_string(status.architecture),
        json_string_or_null(status.renderer),
        json_string_or_null(status.template_format),
        status.prompt_tokens,
        status.generated_tokens,
        status.prefill_batch,
        json_f64(status.prefill_sec),
        json_f64(status.generation_sec),
        tokens_per_sec
            .map(json_f64)
            .unwrap_or_else(|| "null".to_owned())
    )
}

fn json_string_or_null(value: Option<&str>) -> String {
    value.map(json_string).unwrap_or_else(|| "null".to_owned())
}

fn json_f64(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.6}")
    } else {
        "null".to_owned()
    }
}

fn runtime_options_from_gguf(
    gguf: &gguf::GgufFile,
    selector: q8::Q8DotKernelSelector,
) -> inference::LlamaRuntimeOptions {
    inference::LlamaRuntimeOptions {
        q8_selector: selector,
        compute_logits: true,
        rope_scaling: rope_scaling_from_gguf(gguf),
    }
}

fn parse_tui_command(input: &str) -> TuiCommand {
    let (command, rest) = input.split_once(' ').unwrap_or((input, ""));
    let rest = rest.trim();
    match command {
        "/help" | "/?" => TuiCommand::Help,
        "/exit" | "/quit" => TuiCommand::Exit,
        "/model" if rest.is_empty() => TuiCommand::ModelShow,
        "/model" => TuiCommand::ModelLoad(rest.to_owned()),
        "/models" => TuiCommand::Models,
        "/clear" | "/reset" => TuiCommand::Clear,
        "/status" | "/stats" => TuiCommand::Status,
        "/history" => TuiCommand::History,
        "/save" if rest.is_empty() => TuiCommand::Unknown("/save requires a path".to_owned()),
        "/save" => TuiCommand::Save(rest.to_owned()),
        "/temp" if rest.is_empty() => TuiCommand::TempShow,
        "/temp" => rest
            .parse::<f32>()
            .ok()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .map(TuiCommand::TempSet)
            .unwrap_or_else(|| {
                TuiCommand::Unknown("/temp expects a non-negative number".to_owned())
            }),
        "/tokens" if rest.is_empty() => TuiCommand::TokensShow,
        "/tokens" => rest
            .parse::<usize>()
            .ok()
            .filter(|&value| value > 0)
            .map(TuiCommand::TokensSet)
            .unwrap_or_else(|| {
                TuiCommand::Unknown("/tokens expects a positive integer".to_owned())
            }),
        "/system" if rest.is_empty() => TuiCommand::SystemShow,
        "/system" if rest == "clear" || rest == "off" => TuiCommand::SystemClear,
        "/system" => TuiCommand::SystemSet(rest.to_owned()),
        "/trim" => rest
            .parse::<usize>()
            .ok()
            .map(TuiCommand::Trim)
            .unwrap_or_else(|| {
                TuiCommand::Unknown("/trim expects an integer turn count".to_owned())
            }),
        other => TuiCommand::Unknown(other.to_owned()),
    }
}

fn tui_prompt_history(system_prompt: Option<&str>, history: &[ChatTurn]) -> Vec<ChatTurn> {
    let mut prompt_history =
        Vec::with_capacity(history.len() + usize::from(system_prompt.is_some()));
    if let Some(system_prompt) = system_prompt.filter(|prompt| !prompt.trim().is_empty()) {
        prompt_history.push(ChatTurn {
            role: "system".to_owned(),
            content: system_prompt.to_owned(),
        });
    }
    prompt_history.extend_from_slice(history);
    prompt_history
}

fn trim_tui_history(history: &mut Vec<ChatTurn>, keep_turns: usize) {
    if keep_turns == 0 {
        history.clear();
    } else if history.len() > keep_turns {
        let start = history.len() - keep_turns;
        history.drain(0..start);
    }
}

fn print_tui_models(current_model: &Path) {
    let dir = current_model.parent().unwrap_or_else(|| Path::new("."));
    match fs::read_dir(dir) {
        Ok(entries) => {
            println!("{}models{} {}", ansi::LABEL, ansi::RESET, dir.display());
            let mut models = entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| looks_like_gguf_path(&path.to_string_lossy()))
                .collect::<Vec<_>>();
            models.sort();
            if models.is_empty() {
                println!("  no GGUF files found");
            }
            for path in models.into_iter().take(32) {
                let marker = if path == current_model { "*" } else { " " };
                println!("  {marker} {}", path.display());
            }
        }
        Err(err) => eprintln!(
            "{}model list failed:{} {}: {err}",
            ansi::ERROR,
            ansi::RESET,
            dir.display()
        ),
    }
}

fn print_tui_history(history: &[ChatTurn]) {
    println!(
        "{}history{} {} turns",
        ansi::LABEL,
        ansi::RESET,
        history.len()
    );
    if history.is_empty() {
        println!("  empty");
        return;
    }
    for (idx, turn) in history.iter().enumerate() {
        println!(
            "  {:>2}. {:<9} {}",
            idx + 1,
            turn.role,
            abbreviate_for_status(&turn.content, 120)
        );
    }
}

fn print_tui_system_prompt(system_prompt: Option<&str>) {
    match system_prompt {
        Some(prompt) => println!("{}system{} {}", ansi::LABEL, ansi::RESET, prompt),
        None => println!("{}system{} none", ansi::LABEL, ansi::RESET),
    }
}

fn save_tui_transcript(
    path: &str,
    settings: &TuiSettings,
    history: &[ChatTurn],
) -> Result<(), String> {
    let mut out = String::new();
    out.push_str("# NanoCamelid TUI Transcript\n\n");
    out.push_str(&format!("temperature: {:.2}\n", settings.temp));
    out.push_str(&format!("max_tokens: {}\n\n", settings.max_tokens));
    if let Some(system_prompt) = settings.system_prompt.as_deref() {
        out.push_str("## system\n\n");
        out.push_str(system_prompt);
        out.push_str("\n\n");
    }
    for turn in history {
        out.push_str("## ");
        out.push_str(&turn.role);
        out.push_str("\n\n");
        out.push_str(&turn.content);
        out.push_str("\n\n");
    }
    fs::write(path, out).map_err(|err| err.to_string())
}

fn abbreviate_for_status(value: &str, max_chars: usize) -> String {
    let value = value.replace('\n', " ");
    if value.chars().count() <= max_chars {
        return value;
    }
    let mut out = value
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    out.push_str("...");
    out
}

struct TuiBanner<'a> {
    model_name: &'a str,
    model_path: &'a Path,
    config: &'a model::LlamaModelConfig,
    kernel: &'a str,
    renderer: &'a str,
    settings: &'a TuiSettings,
    load_secs: f64,
    governor_recommendation: Option<&'a str>,
}

struct TuiStatus<'a> {
    model_name: &'a str,
    kernel: &'a str,
    settings: &'a TuiSettings,
    context_tokens: usize,
    turns: usize,
    input_tokens: usize,
    output_tokens: usize,
    total_in: usize,
    total_out: usize,
    ttft_secs: Option<f64>,
    elapsed_secs: f64,
}

fn print_tui_banner(banner: TuiBanner<'_>) {
    println!();
    println!(
        "{}NanoCamelid{} - local model chat",
        ansi::TITLE,
        ansi::RESET
    );
    println!(
        "{}mode{} terminal assistant | type /help for commands",
        ansi::LABEL,
        ansi::RESET
    );
    println!(
        "{}model{} {}  {}path{} {}",
        ansi::LABEL,
        ansi::RESET,
        banner.model_name,
        ansi::LABEL,
        ansi::RESET,
        banner.model_path.display()
    );
    println!(
        "{}runtime{} {} | layers {} | ctx {} | kernel {} | renderer {} | temp {:.2} | max out {} | load {:.2}s",
        ansi::LABEL,
        ansi::RESET,
        banner.config.architecture,
        banner.config.block_count,
        banner.config.context_length,
        banner.kernel,
        banner.renderer,
        banner.settings.temp,
        banner.settings.max_tokens,
        banner.load_secs
    );
    if let Some(system_prompt) = banner.settings.system_prompt.as_deref() {
        println!(
            "{}system{} {}",
            ansi::LABEL,
            ansi::RESET,
            abbreviate_for_status(system_prompt, 96)
        );
    }
    if let Some(recommendation) = banner.governor_recommendation {
        println!(
            "{}governor{} ondemand detected; for repeatable low-latency decode: {}",
            ansi::LABEL,
            ansi::RESET,
            recommendation
        );
    }
    println!(
        "{}commands{} /help /model <path> /models /temp /tokens /system /status /save /clear /exit",
        ansi::LABEL,
        ansi::RESET
    );
    println!();
}

fn print_tui_commands() {
    println!("{}commands{}", ansi::LABEL, ansi::RESET);
    println!("  /model <path>       load another GGUF and reset the conversation");
    println!("  /model              show the current GGUF path");
    println!("  /models             list GGUFs next to the current model");
    println!("  /temp [value]       show or set sampling temperature");
    println!("  /tokens [count]     show or set max assistant tokens per turn");
    println!("  /system [prompt]    show or set the system prompt and reset chat");
    println!("  /system clear       clear the system prompt and reset chat");
    println!("  /status             show model, session, and decoding settings");
    println!("  /history            show the current conversation turns");
    println!("  /trim <turns>       keep only the last N conversation turns");
    println!("  /save <path>        write the conversation transcript");
    println!("  /clear              reset the conversation and token counters");
    println!("  /exit, /quit        leave the chat");
}

fn print_tui_status(status: TuiStatus<'_>) {
    let tok_per_sec = if status.elapsed_secs > 0.0 {
        status.output_tokens as f64 / status.elapsed_secs
    } else {
        0.0
    };
    let ttft = status
        .ttft_secs
        .map(|secs| format!("{:.0} ms", secs * 1000.0))
        .unwrap_or_else(|| "n/a".to_owned());
    println!(
        "{}connected | model {} | kernel {} | temp {:.2} | max {} | turns {} | ctx {} | last in {} | last out {} | total in/out {}/{} | ttft {} | {:.2} tok/sec{}",
        ansi::DIM,
        status.model_name,
        status.kernel,
        status.settings.temp,
        status.settings.max_tokens,
        status.turns,
        status.context_tokens,
        status.input_tokens,
        status.output_tokens,
        status.total_in,
        status.total_out,
        ttft,
        tok_per_sec,
        ansi::RESET
    );
    println!();
}

mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const TITLE: &str = "\x1b[38;5;208;1m";
    pub const LABEL: &str = "\x1b[38;5;226;1m";
    pub const ASSISTANT: &str = "\x1b[38;5;46;1m";
    pub const INPUT: &str = "\x1b[38;5;244m";
    pub const DIM: &str = "\x1b[38;5;240m";
    pub const ERROR: &str = "\x1b[38;5;196;1m";
}

#[cfg(target_arch = "aarch64")]
fn runtime_neon() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}

#[cfg(not(target_arch = "aarch64"))]
fn runtime_neon() -> bool {
    false
}

#[cfg(target_arch = "aarch64")]
fn runtime_dotprod() -> bool {
    std::arch::is_aarch64_feature_detected!("dotprod")
}

#[cfg(not(target_arch = "aarch64"))]
fn runtime_dotprod() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        fs,
        path::{Path, PathBuf},
    };

    use nanocamelid::gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};
    use nanocamelid::inference;
    use nanocamelid::q8;
    use nanocamelid::tokenizer::{SpecialTokens, TokenizerModel};

    use super::{
        ApiChatMessage, ApiCompletionChoice, ChatTurn, DEFAULT_1B_PREFILL_PROMPT,
        DEFAULT_1B_PREFILL_TEMP, DEFAULT_1B_PREFILL_TOKENS, DEFAULT_1B_SMOKE_PROMPT,
        DEFAULT_1B_SMOKE_TOKENS, DEFAULT_MODEL_GGUF_ENV, DEFAULT_Q4_PREFILL_BATCH,
        DEFAULT_Q4_PREFILL_PROMPT_LEN, DEFAULT_Q4_PREFILL_RUNS, DoctorArgs, GenerationStatusJson,
        HelpTopic, InspectTarget, LLAMA32_1B_Q4_MODEL, LLAMA32_1B_Q8_MODEL, LLAMA32_3B_Q4_MODEL,
        ModelsAction, PERFORMANCE_GOVERNOR_COMMAND, PrefillBenchBatchMetrics,
        ReadyDirectChatStatus, SMOKE_MODEL_GGUF_ENV, ServeArgs, Smoke1BArgs, SmokeDefaults,
        SmokeKind, TRACE_ENV, TuiCommand, api_chat_completion_response_json,
        api_completion_response_json, classify_model_quantization, classify_model_target,
        cpu_features, cpu_governor_recommendation, cpu_model, default_llama32_1b_model_path,
        default_llama32_3b_model_path, device_model, evidence_1b_status_json,
        evidence_context_pack_command, evidence_model_command, evidence_prefill_bench_command,
        evidence_prefill_bench_command_with_env, evidence_ready_no_chat_command,
        generation_status_json, help_topic_for_args, help_topic_named, http_status_text,
        inspect_1b_status_json, inspect_runtime_summary, is_generation_stop_token, is_help_flag,
        json_string, llama32_1b_model_not_found_message, llama32_1b_quantization_for_path,
        llama32_1b_shape_audit, llama32_3b_model_not_found_message, looks_like_gguf_path,
        looks_like_non_gguf_model_path, model_1b_status_json, parse_bench_1b_args_with_env,
        parse_bench_1b_args_with_path, parse_bench_q4_layout_args, parse_bench_q4_prefill_args,
        parse_bench_q8_dot_args, parse_content_length, parse_context_packs, parse_cpu_list,
        parse_doctor_args, parse_evidence_1b_args_with_env, parse_evidence_1b_args_with_path,
        parse_generate_args_with_env, parse_generate_args_with_env_and_alias_env_and_workspace,
        parse_generate_args_with_env_and_workspace, parse_http_request,
        parse_inspect_args_with_env, parse_model_1b_args_with_path, parse_models_args,
        parse_prefill_batches, parse_prefill_bench_1b_batch_metrics, parse_ready_1b_args_with_env,
        parse_ready_1b_args_with_env_and_smoke_defaults,
        parse_ready_1b_args_with_env_and_smoke_defaults_and_chat_default, parse_serve_args,
        parse_serve_args_with_defaults, parse_smoke_1b_args_with_env,
        parse_smoke_1b_args_with_env_and_defaults, parse_smoke_3b_args_with_env,
        parse_smoke_3b_args_with_env_and_defaults, parse_smoke_args_with_env,
        parse_tui_args_with_env, parse_tui_args_with_env_and_alias_env_and_workspace,
        parse_tui_args_with_env_and_workspace, parse_tui_command,
        prefill_batch_size_from_env_value, prefill_bench_1b_batch_command,
        prefill_bench_1b_batch_env, prefill_bench_1b_result_json, prefill_bench_1b_smoke_command,
        prefill_bench_1b_smoke_env, prefill_bench_1b_status_json, prefill_prompt_tokens_per_sec,
        print_runtime_trace_summary, ready_1b_status_json, ready_chat_enabled_default_for_args,
        ready_chat_enabled_from_env_value, ready_chat_prompt_from_env_value,
        ready_chat_temp_from_env_value, ready_chat_tokens_from_env_value, request_too_large_error,
        resolve_api_model_path, resolve_llama32_1b_model_path_with_workspace,
        resolve_llama32_3b_model_path_with_workspace, runtime_options_from_gguf,
        serve_metrics_text, serve_models_json, shared_token_prefix_len, shell_command,
        shell_command_with_env, smoke_1b_status_json, smoke_defaults_from_values,
        smoke_plan_command_with_context, smoke_plan_command_with_env, trim_tui_history,
        tui_prompt_history, validate_api_chat_completion_request, validate_api_completion_request,
        validate_api_input_token_cap, validate_generation_budget,
    };

    #[test]
    fn parses_cpu_lists() {
        assert_eq!(parse_cpu_list("1,2,3"), Some(vec![1, 2, 3]));
        assert_eq!(parse_cpu_list("1-3"), Some(vec![1, 2, 3]));
        assert_eq!(parse_cpu_list("3,1-2,2"), Some(vec![1, 2, 3]));
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(parse_cpu_list("core1"), None);
    }

    #[test]
    fn runtime_trace_summary_noops_when_empty() {
        assert_eq!(TRACE_ENV, "NANOCAMELID_TRACE");
        inference::trace_reset();
        print_runtime_trace_summary();
    }

    #[test]
    fn recommends_performance_governor_for_ondemand() {
        assert_eq!(
            cpu_governor_recommendation(Some("ondemand")),
            Some(PERFORMANCE_GOVERNOR_COMMAND)
        );
        assert_eq!(cpu_governor_recommendation(Some("performance")), None);
        assert_eq!(cpu_governor_recommendation(None), None);
    }

    #[test]
    fn ready_chat_env_value_disables_direct_chat_for_falsey_values() {
        assert_eq!(ready_chat_enabled_from_env_value("0"), Ok(false));
        assert_eq!(ready_chat_enabled_from_env_value("false"), Ok(false));
        assert_eq!(ready_chat_enabled_from_env_value(" no "), Ok(false));
        assert_eq!(ready_chat_enabled_from_env_value("off"), Ok(false));
        assert_eq!(ready_chat_enabled_from_env_value("1"), Ok(true));
        assert_eq!(ready_chat_enabled_from_env_value("true"), Ok(true));
        assert_eq!(ready_chat_enabled_from_env_value("yes"), Ok(true));
        assert_eq!(ready_chat_enabled_from_env_value("on"), Ok(true));
        assert_eq!(ready_chat_enabled_from_env_value(""), Ok(true));
        assert_eq!(
            ready_chat_enabled_from_env_value("flase"),
            Err("NANOCAMELID_READY_CHAT must be 0, 1, false, true, no, yes, off, or on")
        );
    }

    #[test]
    fn ready_chat_default_skips_env_when_flag_overrides() {
        assert_eq!(
            ready_chat_enabled_default_for_args(&["--no-chat".to_owned()]),
            Ok(true)
        );
        assert_eq!(
            ready_chat_enabled_default_for_args(&["--chat".to_owned()]),
            Ok(true)
        );
    }

    #[test]
    fn ready_chat_prompt_uses_env_override_when_non_empty() {
        assert_eq!(
            ready_chat_prompt_from_env_value(Some("Give one Pi tip.".to_owned()), "Smoke prompt"),
            "Give one Pi tip."
        );
        assert_eq!(
            ready_chat_prompt_from_env_value(Some(String::new()), "Smoke prompt"),
            "Smoke prompt"
        );
        assert_eq!(
            ready_chat_prompt_from_env_value(None, "Smoke prompt"),
            "Smoke prompt"
        );
    }

    #[test]
    fn ready_chat_tokens_parse_positive_env_override() {
        assert_eq!(
            ready_chat_tokens_from_env_value(Some("4".to_owned()), 8),
            Ok(4)
        );
        assert_eq!(
            ready_chat_tokens_from_env_value(Some("0".to_owned()), 8),
            Err("ready direct chat env token count must be a positive integer")
        );
        assert_eq!(
            ready_chat_tokens_from_env_value(Some("bad".to_owned()), 8),
            Err("ready direct chat env token count must be a positive integer")
        );
        assert_eq!(ready_chat_tokens_from_env_value(None, 8), Ok(8));
    }

    #[test]
    fn ready_chat_temp_rejects_invalid_env() {
        assert_eq!(
            ready_chat_temp_from_env_value(Some("0.2".to_owned())),
            Ok(0.2)
        );
        assert_eq!(
            ready_chat_temp_from_env_value(Some("-0.1".to_owned())),
            Err("ready direct chat env temperature must be a non-negative number")
        );
        assert_eq!(
            ready_chat_temp_from_env_value(Some("bad".to_owned())),
            Err("ready direct chat env temperature must be a non-negative number")
        );
        assert_eq!(ready_chat_temp_from_env_value(None), Ok(0.0));
    }

    #[test]
    fn prefill_batch_env_rejects_invalid_values() {
        assert_eq!(
            prefill_batch_size_from_env_value(None),
            Ok(DEFAULT_Q4_PREFILL_BATCH)
        );
        assert_eq!(
            prefill_batch_size_from_env_value(Some("32".to_owned())),
            Ok(32)
        );
        assert_eq!(
            prefill_batch_size_from_env_value(Some("0".to_owned())),
            Err("NANOCAMELID_PREFILL_BATCH must be a positive integer")
        );
        assert_eq!(
            prefill_batch_size_from_env_value(Some("bad".to_owned())),
            Err("NANOCAMELID_PREFILL_BATCH must be a positive integer")
        );
    }

    #[test]
    fn parses_aarch64_cpuinfo() {
        let cpuinfo = "\
processor\t: 0
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
Hardware\t: BCM2712
";

        assert_eq!(cpu_model(cpuinfo), Some("BCM2712"));
        assert_eq!(
            cpu_features(cpuinfo),
            Some(
                "fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp"
            )
        );
    }

    #[test]
    fn parses_x86_cpuinfo_fallback() {
        let cpuinfo = "\
model name\t: Example CPU
flags\t\t: sse4_2 avx2
";

        assert_eq!(cpu_model(cpuinfo), Some("Example CPU"));
        assert_eq!(cpu_features(cpuinfo), Some("sse4_2 avx2"));
    }

    #[test]
    fn parses_device_tree_model() {
        assert_eq!(
            device_model("Raspberry Pi 5 Model B Rev 1.0\0"),
            Some("Raspberry Pi 5 Model B Rev 1.0")
        );
        assert_eq!(device_model("\0"), None);
    }

    #[test]
    fn prompt_context_validation_accepts_equal_length() {
        assert_eq!(validate_generation_budget(128, 0, 128), Ok(()));
    }

    #[test]
    fn prompt_context_validation_rejects_overflow() {
        assert_eq!(
            validate_generation_budget(129, 0, 128),
            Err("prompt requires 129 tokens but model context length is 128".to_owned())
        );
    }

    #[test]
    fn prompt_context_validation_rejects_generation_budget_overflow() {
        assert_eq!(
            validate_generation_budget(127, 2, 128),
            Err(
                "prompt uses 127 of 128 context tokens; requested 2 generation tokens but only 1 remain"
                    .to_owned()
            )
        );
    }

    #[test]
    fn shared_token_prefix_tracks_reusable_chat_context() {
        assert_eq!(shared_token_prefix_len(&[], &[1, 2, 3]), 0);
        assert_eq!(shared_token_prefix_len(&[1, 2, 3], &[1, 2, 3, 4, 5]), 3);
        assert_eq!(shared_token_prefix_len(&[1, 2, 9], &[1, 2, 3, 4]), 2);
        assert_eq!(shared_token_prefix_len(&[8, 2], &[1, 2]), 0);
    }

    #[test]
    fn tui_command_parser_handles_settings_and_session_commands() {
        assert_eq!(parse_tui_command("/help"), TuiCommand::Help);
        assert_eq!(parse_tui_command("/quit"), TuiCommand::Exit);
        assert_eq!(parse_tui_command("/model"), TuiCommand::ModelShow);
        assert_eq!(
            parse_tui_command("/model /models/llama.gguf"),
            TuiCommand::ModelLoad("/models/llama.gguf".to_owned())
        );
        assert_eq!(parse_tui_command("/models"), TuiCommand::Models);
        assert_eq!(parse_tui_command("/status"), TuiCommand::Status);
        assert_eq!(parse_tui_command("/history"), TuiCommand::History);
        assert_eq!(
            parse_tui_command("/save transcript.md"),
            TuiCommand::Save("transcript.md".to_owned())
        );
        assert_eq!(parse_tui_command("/temp"), TuiCommand::TempShow);
        assert_eq!(parse_tui_command("/temp 0.4"), TuiCommand::TempSet(0.4));
        assert_eq!(parse_tui_command("/tokens"), TuiCommand::TokensShow);
        assert_eq!(parse_tui_command("/tokens 32"), TuiCommand::TokensSet(32));
        assert_eq!(
            parse_tui_command("/system be terse"),
            TuiCommand::SystemSet("be terse".to_owned())
        );
        assert_eq!(parse_tui_command("/system clear"), TuiCommand::SystemClear);
        assert_eq!(parse_tui_command("/trim 4"), TuiCommand::Trim(4));
        assert!(matches!(
            parse_tui_command("/tokens 0"),
            TuiCommand::Unknown(_)
        ));
    }

    #[test]
    fn tui_prompt_history_prepends_system_prompt_without_mutating_history() {
        let history = vec![ChatTurn {
            role: "user".to_owned(),
            content: "hello".to_owned(),
        }];
        let prompt_history = tui_prompt_history(Some("be concise"), &history);

        assert_eq!(history.len(), 1);
        assert_eq!(prompt_history.len(), 2);
        assert_eq!(prompt_history[0].role, "system");
        assert_eq!(prompt_history[0].content, "be concise");
        assert_eq!(prompt_history[1].role, "user");
    }

    #[test]
    fn trim_tui_history_keeps_recent_turns() {
        let mut history = (0..5)
            .map(|idx| ChatTurn {
                role: "user".to_owned(),
                content: format!("turn {idx}"),
            })
            .collect::<Vec<_>>();

        trim_tui_history(&mut history, 2);
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].content, "turn 3");
        assert_eq!(history[1].content, "turn 4");

        trim_tui_history(&mut history, 0);
        assert!(history.is_empty());
    }

    #[test]
    fn generation_stop_token_uses_all_eog_tokens() {
        let special = SpecialTokens {
            eos: Some(1),
            eot: Some(2),
            eom: Some(3),
            eog: [1, 2, 3].into_iter().collect(),
            ..SpecialTokens::default()
        };

        assert!(is_generation_stop_token(&special, 1));
        assert!(is_generation_stop_token(&special, 2));
        assert!(is_generation_stop_token(&special, 3));
        assert!(!is_generation_stop_token(&special, 4));
    }

    #[test]
    fn help_topic_named_maps_supported_commands() {
        assert_eq!(help_topic_named("model"), Some(HelpTopic::Model));
        assert_eq!(help_topic_named("models"), Some(HelpTopic::Models));
        assert_eq!(help_topic_named("doctor"), Some(HelpTopic::Doctor));
        assert_eq!(help_topic_named("serve"), Some(HelpTopic::Serve));
        assert_eq!(help_topic_named("probe"), Some(HelpTopic::Probe));
        assert_eq!(help_topic_named("inspect"), Some(HelpTopic::Inspect));
        assert_eq!(help_topic_named("generate"), Some(HelpTopic::Generate));
        assert_eq!(help_topic_named("chat"), Some(HelpTopic::Chat));
        assert_eq!(help_topic_named("tui"), Some(HelpTopic::Tui));
        assert_eq!(help_topic_named("ready"), Some(HelpTopic::Ready));
        assert_eq!(help_topic_named("evidence"), Some(HelpTopic::Evidence));
        assert_eq!(help_topic_named("bench"), Some(HelpTopic::Bench));
        assert_eq!(help_topic_named("smoke"), Some(HelpTopic::Smoke));
        assert_eq!(help_topic_named("unknown"), None);
    }

    #[test]
    fn help_topic_for_args_detects_top_level_help() {
        assert_eq!(
            help_topic_for_args(&["--help".to_owned()]),
            Some(HelpTopic::TopLevel)
        );
        assert_eq!(
            help_topic_for_args(&["help".to_owned()]),
            Some(HelpTopic::TopLevel)
        );
    }

    #[test]
    fn help_topic_for_args_detects_subcommand_help() {
        assert_eq!(
            help_topic_for_args(&["help".to_owned(), "bench".to_owned()]),
            Some(HelpTopic::Bench)
        );
        assert_eq!(
            help_topic_for_args(&["help".to_owned(), "model".to_owned()]),
            Some(HelpTopic::Model)
        );
        assert_eq!(
            help_topic_for_args(&["help".to_owned(), "models".to_owned()]),
            Some(HelpTopic::Models)
        );
        assert_eq!(
            help_topic_for_args(&["help".to_owned(), "doctor".to_owned()]),
            Some(HelpTopic::Doctor)
        );
        assert_eq!(
            help_topic_for_args(&["help".to_owned(), "serve".to_owned()]),
            Some(HelpTopic::Serve)
        );
        assert_eq!(
            help_topic_for_args(&["help".to_owned(), "smoke".to_owned()]),
            Some(HelpTopic::Smoke)
        );
        assert_eq!(
            help_topic_for_args(&["help".to_owned(), "ready".to_owned()]),
            Some(HelpTopic::Ready)
        );
        assert_eq!(
            help_topic_for_args(&["help".to_owned(), "tui".to_owned()]),
            Some(HelpTopic::Tui)
        );
    }

    #[test]
    fn help_topic_for_args_detects_nested_model_alias_help() {
        assert_eq!(
            help_topic_for_args(&["inspect".to_owned(), "1b".to_owned(), "--help".to_owned()]),
            Some(HelpTopic::Inspect)
        );
        assert_eq!(
            help_topic_for_args(&[
                "inspect".to_owned(),
                "llama32-3b".to_owned(),
                "-h".to_owned()
            ]),
            Some(HelpTopic::Inspect)
        );
        assert_eq!(
            help_topic_for_args(&["generate".to_owned(), "1b".to_owned(), "--help".to_owned()]),
            Some(HelpTopic::Generate)
        );
        assert_eq!(
            help_topic_for_args(&["chat".to_owned(), "llama32-3b".to_owned(), "-h".to_owned()]),
            Some(HelpTopic::Chat)
        );
        assert_eq!(
            help_topic_for_args(&[
                "tui".to_owned(),
                "llama-3.2-1b".to_owned(),
                "--help".to_owned()
            ]),
            Some(HelpTopic::Tui)
        );
        assert_eq!(
            help_topic_for_args(&["ready".to_owned(), "1b".to_owned(), "--help".to_owned()]),
            Some(HelpTopic::Ready)
        );
        assert_eq!(
            help_topic_for_args(&[
                "evidence".to_owned(),
                "llama32-1b".to_owned(),
                "--help".to_owned()
            ]),
            Some(HelpTopic::Evidence)
        );
        assert_eq!(
            help_topic_for_args(&["model".to_owned(), "1b".to_owned(), "--help".to_owned()]),
            Some(HelpTopic::Model)
        );
        assert_eq!(
            help_topic_for_args(&["models".to_owned(), "list".to_owned(), "--help".to_owned()]),
            Some(HelpTopic::Models)
        );
        assert_eq!(
            help_topic_for_args(&["models".to_owned(), "scan".to_owned(), "-h".to_owned()]),
            Some(HelpTopic::Models)
        );
        assert_eq!(
            help_topic_for_args(&[
                "models".to_owned(),
                "inspect".to_owned(),
                "--help".to_owned()
            ]),
            Some(HelpTopic::Models)
        );
        assert_eq!(
            help_topic_for_args(&["bench".to_owned(), "1b".to_owned(), "--help".to_owned()]),
            Some(HelpTopic::Bench)
        );
        assert_eq!(
            help_topic_for_args(&["bench".to_owned(), "q8-dot".to_owned(), "-h".to_owned()]),
            Some(HelpTopic::Bench)
        );
        assert_eq!(
            help_topic_for_args(&[
                "smoke".to_owned(),
                "llama-3.2-1b".to_owned(),
                "-h".to_owned()
            ]),
            Some(HelpTopic::Smoke)
        );
        assert_eq!(
            help_topic_for_args(&[
                "smoke".to_owned(),
                "q8-chat".to_owned(),
                "--help".to_owned()
            ]),
            Some(HelpTopic::Smoke)
        );
    }

    #[test]
    fn help_topic_for_args_leaves_unknown_topics_to_main_parser() {
        assert_eq!(
            help_topic_for_args(&["help".to_owned(), "mystery".to_owned()]),
            None
        );
    }

    #[test]
    fn help_flag_recognizes_short_and_long_variants() {
        assert!(is_help_flag("-h"));
        assert!(is_help_flag("--help"));
        assert!(!is_help_flag("help"));
    }

    #[test]
    fn doctor_args_accept_json_and_dry_run_only() {
        assert_eq!(
            parse_doctor_args(&["--json".to_owned(), "--dry-run".to_owned()]),
            Ok(DoctorArgs {
                json: true,
                dry_run: true
            })
        );
        assert_eq!(
            parse_doctor_args(&["extra".to_owned()]).expect_err("extra doctor arg should fail"),
            "unexpected doctor argument"
        );
        assert_eq!(
            parse_doctor_args(&["--bad".to_owned()]).expect_err("bad doctor flag should fail"),
            "unknown doctor option"
        );
    }

    #[test]
    fn serve_args_parse_defaults_and_options() {
        let parsed = parse_serve_args_with_defaults(
            &["--dry-run".to_owned()],
            "/mnt/nanocamelid/models".to_owned(),
            None,
            None,
            None,
            None,
        )
        .expect("serve should parse");
        assert_eq!(parsed.host, "127.0.0.1");
        assert_eq!(parsed.port, 8080);
        assert_eq!(parsed.model_dir, "/mnt/nanocamelid/models");
        assert_eq!(parsed.api_key, None);
        assert_eq!(parsed.max_request_bytes, 65_536);
        assert_eq!(parsed.max_input_tokens, 2048);
        assert_eq!(parsed.max_output_tokens, 256);
        assert!(parsed.dry_run);

        let parsed = parse_serve_args_with_defaults(
            &[
                "--host=127.0.0.2".to_owned(),
                "--port".to_owned(),
                "9090".to_owned(),
                "--model-dir".to_owned(),
                "/models".to_owned(),
                "--api-key".to_owned(),
                "secret".to_owned(),
                "--max-request-bytes".to_owned(),
                "4096".to_owned(),
                "--max-input-tokens=1024".to_owned(),
                "--max-output-tokens".to_owned(),
                "64".to_owned(),
            ],
            "/unused".to_owned(),
            None,
            None,
            None,
            None,
        )
        .expect("serve explicit options should parse");
        assert_eq!(
            parsed,
            ServeArgs {
                host: "127.0.0.2".to_owned(),
                port: 9090,
                model_dir: "/models".to_owned(),
                api_key: Some("secret".to_owned()),
                max_request_bytes: 4096,
                max_input_tokens: 1024,
                max_output_tokens: 64,
                dry_run: false,
            }
        );
    }

    #[test]
    fn serve_args_reject_bad_shape() {
        assert_eq!(
            parse_serve_args(&["--port=0".to_owned()]).expect_err("zero port should fail"),
            "serve --port must be an integer from 1 to 65535"
        );
        assert_eq!(
            parse_serve_args(&["--host".to_owned()]).expect_err("missing host should fail"),
            "serve --host requires a bind address"
        );
        assert_eq!(
            parse_serve_args(&["--api-key=".to_owned()]).expect_err("empty key should fail"),
            "serve --api-key requires a non-empty token"
        );
        assert_eq!(
            parse_serve_args(&["extra".to_owned()]).expect_err("extra arg should fail"),
            "unexpected serve argument"
        );
    }

    #[test]
    fn serve_args_read_caps_from_default_env_values() {
        let parsed = parse_serve_args_with_defaults(
            &[],
            "/models".to_owned(),
            Some("env-secret".to_owned()),
            Some("2048".to_owned()),
            Some("512".to_owned()),
            Some("32".to_owned()),
        )
        .expect("serve env caps should parse");
        assert_eq!(parsed.model_dir, "/models");
        assert_eq!(parsed.api_key.as_deref(), Some("env-secret"));
        assert_eq!(parsed.max_request_bytes, 2048);
        assert_eq!(parsed.max_input_tokens, 512);
        assert_eq!(parsed.max_output_tokens, 32);
    }

    #[test]
    fn http_request_parser_reads_method_path_and_auth() {
        let request = parse_http_request(
            "POST /v1/models?limit=10 HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret\r\n\r\n{\"ok\":true}",
        )
        .expect("request should parse");
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/models");
        assert_eq!(request.authorization.as_deref(), Some("Bearer secret"));
        assert_eq!(request.body, "{\"ok\":true}");
        assert_eq!(parse_http_request("bad"), None);
    }

    #[test]
    fn serve_content_length_parser_reports_cap_errors() {
        assert_eq!(
            parse_content_length("Host: 127.0.0.1\r\nContent-Length: 42\r\n")
                .expect("content length should parse"),
            Some(42)
        );
        assert_eq!(
            parse_content_length("Host: 127.0.0.1\r\n").expect("missing content length is ok"),
            None
        );
        let err = parse_content_length("Content-Length: nope\r\n")
            .expect_err("invalid content length should fail");
        assert_eq!(err.code, "invalid_content_length");

        let err = request_too_large_error();
        assert_eq!(err.status, 413);
        assert_eq!(err.code, "request_too_large");
    }

    #[test]
    fn serve_completion_request_validation_enforces_openai_shape_and_caps() {
        let parsed = ServeArgs {
            host: "127.0.0.1".to_owned(),
            port: 8080,
            model_dir: "/models".to_owned(),
            api_key: None,
            max_request_bytes: 65_536,
            max_input_tokens: 6,
            max_output_tokens: 8,
            dry_run: false,
        };

        let request = validate_api_completion_request(
            r#"{"model":"tiny","prompt":["hello","small prompt"],"max_tokens":4}"#,
            &parsed,
        )
        .expect("completion request should validate");
        assert_eq!(request.model, "tiny");
        assert_eq!(request.prompts, vec!["hello", "small prompt"]);
        assert_eq!(request.input_tokens, 5);
        assert_eq!(request.requested_output_tokens, 4);
        assert_eq!(request.temperature, 0.0);

        let request = validate_api_completion_request(
            r#"{"model":"tiny","prompt":"hello","temperature":0.2}"#,
            &parsed,
        )
        .expect("completion request with temperature should validate");
        assert_eq!(request.temperature, 0.2);

        let err = validate_api_completion_request(
            r#"{"model":"tiny","prompt":"hello","max_tokens":9}"#,
            &parsed,
        )
        .expect_err("output cap should fail");
        assert_eq!(err.code, "output_tokens_exceeded");

        let request = validate_api_completion_request(
            r#"{"model":"tiny","prompt":"one two three four five six seven","max_tokens":4}"#,
            &parsed,
        )
        .expect("estimated input cap should not reject before tokenizer encoding");
        assert_eq!(request.input_tokens, 9);

        let err = validate_api_input_token_cap(7, &parsed)
            .expect_err("exact tokenizer input cap should fail");
        assert_eq!(err.code, "input_tokens_exceeded");

        let err = validate_api_completion_request(r#"{"model":"tiny"}"#, &parsed)
            .expect_err("prompt should be required");
        assert_eq!(err.code, "missing_prompt");

        let err = validate_api_completion_request(r#""model":"tiny","prompt":"hello""#, &parsed)
            .expect_err("non-object body should fail");
        assert_eq!(err.code, "invalid_json");

        let err = validate_api_completion_request(
            r#"{"model":"tiny","prompt":"hello","temperature":-0.1}"#,
            &parsed,
        )
        .expect_err("negative temperature should fail");
        assert_eq!(err.code, "invalid_temperature");
    }

    #[test]
    fn serve_chat_completion_request_validation_reads_messages() {
        let parsed = ServeArgs {
            host: "127.0.0.1".to_owned(),
            port: 8080,
            model_dir: "/models".to_owned(),
            api_key: None,
            max_request_bytes: 65_536,
            max_input_tokens: 16,
            max_output_tokens: 8,
            dry_run: false,
        };

        let request = validate_api_chat_completion_request(
            r#"{"model":"tiny","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"say hi"}],"max_tokens":2}"#,
            &parsed,
        )
        .expect("chat request should validate");
        assert_eq!(request.model, "tiny");
        assert_eq!(
            request.messages,
            vec![
                ApiChatMessage {
                    role: "system".to_owned(),
                    content: "be brief".to_owned(),
                },
                ApiChatMessage {
                    role: "user".to_owned(),
                    content: "say hi".to_owned(),
                },
            ]
        );
        assert_eq!(request.input_tokens, 4);
        assert_eq!(request.requested_output_tokens, 2);

        let err = validate_api_chat_completion_request(
            r#"{"model":"tiny","messages":[{"role":"user","content":""}]}"#,
            &parsed,
        )
        .expect_err("empty content should fail");
        assert_eq!(err.code, "invalid_messages");

        let err = validate_api_chat_completion_request(
            r#"{"model":"tiny","messages":[{"role":"tool","content":"hello"}],"max_tokens":2}"#,
            &parsed,
        )
        .expect_err("unsupported role should fail");
        assert_eq!(err.code, "invalid_messages");

        let err = validate_api_chat_completion_request(
            r#"{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":0}"#,
            &parsed,
        )
        .expect_err("zero max_tokens should fail");
        assert_eq!(err.code, "invalid_max_tokens");
    }

    #[test]
    fn serve_resolves_api_model_ids_from_model_directory() {
        let dir =
            std::env::temp_dir().join(format!("nanocamelid-api-model-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp model dir should be created");
        let q4_path = dir.join(LLAMA32_1B_Q4_MODEL);
        let qwen_path = dir.join("qwen2.5-coder-0.5b-instruct-q4_0.gguf");
        fs::write(&q4_path, b"placeholder").expect("q4 placeholder should be written");
        fs::write(&qwen_path, b"placeholder").expect("qwen placeholder should be written");

        let parsed = ServeArgs {
            host: "127.0.0.1".to_owned(),
            port: 8080,
            model_dir: dir.to_string_lossy().into_owned(),
            api_key: None,
            max_request_bytes: 65_536,
            max_input_tokens: 2048,
            max_output_tokens: 256,
            dry_run: false,
        };

        assert_eq!(
            resolve_api_model_path("1b", &parsed).expect("1b alias should resolve"),
            q4_path
        );
        assert_eq!(
            resolve_api_model_path("qwen2.5-coder-0.5b-instruct-q4_0", &parsed)
                .expect("stem id should resolve"),
            qwen_path
        );
        assert_eq!(
            resolve_api_model_path(
                &dir.join("qwen2.5-coder-0.5b-instruct-q4_0.gguf")
                    .to_string_lossy(),
                &parsed,
            )
            .expect("explicit path should resolve"),
            qwen_path
        );
        let err =
            resolve_api_model_path("missing", &parsed).expect_err("missing model should fail");
        assert_eq!(err.status, 404);
        assert_eq!(err.code, "model_not_found");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn api_completion_response_json_reports_choices_and_usage() {
        let json = api_completion_response_json(
            "tiny",
            &[
                ApiCompletionChoice {
                    index: 0,
                    text: "hello".to_owned(),
                    prompt_tokens: 2,
                    generated_tokens: 1,
                    finish_reason: "stop",
                },
                ApiCompletionChoice {
                    index: 1,
                    text: "world".to_owned(),
                    prompt_tokens: 3,
                    generated_tokens: 2,
                    finish_reason: "length",
                },
            ],
        );
        assert!(json.contains("\"object\":\"text_completion\""));
        assert!(json.contains("\"model\":\"tiny\""));
        assert!(json.contains("\"text\":\"hello\""));
        assert!(json.contains("\"finish_reason\":\"length\""));
        assert!(json.contains("\"prompt_tokens\":5"));
        assert!(json.contains("\"completion_tokens\":3"));
        assert!(json.contains("\"total_tokens\":8"));
    }

    #[test]
    fn api_chat_completion_response_json_reports_message_choices_and_usage() {
        let json = api_chat_completion_response_json(
            "tiny",
            &[ApiCompletionChoice {
                index: 0,
                text: "hello".to_owned(),
                prompt_tokens: 2,
                generated_tokens: 1,
                finish_reason: "stop",
            }],
        );
        assert!(json.contains("\"object\":\"chat.completion\""));
        assert!(json.contains("\"model\":\"tiny\""));
        assert!(json.contains("\"message\":{\"role\":\"assistant\",\"content\":\"hello\"}"));
        assert!(json.contains("\"finish_reason\":\"stop\""));
        assert!(json.contains("\"prompt_tokens\":2"));
        assert!(json.contains("\"completion_tokens\":1"));
        assert!(json.contains("\"total_tokens\":3"));
    }

    #[test]
    fn serve_models_json_reports_missing_dir_as_empty_list() {
        let parsed = ServeArgs {
            host: "127.0.0.1".to_owned(),
            port: 8080,
            model_dir: "/definitely/missing/nanocamelid-models".to_owned(),
            api_key: None,
            max_request_bytes: 65_536,
            max_input_tokens: 2048,
            max_output_tokens: 256,
            dry_run: false,
        };
        let json = serve_models_json(&parsed);
        assert!(json.contains("\"object\":\"list\""));
        assert!(json.contains("\"model_dir_exists\":false"));
        assert!(json.contains("\"data\":[]"));
    }

    #[test]
    fn serve_metrics_use_prometheus_text_shape() {
        let parsed = ServeArgs {
            host: "127.0.0.1".to_owned(),
            port: 8080,
            model_dir: "/models".to_owned(),
            api_key: None,
            max_request_bytes: 4096,
            max_input_tokens: 1024,
            max_output_tokens: 64,
            dry_run: false,
        };
        let metrics = serve_metrics_text(3, 1.25, &parsed);
        assert!(metrics.contains("nanocamelid_requests_total 3"));
        assert!(metrics.contains("nanocamelid_uptime_seconds 1.250"));
        assert!(metrics.contains("nanocamelid_max_request_bytes 4096"));
        assert!(metrics.contains("nanocamelid_max_input_tokens 1024"));
        assert_eq!(http_status_text(413), "Payload Too Large");
        assert_eq!(http_status_text(501), "Not Implemented");
    }

    #[test]
    fn models_args_parse_list_scan_and_inspect() {
        let parsed = parse_models_args(&[
            "list".to_owned(),
            "--dir".to_owned(),
            "/models".to_owned(),
            "--json".to_owned(),
            "--dry-run".to_owned(),
        ])
        .expect("models list should parse");
        assert!(parsed.json);
        assert!(parsed.dry_run);
        assert_eq!(
            parsed.action,
            ModelsAction::List {
                dir: "/models".to_owned()
            }
        );

        let parsed = parse_models_args(&["scan".to_owned(), "--dir=/models".to_owned()])
            .expect("models scan should parse");
        assert_eq!(
            parsed.action,
            ModelsAction::Scan {
                dir: "/models".to_owned()
            }
        );

        let parsed = parse_models_args(&[
            "inspect".to_owned(),
            "/models/Llama-3.2-1B-Instruct-Q4_0.gguf".to_owned(),
            "--dry-run".to_owned(),
        ])
        .expect("models inspect should parse");
        match parsed.action {
            ModelsAction::Inspect(inspect) => {
                assert_eq!(
                    inspect.model_path,
                    "/models/Llama-3.2-1B-Instruct-Q4_0.gguf"
                );
                assert!(inspect.dry_run);
            }
            other => panic!("expected inspect action, got {other:?}"),
        }
    }

    #[test]
    fn models_args_reject_bad_shape() {
        assert_eq!(
            parse_models_args(&[]).expect_err("missing models command should fail"),
            "missing models command; expected list, scan, or inspect"
        );
        assert_eq!(
            parse_models_args(&["bad".to_owned()]).expect_err("unknown models command should fail"),
            "unknown models command; expected list, scan, or inspect"
        );
        assert_eq!(
            parse_models_args(&["list".to_owned(), "--dir".to_owned()])
                .expect_err("missing models dir should fail"),
            "models --dir requires a path"
        );
    }

    #[test]
    fn model_scan_classifies_common_filename_hints() {
        let llama_path = Path::new("/models/Llama-3.2-1B-Instruct-Q8_0.gguf");
        assert_eq!(classify_model_target(llama_path), Some("llama32-1b"));
        assert_eq!(classify_model_quantization(llama_path), Some("Q8_0"));

        let qwen_path = Path::new("/models/qwen2.5-coder-0.5b-instruct-q5_k_m.gguf");
        assert_eq!(classify_model_target(qwen_path), Some("qwen2"));
        assert_eq!(classify_model_quantization(qwen_path), Some("Q5_K"));

        let unknown_path = Path::new("/models/custom.gguf");
        assert_eq!(classify_model_target(unknown_path), None);
        assert_eq!(classify_model_quantization(unknown_path), None);
    }

    #[test]
    fn bench_q8_dot_args_default_and_reject_bad_values() {
        let defaults = parse_bench_q8_dot_args(&[]).expect("default q8-dot args should parse");
        assert_eq!(defaults.iterations, q8::DEFAULT_DOT_BENCH_ITERATIONS);
        assert_eq!(defaults.runs, q8::DEFAULT_DOT_BENCH_RUNS);

        let parsed = parse_bench_q8_dot_args(&["11".to_owned(), "3".to_owned()])
            .expect("explicit q8-dot args should parse");
        assert_eq!(parsed.iterations, 11);
        assert_eq!(parsed.runs, 3);

        assert_eq!(
            parse_bench_q8_dot_args(&["0".to_owned()]).expect_err("zero iterations should fail"),
            "q8-dot iterations must be a positive integer"
        );
        assert_eq!(
            parse_bench_q8_dot_args(&["1".to_owned(), "bad".to_owned()])
                .expect_err("bad runs should fail"),
            "q8-dot runs must be a positive integer"
        );
        assert_eq!(
            parse_bench_q8_dot_args(&["1".to_owned(), "1".to_owned(), "extra".to_owned()])
                .expect_err("extra q8-dot arg should fail"),
            "unexpected extra q8-dot benchmark argument"
        );
    }

    #[test]
    fn bench_q4_layout_args_default_and_reject_bad_values() {
        let defaults =
            parse_bench_q4_layout_args(&[]).expect("default q4-layout args should parse");
        assert_eq!(defaults.rows, q8::DEFAULT_Q4_LAYOUT_BENCH_ROWS);
        assert_eq!(defaults.cols, q8::DEFAULT_Q4_LAYOUT_BENCH_COLS);
        assert_eq!(defaults.runs, q8::DEFAULT_DOT_BENCH_RUNS);

        let parsed =
            parse_bench_q4_layout_args(&["256".to_owned(), "1024".to_owned(), "2".to_owned()])
                .expect("explicit q4-layout args should parse");
        assert_eq!(parsed.rows, 256);
        assert_eq!(parsed.cols, 1024);
        assert_eq!(parsed.runs, 2);

        assert_eq!(
            parse_bench_q4_layout_args(&["bad".to_owned()]).expect_err("bad rows should fail"),
            "q4-layout rows must be a positive integer"
        );
        assert_eq!(
            parse_bench_q4_layout_args(&["1".to_owned(), "0".to_owned()])
                .expect_err("zero cols should fail"),
            "q4-layout cols must be a positive integer"
        );
        assert_eq!(
            parse_bench_q4_layout_args(&[
                "1".to_owned(),
                "1".to_owned(),
                "1".to_owned(),
                "extra".to_owned(),
            ])
            .expect_err("extra q4-layout arg should fail"),
            "unexpected extra q4-layout benchmark argument"
        );
    }

    #[test]
    fn bench_q4_prefill_args_default_and_reject_bad_values() {
        let defaults =
            parse_bench_q4_prefill_args(&[]).expect("default q4-prefill args should parse");
        assert_eq!(defaults.prompt_len, DEFAULT_Q4_PREFILL_PROMPT_LEN);
        assert_eq!(defaults.batch_size, DEFAULT_Q4_PREFILL_BATCH);
        assert_eq!(defaults.runs, DEFAULT_Q4_PREFILL_RUNS);

        let parsed =
            parse_bench_q4_prefill_args(&["96".to_owned(), "8".to_owned(), "2".to_owned()])
                .expect("explicit q4-prefill args should parse");
        assert_eq!(parsed.prompt_len, 96);
        assert_eq!(parsed.batch_size, 8);
        assert_eq!(parsed.runs, 2);

        assert_eq!(
            parse_bench_q4_prefill_args(&["0".to_owned()])
                .expect_err("zero prompt len should fail"),
            "q4-prefill prompt_len must be a positive integer"
        );
        assert_eq!(
            parse_bench_q4_prefill_args(&["1".to_owned(), "bad".to_owned()])
                .expect_err("bad batch size should fail"),
            "q4-prefill batch_size must be a positive integer"
        );
        assert_eq!(
            parse_bench_q4_prefill_args(&[
                "1".to_owned(),
                "1".to_owned(),
                "1".to_owned(),
                "extra".to_owned(),
            ])
            .expect_err("extra q4-prefill arg should fail"),
            "unexpected extra q4-prefill benchmark argument"
        );
    }

    #[test]
    fn bench_1b_args_default_and_reject_bad_values() {
        let defaults = parse_bench_1b_args_with_path(
            &["--dry-run".to_owned()],
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("default 1B prefill benchmark args should parse");

        assert_eq!(defaults.workspace, "/mnt/nanocamelid");
        assert_eq!(
            defaults.q4_model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(
            defaults.q8_model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q8_MODEL}")
        );
        assert_eq!(defaults.model_path, defaults.q8_model_path);
        assert_eq!(defaults.model_source, "workspace Q8_0 fallback");
        assert_eq!(defaults.prompt, DEFAULT_1B_PREFILL_PROMPT);
        assert_eq!(defaults.max_tokens, DEFAULT_1B_PREFILL_TOKENS);
        assert_eq!(defaults.temp, DEFAULT_1B_PREFILL_TEMP);
        assert_eq!(defaults.batches, vec![1, 16, 32, 64]);
        assert!(defaults.dry_run);

        let parsed = parse_bench_1b_args_with_path(
            &[
                "/models/custom.GGUF".to_owned(),
                "Hello".to_owned(),
                "3".to_owned(),
                "0.2".to_owned(),
                "1,8,16".to_owned(),
                "--dry-run".to_owned(),
            ],
            Some("/models/env.gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect("explicit 1B prefill benchmark args should parse");
        assert_eq!(parsed.model_path, "/models/custom.GGUF");
        assert_eq!(parsed.model_source, "explicit argument");
        assert_eq!(parsed.prompt, "Hello");
        assert_eq!(parsed.max_tokens, 3);
        assert_eq!(parsed.temp, "0.2");
        assert_eq!(parsed.batches, vec![1, 8, 16]);

        let smoke_env = parse_bench_1b_args_with_env(
            &["--dry-run".to_owned()],
            Some((
                "/models/smoke-override.gguf".to_owned(),
                SMOKE_MODEL_GGUF_ENV,
            )),
            "/mnt/nanocamelid",
            true,
        )
        .expect("smoke env 1B prefill benchmark path should parse");
        assert_eq!(smoke_env.model_path, "/models/smoke-override.gguf");
        assert_eq!(smoke_env.model_source, SMOKE_MODEL_GGUF_ENV);

        let q4 = parse_bench_1b_args_with_path(
            &["--q4".to_owned(), "--dry-run".to_owned()],
            Some("/models/env.gguf".to_owned()),
            "/mnt/nanocamelid",
            false,
        )
        .expect("q4 selector should parse");
        assert_eq!(
            q4.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(q4.model_source, "workspace Q4_0 requested");

        let explicit = parse_bench_1b_args_with_path(
            &[
                "/models/custom.gguf".to_owned(),
                "--q8".to_owned(),
                "--dry-run".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect("explicit model should override quant selector");
        assert_eq!(explicit.model_path, "/models/custom.gguf");
        assert_eq!(explicit.model_source, "explicit argument");

        assert_eq!(
            parse_bench_1b_args_with_path(
                &["--q4".to_owned(), "--q8".to_owned(), "--dry-run".to_owned()],
                None,
                "/mnt/nanocamelid",
                true,
            )
            .expect_err("conflicting quant selectors should fail"),
            "1B prefill benchmark accepts only one quantization selector"
        );

        assert_eq!(
            parse_bench_1b_args_with_path(
                &["/models/not-a-gguf".to_owned(), "--dry-run".to_owned()],
                None,
                "/mnt/nanocamelid",
                true,
            )
            .expect_err("path-like non-GGUF benchmark arg should fail"),
            "1B prefill benchmark model argument must be a .gguf path"
        );
        assert_eq!(
            parse_bench_1b_args_with_path(
                &["--oops".to_owned(), "--dry-run".to_owned()],
                None,
                "/mnt/nanocamelid",
                true,
            )
            .expect_err("unknown long option should fail before becoming a prompt"),
            "unknown 1B prefill benchmark option"
        );
        assert_eq!(
            parse_bench_1b_args_with_path(
                &["not-a-model".to_owned(), "0".to_owned()],
                None,
                "/mnt/nanocamelid",
                true,
            )
            .expect_err("zero token count should fail"),
            "1B prefill benchmark max_tokens must be a positive integer"
        );
        assert_eq!(
            parse_bench_1b_args_with_path(
                &[
                    "prompt".to_owned(),
                    "1".to_owned(),
                    "bad".to_owned(),
                    "1".to_owned(),
                ],
                None,
                "/mnt/nanocamelid",
                true,
            )
            .expect_err("bad temp should fail"),
            "1B prefill benchmark temp must be a non-negative number"
        );
        assert_eq!(
            parse_prefill_batches("1,0").expect_err("zero batch should fail"),
            "1B prefill benchmark batches must be positive integers"
        );
        assert_eq!(
            parse_prefill_batches("1,,16").expect_err("empty batch should fail"),
            "1B prefill benchmark batches must be positive integers"
        );
        assert_eq!(
            parse_prefill_batches("1, ,16").expect_err("blank batch should fail"),
            "1B prefill benchmark batches must be positive integers"
        );
        assert_eq!(
            parse_prefill_batches("16, 32 16").expect_err("duplicate batch should fail"),
            "1B prefill benchmark batches must be unique"
        );
        assert_eq!(
            parse_bench_1b_args_with_path(
                &[
                    "prompt".to_owned(),
                    "1".to_owned(),
                    "0".to_owned(),
                    "1".to_owned(),
                    "extra".to_owned(),
                ],
                None,
                "/mnt/nanocamelid",
                true,
            )
            .expect_err("extra 1B benchmark arg should fail"),
            "unexpected extra 1B prefill benchmark argument"
        );
    }

    #[test]
    fn evidence_1b_args_default_and_reject_bad_values() {
        let defaults = parse_evidence_1b_args_with_path(
            &["--dry-run".to_owned()],
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("default 1B evidence args should parse");

        assert_eq!(defaults.workspace, "/mnt/nanocamelid");
        assert_eq!(
            defaults.q4_model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(
            defaults.q8_model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q8_MODEL}")
        );
        assert_eq!(defaults.model_path, defaults.q8_model_path);
        assert_eq!(defaults.model_source, "workspace Q8_0 fallback");
        assert_eq!(defaults.smoke.kind, SmokeKind::Q8Chat);
        assert_eq!(defaults.smoke.prompt, DEFAULT_1B_SMOKE_PROMPT);
        assert_eq!(defaults.smoke.max_tokens, DEFAULT_1B_SMOKE_TOKENS);
        assert_eq!(defaults.prefill_batch, DEFAULT_Q4_PREFILL_BATCH);
        assert_eq!(defaults.context_packs, vec![512, 1024, 2048, 4096, 8192]);
        assert_eq!(defaults.prefill.prompt, DEFAULT_1B_PREFILL_PROMPT);
        assert_eq!(defaults.prefill.max_tokens, DEFAULT_1B_PREFILL_TOKENS);
        assert_eq!(defaults.prefill.temp, DEFAULT_1B_PREFILL_TEMP);
        assert_eq!(defaults.prefill.batches, vec![1, 16, 32, 64]);
        assert!(defaults.dry_run);

        let parsed = parse_evidence_1b_args_with_env(
            &["/models/custom.GGUF".to_owned(), "--dry-run".to_owned()],
            Some((
                "/models/smoke-override.gguf".to_owned(),
                SMOKE_MODEL_GGUF_ENV,
            )),
            "/mnt/nanocamelid",
            true,
        )
        .expect("explicit 1B evidence model should parse");
        assert_eq!(parsed.model_path, "/models/custom.GGUF");
        assert_eq!(parsed.model_source, "explicit argument");

        let smoke_env = parse_evidence_1b_args_with_env(
            &["--dry-run".to_owned()],
            Some((
                "/models/smoke-override.gguf".to_owned(),
                SMOKE_MODEL_GGUF_ENV,
            )),
            "/mnt/nanocamelid",
            true,
        )
        .expect("smoke env 1B evidence path should parse");
        assert_eq!(smoke_env.model_path, "/models/smoke-override.gguf");
        assert_eq!(smoke_env.model_source, SMOKE_MODEL_GGUF_ENV);

        let q8 = parse_evidence_1b_args_with_path(
            &["--q8".to_owned(), "--dry-run".to_owned()],
            Some("/models/env.gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect("q8 selector should parse");
        assert_eq!(
            q8.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q8_MODEL}")
        );
        assert_eq!(q8.model_source, "workspace Q8_0 requested");
        assert_eq!(q8.prefill.model_path, q8.model_path);
        assert_eq!(q8.prefill.model_source, q8.model_source);

        let explicit = parse_evidence_1b_args_with_path(
            &[
                "/models/custom.gguf".to_owned(),
                "--q4".to_owned(),
                "--dry-run".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect("explicit model should override quant selector");
        assert_eq!(explicit.model_path, "/models/custom.gguf");
        assert_eq!(explicit.model_source, "explicit argument");

        assert_eq!(
            parse_evidence_1b_args_with_path(
                &["--q4".to_owned(), "--q8".to_owned(), "--dry-run".to_owned()],
                None,
                "/mnt/nanocamelid",
                true,
            )
            .expect_err("conflicting quant selectors should fail"),
            "1B evidence accepts only one quantization selector"
        );

        assert_eq!(
            parse_evidence_1b_args_with_path(
                &["/models/not-a-gguf".to_owned(), "--dry-run".to_owned()],
                None,
                "/mnt/nanocamelid",
                true,
            )
            .expect_err("path-like non-GGUF evidence arg should fail"),
            "1B evidence model argument must be a .gguf path"
        );
        assert_eq!(
            parse_evidence_1b_args_with_path(
                &["/models/custom.gguf".to_owned(), "extra".to_owned()],
                None,
                "/mnt/nanocamelid",
                true,
            )
            .expect_err("extra evidence arg should fail"),
            "unexpected extra 1B evidence argument"
        );
        assert_eq!(
            parse_context_packs("512,0").expect_err("zero context pack should fail"),
            "1B evidence context packs must be positive integers"
        );
        assert_eq!(
            parse_context_packs("512,,1024").expect_err("empty context pack should fail"),
            "1B evidence context packs must be positive integers"
        );
        assert_eq!(
            parse_context_packs("512, ,1024").expect_err("blank context pack should fail"),
            "1B evidence context packs must be positive integers"
        );
        assert_eq!(
            parse_context_packs("512, 1024 512").expect_err("duplicate context pack should fail"),
            "1B evidence context packs must be unique"
        );
    }

    #[test]
    fn evidence_1b_plan_records_bundle_commands_and_json() {
        let parsed = parse_evidence_1b_args_with_path(
            &[
                "/models/Llama-3.2-1B-Instruct-Q4_0.gguf".to_owned(),
                "--dry-run".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect("explicit 1B evidence args should parse");

        assert_eq!(
            evidence_model_command(&parsed),
            "nanocamelid model 1b /models/Llama-3.2-1B-Instruct-Q4_0.gguf"
        );
        assert_eq!(
            evidence_ready_no_chat_command(&parsed, Some("512")),
            "NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid ready 1b /models/Llama-3.2-1B-Instruct-Q4_0.gguf chat 'Say hello in one sentence.' 8 --no-chat"
        );
        assert_eq!(
            evidence_context_pack_command(&parsed, 1024),
            "NANOCAMELID_CONTEXT_LIMIT=1024 nanocamelid smoke 1b /models/Llama-3.2-1B-Instruct-Q4_0.gguf chat 'Say hello in one sentence.' 8"
        );
        assert_eq!(
            evidence_prefill_bench_command(&parsed, Some("512")),
            "NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid bench 1b /models/Llama-3.2-1B-Instruct-Q4_0.gguf 'Explain one practical Raspberry Pi inference bottleneck in two short sentences.' 2 0.0 '1,16,32,64'"
        );
        assert_eq!(
            evidence_prefill_bench_command_with_env(&parsed, Some("512"), Some("32")),
            "NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid bench 1b /models/Llama-3.2-1B-Instruct-Q4_0.gguf 'Explain one practical Raspberry Pi inference bottleneck in two short sentences.' 2 0.0 '1,16,32,64'"
        );
        assert_eq!(
            evidence_1b_status_json(&parsed),
            "{\"target\":\"llama32-1b\",\"status\":\"ok\",\"model\":\"/models/Llama-3.2-1B-Instruct-Q4_0.gguf\",\"selected_source\":\"explicit argument\",\"quantization\":\"q4_0\",\"shape\":\"llama32_1b\",\"shape_ready\":true,\"context_limit\":\"unset\",\"ready_no_chat\":true,\"context_pack\":true,\"prefill_bench\":true,\"smoke_prompt\":\"Say hello in one sentence.\",\"smoke_kind\":\"chat\",\"smoke_tokens\":8,\"prefill_batch\":16,\"context_pack_caps\":[512,1024,2048,4096,8192],\"prefill_prompt\":\"Explain one practical Raspberry Pi inference bottleneck in two short sentences.\",\"prefill_tokens\":2,\"prefill_temp\":0.0,\"prefill_batches\":[1,16,32,64]}"
        );
    }

    #[test]
    fn gguf_path_detection_accepts_uppercase_extensions() {
        assert!(looks_like_gguf_path(
            "/models/Llama-3.2-1B-Instruct.Q4_0.GGUF"
        ));
        assert!(looks_like_gguf_path(
            "/models/Llama-3.2-1B-Instruct.Q4_0.GgUf/"
        ));
        assert!(!looks_like_gguf_path(
            "/models/Llama-3.2-1B-Instruct.gguf.tmp"
        ));
        assert!(looks_like_non_gguf_model_path("/models/not-a-gguf"));
        assert!(looks_like_non_gguf_model_path("models/not-a-gguf.bin"));
        assert!(!looks_like_non_gguf_model_path("Say hello"));
    }

    #[test]
    fn generate_args_use_explicit_model_path_without_env() {
        let parsed = parse_generate_args_with_env(
            &[
                "/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned(),
                "Hello".to_owned(),
                "0.5".to_owned(),
                "32".to_owned(),
            ],
            None,
        )
        .expect("explicit model path should parse");

        assert_eq!(parsed.model_path, "/models/Llama-3.2-1B-Instruct.Q8_0.gguf");
        assert_eq!(parsed.model_source, "explicit argument");
        assert_eq!(parsed.prompt, "Hello");
        assert_eq!(parsed.temp, 0.5);
        assert_eq!(parsed.max_tokens, 32);
    }

    #[test]
    fn generate_args_fall_back_to_env_model_path() {
        let parsed = parse_generate_args_with_env(
            &[
                "Explain grouped-query attention".to_owned(),
                "16".to_owned(),
            ],
            Some("/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned()),
        )
        .expect("env-backed generate path should parse");

        assert_eq!(parsed.model_path, "/models/Llama-3.2-1B-Instruct.Q8_0.gguf");
        assert_eq!(parsed.model_source, DEFAULT_MODEL_GGUF_ENV);
        assert_eq!(parsed.prompt, "Explain grouped-query attention");
        assert_eq!(parsed.temp, 16.0);
        assert_eq!(parsed.max_tokens, 128);
    }

    #[test]
    fn generate_args_accept_1b_alias() {
        let parsed = parse_generate_args_with_env_and_workspace(
            &[
                "1b".to_owned(),
                "Say hello".to_owned(),
                "0.0".to_owned(),
                "8".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect("1B alias should parse");

        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(parsed.model_source, "workspace Q4_0 default");
        assert_eq!(parsed.prompt, "Say hello");
        assert_eq!(parsed.temp, 0.0);
        assert_eq!(parsed.max_tokens, 8);
        assert!(parsed.audit_1b_shape);
    }

    #[test]
    fn generate_args_accept_1b_dry_run_without_prompt() {
        let parsed = parse_generate_args_with_env_and_workspace(
            &["1b".to_owned(), "--dry-run".to_owned()],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect("1B dry-run path should not require a prompt");

        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(parsed.model_source, "workspace Q4_0 default");
        assert_eq!(parsed.prompt, "<prompt>");
        assert!(parsed.dry_run);
        assert!(parsed.audit_1b_shape);
    }

    #[test]
    fn chat_style_generate_args_do_not_treat_dry_run_as_prompt() {
        let parsed = parse_generate_args_with_env_and_workspace(
            &[
                "1b".to_owned(),
                "--dry-run".to_owned(),
                "Say hello".to_owned(),
                "0.0".to_owned(),
                "8".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect("dry-run flag should be removed from chat-style args");

        assert_eq!(parsed.prompt, "Say hello");
        assert_eq!(parsed.model_source, "workspace Q4_0 default");
        assert_eq!(parsed.temp, 0.0);
        assert_eq!(parsed.max_tokens, 8);
        assert!(parsed.audit_1b_shape);
        assert!(parsed.dry_run);
    }

    #[test]
    fn generate_args_accept_3b_alias() {
        let parsed = parse_generate_args_with_env_and_workspace(
            &[
                "llama32-3b".to_owned(),
                "Say hello".to_owned(),
                "0.0".to_owned(),
                "8".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect("3B alias should parse");

        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_3B_Q4_MODEL}")
        );
        assert_eq!(parsed.model_source, "workspace 3B Q4_0 default");
        assert_eq!(parsed.prompt, "Say hello");
        assert_eq!(parsed.temp, 0.0);
        assert_eq!(parsed.max_tokens, 8);
    }

    #[test]
    fn generate_args_1b_alias_honors_env_model_override() {
        let parsed = parse_generate_args_with_env_and_workspace(
            &["llama-3.2-1b".to_owned(), "Say hello".to_owned()],
            Some("/models/custom-1b.gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect("1B alias with env override should parse");

        assert_eq!(parsed.model_path, "/models/custom-1b.gguf");
        assert_eq!(parsed.model_source, DEFAULT_MODEL_GGUF_ENV);
        assert_eq!(parsed.prompt, "Say hello");
        assert!(parsed.audit_1b_shape);
    }

    #[test]
    fn generate_args_1b_alias_prefers_smoke_env_model_override() {
        let parsed = parse_generate_args_with_env_and_alias_env_and_workspace(
            &["llama-3.2-1b".to_owned(), "Say hello".to_owned()],
            Some("/models/default-1b.gguf".to_owned()),
            Some(("/models/smoke-1b.gguf".to_owned(), SMOKE_MODEL_GGUF_ENV)),
            "/mnt/nanocamelid",
            true,
        )
        .expect("1B alias should prefer smoke env override");

        assert_eq!(parsed.model_path, "/models/smoke-1b.gguf");
        assert_eq!(parsed.model_source, SMOKE_MODEL_GGUF_ENV);
        assert_eq!(parsed.prompt, "Say hello");
        assert!(parsed.audit_1b_shape);
    }

    #[test]
    fn generate_args_reject_non_gguf_env_model_path() {
        let err = parse_generate_args_with_env_and_workspace(
            &["1b".to_owned(), "--dry-run".to_owned()],
            Some("/models/not-a-gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("non-GGUF generate env path should fail");

        assert_eq!(err, "model alias env path must be a .gguf path");
    }

    #[test]
    fn generate_args_reject_alias_prompt_that_looks_like_model_path() {
        let non_gguf = parse_generate_args_with_env_and_workspace(
            &[
                "1b".to_owned(),
                "/models/not-a-gguf".to_owned(),
                "--dry-run".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("path-like 1B alias prompt should fail");

        assert_eq!(
            non_gguf,
            "model alias prompt must not look like a model path; omit the alias when passing an explicit model"
        );

        let gguf = parse_generate_args_with_env_and_workspace(
            &[
                "llama32-3b".to_owned(),
                "/models/custom.gguf".to_owned(),
                "--dry-run".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("explicit model after alias should fail");

        assert_eq!(
            gguf,
            "model alias prompt must not look like a model path; omit the alias when passing an explicit model"
        );
    }

    #[test]
    fn generate_args_3b_alias_honors_env_model_override() {
        let parsed = parse_generate_args_with_env_and_workspace(
            &["llama-3.2-3b".to_owned(), "Say hello".to_owned()],
            Some("/models/custom-3b.gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect("3B alias with env override should parse");

        assert_eq!(parsed.model_path, "/models/custom-3b.gguf");
        assert_eq!(parsed.model_source, DEFAULT_MODEL_GGUF_ENV);
        assert_eq!(parsed.prompt, "Say hello");
        assert!(!parsed.audit_1b_shape);
    }

    #[test]
    fn generate_args_require_prompt_even_with_env_model_path() {
        let err = parse_generate_args_with_env(
            &[],
            Some("/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned()),
        )
        .expect_err("missing prompt should fail");

        assert_eq!(
            err,
            "missing prompt; pass one after the GGUF path or set NANOCAMELID_MODEL_GGUF and pass the prompt first"
        );
    }

    #[test]
    fn generate_args_reject_invalid_temp_token_count_and_extra_args() {
        let bad_temp = parse_generate_args_with_env(
            &[
                "/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned(),
                "Hello".to_owned(),
                "bad".to_owned(),
            ],
            None,
        )
        .expect_err("invalid generate temp should fail");
        assert_eq!(bad_temp, "generate temp must be a non-negative number");

        let bad_tokens = parse_generate_args_with_env(
            &[
                "/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned(),
                "Hello".to_owned(),
                "0.0".to_owned(),
                "0".to_owned(),
            ],
            None,
        )
        .expect_err("zero generate token count should fail");
        assert_eq!(bad_tokens, "generate max_tokens must be a positive integer");

        let extra = parse_generate_args_with_env(
            &[
                "/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned(),
                "Hello".to_owned(),
                "0.0".to_owned(),
                "8".to_owned(),
                "extra".to_owned(),
            ],
            None,
        )
        .expect_err("extra generate argument should fail");
        assert_eq!(extra, "unexpected extra generate argument");
    }

    #[test]
    fn tui_args_use_explicit_model_path_without_env() {
        let parsed = parse_tui_args_with_env(
            &[
                "/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned(),
                "0.2".to_owned(),
                "64".to_owned(),
            ],
            None,
        )
        .expect("explicit model path should parse");

        assert_eq!(parsed.model_path, "/models/Llama-3.2-1B-Instruct.Q8_0.gguf");
        assert_eq!(parsed.model_source, "explicit argument");
        assert_eq!(parsed.temp, 0.2);
        assert_eq!(parsed.max_tokens, 64);
        assert!(!parsed.dry_run);
    }

    #[test]
    fn tui_args_fall_back_to_env_model_path() {
        let parsed = parse_tui_args_with_env(
            &["0.1".to_owned(), "32".to_owned()],
            Some("/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned()),
        )
        .expect("env-backed tui path should parse");

        assert_eq!(parsed.model_path, "/models/Llama-3.2-1B-Instruct.Q8_0.gguf");
        assert_eq!(parsed.model_source, DEFAULT_MODEL_GGUF_ENV);
        assert_eq!(parsed.temp, 0.1);
        assert_eq!(parsed.max_tokens, 32);
        assert!(!parsed.dry_run);
    }

    #[test]
    fn tui_args_accept_1b_alias() {
        let parsed = parse_tui_args_with_env_and_workspace(
            &["llama32-1b".to_owned(), "0.1".to_owned(), "64".to_owned()],
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("1B TUI alias should parse");

        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q8_MODEL}")
        );
        assert_eq!(parsed.model_source, "workspace Q8_0 fallback");
        assert_eq!(parsed.temp, 0.1);
        assert_eq!(parsed.max_tokens, 64);
        assert!(!parsed.dry_run);
        assert!(parsed.audit_1b_shape);
    }

    #[test]
    fn tui_args_1b_alias_prefers_smoke_env_model_override() {
        let parsed = parse_tui_args_with_env_and_alias_env_and_workspace(
            &["llama32-1b".to_owned(), "0.1".to_owned(), "64".to_owned()],
            Some("/models/default-1b.gguf".to_owned()),
            Some(("/models/smoke-1b.gguf".to_owned(), SMOKE_MODEL_GGUF_ENV)),
            "/mnt/nanocamelid",
            false,
        )
        .expect("1B TUI alias should prefer smoke env override");

        assert_eq!(parsed.model_path, "/models/smoke-1b.gguf");
        assert_eq!(parsed.model_source, SMOKE_MODEL_GGUF_ENV);
        assert_eq!(parsed.temp, 0.1);
        assert_eq!(parsed.max_tokens, 64);
        assert!(parsed.audit_1b_shape);
    }

    #[test]
    fn tui_args_accept_3b_alias() {
        let parsed = parse_tui_args_with_env_and_workspace(
            &["llama32-3b".to_owned(), "0.1".to_owned(), "64".to_owned()],
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("3B TUI alias should parse");

        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_3B_Q4_MODEL}")
        );
        assert_eq!(parsed.model_source, "workspace 3B Q4_0 default");
        assert_eq!(parsed.temp, 0.1);
        assert_eq!(parsed.max_tokens, 64);
        assert!(!parsed.dry_run);
        assert!(!parsed.audit_1b_shape);
    }

    #[test]
    fn tui_args_accept_dry_run_anywhere() {
        let parsed = parse_tui_args_with_env_and_workspace(
            &[
                "--dry-run".to_owned(),
                "llama32-1b".to_owned(),
                "0.1".to_owned(),
                "64".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("1B TUI dry-run alias should parse");

        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q8_MODEL}")
        );
        assert_eq!(parsed.model_source, "workspace Q8_0 fallback");
        assert_eq!(parsed.temp, 0.1);
        assert_eq!(parsed.max_tokens, 64);
        assert!(parsed.dry_run);
        assert!(parsed.audit_1b_shape);
    }

    #[test]
    fn tui_args_reject_non_gguf_env_model_path() {
        let err = parse_tui_args_with_env_and_workspace(
            &["1b".to_owned(), "--dry-run".to_owned()],
            Some("/models/not-a-gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("non-GGUF TUI env path should fail");

        assert_eq!(err, "model alias env path must be a .gguf path");
    }

    #[test]
    fn tui_args_reject_alias_option_that_looks_like_model_path() {
        let err = parse_tui_args_with_env_and_workspace(
            &[
                "1b".to_owned(),
                "/models/not-a-gguf".to_owned(),
                "--dry-run".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("path-like 1B TUI alias option should fail");

        assert_eq!(
            err,
            "model alias argument must not be a path; use `tui <model.gguf>` for explicit models"
        );
    }

    #[test]
    fn tui_args_require_model_without_env() {
        let err = parse_tui_args_with_env(&[], None)
            .expect_err("missing model path should fail without env");

        assert_eq!(
            err,
            "missing GGUF model path; pass one or set NANOCAMELID_MODEL_GGUF"
        );
    }

    #[test]
    fn tui_args_reject_invalid_temp_token_count_and_extra_args() {
        let bad_temp = parse_tui_args_with_env(
            &[
                "/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned(),
                "-0.1".to_owned(),
            ],
            None,
        )
        .expect_err("negative TUI temp should fail");
        assert_eq!(bad_temp, "tui temp must be a non-negative number");

        let bad_tokens = parse_tui_args_with_env(
            &[
                "/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned(),
                "0.0".to_owned(),
                "bad".to_owned(),
            ],
            None,
        )
        .expect_err("invalid TUI token count should fail");
        assert_eq!(bad_tokens, "tui max_tokens must be a positive integer");

        let extra = parse_tui_args_with_env(
            &[
                "/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned(),
                "0.0".to_owned(),
                "8".to_owned(),
                "extra".to_owned(),
            ],
            None,
        )
        .expect_err("extra TUI arg should fail");
        assert_eq!(extra, "unexpected extra tui argument");
    }

    #[test]
    fn smoke_q8_model_args_use_explicit_model_path_without_env() {
        let parsed = parse_smoke_args_with_env(
            &[
                "/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned(),
                "Hello".to_owned(),
                "4".to_owned(),
            ],
            None,
        )
        .expect("explicit model path should parse");

        assert_eq!(parsed.model_path, "/models/Llama-3.2-1B-Instruct.Q8_0.gguf");
        assert_eq!(parsed.prompt, "Hello");
        assert_eq!(parsed.max_tokens, 4);
    }

    #[test]
    fn smoke_q8_model_args_fall_back_to_env_model_path() {
        let parsed = parse_smoke_args_with_env(
            &["Explain rotary embeddings".to_owned(), "2".to_owned()],
            Some("/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned()),
        )
        .expect("env-backed smoke path should parse");

        assert_eq!(parsed.model_path, "/models/Llama-3.2-1B-Instruct.Q8_0.gguf");
        assert_eq!(parsed.prompt, "Explain rotary embeddings");
        assert_eq!(parsed.max_tokens, 2);
    }

    #[test]
    fn smoke_q8_model_args_prefer_explicit_gguf_even_when_env_is_set() {
        let parsed = parse_smoke_args_with_env(
            &[
                "/override/model.gguf".to_owned(),
                "Hello".to_owned(),
                "3".to_owned(),
            ],
            Some("/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned()),
        )
        .expect("explicit gguf path should override env");

        assert_eq!(parsed.model_path, "/override/model.gguf");
        assert_eq!(parsed.prompt, "Hello");
        assert_eq!(parsed.max_tokens, 3);
    }

    #[test]
    fn smoke_q8_model_args_require_model_when_env_is_missing() {
        let err = parse_smoke_args_with_env(&[], None)
            .expect_err("missing model path should fail without env");

        assert_eq!(
            err,
            "missing GGUF model path; pass one or set NANOCAMELID_SMOKE_GGUF or NANOCAMELID_MODEL_GGUF"
        );
    }

    #[test]
    fn smoke_q8_model_args_reject_non_positive_token_count() {
        let err = parse_smoke_args_with_env(
            &[
                "/models/model.gguf".to_owned(),
                "Hello".to_owned(),
                "0".to_owned(),
            ],
            None,
        )
        .expect_err("zero-token smoke should fail");

        assert_eq!(err, "smoke max_tokens must be a positive integer");
    }

    #[test]
    fn smoke_q8_model_args_reject_extra_positionals_after_token_count() {
        let err = parse_smoke_args_with_env(
            &[
                "/models/model.gguf".to_owned(),
                "Hello".to_owned(),
                "2".to_owned(),
                "unexpected".to_owned(),
            ],
            None,
        )
        .expect_err("extra q8 smoke arg should fail");

        assert_eq!(err, "unexpected extra smoke argument");
    }

    #[test]
    fn default_1b_smoke_path_prefers_q4_when_present() {
        assert_eq!(
            default_llama32_1b_model_path("/mnt/nanocamelid", true),
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(
            default_llama32_1b_model_path("/mnt/nanocamelid", false),
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q8_MODEL}")
        );
    }

    #[test]
    fn default_3b_smoke_path_uses_q4_model() {
        assert_eq!(
            default_llama32_3b_model_path("/mnt/nanocamelid"),
            format!("/mnt/nanocamelid/models/{LLAMA32_3B_Q4_MODEL}")
        );
    }

    #[test]
    fn resolve_1b_model_path_prefers_env_before_pi_default() {
        assert_eq!(
            resolve_llama32_1b_model_path_with_workspace(
                Some("/models/env-1b.gguf".to_owned()),
                "/mnt/nanocamelid",
                true,
            ),
            "/models/env-1b.gguf"
        );
    }

    #[test]
    fn resolve_1b_model_path_uses_workspace_default_without_env() {
        assert_eq!(
            resolve_llama32_1b_model_path_with_workspace(None, "/mnt/nanocamelid", true),
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(
            resolve_llama32_1b_model_path_with_workspace(None, "/mnt/nanocamelid", false),
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q8_MODEL}")
        );
    }

    #[test]
    fn resolve_3b_model_path_prefers_env_before_pi_default() {
        assert_eq!(
            resolve_llama32_3b_model_path_with_workspace(
                Some("/models/env-3b.gguf".to_owned()),
                "/mnt/nanocamelid",
            ),
            "/models/env-3b.gguf"
        );
    }

    #[test]
    fn resolve_3b_model_path_uses_workspace_default_without_env() {
        assert_eq!(
            resolve_llama32_3b_model_path_with_workspace(None, "/mnt/nanocamelid"),
            format!("/mnt/nanocamelid/models/{LLAMA32_3B_Q4_MODEL}")
        );
    }

    #[test]
    fn inspect_1b_dry_run_resolves_model_without_requiring_file() {
        let parsed = parse_inspect_args_with_env(
            &["1b".to_owned(), "--dry-run".to_owned()],
            None,
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("1B inspect dry run should parse");

        assert_eq!(parsed.target, Some(InspectTarget::Llama32_1B));
        assert_eq!(parsed.model_source, "workspace Q8_0 fallback");
        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q8_MODEL}")
        );
        assert!(parsed.dry_run);
    }

    #[test]
    fn inspect_alias_prefers_smoke_env_model_path() {
        let parsed = parse_inspect_args_with_env(
            &["1b".to_owned()],
            None,
            Some(("/models/custom.GGUF".to_owned(), SMOKE_MODEL_GGUF_ENV)),
            "/mnt/nanocamelid",
            true,
        )
        .expect("env-backed 1B inspect should parse");

        assert_eq!(parsed.model_path, "/models/custom.GGUF");
        assert_eq!(parsed.model_source, SMOKE_MODEL_GGUF_ENV);
    }

    #[test]
    fn inspect_1b_prefers_explicit_path_over_env_and_defaults() {
        let parsed = parse_inspect_args_with_env(
            &[
                "1b".to_owned(),
                "/models/custom.GGUF".to_owned(),
                "--dry-run".to_owned(),
            ],
            None,
            Some(("/models/env.gguf".to_owned(), SMOKE_MODEL_GGUF_ENV)),
            "/mnt/nanocamelid",
            true,
        )
        .expect("explicit 1B inspect model should parse");

        assert_eq!(parsed.target, Some(InspectTarget::Llama32_1B));
        assert_eq!(parsed.model_path, "/models/custom.GGUF");
        assert_eq!(parsed.model_source, "explicit argument");
        assert!(parsed.dry_run);
    }

    #[test]
    fn inspect_1b_quant_selector_forces_workspace_row() {
        let parsed = parse_inspect_args_with_env(
            &["1b".to_owned(), "--q4".to_owned()],
            None,
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("q4 inspect selector should parse");

        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(parsed.model_source, "workspace Q4_0 requested");
    }

    #[test]
    fn inspect_args_reject_extra_positionals() {
        let err = parse_inspect_args_with_env(
            &[
                "1b".to_owned(),
                "/models/custom.gguf".to_owned(),
                "extra".to_owned(),
            ],
            None,
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect_err("extra inspect arg should fail");

        assert_eq!(err, "unexpected extra inspect 1B argument");
    }

    #[test]
    fn inspect_1b_rejects_non_gguf_model_arg() {
        let err = parse_inspect_args_with_env(
            &["1b".to_owned(), "/models/not-a-gguf".to_owned()],
            None,
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect_err("non-GGUF inspect 1B model arg should fail");

        assert_eq!(err, "inspect 1B model argument must be a .gguf path");
    }

    #[test]
    fn llama32_missing_model_messages_include_actionable_defaults() {
        let one_b = llama32_1b_model_not_found_message(Path::new(
            "/mnt/nanocamelid/models/missing-1b.gguf",
        ));
        assert!(one_b.contains("1B model not found: /mnt/nanocamelid/models/missing-1b.gguf"));
        assert!(one_b.contains(LLAMA32_1B_Q4_MODEL));
        assert!(one_b.contains(LLAMA32_1B_Q8_MODEL));
        assert!(one_b.contains("${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}/models"));

        let three_b = llama32_3b_model_not_found_message(Path::new(
            "/mnt/nanocamelid/models/missing-3b.gguf",
        ));
        assert!(three_b.contains("3B model not found: /mnt/nanocamelid/models/missing-3b.gguf"));
        assert!(three_b.contains(LLAMA32_3B_Q4_MODEL));
        assert!(three_b.contains("${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}/models"));
    }

    #[test]
    fn model_1b_audit_defaults_to_q4_when_present() {
        let parsed = parse_model_1b_args_with_path(&[], None, "/mnt/nanocamelid", true)
            .expect("default model audit should parse");

        assert_eq!(parsed.workspace, "/mnt/nanocamelid");
        assert_eq!(
            parsed.q4_model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(
            parsed.q8_model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q8_MODEL}")
        );
        assert_eq!(parsed.model_path, parsed.q4_model_path);
        assert_eq!(parsed.model_source, "workspace Q4_0 default");
        assert!(!parsed.dry_run);
    }

    #[test]
    fn model_1b_audit_falls_back_to_q8_and_accepts_dry_run() {
        let parsed = parse_model_1b_args_with_path(
            &["--dry-run".to_owned()],
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("dry-run model audit should parse");

        assert_eq!(parsed.model_path, parsed.q8_model_path);
        assert_eq!(parsed.model_source, "workspace Q8_0 fallback");
        assert!(parsed.dry_run);
    }

    #[test]
    fn model_1b_audit_quant_selector_forces_workspace_row() {
        let q8 = parse_model_1b_args_with_path(
            &["--q8".to_owned(), "--dry-run".to_owned()],
            Some("/models/env.gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect("q8 selector should parse");

        assert_eq!(
            q8.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q8_MODEL}")
        );
        assert_eq!(q8.model_source, "workspace Q8_0 requested");
        assert!(q8.dry_run);

        let q4 =
            parse_model_1b_args_with_path(&["--q4".to_owned()], None, "/mnt/nanocamelid", false)
                .expect("q4 selector should parse");

        assert_eq!(
            q4.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(q4.model_source, "workspace Q4_0 requested");
    }

    #[test]
    fn model_1b_audit_rejects_conflicting_quant_selectors() {
        let err = parse_model_1b_args_with_path(
            &["--q4".to_owned(), "--q8".to_owned()],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("conflicting selectors should fail");

        assert_eq!(err, "1B model audit accepts only one quantization selector");
    }

    #[test]
    fn model_1b_audit_prefers_explicit_path_over_env_and_defaults() {
        let parsed = parse_model_1b_args_with_path(
            &["/models/custom.GGUF".to_owned()],
            Some("/models/env.gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect("explicit model audit path should parse");

        assert_eq!(parsed.model_path, "/models/custom.GGUF");
        assert_eq!(parsed.model_source, "explicit argument");
    }

    #[test]
    fn model_1b_audit_rejects_non_gguf_path() {
        let err = parse_model_1b_args_with_path(
            &["not-a-model".to_owned()],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("non-GGUF model audit arg should fail");

        assert_eq!(err, "1B model audit argument must be a .gguf path");
    }

    #[test]
    fn model_1b_audit_rejects_non_gguf_env_path() {
        let err = parse_model_1b_args_with_path(
            &[],
            Some("/models/not-a-gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("non-GGUF env model audit path should fail");

        assert_eq!(err, "1B model audit env path must be a .gguf path");
    }

    #[test]
    fn smoke_1b_args_default_to_chat_prompt_and_pi_model() {
        let parsed = parse_smoke_1b_args_with_env(&[], None, "/mnt/nanocamelid", true)
            .expect("default 1B smoke args should parse");

        assert_eq!(parsed.kind, SmokeKind::Q8Chat);
        assert_eq!(parsed.model_source, "workspace Q4_0 default");
        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(parsed.prompt, DEFAULT_1B_SMOKE_PROMPT);
        assert_eq!(parsed.max_tokens, DEFAULT_1B_SMOKE_TOKENS);
    }

    #[test]
    fn smoke_1b_args_use_env_style_defaults_when_positionals_are_missing() {
        let parsed = parse_smoke_1b_args_with_env_and_defaults(
            &[],
            None,
            "/mnt/nanocamelid",
            true,
            SmokeDefaults {
                kind: SmokeKind::Q8Model,
                prompt: "One token check".to_owned(),
                max_tokens: 1,
            },
        )
        .expect("custom-default 1B smoke args should parse");

        assert_eq!(parsed.kind, SmokeKind::Q8Model);
        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(parsed.prompt, "One token check");
        assert_eq!(parsed.max_tokens, 1);
    }

    #[test]
    fn smoke_3b_args_use_env_style_defaults_when_positionals_are_missing() {
        let parsed = parse_smoke_3b_args_with_env_and_defaults(
            &[],
            None,
            "/mnt/nanocamelid",
            SmokeDefaults {
                kind: SmokeKind::Q8Model,
                prompt: "Three billion check".to_owned(),
                max_tokens: 3,
            },
        )
        .expect("custom-default 3B smoke args should parse");

        assert_eq!(parsed.kind, SmokeKind::Q8Model);
        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_3B_Q4_MODEL}")
        );
        assert_eq!(parsed.prompt, "Three billion check");
        assert_eq!(parsed.max_tokens, 3);
    }

    #[test]
    fn smoke_defaults_from_values_reject_invalid_kind_and_tokens() {
        let invalid_kind = smoke_defaults_from_values(
            SmokeDefaults::default(),
            Some("broken".to_owned()),
            None,
            None,
            "bad kind",
            "bad tokens",
        )
        .expect_err("invalid smoke kind should fail");
        assert_eq!(invalid_kind, "bad kind");

        let invalid_tokens = smoke_defaults_from_values(
            SmokeDefaults::default(),
            None,
            None,
            Some("0".to_owned()),
            "bad kind",
            "bad tokens",
        )
        .expect_err("zero smoke token default should fail");
        assert_eq!(invalid_tokens, "bad tokens");
    }

    #[test]
    fn smoke_1b_args_accept_kind_prompt_and_tokens() {
        let parsed = parse_smoke_1b_args_with_env(
            &["model".to_owned(), "Hello".to_owned(), "2".to_owned()],
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("custom 1B smoke args should parse");

        assert_eq!(parsed.kind, SmokeKind::Q8Model);
        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q8_MODEL}")
        );
        assert_eq!(parsed.prompt, "Hello");
        assert_eq!(parsed.max_tokens, 2);
    }

    #[test]
    fn smoke_3b_args_accept_kind_prompt_and_tokens() {
        let parsed = parse_smoke_3b_args_with_env(
            &["model".to_owned(), "Hello".to_owned(), "2".to_owned()],
            None,
            "/mnt/nanocamelid",
        )
        .expect("custom 3B smoke args should parse");

        assert_eq!(parsed.kind, SmokeKind::Q8Model);
        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_3B_Q4_MODEL}")
        );
        assert_eq!(parsed.prompt, "Hello");
        assert_eq!(parsed.max_tokens, 2);
    }

    #[test]
    fn smoke_1b_args_keep_legacy_q8_kind_aliases() {
        let parsed = parse_smoke_1b_args_with_env(
            &["q8-model".to_owned(), "Hello".to_owned(), "2".to_owned()],
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("legacy 1B smoke kind should parse");

        assert_eq!(parsed.kind, SmokeKind::Q8Model);
        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q8_MODEL}")
        );
        assert_eq!(parsed.prompt, "Hello");
        assert_eq!(parsed.max_tokens, 2);
    }

    #[test]
    fn smoke_1b_args_prefer_explicit_gguf_over_env_and_defaults() {
        let parsed = parse_smoke_1b_args_with_env(
            &[
                "/models/custom.GGUF".to_owned(),
                "q8-chat".to_owned(),
                "Say hi".to_owned(),
                "4".to_owned(),
            ],
            Some("/models/env.gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect("explicit 1B smoke model should parse");

        assert_eq!(parsed.kind, SmokeKind::Q8Chat);
        assert_eq!(parsed.model_path, "/models/custom.GGUF");
        assert_eq!(parsed.prompt, "Say hi");
        assert_eq!(parsed.max_tokens, 4);
    }

    #[test]
    fn smoke_1b_args_use_env_model_path_before_pi_default() {
        let parsed = parse_smoke_1b_args_with_env(
            &["chat".to_owned()],
            Some("/models/env.gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect("env-backed 1B smoke path should parse");

        assert_eq!(parsed.kind, SmokeKind::Q8Chat);
        assert_eq!(parsed.model_path, "/models/env.gguf");
        assert_eq!(parsed.model_source, "NANOCAMELID_MODEL_GGUF");
    }

    #[test]
    fn smoke_1b_args_reject_non_gguf_env_model_path() {
        let err = parse_smoke_1b_args_with_env(
            &["chat".to_owned()],
            Some("/models/not-a-gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("non-GGUF env smoke path should fail");

        assert_eq!(err, "1B smoke env model path must be a .gguf path");
    }

    #[test]
    fn smoke_1b_args_reject_path_like_non_gguf_model_arg() {
        let err = parse_smoke_1b_args_with_env(
            &["/models/not-a-gguf".to_owned(), "--dry-run".to_owned()],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("path-like non-GGUF smoke arg should fail");

        assert_eq!(err, "1B smoke model argument must be a .gguf path");
    }

    #[test]
    fn smoke_1b_args_accept_dry_run_without_consuming_positionals() {
        let parsed = parse_smoke_1b_args_with_env(
            &[
                "--dry-run".to_owned(),
                "/models/custom.GGUF".to_owned(),
                "model".to_owned(),
                "Hello".to_owned(),
                "2".to_owned(),
            ],
            Some("/models/env.gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect("dry-run smoke args should parse");

        assert!(parsed.dry_run);
        assert_eq!(parsed.kind, SmokeKind::Q8Model);
        assert_eq!(parsed.model_path, "/models/custom.GGUF");
        assert_eq!(parsed.prompt, "Hello");
        assert_eq!(parsed.max_tokens, 2);
    }

    #[test]
    fn dry_run_commands_quote_prompts_for_shell_reuse() {
        let command = shell_command(&[
            "nanocamelid",
            "smoke",
            "1b",
            "/models/Llama-3.2-1B-Instruct-Q4_0.gguf",
            "chat",
            "Say hello in one sentence.",
            "8",
        ]);

        assert_eq!(
            command,
            "nanocamelid smoke 1b /models/Llama-3.2-1B-Instruct-Q4_0.gguf chat 'Say hello in one sentence.' 8"
        );
    }

    #[test]
    fn dry_run_commands_escape_single_quotes() {
        let command = shell_command(&["nanocamelid", "chat", "/models/model.gguf", "can't", "0"]);

        assert_eq!(command, "nanocamelid chat /models/model.gguf 'can'\\''t' 0");
    }

    #[test]
    fn dry_run_commands_prefix_context_limit_for_reusable_plans() {
        let command = shell_command_with_env(
            &[
                "nanocamelid",
                "chat",
                "/models/model.gguf",
                "Say hello",
                "0",
                "8",
            ],
            &[("NANOCAMELID_CONTEXT_LIMIT", "512")],
        );

        assert_eq!(
            command,
            "NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid chat /models/model.gguf 'Say hello' 0 8"
        );
    }

    #[test]
    fn dry_run_env_assignments_escape_shell_values() {
        let command = shell_command_with_env(
            &[
                "nanocamelid",
                "chat",
                "/models/model.gguf",
                "can't",
                "0",
                "8",
            ],
            &[("NANOCAMELID_CONTEXT_LIMIT", "5 12")],
        );

        assert_eq!(
            command,
            "NANOCAMELID_CONTEXT_LIMIT='5 12' nanocamelid chat /models/model.gguf 'can'\\''t' 0 8"
        );
    }

    #[test]
    fn prefill_bench_1b_batch_command_records_kernel_and_batch_env() {
        let parsed = parse_bench_1b_args_with_path(
            &[
                "/models/Llama-3.2-1B-Instruct-Q8_0.gguf".to_owned(),
                "Say hello".to_owned(),
                "2".to_owned(),
                "0.0".to_owned(),
                "16".to_owned(),
                "--dry-run".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("1B prefill benchmark plan should parse");

        let command = prefill_bench_1b_batch_command(&parsed, 16, Some("512"));

        assert_eq!(
            command,
            "NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot NANOCAMELID_PREFILL_BATCH=16 nanocamelid chat /models/Llama-3.2-1B-Instruct-Q8_0.gguf 'Say hello' 0.0 2"
        );
    }

    #[test]
    fn prefill_bench_1b_smoke_command_records_kernel_preflight() {
        let parsed = parse_bench_1b_args_with_path(
            &[
                "/models/Llama-3.2-1B-Instruct-Q8_0.gguf".to_owned(),
                "Say hello".to_owned(),
                "2".to_owned(),
                "0.0".to_owned(),
                "16".to_owned(),
                "--dry-run".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("1B prefill benchmark plan should parse");

        let command = prefill_bench_1b_smoke_command(&parsed, Some("512"));

        assert_eq!(
            command,
            "NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot nanocamelid smoke 1b /models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello' 2"
        );
    }

    #[test]
    fn prefill_bench_1b_batch_env_applies_context_limit_to_actual_batch_runs() {
        assert_eq!(
            prefill_bench_1b_batch_env(16, Some("512")),
            vec![
                ("NANOCAMELID_CONTEXT_LIMIT", "512".to_owned()),
                ("NANOCAMELID_Q8_DOT_SDOT", "1".to_owned()),
                ("NANOCAMELID_Q8_DOT_KERNEL", "sdot".to_owned()),
                ("NANOCAMELID_PREFILL_BATCH", "16".to_owned()),
            ]
        );
        assert_eq!(
            prefill_bench_1b_batch_env(32, None),
            vec![
                ("NANOCAMELID_Q8_DOT_SDOT", "1".to_owned()),
                ("NANOCAMELID_Q8_DOT_KERNEL", "sdot".to_owned()),
                ("NANOCAMELID_PREFILL_BATCH", "32".to_owned()),
            ]
        );
    }

    #[test]
    fn prefill_bench_1b_smoke_env_applies_context_limit_to_preflight() {
        assert_eq!(
            prefill_bench_1b_smoke_env(Some("512")),
            vec![
                ("NANOCAMELID_CONTEXT_LIMIT", "512".to_owned()),
                ("NANOCAMELID_Q8_DOT_SDOT", "1".to_owned()),
                ("NANOCAMELID_Q8_DOT_KERNEL", "sdot".to_owned()),
            ]
        );
        assert_eq!(
            prefill_bench_1b_smoke_env(None),
            vec![
                ("NANOCAMELID_Q8_DOT_SDOT", "1".to_owned()),
                ("NANOCAMELID_Q8_DOT_KERNEL", "sdot".to_owned()),
            ]
        );
    }

    #[test]
    fn smoke_plan_command_uses_resolved_model_and_normalized_kind() {
        let parsed = Smoke1BArgs {
            kind: SmokeKind::Q8Model,
            model_path: "/models/ignored.gguf".to_owned(),
            model_source: "explicit argument",
            prompt: "Hello Pi".to_owned(),
            max_tokens: 2,
            dry_run: true,
        };

        assert_eq!(
            smoke_plan_command_with_context("1b", Path::new("/models/custom.gguf"), &parsed, None),
            "nanocamelid smoke 1b /models/custom.gguf model 'Hello Pi' 2"
        );
    }

    #[test]
    fn smoke_plan_command_can_include_context_limit_prefix() {
        let parsed = Smoke1BArgs {
            kind: SmokeKind::Q8Chat,
            model_path: "/models/ignored.gguf".to_owned(),
            model_source: "explicit argument",
            prompt: "Hello Pi".to_owned(),
            max_tokens: 2,
            dry_run: true,
        };

        assert_eq!(
            smoke_plan_command_with_context(
                "1b",
                Path::new("/models/custom.gguf"),
                &parsed,
                Some("512"),
            ),
            "NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid smoke 1b /models/custom.gguf chat 'Hello Pi' 2"
        );
    }

    #[test]
    fn smoke_plan_command_can_include_context_limit_and_prefill_batch_prefix() {
        let parsed = Smoke1BArgs {
            kind: SmokeKind::Q8Chat,
            model_path: "/models/ignored.gguf".to_owned(),
            model_source: "explicit argument",
            prompt: "Hello Pi".to_owned(),
            max_tokens: 2,
            dry_run: true,
        };

        assert_eq!(
            smoke_plan_command_with_env(
                "1b",
                Path::new("/models/custom.gguf"),
                &parsed,
                Some("512"),
                Some("32"),
            ),
            "NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid smoke 1b /models/custom.gguf chat 'Hello Pi' 2"
        );
    }

    #[test]
    fn json_string_escapes_log_values() {
        assert_eq!(json_string("plain"), "\"plain\"");
        assert_eq!(json_string("can't \"skip\"\n"), "\"can't \\\"skip\\\"\\n\"");
        assert_eq!(json_string("back\\slash"), "\"back\\\\slash\"");
    }

    #[test]
    fn generation_status_json_records_machine_readable_timing() {
        assert_eq!(
            generation_status_json(GenerationStatusJson {
                command: "chat",
                model_path: Path::new("/models/Llama-3.2-1B-Instruct-Q4_0.gguf"),
                model_source: "workspace Q4_0 default",
                audit_1b_shape: true,
                architecture: "llama",
                renderer: Some("llama3_instruct"),
                template_format: Some("llama3"),
                prompt_tokens: 19,
                generated_tokens: 2,
                prefill_batch: 16,
                prefill_sec: 0.5,
                generation_sec: 0.25,
            }),
            "{\"command\":\"chat\",\"status\":\"ok\",\"model\":\"/models/Llama-3.2-1B-Instruct-Q4_0.gguf\",\"selected_source\":\"workspace Q4_0 default\",\"target\":\"llama32-1b\",\"shape\":\"llama32_1b\",\"shape_ready\":true,\"architecture\":\"llama\",\"renderer\":\"llama3_instruct\",\"template_format\":\"llama3\",\"prompt_tokens\":19,\"generated_tokens\":2,\"prefill_batch\":16,\"prefill_sec\":0.500000,\"generation_sec\":0.250000,\"tokens_per_sec\":8.000000}"
        );
    }

    #[test]
    fn generation_status_json_uses_null_for_plain_generation_renderer() {
        assert_eq!(
            generation_status_json(GenerationStatusJson {
                command: "generate",
                model_path: Path::new("/models/model.gguf"),
                model_source: "explicit argument",
                audit_1b_shape: false,
                architecture: "llama",
                renderer: None,
                template_format: None,
                prompt_tokens: 1,
                generated_tokens: 0,
                prefill_batch: 16,
                prefill_sec: f64::INFINITY,
                generation_sec: 0.0,
            }),
            "{\"command\":\"generate\",\"status\":\"ok\",\"model\":\"/models/model.gguf\",\"selected_source\":\"explicit argument\",\"target\":null,\"shape\":null,\"shape_ready\":null,\"architecture\":\"llama\",\"renderer\":null,\"template_format\":null,\"prompt_tokens\":1,\"generated_tokens\":0,\"prefill_batch\":16,\"prefill_sec\":null,\"generation_sec\":0.000000,\"tokens_per_sec\":null}"
        );
    }

    #[test]
    fn model_1b_status_json_records_shape_audit_success() {
        assert_eq!(
            model_1b_status_json(
                Path::new("/models/Llama-3.2-1B-Instruct-Q4_0.gguf"),
                "workspace Q4_0 default"
            ),
            "{\"target\":\"llama32-1b\",\"status\":\"ok\",\"model\":\"/models/Llama-3.2-1B-Instruct-Q4_0.gguf\",\"selected_source\":\"workspace Q4_0 default\",\"quantization\":\"q4_0\",\"shape\":\"llama32_1b\",\"shape_ready\":true}"
        );
    }

    #[test]
    fn inspect_1b_status_json_records_shape_audit_success() {
        assert_eq!(
            inspect_1b_status_json(
                Path::new("/models/Llama-3.2-1B-Instruct-Q8_0.gguf"),
                "workspace Q8_0 fallback"
            ),
            "{\"target\":\"llama32-1b\",\"command\":\"inspect\",\"status\":\"ok\",\"model\":\"/models/Llama-3.2-1B-Instruct-Q8_0.gguf\",\"selected_source\":\"workspace Q8_0 fallback\",\"quantization\":\"q8_0\",\"shape\":\"llama32_1b\",\"shape_ready\":true}"
        );
    }

    #[test]
    fn llama32_1b_quantization_comes_from_selected_model_filename() {
        assert_eq!(
            llama32_1b_quantization_for_path(Path::new(
                "/mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf"
            )),
            "q4_0"
        );
        assert_eq!(
            llama32_1b_quantization_for_path(Path::new(
                "/mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
            )),
            "q8_0"
        );
        assert_eq!(
            llama32_1b_quantization_for_path(Path::new("/models/custom-1b.gguf")),
            "unknown"
        );
    }

    #[test]
    fn ready_1b_status_json_records_success_plan() {
        let smoke = Smoke1BArgs {
            kind: SmokeKind::Q8Chat,
            model_path: "/models/ignored.gguf".to_owned(),
            model_source: "explicit argument",
            prompt: "Say hello".to_owned(),
            max_tokens: 8,
            dry_run: false,
        };

        assert_eq!(
            ready_1b_status_json(
                Path::new("/models/Llama-3.2-1B-Instruct-Q4_0.gguf"),
                &smoke,
                ReadyDirectChatStatus {
                    enabled: true,
                    prompt: Some("Direct hello"),
                    tokens: Some(4),
                    temp: Some(0.2),
                },
                "512",
                32,
            ),
            "{\"target\":\"llama32-1b\",\"status\":\"ok\",\"model\":\"/models/Llama-3.2-1B-Instruct-Q4_0.gguf\",\"selected_source\":\"explicit argument\",\"quantization\":\"q4_0\",\"probe\":true,\"shape\":\"llama32_1b\",\"shape_ready\":true,\"context_limit\":\"512\",\"smoke_prompt\":\"Say hello\",\"smoke_kind\":\"chat\",\"smoke_tokens\":8,\"prefill_batch\":32,\"direct_chat\":true,\"chat_prompt\":\"Direct hello\",\"chat_tokens\":4,\"chat_temp\":0.2}"
        );
    }

    #[test]
    fn smoke_1b_status_json_records_success_plan() {
        let smoke = Smoke1BArgs {
            kind: SmokeKind::Q8Model,
            model_path: "/models/ignored.gguf".to_owned(),
            model_source: "NANOCAMELID_SMOKE_GGUF",
            prompt: "Say hello".to_owned(),
            max_tokens: 2,
            dry_run: false,
        };

        assert_eq!(
            smoke_1b_status_json(
                Path::new("/models/Llama-3.2-1B-Instruct-Q8_0.gguf"),
                &smoke,
                "unset",
                16,
            ),
            "{\"target\":\"llama32-1b\",\"status\":\"ok\",\"model\":\"/models/Llama-3.2-1B-Instruct-Q8_0.gguf\",\"selected_source\":\"NANOCAMELID_SMOKE_GGUF\",\"quantization\":\"q8_0\",\"shape\":\"llama32_1b\",\"shape_ready\":true,\"context_limit\":\"unset\",\"smoke_prompt\":\"Say hello\",\"smoke_kind\":\"model\",\"smoke_tokens\":2,\"prefill_batch\":16}"
        );
    }

    #[test]
    fn prefill_bench_1b_status_json_records_sweep_plan() {
        let parsed = parse_bench_1b_args_with_path(
            &[
                "/models/Llama-3.2-1B-Instruct-Q8_0.gguf".to_owned(),
                "Say hello".to_owned(),
                "2".to_owned(),
                "0.0".to_owned(),
                "1,16".to_owned(),
                "--dry-run".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("1B prefill benchmark plan should parse");

        assert_eq!(
            prefill_bench_1b_status_json(&parsed, "unset"),
            "{\"benchmark\":\"llama32-1b-prefill\",\"target\":\"llama32-1b\",\"status\":\"ok\",\"model\":\"/models/Llama-3.2-1B-Instruct-Q8_0.gguf\",\"selected_source\":\"explicit argument\",\"quantization\":\"q8_0\",\"probe\":true,\"shape\":\"llama32_1b\",\"shape_ready\":true,\"context_limit\":\"unset\",\"prompt\":\"Say hello\",\"max_tokens\":2,\"temp\":0.0,\"batches\":[1,16],\"best_prefill_batch\":null,\"best_prefill_sec\":null,\"best_prefill_prompt_tokens_per_sec\":null,\"best_decode_batch\":null,\"best_tokens_per_sec\":null}"
        );
    }

    #[test]
    fn prefill_bench_1b_batch_metrics_parse_generation_output() {
        let metrics = parse_prefill_bench_1b_batch_metrics(
            "Prompt ingested in 0.38s with prefill batch 16\nGenerated 8 tokens in 1.91s (4.18 tokens/sec)\njson: {\"command\":\"chat\",\"status\":\"ok\",\"prompt_tokens\":142,\"generated_tokens\":8,\"prefill_batch\":16,\"prefill_sec\":0.376000,\"generation_sec\":1.912000,\"tokens_per_sec\":4.184100}\n",
        );

        assert_eq!(
            metrics,
            PrefillBenchBatchMetrics {
                prompt_tokens: Some(142),
                prefill_sec: Some(0.376),
                generated_tokens: Some(8),
                generation_sec: Some(1.912),
                tokens_per_sec: Some(4.1841),
            }
        );
        assert_eq!(
            prefill_prompt_tokens_per_sec(metrics),
            Some(377.6595744680851)
        );
    }

    #[test]
    fn prefill_bench_1b_result_json_records_best_observed_batches() {
        let parsed = parse_bench_1b_args_with_path(
            &[
                "/models/Llama-3.2-1B-Instruct-Q4_0.gguf".to_owned(),
                "Say hello".to_owned(),
                "2".to_owned(),
                "0.0".to_owned(),
                "1,16".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect("1B prefill benchmark plan should parse");

        assert_eq!(
            prefill_bench_1b_result_json(
                &parsed,
                "512",
                Some((16, 0.38)),
                Some(373.6842105263158),
                Some((1, 4.18)),
            ),
            "{\"benchmark\":\"llama32-1b-prefill\",\"target\":\"llama32-1b\",\"status\":\"ok\",\"model\":\"/models/Llama-3.2-1B-Instruct-Q4_0.gguf\",\"selected_source\":\"explicit argument\",\"quantization\":\"q4_0\",\"probe\":true,\"shape\":\"llama32_1b\",\"shape_ready\":true,\"context_limit\":\"512\",\"prompt\":\"Say hello\",\"max_tokens\":2,\"temp\":0.0,\"batches\":[1,16],\"best_prefill_batch\":16,\"best_prefill_sec\":0.380000,\"best_prefill_prompt_tokens_per_sec\":373.684211,\"best_decode_batch\":1,\"best_tokens_per_sec\":4.180000}"
        );
    }

    #[test]
    fn ready_1b_args_accept_no_chat_flag_after_chat_args() {
        let parsed = parse_ready_1b_args_with_env(
            &[
                "/models/custom.GGUF".to_owned(),
                "chat".to_owned(),
                "Say hi".to_owned(),
                "4".to_owned(),
                "--no-chat".to_owned(),
            ],
            Some("/models/env.gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect("ready args should parse");

        assert_eq!(parsed.smoke.kind, SmokeKind::Q8Chat);
        assert_eq!(parsed.smoke.model_path, "/models/custom.GGUF");
        assert_eq!(parsed.smoke.prompt, "Say hi");
        assert_eq!(parsed.smoke.max_tokens, 4);
        assert_eq!(parsed.chat_enabled_override, Some(false));
        assert_eq!(parsed.chat_prompt_override, None);
        assert_eq!(parsed.chat_tokens_override, None);
    }

    #[test]
    fn ready_1b_args_accept_smoke_only_before_kind() {
        let parsed = parse_ready_1b_args_with_env(
            &[
                "--smoke-only".to_owned(),
                "model".to_owned(),
                "Hello".to_owned(),
                "2".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("ready smoke-only args should parse");

        assert_eq!(parsed.smoke.kind, SmokeKind::Q8Model);
        assert_eq!(
            parsed.smoke.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q8_MODEL}")
        );
        assert_eq!(parsed.smoke.prompt, "Hello");
        assert_eq!(parsed.smoke.max_tokens, 2);
        assert_eq!(parsed.chat_enabled_override, Some(false));
        assert_eq!(parsed.chat_prompt_override, None);
        assert_eq!(parsed.chat_tokens_override, None);
    }

    #[test]
    fn ready_1b_args_no_chat_uses_positionals_for_smoke_gate() {
        let parsed = parse_ready_1b_args_with_env_and_smoke_defaults(
            &[
                "chat".to_owned(),
                "Smoke this".to_owned(),
                "3".to_owned(),
                "--no-chat".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            true,
            SmokeDefaults {
                kind: SmokeKind::Q8Model,
                prompt: "Default smoke".to_owned(),
                max_tokens: 2,
            },
        )
        .expect("no-chat ready args should parse");

        assert_eq!(parsed.smoke.kind, SmokeKind::Q8Chat);
        assert_eq!(parsed.smoke.prompt, "Smoke this");
        assert_eq!(parsed.smoke.max_tokens, 3);
        assert_eq!(parsed.chat_enabled_override, Some(false));
        assert_eq!(parsed.chat_prompt_override, None);
        assert_eq!(parsed.chat_tokens_override, None);
    }

    #[test]
    fn ready_1b_args_env_disabled_chat_uses_positionals_for_smoke_gate() {
        let parsed = parse_ready_1b_args_with_env_and_smoke_defaults_and_chat_default(
            &["chat".to_owned(), "Smoke this".to_owned(), "3".to_owned()],
            None,
            "/mnt/nanocamelid",
            true,
            SmokeDefaults {
                kind: SmokeKind::Q8Model,
                prompt: "Default smoke".to_owned(),
                max_tokens: 2,
            },
            false,
        )
        .expect("env-disabled ready args should parse");

        assert_eq!(parsed.smoke.kind, SmokeKind::Q8Chat);
        assert_eq!(parsed.smoke.prompt, "Smoke this");
        assert_eq!(parsed.smoke.max_tokens, 3);
        assert_eq!(parsed.chat_enabled_override, None);
        assert_eq!(parsed.chat_prompt_override, None);
        assert_eq!(parsed.chat_tokens_override, None);
    }

    #[test]
    fn ready_1b_args_accept_chat_flag_before_kind() {
        let parsed = parse_ready_1b_args_with_env(
            &["--chat".to_owned(), "model".to_owned(), "Hello".to_owned()],
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect("ready chat override args should parse");

        assert_eq!(parsed.smoke.kind, SmokeKind::Q8Model);
        assert_eq!(
            parsed.smoke.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q8_MODEL}")
        );
        assert_eq!(parsed.smoke.prompt, DEFAULT_1B_SMOKE_PROMPT);
        assert_eq!(parsed.chat_enabled_override, Some(true));
        assert_eq!(parsed.chat_prompt_override, Some("Hello".to_owned()));
        assert!(!parsed.dry_run);
    }

    #[test]
    fn ready_1b_args_accept_dry_run_with_other_flags() {
        let parsed = parse_ready_1b_args_with_env(
            &[
                "--dry-run".to_owned(),
                "/models/custom.GGUF".to_owned(),
                "q8-chat".to_owned(),
                "Say hi".to_owned(),
                "4".to_owned(),
                "--smoke-only".to_owned(),
            ],
            Some("/models/env.gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect("dry-run ready args should parse");

        assert!(parsed.dry_run);
        assert_eq!(parsed.smoke.kind, SmokeKind::Q8Chat);
        assert_eq!(parsed.smoke.model_path, "/models/custom.GGUF");
        assert_eq!(parsed.smoke.model_source, "explicit argument");
        assert_eq!(parsed.smoke.prompt, "Say hi");
        assert_eq!(parsed.smoke.max_tokens, 4);
        assert_eq!(parsed.chat_enabled_override, Some(false));
        assert_eq!(parsed.chat_prompt_override, None);
        assert_eq!(parsed.chat_tokens_override, None);
    }

    #[test]
    fn ready_1b_args_quant_selector_forces_workspace_row() {
        let parsed = parse_ready_1b_args_with_env(
            &["--q4".to_owned(), "--dry-run".to_owned()],
            Some("/models/env.gguf".to_owned()),
            "/mnt/nanocamelid",
            false,
        )
        .expect("q4 readiness selector should parse");

        assert_eq!(
            parsed.smoke.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(parsed.smoke.model_source, "workspace Q4_0 requested");
        assert!(parsed.dry_run);
    }

    #[test]
    fn ready_1b_args_explicit_model_overrides_quant_selector() {
        let parsed = parse_ready_1b_args_with_env(
            &[
                "/models/custom.GGUF".to_owned(),
                "--q8".to_owned(),
                "--dry-run".to_owned(),
            ],
            Some("/models/env.gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect("explicit readiness model should parse");

        assert_eq!(parsed.smoke.model_path, "/models/custom.GGUF");
        assert_eq!(parsed.smoke.model_source, "explicit argument");
        assert!(parsed.dry_run);
    }

    #[test]
    fn ready_1b_args_reject_conflicting_quant_selectors() {
        let err = parse_ready_1b_args_with_env(
            &["--q4".to_owned(), "--q8".to_owned(), "--dry-run".to_owned()],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("conflicting readiness selectors should fail");

        assert_eq!(err, "ready 1B accepts only one quantization selector");
    }

    #[test]
    fn ready_1b_args_reject_non_gguf_env_model_path() {
        let err = parse_ready_1b_args_with_env(
            &["--dry-run".to_owned()],
            Some("/models/not-a-gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("non-GGUF env readiness path should fail");

        assert_eq!(err, "ready 1B env model path must be a .gguf path");
    }

    #[test]
    fn ready_1b_args_reject_path_like_non_gguf_model_arg() {
        let err = parse_ready_1b_args_with_env(
            &["/models/not-a-gguf".to_owned(), "--dry-run".to_owned()],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("path-like non-GGUF ready arg should fail");

        assert_eq!(err, "ready 1B model argument must be a .gguf path");
    }

    #[test]
    fn ready_1b_args_leave_chat_default_without_flag() {
        let parsed = parse_ready_1b_args_with_env(&[], None, "/mnt/nanocamelid", true)
            .expect("default ready args should parse");

        assert_eq!(parsed.smoke.kind, SmokeKind::Q8Chat);
        assert_eq!(
            parsed.smoke.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(parsed.chat_enabled_override, None);
        assert_eq!(parsed.chat_prompt_override, None);
        assert_eq!(parsed.chat_tokens_override, None);
        assert!(!parsed.dry_run);
    }

    #[test]
    fn ready_1b_args_use_ready_smoke_defaults_when_positionals_are_missing() {
        let parsed = parse_ready_1b_args_with_env_and_smoke_defaults(
            &[],
            None,
            "/mnt/nanocamelid",
            true,
            SmokeDefaults {
                kind: SmokeKind::Q8Model,
                prompt: "Hello".to_owned(),
                max_tokens: 2,
            },
        )
        .expect("ready smoke defaults should parse");

        assert_eq!(parsed.smoke.kind, SmokeKind::Q8Model);
        assert_eq!(parsed.smoke.prompt, "Hello");
        assert_eq!(parsed.smoke.max_tokens, 2);
        assert_eq!(parsed.chat_enabled_override, None);
        assert_eq!(parsed.chat_prompt_override, None);
        assert_eq!(parsed.chat_tokens_override, None);
    }

    #[test]
    fn ready_1b_args_positionals_override_direct_chat_defaults() {
        let parsed = parse_ready_1b_args_with_env_and_smoke_defaults(
            &["chat".to_owned(), "Say hi".to_owned(), "4".to_owned()],
            None,
            "/mnt/nanocamelid",
            true,
            SmokeDefaults {
                kind: SmokeKind::Q8Model,
                prompt: "Hello".to_owned(),
                max_tokens: 2,
            },
        )
        .expect("ready chat positional overrides should parse");

        assert_eq!(parsed.smoke.kind, SmokeKind::Q8Chat);
        assert_eq!(parsed.smoke.prompt, "Hello");
        assert_eq!(parsed.smoke.max_tokens, 2);
        assert_eq!(parsed.chat_prompt_override, Some("Say hi".to_owned()));
        assert_eq!(parsed.chat_tokens_override, Some(4));
    }

    #[test]
    fn ready_1b_args_reject_invalid_direct_chat_token_count() {
        let err = parse_ready_1b_args_with_env(
            &["chat".to_owned(), "Say hi".to_owned(), "0".to_owned()],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("zero-token ready chat should fail");

        assert_eq!(err, "ready 1B token count must be a positive integer");
    }

    #[test]
    fn ready_1b_args_reject_extra_positionals_after_token_count() {
        let err = parse_ready_1b_args_with_env(
            &[
                "chat".to_owned(),
                "Say hi".to_owned(),
                "4".to_owned(),
                "unexpected".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("extra ready 1B arg should fail");

        assert_eq!(err, "unexpected extra ready 1B argument");
    }

    #[test]
    fn smoke_1b_args_reject_unknown_q8_kind() {
        let err =
            parse_smoke_1b_args_with_env(&["q8-broken".to_owned()], None, "/mnt/nanocamelid", true)
                .expect_err("unknown q8 kind should fail");

        assert_eq!(
            err,
            "unknown 1B smoke kind; expected chat, model, q8-chat, or q8-model"
        );
    }

    #[test]
    fn smoke_1b_args_quant_selector_forces_workspace_row() {
        let parsed = parse_smoke_1b_args_with_env(
            &["--q4".to_owned(), "--dry-run".to_owned()],
            Some("/models/env.gguf".to_owned()),
            "/mnt/nanocamelid",
            false,
        )
        .expect("q4 smoke selector should parse");

        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_1B_Q4_MODEL}")
        );
        assert_eq!(parsed.model_source, "workspace Q4_0 requested");
        assert!(parsed.dry_run);
    }

    #[test]
    fn smoke_1b_args_explicit_model_overrides_quant_selector() {
        let parsed = parse_smoke_1b_args_with_env(
            &[
                "/models/custom.GGUF".to_owned(),
                "--q8".to_owned(),
                "--dry-run".to_owned(),
            ],
            Some("/models/env.gguf".to_owned()),
            "/mnt/nanocamelid",
            true,
        )
        .expect("explicit smoke model should parse");

        assert_eq!(parsed.model_path, "/models/custom.GGUF");
        assert_eq!(parsed.model_source, "explicit argument");
        assert!(parsed.dry_run);
    }

    #[test]
    fn smoke_1b_args_reject_conflicting_quant_selectors() {
        let err = parse_smoke_1b_args_with_env(
            &["--q4".to_owned(), "--q8".to_owned(), "--dry-run".to_owned()],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("conflicting smoke selectors should fail");

        assert_eq!(err, "1B smoke accepts only one quantization selector");
    }

    #[test]
    fn smoke_1b_args_reject_invalid_token_count() {
        let err = parse_smoke_1b_args_with_env(
            &["chat".to_owned(), "Hello".to_owned(), "bad".to_owned()],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("invalid 1B smoke token count should fail");

        assert_eq!(err, "1B smoke max_tokens must be a positive integer");
    }

    #[test]
    fn smoke_1b_args_reject_extra_positionals_after_token_count() {
        let err = parse_smoke_1b_args_with_env(
            &[
                "chat".to_owned(),
                "Hello".to_owned(),
                "2".to_owned(),
                "unexpected".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
            true,
        )
        .expect_err("extra 1B smoke arg should fail");

        assert_eq!(err, "unexpected extra 1B smoke argument");
    }

    #[test]
    fn smoke_3b_args_default_to_chat_prompt_and_pi_model() {
        let parsed = parse_smoke_3b_args_with_env(&[], None, "/mnt/nanocamelid")
            .expect("default 3B smoke args should parse");

        assert_eq!(parsed.kind, SmokeKind::Q8Chat);
        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_3B_Q4_MODEL}")
        );
        assert_eq!(parsed.prompt, DEFAULT_1B_SMOKE_PROMPT);
        assert_eq!(parsed.max_tokens, DEFAULT_1B_SMOKE_TOKENS);
    }

    #[test]
    fn smoke_3b_args_prefer_explicit_gguf_over_env_and_defaults() {
        let parsed = parse_smoke_3b_args_with_env(
            &[
                "/models/custom-3b.gguf".to_owned(),
                "q8-model".to_owned(),
                "Hello".to_owned(),
                "2".to_owned(),
            ],
            Some("/models/env-3b.gguf".to_owned()),
            "/mnt/nanocamelid",
        )
        .expect("explicit 3B smoke model should parse");

        assert_eq!(parsed.kind, SmokeKind::Q8Model);
        assert_eq!(parsed.model_path, "/models/custom-3b.gguf");
        assert_eq!(parsed.prompt, "Hello");
        assert_eq!(parsed.max_tokens, 2);
    }

    #[test]
    fn smoke_3b_args_accept_dry_run_without_model_file() {
        let parsed = parse_smoke_3b_args_with_env(
            &[
                "--dry-run".to_owned(),
                "chat".to_owned(),
                "Hello".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
        )
        .expect("dry-run 3B smoke args should parse");

        assert!(parsed.dry_run);
        assert_eq!(parsed.kind, SmokeKind::Q8Chat);
        assert_eq!(
            parsed.model_path,
            format!("/mnt/nanocamelid/models/{LLAMA32_3B_Q4_MODEL}")
        );
        assert_eq!(parsed.prompt, "Hello");
        assert_eq!(parsed.max_tokens, DEFAULT_1B_SMOKE_TOKENS);
    }

    #[test]
    fn smoke_3b_args_reject_path_like_non_gguf_model_arg() {
        let err = parse_smoke_3b_args_with_env(
            &["models/not-a-gguf.bin".to_owned(), "--dry-run".to_owned()],
            None,
            "/mnt/nanocamelid",
        )
        .expect_err("path-like non-GGUF 3B smoke arg should fail");

        assert_eq!(err, "3B smoke model argument must be a .gguf path");
    }

    #[test]
    fn smoke_3b_args_reject_unknown_q8_kind() {
        let err = parse_smoke_3b_args_with_env(&["q8-broken".to_owned()], None, "/mnt/nanocamelid")
            .expect_err("unknown q8 kind should fail");

        assert_eq!(
            err,
            "unknown 3B smoke kind; expected chat, model, q8-chat, or q8-model"
        );
    }

    #[test]
    fn smoke_3b_args_reject_invalid_token_count() {
        let err = parse_smoke_3b_args_with_env(
            &["chat".to_owned(), "Hello".to_owned(), "0".to_owned()],
            None,
            "/mnt/nanocamelid",
        )
        .expect_err("zero-token 3B smoke should fail");

        assert_eq!(err, "3B smoke max_tokens must be a positive integer");
    }

    #[test]
    fn smoke_3b_args_reject_extra_positionals_after_token_count() {
        let err = parse_smoke_3b_args_with_env(
            &[
                "chat".to_owned(),
                "Hello".to_owned(),
                "2".to_owned(),
                "unexpected".to_owned(),
            ],
            None,
            "/mnt/nanocamelid",
        )
        .expect_err("extra 3B smoke arg should fail");

        assert_eq!(err, "unexpected extra 3B smoke argument");
    }

    #[test]
    fn inspect_runtime_summary_reports_ready_q8_llama_fixture() {
        let summary = inspect_runtime_summary(&inspect_fixture(false));
        assert!(summary.ready);
        assert!(summary.tensor_layouts.is_ok());
        assert!(summary.tied_output);
        assert!(summary.unsupported_tensor_types.is_empty());
        assert!(summary.supported_tensor_types.contains(&"Q8_0".to_owned()));
        assert_eq!(
            summary
                .model_config
                .expect("config should parse for ready fixture")
                .vocab_size,
            64
        );
        assert_eq!(
            summary
                .tokenizer
                .expect("tokenizer should parse for ready fixture")
                .model,
            TokenizerModel::LlamaSpm
        );
    }

    #[test]
    fn inspect_runtime_summary_surfaces_tensor_layout_errors() {
        let summary = inspect_runtime_summary(&inspect_fixture(true));
        assert!(!summary.ready);
        let err = summary
            .tensor_layouts
            .expect_err("broken fixture should fail tensor layout validation");
        assert!(err.contains("blk.0.ffn_down.weight"));
    }

    #[test]
    fn inspect_runtime_summary_reports_unsupported_tensor_types() {
        let mut fixture = inspect_fixture(false);
        fixture.tensors.push(tensor_desc(
            "unsupported.weight",
            vec![32, 32],
            GgufTensorType::Q8_1,
            36,
        ));

        let summary = inspect_runtime_summary(&fixture);
        assert!(!summary.ready);
        assert_eq!(summary.unsupported_tensor_types, vec!["Q8_1".to_owned()]);
    }

    #[test]
    fn inspect_runtime_summary_accepts_low_bit_k_tensor_types() {
        let mut fixture = inspect_fixture(false);
        fixture.tensors.push(tensor_desc(
            "q2.weight",
            vec![256, 1],
            GgufTensorType::Q2K,
            84,
        ));
        fixture.tensors.push(tensor_desc(
            "q3.weight",
            vec![256, 1],
            GgufTensorType::Q3K,
            110,
        ));

        let summary = inspect_runtime_summary(&fixture);
        assert!(summary.unsupported_tensor_types.is_empty());
        assert!(summary.supported_tensor_types.contains(&"Q2_K".to_owned()));
        assert!(summary.supported_tensor_types.contains(&"Q3_K".to_owned()));
    }

    #[test]
    fn inspect_runtime_summary_accepts_q8_k_and_iq4_nl_tensor_types() {
        let mut fixture = inspect_fixture(false);
        fixture.tensors.push(tensor_desc(
            "q8_k.weight",
            vec![256, 1],
            GgufTensorType::Q8K,
            292,
        ));
        fixture.tensors.push(tensor_desc(
            "iq4_nl.weight",
            vec![32, 1],
            GgufTensorType::IQ4NL,
            18,
        ));

        let summary = inspect_runtime_summary(&fixture);
        assert!(summary.unsupported_tensor_types.is_empty());
        assert!(summary.supported_tensor_types.contains(&"Q8_K".to_owned()));
        assert!(
            summary
                .supported_tensor_types
                .contains(&"IQ4_NL".to_owned())
        );
    }

    #[test]
    fn llama32_1b_shape_audit_accepts_expected_metadata() {
        let fixture = llama32_1b_shape_fixture();
        let audit = llama32_1b_shape_audit(&fixture);

        assert_eq!(audit.label, "llama32_1b");
        assert!(audit.ready);
        assert!(audit.mismatches.is_empty());
    }

    #[test]
    fn llama32_1b_shape_audit_reports_shape_mismatches() {
        let mut fixture = llama32_1b_shape_fixture();
        fixture
            .metadata
            .insert("llama.block_count".to_owned(), GgufMetadataValue::U32(15));
        fixture.metadata.insert(
            "llama.vocab_size".to_owned(),
            GgufMetadataValue::U32(32_000),
        );

        let audit = llama32_1b_shape_audit(&fixture);

        assert!(!audit.ready);
        assert!(
            audit
                .mismatches
                .contains(&"block_count expected 16 got 15".to_owned())
        );
        assert!(
            audit
                .mismatches
                .contains(&"vocab_size expected 128256 got 32000".to_owned())
        );
    }

    #[test]
    fn llama32_1b_shape_audit_reports_tensor_shape_mismatches() {
        let mut fixture = llama32_1b_shape_fixture();
        set_tensor_dimensions(&mut fixture, "token_embd.weight", vec![2_048, 128_000]);

        let audit = llama32_1b_shape_audit(&fixture);

        assert!(!audit.ready);
        assert!(audit.mismatches.contains(
            &"token_embd.weight dims expected [2048, 128256] got [2048, 128000]".to_owned()
        ));
    }

    #[test]
    fn llama32_1b_shape_audit_accepts_output_projection_when_present() {
        let mut fixture = llama32_1b_shape_fixture();
        fixture.tensors.push(tensor_desc(
            "output.weight",
            vec![128_256, 2_048],
            GgufTensorType::Q8_0,
            q8_bytes(2_048, 128_256),
        ));
        fixture.tensor_count = fixture.tensors.len() as u64;

        let audit = llama32_1b_shape_audit(&fixture);

        assert!(audit.ready, "{:?}", audit.mismatches);
    }

    #[test]
    fn llama32_1b_shape_audit_reports_output_projection_mismatches() {
        let mut fixture = llama32_1b_shape_fixture();
        fixture.tensors.push(tensor_desc(
            "output.weight",
            vec![128_000, 2_048],
            GgufTensorType::Q8_0,
            q8_bytes(2_048, 128_000),
        ));
        fixture.tensor_count = fixture.tensors.len() as u64;

        let audit = llama32_1b_shape_audit(&fixture);

        assert!(!audit.ready);
        assert!(
            audit.mismatches.contains(
                &"output.weight dims expected [2048, 128256] or [128256, 2048] got [128000, 2048]"
                    .to_owned()
            )
        );
    }

    #[test]
    fn llama32_1b_shape_audit_checks_every_block_tensor_shape() {
        let mut fixture = llama32_1b_shape_fixture();
        set_tensor_dimensions(&mut fixture, "blk.15.ffn_down.weight", vec![8_192, 4_096]);

        let audit = llama32_1b_shape_audit(&fixture);

        assert!(!audit.ready);
        assert!(audit.mismatches.contains(
            &"blk.15.ffn_down.weight dims expected [8192, 2048] got [8192, 4096]".to_owned()
        ));
    }

    #[test]
    fn llama32_1b_shape_audit_reports_missing_later_block_tensor() {
        let mut fixture = llama32_1b_shape_fixture();
        fixture
            .tensors
            .retain(|tensor| tensor.name != "blk.12.attn_q.weight");
        fixture.tensor_count = fixture.tensors.len() as u64;

        let audit = llama32_1b_shape_audit(&fixture);

        assert!(!audit.ready);
        assert!(
            audit
                .mismatches
                .contains(&"blk.12.attn_q.weight missing".to_owned())
        );
    }

    #[test]
    fn llama32_1b_shape_audit_requires_llama3_instruct_renderer() {
        let mut fixture = llama32_1b_shape_fixture();
        fixture.metadata.insert(
            "tokenizer.chat_template".to_owned(),
            GgufMetadataValue::String("{{ bos_token }}{{ messages }}".to_owned()),
        );

        let audit = llama32_1b_shape_audit(&fixture);

        assert!(!audit.ready);
        assert!(
            audit.mismatches.contains(
                &"tokenizer_chat_template_format expected llama3_instruct got metadata_unparsed"
                    .to_owned()
            )
        );
    }

    #[test]
    fn runtime_options_read_qwen2_rope_scaling_metadata() {
        let mut fixture = inspect_fixture(false);
        fixture.metadata.insert(
            "general.architecture".to_owned(),
            GgufMetadataValue::String("qwen2".to_owned()),
        );
        fixture.metadata.remove("llama.context_length");
        fixture.metadata.remove("llama.embedding_length");
        fixture.metadata.remove("llama.block_count");
        fixture.metadata.remove("llama.feed_forward_length");
        fixture.metadata.remove("llama.attention.head_count");
        fixture.metadata.remove("llama.attention.head_count_kv");
        fixture.metadata.remove("llama.vocab_size");
        fixture.metadata.insert(
            "qwen2.context_length".to_owned(),
            GgufMetadataValue::U32(128),
        );
        fixture.metadata.insert(
            "qwen2.embedding_length".to_owned(),
            GgufMetadataValue::U32(32),
        );
        fixture
            .metadata
            .insert("qwen2.block_count".to_owned(), GgufMetadataValue::U32(1));
        fixture.metadata.insert(
            "qwen2.feed_forward_length".to_owned(),
            GgufMetadataValue::U32(64),
        );
        fixture.metadata.insert(
            "qwen2.attention.head_count".to_owned(),
            GgufMetadataValue::U32(4),
        );
        fixture.metadata.insert(
            "qwen2.attention.head_count_kv".to_owned(),
            GgufMetadataValue::U32(4),
        );
        fixture
            .metadata
            .insert("qwen2.vocab_size".to_owned(), GgufMetadataValue::U32(64));
        fixture.metadata.insert(
            "qwen2.rope.scaling.factor".to_owned(),
            GgufMetadataValue::F32(2.0),
        );
        fixture.metadata.insert(
            "qwen2.rope.scaling.original_context_length".to_owned(),
            GgufMetadataValue::U32(4096),
        );
        fixture.metadata.insert(
            "qwen2.rope.scaling.low_freq_factor".to_owned(),
            GgufMetadataValue::F32(1.0),
        );
        fixture.metadata.insert(
            "qwen2.rope.scaling.high_freq_factor".to_owned(),
            GgufMetadataValue::F32(4.0),
        );

        let runtime = runtime_options_from_gguf(
            &fixture,
            q8::Q8DotKernelSelector {
                requested: None,
                selected: q8::Q8DotKernel::Scalar,
                fallback_reason: None,
            },
        );
        assert_eq!(runtime.rope_scaling.factor, Some(2.0));
        assert_eq!(runtime.rope_scaling.original_context_length, Some(4096.0));
        assert_eq!(runtime.rope_scaling.low_freq_factor, Some(1.0));
        assert_eq!(runtime.rope_scaling.high_freq_factor, Some(4.0));
    }

    #[test]
    fn runtime_options_read_mistral_rope_scaling_metadata() {
        let mut fixture = inspect_fixture(false);
        fixture.metadata.insert(
            "general.architecture".to_owned(),
            GgufMetadataValue::String("mistral".to_owned()),
        );
        fixture.metadata.insert(
            "mistral.rope.scaling.factor".to_owned(),
            GgufMetadataValue::F32(1.5),
        );
        fixture.metadata.insert(
            "mistral.rope.scaling.original_context_length".to_owned(),
            GgufMetadataValue::U32(8192),
        );

        let runtime = runtime_options_from_gguf(
            &fixture,
            q8::Q8DotKernelSelector {
                requested: None,
                selected: q8::Q8DotKernel::Scalar,
                fallback_reason: None,
            },
        );
        assert_eq!(runtime.rope_scaling.factor, Some(1.5));
        assert_eq!(runtime.rope_scaling.original_context_length, Some(8192.0));
    }

    fn inspect_fixture(break_ffn_down: bool) -> GgufFile {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            GgufMetadataValue::String("llama".to_owned()),
        );
        metadata.insert(
            "llama.context_length".to_owned(),
            GgufMetadataValue::U32(128),
        );
        metadata.insert(
            "llama.embedding_length".to_owned(),
            GgufMetadataValue::U32(32),
        );
        metadata.insert("llama.block_count".to_owned(), GgufMetadataValue::U32(1));
        metadata.insert(
            "llama.feed_forward_length".to_owned(),
            GgufMetadataValue::U32(64),
        );
        metadata.insert(
            "llama.attention.head_count".to_owned(),
            GgufMetadataValue::U32(4),
        );
        metadata.insert(
            "llama.attention.head_count_kv".to_owned(),
            GgufMetadataValue::U32(4),
        );
        metadata.insert("llama.vocab_size".to_owned(), GgufMetadataValue::U32(64));
        metadata.insert(
            "tokenizer.ggml.model".to_owned(),
            GgufMetadataValue::String("llama".to_owned()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            GgufMetadataValue::Array(vec![
                GgufMetadataValue::String("<unk>".to_owned()),
                GgufMetadataValue::String("<s>".to_owned()),
                GgufMetadataValue::String("</s>".to_owned()),
                GgufMetadataValue::String("▁hello".to_owned()),
                GgufMetadataValue::String("hello".to_owned()),
                GgufMetadataValue::String("▁".to_owned()),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.scores".to_owned(),
            GgufMetadataValue::Array(vec![
                GgufMetadataValue::F32(0.0),
                GgufMetadataValue::F32(0.0),
                GgufMetadataValue::F32(0.0),
                GgufMetadataValue::F32(10.0),
                GgufMetadataValue::F32(2.0),
                GgufMetadataValue::F32(1.0),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_owned(),
            GgufMetadataValue::Array(vec![
                GgufMetadataValue::I32(2),
                GgufMetadataValue::I32(3),
                GgufMetadataValue::I32(3),
                GgufMetadataValue::I32(1),
                GgufMetadataValue::I32(1),
                GgufMetadataValue::I32(1),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_owned(),
            GgufMetadataValue::U32(1),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_owned(),
            GgufMetadataValue::U32(2),
        );
        metadata.insert(
            "tokenizer.chat_template".to_owned(),
            GgufMetadataValue::String("{{ bos_token }}{{ messages }}".to_owned()),
        );

        let ffn_down_dims = if break_ffn_down {
            vec![64, 64]
        } else {
            vec![64, 32]
        };

        let tensors = vec![
            tensor_desc(
                "token_embd.weight",
                vec![32, 64],
                GgufTensorType::F16,
                f16_bytes(32, 64),
            ),
            tensor_desc("output_norm.weight", vec![32], GgufTensorType::F32, 32 * 4),
            tensor_desc(
                "blk.0.attn_norm.weight",
                vec![32],
                GgufTensorType::F32,
                32 * 4,
            ),
            tensor_desc(
                "blk.0.attn_q.weight",
                vec![32, 32],
                GgufTensorType::Q8_0,
                q8_bytes(32, 32),
            ),
            tensor_desc(
                "blk.0.attn_k.weight",
                vec![32, 32],
                GgufTensorType::Q8_0,
                q8_bytes(32, 32),
            ),
            tensor_desc(
                "blk.0.attn_v.weight",
                vec![32, 32],
                GgufTensorType::Q8_0,
                q8_bytes(32, 32),
            ),
            tensor_desc(
                "blk.0.attn_output.weight",
                vec![32, 32],
                GgufTensorType::Q8_0,
                q8_bytes(32, 32),
            ),
            tensor_desc(
                "blk.0.ffn_norm.weight",
                vec![32],
                GgufTensorType::F32,
                32 * 4,
            ),
            tensor_desc(
                "blk.0.ffn_gate.weight",
                vec![32, 64],
                GgufTensorType::Q8_0,
                q8_bytes(32, 64),
            ),
            tensor_desc(
                "blk.0.ffn_up.weight",
                vec![32, 64],
                GgufTensorType::Q8_0,
                q8_bytes(32, 64),
            ),
            tensor_desc(
                "blk.0.ffn_down.weight",
                ffn_down_dims,
                GgufTensorType::Q8_0,
                q8_bytes(64, 32),
            ),
        ];

        GgufFile {
            path: PathBuf::from("inspect-fixture.gguf"),
            version: 3,
            tensor_count: tensors.len() as u64,
            metadata_count: metadata.len() as u64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors,
        }
    }

    fn llama32_1b_shape_fixture() -> GgufFile {
        let mut fixture = inspect_fixture(false);
        fixture.metadata.insert(
            "llama.context_length".to_owned(),
            GgufMetadataValue::U32(131_072),
        );
        fixture.metadata.insert(
            "llama.embedding_length".to_owned(),
            GgufMetadataValue::U32(2_048),
        );
        fixture
            .metadata
            .insert("llama.block_count".to_owned(), GgufMetadataValue::U32(16));
        fixture.metadata.insert(
            "llama.feed_forward_length".to_owned(),
            GgufMetadataValue::U32(8_192),
        );
        fixture.metadata.insert(
            "llama.attention.head_count".to_owned(),
            GgufMetadataValue::U32(32),
        );
        fixture.metadata.insert(
            "llama.attention.head_count_kv".to_owned(),
            GgufMetadataValue::U32(8),
        );
        fixture.metadata.insert(
            "llama.rope.dimension_count".to_owned(),
            GgufMetadataValue::U32(64),
        );
        fixture.metadata.insert(
            "llama.rope.freq_base".to_owned(),
            GgufMetadataValue::F32(500_000.0),
        );
        fixture.metadata.insert(
            "llama.vocab_size".to_owned(),
            GgufMetadataValue::U32(128_256),
        );
        fixture.metadata.insert(
            "tokenizer.chat_template".to_owned(),
            GgufMetadataValue::String(
                "{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}<|start_header_id|>assistant<|end_header_id|>\n\n"
                    .to_owned(),
            ),
        );
        set_tensor_dimensions(&mut fixture, "token_embd.weight", vec![2_048, 128_256]);
        set_tensor_dimensions(&mut fixture, "output_norm.weight", vec![2_048]);
        set_tensor_dimensions(&mut fixture, "blk.0.attn_norm.weight", vec![2_048]);
        set_tensor_dimensions(&mut fixture, "blk.0.attn_q.weight", vec![2_048, 2_048]);
        set_tensor_dimensions(&mut fixture, "blk.0.attn_k.weight", vec![2_048, 512]);
        set_tensor_dimensions(&mut fixture, "blk.0.attn_v.weight", vec![2_048, 512]);
        set_tensor_dimensions(&mut fixture, "blk.0.attn_output.weight", vec![2_048, 2_048]);
        set_tensor_dimensions(&mut fixture, "blk.0.ffn_norm.weight", vec![2_048]);
        set_tensor_dimensions(&mut fixture, "blk.0.ffn_gate.weight", vec![2_048, 8_192]);
        set_tensor_dimensions(&mut fixture, "blk.0.ffn_up.weight", vec![2_048, 8_192]);
        set_tensor_dimensions(&mut fixture, "blk.0.ffn_down.weight", vec![8_192, 2_048]);
        expand_llama32_1b_block_tensors(&mut fixture);
        fixture
    }

    fn expand_llama32_1b_block_tensors(gguf: &mut GgufFile) {
        let block_zero_tensors = gguf
            .tensors
            .iter()
            .filter(|tensor| tensor.name.starts_with("blk.0."))
            .cloned()
            .collect::<Vec<_>>();

        for layer_idx in 1..16 {
            for tensor in &block_zero_tensors {
                let mut layer_tensor = tensor.clone();
                layer_tensor.name = tensor
                    .name
                    .replacen("blk.0.", &format!("blk.{layer_idx}."), 1);
                gguf.tensors.push(layer_tensor);
            }
        }

        gguf.tensor_count = gguf.tensors.len() as u64;
    }

    fn set_tensor_dimensions(gguf: &mut GgufFile, name: &str, dimensions: Vec<u64>) {
        if let Some(tensor) = gguf.tensors.iter_mut().find(|tensor| tensor.name == name) {
            tensor.dimensions = dimensions;
        }
    }

    fn tensor_desc(
        name: &str,
        dimensions: Vec<u64>,
        tensor_type: GgufTensorType,
        n_bytes: u64,
    ) -> GgufTensorDescriptor {
        GgufTensorDescriptor {
            name: name.to_owned(),
            dimensions,
            tensor_type,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes,
        }
    }

    fn q8_bytes(row_values: u64, row_count: u64) -> u64 {
        row_values / 32 * 34 * row_count
    }

    fn f16_bytes(row_values: u64, row_count: u64) -> u64 {
        row_values * row_count * 2
    }
}
