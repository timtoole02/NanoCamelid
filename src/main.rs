use std::{
    collections::BTreeSet,
    env, fs,
    hint::black_box,
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    time::Duration,
};

use nanocamelid::{gguf, inference, model, q8, speculative, tokenizer};

const DEFAULT_MODEL_GGUF_ENV: &str = "NANOCAMELID_MODEL_GGUF";
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
const DEFAULT_PI_WORKSPACE: &str = "/mnt/nanocamelid";
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
    println!();
    println!("Commands:");
    println!(
        "  probe                                     Print host CPU and runtime feature information"
    );
    println!("  model 1b [model.gguf] [--dry-run]");
    println!(
        "                                            Audit the default Llama 3.2 1B model path"
    );
    println!(
        "  inspect <model.gguf>                      Inspect GGUF metadata and tensor layouts"
    );
    println!("  inspect 1b [--dry-run]                    Inspect the default Llama 3.2 1B path");
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
        "  ready 1b [model.gguf] [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--no-chat|--smoke-only|--chat|--dry-run]"
    );
    println!(
        "                                            Run inspect, smoke, and direct chat gates for 1B"
    );
    println!("  evidence 1b [model.gguf] [--dry-run]");
    println!("                                            Run the bounded 1B evidence bundle");
    println!("  bench q8-dot [iterations] [runs]          Benchmark Q8 dot product kernels");
    println!("  bench 1b [model.gguf] [prompt] [max_tokens] [temp] [batches] [--dry-run]");
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
    println!();
    println!("Run `nanocamelid help <command>` or `nanocamelid <command> --help` for details.");
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
    println!("  nanocamelid model 1b [model.gguf] [--dry-run]");
    println!("  nanocamelid model llama32-1b [model.gguf] [--dry-run]");
    println!();
    println!(
        "Audit the Llama 3.2 1B model selection plan and verify that the selected GGUF exists."
    );
    println!();
    println!("Options:");
    println!(
        "  --dry-run                                Print the audit without failing when the selected model is missing"
    );
    println!();
    println!("Env:");
    println!("  {SMOKE_MODEL_GGUF_ENV:<38} Override the 1B model audit GGUF path");
    println!("  {DEFAULT_MODEL_GGUF_ENV:<38} Shared default GGUF path for inspect/generate/smoke");
    println!("  {WORKSPACE_ENV:<38} Pi workspace for 1B defaults; default {DEFAULT_PI_WORKSPACE}");
    println!();
    println!("`1b` prefers the Pi-local Q4_0 Llama 3.2 1B GGUF, then falls back to Q8_0.");
}

fn print_inspect_usage() {
    println!("NanoCamelid inspect");
    println!();
    println!("Usage:");
    println!("  nanocamelid inspect <model.gguf> [--dry-run]");
    println!(
        "  nanocamelid inspect 1b [--dry-run]         inspect and strictly gate the Llama 3.2 1B path"
    );
    println!("  nanocamelid inspect 3b [--dry-run]         inspect the default Llama 3.2 3B path");
    println!("  nanocamelid inspect                        with NANOCAMELID_MODEL_GGUF set");
    println!();
    println!(
        "Inspect GGUF metadata, runtime-ready LLaMA config, tokenizer support, and the first tensor layouts."
    );
    println!("Use --dry-run to print the resolved inspect command without reading the GGUF.");
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
        "  nanocamelid ready 1b [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--no-chat|--smoke-only|--chat|--dry-run]"
    );
    println!(
        "  nanocamelid ready 1b <model.gguf> [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--no-chat|--smoke-only|--chat|--dry-run]"
    );
    println!();
    println!(
        "Run the Llama 3.2 1B readiness gate: audit shape, inspect metadata, smoke scalar-vs-selected logits, then run one direct chat turn."
    );
    println!();
    println!("Options:");
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
}

fn print_evidence_usage() {
    println!("NanoCamelid evidence");
    println!();
    println!("Usage:");
    println!("  nanocamelid evidence 1b [model.gguf] [--dry-run]");
    println!("  nanocamelid evidence llama32-1b [model.gguf] [--dry-run]");
    println!();
    println!(
        "Run the bounded Llama 3.2 1B evidence bundle: readiness without final chat, context-pack smoke, and prefill batch sweep."
    );
    println!();
    println!("Options:");
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
}

fn print_bench_usage() {
    println!("NanoCamelid bench");
    println!();
    println!("Usage:");
    println!("  nanocamelid bench q8-dot [iterations] [runs]");
    println!("  nanocamelid bench q4-layout [rows] [cols] [runs]");
    println!("  nanocamelid bench q4-prefill [prompt_len] [batch_size] [runs]");
    println!(
        "  nanocamelid bench 1b [model.gguf] [prompt] [max_tokens] [temp] [batches] [--dry-run]"
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
        "  nanocamelid smoke 1b [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--dry-run]"
    );
    println!(
        "  nanocamelid smoke 3b [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--dry-run]"
    );
    println!("  nanocamelid smoke q8-model [prompt] [max_tokens]  with NANOCAMELID_SMOKE_GGUF set");
    println!("  nanocamelid smoke q8-chat [prompt] [max_tokens]   with NANOCAMELID_SMOKE_GGUF set");
    println!(
        "  nanocamelid smoke 1b <model.gguf> [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--dry-run]"
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
    println!();
    println!(
        "When {SMOKE_MODEL_GGUF_ENV} or {DEFAULT_MODEL_GGUF_ENV} is set, the first positional argument is treated as the prompt unless it looks like a .gguf path."
    );
    println!();
    println!(
        "`q8-model` tokenizes the prompt directly. `q8-chat` renders a single-turn user message through the model tokenizer chat template before parity/generation."
    );
    println!(
        "`1b` defaults to chat, prompt {DEFAULT_1B_SMOKE_PROMPT:?}, and {DEFAULT_1B_SMOKE_TOKENS} tokens. It prefers the Pi-local Q4_0 Llama 3.2 1B GGUF, then Q8_0. The legacy q8-chat and q8-model aliases are still accepted."
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
    let mut positionals = Vec::with_capacity(args.len());
    for arg in args {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
            arg if arg.starts_with('-') => return Err("unknown inspect option"),
            _ => positionals.push(arg.clone()),
        }
    }

    if positionals.len() > 1 {
        return Err("unexpected extra inspect argument");
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
        let (model_path, model_source) =
            resolve_llama32_1b_model_path_and_source(alias_env_model_path, workspace, q4_exists)?;
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
    let mut positionals = Vec::with_capacity(args.len());

    for arg in args {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
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
    let mut smoke_args = Vec::with_capacity(args.len());

    for arg in args {
        match arg.as_str() {
            "--no-chat" | "--smoke-only" => chat_enabled_override = Some(false),
            "--chat" => chat_enabled_override = Some(true),
            "--dry-run" => dry_run = true,
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
    let mut positionals = Vec::with_capacity(args.len());
    for arg in args {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
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
    let mut positionals = Vec::with_capacity(args.len());
    for arg in args {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
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
    let mut positionals = Vec::with_capacity(args.len());
    for arg in args {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
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
        shell_command(&["nanocamelid", "inspect", &model_path.display().to_string()])
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
            shell_command(&["nanocamelid", "inspect", &model_path.display().to_string()])
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
            shell_command(&["nanocamelid", "inspect", &model_arg])
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
            shell_command(&["nanocamelid", "inspect", &model_path.display().to_string()])
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
        path::{Path, PathBuf},
    };

    use nanocamelid::gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};
    use nanocamelid::inference;
    use nanocamelid::q8;
    use nanocamelid::tokenizer::{SpecialTokens, TokenizerModel};

    use super::{
        ChatTurn, DEFAULT_1B_PREFILL_PROMPT, DEFAULT_1B_PREFILL_TEMP, DEFAULT_1B_PREFILL_TOKENS,
        DEFAULT_1B_SMOKE_PROMPT, DEFAULT_1B_SMOKE_TOKENS, DEFAULT_MODEL_GGUF_ENV,
        DEFAULT_Q4_PREFILL_BATCH, DEFAULT_Q4_PREFILL_PROMPT_LEN, DEFAULT_Q4_PREFILL_RUNS,
        GenerationStatusJson, HelpTopic, InspectTarget, LLAMA32_1B_Q4_MODEL, LLAMA32_1B_Q8_MODEL,
        LLAMA32_3B_Q4_MODEL, PERFORMANCE_GOVERNOR_COMMAND, PrefillBenchBatchMetrics,
        ReadyDirectChatStatus, SMOKE_MODEL_GGUF_ENV, Smoke1BArgs, SmokeDefaults, SmokeKind,
        TRACE_ENV, TuiCommand, cpu_features, cpu_governor_recommendation, cpu_model,
        default_llama32_1b_model_path, default_llama32_3b_model_path, device_model,
        evidence_1b_status_json, evidence_context_pack_command, evidence_model_command,
        evidence_prefill_bench_command, evidence_prefill_bench_command_with_env,
        evidence_ready_no_chat_command, generation_status_json, help_topic_for_args,
        help_topic_named, inspect_1b_status_json, inspect_runtime_summary,
        is_generation_stop_token, is_help_flag, json_string, llama32_1b_model_not_found_message,
        llama32_1b_quantization_for_path, llama32_1b_shape_audit,
        llama32_3b_model_not_found_message, looks_like_gguf_path, looks_like_non_gguf_model_path,
        model_1b_status_json, parse_bench_1b_args_with_env, parse_bench_1b_args_with_path,
        parse_bench_q4_layout_args, parse_bench_q4_prefill_args, parse_bench_q8_dot_args,
        parse_context_packs, parse_cpu_list, parse_evidence_1b_args_with_env,
        parse_evidence_1b_args_with_path, parse_generate_args_with_env,
        parse_generate_args_with_env_and_alias_env_and_workspace,
        parse_generate_args_with_env_and_workspace, parse_inspect_args_with_env,
        parse_model_1b_args_with_path, parse_prefill_batches, parse_prefill_bench_1b_batch_metrics,
        parse_ready_1b_args_with_env, parse_ready_1b_args_with_env_and_smoke_defaults,
        parse_ready_1b_args_with_env_and_smoke_defaults_and_chat_default,
        parse_smoke_1b_args_with_env, parse_smoke_1b_args_with_env_and_defaults,
        parse_smoke_3b_args_with_env, parse_smoke_3b_args_with_env_and_defaults,
        parse_smoke_args_with_env, parse_tui_args_with_env,
        parse_tui_args_with_env_and_alias_env_and_workspace, parse_tui_args_with_env_and_workspace,
        parse_tui_command, prefill_batch_size_from_env_value, prefill_bench_1b_batch_command,
        prefill_bench_1b_batch_env, prefill_bench_1b_result_json, prefill_bench_1b_smoke_command,
        prefill_bench_1b_smoke_env, prefill_bench_1b_status_json, prefill_prompt_tokens_per_sec,
        print_runtime_trace_summary, ready_1b_status_json, ready_chat_enabled_default_for_args,
        ready_chat_enabled_from_env_value, ready_chat_prompt_from_env_value,
        ready_chat_temp_from_env_value, ready_chat_tokens_from_env_value,
        resolve_llama32_1b_model_path_with_workspace, resolve_llama32_3b_model_path_with_workspace,
        runtime_options_from_gguf, shared_token_prefix_len, shell_command, shell_command_with_env,
        smoke_1b_status_json, smoke_defaults_from_values, smoke_plan_command_with_context,
        smoke_plan_command_with_env, trim_tui_history, tui_prompt_history,
        validate_generation_budget,
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
    fn inspect_args_reject_extra_positionals() {
        let err = parse_inspect_args_with_env(
            &["1b".to_owned(), "extra".to_owned()],
            None,
            None,
            "/mnt/nanocamelid",
            false,
        )
        .expect_err("extra inspect arg should fail");

        assert_eq!(err, "unexpected extra inspect argument");
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
