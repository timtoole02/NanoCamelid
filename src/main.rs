use std::{
    env, fs,
    hint::black_box,
    io::{self, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    time::Duration,
};

use nanocamelid::{gguf, inference, model, q8, tokenizer};

const DEFAULT_MODEL_GGUF_ENV: &str = "NANOCAMELID_MODEL_GGUF";
const SMOKE_MODEL_GGUF_ENV: &str = "NANOCAMELID_SMOKE_GGUF";
const RAYON_THREADS_ENV: &str = "NANOCAMELID_RAYON_THREADS";
const PREFILL_BATCH_ENV: &str = "NANOCAMELID_PREFILL_BATCH";
const CONTEXT_LIMIT_ENV: &str = "NANOCAMELID_CONTEXT_LIMIT";
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
            match resolve_model_path_arg(args.get(1), default_model_path_from_env()) {
                Some(path) => inspect_gguf(Path::new(&path)),
                None => {
                    eprintln!("missing GGUF path; pass one or set {DEFAULT_MODEL_GGUF_ENV}");
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
                Ok(parsed) => run_generation(
                    Path::new(&parsed.model_path),
                    &parsed.prompt,
                    parsed.temp,
                    parsed.max_tokens,
                ),
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
                Ok(parsed) => run_chat(
                    Path::new(&parsed.model_path),
                    &parsed.prompt,
                    parsed.temp,
                    parsed.max_tokens,
                ),
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
                Ok(parsed) => run_chat_tui(
                    Path::new(&parsed.model_path),
                    parsed.temp,
                    parsed.max_tokens,
                ),
                Err(err) => {
                    eprintln!("{err}");
                    print_help(HelpTopic::Tui);
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
                Some("q8-dot") => {
                    let iterations = args
                        .get(2)
                        .and_then(|value| value.parse::<usize>().ok())
                        .unwrap_or(q8::DEFAULT_DOT_BENCH_ITERATIONS);
                    let runs = args
                        .get(3)
                        .and_then(|value| value.parse::<usize>().ok())
                        .unwrap_or(q8::DEFAULT_DOT_BENCH_RUNS);
                    bench_q8_dot(iterations, runs)
                }
                Some("q4-layout") => {
                    let rows = args
                        .get(2)
                        .and_then(|value| value.parse::<usize>().ok())
                        .unwrap_or(q8::DEFAULT_Q4_LAYOUT_BENCH_ROWS);
                    let cols = args
                        .get(3)
                        .and_then(|value| value.parse::<usize>().ok())
                        .unwrap_or(q8::DEFAULT_Q4_LAYOUT_BENCH_COLS);
                    let runs = args
                        .get(4)
                        .and_then(|value| value.parse::<usize>().ok())
                        .unwrap_or(q8::DEFAULT_DOT_BENCH_RUNS);
                    bench_q4_layout(rows, cols, runs)
                }
                Some("q4-prefill") => {
                    let prompt_len = args
                        .get(2)
                        .and_then(|value| value.parse::<usize>().ok())
                        .unwrap_or(DEFAULT_Q4_PREFILL_PROMPT_LEN);
                    let batch_size = args
                        .get(3)
                        .and_then(|value| value.parse::<usize>().ok())
                        .unwrap_or(DEFAULT_Q4_PREFILL_BATCH);
                    bench_q4_prefill(prompt_len, batch_size, DEFAULT_Q4_PREFILL_RUNS)
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

fn setup_thread_pool() {
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let default_threads = core_ids.len().clamp(1, DEFAULT_RAYON_THREADS);
    let thread_count = env::var(RAYON_THREADS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default_threads);

    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .start_handler(move |thread_idx| {
            if let Some(core_id) = core_ids.get(thread_idx % core_ids.len().max(1)) {
                core_affinity::set_for_current(*core_id);
            }
        })
        .build_global();
}

fn prefill_batch_size() -> usize {
    env::var(PREFILL_BATCH_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_Q4_PREFILL_BATCH)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HelpTopic {
    TopLevel,
    Probe,
    Inspect,
    Generate,
    Chat,
    Tui,
    Bench,
    Smoke,
}

fn help_topic_for_args(args: &[String]) -> Option<HelpTopic> {
    match args.first().map(String::as_str) {
        Some("-h" | "--help") | Some("help") if args.len() == 1 => Some(HelpTopic::TopLevel),
        Some("help") => help_topic_named(args.get(1).map(String::as_str).unwrap_or_default()),
        _ => None,
    }
}

fn help_topic_named(name: &str) -> Option<HelpTopic> {
    match name {
        "probe" => Some(HelpTopic::Probe),
        "inspect" => Some(HelpTopic::Inspect),
        "generate" => Some(HelpTopic::Generate),
        "chat" => Some(HelpTopic::Chat),
        "tui" => Some(HelpTopic::Tui),
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
        HelpTopic::Probe => print_probe_usage(),
        HelpTopic::Inspect => print_inspect_usage(),
        HelpTopic::Generate => print_generate_usage(),
        HelpTopic::Chat => print_chat_usage(),
        HelpTopic::Tui => print_tui_usage(),
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
    println!(
        "  inspect <model.gguf>                      Inspect GGUF metadata and tensor layouts"
    );
    println!("  generate <model.gguf> <prompt> [temp] [max_tokens]");
    println!(
        "                                            Generate text from prompt on Raspberry Pi 5"
    );
    println!("  chat <model.gguf> <prompt> [temp] [max_tokens]");
    println!(
        "                                            Render a single-turn chat prompt before generation"
    );
    println!("  tui <model.gguf> [temp] [max_tokens]");
    println!("                                            Open an interactive terminal chat");
    println!("  bench q8-dot [iterations] [runs]          Benchmark Q8 dot product kernels");
    println!("  smoke q8-model <model.gguf> [prompt] [max_tokens]");
    println!(
        "                                            Compare scalar vs selected Q8 model logits and greedy generation"
    );
    println!("  smoke q8-chat <model.gguf> [prompt] [max_tokens]");
    println!(
        "                                            Compare scalar vs selected Q8 model logits through the tokenizer chat template"
    );
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
    println!("Print host CPU model, feature flags, and runtime SIMD detection.");
}

fn print_inspect_usage() {
    println!("NanoCamelid inspect");
    println!();
    println!("Usage:");
    println!("  nanocamelid inspect <model.gguf>");
    println!("  nanocamelid inspect                        with NANOCAMELID_MODEL_GGUF set");
    println!();
    println!(
        "Inspect GGUF metadata, runtime-ready LLaMA config, tokenizer support, and the first tensor layouts."
    );
    println!();
    println!("Env:");
    println!(
        "  {DEFAULT_MODEL_GGUF_ENV}                    Default GGUF path for inspect and generate"
    );
}

fn print_generate_usage() {
    println!("NanoCamelid generate");
    println!();
    println!("Usage:");
    println!("  nanocamelid generate <model.gguf> <prompt> [temp] [max_tokens]");
    println!(
        "  nanocamelid generate <prompt> [temp] [max_tokens]   with NANOCAMELID_MODEL_GGUF set"
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
    println!("Env:");
    println!(
        "  {DEFAULT_MODEL_GGUF_ENV}                    Default GGUF path for inspect and generate"
    );
    println!(
        "  {PREFILL_BATCH_ENV}                         Prefill prompt token batch size; default {DEFAULT_Q4_PREFILL_BATCH}, set 1 for single-token prefill"
    );
    println!(
        "  {CONTEXT_LIMIT_ENV}                         Optional runtime context cap for short long-context smoke runs"
    );
    println!();
    println!(
        "When {DEFAULT_MODEL_GGUF_ENV} is set, the first positional argument is treated as the prompt unless it looks like a .gguf path."
    );
}

fn print_chat_usage() {
    println!("NanoCamelid chat");
    println!();
    println!("Usage:");
    println!("  nanocamelid chat <model.gguf> <prompt> [temp] [max_tokens]");
    println!("  nanocamelid chat <prompt> [temp] [max_tokens]   with NANOCAMELID_MODEL_GGUF set");
    println!();
    println!("Args:");
    println!("  <model.gguf>                              Path to the GGUF model file");
    println!(
        "  <prompt>                                  User message content for a single-turn chat request"
    );
    println!("  [temp]                                    Sampling temperature, default 0.0");
    println!("  [max_tokens]                              Maximum tokens to generate, default 128");
    println!();
    println!("Env:");
    println!(
        "  {DEFAULT_MODEL_GGUF_ENV}                    Default GGUF path for inspect, generate, and chat"
    );
    println!(
        "  {PREFILL_BATCH_ENV}                         Prefill prompt token batch size; default {DEFAULT_Q4_PREFILL_BATCH}, set 1 for single-token prefill"
    );
    println!(
        "  {CONTEXT_LIMIT_ENV}                         Optional runtime context cap for short long-context smoke runs"
    );
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
    println!("  nanocamelid tui [temp] [max_tokens]   with NANOCAMELID_MODEL_GGUF set");
    println!();
    println!("Args:");
    println!("  <model.gguf>                              Path to the GGUF model file");
    println!("  [temp]                                    Sampling temperature, default 0.0");
    println!(
        "  [max_tokens]                              Maximum tokens per assistant turn, default 128"
    );
    println!();
    println!("Env:");
    println!(
        "  {DEFAULT_MODEL_GGUF_ENV}                    Default GGUF path for inspect, generate, chat, and tui"
    );
    println!(
        "  {RAYON_THREADS_ENV}                         Rayon worker count; defaults to up to 4 pinned workers"
    );
    println!(
        "  {PREFILL_BATCH_ENV}                         Prefill prompt token batch size; default {DEFAULT_Q4_PREFILL_BATCH}, set 1 for single-token prefill"
    );
    println!(
        "  {CONTEXT_LIMIT_ENV}                         Optional runtime context cap for short long-context smoke runs"
    );
    println!();
    println!("Commands inside the TUI: /help, /model <path>, /clear, /exit, /quit");
}

fn print_bench_usage() {
    println!("NanoCamelid bench");
    println!();
    println!("Usage:");
    println!("  nanocamelid bench q8-dot [iterations] [runs]");
    println!("  nanocamelid bench q4-layout [rows] [cols] [runs]");
    println!("  nanocamelid bench q4-prefill [prompt_len] [batch_size]");
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
    println!();
    println!("Env:");
    println!("  NANOCAMELID_Q8_DOT_KERNEL                 Force scalar, neon, or sdot selection");
    println!(
        "  NANOCAMELID_Q8_DOT_SDOT                   Enable SDOT candidate benchmarking when supported"
    );
    println!("  NANOCAMELID_Q6K_SDOT                      Enable experimental Q6_K SDOT matmuls");
    println!(
        "  {RAYON_THREADS_ENV}                         Rayon worker count; defaults to up to 4 pinned workers"
    );
}

fn print_smoke_usage() {
    println!("NanoCamelid smoke");
    println!();
    println!("Usage:");
    println!("  nanocamelid smoke q8-model <model.gguf> [prompt] [max_tokens]");
    println!("  nanocamelid smoke q8-chat <model.gguf> [prompt] [max_tokens]");
    println!("  nanocamelid smoke q8-model [prompt] [max_tokens]  with NANOCAMELID_SMOKE_GGUF set");
    println!("  nanocamelid smoke q8-chat [prompt] [max_tokens]   with NANOCAMELID_SMOKE_GGUF set");
    println!();
    println!("Args:");
    println!("  <model.gguf>                              Path to the GGUF model file");
    println!("  [prompt]                                  Prompt text, default \"Hello\"");
    println!(
        "  [max_tokens]                              Greedy tokens to generate after parity, default 1"
    );
    println!();
    println!("Env:");
    println!("  {SMOKE_MODEL_GGUF_ENV}                    Default GGUF path for smoke validation");
    println!(
        "  {DEFAULT_MODEL_GGUF_ENV}                    Shared default GGUF path for inspect/generate/smoke"
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
    println!(
        "When {SMOKE_MODEL_GGUF_ENV} or {DEFAULT_MODEL_GGUF_ENV} is set, the first positional argument is treated as the prompt unless it looks like a .gguf path."
    );
    println!();
    println!(
        "`q8-model` tokenizes the prompt directly. `q8-chat` renders a single-turn user message through the model tokenizer chat template before parity/generation."
    );
}

#[derive(Debug, PartialEq)]
struct GenerateArgs {
    model_path: String,
    prompt: String,
    temp: f32,
    max_tokens: usize,
}

#[derive(Debug, PartialEq)]
struct TuiArgs {
    model_path: String,
    temp: f32,
    max_tokens: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct SmokeQ8ModelArgs {
    model_path: String,
    prompt: String,
    max_tokens: usize,
}

fn parse_generate_args(args: &[String]) -> Result<GenerateArgs, &'static str> {
    parse_generate_args_with_env(args, default_model_path_from_env())
}

fn parse_tui_args(args: &[String]) -> Result<TuiArgs, &'static str> {
    parse_tui_args_with_env(args, default_model_path_from_env())
}

fn parse_tui_args_with_env(
    args: &[String],
    env_model_path: Option<String>,
) -> Result<TuiArgs, &'static str> {
    let first_looks_like_model = args
        .first()
        .is_some_and(|value| looks_like_gguf_path(value));

    let (model_path, option_idx) = match (args.first(), env_model_path) {
        (Some(path), _) if first_looks_like_model => (path.clone(), 1),
        (Some(path), None) => (path.clone(), 1),
        (_, Some(path)) => (path, 0),
        (None, None) => {
            return Err("missing GGUF model path; pass one or set NANOCAMELID_MODEL_GGUF");
        }
    };

    let temp = args
        .get(option_idx)
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.0);
    let max_tokens = args
        .get(option_idx + 1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(128);

    Ok(TuiArgs {
        model_path,
        temp,
        max_tokens,
    })
}

fn parse_generate_args_with_env(
    args: &[String],
    env_model_path: Option<String>,
) -> Result<GenerateArgs, &'static str> {
    let first_looks_like_model = args
        .first()
        .is_some_and(|value| looks_like_gguf_path(value));

    let (model_path, prompt_idx) = match (args.first(), env_model_path) {
        (Some(path), _) if first_looks_like_model => (path.clone(), 1),
        (Some(path), None) => (path.clone(), 1),
        (_, Some(path)) => (path, 0),
        (None, None) => {
            return Err("missing GGUF model path; pass one or set NANOCAMELID_MODEL_GGUF");
        }
    };

    let prompt = match args.get(prompt_idx) {
        Some(prompt) => prompt.clone(),
        None => {
            return Err(
                "missing prompt; pass one after the GGUF path or set NANOCAMELID_MODEL_GGUF and pass the prompt first",
            );
        }
    };
    let temp = args
        .get(prompt_idx + 1)
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.0);
    let max_tokens = args
        .get(prompt_idx + 2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(128);

    Ok(GenerateArgs {
        model_path,
        prompt,
        temp,
        max_tokens,
    })
}

fn parse_smoke_args(args: &[String]) -> Result<SmokeQ8ModelArgs, &'static str> {
    parse_smoke_args_with_env(args, smoke_model_path_from_env())
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
    let max_tokens = args
        .get(prompt_idx + 1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);

    Ok(SmokeQ8ModelArgs {
        model_path,
        prompt,
        max_tokens,
    })
}

fn looks_like_gguf_path(value: &str) -> bool {
    value.ends_with(".gguf") || value.ends_with(".gguf/")
}

fn resolve_model_path_arg(path: Option<&String>, env_model_path: Option<String>) -> Option<String> {
    path.cloned().or(env_model_path)
}

fn default_model_path_from_env() -> Option<String> {
    env::var(DEFAULT_MODEL_GGUF_ENV).ok()
}

fn smoke_model_path_from_env() -> Option<String> {
    env::var(SMOKE_MODEL_GGUF_ENV)
        .ok()
        .or_else(default_model_path_from_env)
}

fn print_probe() {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let device_tree_model = fs::read_to_string("/proc/device-tree/model").unwrap_or_default();
    let model = device_model(&device_tree_model)
        .or_else(|| cpu_model(&cpuinfo))
        .unwrap_or("unknown");
    let features = cpu_features(&cpuinfo);

    println!("NanoCamelid host probe");
    println!("arch: {}", env::consts::ARCH);
    println!("os: {}", env::consts::OS);
    println!("cpu_model: {model}");
    println!("cpu_features: {}", features.unwrap_or("unknown"));
    println!("runtime_neon: {}", runtime_neon());
    println!("runtime_dotprod: {}", runtime_dotprod());
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
        "json: {{\"benchmark\":\"q4-layout\",\"rows\":{},\"cols\":{},\"runs\":{},\"blocks_per_row\":{},\"dotprod_feature_detected\":{},\"row_major\":{{\"checksum\":{},\"run_ms\":{}}},\"swizzled_1x4\":{{\"checksum\":{},\"run_ms\":{}}},\"swizzled_speedup\":{:.6}}}",
        report.rows,
        report.cols,
        report.runs,
        report.blocks_per_row,
        report.dotprod_feature_detected,
        report.row_major.checksum,
        duration_ms_json(&report.row_major.elapsed_runs),
        report.swizzled_1x4.checksum,
        duration_ms_json(&report.swizzled_1x4.elapsed_runs),
        report.swizzled_speedup()
    );

    if report.row_major.checksum != report.swizzled_1x4.checksum {
        eprintln!("swizzled layout checksum mismatch");
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

    let rows = Q4_PREFILL_ROWS;
    let cols = Q4_PREFILL_COLS;
    let blocks_per_row = cols / q8::Q8_BLOCK_SIZE;
    let q8_blocks_per_token = cols / q8::Q8_BLOCK_SIZE;
    let selector = q8::Q8DotKernelSelector::from_env();
    let row_major = synthetic_q4_blocks(rows, blocks_per_row);
    let swizzled_1x4 = swizzle_q4_0_1x4(&row_major, rows, blocks_per_row);
    let matrix = model::QuantizedMatrix::Q4_0Swizzled1x4(model::Q4_0Swizzled1x4Matrix {
        swizzled_1x4,
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
    let min_ms = elapsed_runs
        .iter()
        .min()
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .unwrap_or_default();

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
    println!("min_ms: {min_ms:.3}");
    println!(
        "json: {{\"benchmark\":\"q4-prefill\",\"prompt_len\":{},\"batch_size\":{},\"runs\":{},\"rows\":{},\"cols\":{},\"kernel\":\"{}\",\"run_ms\":{},\"median_ms\":{:.6},\"median_ms_per_token\":{:.6},\"min_ms\":{:.6}}}",
        prompt_len,
        batch_size,
        runs,
        rows,
        cols,
        selector.selected.name(),
        duration_ms_json(&elapsed_runs),
        median_ms,
        median_ms_per_token,
        min_ms
    );

    ExitCode::SUCCESS
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

fn inspect_gguf(path: &Path) -> ExitCode {
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
    let ready = model_config.is_ok() && tensor_layouts.is_ok() && tokenizer.is_ok();

    InspectRuntimeSummary {
        ready,
        tied_output,
        model_config,
        tensor_layouts,
        tokenizer,
    }
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
    let mut selected_batch_ws = inference::LlamaBatchWorkspace::new(&config, prefill_batch_size());
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
        if Some(scalar_next as u32) == tokenizer.special.eos
            || Some(scalar_next as u32) == tokenizer.special.eot
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

fn rope_scaling_from_gguf(gguf: &gguf::GgufFile) -> inference::RopeScaling {
    let prefix = gguf
        .metadata_string("general.architecture")
        .filter(|arch| matches!(*arch, "llama" | "qwen2"))
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
    let Some(limit) = env::var(CONTEXT_LIMIT_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(());
    };
    let limit = limit
        .parse::<usize>()
        .map_err(|err| format!("{CONTEXT_LIMIT_ENV} must be a positive integer: {err}"))?;
    if limit == 0 {
        return Err(format!("{CONTEXT_LIMIT_ENV} must be greater than zero"));
    }
    config.context_length = config.context_length.min(limit);
    Ok(())
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

fn run_generation(model_path: &Path, prompt: &str, temp: f32, max_tokens: usize) -> ExitCode {
    run_generation_with_prompt_builder(model_path, temp, max_tokens, |_tokenizer| {
        Ok(GenerationPromptPlan {
            text: prompt.to_owned(),
            add_special: true,
            parse_special: true,
            renderer: None,
            template_format: None,
        })
    })
}

fn run_chat(model_path: &Path, prompt: &str, temp: f32, max_tokens: usize) -> ExitCode {
    run_generation_with_prompt_builder(model_path, temp, max_tokens, |tokenizer| {
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
    })
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
    Ok(TuiLoadedModel {
        model_path: model_path.to_path_buf(),
        config,
        weights,
        tokenizer,
        runtime_options,
        model_name,
        renderer,
        load_secs: started_load.elapsed().as_secs_f64(),
    })
}

fn run_chat_tui(model_path: &Path, temp: f32, max_tokens: usize) -> ExitCode {
    let selector = q8::Q8DotKernelSelector::from_env();
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

    print_tui_banner(TuiBanner {
        model_name: &loaded.model_name,
        model_path: &loaded.model_path,
        config: &loaded.config,
        kernel: selector.selected.name(),
        renderer: &loaded.renderer,
        temp,
        max_tokens,
        load_secs: loaded.load_secs,
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
        if input == "/model" {
            println!(
                "{}current model{} {}",
                ansi::LABEL,
                ansi::RESET,
                loaded.model_path.display()
            );
            continue;
        }
        if let Some(next_model) = input.strip_prefix("/model ") {
            let next_model = next_model.trim();
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
                        temp,
                        max_tokens,
                        load_secs: loaded.load_secs,
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
            continue;
        }

        match input {
            "/exit" | "/quit" => break,
            "/help" => {
                print_tui_commands();
                continue;
            }
            "/clear" => {
                history.clear();
                session.reset(&loaded.config);
                total_in = 0;
                total_out = 0;
                println!("{}conversation cleared{}", ansi::DIM, ansi::RESET);
                print_tui_status(TuiStatus {
                    model_name: &loaded.model_name,
                    kernel: selector.selected.name(),
                    input_tokens: 0,
                    output_tokens: 0,
                    total_in,
                    total_out,
                    ttft_secs: None,
                    elapsed_secs: 0.0,
                });
                continue;
            }
            _ => {}
        }

        history.push(ChatTurn {
            role: "user".to_owned(),
            content: input.to_owned(),
        });

        print!("{}assistant>{} ", ansi::ASSISTANT, ansi::RESET);
        if io::stdout().flush().is_err() {
            return ExitCode::FAILURE;
        }

        let report = match generate_chat_turn(
            &history,
            &mut session,
            ChatGenerationEnv {
                config: &loaded.config,
                weights: &loaded.weights,
                tokenizer: &loaded.tokenizer,
                runtime_options: loaded.runtime_options,
            },
            temp,
            max_tokens,
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

        total_in += report.input_tokens;
        total_out += report.output_tokens;
        print_tui_status(TuiStatus {
            model_name: &loaded.model_name,
            kernel: selector.selected.name(),
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
    env: ChatGenerationEnv<'_>,
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
    loop {
        let next_token = inference::sample_logits(&session.ws.logits, temp);
        if Some(next_token as u32) == env.tokenizer.special.eos
            || Some(next_token as u32) == env.tokenizer.special.eot
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
    model_path: &Path,
    temp: f32,
    max_tokens: usize,
    prompt_builder: F,
) -> ExitCode
where
    F: FnOnce(&tokenizer::Tokenizer) -> Result<GenerationPromptPlan, String>,
{
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

    println!("Selected dot-product kernel: {}", selector.selected.name());
    println!("\nGenerating response:\n");

    let mut pos = 0;
    let started_prefill = std::time::Instant::now();

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
    println!(
        "Prompt ingested in {:.2}s with prefill batch {}",
        started_prefill.elapsed().as_secs_f64(),
        batch_ws.max_batch
    );

    // Now generate the next tokens
    let mut generated_tokens = Vec::new();
    let mut last_printed_len = 0;
    let start_gen = std::time::Instant::now();

    loop {
        let next_token = inference::sample_logits(&ws.logits, temp);

        if Some(next_token as u32) == tokenizer.special.eos
            || Some(next_token as u32) == tokenizer.special.eot
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

    ExitCode::SUCCESS
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

struct TuiBanner<'a> {
    model_name: &'a str,
    model_path: &'a Path,
    config: &'a model::LlamaModelConfig,
    kernel: &'a str,
    renderer: &'a str,
    temp: f32,
    max_tokens: usize,
    load_secs: f64,
}

struct TuiStatus<'a> {
    model_name: &'a str,
    kernel: &'a str,
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
        "{}llama NanoCamelid 0.1.0{} - Pi-local terminal chat",
        ansi::TITLE,
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
        banner.temp,
        banner.max_tokens,
        banner.load_secs
    );
    println!(
        "{}commands{} /help /model <path> /clear /exit",
        ansi::LABEL,
        ansi::RESET
    );
    println!();
}

fn print_tui_commands() {
    println!("{}commands{}", ansi::LABEL, ansi::RESET);
    println!("  /model <path>  load another GGUF and reset the conversation");
    println!("  /model         show the current GGUF path");
    println!("  /clear  reset the conversation and token counters");
    println!("  /exit   leave the chat");
    println!("  /quit   leave the chat");
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
        "{}connected | model {} | kernel {} | last in {} | last out {} | total in/out {}/{} | ttft {} | {:.2} tok/sec{}",
        ansi::DIM,
        status.model_name,
        status.kernel,
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
    use std::{collections::BTreeMap, path::PathBuf};

    use nanocamelid::gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};
    use nanocamelid::q8;
    use nanocamelid::tokenizer::TokenizerModel;

    use super::{
        HelpTopic, cpu_features, cpu_model, device_model, help_topic_for_args, help_topic_named,
        inspect_runtime_summary, is_help_flag, parse_generate_args_with_env,
        parse_smoke_args_with_env, parse_tui_args_with_env, resolve_model_path_arg,
        runtime_options_from_gguf, shared_token_prefix_len, validate_generation_budget,
    };
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
    fn help_topic_named_maps_supported_commands() {
        assert_eq!(help_topic_named("probe"), Some(HelpTopic::Probe));
        assert_eq!(help_topic_named("inspect"), Some(HelpTopic::Inspect));
        assert_eq!(help_topic_named("generate"), Some(HelpTopic::Generate));
        assert_eq!(help_topic_named("tui"), Some(HelpTopic::Tui));
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
            help_topic_for_args(&["help".to_owned(), "smoke".to_owned()]),
            Some(HelpTopic::Smoke)
        );
        assert_eq!(
            help_topic_for_args(&["help".to_owned(), "tui".to_owned()]),
            Some(HelpTopic::Tui)
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
    fn resolve_model_path_arg_prefers_explicit_path() {
        assert_eq!(
            resolve_model_path_arg(
                Some(&"/models/explicit.gguf".to_owned()),
                Some("/models/env.gguf".to_owned())
            ),
            Some("/models/explicit.gguf".to_owned())
        );
    }

    #[test]
    fn resolve_model_path_arg_falls_back_to_env_path() {
        assert_eq!(
            resolve_model_path_arg(None, Some("/models/env.gguf".to_owned())),
            Some("/models/env.gguf".to_owned())
        );
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
        assert_eq!(parsed.prompt, "Explain grouped-query attention");
        assert_eq!(parsed.temp, 16.0);
        assert_eq!(parsed.max_tokens, 128);
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
        assert_eq!(parsed.temp, 0.2);
        assert_eq!(parsed.max_tokens, 64);
    }

    #[test]
    fn tui_args_fall_back_to_env_model_path() {
        let parsed = parse_tui_args_with_env(
            &["0.1".to_owned(), "32".to_owned()],
            Some("/models/Llama-3.2-1B-Instruct.Q8_0.gguf".to_owned()),
        )
        .expect("env-backed tui path should parse");

        assert_eq!(parsed.model_path, "/models/Llama-3.2-1B-Instruct.Q8_0.gguf");
        assert_eq!(parsed.temp, 0.1);
        assert_eq!(parsed.max_tokens, 32);
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
    fn inspect_runtime_summary_reports_ready_q8_llama_fixture() {
        let summary = inspect_runtime_summary(&inspect_fixture(false));
        assert!(summary.ready);
        assert!(summary.tensor_layouts.is_ok());
        assert!(summary.tied_output);
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
