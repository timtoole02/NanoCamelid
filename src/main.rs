use std::{
    env, fs,
    io::{self, Write},
    path::Path,
    process::ExitCode,
};

use nanocamelid::{gguf, inference, model, q8, tokenizer};

const DEFAULT_MODEL_GGUF_ENV: &str = "NANOCAMELID_MODEL_GGUF";
const SMOKE_MODEL_GGUF_ENV: &str = "NANOCAMELID_SMOKE_GGUF";

fn main() -> ExitCode {
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
    println!();
    println!("Commands inside the TUI: /help, /clear, /exit, /quit");
}

fn print_bench_usage() {
    println!("NanoCamelid bench");
    println!();
    println!("Usage:");
    println!("  nanocamelid bench q8-dot [iterations] [runs]");
    println!();
    println!("Args:");
    println!(
        "  [iterations]                              Blocks per run, default {}",
        q8::DEFAULT_DOT_BENCH_ITERATIONS
    );
    println!(
        "  [runs]                                    Repeated timing samples, default {}",
        q8::DEFAULT_DOT_BENCH_RUNS
    );
    println!();
    println!("Env:");
    println!("  NANOCAMELID_Q8_DOT_KERNEL                 Force scalar, neon, or sdot selection");
    println!(
        "  NANOCAMELID_Q8_DOT_SDOT                   Enable SDOT candidate benchmarking when supported"
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
                    println!("  architecture: llama");
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

            if let Some(factor) = file.metadata_f32("llama.rope.scaling.factor") {
                println!("  rope_scaling_factor: {factor}");
            }
            if let Some(original_context_length) =
                file.metadata_u32("llama.rope.scaling.original_context_length")
            {
                println!("  rope_scaling_original_context_length: {original_context_length}");
            }
            if let Some(low_freq_factor) = file.metadata_f32("llama.rope.scaling.low_freq_factor") {
                println!("  rope_scaling_low_freq_factor: {low_freq_factor}");
            }
            if let Some(high_freq_factor) = file.metadata_f32("llama.rope.scaling.high_freq_factor")
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
    let config = model::LlamaModelConfig::from_gguf(&gguf)
        .map_err(|err| format!("failed to parse config: {err}"))?;
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
    };
    let selected_options = inference::LlamaRuntimeOptions {
        q8_selector: selected,
        rope_scaling,
    };

    let mut scalar_cache =
        inference::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut selected_cache =
        inference::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut scalar_ws = inference::LlamaWorkspace::new(&config);
    let mut selected_ws = inference::LlamaWorkspace::new(&config);
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
        inference::forward_pass(
            token as usize,
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
    inference::RopeScaling {
        factor: gguf.metadata_f32("llama.rope.scaling.factor"),
        original_context_length: gguf
            .metadata_u32("llama.rope.scaling.original_context_length")
            .map(|value| value as f32),
        low_freq_factor: gguf.metadata_f32("llama.rope.scaling.low_freq_factor"),
        high_freq_factor: gguf.metadata_f32("llama.rope.scaling.high_freq_factor"),
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
    elapsed_secs: f64,
}

fn run_chat_tui(model_path: &Path, temp: f32, max_tokens: usize) -> ExitCode {
    println!("Loading GGUF file: {}...", model_path.display());
    let gguf = match gguf::read_file(model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to read GGUF: {e}");
            return ExitCode::FAILURE;
        }
    };

    let config = match model::LlamaModelConfig::from_gguf(&gguf) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to parse config: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tokenizer = match tokenizer::Tokenizer::from_gguf(&gguf) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to load tokenizer: {e}");
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

    let selector = q8::Q8DotKernelSelector::from_env();
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
    let mut history = Vec::<ChatTurn>::new();
    let mut total_in = 0usize;
    let mut total_out = 0usize;

    print_tui_banner(TuiBanner {
        model_name: &model_name,
        model_path,
        config: &config,
        kernel: selector.selected.name(),
        renderer: &renderer,
        temp,
        max_tokens,
        load_secs: started_load.elapsed().as_secs_f64(),
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
        match input {
            "/exit" | "/quit" => break,
            "/help" => {
                print_tui_commands();
                continue;
            }
            "/clear" => {
                history.clear();
                total_in = 0;
                total_out = 0;
                println!("{}conversation cleared{}", ansi::DIM, ansi::RESET);
                print_tui_status(
                    &model_name,
                    selector.selected.name(),
                    0,
                    0,
                    total_in,
                    total_out,
                    0.0,
                );
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
            &config,
            &weights,
            &tokenizer,
            runtime_options,
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
        print_tui_status(
            &model_name,
            selector.selected.name(),
            report.input_tokens,
            report.output_tokens,
            total_in,
            total_out,
            report.elapsed_secs,
        );
    }

    println!("{}disconnected{}", ansi::DIM, ansi::RESET);
    ExitCode::SUCCESS
}

fn generate_chat_turn(
    history: &[ChatTurn],
    config: &model::LlamaModelConfig,
    weights: &model::LlamaWeights,
    tokenizer: &tokenizer::Tokenizer,
    runtime_options: inference::LlamaRuntimeOptions,
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
    let rendered = tokenizer.render_chat_prompt(&messages);
    let prompt_tokens =
        tokenizer.encode(&rendered.text, rendered.add_special, rendered.parse_special)?;
    if prompt_tokens.is_empty() {
        return Err("prompt tokenized to an empty sequence".to_owned());
    }
    validate_generation_budget(prompt_tokens.len(), max_tokens, config.context_length)?;

    let mut cache =
        inference::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut ws = inference::LlamaWorkspace::new(config);
    let mut pos = 0usize;
    for &token in &prompt_tokens {
        inference::forward_pass(
            token as usize,
            pos,
            config,
            weights,
            &mut cache,
            &mut ws,
            runtime_options,
        );
        pos += 1;
    }

    let mut generated_tokens = Vec::new();
    let mut last_printed_len = 0usize;
    let started = std::time::Instant::now();
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
            io::stdout()
                .flush()
                .map_err(|err| format!("failed to flush stdout: {err}"))?;
            last_printed_len = full_text.len();
        }

        inference::forward_pass(
            next_token,
            pos,
            config,
            weights,
            &mut cache,
            &mut ws,
            runtime_options,
        );
        pos += 1;
    }

    let assistant_text = tokenizer.decode(&generated_tokens, true)?;
    println!();
    Ok((
        assistant_text,
        ChatTurnReport {
            input_tokens: prompt_tokens.len(),
            output_tokens: generated_tokens.len(),
            elapsed_secs: started.elapsed().as_secs_f64(),
        },
    ))
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

    let config = match model::LlamaModelConfig::from_gguf(&gguf) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to parse config: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("Architecture: LLaMA");
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
    let selector = q8::Q8DotKernelSelector::from_env();
    let runtime_options = runtime_options_from_gguf(&gguf, selector);

    println!("Selected dot-product kernel: {}", selector.selected.name());
    println!("\nGenerating response:\n");

    let mut pos = 0;

    // Decode prompt tokens (prefill path)
    for &token in &prompt_tokens {
        inference::forward_pass(
            token as usize,
            pos,
            &config,
            &weights,
            &mut cache,
            &mut ws,
            runtime_options,
        );
        pos += 1;
    }

    // Now generate the next tokens
    let mut generated_count = 0;
    let mut generated_tokens = Vec::new();
    let mut last_printed_len = 0;
    let start_gen = std::time::Instant::now();

    loop {
        let next_token = inference::sample_logits(&ws.logits, temp);

        if Some(next_token as u32) == tokenizer.special.eos
            || Some(next_token as u32) == tokenizer.special.eot
            || pos >= config.context_length
            || generated_count >= max_tokens
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
        generated_count += 1;
    }

    let elapsed = start_gen.elapsed().as_secs_f64();
    println!(
        "\n\nGenerated {} tokens in {:.2}s ({:.2} tokens/sec)",
        generated_count,
        elapsed,
        generated_count as f64 / elapsed
    );

    ExitCode::SUCCESS
}

fn runtime_options_from_gguf(
    gguf: &gguf::GgufFile,
    selector: q8::Q8DotKernelSelector,
) -> inference::LlamaRuntimeOptions {
    inference::LlamaRuntimeOptions {
        q8_selector: selector,
        rope_scaling: inference::RopeScaling {
            factor: gguf.metadata_f32("llama.rope.scaling.factor"),
            original_context_length: gguf
                .metadata_u32("llama.rope.scaling.original_context_length")
                .map(|v| v as f32),
            low_freq_factor: gguf.metadata_f32("llama.rope.scaling.low_freq_factor"),
            high_freq_factor: gguf.metadata_f32("llama.rope.scaling.high_freq_factor"),
        },
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
        "{}runtime{} LLaMA | layers {} | ctx {} | kernel {} | renderer {} | temp {:.2} | max out {} | load {:.2}s",
        ansi::LABEL,
        ansi::RESET,
        banner.config.block_count,
        banner.config.context_length,
        banner.kernel,
        banner.renderer,
        banner.temp,
        banner.max_tokens,
        banner.load_secs
    );
    println!("{}commands{} /help /clear /exit", ansi::LABEL, ansi::RESET);
    println!();
}

fn print_tui_commands() {
    println!("{}commands{}", ansi::LABEL, ansi::RESET);
    println!("  /clear  reset the conversation and token counters");
    println!("  /exit   leave the chat");
    println!("  /quit   leave the chat");
}

fn print_tui_status(
    model_name: &str,
    kernel: &str,
    input_tokens: usize,
    output_tokens: usize,
    total_in: usize,
    total_out: usize,
    elapsed_secs: f64,
) {
    let tok_per_sec = if elapsed_secs > 0.0 {
        output_tokens as f64 / elapsed_secs
    } else {
        0.0
    };
    println!(
        "{}connected | model {} | kernel {} | last in {} | last out {} | total in/out {}/{} | {:.2} tok/sec{}",
        ansi::DIM,
        model_name,
        kernel,
        input_tokens,
        output_tokens,
        total_in,
        total_out,
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
    use nanocamelid::tokenizer::TokenizerModel;

    use super::{
        HelpTopic, cpu_features, cpu_model, device_model, help_topic_for_args, help_topic_named,
        inspect_runtime_summary, is_help_flag, parse_generate_args_with_env,
        parse_smoke_args_with_env, parse_tui_args_with_env, resolve_model_path_arg,
        validate_generation_budget,
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
