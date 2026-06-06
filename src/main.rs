use std::{env, fs, path::Path, process::ExitCode};

use nanocamelid::{distributed, gguf, inference, model, q4, q8, runtime, tokenizer};

fn main() -> ExitCode {
    let command = env::args().nth(1);

    match command.as_deref() {
        Some("probe") => {
            print_probe();
            ExitCode::SUCCESS
        }
        Some("inspect") => match env::args().nth(2) {
            Some(path) => inspect_gguf(Path::new(&path)),
            None => {
                eprintln!("missing GGUF path");
                print_usage();
                ExitCode::from(2)
            }
        },
        Some("generate") => {
            let model_path = env::args().nth(2);
            let prompt = env::args().nth(3);
            if model_path.is_none() || prompt.is_none() {
                eprintln!("missing GGUF model path or prompt");
                print_usage();
                ExitCode::from(2)
            } else {
                let temp = env::args().nth(4)
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(0.0);
                let max_tokens = env::args().nth(5)
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(128);
                run_generation(Path::new(&model_path.unwrap()), &prompt.unwrap(), temp, max_tokens)
            }
        }
        Some("serve-stage") => {
            let config_path = env::args().nth(2);
            let node_name = env::args().nth(3);
            match (config_path, node_name) {
                (Some(config_path), Some(node_name)) => {
                    run_serve_stage(&config_path, &node_name, env::args().nth(4))
                }
                _ => {
                    eprintln!("missing cluster config path or node name");
                    print_usage();
                    ExitCode::from(2)
                }
            }
        }
        Some("generate-distributed") => {
            let config_path = env::args().nth(2);
            let prompt = env::args().nth(3);
            if config_path.is_none() || prompt.is_none() {
                eprintln!("missing cluster config path or prompt");
                print_usage();
                ExitCode::from(2)
            } else {
                let temp = env::args().nth(4)
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(0.0);
                let max_tokens = env::args().nth(5)
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(128);
                run_generate_distributed(&config_path.unwrap(), &prompt.unwrap(), temp, max_tokens)
            }
        }
        Some("bench") => match env::args().nth(2).as_deref() {
            Some("q8-dot") => {
                let iterations = env::args()
                    .nth(3)
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(q8::DEFAULT_DOT_BENCH_ITERATIONS);
                let runs = env::args()
                    .nth(4)
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(q8::DEFAULT_DOT_BENCH_RUNS);
                bench_q8_dot(iterations, runs)
            }
            Some("q4-dot") => {
                let iterations = env::args()
                    .nth(3)
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(q8::DEFAULT_DOT_BENCH_ITERATIONS);
                let runs = env::args()
                    .nth(4)
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(q8::DEFAULT_DOT_BENCH_RUNS);
                bench_q4_dot(iterations, runs)
            }
            Some(other) => {
                eprintln!("unknown benchmark: {other}");
                print_usage();
                ExitCode::from(2)
            }
            None => {
                eprintln!("missing benchmark name");
                print_usage();
                ExitCode::from(2)
            }
        },
        Some("-h" | "--help") | None => {
            print_usage();
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown command: {other}");
            print_usage();
            ExitCode::from(2)
        }
    }
}

fn print_usage() {
    println!("NanoCamelid");
    println!();
    println!("Usage:");
    println!("  nanocamelid probe                            Print host CPU and runtime feature information");
    println!("  nanocamelid inspect <model.gguf>             Inspect GGUF metadata and tensor layouts");
    println!("  nanocamelid generate <model.gguf> <prompt> [temp] [max_tokens]");
    println!("                                               Generate text from prompt on Raspberry Pi 5");
    println!("  nanocamelid serve-stage <nodes.toml> <node-name> [model.gguf]");
    println!("                                               Run a pipeline stage server (node1/node2)");
    println!("  nanocamelid generate-distributed <nodes.toml> <prompt> [temp] [max_tokens]");
    println!("                                               Generate using the 3-node pipeline (run on node0)");
    println!("  nanocamelid bench q8-dot [iterations] [runs] Benchmark Q8 dot product kernels");
    println!("  nanocamelid bench q4-dot [iterations] [runs] Benchmark Q4 dot product kernels");
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
    match runtime::read_cpu_temp_celsius() {
        Some(temp) => println!("cpu_temp_c: {temp:.1}"),
        None => println!("cpu_temp_c: unknown"),
    }
    match runtime::read_throttled_flags() {
        Some(flags) => println!("throttled: 0x{flags:x}"),
        None => println!("throttled: unknown"),
    }
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
    match gguf::inspect(path) {
        Ok(summary) => {
            println!("NanoCamelid GGUF inspect");
            println!("path: {}", path.display());
            println!("version: {}", summary.version);
            println!("tensor_count: {}", summary.tensor_count);
            println!("metadata_count: {}", summary.metadata_count);
            println!("alignment: {}", summary.alignment);
            println!("data_start_offset: {}", summary.data_start_offset);

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

fn run_generation(model_path: &Path, prompt: &str, temp: f32, max_tokens: usize) -> ExitCode {
    // Pin the rayon workers to the A76 cores. Scheduling only — results are unchanged.
    if let Err(e) = runtime::configure_compute_pool() {
        eprintln!("warning: {e}");
    }
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

    println!("Loading model weights into memory...");
    let started_load = std::time::Instant::now();
    let weights = match model::LlamaWeights::load(model_path, &config, &gguf) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to load weights: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("Weights loaded in {:.2}s", started_load.elapsed().as_secs_f64());

    let prompt_tokens = match tokenizer.encode(prompt, true, true) {
        Ok(tokens) => tokens,
        Err(e) => {
            eprintln!("Failed to tokenize prompt: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("Prompt tokens: {:?}", prompt_tokens);

    let mut cache = inference::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut ws = inference::LlamaWorkspace::new(&config);
    let selector_q8 = q8::Q8DotKernelSelector::from_env_or_auto();
    let selector_q4 = q4::Q4DotKernelSelector::from_env_or_auto();
    
    match &weights {
        model::LlamaWeights::Q8_0(_) => {
            println!("Selected Q8 dot-product kernel: {}", selector_q8.selected.name());
        }
        model::LlamaWeights::Q4_0(_) => {
            println!("Selected Q4 dot-product kernel: {}", selector_q4.selected.name());
        }
    }
    println!("\nGenerating response:\n");

    let mut input_token;
    let mut pos = 0;
    
    // Decode prompt tokens (prefill path)
    for &token in &prompt_tokens {
        input_token = token as usize;
        inference::forward_pass(
            input_token,
            pos,
            &config,
            &weights,
            &mut cache,
            &mut ws,
            selector_q8,
            selector_q4,
            gguf.metadata_f32("llama.rope.scaling.factor"),
            gguf.metadata_u32("llama.rope.scaling.original_context_length").map(|v| v as f32),
            gguf.metadata_f32("llama.rope.scaling.low_freq_factor"),
            gguf.metadata_f32("llama.rope.scaling.high_freq_factor"),
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
        if let Ok(full_text) = tokenizer.decode(&generated_tokens, true) {
            if full_text.len() > last_printed_len {
                print!("{}", &full_text[last_printed_len..]);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                last_printed_len = full_text.len();
            }
        }

        input_token = next_token;
        inference::forward_pass(
            input_token,
            pos,
            &config,
            &weights,
            &mut cache,
            &mut ws,
            selector_q8,
            selector_q4,
            gguf.metadata_f32("llama.rope.scaling.factor"),
            gguf.metadata_u32("llama.rope.scaling.original_context_length").map(|v| v as f32),
            gguf.metadata_f32("llama.rope.scaling.low_freq_factor"),
            gguf.metadata_f32("llama.rope.scaling.high_freq_factor"),
        );

        pos += 1;
        generated_count += 1;
    }

    let elapsed = start_gen.elapsed().as_secs_f64();
    println!("\n\nGenerated {} tokens in {:.2}s ({:.2} tokens/sec)", 
        generated_count, 
        elapsed, 
        generated_count as f64 / elapsed
    );

    ExitCode::SUCCESS
}

/// Run this Pi as a pipeline stage server (node1 / node2 in nodes.toml).
fn run_serve_stage(config_path: &str, node_name: &str, model_override: Option<String>) -> ExitCode {
    if let Err(e) = runtime::configure_compute_pool() {
        eprintln!("warning: {e}");
    }
    let _thermal = runtime::ThermalMonitor::spawn(
        runtime::DEFAULT_THERMAL_INTERVAL,
        runtime::DEFAULT_THERMAL_THRESHOLD_C,
    );

    let cluster = match distributed::config::ClusterConfig::load(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let model = match model_override.or_else(|| cluster.model.clone()) {
        Some(m) => m,
        None => {
            eprintln!(
                "no model path: set `model = \"...\"` in {config_path} or pass it as the third argument"
            );
            return ExitCode::from(2);
        }
    };

    // Network I/O is tiny (one peer, ~KB frames); a current-thread tokio runtime keeps all
    // 4 cores free for the pinned rayon compute pool.
    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("failed to start tokio runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    match rt.block_on(distributed::run_stage(&cluster, node_name, &model)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("serve-stage failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Run distributed generation from the head node (node0 in nodes.toml).
fn run_generate_distributed(
    config_path: &str,
    prompt: &str,
    temp: f32,
    max_tokens: usize,
) -> ExitCode {
    if let Err(e) = runtime::configure_compute_pool() {
        eprintln!("warning: {e}");
    }
    let _thermal = runtime::ThermalMonitor::spawn(
        runtime::DEFAULT_THERMAL_INTERVAL,
        runtime::DEFAULT_THERMAL_THRESHOLD_C,
    );

    let cluster = match distributed::config::ClusterConfig::load(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let model = match cluster.model.clone() {
        Some(m) => m,
        None => {
            eprintln!("no model path: set `model = \"...\"` in {config_path}");
            return ExitCode::from(2);
        }
    };

    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("failed to start tokio runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    match rt.block_on(distributed::run_distributed_generation(
        &cluster, &model, prompt, temp, max_tokens,
    )) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("distributed generation failed: {e}");
            ExitCode::FAILURE
        }
    }
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
    use super::{cpu_features, cpu_model, device_model};

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
}

fn bench_q4_dot(iterations: usize, runs: usize) -> ExitCode {
    if iterations == 0 {
        eprintln!("iterations must be greater than zero");
        return ExitCode::from(2);
    }
    if runs == 0 {
        eprintln!("runs must be greater than zero");
        return ExitCode::from(2);
    }

    let report = q4::bench_dot_runs(iterations, runs);

    println!("NanoCamelid Q4 dot benchmark");
    println!("iterations: {}", report.iterations);
    println!("runs: {}", report.runs);
    println!("blocks_per_iteration: {}", report.blocks_per_iteration);
    println!("elements_per_iteration: {}", report.elements_per_iteration);
    println!(
        "kernel_selector_requested: {}",
        report
            .kernel_selector
            .requested
            .map(q4::Q4DotKernel::name)
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
                q4::sdot_candidate_requested()
            );
            println!("sdot_candidate_enabled: {}", q4::sdot_candidate_enabled());
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
                q4::sdot_candidate_requested()
            );
            println!("sdot_candidate_enabled: {}", q4::sdot_candidate_enabled());
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

    println!("json: {}", q4_dot_json(&report));

    ExitCode::SUCCESS
}

fn q4_dot_json(report: &q4::Q4DotBenchmarkReport) -> String {
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
    let kernel_json = format!(
        ",\"kernel_selector\":{{\"requested\":{},\"selected\":\"{}\",\"fallback_reason\":{}}}",
        report.kernel_selector.requested.map(|k| format!("\"{}\"", k.name())).unwrap_or_else(|| "null".to_string()),
        report.kernel_selector.selected.name(),
        report.kernel_selector.fallback_reason.map(|r| format!("\"{}\"", r)).unwrap_or_else(|| "null".to_string())
    );
    let suffix_json = format!("{kernel_json}{neon_json}{sdot_json}");

    format!(
        "{{\"benchmark\":\"q4-dot\",\"iterations\":{},\"runs\":{},\"blocks_per_iteration\":{},\"elements_per_iteration\":{},\"selected\":{{\"checksum\":{},\"run_ms\":{}}},\"scalar\":{{\"checksum\":{},\"run_ms\":{},\"min_ns_per_block\":{:.6},\"median_ns_per_block\":{:.6}}},\"neon_available\":{},\"dotprod_feature_detected\":{},\"sdot_candidate_requested\":{},\"sdot_candidate_enabled\":{}{}}}",
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
        q4::sdot_candidate_requested(),
        q4::sdot_candidate_enabled(),
        suffix_json
    )
}
