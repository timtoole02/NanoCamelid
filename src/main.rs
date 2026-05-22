mod gguf;

use std::{env, fs, path::Path, process::ExitCode};

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
    println!("  nanocamelid probe    Print host CPU and runtime feature information");
    println!("  nanocamelid inspect <model.gguf>");
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

fn inspect_gguf(path: &Path) -> ExitCode {
    match gguf::inspect(path) {
        Ok(summary) => {
            println!("NanoCamelid GGUF inspect");
            println!("path: {}", path.display());
            println!("version: {}", summary.version);
            println!("tensor_count: {}", summary.tensor_count);
            println!("metadata_count: {}", summary.metadata_count);

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
