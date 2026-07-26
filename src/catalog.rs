//! Model catalog and downloads for the web UI.
//!
//! Shared by the single-node `webui` router and the cluster head so the two
//! cannot drift. Downloads shell out to `curl` because NanoCamelid carries no
//! HTTP client and no TLS crate, and they are always spawned, never awaited:
//! both servers are single-threaded accept loops and a multi-GB transfer would
//! otherwise wedge them.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use crate::model;

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Model catalog + downloads.
//
// The UI's Models page needs a curated catalog it can browse and a way to fetch
// a GGUF. NanoCamelid has a hard minimal-dependency policy (core_affinity,
// memmap2, rayon, serde_json) so there is no HTTP client and no TLS crate here;
// HuggingFace is HTTPS-only, so downloads shell out to `curl`, which is present
// on every target.
//
// The accept loop is single-threaded, so the transfer MUST NOT be awaited
// inline: `install` spawns curl and returns immediately, and progress is read
// from the size of the .part file plus a non-blocking try_wait() on the child.
// A multi-GB download would otherwise wedge the whole server.

#[derive(Debug)]
struct DownloadJob {
    id: String,
    repo_id: String,
    filename: String,
    continuation_mode: String,
    total_bytes: u64,
    part_path: PathBuf,
    final_path: PathBuf,
    child: Option<std::process::Child>,
    status: String,
}

fn download_registry() -> &'static Mutex<Vec<DownloadJob>> {
    static REG: OnceLock<Mutex<Vec<DownloadJob>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(Vec::new()))
}

/// A small curated catalog of GGUFs known to run on Pi-class hardware.
///
/// Deliberately static: the UI debounces its search box to 350 ms per keystroke,
/// and a live HuggingFace query would put a blocking network round trip on the
/// single-threaded accept loop for every one of them.
const CATALOG: &[(&str, &str, &str, u64)] = &[
    // (catalog_id, repo_id, filename, size_bytes)
    (
        "llama32-1b-instruct-q4_0",
        "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-Q4_0.gguf",
        770928480,
    ),
    (
        "llama32-3b-instruct-q4_0",
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-Q4_0.gguf",
        1917190432,
    ),
    (
        "qwen25-05b-instruct-q4_0",
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_0.gguf",
        428730208,
    ),
    (
        "smollm2-360m-instruct-q8_0",
        "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
        "smollm2-360m-instruct-q8_0.gguf",
        386404992,
    ),
];

pub fn catalog_json() -> String {
    let mut items = String::new();
    for (i, (cid, repo, file, bytes)) in CATALOG.iter().enumerate() {
        if i > 0 {
            items.push(',');
        }
        let quant = model::quant_label_from_filename(file).unwrap_or("Q4_0");
        items.push_str(&format!(
            "{{\"catalog_id\":\"{cid}\",\"name\":\"{name}\",\"repo_id\":\"{repo}\",\"filename\":\"{file}\",\"size_bytes\":{bytes},\"group\":\"curated\",\"architecture\":\"llama\",\"quant\":\"{quant}\",\"oracle_qualified\":false,\"downloads\":0,\"likes\":0}}",
            cid = json_escape(cid),
            name = json_escape(file.trim_end_matches(".gguf")),
            repo = json_escape(repo),
            file = json_escape(file),
            quant = json_escape(quant),
        ));
    }
    format!("{{\"items\":[{items}],\"next_cursor\":null}}")
}

/// Refresh each job's status without blocking, then render the array the UI polls.
pub fn downloads_json(model_dir: &Path) -> String {
    let Ok(mut jobs) = download_registry().lock() else {
        return String::from("[]");
    };
    let mut out = String::from("[");
    for (i, job) in jobs.iter_mut().enumerate() {
        if job.status == "downloading" {
            // try_wait is non-blocking; .wait() here would hang the accept loop
            // for the length of a multi-GB transfer.
            match job.child.as_mut().map(|c| c.try_wait()) {
                Some(Ok(Some(status))) => {
                    if status.success() && std::fs::rename(&job.part_path, &job.final_path).is_ok()
                    {
                        job.status = "completed".to_owned();
                    } else {
                        let _ = std::fs::remove_file(&job.part_path);
                        job.status = "failed".to_owned();
                    }
                    job.child = None;
                }
                Some(Err(_)) => {
                    job.status = "failed".to_owned();
                    job.child = None;
                }
                _ => {}
            }
        }
        let observed = if job.status == "completed" {
            std::fs::metadata(&job.final_path)
                .map(|m| m.len())
                .unwrap_or(job.total_bytes)
        } else {
            std::fs::metadata(&job.part_path)
                .map(|m| m.len())
                .unwrap_or(0)
        };
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"id\":\"{id}\",\"repo_id\":\"{repo}\",\"filename\":\"{file}\",\"continuation_mode\":\"{mode}\",\"total_bytes\":{total},\"bytes_downloaded\":{done},\"status\":\"{status}\"}}",
            id = json_escape(&job.id),
            repo = json_escape(&job.repo_id),
            file = json_escape(&job.filename),
            mode = json_escape(&job.continuation_mode),
            total = job.total_bytes,
            done = observed,
            status = job.status,
        ));
    }
    let _ = model_dir;
    out.push(']');
    out
}

/// Minimal string-field reader for the small flat JSON bodies the UI posts.
pub fn json_field<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let start = body.find(&needle)? + needle.len();
    let rest = &body[start..];
    let colon = rest.find(':')? + 1;
    let rest = rest[colon..].trim_start();
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        Some(&stripped[..end])
    } else {
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        Some(rest[..end].trim())
    }
}

pub fn start_download(body: &str, model_dir: &Path) -> (u16, String) {
    let (Some(catalog_id), Some(repo_id), Some(filename)) = (
        json_field(body, "catalog_id"),
        json_field(body, "repo_id"),
        json_field(body, "filename"),
    ) else {
        return (
            400,
            String::from("{\"error\":{\"message\":\"missing catalog_id, repo_id or filename\"}}"),
        );
    };
    if !model::is_model_gguf(filename) || filename.contains('/') || filename.contains("..") {
        return (
            400,
            String::from("{\"error\":{\"message\":\"invalid filename\"}}"),
        );
    }
    let total_bytes: u64 = json_field(body, "size_bytes")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mode = json_field(body, "continuation_mode")
        .unwrap_or("download")
        .to_owned();

    let Ok(mut jobs) = download_registry().lock() else {
        return (
            500,
            String::from("{\"error\":{\"message\":\"registry unavailable\"}}"),
        );
    };
    if jobs
        .iter()
        .any(|j| j.id == catalog_id && j.status == "downloading")
    {
        // The UI treats this as a rejoin of an in-flight transfer, not an error.
        return (
            409,
            String::from("{\"error\":{\"code\":\"download_already_running\"}}"),
        );
    }
    jobs.retain(|j| j.id != catalog_id);

    let final_path = model_dir.join(filename);
    let part_path = model_dir.join(format!("{filename}.part"));
    let url = format!("https://huggingface.co/{repo_id}/resolve/main/{filename}");
    // -f so an HTML error page is never written as if it were a model, -L because
    // HuggingFace 302s to a CDN, -C - to resume a partial. Spawned, never waited on.
    let child = Command::new("curl")
        .args([
            "-f",
            "-L",
            "-C",
            "-",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "30",
            "--retry",
            "10",
            "--retry-delay",
            "2",
            "--retry-all-errors",
            "-o",
        ])
        .arg(&part_path)
        .arg(&url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
    match child {
        Ok(child) => {
            jobs.push(DownloadJob {
                id: catalog_id.to_owned(),
                repo_id: repo_id.to_owned(),
                filename: filename.to_owned(),
                continuation_mode: mode,
                total_bytes,
                part_path,
                final_path,
                child: Some(child),
                status: String::from("downloading"),
            });
            (200, String::from("{\"status\":\"started\"}"))
        }
        Err(err) => (
            500,
            format!(
                "{{\"error\":{{\"message\":\"could not start curl: {}\"}}}}",
                json_escape(&err.to_string())
            ),
        ),
    }
}

pub fn cancel_download(body: &str) -> (u16, String) {
    let Some(id) = json_field(body, "id") else {
        return (
            400,
            String::from("{\"error\":{\"message\":\"missing id\"}}"),
        );
    };
    let Ok(mut jobs) = download_registry().lock() else {
        return (
            500,
            String::from("{\"error\":{\"message\":\"registry unavailable\"}}"),
        );
    };
    let Some(job) = jobs.iter_mut().find(|j| j.id == id) else {
        return (
            404,
            String::from("{\"error\":{\"message\":\"unknown download\"}}"),
        );
    };
    if job.status != "downloading" {
        return (
            409,
            String::from("{\"error\":{\"code\":\"download_already_completed\"}}"),
        );
    }
    // Kill the recorded child directly. Never match by process name: a pattern
    // kill over ssh has taken out the invoking shell here before.
    if let Some(child) = job.child.as_mut() {
        let _ = child.kill();
        let _ = child.wait();
    }
    job.child = None;
    job.status = String::from("canceled");
    let _ = std::fs::remove_file(&job.part_path);
    (200, String::from("{\"status\":\"canceled\"}"))
}

/// Drop a finished record. REQUIRED: the UI throws if this is not 2xx and
/// reports a perfectly good download as failed.
pub fn ack_download(body: &str) -> (u16, String) {
    let Some(id) = json_field(body, "id") else {
        return (
            400,
            String::from("{\"error\":{\"message\":\"missing id\"}}"),
        );
    };
    let Ok(mut jobs) = download_registry().lock() else {
        return (
            500,
            String::from("{\"error\":{\"message\":\"registry unavailable\"}}"),
        );
    };
    if jobs.iter().any(|j| j.id == id && j.status == "downloading") {
        return (
            409,
            String::from("{\"error\":{\"code\":\"download_still_running\"}}"),
        );
    }
    jobs.retain(|j| j.id != id);
    (200, String::from("{\"status\":\"ok\"}"))
}
