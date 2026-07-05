//! `nanocamelid up` / `down`: one-command bring-up of a tensor-parallel Pi
//! cluster behind a single OpenAI-compatible endpoint.
//!
//! This module is PURE ORCHESTRATION. Nothing on the numerically-sensitive data
//! path lives here: the TP wire protocol, the shard-direct GGUF loader
//! (`tp::load_tp_shard_direct`), the geometry (`tp::shard_geometry`), and the
//! `cluster_tp_node master-serve` HTTP endpoint are all consumed as-is. `up`
//! only does, mechanically and idempotently, the four things a human does by
//! hand today: read a manifest, size each node's shard from its RAM, launch the
//! workers-first / head-last star in the order the protocol requires, and
//! health-gate the endpoint.
//!
//! Phase 1 (this file) ships the orchestration brain and a fully-offline
//! `--dry-run`: parse the manifest, auto-compute a weighted KV-head split that
//! the real `shard_geometry` accepts, and print the exact per-node launch plan.
//! Live SSH launch / gate / `down` build on these same pieces.

use std::path::Path;
use std::process::ExitCode;

use crate::model::LlamaModelConfig;
use crate::{gguf, tp};

/// Cluster-wide manifest settings (`[cluster]` table).
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterCfg {
    pub model: String,
    pub serve_port: u16,
    pub worker_port: u16,
    pub ssh_user: Option<String>,
    pub ssh_key: Option<String>,
    pub bin_path: Option<String>,
    pub parity: bool,
    pub reserve_ram_mb: u64,
}

impl Default for ClusterCfg {
    fn default() -> Self {
        ClusterCfg {
            model: String::new(),
            serve_port: 8090,
            worker_port: 5921,
            ssh_user: None,
            ssh_key: None,
            bin_path: None,
            parity: false,
            reserve_ram_mb: 1500,
        }
    }
}

/// One machine in the cluster (`[[node]]` entry).
#[derive(Debug, Clone, PartialEq)]
pub struct NodeCfg {
    pub name: String,
    pub host: String,
    pub role: NodeRole,
    pub worker_port: Option<u16>,
    pub max_cores: Option<usize>,
    /// Manual weight override; when absent the planner uses probed RAM.
    pub weight: Option<f64>,
    pub remote_model_path: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    Head,
    Worker,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub cluster: ClusterCfg,
    pub nodes: Vec<NodeCfg>,
}

impl Manifest {
    pub fn head(&self) -> &NodeCfg {
        self.nodes
            .iter()
            .find(|n| n.role == NodeRole::Head)
            .expect("validated: exactly one head")
    }

    /// Workers in shard-index order (manifest order). Shard 1 is the first
    /// worker, matching master-serve's address-order connect + HELLO check.
    pub fn workers(&self) -> impl Iterator<Item = &NodeCfg> {
        self.nodes.iter().filter(|n| n.role == NodeRole::Worker)
    }

    fn validate(&self) -> Result<(), String> {
        if self.cluster.model.trim().is_empty() {
            return Err("[cluster].model is required".to_owned());
        }
        let heads = self
            .nodes
            .iter()
            .filter(|n| n.role == NodeRole::Head)
            .count();
        if heads != 1 {
            return Err(format!(
                "manifest needs exactly one head node, found {heads}"
            ));
        }
        if self.workers().count() == 0 {
            return Err("manifest needs at least one worker node".to_owned());
        }
        let mut seen = std::collections::BTreeSet::new();
        for node in &self.nodes {
            if node.name.trim().is_empty() {
                return Err("every [[node]] needs a name".to_owned());
            }
            if node.host.trim().is_empty() {
                return Err(format!("node {} needs a host", node.name));
            }
            if !seen.insert(&node.name) {
                return Err(format!("duplicate node name {}", node.name));
            }
        }
        Ok(())
    }
}

/// The model geometry the planner keys off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    pub kv_total: usize,
    pub ffn_len: usize,
}

impl ModelShape {
    fn from_config(config: &LlamaModelConfig) -> ModelShape {
        ModelShape {
            kv_total: config.attention_head_count_kv,
            ffn_len: config.feed_forward_length,
        }
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Auto-compute the weighted KV-head shard split.
///
/// Each share `s` must satisfy `shard_geometry`'s per-share FFN constraint
/// `(ffn_len * s) % (kv_total*32) == 0`, so every valid share is a multiple of
/// `step = (kv_total*32) / gcd(ffn_len, kv_total*32)`. We therefore apportion in
/// units of `step`: `kv_total/step` units are handed out across the nodes by
/// weight (largest-remainder), one unit minimum each, and each share is
/// `units * step`. The result sums to `kv_total`, is all-nonzero, and satisfies
/// the FFN constraint by construction — but we still hand it to the real
/// `shard_geometry` as the final authority when a config is available (see
/// `plan_and_validate`).
pub fn plan_shares(weights: &[f64], shape: ModelShape) -> Result<Vec<usize>, String> {
    let node_count = weights.len();
    if node_count == 0 {
        return Err("no nodes to plan over".to_owned());
    }
    if node_count < 2 {
        return Err("tensor parallelism needs at least 2 nodes (1 head + 1 worker)".to_owned());
    }
    let ModelShape { kv_total, ffn_len } = shape;
    if kv_total == 0 {
        return Err("model reports zero kv heads".to_owned());
    }
    let step = (kv_total * 32) / gcd(ffn_len, kv_total * 32);
    if !kv_total.is_multiple_of(step) {
        return Err(format!(
            "model geometry admits no valid split: share step {step} does not divide {kv_total} kv heads (ffn {ffn_len})"
        ));
    }
    let total_units = kv_total / step;
    if total_units < node_count {
        return Err(format!(
            "too many nodes for this model: {node_count} nodes but only {total_units} allocatable shard unit(s) (kv_total {kv_total}, step {step}). Use fewer nodes or a larger model."
        ));
    }

    // Largest-remainder apportionment of `total_units` by weight, min 1 each.
    let weight_sum: f64 = weights.iter().sum();
    let weight_sum = if weight_sum > 0.0 {
        weight_sum
    } else {
        node_count as f64
    };
    let ideal: Vec<f64> = weights
        .iter()
        .map(|&w| {
            let w = if w > 0.0 { w } else { 1.0 };
            (w / weight_sum) * total_units as f64
        })
        .collect();
    let mut units: Vec<usize> = ideal.iter().map(|&x| (x.floor() as usize).max(1)).collect();
    // Fix the sum: we start with at-least-1 each, so we may be over or under.
    let assigned: usize = units.iter().sum();
    if assigned < total_units {
        // Hand out the remaining units to the largest fractional remainders.
        let mut order: Vec<usize> = (0..node_count).collect();
        order.sort_by(|&a, &b| {
            (ideal[b] - ideal[b].floor())
                .partial_cmp(&(ideal[a] - ideal[a].floor()))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut extra = total_units - assigned;
        for &i in order.iter().cycle() {
            if extra == 0 {
                break;
            }
            units[i] += 1;
            extra -= 1;
        }
    } else if assigned > total_units {
        // Trim from the nodes with the most units (never below 1).
        let mut over = assigned - total_units;
        while over > 0 {
            let (idx, _) = units
                .iter()
                .enumerate()
                .filter(|(_, u)| **u > 1)
                .max_by_key(|(_, u)| **u)
                .ok_or_else(|| "cannot fit one unit per node".to_owned())?;
            units[idx] -= 1;
            over -= 1;
        }
    }
    let shares: Vec<usize> = units.iter().map(|&u| u * step).collect();
    debug_assert_eq!(shares.iter().sum::<usize>(), kv_total);
    Ok(shares)
}

/// Plan the split and validate it against the real library geometry.
pub fn plan_and_validate(weights: &[f64], config: &LlamaModelConfig) -> Result<Vec<usize>, String> {
    let shares = plan_shares(weights, ModelShape::from_config(config))?;
    for idx in 0..shares.len() {
        tp::shard_geometry(config, &shares, idx)
            .map_err(|e| format!("planned shares {shares:?} rejected by shard_geometry: {e}"))?;
    }
    Ok(shares)
}

/// The env prefix every launched process gets: spin-pool off (starvation fix),
/// swizzle off (TP requirement), per-node core cap (brown-out mitigation), and
/// optional parity pin.
pub fn env_prefix(node: &NodeCfg, cluster: &ClusterCfg) -> String {
    let mut parts = vec![
        "NANOCAMELID_SPIN_POOL=0".to_owned(),
        "NANOCAMELID_Q4_SWIZZLE_1X4=0".to_owned(),
        "NANOCAMELID_Q8_SWIZZLE_1X4=0".to_owned(),
    ];
    if cluster.parity {
        parts.push("NANOCAMELID_TP_PARITY=1".to_owned());
    }
    let mut prefix = parts.join(" ");
    if let Some(n) = node.max_cores {
        // taskset pins cores; RAYON_NUM_THREADS bounds the pool. Belt-and-braces
        // against the camelid1-style PSU brown-out under full-core load.
        let last = n.saturating_sub(1);
        prefix = format!("taskset -c 0-{last} env RAYON_NUM_THREADS={n} {prefix}");
    }
    prefix
}

fn shares_arg(shares: &[usize]) -> String {
    shares
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn node_model_path(node: &NodeCfg, cluster: &ClusterCfg) -> String {
    node.remote_model_path
        .clone()
        .unwrap_or_else(|| cluster.model.clone())
}

fn bin(cluster: &ClusterCfg) -> String {
    cluster
        .bin_path
        .clone()
        .unwrap_or_else(|| "cluster_tp_node".to_owned())
}

/// The exact remote command to launch a worker (shard `shard_idx`).
pub fn worker_cmd(
    node: &NodeCfg,
    cluster: &ClusterCfg,
    shard_idx: usize,
    shares: &[usize],
) -> String {
    let port = node.worker_port.unwrap_or(cluster.worker_port);
    format!(
        "{} {} worker {} 0.0.0.0:{} {} {}",
        env_prefix(node, cluster),
        bin(cluster),
        node_model_path(node, cluster),
        port,
        shard_idx,
        shares_arg(shares),
    )
}

/// The exact remote command to launch the head (master-serve = the endpoint).
pub fn master_cmd(manifest: &Manifest, shares: &[usize]) -> String {
    let cluster = &manifest.cluster;
    let head = manifest.head();
    let worker_addrs = manifest
        .workers()
        .map(|w| {
            format!(
                "{}:{}",
                w.host,
                w.worker_port.unwrap_or(cluster.worker_port)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{} {} master-serve {} {} {} {}",
        env_prefix(head, cluster),
        bin(cluster),
        node_model_path(head, cluster),
        worker_addrs,
        shares_arg(shares),
        cluster.serve_port,
    )
}

/// Read the model shape from a local GGUF (used by `--dry-run` and by the head
/// when the model path resolves locally).
fn read_local_shape(model_path: &str) -> Result<(LlamaModelConfig, ModelShape), String> {
    let gguf = gguf::read_file(Path::new(model_path))
        .map_err(|e| format!("cannot read model {model_path}: {e}"))?;
    let config = LlamaModelConfig::from_gguf(&gguf)
        .map_err(|e| format!("cannot parse model config from {model_path}: {e}"))?;
    let shape = ModelShape::from_config(&config);
    Ok((config, shape))
}

/// Node weights for the planner: manual `weight` override, else probed RAM
/// (Phase 1b), else equal weighting so `--dry-run` works fully offline.
fn planner_weights(manifest: &Manifest) -> Vec<f64> {
    manifest
        .nodes
        .iter()
        .map(|n| n.weight.unwrap_or(1.0))
        .collect()
}

fn print_plan(manifest: &Manifest, shares: &[usize]) {
    let cluster = &manifest.cluster;
    println!("cluster launch plan (workers first, head last):");
    println!("  model:      {}", cluster.model);
    println!("  serve_port: {}", cluster.serve_port);
    println!(
        "  shares:     {} (sum {})",
        shares_arg(shares),
        shares.iter().sum::<usize>()
    );
    println!();
    // Workers are shard 1..N in manifest order.
    for (worker_idx, w) in manifest.workers().enumerate() {
        let shard_idx = worker_idx + 1;
        println!(
            "  [worker {shard_idx}] {} ({})  ssh> {}",
            w.name,
            w.host,
            worker_cmd(w, cluster, shard_idx, shares)
        );
    }
    let head = manifest.head();
    println!(
        "  [head    0] {} ({})  ssh> {}",
        head.name,
        head.host,
        master_cmd(manifest, shares)
    );
    println!();
    println!(
        "endpoint (after bring-up): http://{}:{}/v1/chat/completions",
        head.host, cluster.serve_port
    );
}

/// `nanocamelid up --cluster <path> [--model X] [--dry-run]`.
pub fn run_up(args: &[String]) -> ExitCode {
    let opts = match parse_up_args(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let text = match std::fs::read_to_string(&opts.manifest) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read manifest {}: {e}", opts.manifest);
            return ExitCode::from(2);
        }
    };
    let mut manifest = match parse_manifest(&text) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("manifest error: {e}");
            return ExitCode::from(2);
        }
    };
    if let Some(model) = opts.model {
        manifest.cluster.model = model;
    }
    if let Err(e) = manifest.validate() {
        eprintln!("manifest error: {e}");
        return ExitCode::from(2);
    }

    // Model shape: read the GGUF locally when the path resolves here (the common
    // case for --dry-run from a machine that has the model). Over-SSH shape
    // fetch for head-only-staged models lands with live launch (Phase 1b).
    let (config, shape) = match read_local_shape(&manifest.cluster.model) {
        Ok(cs) => cs,
        Err(e) => {
            eprintln!("cannot determine model shape: {e}");
            eprintln!(
                "hint: run `up` from a host that has the GGUF, or point --model at a local copy"
            );
            return ExitCode::from(3);
        }
    };

    let weights = planner_weights(&manifest);
    let shares = match plan_and_validate(&weights, &config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("planning failed: {e}");
            return ExitCode::from(3);
        }
    };
    let _ = shape;

    print_plan(&manifest, &shares);

    if opts.dry_run {
        println!();
        println!(
            "dry-run: no processes launched. Re-run without --dry-run to bring the cluster up."
        );
        return ExitCode::SUCCESS;
    }

    // Live SSH launch + health-gate + state lockfile land in Phase 1b; the
    // planner and plan output above are what `up` commits to now.
    eprintln!(
        "live launch is not yet wired in this build; use --dry-run to review the plan. \
         (The auto-computed split above is validated against the real shard geometry.)"
    );
    ExitCode::from(1)
}

/// `nanocamelid down` (Phase 1b: reads the state lockfile and reverses launch).
pub fn run_down(_args: &[String]) -> ExitCode {
    eprintln!("`nanocamelid down` lands with live launch (Phase 1b).");
    ExitCode::from(1)
}

struct UpOpts {
    manifest: String,
    model: Option<String>,
    dry_run: bool,
}

fn parse_up_args(args: &[String]) -> Result<UpOpts, String> {
    let mut manifest = None;
    let mut model = None;
    let mut dry_run = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--cluster" => {
                manifest = Some(args.get(i + 1).ok_or("--cluster needs a path")?.clone());
                i += 2;
            }
            "--model" => {
                model = Some(args.get(i + 1).ok_or("--model needs a path")?.clone());
                i += 2;
            }
            "--dry-run" => {
                dry_run = true;
                i += 1;
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(UpOpts {
        manifest: manifest.ok_or("up requires --cluster <manifest.toml>")?,
        model,
        dry_run,
    })
}

// -------------------------------------------------------------------------
// Minimal TOML-subset parser for the manifest. Supports `[table]`,
// `[[array]]`, `key = "string" | int | float | true/false`, and `#` comments.
// Kept in-house to preserve the zero-extra-dependency footprint; anything it
// does not understand is a hard error, and the parsed plan is echoed by
// `--dry-run` so the operator can audit it.
// -------------------------------------------------------------------------

pub fn parse_manifest(text: &str) -> Result<Manifest, String> {
    let mut cluster = ClusterCfg::default();
    let mut nodes: Vec<NodeCfg> = Vec::new();
    let mut section = Section::None;

    for (lineno, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        let n = lineno + 1;
        if line == "[cluster]" {
            section = Section::Cluster;
            continue;
        }
        if line == "[[node]]" {
            nodes.push(NodeCfg {
                name: String::new(),
                host: String::new(),
                role: NodeRole::Worker,
                worker_port: None,
                max_cores: None,
                weight: None,
                remote_model_path: None,
            });
            section = Section::Node;
            continue;
        }
        if line.starts_with('[') {
            return Err(format!("line {n}: unsupported section {line}"));
        }
        let (key, val) = line
            .split_once('=')
            .ok_or_else(|| format!("line {n}: expected `key = value`, got {line}"))?;
        let key = key.trim();
        let val = val.trim();
        match section {
            Section::None => return Err(format!("line {n}: key {key} outside any [section]")),
            Section::Cluster => apply_cluster(&mut cluster, key, val, n)?,
            Section::Node => {
                let node = nodes.last_mut().expect("node section pushed one");
                apply_node(node, key, val, n)?;
            }
        }
    }

    let manifest = Manifest { cluster, nodes };
    manifest.validate()?;
    Ok(manifest)
}

enum Section {
    None,
    Cluster,
    Node,
}

fn strip_comment(line: &str) -> &str {
    // Comments start at the first `#` that is not inside a quoted string.
    let mut in_str = false;
    for (i, ch) in line.char_indices() {
        match ch {
            '"' => in_str = !in_str,
            '#' if !in_str => return &line[..i],
            _ => {}
        }
    }
    line
}

fn parse_string(val: &str, n: usize) -> Result<String, String> {
    let bytes = val.as_bytes();
    if bytes.len() >= 2 && bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"' {
        Ok(val[1..val.len() - 1].to_owned())
    } else {
        Err(format!("line {n}: expected a quoted string, got {val}"))
    }
}

fn parse_u64(val: &str, n: usize) -> Result<u64, String> {
    val.parse::<u64>()
        .map_err(|_| format!("line {n}: expected an integer, got {val}"))
}

fn apply_cluster(cfg: &mut ClusterCfg, key: &str, val: &str, n: usize) -> Result<(), String> {
    match key {
        "model" => cfg.model = parse_string(val, n)?,
        "serve_port" => {
            cfg.serve_port = parse_u64(val, n)?
                .try_into()
                .map_err(|_| format!("line {n}: serve_port out of range"))?
        }
        "worker_port" => {
            cfg.worker_port = parse_u64(val, n)?
                .try_into()
                .map_err(|_| format!("line {n}: worker_port out of range"))?
        }
        "ssh_user" => cfg.ssh_user = Some(parse_string(val, n)?),
        "ssh_key" => cfg.ssh_key = Some(parse_string(val, n)?),
        "bin_path" => cfg.bin_path = Some(parse_string(val, n)?),
        "reserve_ram_mb" => cfg.reserve_ram_mb = parse_u64(val, n)?,
        "parity" => cfg.parity = val == "true",
        other => return Err(format!("line {n}: unknown [cluster] key {other}")),
    }
    Ok(())
}

fn apply_node(node: &mut NodeCfg, key: &str, val: &str, n: usize) -> Result<(), String> {
    match key {
        "name" => node.name = parse_string(val, n)?,
        "host" => node.host = parse_string(val, n)?,
        "role" => {
            node.role = match parse_string(val, n)?.as_str() {
                "head" => NodeRole::Head,
                "worker" => NodeRole::Worker,
                other => return Err(format!("line {n}: role must be head|worker, got {other}")),
            }
        }
        "worker_port" => {
            node.worker_port = Some(
                parse_u64(val, n)?
                    .try_into()
                    .map_err(|_| format!("line {n}: worker_port out of range"))?,
            )
        }
        "max_cores" => node.max_cores = Some(parse_u64(val, n)? as usize),
        "weight" => {
            node.weight = Some(
                val.parse::<f64>()
                    .map_err(|_| format!("line {n}: weight must be a number, got {val}"))?,
            )
        }
        "remote_model_path" => node.remote_model_path = Some(parse_string(val, n)?),
        other => return Err(format!("line {n}: unknown [[node]] key {other}")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
# a small cluster
[cluster]
model = "/mnt/nanocamelid/models/llama-70b-q4_0.gguf"
serve_port = 8090
ssh_user = "tooleman"

[[node]]
name = "camelid0"   # head
host = "camelid0.local"
role = "head"

[[node]]
name = "camelid1"
host = "camelid1.local"
role = "worker"
max_cores = 2       # brown-out cap

[[node]]
name = "camelid2"
host = "camelid2.local"
role = "worker"
weight = 1.5
"#;

    #[test]
    fn parse_manifest_reads_a_full_cluster() {
        let m = parse_manifest(SAMPLE).expect("parses");
        assert_eq!(m.cluster.serve_port, 8090);
        assert_eq!(m.cluster.ssh_user.as_deref(), Some("tooleman"));
        assert_eq!(m.nodes.len(), 3);
        assert_eq!(m.head().name, "camelid0");
        assert_eq!(m.workers().count(), 2);
        let c1 = m.nodes.iter().find(|n| n.name == "camelid1").unwrap();
        assert_eq!(c1.max_cores, Some(2));
        let c2 = m.nodes.iter().find(|n| n.name == "camelid2").unwrap();
        assert_eq!(c2.weight, Some(1.5));
    }

    #[test]
    fn parse_manifest_rejects_zero_or_many_heads() {
        let none =
            "[cluster]\nmodel=\"m.gguf\"\n[[node]]\nname=\"a\"\nhost=\"a\"\nrole=\"worker\"\n";
        assert!(parse_manifest(none).is_err());
        let two = "[cluster]\nmodel=\"m.gguf\"\n[[node]]\nname=\"a\"\nhost=\"a\"\nrole=\"head\"\n[[node]]\nname=\"b\"\nhost=\"b\"\nrole=\"head\"\n";
        assert!(parse_manifest(two).is_err());
    }

    #[test]
    fn parse_manifest_rejects_unknown_keys() {
        let bad = "[cluster]\nmodel=\"m.gguf\"\nbogus=\"x\"\n";
        assert!(parse_manifest(bad).is_err());
    }

    #[test]
    fn strip_comment_keeps_hashes_inside_strings() {
        assert_eq!(
            strip_comment(r#"host = "a#b" # trailing"#).trim(),
            r#"host = "a#b""#
        );
    }

    // --- planner ---

    #[test]
    fn plan_shares_llama70b_step_one_weighted() {
        // Llama-70B: 8 kv heads, ffn 28672. step = 256/gcd(28672,256)=256/256=1.
        let shape = ModelShape {
            kv_total: 8,
            ffn_len: 28672,
        };
        // Head weak, two workers strong -> heavier nodes get more heads.
        let shares = plan_shares(&[1.0, 3.0, 4.0], shape).unwrap();
        assert_eq!(shares.iter().sum::<usize>(), 8);
        assert!(shares.iter().all(|&s| s >= 1));
        // Monotonic with weight (largest-remainder): node 2 >= node 1 >= node 0.
        assert!(shares[2] >= shares[1] && shares[1] >= shares[0]);
    }

    #[test]
    fn plan_shares_equal_weights_splits_evenly() {
        let shape = ModelShape {
            kv_total: 8,
            ffn_len: 28672,
        };
        let shares = plan_shares(&[1.0, 1.0, 1.0, 1.0], shape).unwrap();
        assert_eq!(shares, vec![2, 2, 2, 2]);
    }

    #[test]
    fn plan_shares_respects_ffn_step_gt_one() {
        // Contrived: kv_total=4, ffn_len=32. step=(4*32)/gcd(32,128)=128/32=4.
        // total_units = 4/4 = 1 -> only ONE node can be served; 2 nodes must fail.
        let shape = ModelShape {
            kv_total: 4,
            ffn_len: 32,
        };
        assert!(plan_shares(&[1.0, 1.0], shape).is_err());
        // kv_total=8, ffn_len=64: step=(8*32)/gcd(64,256)=256/64=4, units=2.
        let shape2 = ModelShape {
            kv_total: 8,
            ffn_len: 64,
        };
        let shares = plan_shares(&[1.0, 1.0], shape2).unwrap();
        assert_eq!(shares, vec![4, 4]); // each share a multiple of step=4
        // 3 nodes into 2 units must fail.
        assert!(plan_shares(&[1.0, 1.0, 1.0], shape2).is_err());
    }

    #[test]
    fn plan_shares_needs_two_nodes() {
        let shape = ModelShape {
            kv_total: 8,
            ffn_len: 28672,
        };
        assert!(plan_shares(&[1.0], shape).is_err());
    }

    // --- launch command builders ---

    fn sample_manifest() -> Manifest {
        parse_manifest(SAMPLE).unwrap()
    }

    #[test]
    fn env_prefix_has_safety_defaults_and_core_cap() {
        let m = sample_manifest();
        let c1 = m.nodes.iter().find(|n| n.name == "camelid1").unwrap();
        let p = env_prefix(c1, &m.cluster);
        assert!(p.contains("NANOCAMELID_SPIN_POOL=0"));
        assert!(p.contains("NANOCAMELID_Q4_SWIZZLE_1X4=0"));
        assert!(p.contains("NANOCAMELID_Q8_SWIZZLE_1X4=0"));
        // camelid1 has max_cores=2 -> taskset + RAYON cap.
        assert!(p.contains("taskset -c 0-1"));
        assert!(p.contains("RAYON_NUM_THREADS=2"));
    }

    #[test]
    fn worker_and_master_commands_are_well_formed() {
        let m = sample_manifest();
        let shares = vec![2usize, 3, 3];
        let c1 = m.nodes.iter().find(|n| n.name == "camelid1").unwrap();
        let w = worker_cmd(c1, &m.cluster, 1, &shares);
        assert!(w.contains("cluster_tp_node worker"));
        assert!(w.contains("0.0.0.0:5921 1 2,3,3"));
        let master = master_cmd(&m, &shares);
        assert!(master.contains("cluster_tp_node master-serve"));
        // worker addresses in shard-index (manifest) order, then shares, then port
        assert!(master.contains("camelid1.local:5921,camelid2.local:5921 2,3,3 8090"));
    }
}
