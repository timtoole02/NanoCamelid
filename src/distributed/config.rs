//! Cluster node configuration (`config/nodes.toml`).
//!
//! Defines the 3 pipeline nodes (in order) and an optional explicit layer split. When no
//! explicit `layers` are given the layers are auto-split into contiguous, near-equal ranges
//! deterministically, so every node computes the same boundaries from `block_count`.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ClusterConfig {
    /// Default model path (used if `--model` is not passed). Each node reads the same file.
    #[serde(default)]
    pub model: Option<String>,
    /// Pipeline nodes, in stage order (node 0 = head).
    pub nodes: Vec<NodeConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeConfig {
    pub name: String,
    pub host: String,
    pub port: u16,
    /// Optional explicit `[start, end)` layer range for this node. If any node omits it, the
    /// whole cluster falls back to an even auto-split.
    #[serde(default)]
    pub layers: Option<[usize; 2]>,
}

impl ClusterConfig {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read cluster config {path}: {e}"))?;
        let cfg: ClusterConfig =
            toml::from_str(&text).map_err(|e| format!("failed to parse {path}: {e}"))?;
        if cfg.nodes.is_empty() {
            return Err("cluster config has no [[nodes]] entries".to_string());
        }
        Ok(cfg)
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.name == name)
    }

    pub fn node(&self, name: &str) -> Option<&NodeConfig> {
        self.nodes.iter().find(|n| n.name == name)
    }

    pub fn addr(&self, name: &str) -> Option<String> {
        self.node(name).map(|n| format!("{}:{}", n.host, n.port))
    }

    /// Per-node `[start, end)` layer ranges, in node order. Uses explicit `layers` if every
    /// node provides them; otherwise an even contiguous auto-split of `block_count`.
    pub fn ranges(&self, block_count: usize) -> Result<Vec<(usize, usize)>, String> {
        if self.nodes.iter().all(|n| n.layers.is_some()) {
            let ranges: Vec<(usize, usize)> =
                self.nodes.iter().map(|n| n.layers.map(|r| (r[0], r[1])).unwrap()).collect();
            // Validate the explicit split covers [0, block_count) contiguously.
            let mut expected = 0usize;
            for (start, end) in &ranges {
                if *start != expected || end < start {
                    return Err(format!(
                        "explicit layer ranges must be contiguous from 0; got start={start} end={end}, expected start={expected}"
                    ));
                }
                expected = *end;
            }
            if expected != block_count {
                return Err(format!(
                    "explicit layer ranges cover {expected} layers but model has {block_count}"
                ));
            }
            Ok(ranges)
        } else {
            Ok(auto_split(block_count, self.nodes.len()))
        }
    }
}

/// Split `block_count` layers into `n` contiguous, near-equal ranges. Earlier nodes take the
/// remainder so the result is fully deterministic and identical on every node.
pub fn auto_split(block_count: usize, n: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(n);
    let base = block_count / n;
    let rem = block_count % n;
    let mut start = 0;
    for i in 0..n {
        let size = base + if i < rem { 1 } else { 0 };
        ranges.push((start, start + size));
        start += size;
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::auto_split;

    #[test]
    fn even_split() {
        assert_eq!(auto_split(24, 3), vec![(0, 8), (8, 16), (16, 24)]);
    }

    #[test]
    fn remainder_to_earlier_nodes() {
        assert_eq!(auto_split(22, 3), vec![(0, 8), (8, 15), (15, 22)]);
        assert_eq!(auto_split(32, 3), vec![(0, 11), (11, 22), (22, 32)]);
    }
}
