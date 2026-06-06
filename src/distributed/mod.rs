//! Pipeline-parallel distribution across the 3-node cluster.
//!
//! Layers are split into contiguous ranges (one per node). The head (node0) embeds and runs
//! its own layer range, then RPCs each downstream stage with the hidden-state vector; the
//! tail stage runs the final RMSNorm + output projection and returns logits. Sampling/RNG
//! stays entirely on the head, so output is bit-identical to a single-node run.
//!
//! Transport is raw tokio TCP with length-prefixed binary frames (see [`frame`]).

pub mod client;
pub mod config;
pub mod frame;
pub mod node;

use crate::gguf::GgufFile;

pub use client::run_distributed_generation;
pub use node::run_stage;

/// RoPE scaling parameters read from GGUF metadata. Every node reads them from the same
/// model file, so they are identical across the pipeline (required for bit-identical output).
#[derive(Debug, Clone, Copy, Default)]
pub struct RopeParams {
    pub factor: Option<f32>,
    pub orig_ctx: Option<f32>,
    pub low: Option<f32>,
    pub high: Option<f32>,
}

impl RopeParams {
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        Self {
            factor: gguf.metadata_f32("llama.rope.scaling.factor"),
            orig_ctx: gguf
                .metadata_u32("llama.rope.scaling.original_context_length")
                .map(|v| v as f32),
            low: gguf.metadata_f32("llama.rope.scaling.low_freq_factor"),
            high: gguf.metadata_f32("llama.rope.scaling.high_freq_factor"),
        }
    }
}
