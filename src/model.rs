use std::{
    alloc::{Layout, alloc, dealloc, handle_alloc_error},
    env,
    fs::File,
    path::Path,
    ptr::NonNull,
    sync::OnceLock,
};

use crate::gguf::{GgufFile, GgufTensorDescriptor, GgufTensorType};
use crate::q8::{
    Q4_0Block, Q4_1Block, Q6KBlock, Q8_0Block, QK_K_BLOCK_SIZE, decode_q4_0_blocks,
    decode_q4_1_blocks, decode_q6_k_blocks, decode_q8_0_blocks, swizzle_q4_0_1x4,
};
use memmap2::Mmap;

pub const Q4_SWIZZLE_1X4_ENV: &str = "NANOCAMELID_Q4_SWIZZLE_1X4";
pub const Q4_PAGE_ALIGN_1X4_ENV: &str = "NANOCAMELID_Q4_PAGE_ALIGN_1X4";

#[derive(Clone, Debug, PartialEq)]
pub struct LlamaModelConfig {
    pub architecture: String,
    pub metadata_prefix: String,
    pub context_length: usize,
    pub embedding_length: usize,
    pub block_count: usize,
    pub feed_forward_length: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub rope_dimension_count: usize,
    pub rope_freq_base: f32,
    pub rms_norm_epsilon: f32,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub attention_output_width: usize,
    pub kv_width: usize,
    pub expert_count: usize,
    pub expert_used_count: usize,
}

impl LlamaModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| "missing general.architecture".to_owned())?;
        let metadata_prefix = metadata_prefix_for_arch(arch)
            .ok_or_else(|| format!("unsupported architecture: {arch}"))?;

        let context_length = metadata_u32(gguf, metadata_prefix, "context_length")?;
        let embedding_length = metadata_u32(gguf, metadata_prefix, "embedding_length")?;
        let block_count = metadata_u32(gguf, metadata_prefix, "block_count")?;
        let feed_forward_length = gguf
            .metadata_u32(&metadata_key(metadata_prefix, "feed_forward_length"))
            .or_else(|| {
                gguf.metadata_u32(&metadata_key(metadata_prefix, "feed_forward_length_expert"))
            })
            .ok_or_else(|| format!("missing {metadata_prefix}.feed_forward_length"))?
            as usize;
        let attention_head_count = gguf
            .metadata_u32(&metadata_key(metadata_prefix, "attention.head_count"))
            .ok_or_else(|| format!("missing {metadata_prefix}.attention.head_count"))?
            as usize;
        let attention_head_count_kv = gguf
            .metadata_u32(&metadata_key(metadata_prefix, "attention.head_count_kv"))
            .unwrap_or(attention_head_count as u32) as usize;
        if attention_head_count == 0 {
            return Err(format!(
                "{metadata_prefix}.attention.head_count must be greater than zero"
            ));
        }
        if !embedding_length.is_multiple_of(attention_head_count) {
            return Err(format!(
                "embedding length {embedding_length} is not divisible by attention head count {attention_head_count}"
            ));
        }
        if attention_head_count_kv == 0 {
            return Err(format!(
                "{metadata_prefix}.attention.head_count_kv must be greater than zero"
            ));
        }
        if !attention_head_count.is_multiple_of(attention_head_count_kv) {
            return Err(format!(
                "attention head count {attention_head_count} must be a multiple of kv head count {attention_head_count_kv}"
            ));
        }

        let rope_freq_base = gguf
            .metadata_f32(&metadata_key(metadata_prefix, "rope.freq_base"))
            .unwrap_or(10000.0);
        let rms_norm_epsilon = gguf
            .metadata_f32(&metadata_key(
                metadata_prefix,
                "attention.layer_norm_rms_epsilon",
            ))
            .unwrap_or(1e-5);

        // Find token embedding weight to infer vocab size if not explicitly given
        let token_emb_desc = gguf
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .ok_or_else(|| "missing token_embd.weight tensor".to_owned())?;

        let vocab_size = if let Some(v) =
            gguf.metadata_u32(&metadata_key(metadata_prefix, "vocab_size"))
        {
            v as usize
        } else {
            if token_emb_desc.dimensions.len() == 2 {
                let dims = token_emb_desc.dimensions.as_slice();
                if dims[0] == embedding_length as u64 {
                    dims[1] as usize
                } else if dims[1] == embedding_length as u64 {
                    dims[0] as usize
                } else {
                    return Err(format!(
                        "cannot infer vocab size from token_embd.weight dimensions {:?} for embedding length {embedding_length}",
                        token_emb_desc.dimensions
                    ));
                }
            } else {
                return Err("cannot infer vocab size from token_embd.weight dimensions".to_owned());
            }
        };

        let head_dim =
            gguf.metadata_u32(&metadata_key(metadata_prefix, "attention.key_length"))
                .unwrap_or((embedding_length / attention_head_count) as u32) as usize;
        let rope_dimension_count = gguf
            .metadata_u32(&metadata_key(metadata_prefix, "rope.dimension_count"))
            .unwrap_or(head_dim as u32) as usize;
        if rope_dimension_count == 0
            || rope_dimension_count > head_dim
            || !rope_dimension_count.is_multiple_of(2)
        {
            return Err(format!(
                "RoPE dimension count {rope_dimension_count} must be even and within head dimension {head_dim}"
            ));
        }
        let attention_output_width = attention_head_count * head_dim;
        let kv_width = attention_head_count_kv * head_dim;
        let expert_count = gguf
            .metadata_u32(&metadata_key(metadata_prefix, "expert_count"))
            .unwrap_or(0) as usize;
        let expert_used_count = gguf
            .metadata_u32(&metadata_key(metadata_prefix, "expert_used_count"))
            .unwrap_or(0) as usize;
        if expert_count == 0 && expert_used_count != 0 {
            return Err(format!(
                "{metadata_prefix}.expert_used_count is set but expert_count is missing"
            ));
        }
        if expert_used_count > expert_count {
            return Err(format!(
                "{metadata_prefix}.expert_used_count {expert_used_count} exceeds expert_count {expert_count}"
            ));
        }

        Ok(Self {
            architecture: arch.to_owned(),
            metadata_prefix: metadata_prefix.to_owned(),
            context_length,
            embedding_length,
            block_count,
            feed_forward_length,
            attention_head_count,
            attention_head_count_kv,
            rope_dimension_count,
            rope_freq_base,
            rms_norm_epsilon,
            vocab_size,
            head_dim,
            attention_output_width,
            kv_width,
            expert_count,
            expert_used_count,
        })
    }
}

fn metadata_key(prefix: &str, suffix: &str) -> String {
    format!("{prefix}.{suffix}")
}

pub fn metadata_prefix_for_arch(arch: &str) -> Option<&'static str> {
    match arch {
        "llama" => Some("llama"),
        "qwen2" => Some("qwen2"),
        "qwen3" => Some("qwen3"),
        "smollm3" => Some("smollm3"),
        "gemma3" => Some("gemma3"),
        "phi3" => Some("phi3"),
        "lfm2" => Some("lfm2"),
        "mistral" => Some("mistral"),
        _ => None,
    }
}

fn metadata_u32(gguf: &GgufFile, prefix: &str, suffix: &str) -> Result<usize, String> {
    gguf.metadata_u32(&metadata_key(prefix, suffix))
        .map(|value| value as usize)
        .ok_or_else(|| format!("missing {prefix}.{suffix}"))
}

pub struct LlamaLayerWeights {
    pub attention_norm: Vec<f32>,
    pub wq: QuantizedMatrix,
    pub wk: QuantizedMatrix,
    pub wav: QuantizedMatrix, // or attention_v
    pub wq_bias: Option<Vec<f32>>,
    pub wk_bias: Option<Vec<f32>>,
    pub wav_bias: Option<Vec<f32>>,
    pub wo: QuantizedMatrix,
    pub ffn_norm: Vec<f32>,
    pub ffn: LlamaFfnWeights,
}

#[allow(clippy::large_enum_variant)]
pub enum LlamaFfnWeights {
    Dense {
        w1: QuantizedMatrix,
        w3: QuantizedMatrix,
        w2: QuantizedMatrix,
    },
    MoE {
        router: Vec<f32>,
        expert_used_count: usize,
        experts: Vec<MoeExpertWeights>,
    },
}

pub struct MoeExpertWeights {
    pub w1: QuantizedMatrix,
    pub w3: QuantizedMatrix,
    pub w2: QuantizedMatrix,
}

pub struct LlamaWeights {
    pub token_embeddings: Vec<f32>, // vocab_size * embedding_length
    pub output_norm: Vec<f32>,
    pub output_projection: Option<QuantizedMatrix>, // None if tied output
    pub layers: Vec<LlamaLayerWeights>,
}

pub struct DistributedLlamaWeights {
    pub token_embeddings: Option<Vec<f32>>,
    pub output_norm: Option<Vec<f32>>,
    pub output_projection: Option<QuantizedMatrix>,
    pub layer_start: usize,
    pub layer_end: usize,
    pub layers: Vec<LlamaLayerWeights>,
}

impl DistributedLlamaWeights {
    pub fn owns_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.layer_start && layer_idx < self.layer_end
    }
}

pub enum QuantizedMatrix {
    Q8_0(Vec<Q8_0Block>),
    Q4_0(Vec<Q4_0Block>),
    Q4_1(Vec<Q4_1Block>),
    Q4_0Swizzled1x4(Q4_0Swizzled1x4Matrix),
    Q6K(Vec<Q6KBlock>),
}

pub struct Q4_0Swizzled1x4Matrix {
    pub swizzled_1x4: Vec<Q4_0Block>,
    pub page_aligned_1x4: Option<PageAlignedQ4_0Swizzled1x4>,
    pub rows: usize,
    pub cols: usize,
}

pub struct PageAlignedQ4_0Swizzled1x4 {
    chunks: Vec<NonNull<Q4_0Block>>,
    blocks_per_row: usize,
    layout: Layout,
}

unsafe impl Send for PageAlignedQ4_0Swizzled1x4 {}
unsafe impl Sync for PageAlignedQ4_0Swizzled1x4 {}

impl PageAlignedQ4_0Swizzled1x4 {
    pub(crate) fn from_swizzled(
        swizzled: &[Q4_0Block],
        rows: usize,
        blocks_per_row: usize,
    ) -> Self {
        debug_assert!(rows.is_multiple_of(4));
        debug_assert_eq!(swizzled.len(), rows * blocks_per_row);
        let chunk_blocks = blocks_per_row * 4;
        let chunk_bytes = chunk_blocks * std::mem::size_of::<Q4_0Block>();
        let layout = Layout::from_size_align(chunk_bytes, 4096).unwrap();
        let mut chunks = Vec::with_capacity(rows / 4);
        for chunk_idx in 0..rows / 4 {
            let ptr = unsafe { alloc(layout) as *mut Q4_0Block };
            let Some(non_null) = NonNull::new(ptr) else {
                handle_alloc_error(layout);
            };
            let src_start = chunk_idx * chunk_blocks;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    swizzled[src_start..src_start + chunk_blocks].as_ptr(),
                    non_null.as_ptr(),
                    chunk_blocks,
                );
            }
            chunks.push(non_null);
        }

        Self {
            chunks,
            blocks_per_row,
            layout,
        }
    }

    pub(crate) fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub(crate) fn blocks_per_row(&self) -> usize {
        self.blocks_per_row
    }

    pub(crate) fn chunk_ptr(&self, chunk_idx: usize) -> *const Q4_0Block {
        self.chunks[chunk_idx].as_ptr()
    }
}

impl Drop for PageAlignedQ4_0Swizzled1x4 {
    fn drop(&mut self) {
        for ptr in &self.chunks {
            unsafe {
                dealloc(ptr.as_ptr().cast::<u8>(), self.layout);
            }
        }
    }
}

impl LlamaWeights {
    pub fn load(path: &Path, config: &LlamaModelConfig, gguf: &GgufFile) -> Result<Self, String> {
        validate_model_tensors(gguf, config)?;
        let file = File::open(path).map_err(|e| e.to_string())?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| e.to_string())? };

        let token_embeddings = load_f32_or_f16(&mmap, gguf, "token_embd.weight")?;
        let output_norm = load_f32_or_f16(&mmap, gguf, "output_norm.weight")?;

        let output_projection = if gguf.tensors.iter().any(|t| t.name == "output.weight") {
            Some(load_quantized_matrix(&mmap, gguf, "output.weight")?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(config.block_count);
        for i in 0..config.block_count {
            layers.push(load_layer_weights(&mmap, gguf, i)?);
        }

        Ok(Self {
            token_embeddings,
            output_norm,
            output_projection,
            layers,
        })
    }

    pub fn load_distributed(
        path: &Path,
        config: &LlamaModelConfig,
        gguf: &GgufFile,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<DistributedLlamaWeights, String> {
        validate_model_tensors(gguf, config)?;
        validate_distributed_layer_range(config, start_layer, end_layer)?;
        let file = File::open(path).map_err(|e| e.to_string())?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| e.to_string())? };

        let is_first = start_layer == 0;
        let is_last = end_layer == config.block_count;
        let has_output_projection = gguf.tensors.iter().any(|t| t.name == "output.weight");

        let token_embeddings = if is_first || (is_last && !has_output_projection) {
            Some(load_f32_or_f16(&mmap, gguf, "token_embd.weight")?)
        } else {
            None
        };

        let output_norm = if is_last {
            Some(load_f32_or_f16(&mmap, gguf, "output_norm.weight")?)
        } else {
            None
        };

        let output_projection = if is_last && has_output_projection {
            Some(load_quantized_matrix(&mmap, gguf, "output.weight")?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(end_layer - start_layer);
        for i in start_layer..end_layer {
            layers.push(load_layer_weights(&mmap, gguf, i)?);
        }

        Ok(DistributedLlamaWeights {
            token_embeddings,
            output_norm,
            output_projection,
            layer_start: start_layer,
            layer_end: end_layer,
            layers,
        })
    }
}

fn load_layer_weights(
    mmap: &Mmap,
    gguf: &GgufFile,
    layer_idx: usize,
) -> Result<LlamaLayerWeights, String> {
    let i = layer_idx;
    let attention_norm = load_f32_or_f16(mmap, gguf, &format!("blk.{i}.attn_norm.weight"))?;
    let wq = load_quantized_matrix(mmap, gguf, &format!("blk.{i}.attn_q.weight"))?;
    let wk = load_quantized_matrix(mmap, gguf, &format!("blk.{i}.attn_k.weight"))?;
    let wav = load_quantized_matrix(mmap, gguf, &format!("blk.{i}.attn_v.weight"))?;
    let wq_bias = load_optional_f32_or_f16(mmap, gguf, &format!("blk.{i}.attn_q.bias"))?;
    let wk_bias = load_optional_f32_or_f16(mmap, gguf, &format!("blk.{i}.attn_k.bias"))?;
    let wav_bias = load_optional_f32_or_f16(mmap, gguf, &format!("blk.{i}.attn_v.bias"))?;
    let wo = load_quantized_matrix(mmap, gguf, &format!("blk.{i}.attn_output.weight"))?;

    let ffn_norm = load_f32_or_f16(mmap, gguf, &format!("blk.{i}.ffn_norm.weight"))?;
    let ffn = if gguf
        .tensors
        .iter()
        .any(|tensor| tensor.name == format!("blk.{i}.ffn_gate.0.weight"))
    {
        let expert_count =
            gguf.metadata_u32("llama.expert_count")
                .or_else(|| gguf.metadata_u32("mistral.expert_count"))
                .unwrap_or_else(|| count_layer_experts(gguf, i) as u32) as usize;
        let expert_used_count = gguf
            .metadata_u32("llama.expert_used_count")
            .or_else(|| gguf.metadata_u32("mistral.expert_used_count"))
            .unwrap_or(2) as usize;
        if expert_count == 0 {
            return Err(format!("layer {i} has MoE tensors but no experts"));
        }
        if expert_used_count == 0 || expert_used_count > expert_count {
            return Err(format!(
                "layer {i} invalid expert_used_count {expert_used_count} for {expert_count} experts"
            ));
        }

        let router = load_f32_or_f16(mmap, gguf, &format!("blk.{i}.ffn_gate_inp.weight"))?;
        let mut experts = Vec::with_capacity(expert_count);
        for expert_idx in 0..expert_count {
            experts.push(MoeExpertWeights {
                w1: load_quantized_matrix(
                    mmap,
                    gguf,
                    &format!("blk.{i}.ffn_gate.{expert_idx}.weight"),
                )?,
                w3: load_quantized_matrix(
                    mmap,
                    gguf,
                    &format!("blk.{i}.ffn_up.{expert_idx}.weight"),
                )?,
                w2: load_quantized_matrix(
                    mmap,
                    gguf,
                    &format!("blk.{i}.ffn_down.{expert_idx}.weight"),
                )?,
            });
        }

        LlamaFfnWeights::MoE {
            router,
            expert_used_count,
            experts,
        }
    } else {
        LlamaFfnWeights::Dense {
            w1: load_quantized_matrix(mmap, gguf, &format!("blk.{i}.ffn_gate.weight"))?,
            w3: load_quantized_matrix(mmap, gguf, &format!("blk.{i}.ffn_up.weight"))?,
            w2: load_quantized_matrix(mmap, gguf, &format!("blk.{i}.ffn_down.weight"))?,
        }
    };

    Ok(LlamaLayerWeights {
        attention_norm,
        wq,
        wk,
        wav,
        wq_bias,
        wk_bias,
        wav_bias,
        wo,
        ffn_norm,
        ffn,
    })
}

fn count_layer_experts(gguf: &GgufFile, layer_idx: usize) -> usize {
    let prefix = format!("blk.{layer_idx}.ffn_gate.");
    gguf.tensors
        .iter()
        .filter_map(|tensor| {
            let suffix = tensor.name.strip_prefix(&prefix)?;
            let (idx, tail) = suffix.split_once('.')?;
            (tail == "weight")
                .then(|| idx.parse::<usize>().ok())
                .flatten()
        })
        .max()
        .map(|idx| idx + 1)
        .unwrap_or(0)
}

fn validate_distributed_layer_range(
    config: &LlamaModelConfig,
    start_layer: usize,
    end_layer: usize,
) -> Result<(), String> {
    if start_layer >= end_layer {
        return Err(format!(
            "distributed layer range must be non-empty, got {start_layer}..{end_layer}"
        ));
    }
    if end_layer > config.block_count {
        return Err(format!(
            "distributed layer range {start_layer}..{end_layer} exceeds block count {}",
            config.block_count
        ));
    }
    Ok(())
}

pub fn validate_model_tensors(gguf: &GgufFile, config: &LlamaModelConfig) -> Result<(), String> {
    let token_embedding = load_tensor_desc(gguf, "token_embd.weight")?;
    require_descriptor_matrix_shape(
        token_embedding,
        config.embedding_length,
        config.vocab_size,
        "token embedding",
    )?;
    validate_token_row_storage_layout(
        token_embedding,
        config.embedding_length,
        config.vocab_size,
        "token embedding",
    )?;

    let output_norm = load_tensor_desc(gguf, "output_norm.weight")?;
    require_descriptor_shape(output_norm, &[config.embedding_length], "output norm")?;

    if let Ok(output) = load_tensor_desc(gguf, "output.weight") {
        require_descriptor_matrix_shape(
            output,
            config.embedding_length,
            config.vocab_size,
            "output projection",
        )?;
        validate_token_row_storage_layout(
            output,
            config.embedding_length,
            config.vocab_size,
            "output projection",
        )?;
    }

    for layer_idx in 0..config.block_count {
        require_descriptor_shape(
            load_tensor_desc(gguf, &format!("blk.{layer_idx}.attn_norm.weight"))?,
            &[config.embedding_length],
            &format!("layer {layer_idx} attention norm"),
        )?;
        require_descriptor_matrix_shape(
            load_tensor_desc(gguf, &format!("blk.{layer_idx}.attn_q.weight"))?,
            config.embedding_length,
            config.attention_output_width,
            &format!("layer {layer_idx} attention q"),
        )?;
        if let Ok(bias) = load_tensor_desc(gguf, &format!("blk.{layer_idx}.attn_q.bias")) {
            require_descriptor_shape(
                bias,
                &[config.attention_output_width],
                &format!("layer {layer_idx} attention q bias"),
            )?;
        }
        require_descriptor_matrix_shape(
            load_tensor_desc(gguf, &format!("blk.{layer_idx}.attn_k.weight"))?,
            config.embedding_length,
            config.kv_width,
            &format!("layer {layer_idx} attention k"),
        )?;
        if let Ok(bias) = load_tensor_desc(gguf, &format!("blk.{layer_idx}.attn_k.bias")) {
            require_descriptor_shape(
                bias,
                &[config.kv_width],
                &format!("layer {layer_idx} attention k bias"),
            )?;
        }
        require_descriptor_matrix_shape(
            load_tensor_desc(gguf, &format!("blk.{layer_idx}.attn_v.weight"))?,
            config.embedding_length,
            config.kv_width,
            &format!("layer {layer_idx} attention v"),
        )?;
        if let Ok(bias) = load_tensor_desc(gguf, &format!("blk.{layer_idx}.attn_v.bias")) {
            require_descriptor_shape(
                bias,
                &[config.kv_width],
                &format!("layer {layer_idx} attention v bias"),
            )?;
        }
        require_descriptor_matrix_shape(
            load_tensor_desc(gguf, &format!("blk.{layer_idx}.attn_output.weight"))?,
            config.attention_output_width,
            config.embedding_length,
            &format!("layer {layer_idx} attention output"),
        )?;
        require_descriptor_shape(
            load_tensor_desc(gguf, &format!("blk.{layer_idx}.ffn_norm.weight"))?,
            &[config.embedding_length],
            &format!("layer {layer_idx} ffn norm"),
        )?;
        if has_moe_layer(gguf, layer_idx) {
            let expert_count = config
                .expert_count
                .max(count_layer_experts(gguf, layer_idx));
            if expert_count == 0 {
                return Err(format!("layer {layer_idx} has MoE tensors but no experts"));
            }
            let expert_used_count = if config.expert_used_count == 0 {
                2
            } else {
                config.expert_used_count
            };
            if expert_used_count == 0 || expert_used_count > expert_count {
                return Err(format!(
                    "layer {layer_idx} invalid expert_used_count {expert_used_count} for {expert_count} experts"
                ));
            }
            require_descriptor_matrix_shape(
                load_tensor_desc(gguf, &format!("blk.{layer_idx}.ffn_gate_inp.weight"))?,
                config.embedding_length,
                expert_count,
                &format!("layer {layer_idx} MoE router"),
            )?;
            for expert_idx in 0..expert_count {
                require_descriptor_matrix_shape(
                    load_tensor_desc(
                        gguf,
                        &format!("blk.{layer_idx}.ffn_gate.{expert_idx}.weight"),
                    )?,
                    config.embedding_length,
                    config.feed_forward_length,
                    &format!("layer {layer_idx} expert {expert_idx} ffn gate"),
                )?;
                require_descriptor_matrix_shape(
                    load_tensor_desc(gguf, &format!("blk.{layer_idx}.ffn_up.{expert_idx}.weight"))?,
                    config.embedding_length,
                    config.feed_forward_length,
                    &format!("layer {layer_idx} expert {expert_idx} ffn up"),
                )?;
                require_descriptor_matrix_shape(
                    load_tensor_desc(
                        gguf,
                        &format!("blk.{layer_idx}.ffn_down.{expert_idx}.weight"),
                    )?,
                    config.feed_forward_length,
                    config.embedding_length,
                    &format!("layer {layer_idx} expert {expert_idx} ffn down"),
                )?;
            }
        } else {
            require_descriptor_matrix_shape(
                load_tensor_desc(gguf, &format!("blk.{layer_idx}.ffn_gate.weight"))?,
                config.embedding_length,
                config.feed_forward_length,
                &format!("layer {layer_idx} ffn gate"),
            )?;
            require_descriptor_matrix_shape(
                load_tensor_desc(gguf, &format!("blk.{layer_idx}.ffn_up.weight"))?,
                config.embedding_length,
                config.feed_forward_length,
                &format!("layer {layer_idx} ffn up"),
            )?;
            require_descriptor_matrix_shape(
                load_tensor_desc(gguf, &format!("blk.{layer_idx}.ffn_down.weight"))?,
                config.feed_forward_length,
                config.embedding_length,
                &format!("layer {layer_idx} ffn down"),
            )?;
        }
    }

    Ok(())
}

fn has_moe_layer(gguf: &GgufFile, layer_idx: usize) -> bool {
    gguf.tensors
        .iter()
        .any(|tensor| tensor.name == format!("blk.{layer_idx}.ffn_gate.0.weight"))
}

fn load_tensor_desc<'a>(
    gguf: &'a GgufFile,
    name: &str,
) -> Result<&'a GgufTensorDescriptor, String> {
    gguf.tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("tensor {name} not found in GGUF"))
}

fn tensor_bytes<'a>(mmap: &'a Mmap, desc: &GgufTensorDescriptor) -> Result<&'a [u8], String> {
    let start = usize::try_from(desc.absolute_offset).map_err(|_| {
        format!(
            "tensor {} offset {} does not fit in usize",
            desc.name, desc.absolute_offset
        )
    })?;
    let len = usize::try_from(desc.n_bytes).map_err(|_| {
        format!(
            "tensor {} byte length {} does not fit in usize",
            desc.name, desc.n_bytes
        )
    })?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| format!("tensor {} byte range overflows usize", desc.name))?;
    mmap.get(start..end).ok_or_else(|| {
        format!(
            "tensor {} byte range {}..{} exceeds mapped GGUF size {}",
            desc.name,
            start,
            end,
            mmap.len()
        )
    })
}

fn load_f32_or_f16(mmap: &Mmap, gguf: &GgufFile, name: &str) -> Result<Vec<f32>, String> {
    let desc = load_tensor_desc(gguf, name)?;
    let bytes = tensor_bytes(mmap, desc)?;

    match desc.tensor_type {
        GgufTensorType::F32 => {
            let mut data = vec![0.0; bytes.len() / 4];
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                data[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
            Ok(data)
        }
        GgufTensorType::F16 => {
            let mut data = vec![0.0; bytes.len() / 2];
            for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                data[i] = f16_bits_to_f32(bits);
            }
            Ok(data)
        }
        GgufTensorType::Q8_0 => {
            let blocks = decode_q8_0_blocks(bytes).map_err(|e| e.to_string())?;
            let mut data = Vec::with_capacity(blocks.len() * 32);
            for block in blocks {
                let scale = block.scale_f32();
                for &val in block.values() {
                    data.push(val as f32 * scale);
                }
            }
            Ok(data)
        }
        GgufTensorType::Q4_0 => {
            let blocks = decode_q4_0_blocks(bytes).map_err(|e| e.to_string())?;
            let mut data = Vec::with_capacity(blocks.len() * 32);
            for block in blocks {
                let scale = block.scale_f32();
                for val in block.unpack_values() {
                    data.push(val as f32 * scale);
                }
            }
            Ok(data)
        }
        GgufTensorType::Q4_1 => {
            let blocks = decode_q4_1_blocks(bytes).map_err(|e| e.to_string())?;
            let mut data = Vec::with_capacity(blocks.len() * 32);
            for block in blocks {
                let scale = block.scale_f32();
                let min = block.min_f32();
                for val in block.unpack_values() {
                    data.push(val as f32 * scale + min);
                }
            }
            Ok(data)
        }
        GgufTensorType::Q6K => {
            let blocks = decode_q6_k_blocks(bytes).map_err(|e| e.to_string())?;
            let mut data = Vec::with_capacity(blocks.len() * QK_K_BLOCK_SIZE);
            for block in blocks {
                let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
                block.dequantize(&mut values);
                data.extend_from_slice(&values);
            }
            Ok(data)
        }
        other => Err(format!(
            "unsupported floating point type for {name}: {other:?}"
        )),
    }
}

fn load_optional_f32_or_f16(
    mmap: &Mmap,
    gguf: &GgufFile,
    name: &str,
) -> Result<Option<Vec<f32>>, String> {
    if gguf.tensors.iter().any(|tensor| tensor.name == name) {
        load_f32_or_f16(mmap, gguf, name).map(Some)
    } else {
        Ok(None)
    }
}

fn load_quantized_matrix(
    mmap: &Mmap,
    gguf: &GgufFile,
    name: &str,
) -> Result<QuantizedMatrix, String> {
    let desc = load_tensor_desc(gguf, name)?;
    let bytes = tensor_bytes(mmap, desc)?;

    match desc.tensor_type {
        GgufTensorType::Q8_0 => decode_q8_0_blocks(bytes)
            .map(QuantizedMatrix::Q8_0)
            .map_err(|e| e.to_string()),
        GgufTensorType::Q4_0 => load_q4_0_matrix(bytes, desc),
        GgufTensorType::Q4_1 => decode_q4_1_blocks(bytes)
            .map(QuantizedMatrix::Q4_1)
            .map_err(|e| e.to_string()),
        GgufTensorType::Q6K => decode_q6_k_blocks(bytes)
            .map(QuantizedMatrix::Q6K)
            .map_err(|e| e.to_string()),
        other => Err(format!(
            "expected Q8_0, Q4_0, Q4_1, or Q6_K tensor type for {name}, got {other:?}"
        )),
    }
}

fn load_q4_0_matrix(bytes: &[u8], desc: &GgufTensorDescriptor) -> Result<QuantizedMatrix, String> {
    let row_major = decode_q4_0_blocks(bytes).map_err(|e| e.to_string())?;
    if !q4_swizzle_1x4_enabled() {
        return Ok(QuantizedMatrix::Q4_0(row_major));
    }

    let dims = descriptor_dims(desc)?;
    let Some((&cols, rest)) = dims.split_first() else {
        return Ok(QuantizedMatrix::Q4_0(row_major));
    };
    let &[rows] = rest else {
        return Ok(QuantizedMatrix::Q4_0(row_major));
    };
    if rows == 0
        || cols == 0
        || !rows.is_multiple_of(4)
        || !cols.is_multiple_of(32)
        || row_major.len() != rows * (cols / 32)
    {
        return Ok(QuantizedMatrix::Q4_0(row_major));
    }

    let blocks_per_row = cols / 32;
    let swizzled_1x4 = swizzle_q4_0_1x4(&row_major, rows, blocks_per_row);
    let page_aligned_1x4 = q4_page_align_1x4_enabled()
        .then(|| PageAlignedQ4_0Swizzled1x4::from_swizzled(&swizzled_1x4, rows, blocks_per_row));
    Ok(QuantizedMatrix::Q4_0Swizzled1x4(Q4_0Swizzled1x4Matrix {
        swizzled_1x4,
        page_aligned_1x4,
        rows,
        cols,
    }))
}

fn q4_swizzle_1x4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(Q4_SWIZZLE_1X4_ENV)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
            .unwrap_or(true)
    })
}

fn q4_page_align_1x4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(Q4_PAGE_ALIGN_1X4_ENV)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
            .unwrap_or(false)
    })
}

fn require_descriptor_shape(
    tensor: &GgufTensorDescriptor,
    expected: &[usize],
    role: &str,
) -> Result<(), String> {
    let actual = descriptor_dims(tensor)?;
    if actual != expected {
        return Err(format!(
            "{role} tensor {} expected descriptor shape {:?}, got {:?}",
            tensor.name, expected, actual
        ));
    }
    Ok(())
}

fn require_descriptor_matrix_shape(
    tensor: &GgufTensorDescriptor,
    input_width: usize,
    output_width: usize,
    role: &str,
) -> Result<(), String> {
    let actual = descriptor_dims(tensor)?;
    let direct = [input_width, output_width];
    let transposed = [output_width, input_width];
    if actual.as_slice() != direct && actual.as_slice() != transposed {
        return Err(format!(
            "{role} tensor {} expected descriptor shape {:?} or {:?}, got {:?}",
            tensor.name, direct, transposed, actual
        ));
    }
    Ok(())
}

fn validate_token_row_storage_layout(
    tensor: &GgufTensorDescriptor,
    hidden_width: usize,
    vocab_size: usize,
    role: &str,
) -> Result<(), String> {
    let actual = descriptor_dims(tensor)?;
    let (row_values, row_count, layout) = match actual.as_slice() {
        [hidden, vocab] if *hidden == hidden_width && *vocab == vocab_size => {
            (*hidden, *vocab, "gguf_hidden_vocab_token_rows")
        }
        [vocab, hidden] if *hidden == hidden_width && *vocab == vocab_size => {
            (*hidden, *vocab, "output_input_token_rows")
        }
        _ => return Ok(()),
    };

    let (block_size, type_size_bytes) = tensor.tensor_type.layout().ok_or_else(|| {
        format!(
            "{role} tensor {} has unsupported storage type {:?} for token-row validation",
            tensor.name, tensor.tensor_type
        )
    })?;
    let row_values = u64::try_from(row_values)
        .map_err(|_| format!("{role} tensor {} row width overflow", tensor.name))?;
    let row_count = u64::try_from(row_count)
        .map_err(|_| format!("{role} tensor {} row count overflow", tensor.name))?;
    if row_values % block_size != 0 {
        return Err(format!(
            "{role} tensor {} token-row width {row_values} is not divisible by {:?} block size {block_size}",
            tensor.name, tensor.tensor_type
        ));
    }

    let row_size_bytes = row_values
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(type_size_bytes))
        .ok_or_else(|| format!("{role} tensor {} row size overflow", tensor.name))?;
    let expected_bytes = row_size_bytes
        .checked_mul(row_count)
        .ok_or_else(|| format!("{role} tensor {} byte size overflow", tensor.name))?;

    if tensor.n_bytes != expected_bytes {
        return Err(format!(
            "{role} tensor {} token-major storage validation failed for {layout}: row_values={row_values}, row_count={row_count}, row_size_bytes={row_size_bytes}, expected_n_bytes={expected_bytes}, actual_n_bytes={}",
            tensor.name, tensor.n_bytes
        ));
    }

    Ok(())
}

fn descriptor_dims(tensor: &GgufTensorDescriptor) -> Result<Vec<usize>, String> {
    tensor
        .dimensions
        .iter()
        .map(|dim| {
            usize::try_from(*dim)
                .map_err(|_| format!("tensor {} dimension {dim} does not fit usize", tensor.name))
        })
        .collect()
}

// Convert F16 bits to F32 (duplicate helper from q8.rs to keep model.rs self-contained)
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);

    match exponent {
        0 if fraction == 0 => f32::from_bits(sign),
        0 => {
            let mut mantissa = fraction;
            let mut exponent = -14_i32;
            while (mantissa & 0x0400) == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            mantissa &= 0x03ff;
            let f32_exponent = u32::try_from(exponent + 127).expect("subnormal exponent fits");
            f32::from_bits(sign | (f32_exponent << 23) | (mantissa << 13))
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (fraction << 13)),
        _ => {
            let f32_exponent = u32::from(exponent) + (127 - 15);
            f32::from_bits(sign | (f32_exponent << 23) | (fraction << 13))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::PathBuf};

    use crate::gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};

    use super::{
        DistributedLlamaWeights, LlamaModelConfig, validate_distributed_layer_range,
        validate_model_tensors,
    };

    #[test]
    fn infers_vocab_size_from_transposed_token_embedding() {
        let gguf = gguf_fixture(
            [
                ("llama.context_length", GgufMetadataValue::U32(2048)),
                ("llama.embedding_length", GgufMetadataValue::U32(2048)),
                ("llama.block_count", GgufMetadataValue::U32(1)),
                ("llama.feed_forward_length", GgufMetadataValue::U32(5632)),
                ("llama.attention.head_count", GgufMetadataValue::U32(32)),
                ("llama.attention.head_count_kv", GgufMetadataValue::U32(4)),
            ],
            vec![tensor_desc(
                "token_embd.weight",
                vec![32000, 2048],
                GgufTensorType::Q8_0,
                q8_bytes(2048, 32000),
            )],
        );

        let config = LlamaModelConfig::from_gguf(&gguf).expect("config should parse");
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.rope_dimension_count, 64);
    }

    #[test]
    fn parses_llama32_1b_metadata_shape() {
        let gguf = gguf_fixture(
            [
                ("llama.context_length", GgufMetadataValue::U32(131072)),
                ("llama.embedding_length", GgufMetadataValue::U32(2048)),
                ("llama.block_count", GgufMetadataValue::U32(16)),
                ("llama.feed_forward_length", GgufMetadataValue::U32(8192)),
                ("llama.attention.head_count", GgufMetadataValue::U32(32)),
                ("llama.attention.head_count_kv", GgufMetadataValue::U32(8)),
                ("llama.rope.freq_base", GgufMetadataValue::F32(500000.0)),
                ("llama.vocab_size", GgufMetadataValue::U32(128256)),
            ],
            vec![tensor_desc(
                "token_embd.weight",
                vec![128256, 2048],
                GgufTensorType::Q8_0,
                q8_bytes(2048, 128256),
            )],
        );

        let config = LlamaModelConfig::from_gguf(&gguf).expect("Llama 3.2 1B config should parse");
        assert_eq!(config.context_length, 131072);
        assert_eq!(config.embedding_length, 2048);
        assert_eq!(config.block_count, 16);
        assert_eq!(config.feed_forward_length, 8192);
        assert_eq!(config.attention_head_count, 32);
        assert_eq!(config.attention_head_count_kv, 8);
        assert_eq!(config.vocab_size, 128256);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.attention_output_width, 2048);
        assert_eq!(config.kv_width, 512);
        assert_eq!(config.rope_freq_base, 500000.0);
    }

    #[test]
    fn rejects_invalid_rope_dimension_count() {
        let gguf = gguf_fixture(
            [
                ("llama.context_length", GgufMetadataValue::U32(2048)),
                ("llama.embedding_length", GgufMetadataValue::U32(2048)),
                ("llama.block_count", GgufMetadataValue::U32(1)),
                ("llama.feed_forward_length", GgufMetadataValue::U32(5632)),
                ("llama.attention.head_count", GgufMetadataValue::U32(32)),
                ("llama.attention.head_count_kv", GgufMetadataValue::U32(4)),
                ("llama.rope.dimension_count", GgufMetadataValue::U32(65)),
            ],
            vec![tensor_desc(
                "token_embd.weight",
                vec![2048, 32000],
                GgufTensorType::Q8_0,
                q8_bytes(2048, 32000),
            )],
        );

        let err = LlamaModelConfig::from_gguf(&gguf).unwrap_err();
        assert!(err.contains("RoPE dimension count 65"));
    }

    #[test]
    fn parses_qwen2_metadata_namespace() {
        let gguf = gguf_fixture(
            [
                (
                    "general.architecture",
                    GgufMetadataValue::String("qwen2".to_owned()),
                ),
                ("qwen2.context_length", GgufMetadataValue::U32(32768)),
                ("qwen2.embedding_length", GgufMetadataValue::U32(3584)),
                ("qwen2.block_count", GgufMetadataValue::U32(28)),
                ("qwen2.feed_forward_length", GgufMetadataValue::U32(18944)),
                ("qwen2.attention.head_count", GgufMetadataValue::U32(28)),
                ("qwen2.attention.head_count_kv", GgufMetadataValue::U32(4)),
                ("qwen2.rope.freq_base", GgufMetadataValue::F32(1_000_000.0)),
            ],
            vec![tensor_desc(
                "token_embd.weight",
                vec![152064, 3584],
                GgufTensorType::Q4_0,
                q8_bytes(3584, 152064),
            )],
        );

        let config = LlamaModelConfig::from_gguf(&gguf).expect("qwen2 config should parse");
        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.metadata_prefix, "qwen2");
        assert_eq!(config.vocab_size, 152064);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.kv_width, 512);
        assert_eq!(config.rope_freq_base, 1_000_000.0);
    }

    #[test]
    fn parses_qwen3_metadata_namespace() {
        let gguf = gguf_fixture(
            [
                (
                    "general.architecture",
                    GgufMetadataValue::String("qwen3".to_owned()),
                ),
                ("qwen3.context_length", GgufMetadataValue::U32(40960)),
                ("qwen3.embedding_length", GgufMetadataValue::U32(1024)),
                ("qwen3.block_count", GgufMetadataValue::U32(28)),
                ("qwen3.feed_forward_length", GgufMetadataValue::U32(3072)),
                ("qwen3.attention.head_count", GgufMetadataValue::U32(16)),
                ("qwen3.attention.head_count_kv", GgufMetadataValue::U32(8)),
            ],
            vec![tensor_desc(
                "token_embd.weight",
                vec![151936, 1024],
                GgufTensorType::Q8_0,
                q8_bytes(1024, 151936),
            )],
        );

        let config = LlamaModelConfig::from_gguf(&gguf).expect("qwen3 config should parse");
        assert_eq!(config.architecture, "qwen3");
        assert_eq!(config.metadata_prefix, "qwen3");
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.kv_width, 512);
    }

    #[test]
    fn parses_mistral_metadata_namespace() {
        let gguf = gguf_fixture(
            [
                (
                    "general.architecture",
                    GgufMetadataValue::String("mistral".to_owned()),
                ),
                ("mistral.context_length", GgufMetadataValue::U32(32768)),
                ("mistral.embedding_length", GgufMetadataValue::U32(4096)),
                ("mistral.block_count", GgufMetadataValue::U32(32)),
                ("mistral.feed_forward_length", GgufMetadataValue::U32(14336)),
                ("mistral.attention.head_count", GgufMetadataValue::U32(32)),
                ("mistral.attention.head_count_kv", GgufMetadataValue::U32(8)),
                (
                    "mistral.rope.freq_base",
                    GgufMetadataValue::F32(1_000_000.0),
                ),
                ("mistral.vocab_size", GgufMetadataValue::U32(32000)),
            ],
            vec![tensor_desc(
                "token_embd.weight",
                vec![4096, 32000],
                GgufTensorType::Q4_0,
                q8_bytes(4096, 32000),
            )],
        );

        let config = LlamaModelConfig::from_gguf(&gguf).expect("mistral config should parse");
        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.metadata_prefix, "mistral");
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.kv_width, 1024);
        assert_eq!(config.rope_freq_base, 1_000_000.0);
    }

    #[test]
    fn validates_tinyllama_style_hidden_vocab_storage_layouts() {
        let gguf = full_tensor_fixture(q8_bytes(2048, 32000));
        validate_model_tensors(&gguf, &base_config()).expect("fixture shapes should validate");
    }

    #[test]
    fn parses_and_validates_moe_expert_tensor_layouts() {
        let gguf = moe_tensor_fixture(2, 1);
        let config = LlamaModelConfig::from_gguf(&gguf).expect("moe config should parse");

        assert_eq!(config.expert_count, 2);
        assert_eq!(config.expert_used_count, 1);
        validate_model_tensors(&gguf, &config).expect("moe tensors should validate");
    }

    #[test]
    fn rejects_moe_expert_used_count_above_expert_count() {
        let gguf = moe_tensor_fixture(2, 3);

        let err = LlamaModelConfig::from_gguf(&gguf).unwrap_err();
        assert!(err.contains("expert_used_count 3 exceeds expert_count 2"));
    }

    #[test]
    fn rejects_mismatched_output_row_storage_bytes() {
        let err = validate_model_tensors(
            &full_tensor_fixture(q8_bytes(2048, 32000) + 34),
            &base_config(),
        )
        .unwrap_err();
        assert!(err.contains("output.weight"));
        assert!(err.contains("expected_n_bytes=69632000"));
    }

    #[test]
    fn validates_distributed_layer_ranges() {
        let config = LlamaModelConfig {
            block_count: 28,
            ..base_config()
        };

        validate_distributed_layer_range(&config, 0, 14).expect("first half is valid");
        validate_distributed_layer_range(&config, 14, 28).expect("second half is valid");

        let err = validate_distributed_layer_range(&config, 4, 4).unwrap_err();
        assert!(err.contains("non-empty"));

        let err = validate_distributed_layer_range(&config, 20, 29).unwrap_err();
        assert!(err.contains("exceeds block count 28"));
    }

    #[test]
    fn distributed_weights_report_owned_layer_range() {
        let weights = DistributedLlamaWeights {
            token_embeddings: None,
            output_norm: None,
            output_projection: None,
            layer_start: 7,
            layer_end: 14,
            layers: Vec::new(),
        };

        assert!(!weights.owns_layer(6));
        assert!(weights.owns_layer(7));
        assert!(weights.owns_layer(13));
        assert!(!weights.owns_layer(14));
    }

    fn base_config() -> LlamaModelConfig {
        LlamaModelConfig {
            architecture: "llama".to_owned(),
            metadata_prefix: "llama".to_owned(),
            context_length: 2048,
            embedding_length: 2048,
            block_count: 1,
            feed_forward_length: 5632,
            attention_head_count: 32,
            attention_head_count_kv: 4,
            rope_dimension_count: 64,
            rope_freq_base: 10000.0,
            rms_norm_epsilon: 1e-5,
            vocab_size: 32000,
            head_dim: 64,
            attention_output_width: 2048,
            kv_width: 256,
            expert_count: 0,
            expert_used_count: 0,
        }
    }

    fn full_tensor_fixture(output_n_bytes: u64) -> GgufFile {
        gguf_fixture(
            [
                ("llama.context_length", GgufMetadataValue::U32(2048)),
                ("llama.embedding_length", GgufMetadataValue::U32(2048)),
                ("llama.block_count", GgufMetadataValue::U32(1)),
                ("llama.feed_forward_length", GgufMetadataValue::U32(5632)),
                ("llama.attention.head_count", GgufMetadataValue::U32(32)),
                ("llama.attention.head_count_kv", GgufMetadataValue::U32(4)),
                ("llama.vocab_size", GgufMetadataValue::U32(32000)),
            ],
            vec![
                tensor_desc(
                    "token_embd.weight",
                    vec![2048, 32000],
                    GgufTensorType::Q8_0,
                    q8_bytes(2048, 32000),
                ),
                tensor_desc(
                    "output_norm.weight",
                    vec![2048],
                    GgufTensorType::F32,
                    2048 * 4,
                ),
                tensor_desc(
                    "output.weight",
                    vec![2048, 32000],
                    GgufTensorType::Q8_0,
                    output_n_bytes,
                ),
                tensor_desc(
                    "blk.0.attn_norm.weight",
                    vec![2048],
                    GgufTensorType::F32,
                    2048 * 4,
                ),
                tensor_desc(
                    "blk.0.attn_q.weight",
                    vec![2048, 2048],
                    GgufTensorType::Q8_0,
                    q8_bytes(2048, 2048),
                ),
                tensor_desc(
                    "blk.0.attn_k.weight",
                    vec![2048, 256],
                    GgufTensorType::Q8_0,
                    q8_bytes(2048, 256),
                ),
                tensor_desc(
                    "blk.0.attn_v.weight",
                    vec![2048, 256],
                    GgufTensorType::Q8_0,
                    q8_bytes(2048, 256),
                ),
                tensor_desc(
                    "blk.0.attn_output.weight",
                    vec![2048, 2048],
                    GgufTensorType::Q8_0,
                    q8_bytes(2048, 2048),
                ),
                tensor_desc(
                    "blk.0.ffn_norm.weight",
                    vec![2048],
                    GgufTensorType::F32,
                    2048 * 4,
                ),
                tensor_desc(
                    "blk.0.ffn_gate.weight",
                    vec![2048, 5632],
                    GgufTensorType::Q8_0,
                    q8_bytes(2048, 5632),
                ),
                tensor_desc(
                    "blk.0.ffn_up.weight",
                    vec![2048, 5632],
                    GgufTensorType::Q8_0,
                    q8_bytes(2048, 5632),
                ),
                tensor_desc(
                    "blk.0.ffn_down.weight",
                    vec![5632, 2048],
                    GgufTensorType::Q8_0,
                    q8_bytes(5632, 2048),
                ),
            ],
        )
    }

    fn moe_tensor_fixture(expert_count: usize, expert_used_count: usize) -> GgufFile {
        let mut fixture = full_tensor_fixture(q8_bytes(2048, 32000));
        fixture.metadata.insert(
            "llama.expert_count".to_owned(),
            GgufMetadataValue::U32(expert_count as u32),
        );
        fixture.metadata.insert(
            "llama.expert_used_count".to_owned(),
            GgufMetadataValue::U32(expert_used_count as u32),
        );
        fixture.metadata_count = fixture.metadata.len() as u64;

        fixture.tensors.push(tensor_desc(
            "blk.0.ffn_gate_inp.weight",
            vec![2048, expert_count as u64],
            GgufTensorType::F32,
            2048 * expert_count as u64 * 4,
        ));
        for expert_idx in 0..expert_count {
            fixture.tensors.extend([
                tensor_desc(
                    &format!("blk.0.ffn_gate.{expert_idx}.weight"),
                    vec![2048, 5632],
                    GgufTensorType::Q8_0,
                    q8_bytes(2048, 5632),
                ),
                tensor_desc(
                    &format!("blk.0.ffn_up.{expert_idx}.weight"),
                    vec![2048, 5632],
                    GgufTensorType::Q8_0,
                    q8_bytes(2048, 5632),
                ),
                tensor_desc(
                    &format!("blk.0.ffn_down.{expert_idx}.weight"),
                    vec![5632, 2048],
                    GgufTensorType::Q8_0,
                    q8_bytes(5632, 2048),
                ),
            ]);
        }
        fixture.tensor_count = fixture.tensors.len() as u64;

        fixture
    }

    fn gguf_fixture<const N: usize>(
        overrides: [(&str, GgufMetadataValue); N],
        tensors: Vec<GgufTensorDescriptor>,
    ) -> GgufFile {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            GgufMetadataValue::String("llama".to_owned()),
        );
        for (key, value) in overrides {
            metadata.insert(key.to_owned(), value);
        }

        GgufFile {
            path: PathBuf::from("model-fixture.gguf"),
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
}
