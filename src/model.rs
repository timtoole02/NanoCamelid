use std::{
    fs::File,
    os::unix::fs::FileExt,
    path::Path,
};

use crate::gguf::{GgufFile, GgufTensorDescriptor, GgufTensorType};
use crate::q8::{Q8_0Block, decode_q8_0_blocks};
use crate::q4::{Q4_0Block, decode_q4_0_blocks};

#[derive(Clone, Debug)]
pub struct LlamaModelConfig {
    pub context_length: usize,
    pub embedding_length: usize,
    pub block_count: usize,
    pub feed_forward_length: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub rope_freq_base: f32,
    pub rms_norm_epsilon: f32,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub kv_width: usize,
}

impl LlamaModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf.metadata_string("general.architecture")
            .ok_or_else(|| "missing general.architecture".to_owned())?;
        if arch != "llama" {
            return Err(format!("unsupported architecture: {arch}"));
        }

        let context_length = gguf.metadata_u32("llama.context_length")
            .ok_or_else(|| "missing llama.context_length".to_owned())? as usize;
        let embedding_length = gguf.metadata_u32("llama.embedding_length")
            .ok_or_else(|| "missing llama.embedding_length".to_owned())? as usize;
        let block_count = gguf.metadata_u32("llama.block_count")
            .ok_or_else(|| "missing llama.block_count".to_owned())? as usize;
        let feed_forward_length = gguf.metadata_u32("llama.feed_forward_length")
            .ok_or_else(|| "missing llama.feed_forward_length".to_owned())? as usize;
        let attention_head_count = gguf.metadata_u32("llama.attention.head_count")
            .ok_or_else(|| "missing llama.attention.head_count".to_owned())? as usize;
        let attention_head_count_kv = gguf.metadata_u32("llama.attention.head_count_kv")
            .unwrap_or(attention_head_count as u32) as usize;

        let rope_freq_base = gguf.metadata_f32("llama.rope.freq_base").unwrap_or(10000.0);
        let rms_norm_epsilon = gguf.metadata_f32("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);

        // Find token embedding weight to infer vocab size if not explicitly given
        let token_emb_desc = gguf.tensors.iter()
            .find(|t| t.name == "token_embd.weight")
            .ok_or_else(|| "missing token_embd.weight tensor".to_owned())?;
        
        let vocab_size = if let Some(v) = gguf.metadata_u32("llama.vocab_size") {
            v as usize
        } else {
            // dimensions[1] of token_embd.weight is usually the vocab size
            if token_emb_desc.dimensions.len() >= 2 {
                token_emb_desc.dimensions[1] as usize
            } else {
                return Err("cannot infer vocab size from token_embd.weight dimensions".to_owned());
            }
        };

        let head_dim = embedding_length / attention_head_count;
        let kv_width = attention_head_count_kv * head_dim;

        Ok(Self {
            context_length,
            embedding_length,
            block_count,
            feed_forward_length,
            attention_head_count,
            attention_head_count_kv,
            rope_freq_base,
            rms_norm_epsilon,
            vocab_size,
            head_dim,
            kv_width,
        })
    }
}

pub struct LlamaLayerWeightsQ8 {
    pub attention_norm: Vec<f32>,
    pub wq: Vec<Q8_0Block>,
    pub wk: Vec<Q8_0Block>,
    pub wav: Vec<Q8_0Block>,
    pub wo: Vec<Q8_0Block>,
    pub ffn_norm: Vec<f32>,
    pub w1: Vec<Q8_0Block>,
    pub w3: Vec<Q8_0Block>,
    pub w2: Vec<Q8_0Block>,
}

pub struct LlamaLayerWeightsQ4 {
    pub attention_norm: Vec<f32>,
    pub wq: Vec<Q4_0Block>,
    pub wk: Vec<Q4_0Block>,
    pub wav: Vec<Q4_0Block>,
    pub wo: Vec<Q4_0Block>,
    pub ffn_norm: Vec<f32>,
    pub w1: Vec<Q4_0Block>,
    pub w3: Vec<Q4_0Block>,
    pub w2: Vec<Q4_0Block>,
}

pub struct LlamaWeightsQ8 {
    pub token_embeddings: Vec<f32>,
    pub output_norm: Vec<f32>,
    pub output_projection: Option<Vec<Q8_0Block>>,
    pub layers: Vec<LlamaLayerWeightsQ8>,
}

pub struct LlamaWeightsQ4 {
    pub token_embeddings: Vec<f32>,
    pub output_norm: Vec<f32>,
    pub output_projection: Option<Vec<Q4_0Block>>,
    pub layers: Vec<LlamaLayerWeightsQ4>,
}

pub enum LlamaWeights {
    Q8_0(LlamaWeightsQ8),
    Q4_0(LlamaWeightsQ4),
}

impl LlamaWeights {
    /// Load the whole model onto a single node.
    pub fn load(path: &Path, config: &LlamaModelConfig, gguf: &GgufFile) -> Result<Self, String> {
        Self::load_range(path, config, gguf, 0, config.block_count, true, true)
    }

    /// Load only the transformer layers in `[start_layer, end_layer)` plus the optional
    /// embedding / output tensors this pipeline node needs.
    ///
    /// - `need_embeddings`: load `token_embd.weight` — node0 needs it for the embedding lookup.
    /// - `need_output`: load `output_norm.weight` (+ `output.weight` if present) — the tail
    ///   node needs them for the final RMSNorm and logits projection.
    ///
    /// Tied-embeddings models (no `output.weight`) compute logits from `token_embeddings` in
    /// `finalize`, so when `need_output` is set on such a model the embedding table is force
    /// loaded as well. The returned `layers` Vec is **locally indexed** `0..(end-start)` to
    /// match `forward_layers`. Tensors are read independently by file offset, so a shard only
    /// materializes the bytes it asks for — this is what lets a model exceed one node's RAM.
    pub fn load_range(
        path: &Path,
        config: &LlamaModelConfig,
        gguf: &GgufFile,
        start_layer: usize,
        end_layer: usize,
        need_embeddings: bool,
        need_output: bool,
    ) -> Result<Self, String> {
        let _ = config;
        let file = File::open(path).map_err(|e| e.to_string())?;
        let has_output_weight = gguf.tensors.iter().any(|t| t.name == "output.weight");
        // Tied embeddings: the tail node's finalize() multiplies by token_embeddings.
        let load_embeddings = need_embeddings || (need_output && !has_output_weight);

        // Inspect format from the first layer query tensor type
        let sample_desc = load_tensor_desc(gguf, "blk.0.attn_q.weight")?;
        match sample_desc.tensor_type {
            GgufTensorType::Q8_0 => {
                let token_embeddings = if load_embeddings {
                    load_f32_or_f16(&file, gguf, "token_embd.weight")?
                } else {
                    Vec::new()
                };
                let output_norm = if need_output {
                    load_f32_or_f16(&file, gguf, "output_norm.weight")?
                } else {
                    Vec::new()
                };
                let output_projection = if need_output && has_output_weight {
                    Some(load_q8_0(&file, gguf, "output.weight")?)
                } else {
                    None
                };

                let mut layers = Vec::with_capacity(end_layer - start_layer);
                for i in start_layer..end_layer {
                    let attention_norm = load_f32_or_f16(&file, gguf, &format!("blk.{i}.attn_norm.weight"))?;
                    let wq = load_q8_0(&file, gguf, &format!("blk.{i}.attn_q.weight"))?;
                    let wk = load_q8_0(&file, gguf, &format!("blk.{i}.attn_k.weight"))?;
                    let wav = load_q8_0(&file, gguf, &format!("blk.{i}.attn_v.weight"))?;
                    let wo = load_q8_0(&file, gguf, &format!("blk.{i}.attn_output.weight"))?;
                    let ffn_norm = load_f32_or_f16(&file, gguf, &format!("blk.{i}.ffn_norm.weight"))?;
                    let w1 = load_q8_0(&file, gguf, &format!("blk.{i}.ffn_gate.weight"))?;
                    let w3 = load_q8_0(&file, gguf, &format!("blk.{i}.ffn_up.weight"))?;
                    let w2 = load_q8_0(&file, gguf, &format!("blk.{i}.ffn_down.weight"))?;

                    layers.push(LlamaLayerWeightsQ8 {
                        attention_norm,
                        wq,
                        wk,
                        wav,
                        wo,
                        ffn_norm,
                        w1,
                        w3,
                        w2,
                    });
                }

                Ok(Self::Q8_0(LlamaWeightsQ8 {
                    token_embeddings,
                    output_norm,
                    output_projection,
                    layers,
                }))
            }
            GgufTensorType::Q4_0 => {
                let token_embeddings = if load_embeddings {
                    load_f32_or_f16(&file, gguf, "token_embd.weight")?
                } else {
                    Vec::new()
                };
                let output_norm = if need_output {
                    load_f32_or_f16(&file, gguf, "output_norm.weight")?
                } else {
                    Vec::new()
                };
                let output_projection = if need_output && has_output_weight {
                    Some(load_q4_0(&file, gguf, "output.weight")?)
                } else {
                    None
                };

                let mut layers = Vec::with_capacity(end_layer - start_layer);
                for i in start_layer..end_layer {
                    let attention_norm = load_f32_or_f16(&file, gguf, &format!("blk.{i}.attn_norm.weight"))?;
                    let wq = load_q4_0(&file, gguf, &format!("blk.{i}.attn_q.weight"))?;
                    let wk = load_q4_0(&file, gguf, &format!("blk.{i}.attn_k.weight"))?;
                    let wav = load_q4_0(&file, gguf, &format!("blk.{i}.attn_v.weight"))?;
                    let wo = load_q4_0(&file, gguf, &format!("blk.{i}.attn_output.weight"))?;
                    let ffn_norm = load_f32_or_f16(&file, gguf, &format!("blk.{i}.ffn_norm.weight"))?;
                    let w1 = load_q4_0(&file, gguf, &format!("blk.{i}.ffn_gate.weight"))?;
                    let w3 = load_q4_0(&file, gguf, &format!("blk.{i}.ffn_up.weight"))?;
                    let w2 = load_q4_0(&file, gguf, &format!("blk.{i}.ffn_down.weight"))?;

                    layers.push(LlamaLayerWeightsQ4 {
                        attention_norm,
                        wq,
                        wk,
                        wav,
                        wo,
                        ffn_norm,
                        w1,
                        w3,
                        w2,
                    });
                }

                Ok(Self::Q4_0(LlamaWeightsQ4 {
                    token_embeddings,
                    output_norm,
                    output_projection,
                    layers,
                }))
            }
            other => Err(format!("Unsupported model quantization format: {:?}", other)),
        }
    }
}

fn load_tensor_desc<'a>(gguf: &'a GgufFile, name: &str) -> Result<&'a GgufTensorDescriptor, String> {
    gguf.tensors.iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("tensor {name} not found in GGUF"))
}

fn load_f32_or_f16(file: &File, gguf: &GgufFile, name: &str) -> Result<Vec<f32>, String> {
    let desc = load_tensor_desc(gguf, name)?;
    let mut bytes = vec![0; desc.n_bytes as usize];
    file.read_exact_at(&mut bytes, desc.absolute_offset).map_err(|e| e.to_string())?;

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
            let blocks = decode_q8_0_blocks(&bytes).map_err(|e| e.to_string())?;
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
            let blocks = decode_q4_0_blocks(&bytes).map_err(|e| e.to_string())?;
            let mut data = Vec::with_capacity(blocks.len() * 32);
            for block in blocks {
                let mut block_data = [0.0f32; 32];
                block.dequantize(&mut block_data);
                data.extend_from_slice(&block_data);
            }
            Ok(data)
        }
        other => Err(format!("unsupported floating point type for {name}: {other:?}")),
    }
}

fn load_q8_0(file: &File, gguf: &GgufFile, name: &str) -> Result<Vec<Q8_0Block>, String> {
    let desc = load_tensor_desc(gguf, name)?;
    if desc.tensor_type != GgufTensorType::Q8_0 {
        return Err(format!("expected Q8_0 tensor type for {name}, got {:?}", desc.tensor_type));
    }
    let mut bytes = vec![0; desc.n_bytes as usize];
    file.read_exact_at(&mut bytes, desc.absolute_offset).map_err(|e| e.to_string())?;

    decode_q8_0_blocks(&bytes).map_err(|e| e.to_string())
}

fn load_q4_0(file: &File, gguf: &GgufFile, name: &str) -> Result<Vec<Q4_0Block>, String> {
    let desc = load_tensor_desc(gguf, name)?;
    if desc.tensor_type != GgufTensorType::Q4_0 {
        return Err(format!("expected Q4_0 tensor type for {name}, got {:?}", desc.tensor_type));
    }
    let mut bytes = vec![0; desc.n_bytes as usize];
    file.read_exact_at(&mut bytes, desc.absolute_offset).map_err(|e| e.to_string())?;

    decode_q4_0_blocks(&bytes).map_err(|e| e.to_string())
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
