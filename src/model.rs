use std::{env, fs::File, path::Path, sync::OnceLock};

use crate::gguf::{GgufFile, GgufTensorDescriptor, GgufTensorType};
use crate::q8::{
    Q4_0Block, Q6KBlock, Q8_0Block, QK_K_BLOCK_SIZE, decode_q4_0_blocks, decode_q6_k_blocks,
    decode_q8_0_blocks, swizzle_q4_0_1x4,
};
use memmap2::Mmap;

pub const Q4_SWIZZLE_1X4_ENV: &str = "NANOCAMELID_Q4_SWIZZLE_1X4";

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
    pub kv_width: usize,
}

impl LlamaModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| "missing general.architecture".to_owned())?;
        let metadata_prefix = match arch {
            "llama" => "llama",
            "qwen2" => "qwen2",
            _ => {
                return Err(format!("unsupported architecture: {arch}"));
            }
        };

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

        let head_dim = embedding_length / attention_head_count;
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
        let kv_width = attention_head_count_kv * head_dim;

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
            kv_width,
        })
    }
}

fn metadata_key(prefix: &str, suffix: &str) -> String {
    format!("{prefix}.{suffix}")
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

pub enum QuantizedMatrix {
    Q8_0(Vec<Q8_0Block>),
    Q4_0(Vec<Q4_0Block>),
    Q4_0Swizzled1x4(Q4_0Swizzled1x4Matrix),
    Q6K(Vec<Q6KBlock>),
}

pub struct Q4_0Swizzled1x4Matrix {
    pub swizzled_1x4: Vec<Q4_0Block>,
    pub rows: usize,
    pub cols: usize,
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
            let attention_norm =
                load_f32_or_f16(&mmap, gguf, &format!("blk.{i}.attn_norm.weight"))?;
            let wq = load_quantized_matrix(&mmap, gguf, &format!("blk.{i}.attn_q.weight"))?;
            let wk = load_quantized_matrix(&mmap, gguf, &format!("blk.{i}.attn_k.weight"))?;
            let wav = load_quantized_matrix(&mmap, gguf, &format!("blk.{i}.attn_v.weight"))?;
            let wq_bias = load_optional_f32_or_f16(&mmap, gguf, &format!("blk.{i}.attn_q.bias"))?;
            let wk_bias = load_optional_f32_or_f16(&mmap, gguf, &format!("blk.{i}.attn_k.bias"))?;
            let wav_bias = load_optional_f32_or_f16(&mmap, gguf, &format!("blk.{i}.attn_v.bias"))?;
            let wo = load_quantized_matrix(&mmap, gguf, &format!("blk.{i}.attn_output.weight"))?;

            let ffn_norm = load_f32_or_f16(&mmap, gguf, &format!("blk.{i}.ffn_norm.weight"))?;
            let w1 = load_quantized_matrix(&mmap, gguf, &format!("blk.{i}.ffn_gate.weight"))?;
            let w3 = load_quantized_matrix(&mmap, gguf, &format!("blk.{i}.ffn_up.weight"))?;
            let w2 = load_quantized_matrix(&mmap, gguf, &format!("blk.{i}.ffn_down.weight"))?;

            layers.push(LlamaLayerWeights {
                attention_norm,
                wq,
                wk,
                wav,
                wq_bias,
                wk_bias,
                wav_bias,
                wo,
                ffn_norm,
                w1,
                w3,
                w2,
            });
        }

        Ok(Self {
            token_embeddings,
            output_norm,
            output_projection,
            layers,
        })
    }
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
            config.embedding_length,
            &format!("layer {layer_idx} attention q"),
        )?;
        if let Ok(bias) = load_tensor_desc(gguf, &format!("blk.{layer_idx}.attn_q.bias")) {
            require_descriptor_shape(
                bias,
                &[config.embedding_length],
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
            config.embedding_length,
            config.embedding_length,
            &format!("layer {layer_idx} attention output"),
        )?;
        require_descriptor_shape(
            load_tensor_desc(gguf, &format!("blk.{layer_idx}.ffn_norm.weight"))?,
            &[config.embedding_length],
            &format!("layer {layer_idx} ffn norm"),
        )?;
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

    Ok(())
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
                let scale = block.scale_f32();
                for val in block.unpack_values() {
                    data.push(val as f32 * scale);
                }
            }
            Ok(data)
        }
        GgufTensorType::Q6K => {
            let blocks = decode_q6_k_blocks(&bytes).map_err(|e| e.to_string())?;
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
        GgufTensorType::Q8_0 => decode_q8_0_blocks(&bytes)
            .map(QuantizedMatrix::Q8_0)
            .map_err(|e| e.to_string()),
        GgufTensorType::Q4_0 => load_q4_0_matrix(&bytes, desc),
        GgufTensorType::Q6K => decode_q6_k_blocks(&bytes)
            .map(QuantizedMatrix::Q6K)
            .map_err(|e| e.to_string()),
        other => Err(format!(
            "expected Q8_0, Q4_0, or Q6_K tensor type for {name}, got {other:?}"
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

    let swizzled_1x4 = swizzle_q4_0_1x4(&row_major, rows, cols / 32);
    Ok(QuantizedMatrix::Q4_0Swizzled1x4(Q4_0Swizzled1x4Matrix {
        swizzled_1x4,
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

    use super::{LlamaModelConfig, validate_model_tensors};

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
    fn validates_tinyllama_style_hidden_vocab_storage_layouts() {
        let gguf = full_tensor_fixture(q8_bytes(2048, 32000));
        validate_model_tensors(&gguf, &base_config()).expect("fixture shapes should validate");
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
            kv_width: 256,
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
