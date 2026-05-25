use crate::model::{LlamaModelConfig, LlamaWeights, PageAlignedQ4_0Swizzled1x4, QuantizedMatrix};
use crate::q8::{
    Q4_0Block, Q6KBlock, Q8_0Block, Q8_BLOCK_SIZE, Q8DotKernel, Q8DotKernelSelector,
    QK_K_BLOCK_SIZE,
};
use rayon::prelude::*;
use std::{env, sync::OnceLock};

pub const MATMUL_MIN_ROWS_ENV: &str = "NANOCAMELID_MATMUL_MIN_ROWS";
pub const Q4_1X4_SDOT_ENV: &str = "NANOCAMELID_Q4_1X4_SDOT";
pub const Q6K_SDOT_ENV: &str = "NANOCAMELID_Q6K_SDOT";
pub const ATTENTION_HEAD_PARALLEL_ENV: &str = "NANOCAMELID_ATTENTION_HEAD_PARALLEL";
pub const KV_CACHE_F16_ENV: &str = "NANOCAMELID_KV_CACHE_F16";
const DEFAULT_MATMUL_MIN_ROWS: usize = 128;

fn matmul_min_rows() -> usize {
    static MIN_ROWS: OnceLock<usize> = OnceLock::new();
    *MIN_ROWS.get_or_init(|| {
        env::var(MATMUL_MIN_ROWS_ENV)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_MATMUL_MIN_ROWS)
    })
}

fn should_parallelize_matmul(rows: usize) -> bool {
    rows >= matmul_min_rows()
}

fn q4_1x4_sdot_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(Q4_1X4_SDOT_ENV)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
            .unwrap_or(true)
    })
}

fn q6_k_sdot_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(Q6K_SDOT_ENV)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
            .unwrap_or(true)
    })
}

fn attention_head_parallel_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(ATTENTION_HEAD_PARALLEL_ENV)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
            .unwrap_or(true)
    })
}

fn kv_cache_f16_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(KV_CACHE_F16_ENV)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
            .unwrap_or(false)
    })
}

fn for_each_matmul_row<F>(out: &mut [f32], compute_row: F)
where
    F: Fn(usize, &mut f32) + Sync,
{
    if should_parallelize_matmul(out.len()) {
        out.par_iter_mut()
            .with_min_len(matmul_min_rows())
            .enumerate()
            .for_each(|(r, out_val)| compute_row(r, out_val));
    } else {
        out.iter_mut()
            .enumerate()
            .for_each(|(r, out_val)| compute_row(r, out_val));
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RopeScaling {
    pub factor: Option<f32>,
    pub original_context_length: Option<f32>,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct LlamaRuntimeOptions {
    pub q8_selector: Q8DotKernelSelector,
    pub rope_scaling: RopeScaling,
    pub compute_logits: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct BatchMatmulShape {
    pub batch_size: usize,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy)]
pub enum KvCacheSlice<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
}

enum LlamaKvCacheStorage {
    F32 {
        k_cache: Vec<f32>,
        v_cache: Vec<f32>,
    },
    F16 {
        k_cache: Vec<u16>,
        v_cache: Vec<u16>,
    },
}

pub struct LlamaKvCache {
    storage: LlamaKvCacheStorage,
    pub kv_width: usize,
    pub max_seq_len: usize,
}

impl LlamaKvCache {
    pub fn new(block_count: usize, max_seq_len: usize, kv_width: usize) -> Self {
        let size = block_count * max_seq_len * kv_width;
        let storage = if kv_cache_f16_enabled() {
            LlamaKvCacheStorage::F16 {
                k_cache: vec![0; size],
                v_cache: vec![0; size],
            }
        } else {
            LlamaKvCacheStorage::F32 {
                k_cache: vec![0.0; size],
                v_cache: vec![0.0; size],
            }
        };
        Self {
            storage,
            kv_width,
            max_seq_len,
        }
    }

    pub fn store_kv(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let layer_offset = layer * self.max_seq_len * self.kv_width;
        let pos_offset = pos * self.kv_width;
        let start = layer_offset + pos_offset;

        match &mut self.storage {
            LlamaKvCacheStorage::F32 { k_cache, v_cache } => {
                k_cache[start..start + self.kv_width].copy_from_slice(k);
                v_cache[start..start + self.kv_width].copy_from_slice(v);
            }
            LlamaKvCacheStorage::F16 { k_cache, v_cache } => {
                for idx in 0..self.kv_width {
                    k_cache[start + idx] = crate::q8::fast_f32_to_f16(k[idx]);
                    v_cache[start + idx] = crate::q8::fast_f32_to_f16(v[idx]);
                }
            }
        }
    }

    pub fn get_k_cache(&self, layer: usize) -> KvCacheSlice<'_> {
        let layer_offset = layer * self.max_seq_len * self.kv_width;
        let range = layer_offset..layer_offset + self.max_seq_len * self.kv_width;
        match &self.storage {
            LlamaKvCacheStorage::F32 { k_cache, .. } => KvCacheSlice::F32(&k_cache[range]),
            LlamaKvCacheStorage::F16 { k_cache, .. } => KvCacheSlice::F16(&k_cache[range]),
        }
    }

    pub fn get_v_cache(&self, layer: usize) -> KvCacheSlice<'_> {
        let layer_offset = layer * self.max_seq_len * self.kv_width;
        let range = layer_offset..layer_offset + self.max_seq_len * self.kv_width;
        match &self.storage {
            LlamaKvCacheStorage::F32 { v_cache, .. } => KvCacheSlice::F32(&v_cache[range]),
            LlamaKvCacheStorage::F16 { v_cache, .. } => KvCacheSlice::F16(&v_cache[range]),
        }
    }
}

pub struct LlamaWorkspace {
    pub hidden: Vec<f32>,
    pub residual: Vec<f32>,
    pub norm_x: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_output: Vec<f32>,
    pub attn_scores: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_gate_up: Vec<f32>,
    pub x_i8: Vec<i8>,
    pub x_scales: Vec<f32>,
    pub logits: Vec<f32>,
}

pub struct LlamaBatchWorkspace {
    pub max_batch: usize,
    pub hidden: Vec<f32>,
    pub residual: Vec<f32>,
    pub norm_x: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_output: Vec<f32>,
    pub attn_scores: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_gate_up: Vec<f32>,
    pub x_i8: Vec<i8>,
    pub x_scales: Vec<f32>,
}

impl LlamaWorkspace {
    pub fn new(config: &LlamaModelConfig) -> Self {
        let max_size = config
            .vocab_size
            .max(config.feed_forward_length)
            .max(config.embedding_length);
        let attn_score_len = if attention_head_parallel_enabled() {
            config.attention_head_count * config.context_length
        } else {
            config.context_length
        };
        Self {
            hidden: vec![0.0; config.embedding_length],
            residual: vec![0.0; config.embedding_length],
            norm_x: vec![0.0; config.embedding_length],
            q: vec![0.0; config.embedding_length],
            k: vec![0.0; config.kv_width],
            v: vec![0.0; config.kv_width],
            attn_output: vec![0.0; config.embedding_length],
            attn_scores: vec![0.0; attn_score_len],
            ffn_gate: vec![0.0; config.feed_forward_length],
            ffn_up: vec![0.0; config.feed_forward_length],
            ffn_gate_up: vec![0.0; config.feed_forward_length],
            x_i8: vec![0; max_size],
            x_scales: vec![0.0; max_size / 32 + 1],
            logits: vec![0.0; config.vocab_size],
        }
    }
}

impl LlamaBatchWorkspace {
    pub fn new(config: &LlamaModelConfig, max_batch: usize) -> Self {
        let max_size = config
            .vocab_size
            .max(config.feed_forward_length)
            .max(config.embedding_length);
        let attn_score_len = if attention_head_parallel_enabled() {
            max_batch * config.attention_head_count * config.context_length
        } else {
            max_batch * config.context_length
        };
        Self {
            max_batch,
            hidden: vec![0.0; max_batch * config.embedding_length],
            residual: vec![0.0; max_batch * config.embedding_length],
            norm_x: vec![0.0; max_batch * config.embedding_length],
            q: vec![0.0; max_batch * config.embedding_length],
            k: vec![0.0; max_batch * config.kv_width],
            v: vec![0.0; max_batch * config.kv_width],
            attn_output: vec![0.0; max_batch * config.embedding_length],
            attn_scores: vec![0.0; attn_score_len],
            ffn_gate: vec![0.0; max_batch * config.feed_forward_length],
            ffn_up: vec![0.0; max_batch * config.feed_forward_length],
            ffn_gate_up: vec![0.0; max_batch * config.feed_forward_length],
            x_i8: vec![0; max_batch * max_size],
            x_scales: vec![0.0; max_batch * (max_size / Q8_BLOCK_SIZE + 1)],
        }
    }
}

pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], epsilon: f32) {
    let mut sum = 0.0_f32;
    for &val in x {
        sum += val * val;
    }
    let scale = 1.0_f32 / (sum / x.len() as f32 + epsilon).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

pub fn apply_rope(
    data: &mut [f32],
    pos: usize,
    head_count: usize,
    head_dim: usize,
    rope_dim: usize,
    freq_base: f32,
    rope_scaling: RopeScaling,
) {
    for head in 0..head_count {
        let head_start = head * head_dim;
        for i in 0..(rope_dim / 2) {
            let dim0 = head_start + (i * 2);
            let dim1 = dim0 + 1;

            let mut theta = freq_base.powf(-((i * 2) as f32) / rope_dim as f32);

            // Apply Llama 3 / 3.2 scaled RoPE if config is present
            if let (Some(factor), Some(orig_len), Some(low), Some(high)) = (
                rope_scaling.factor,
                rope_scaling.original_context_length,
                rope_scaling.low_freq_factor,
                rope_scaling.high_freq_factor,
            ) {
                let wavelength = (2.0 * std::f32::consts::PI) / theta;
                let low_freq_wavelength = orig_len / low;
                let high_freq_wavelength = orig_len / high;
                if wavelength >= high_freq_wavelength {
                    if wavelength > low_freq_wavelength {
                        theta /= factor;
                    } else {
                        let smooth = (orig_len / wavelength - low) / (high - low);
                        theta = ((1.0 - smooth) * theta / factor) + (smooth * theta);
                    }
                }
            }

            let angle = pos as f32 * theta;
            let (sin, cos) = angle.sin_cos();

            let x0 = data[dim0];
            let x1 = data[dim1];

            data[dim0] = x0 * cos - x1 * sin;
            data[dim1] = x0 * sin + x1 * cos;
        }
    }
}

pub fn quantize_f32_to_q8_0(x: &[f32], x_i8: &mut [i8], x_scales: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 guarantees NEON/ASIMD support.
        unsafe {
            quantize_f32_to_q8_0_neon_max(x, x_i8, x_scales);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        quantize_f32_to_q8_0_scalar(x, x_i8, x_scales);
    }
}

pub fn rms_norm_batch(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    epsilon: f32,
    batch_size: usize,
    dim: usize,
) {
    debug_assert_eq!(out.len(), batch_size * dim);
    debug_assert_eq!(x.len(), batch_size * dim);
    debug_assert_eq!(weight.len(), dim);

    out.par_chunks_mut(dim)
        .zip(x.par_chunks(dim))
        .take(batch_size)
        .for_each(|(out_token, x_token)| {
            rms_norm(out_token, x_token, weight, epsilon);
        });
}

pub fn quantize_f32_to_q8_0_batch(
    x: &[f32],
    x_i8: &mut [i8],
    x_scales: &mut [f32],
    batch_size: usize,
    dim: usize,
) {
    let blocks_per_token = dim / 32;
    debug_assert_eq!(x.len(), batch_size * dim);
    debug_assert_eq!(x_i8.len(), batch_size * dim);
    debug_assert!(x_scales.len() >= batch_size * blocks_per_token);

    x_i8.par_chunks_mut(dim)
        .zip(x_scales.par_chunks_mut(blocks_per_token))
        .zip(x.par_chunks(dim))
        .take(batch_size)
        .for_each(|((out_i8, out_scales), x_token)| {
            quantize_f32_to_q8_0(x_token, out_i8, out_scales);
        });
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn quantize_f32_to_q8_0_scalar(x: &[f32], x_i8: &mut [i8], x_scales: &mut [f32]) {
    let num_blocks = x.len() / 32;
    for (b, scale_out) in x_scales.iter_mut().enumerate().take(num_blocks) {
        let chunk = &x[b * 32..(b + 1) * 32];
        let max_abs = max_abs_scalar(chunk);
        quantize_q8_block_with_max_abs(chunk, &mut x_i8[b * 32..(b + 1) * 32], scale_out, max_abs);
    }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn max_abs_scalar(chunk: &[f32]) -> f32 {
    let mut max_abs = 0.0_f32;
    for &val in chunk {
        let abs = val.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    max_abs
}

fn quantize_q8_block_with_max_abs(
    chunk: &[f32],
    out: &mut [i8],
    scale_out: &mut f32,
    max_abs: f32,
) {
    let scale = max_abs / 127.0;
    *scale_out = scale;
    if scale > 0.0 {
        let inv_scale = 1.0 / scale;
        for (dst, &value) in out.iter_mut().zip(chunk) {
            let q = (value * inv_scale).round();
            *dst = q.clamp(-127.0, 127.0) as i8;
        }
    } else {
        out.fill(0);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn quantize_f32_to_q8_0_neon_max(x: &[f32], x_i8: &mut [i8], x_scales: &mut [f32]) {
    let num_blocks = x.len() / 32;
    for (b, scale_out) in x_scales.iter_mut().enumerate().take(num_blocks) {
        let offset = b * 32;
        let chunk = &x[offset..offset + 32];
        let max_abs = unsafe { max_abs_32_neon(chunk.as_ptr()) };
        quantize_q8_block_with_max_abs(chunk, &mut x_i8[offset..offset + 32], scale_out, max_abs);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn max_abs_32_neon(ptr: *const f32) -> f32 {
    use std::arch::aarch64::{vabsq_f32, vld1q_f32, vmaxq_f32, vmaxvq_f32};

    let mut max_abs = unsafe { vabsq_f32(vld1q_f32(ptr)) };
    for offset in [4, 8, 12, 16, 20, 24, 28] {
        let values = unsafe { vabsq_f32(vld1q_f32(ptr.add(offset))) };
        max_abs = unsafe { vmaxq_f32(max_abs, values) };
    }
    unsafe { vmaxvq_f32(max_abs) }
}

unsafe fn activation_block_ptr(x_i8: &[i8], block_idx: usize) -> &[i8; 32] {
    debug_assert!((block_idx + 1) * 32 <= x_i8.len());
    // SAFETY: callers use model-derived dimensions where each activation block is
    // exactly 32 i8 lanes and block_idx is bounded by blocks_per_row.
    unsafe { &*(x_i8.as_ptr().add(block_idx * 32) as *const [i8; 32]) }
}

pub fn matmul_q8_0(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q8_0Block],
    _rows: usize,
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    let blocks_per_row = cols / 32;
    match selector.selected {
        Q8DotKernel::Scalar => {
            for_each_matmul_row(out, |r, out_val| {
                let mut sum = 0.0_f32;
                let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
                for b in 0..blocks_per_row {
                    let w_block = &w_row[b];
                    let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
                    let dot_val = crate::q8::dot_i8_scalar(w_block.values(), x_block_vals);
                    sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
                }
                *out_val = sum;
            });
        }
        Q8DotKernel::Neon => {
            for_each_matmul_row(out, |r, out_val| {
                let mut sum = 0.0_f32;
                let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
                for b in 0..blocks_per_row {
                    let w_block = &w_row[b];
                    let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
                    let dot_val =
                        crate::q8::dot_i8_neon_32_selected(w_block.values(), x_block_vals);
                    sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
                }
                *out_val = sum;
            });
        }
        Q8DotKernel::Sdot => {
            for_each_matmul_row(out, |r, out_val| {
                let mut sum = 0.0_f32;
                let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
                for b in 0..blocks_per_row {
                    let w_block = &w_row[b];
                    let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
                    let dot_val =
                        crate::q8::dot_i8_sdot_32_selected(w_block.values(), x_block_vals);
                    sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
                }
                *out_val = sum;
            });
        }
    }
}

pub fn matmul_quantized(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &QuantizedMatrix,
    rows: usize,
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    match w {
        QuantizedMatrix::Q8_0(blocks) => {
            matmul_q8_0(out, x_i8, x_scales, blocks, rows, cols, selector)
        }
        QuantizedMatrix::Q4_0(blocks) => {
            matmul_q4_0(out, x_i8, x_scales, blocks, rows, cols, selector)
        }
        QuantizedMatrix::Q4_0Swizzled1x4(matrix) => {
            debug_assert_eq!(matrix.rows, rows);
            debug_assert_eq!(matrix.cols, cols);
            matmul_q4_0_swizzled_1x4(out, x_i8, x_scales, matrix, cols, selector);
        }
        QuantizedMatrix::Q6K(blocks) => matmul_q6_k(out, x_i8, x_scales, blocks, rows, cols),
    }
}

pub fn matmul_quantized_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &QuantizedMatrix,
    shape: BatchMatmulShape,
    selector: Q8DotKernelSelector,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    debug_assert_eq!(out.len(), batch_size * rows);
    debug_assert_eq!(x_i8.len(), batch_size * cols);
    debug_assert!(x_scales.len() >= batch_size * (cols / Q8_BLOCK_SIZE));

    match w {
        QuantizedMatrix::Q8_0(blocks) => {
            matmul_q8_0_batch(out, x_i8, x_scales, blocks, shape, selector)
        }
        QuantizedMatrix::Q4_0(blocks) => {
            matmul_q4_0_batch(out, x_i8, x_scales, blocks, shape, selector)
        }
        QuantizedMatrix::Q4_0Swizzled1x4(matrix) => {
            debug_assert_eq!(matrix.rows, rows);
            debug_assert_eq!(matrix.cols, cols);
            matmul_q4_0_swizzled_1x4_batch(out, x_i8, x_scales, matrix, shape, selector);
        }
        QuantizedMatrix::Q6K(blocks) => matmul_q6_k_batch(out, x_i8, x_scales, blocks, shape),
    }
}

fn for_each_batch_matmul_row<F>(rows: usize, compute_row: F)
where
    F: Fn(usize) + Send + Sync,
{
    if should_parallelize_matmul(rows) {
        (0..rows)
            .into_par_iter()
            .with_min_len(matmul_min_rows())
            .for_each(compute_row);
    } else {
        (0..rows).for_each(compute_row);
    }
}

#[inline]
unsafe fn write_batch_out(out_addr: usize, token_idx: usize, rows: usize, row: usize, value: f32) {
    // SAFETY: batch matmul workers are partitioned by output row. For a fixed
    // row, each token writes token_idx * rows + row, so no two row workers write
    // the same element.
    unsafe {
        let out = out_addr as *mut f32;
        *out.add(token_idx * rows + row) = value;
    }
}

pub fn matmul_q8_0_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q8_0Block],
    shape: BatchMatmulShape,
    selector: Q8DotKernelSelector,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for token_idx in 0..batch_size {
            let x_offset = token_idx * cols;
            let scale_offset = token_idx * blocks_per_row;
            let x_token = &x_i8[x_offset..x_offset + cols];
            let x_token_scales = &x_scales[scale_offset..scale_offset + blocks_per_row];
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let w_block = &w_row[b];
                let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                let dot_val = match selector.selected {
                    Q8DotKernel::Scalar => crate::q8::dot_i8_scalar(w_block.values(), x_block_vals),
                    Q8DotKernel::Neon => {
                        crate::q8::dot_i8_neon_32_selected(w_block.values(), x_block_vals)
                    }
                    Q8DotKernel::Sdot => {
                        crate::q8::dot_i8_sdot_32_selected(w_block.values(), x_block_vals)
                    }
                };
                sum += w_block.scale_f32() * x_token_scales[b] * dot_val as f32;
            }
            unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
        }
    });
}

pub fn matmul_q4_0_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_0Block],
    shape: BatchMatmulShape,
    selector: Q8DotKernelSelector,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for token_idx in 0..batch_size {
            let x_offset = token_idx * cols;
            let scale_offset = token_idx * blocks_per_row;
            let x_token = &x_i8[x_offset..x_offset + cols];
            let x_token_scales = &x_scales[scale_offset..scale_offset + blocks_per_row];
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let w_block = &w_row[b];
                let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                let dot_val =
                    crate::q8::dot_q4_0_q8_0_with_selector(w_block, x_block_vals, selector);
                sum += w_block.scale_f32() * x_token_scales[b] * dot_val as f32;
            }
            unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
        }
    });
}

fn matmul_q4_0_swizzled_1x4_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &crate::model::Q4_0Swizzled1x4Matrix,
    shape: BatchMatmulShape,
    selector: Q8DotKernelSelector,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    let w = &matrix.swizzled_1x4;
    debug_assert_eq!(w.len(), rows * blocks_per_row);
    debug_assert!(rows.is_multiple_of(4));

    if let Some(aligned) = &matrix.page_aligned_1x4 {
        matmul_q4_0_page_aligned_1x4_batch(out, x_i8, x_scales, aligned, shape, selector);
        return;
    }

    if selector.selected == Q8DotKernel::Sdot && q4_1x4_sdot_enabled() {
        matmul_q4_0_swizzled_1x4_sdot_batch(out, x_i8, x_scales, w, shape, blocks_per_row);
        return;
    }

    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let chunk_idx = r / 4;
        let lane = r % 4;
        let chunk_base = chunk_idx * blocks_per_row * 4;
        for token_idx in 0..batch_size {
            let x_offset = token_idx * cols;
            let scale_offset = token_idx * blocks_per_row;
            let x_token = &x_i8[x_offset..x_offset + cols];
            let x_token_scales = &x_scales[scale_offset..scale_offset + blocks_per_row];
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let w_block = &w[chunk_base + b * 4 + lane];
                let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                let dot_val =
                    crate::q8::dot_q4_0_q8_0_with_selector(w_block, x_block_vals, selector);
                sum += w_block.scale_f32() * x_token_scales[b] * dot_val as f32;
            }
            unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
        }
    });
}

fn matmul_q4_0_swizzled_1x4_sdot_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_0Block],
    shape: BatchMatmulShape,
    blocks_per_row: usize,
) {
    let rows = shape.rows;
    let chunks = rows / 4;
    let out_addr = out.as_mut_ptr() as usize;
    let ctx = SwizzledBatchChunkContext {
        out_addr,
        x_i8,
        x_scales,
        w,
        shape,
        blocks_per_row,
    };
    if should_parallelize_matmul(rows) {
        (0..chunks).into_par_iter().for_each(|chunk_idx| {
            compute_q4_0_swizzled_1x4_sdot_batch_chunk(chunk_idx, ctx);
        });
    } else {
        (0..chunks).for_each(|chunk_idx| {
            compute_q4_0_swizzled_1x4_sdot_batch_chunk(chunk_idx, ctx);
        });
    }
}

#[derive(Clone, Copy)]
struct SwizzledBatchChunkContext<'a> {
    out_addr: usize,
    x_i8: &'a [i8],
    x_scales: &'a [f32],
    w: &'a [Q4_0Block],
    shape: BatchMatmulShape,
    blocks_per_row: usize,
}

fn compute_q4_0_swizzled_1x4_sdot_batch_chunk(
    chunk_idx: usize,
    ctx: SwizzledBatchChunkContext<'_>,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = ctx.shape;
    let chunk_base = chunk_idx * ctx.blocks_per_row * 4;
    let row_base = chunk_idx * 4;
    let mut token_sums = vec![[0.0_f32; 4]; batch_size];
    for b in 0..ctx.blocks_per_row {
        let w_base = chunk_base + b * 4;
        let row0 = &ctx.w[w_base];
        let row1 = &ctx.w[w_base + 1];
        let row2 = &ctx.w[w_base + 2];
        let row3 = &ctx.w[w_base + 3];
        let row0_scale = row0.scale_f32();
        let row1_scale = row1.scale_f32();
        let row2_scale = row2.scale_f32();
        let row3_scale = row3.scale_f32();

        #[cfg(target_arch = "aarch64")]
        let unpacked = unsafe {
            let (w0_low, w0_high) = crate::q8::unpack_q4_0_lanes_aarch64(row0);
            let (w1_low, w1_high) = crate::q8::unpack_q4_0_lanes_aarch64(row1);
            let (w2_low, w2_high) = crate::q8::unpack_q4_0_lanes_aarch64(row2);
            let (w3_low, w3_high) = crate::q8::unpack_q4_0_lanes_aarch64(row3);
            crate::q8::Q4_0Unpacked1x4Aarch64 {
                w0_low,
                w0_high,
                w1_low,
                w1_high,
                w2_low,
                w2_high,
                w3_low,
                w3_high,
            }
        };

        for (token_idx, sums) in token_sums.iter_mut().enumerate() {
            let x_offset = token_idx * cols;
            let x_scale = ctx.x_scales[token_idx * ctx.blocks_per_row + b];
            let x_token = &ctx.x_i8[x_offset..x_offset + cols];
            let x_block_vals = unsafe { activation_block_ptr(x_token, b) };

            #[cfg(target_arch = "aarch64")]
            let dots = unsafe {
                crate::q8::dot_q4_0_q8_0_1x4_sdot_preloaded_aarch64(unpacked, x_block_vals)
            };
            #[cfg(not(target_arch = "aarch64"))]
            let dots = [
                crate::q8::dot_q4_0_q8_0_scalar(row0, x_block_vals),
                crate::q8::dot_q4_0_q8_0_scalar(row1, x_block_vals),
                crate::q8::dot_q4_0_q8_0_scalar(row2, x_block_vals),
                crate::q8::dot_q4_0_q8_0_scalar(row3, x_block_vals),
            ];

            sums[0] += row0_scale * x_scale * dots[0] as f32;
            sums[1] += row1_scale * x_scale * dots[1] as f32;
            sums[2] += row2_scale * x_scale * dots[2] as f32;
            sums[3] += row3_scale * x_scale * dots[3] as f32;
        }
    }

    for (token_idx, sums) in token_sums.into_iter().enumerate() {
        for (lane, sum) in sums.into_iter().enumerate() {
            unsafe {
                write_batch_out(ctx.out_addr, token_idx, rows, row_base + lane, sum);
            }
        }
    }
}

fn matmul_q4_0_page_aligned_1x4_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &PageAlignedQ4_0Swizzled1x4,
    shape: BatchMatmulShape,
    selector: Q8DotKernelSelector,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    debug_assert_eq!(matrix.blocks_per_row(), blocks_per_row);
    debug_assert_eq!(matrix.chunk_count(), rows / 4);

    if selector.selected == Q8DotKernel::Sdot && q4_1x4_sdot_enabled() {
        let chunks = rows / 4;
        let out_addr = out.as_mut_ptr() as usize;
        let ctx = PageAlignedBatchChunkContext {
            out_addr,
            x_i8,
            x_scales,
            matrix,
            shape,
            blocks_per_row,
        };
        if should_parallelize_matmul(rows) {
            (0..chunks).into_par_iter().for_each(|chunk_idx| {
                compute_q4_0_page_aligned_1x4_sdot_batch_chunk(chunk_idx, ctx);
            });
        } else {
            (0..chunks).for_each(|chunk_idx| {
                compute_q4_0_page_aligned_1x4_sdot_batch_chunk(chunk_idx, ctx);
            });
        }
        return;
    }

    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let chunk_idx = r / 4;
        let lane = r % 4;
        let chunk = matrix.chunk_ptr(chunk_idx);
        for token_idx in 0..batch_size {
            let x_offset = token_idx * cols;
            let scale_offset = token_idx * blocks_per_row;
            let x_token = &x_i8[x_offset..x_offset + cols];
            let x_token_scales = &x_scales[scale_offset..scale_offset + blocks_per_row];
            let mut sum = 0.0_f32;
            for (b, x_scale) in x_token_scales
                .iter()
                .copied()
                .enumerate()
                .take(blocks_per_row)
            {
                let w_block = unsafe { &*chunk.add(b * 4 + lane) };
                let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                let dot_val =
                    crate::q8::dot_q4_0_q8_0_with_selector(w_block, x_block_vals, selector);
                sum += w_block.scale_f32() * x_scale * dot_val as f32;
            }
            unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
        }
    });
}

#[derive(Clone, Copy)]
struct PageAlignedBatchChunkContext<'a> {
    out_addr: usize,
    x_i8: &'a [i8],
    x_scales: &'a [f32],
    matrix: &'a PageAlignedQ4_0Swizzled1x4,
    shape: BatchMatmulShape,
    blocks_per_row: usize,
}

fn compute_q4_0_page_aligned_1x4_sdot_batch_chunk(
    chunk_idx: usize,
    ctx: PageAlignedBatchChunkContext<'_>,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = ctx.shape;
    let chunk = ctx.matrix.chunk_ptr(chunk_idx);
    let row_base = chunk_idx * 4;
    let mut token_sums = vec![[0.0_f32; 4]; batch_size];
    for b in 0..ctx.blocks_per_row {
        let w_base = b * 4;
        let row0 = unsafe { &*chunk.add(w_base) };
        let row1 = unsafe { &*chunk.add(w_base + 1) };
        let row2 = unsafe { &*chunk.add(w_base + 2) };
        let row3 = unsafe { &*chunk.add(w_base + 3) };
        let row0_scale = row0.scale_f32();
        let row1_scale = row1.scale_f32();
        let row2_scale = row2.scale_f32();
        let row3_scale = row3.scale_f32();

        #[cfg(target_arch = "aarch64")]
        let unpacked = unsafe {
            let (w0_low, w0_high) = crate::q8::unpack_q4_0_lanes_aarch64(row0);
            let (w1_low, w1_high) = crate::q8::unpack_q4_0_lanes_aarch64(row1);
            let (w2_low, w2_high) = crate::q8::unpack_q4_0_lanes_aarch64(row2);
            let (w3_low, w3_high) = crate::q8::unpack_q4_0_lanes_aarch64(row3);
            crate::q8::Q4_0Unpacked1x4Aarch64 {
                w0_low,
                w0_high,
                w1_low,
                w1_high,
                w2_low,
                w2_high,
                w3_low,
                w3_high,
            }
        };

        for (token_idx, sums) in token_sums.iter_mut().enumerate() {
            let x_offset = token_idx * cols;
            let x_scale = ctx.x_scales[token_idx * ctx.blocks_per_row + b];
            let x_token = &ctx.x_i8[x_offset..x_offset + cols];
            let x_block_vals = unsafe { activation_block_ptr(x_token, b) };

            #[cfg(target_arch = "aarch64")]
            let dots = unsafe {
                crate::q8::dot_q4_0_q8_0_1x4_sdot_preloaded_aarch64(unpacked, x_block_vals)
            };
            #[cfg(not(target_arch = "aarch64"))]
            let dots = [
                crate::q8::dot_q4_0_q8_0_scalar(row0, x_block_vals),
                crate::q8::dot_q4_0_q8_0_scalar(row1, x_block_vals),
                crate::q8::dot_q4_0_q8_0_scalar(row2, x_block_vals),
                crate::q8::dot_q4_0_q8_0_scalar(row3, x_block_vals),
            ];

            sums[0] += row0_scale * x_scale * dots[0] as f32;
            sums[1] += row1_scale * x_scale * dots[1] as f32;
            sums[2] += row2_scale * x_scale * dots[2] as f32;
            sums[3] += row3_scale * x_scale * dots[3] as f32;
        }
    }

    for (token_idx, sums) in token_sums.into_iter().enumerate() {
        for (lane, sum) in sums.into_iter().enumerate() {
            unsafe {
                write_batch_out(ctx.out_addr, token_idx, rows, row_base + lane, sum);
            }
        }
    }
}

pub fn matmul_q6_k_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q6KBlock],
    shape: BatchMatmulShape,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    let q8_blocks_per_token = cols / Q8_BLOCK_SIZE;
    let out_addr = out.as_mut_ptr() as usize;
    #[cfg(target_arch = "aarch64")]
    if q6_k_sdot_enabled() && std::arch::is_aarch64_feature_detected!("dotprod") {
        for_each_batch_matmul_row(rows, |r| {
            let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
            for token_idx in 0..batch_size {
                let x_offset = token_idx * cols;
                let scale_offset = token_idx * q8_blocks_per_token;
                let x_token = &x_i8[x_offset..x_offset + cols];
                let x_token_scales = &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                let mut sum = 0.0_f32;
                for (block_idx, w_block) in w_row.iter().enumerate() {
                    let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                    let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                    sum += unsafe {
                        w_block.dot_q8_scaled_sdot(
                            &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                            &x_token_scales
                                [x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                        )
                    };
                }
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        });
        return;
    }

    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for token_idx in 0..batch_size {
            let x_offset = token_idx * cols;
            let scale_offset = token_idx * q8_blocks_per_token;
            let x_token = &x_i8[x_offset..x_offset + cols];
            let x_token_scales = &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
            let mut sum = 0.0_f32;
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                sum += w_block.dot_q8_scaled(
                    &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                    &x_token_scales[x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / 32)],
                );
            }
            unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
        }
    });
}

pub fn matmul_q4_0(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_0Block],
    _rows: usize,
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    if selector.selected == Q8DotKernel::Sdot && q4_1x4_sdot_enabled() {
        matmul_q4_0_sdot_1x4(out, x_i8, x_scales, w, cols, selector);
        return;
    }

    let blocks_per_row = cols / 32;
    for_each_matmul_row(out, |r, out_val| {
        let mut sum = 0.0_f32;
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for b in 0..blocks_per_row {
            let w_block = &w_row[b];
            let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
            let dot_val = crate::q8::dot_q4_0_q8_0_with_selector(w_block, x_block_vals, selector);
            sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
        }
        *out_val = sum;
    });
}

fn matmul_q4_0_sdot_1x4(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_0Block],
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    let blocks_per_row = cols / 32;
    if should_parallelize_matmul(out.len()) {
        out.par_chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q4_0_sdot_1x4_chunk(
                    chunk_idx * 4,
                    out_chunk,
                    x_i8,
                    x_scales,
                    w,
                    blocks_per_row,
                    selector,
                );
            });
    } else {
        out.chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q4_0_sdot_1x4_chunk(
                    chunk_idx * 4,
                    out_chunk,
                    x_i8,
                    x_scales,
                    w,
                    blocks_per_row,
                    selector,
                );
            });
    }
}

fn compute_q4_0_sdot_1x4_chunk(
    row_base: usize,
    out_chunk: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_0Block],
    blocks_per_row: usize,
    selector: Q8DotKernelSelector,
) {
    if out_chunk.len() == 4 {
        let mut sums = [0.0_f32; 4];
        for (b, x_scale) in x_scales.iter().copied().enumerate().take(blocks_per_row) {
            let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
            let w_base = row_base * blocks_per_row + b;
            let row0 = &w[w_base];
            let row1 = &w[w_base + blocks_per_row];
            let row2 = &w[w_base + 2 * blocks_per_row];
            let row3 = &w[w_base + 3 * blocks_per_row];
            let dots = dot_q4_0_q8_0_1x4_sdot([row0, row1, row2, row3], x_block_vals);
            sums[0] += row0.scale_f32() * x_scale * dots[0] as f32;
            sums[1] += row1.scale_f32() * x_scale * dots[1] as f32;
            sums[2] += row2.scale_f32() * x_scale * dots[2] as f32;
            sums[3] += row3.scale_f32() * x_scale * dots[3] as f32;
        }
        out_chunk.copy_from_slice(&sums);
        return;
    }

    for (lane, out_val) in out_chunk.iter_mut().enumerate() {
        let row = row_base + lane;
        let mut sum = 0.0_f32;
        let w_row = &w[row * blocks_per_row..(row + 1) * blocks_per_row];
        for b in 0..blocks_per_row {
            let w_block = &w_row[b];
            let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
            let dot_val = crate::q8::dot_q4_0_q8_0_with_selector(w_block, x_block_vals, selector);
            sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
        }
        *out_val = sum;
    }
}

fn matmul_q4_0_swizzled_1x4(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &crate::model::Q4_0Swizzled1x4Matrix,
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    let blocks_per_row = cols / 32;
    let w = &matrix.swizzled_1x4;
    debug_assert_eq!(w.len(), out.len() * blocks_per_row);
    debug_assert!(out.len().is_multiple_of(4));

    if let Some(aligned) = &matrix.page_aligned_1x4 {
        matmul_q4_0_page_aligned_1x4(out, x_i8, x_scales, aligned, cols, selector);
        return;
    }

    if should_parallelize_matmul(out.len()) {
        out.par_chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q4_0_sdot_1x4_swizzled_chunk(
                    chunk_idx,
                    out_chunk,
                    x_i8,
                    x_scales,
                    w,
                    blocks_per_row,
                    selector,
                );
            });
    } else {
        out.chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q4_0_sdot_1x4_swizzled_chunk(
                    chunk_idx,
                    out_chunk,
                    x_i8,
                    x_scales,
                    w,
                    blocks_per_row,
                    selector,
                );
            });
    }
}

fn compute_q4_0_sdot_1x4_swizzled_chunk(
    chunk_idx: usize,
    out_chunk: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_0Block],
    blocks_per_row: usize,
    selector: Q8DotKernelSelector,
) {
    debug_assert_eq!(out_chunk.len(), 4);

    let mut sums = [0.0_f32; 4];
    let chunk_base = chunk_idx * blocks_per_row * 4;
    for (b, x_scale) in x_scales.iter().copied().enumerate().take(blocks_per_row) {
        let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
        let w_base = chunk_base + b * 4;
        let row0 = &w[w_base];
        let row1 = &w[w_base + 1];
        let row2 = &w[w_base + 2];
        let row3 = &w[w_base + 3];
        let dots = if selector.selected == Q8DotKernel::Sdot && q4_1x4_sdot_enabled() {
            dot_q4_0_q8_0_1x4_sdot([row0, row1, row2, row3], x_block_vals)
        } else {
            [
                crate::q8::dot_q4_0_q8_0_with_selector(row0, x_block_vals, selector),
                crate::q8::dot_q4_0_q8_0_with_selector(row1, x_block_vals, selector),
                crate::q8::dot_q4_0_q8_0_with_selector(row2, x_block_vals, selector),
                crate::q8::dot_q4_0_q8_0_with_selector(row3, x_block_vals, selector),
            ]
        };
        sums[0] += row0.scale_f32() * x_scale * dots[0] as f32;
        sums[1] += row1.scale_f32() * x_scale * dots[1] as f32;
        sums[2] += row2.scale_f32() * x_scale * dots[2] as f32;
        sums[3] += row3.scale_f32() * x_scale * dots[3] as f32;
    }
    out_chunk.copy_from_slice(&sums);
}

fn dot_q4_0_q8_0_1x4_sdot(weights: [&Q4_0Block; 4], activation: &[i8; 32]) -> [i32; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the 1x4 path is entered only after SDOT selector validation.
        unsafe { crate::q8::dot_q4_0_q8_0_1x4_sdot_aarch64(weights, activation) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        [
            crate::q8::dot_q4_0_q8_0_scalar(weights[0], activation),
            crate::q8::dot_q4_0_q8_0_scalar(weights[1], activation),
            crate::q8::dot_q4_0_q8_0_scalar(weights[2], activation),
            crate::q8::dot_q4_0_q8_0_scalar(weights[3], activation),
        ]
    }
}

fn matmul_q4_0_page_aligned_1x4(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &PageAlignedQ4_0Swizzled1x4,
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    debug_assert_eq!(matrix.blocks_per_row(), blocks_per_row);
    debug_assert_eq!(matrix.chunk_count(), out.len() / 4);
    debug_assert!(out.len().is_multiple_of(4));

    if should_parallelize_matmul(out.len()) {
        out.par_chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q4_0_page_aligned_1x4_chunk(
                    chunk_idx, out_chunk, x_i8, x_scales, matrix, selector,
                );
            });
    } else {
        out.chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q4_0_page_aligned_1x4_chunk(
                    chunk_idx, out_chunk, x_i8, x_scales, matrix, selector,
                );
            });
    }
}

fn compute_q4_0_page_aligned_1x4_chunk(
    chunk_idx: usize,
    out_chunk: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &PageAlignedQ4_0Swizzled1x4,
    selector: Q8DotKernelSelector,
) {
    debug_assert_eq!(out_chunk.len(), 4);

    let mut sums = [0.0_f32; 4];
    let chunk = matrix.chunk_ptr(chunk_idx);
    for (b, x_scale) in x_scales
        .iter()
        .copied()
        .enumerate()
        .take(matrix.blocks_per_row())
    {
        let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
        let w_base = b * 4;
        let row0 = unsafe { &*chunk.add(w_base) };
        let row1 = unsafe { &*chunk.add(w_base + 1) };
        let row2 = unsafe { &*chunk.add(w_base + 2) };
        let row3 = unsafe { &*chunk.add(w_base + 3) };
        let dots = if selector.selected == Q8DotKernel::Sdot && q4_1x4_sdot_enabled() {
            dot_q4_0_q8_0_1x4_sdot([row0, row1, row2, row3], x_block_vals)
        } else {
            [
                crate::q8::dot_q4_0_q8_0_with_selector(row0, x_block_vals, selector),
                crate::q8::dot_q4_0_q8_0_with_selector(row1, x_block_vals, selector),
                crate::q8::dot_q4_0_q8_0_with_selector(row2, x_block_vals, selector),
                crate::q8::dot_q4_0_q8_0_with_selector(row3, x_block_vals, selector),
            ]
        };
        sums[0] += row0.scale_f32() * x_scale * dots[0] as f32;
        sums[1] += row1.scale_f32() * x_scale * dots[1] as f32;
        sums[2] += row2.scale_f32() * x_scale * dots[2] as f32;
        sums[3] += row3.scale_f32() * x_scale * dots[3] as f32;
    }

    out_chunk.copy_from_slice(&sums);
}

pub fn matmul_q6_k(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q6KBlock],
    _rows: usize,
    cols: usize,
) {
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    #[cfg(target_arch = "aarch64")]
    if q6_k_sdot_enabled() && std::arch::is_aarch64_feature_detected!("dotprod") {
        for_each_matmul_row(out, |r, out_val| {
            let mut sum = 0.0_f32;
            let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                sum += unsafe {
                    w_block.dot_q8_scaled_sdot(
                        &x_i8[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                        &x_scales[x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                    )
                };
            }
            *out_val = sum;
        });
        return;
    }

    for_each_matmul_row(out, |r, out_val| {
        let mut sum = 0.0_f32;
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for (block_idx, w_block) in w_row.iter().enumerate() {
            let x_block_start = block_idx * QK_K_BLOCK_SIZE;
            let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
            sum += w_block.dot_q8_scaled(
                &x_i8[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                &x_scales[x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
            );
        }
        *out_val = sum;
    });
}

pub fn matmul_f32(out: &mut [f32], x: &[f32], w: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let mut sum = 0.0_f32;
        let w_row = &w[r * cols..(r + 1) * cols];
        for c in 0..cols {
            sum += x[c] * w_row[c];
        }
        out[r] = sum;
    }
}

fn add_bias(values: &mut [f32], bias: &[f32]) {
    debug_assert_eq!(values.len(), bias.len());
    for (value, bias) in values.iter_mut().zip(bias) {
        *value += bias;
    }
}

pub fn silu(x: &mut [f32]) {
    for val in x.iter_mut() {
        let sigmoid = 1.0 / (1.0 + (-*val).exp());
        *val *= sigmoid;
    }
}

pub fn fused_silu_mul(gate_up: &mut [f32], gate: &[f32], up: &[f32]) {
    debug_assert_eq!(gate_up.len(), gate.len());
    debug_assert_eq!(gate.len(), up.len());
    if gate_up.len() >= matmul_min_rows() {
        gate_up
            .par_iter_mut()
            .with_min_len(matmul_min_rows())
            .zip(gate.par_iter())
            .zip(up.par_iter())
            .for_each(|((dst, &g), &u)| {
                let sigmoid = 1.0 / (1.0 + (-g).exp());
                *dst = g * sigmoid * u;
            });
    } else {
        gate_up
            .iter_mut()
            .zip(gate)
            .zip(up)
            .for_each(|((dst, &g), &u)| {
                let sigmoid = 1.0 / (1.0 + (-g).exp());
                *dst = g * sigmoid * u;
            });
    }
}

pub fn softmax(x: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &val in x.iter() {
        if val > max_val {
            max_val = val;
        }
    }

    let mut sum = 0.0;
    for val in x.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }

    let inv_sum = 1.0 / sum;
    for val in x.iter_mut() {
        *val *= inv_sum;
    }
}

fn dot_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 guarantees NEON/ASIMD support.
        unsafe { dot_f32_neon(lhs, rhs) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_f32_scalar(lhs, rhs)
    }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn dot_f32_scalar(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(&a, &b)| a * b).sum()
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_f32_neon(lhs: &[f32], rhs: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vmlaq_f32};

    let mut acc0 = unsafe { vdupq_n_f32(0.0) };
    let mut acc1 = unsafe { vdupq_n_f32(0.0) };
    let mut acc2 = unsafe { vdupq_n_f32(0.0) };
    let mut acc3 = unsafe { vdupq_n_f32(0.0) };
    let chunks = lhs.len() / 16;
    for chunk_idx in 0..chunks {
        let offset = chunk_idx * 16;
        unsafe {
            acc0 = vmlaq_f32(
                acc0,
                vld1q_f32(lhs.as_ptr().add(offset)),
                vld1q_f32(rhs.as_ptr().add(offset)),
            );
            acc1 = vmlaq_f32(
                acc1,
                vld1q_f32(lhs.as_ptr().add(offset + 4)),
                vld1q_f32(rhs.as_ptr().add(offset + 4)),
            );
            acc2 = vmlaq_f32(
                acc2,
                vld1q_f32(lhs.as_ptr().add(offset + 8)),
                vld1q_f32(rhs.as_ptr().add(offset + 8)),
            );
            acc3 = vmlaq_f32(
                acc3,
                vld1q_f32(lhs.as_ptr().add(offset + 12)),
                vld1q_f32(rhs.as_ptr().add(offset + 12)),
            );
        }
    }

    let mut sum = unsafe { vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3))) };
    for idx in chunks * 16..lhs.len() {
        sum += lhs[idx] * rhs[idx];
    }
    sum
}

fn add_weighted_f32(out: &mut [f32], values: &[f32], weight: f32) {
    debug_assert_eq!(out.len(), values.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 guarantees NEON/ASIMD support.
        unsafe {
            add_weighted_f32_neon(out, values, weight);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        add_weighted_f32_scalar(out, values, weight);
    }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn add_weighted_f32_scalar(out: &mut [f32], values: &[f32], weight: f32) {
    for (dst, &value) in out.iter_mut().zip(values) {
        *dst += weight * value;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn add_weighted_f32_neon(out: &mut [f32], values: &[f32], weight: f32) {
    use std::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmlaq_f32, vst1q_f32};

    let weight_vec = unsafe { vdupq_n_f32(weight) };
    let chunks = out.len() / 16;
    for chunk_idx in 0..chunks {
        let offset = chunk_idx * 16;
        for lane_offset in [0, 4, 8, 12] {
            unsafe {
                let dst_ptr = out.as_mut_ptr().add(offset + lane_offset);
                let dst = vld1q_f32(dst_ptr);
                let src = vld1q_f32(values.as_ptr().add(offset + lane_offset));
                vst1q_f32(dst_ptr, vmlaq_f32(dst, src, weight_vec));
            }
        }
    }

    for idx in chunks * 16..out.len() {
        out[idx] += weight * values[idx];
    }
}

pub fn forward_pass<'a>(
    token_id: usize,
    pos: usize,
    config: &LlamaModelConfig,
    weights: &LlamaWeights,
    cache: &mut LlamaKvCache,
    ws: &'a mut LlamaWorkspace,
    options: LlamaRuntimeOptions,
) -> &'a [f32] {
    forward_pass_inner(
        token_id,
        pos,
        config,
        weights,
        cache,
        ws,
        LlamaRuntimeOptions {
            compute_logits: true,
            ..options
        },
    )
}

pub fn prefill_pass(
    token_id: usize,
    pos: usize,
    config: &LlamaModelConfig,
    weights: &LlamaWeights,
    cache: &mut LlamaKvCache,
    ws: &mut LlamaWorkspace,
    options: LlamaRuntimeOptions,
) {
    let _ = forward_pass_inner(
        token_id,
        pos,
        config,
        weights,
        cache,
        ws,
        LlamaRuntimeOptions {
            compute_logits: false,
            ..options
        },
    );
}

pub fn prefill_pass_batch(
    token_ids: &[u32],
    start_pos: usize,
    config: &LlamaModelConfig,
    weights: &LlamaWeights,
    cache: &mut LlamaKvCache,
    ws: &mut LlamaBatchWorkspace,
    options: LlamaRuntimeOptions,
) {
    let batch_size = token_ids.len();
    if batch_size == 0 {
        return;
    }
    debug_assert!(batch_size <= ws.max_batch);
    debug_assert!(start_pos + batch_size <= config.context_length);

    for (token_idx, &token_id) in token_ids.iter().enumerate() {
        let emb_start = token_id as usize * config.embedding_length;
        let hidden_start = token_idx * config.embedding_length;
        ws.hidden[hidden_start..hidden_start + config.embedding_length].copy_from_slice(
            &weights.token_embeddings[emb_start..emb_start + config.embedding_length],
        );
    }

    for layer_idx in 0..config.block_count {
        let layer = &weights.layers[layer_idx];
        let hidden_len = batch_size * config.embedding_length;
        let kv_len = batch_size * config.kv_width;
        let ffn_len = batch_size * config.feed_forward_length;

        ws.residual[..hidden_len].copy_from_slice(&ws.hidden[..hidden_len]);
        rms_norm_batch(
            &mut ws.norm_x[..hidden_len],
            &ws.hidden[..hidden_len],
            &layer.attention_norm,
            config.rms_norm_epsilon,
            batch_size,
            config.embedding_length,
        );
        quantize_f32_to_q8_0_batch(
            &ws.norm_x[..hidden_len],
            &mut ws.x_i8[..batch_size * config.embedding_length],
            &mut ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
            batch_size,
            config.embedding_length,
        );

        let attention_shape = BatchMatmulShape {
            batch_size,
            rows: config.embedding_length,
            cols: config.embedding_length,
        };
        matmul_quantized_batch(
            &mut ws.q[..hidden_len],
            &ws.x_i8[..batch_size * config.embedding_length],
            &ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
            &layer.wq,
            attention_shape,
            options.q8_selector,
        );
        add_bias_batch(
            &mut ws.q[..hidden_len],
            &layer.wq_bias,
            batch_size,
            config.embedding_length,
        );

        let kv_shape = BatchMatmulShape {
            batch_size,
            rows: config.kv_width,
            cols: config.embedding_length,
        };
        matmul_quantized_batch(
            &mut ws.k[..kv_len],
            &ws.x_i8[..batch_size * config.embedding_length],
            &ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
            &layer.wk,
            kv_shape,
            options.q8_selector,
        );
        add_bias_batch(
            &mut ws.k[..kv_len],
            &layer.wk_bias,
            batch_size,
            config.kv_width,
        );
        matmul_quantized_batch(
            &mut ws.v[..kv_len],
            &ws.x_i8[..batch_size * config.embedding_length],
            &ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
            &layer.wav,
            kv_shape,
            options.q8_selector,
        );
        add_bias_batch(
            &mut ws.v[..kv_len],
            &layer.wav_bias,
            batch_size,
            config.kv_width,
        );

        for token_idx in 0..batch_size {
            let pos = start_pos + token_idx;
            let q_start = token_idx * config.embedding_length;
            let kv_start = token_idx * config.kv_width;
            apply_rope(
                &mut ws.q[q_start..q_start + config.embedding_length],
                pos,
                config.attention_head_count,
                config.head_dim,
                config.rope_dimension_count,
                config.rope_freq_base,
                options.rope_scaling,
            );
            apply_rope(
                &mut ws.k[kv_start..kv_start + config.kv_width],
                pos,
                config.attention_head_count_kv,
                config.head_dim,
                config.rope_dimension_count,
                config.rope_freq_base,
                options.rope_scaling,
            );
            cache.store_kv(
                layer_idx,
                pos,
                &ws.k[kv_start..kv_start + config.kv_width],
                &ws.v[kv_start..kv_start + config.kv_width],
            );
        }

        let k_cache = cache.get_k_cache(layer_idx);
        let v_cache = cache.get_v_cache(layer_idx);
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        ws.attn_output[..hidden_len].fill(0.0);
        let parallel_heads = attention_head_parallel_enabled();

        for token_idx in 0..batch_size {
            let pos = start_pos + token_idx;
            let q_token_start = token_idx * config.embedding_length;
            let out_token_start = token_idx * config.embedding_length;
            let scores = if parallel_heads
                && ws.attn_scores.len()
                    >= batch_size * config.attention_head_count * config.context_length
            {
                let scores_start = token_idx * config.attention_head_count * config.context_length;
                let scores_end = scores_start + config.attention_head_count * config.context_length;
                &mut ws.attn_scores[scores_start..scores_end]
            } else {
                let scores_start = token_idx * config.context_length;
                &mut ws.attn_scores[scores_start..scores_start + config.context_length]
            };
            apply_attention_heads(
                &mut ws.attn_output[out_token_start..out_token_start + config.embedding_length],
                scores,
                AttentionInput {
                    q: &ws.q[q_token_start..q_token_start + config.embedding_length],
                    k_cache,
                    v_cache,
                    pos,
                    head_count: config.attention_head_count,
                    kv_head_count: config.attention_head_count_kv,
                    head_dim: config.head_dim,
                    cache_kv_width: cache.kv_width,
                    context_length: config.context_length,
                    scale,
                },
                parallel_heads,
            );
        }

        quantize_f32_to_q8_0_batch(
            &ws.attn_output[..hidden_len],
            &mut ws.x_i8[..batch_size * config.embedding_length],
            &mut ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
            batch_size,
            config.embedding_length,
        );
        matmul_quantized_batch(
            &mut ws.hidden[..hidden_len],
            &ws.x_i8[..batch_size * config.embedding_length],
            &ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
            &layer.wo,
            attention_shape,
            options.q8_selector,
        );
        add_residual_batch(&mut ws.hidden[..hidden_len], &ws.residual[..hidden_len]);

        ws.residual[..hidden_len].copy_from_slice(&ws.hidden[..hidden_len]);
        rms_norm_batch(
            &mut ws.norm_x[..hidden_len],
            &ws.hidden[..hidden_len],
            &layer.ffn_norm,
            config.rms_norm_epsilon,
            batch_size,
            config.embedding_length,
        );
        quantize_f32_to_q8_0_batch(
            &ws.norm_x[..hidden_len],
            &mut ws.x_i8[..batch_size * config.embedding_length],
            &mut ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
            batch_size,
            config.embedding_length,
        );

        let ffn_shape = BatchMatmulShape {
            batch_size,
            rows: config.feed_forward_length,
            cols: config.embedding_length,
        };
        matmul_quantized_batch(
            &mut ws.ffn_gate[..ffn_len],
            &ws.x_i8[..batch_size * config.embedding_length],
            &ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
            &layer.w1,
            ffn_shape,
            options.q8_selector,
        );
        matmul_quantized_batch(
            &mut ws.ffn_up[..ffn_len],
            &ws.x_i8[..batch_size * config.embedding_length],
            &ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
            &layer.w3,
            ffn_shape,
            options.q8_selector,
        );
        fused_silu_mul(
            &mut ws.ffn_gate_up[..ffn_len],
            &ws.ffn_gate[..ffn_len],
            &ws.ffn_up[..ffn_len],
        );

        quantize_f32_to_q8_0_batch(
            &ws.ffn_gate_up[..ffn_len],
            &mut ws.x_i8[..batch_size * config.feed_forward_length],
            &mut ws.x_scales[..batch_size * (config.feed_forward_length / Q8_BLOCK_SIZE)],
            batch_size,
            config.feed_forward_length,
        );
        let down_shape = BatchMatmulShape {
            batch_size,
            rows: config.embedding_length,
            cols: config.feed_forward_length,
        };
        matmul_quantized_batch(
            &mut ws.hidden[..hidden_len],
            &ws.x_i8[..batch_size * config.feed_forward_length],
            &ws.x_scales[..batch_size * (config.feed_forward_length / Q8_BLOCK_SIZE)],
            &layer.w2,
            down_shape,
            options.q8_selector,
        );
        add_residual_batch(&mut ws.hidden[..hidden_len], &ws.residual[..hidden_len]);
    }
}

fn add_bias_batch(values: &mut [f32], bias: &Option<Vec<f32>>, batch_size: usize, dim: usize) {
    if let Some(bias) = bias {
        for token_idx in 0..batch_size {
            let start = token_idx * dim;
            add_bias(&mut values[start..start + dim], bias);
        }
    }
}

fn add_residual_batch(values: &mut [f32], residual: &[f32]) {
    debug_assert_eq!(values.len(), residual.len());
    for (value, residual) in values.iter_mut().zip(residual) {
        *value += residual;
    }
}

struct AttentionInput<'a> {
    q: &'a [f32],
    k_cache: KvCacheSlice<'a>,
    v_cache: KvCacheSlice<'a>,
    pos: usize,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    cache_kv_width: usize,
    context_length: usize,
    scale: f32,
}

fn apply_attention_heads(
    attn_output: &mut [f32],
    attn_scores: &mut [f32],
    input: AttentionInput<'_>,
    parallel_heads: bool,
) {
    let kv_mul = input.head_count / input.kv_head_count;
    let can_parallelize =
        parallel_heads && attn_scores.len() >= input.head_count * input.context_length;

    if can_parallelize {
        attn_output
            .par_chunks_exact_mut(input.head_dim)
            .zip(attn_scores.par_chunks_exact_mut(input.context_length))
            .enumerate()
            .for_each(|(h, (out_head, scores))| {
                apply_attention_head(out_head, scores, &input, h, kv_mul);
            });
    } else {
        for h in 0..input.head_count {
            let out_start = h * input.head_dim;
            apply_attention_head(
                &mut attn_output[out_start..out_start + input.head_dim],
                attn_scores,
                &input,
                h,
                kv_mul,
            );
        }
    }
}

fn apply_attention_head(
    out_head: &mut [f32],
    scores: &mut [f32],
    input: &AttentionInput<'_>,
    head_idx: usize,
    kv_mul: usize,
) {
    let kv_h = head_idx / kv_mul;
    let q_head = &input.q[head_idx * input.head_dim..(head_idx + 1) * input.head_dim];

    match (input.k_cache, input.v_cache) {
        (KvCacheSlice::F32(k_cache), KvCacheSlice::F32(v_cache)) => {
            for (p, score) in scores.iter_mut().enumerate().take(input.pos + 1) {
                let k_head = &k_cache[p * input.cache_kv_width + kv_h * input.head_dim
                    ..p * input.cache_kv_width + (kv_h + 1) * input.head_dim];
                *score = dot_f32(q_head, k_head) * input.scale;
            }
            softmax(&mut scores[0..=input.pos]);

            for (p, &score) in scores.iter().enumerate().take(input.pos + 1) {
                let v_head = &v_cache[p * input.cache_kv_width + kv_h * input.head_dim
                    ..p * input.cache_kv_width + (kv_h + 1) * input.head_dim];
                add_weighted_f32(out_head, v_head, score);
            }
        }
        (KvCacheSlice::F16(k_cache), KvCacheSlice::F16(v_cache)) => {
            apply_attention_head_f16(out_head, scores, input, kv_h, q_head, k_cache, v_cache);
        }
        _ => unreachable!("KV cache key and value storage kinds must match"),
    }
}

fn apply_attention_head_f16(
    out_head: &mut [f32],
    scores: &mut [f32],
    input: &AttentionInput<'_>,
    kv_h: usize,
    q_head: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
) {
    const MAX_STACK_HEAD_DIM: usize = 256;
    if input.head_dim <= MAX_STACK_HEAD_DIM {
        let mut k_head_f32 = [0.0; MAX_STACK_HEAD_DIM];
        let mut v_head_f32 = [0.0; MAX_STACK_HEAD_DIM];
        apply_attention_head_f16_with_buffers(
            out_head,
            scores,
            input,
            kv_h,
            q_head,
            k_cache,
            v_cache,
            &mut k_head_f32[..input.head_dim],
            &mut v_head_f32[..input.head_dim],
        );
    } else {
        let mut k_head_f32 = vec![0.0; input.head_dim];
        let mut v_head_f32 = vec![0.0; input.head_dim];
        apply_attention_head_f16_with_buffers(
            out_head,
            scores,
            input,
            kv_h,
            q_head,
            k_cache,
            v_cache,
            &mut k_head_f32,
            &mut v_head_f32,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_attention_head_f16_with_buffers(
    out_head: &mut [f32],
    scores: &mut [f32],
    input: &AttentionInput<'_>,
    kv_h: usize,
    q_head: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    k_head_f32: &mut [f32],
    v_head_f32: &mut [f32],
) {
    for (p, score) in scores.iter_mut().enumerate().take(input.pos + 1) {
        let k_head = &k_cache[p * input.cache_kv_width + kv_h * input.head_dim
            ..p * input.cache_kv_width + (kv_h + 1) * input.head_dim];
        decode_f16_head(k_head, k_head_f32);
        *score = dot_f32(q_head, k_head_f32) * input.scale;
    }
    softmax(&mut scores[0..=input.pos]);

    for (p, &score) in scores.iter().enumerate().take(input.pos + 1) {
        let v_head = &v_cache[p * input.cache_kv_width + kv_h * input.head_dim
            ..p * input.cache_kv_width + (kv_h + 1) * input.head_dim];
        decode_f16_head(v_head, v_head_f32);
        add_weighted_f32(out_head, v_head_f32, score);
    }
}

fn decode_f16_head(src: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    for (out, &bits) in dst.iter_mut().zip(src) {
        *out = crate::q8::fast_f16_to_f32(bits);
    }
}

fn forward_pass_inner<'a>(
    token_id: usize,
    pos: usize,
    config: &LlamaModelConfig,
    weights: &LlamaWeights,
    cache: &mut LlamaKvCache,
    ws: &'a mut LlamaWorkspace,
    options: LlamaRuntimeOptions,
) -> &'a [f32] {
    // 1. Embedding lookup
    let emb_start = token_id * config.embedding_length;
    ws.hidden
        .copy_from_slice(&weights.token_embeddings[emb_start..emb_start + config.embedding_length]);

    // 2. Transformer layers
    for layer_idx in 0..config.block_count {
        let layer = &weights.layers[layer_idx];

        // --- Attention block ---
        // Save residual
        ws.residual.copy_from_slice(&ws.hidden);

        // RMSNorm
        rms_norm(
            &mut ws.norm_x,
            &ws.hidden,
            &layer.attention_norm,
            config.rms_norm_epsilon,
        );

        // Quantize normalized activations for Q8 matmul
        quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);

        // Q, K, V Projections
        matmul_quantized(
            &mut ws.q,
            &ws.x_i8,
            &ws.x_scales,
            &layer.wq,
            config.embedding_length,
            config.embedding_length,
            options.q8_selector,
        );
        if let Some(bias) = &layer.wq_bias {
            add_bias(&mut ws.q, bias);
        }
        matmul_quantized(
            &mut ws.k,
            &ws.x_i8,
            &ws.x_scales,
            &layer.wk,
            config.kv_width,
            config.embedding_length,
            options.q8_selector,
        );
        if let Some(bias) = &layer.wk_bias {
            add_bias(&mut ws.k, bias);
        }
        matmul_quantized(
            &mut ws.v,
            &ws.x_i8,
            &ws.x_scales,
            &layer.wav,
            config.kv_width,
            config.embedding_length,
            options.q8_selector,
        );
        if let Some(bias) = &layer.wav_bias {
            add_bias(&mut ws.v, bias);
        }

        // Apply RoPE
        apply_rope(
            &mut ws.q,
            pos,
            config.attention_head_count,
            config.head_dim,
            config.rope_dimension_count,
            config.rope_freq_base,
            options.rope_scaling,
        );

        apply_rope(
            &mut ws.k,
            pos,
            config.attention_head_count_kv,
            config.head_dim,
            config.rope_dimension_count,
            config.rope_freq_base,
            options.rope_scaling,
        );

        // Store to KV Cache
        cache.store_kv(layer_idx, pos, &ws.k, &ws.v);

        // Retrieve K, V history
        let k_cache = cache.get_k_cache(layer_idx);
        let v_cache = cache.get_v_cache(layer_idx);

        // Score scale
        let scale = 1.0 / (config.head_dim as f32).sqrt();

        // Compute attention outputs per head
        ws.attn_output.fill(0.0);
        apply_attention_heads(
            &mut ws.attn_output,
            &mut ws.attn_scores,
            AttentionInput {
                q: &ws.q,
                k_cache,
                v_cache,
                pos,
                head_count: config.attention_head_count,
                kv_head_count: config.attention_head_count_kv,
                head_dim: config.head_dim,
                cache_kv_width: cache.kv_width,
                context_length: config.context_length,
                scale,
            },
            attention_head_parallel_enabled(),
        );

        // Projection O (wo)
        quantize_f32_to_q8_0(&ws.attn_output, &mut ws.x_i8, &mut ws.x_scales);
        matmul_quantized(
            &mut ws.hidden,
            &ws.x_i8,
            &ws.x_scales,
            &layer.wo,
            config.embedding_length,
            config.embedding_length,
            options.q8_selector,
        );

        // Residual addition
        for i in 0..config.embedding_length {
            ws.hidden[i] += ws.residual[i];
        }

        // --- FFN block ---
        ws.residual.copy_from_slice(&ws.hidden);

        // RMSNorm
        rms_norm(
            &mut ws.norm_x,
            &ws.hidden,
            &layer.ffn_norm,
            config.rms_norm_epsilon,
        );

        // Quantize
        quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);

        // Gate (w1) & Up (w3) matmuls
        matmul_quantized(
            &mut ws.ffn_gate,
            &ws.x_i8,
            &ws.x_scales,
            &layer.w1,
            config.feed_forward_length,
            config.embedding_length,
            options.q8_selector,
        );
        matmul_quantized(
            &mut ws.ffn_up,
            &ws.x_i8,
            &ws.x_scales,
            &layer.w3,
            config.feed_forward_length,
            config.embedding_length,
            options.q8_selector,
        );

        // Fused SiLU activation on Gate and element-wise product with Up
        fused_silu_mul(&mut ws.ffn_gate_up, &ws.ffn_gate, &ws.ffn_up);

        // Down projection (w2)
        quantize_f32_to_q8_0(&ws.ffn_gate_up, &mut ws.x_i8, &mut ws.x_scales);
        matmul_quantized(
            &mut ws.hidden,
            &ws.x_i8,
            &ws.x_scales,
            &layer.w2,
            config.embedding_length,
            config.feed_forward_length,
            options.q8_selector,
        );

        // Residual addition
        for i in 0..config.embedding_length {
            ws.hidden[i] += ws.residual[i];
        }
    }

    if !options.compute_logits {
        return &ws.logits;
    }

    // 3. Final RMSNorm
    rms_norm(
        &mut ws.norm_x,
        &ws.hidden,
        &weights.output_norm,
        config.rms_norm_epsilon,
    );

    // 4. Logits Projection
    if let Some(out_proj) = &weights.output_projection {
        quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);
        matmul_quantized(
            &mut ws.logits,
            &ws.x_i8,
            &ws.x_scales,
            out_proj,
            config.vocab_size,
            config.embedding_length,
            options.q8_selector,
        );
    } else {
        // Tied embeddings (multiply norm_x by token_embeddings transposed)
        matmul_f32(
            &mut ws.logits,
            &ws.norm_x,
            &weights.token_embeddings,
            config.vocab_size,
            config.embedding_length,
        );
    }

    &ws.logits
}

pub fn sample_logits(logits: &[f32], temperature: f32) -> usize {
    if temperature <= 0.0 {
        // Greedy sampling: argmax
        let mut max_val = logits[0];
        let mut max_idx = 0;
        for (i, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        max_idx
    } else {
        // Temperature sampling
        let mut scaled_probs = vec![0.0_f32; logits.len()];
        let mut max_logit = logits[0];
        for &val in logits.iter() {
            if val > max_logit {
                max_logit = val;
            }
        }

        let mut sum = 0.0;
        for (i, &val) in logits.iter().enumerate() {
            let p = ((val - max_logit) / temperature).exp();
            scaled_probs[i] = p;
            sum += p;
        }

        // Draw sample
        let mut r = rand_simple() * sum;
        for (i, &p) in scaled_probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
        logits.len() - 1
    }
}

// Simple deterministic pseudo-random number generator for zero-dependency sampling
static mut RNG_STATE: u64 = 0x123456789abcdef0;
fn rand_simple() -> f32 {
    unsafe {
        RNG_STATE = RNG_STATE.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((RNG_STATE >> 32) as u32) as f32;
        x / (u32::MAX as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionInput, BatchMatmulShape, KvCacheSlice, RopeScaling, add_weighted_f32,
        add_weighted_f32_scalar, apply_attention_heads, apply_rope, dot_f32, dot_f32_scalar,
        fused_silu_mul, matmul_q4_0, matmul_q4_0_batch, matmul_q4_0_swizzled_1x4, matmul_q6_k,
        matmul_q6_k_batch, matmul_q8_0, matmul_q8_0_batch, quantize_f32_to_q8_0,
        quantize_f32_to_q8_0_batch, rms_norm, rms_norm_batch, silu,
    };
    use crate::q8::{
        Q4_0Block, Q6KBlock, Q8_0Block, Q8_BLOCK_SIZE, Q8DotKernel, Q8DotKernelSelector,
        QK_K_BLOCK_SIZE, swizzle_q4_0_1x4,
    };

    fn selector(selected: Q8DotKernel) -> Q8DotKernelSelector {
        Q8DotKernelSelector {
            requested: Some(selected),
            selected,
            fallback_reason: None,
        }
    }

    fn q8_block(scale_bits: u16, seed: i16) -> Q8_0Block {
        let mut values = [0_i8; Q8_BLOCK_SIZE];
        for (idx, value) in values.iter_mut().enumerate() {
            *value = ((seed + idx as i16 * 7) % 63 - 31) as i8;
        }
        Q8_0Block::from_parts(scale_bits, values)
    }

    fn q4_block(scale_bits: u16, seed: i16) -> Q4_0Block {
        let mut values = [0_u8; Q8_BLOCK_SIZE / 2];
        for (idx, value) in values.iter_mut().enumerate() {
            let low = ((seed + idx as i16 * 3).rem_euclid(16)) as u8;
            let high = ((seed + (idx as i16 + 16) * 3).rem_euclid(16)) as u8;
            *value = low | (high << 4);
        }
        Q4_0Block::from_parts(scale_bits, values)
    }

    fn q6_k_block(scale_bits: u16, seed: i16) -> Q6KBlock {
        let mut ql = [0_u8; 128];
        let mut qh = [0_u8; 64];
        let mut scales = [0_i8; 16];
        for (idx, value) in ql.iter_mut().enumerate() {
            *value = ((seed + idx as i16 * 5).rem_euclid(256)) as u8;
        }
        for (idx, value) in qh.iter_mut().enumerate() {
            *value = ((seed + idx as i16 * 7).rem_euclid(256)) as u8;
        }
        for (idx, scale) in scales.iter_mut().enumerate() {
            *scale = ((seed + idx as i16 * 3).rem_euclid(15) - 7) as i8;
        }
        Q6KBlock::from_parts(ql, qh, scales, scale_bits)
    }

    #[test]
    fn attention_vector_helpers_match_scalar_reference() {
        let lhs: Vec<f32> = (0..130).map(|idx| idx as f32 * 0.03125 - 2.0).collect();
        let rhs: Vec<f32> = (0..130).map(|idx| 1.0 - idx as f32 * 0.015625).collect();
        let candidate = dot_f32(&lhs, &rhs);
        let expected = dot_f32_scalar(&lhs, &rhs);
        assert!(
            (candidate - expected).abs() < 1e-4,
            "candidate {candidate} expected {expected}"
        );

        let mut candidate_out: Vec<f32> = (0..130).map(|idx| idx as f32 * 0.125).collect();
        let mut expected_out = candidate_out.clone();
        add_weighted_f32(&mut candidate_out, &rhs, 0.375);
        add_weighted_f32_scalar(&mut expected_out, &rhs, 0.375);

        for (idx, (&candidate, &expected)) in candidate_out.iter().zip(&expected_out).enumerate() {
            assert!(
                (candidate - expected).abs() < 1e-6,
                "idx {idx} candidate {candidate} expected {expected}"
            );
        }
    }

    #[test]
    fn head_parallel_attention_matches_serial_attention() {
        let head_count = 4;
        let kv_head_count = 2;
        let head_dim = 8;
        let context_length = 5;
        let cache_kv_width = kv_head_count * head_dim;
        let pos = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q: Vec<f32> = (0..head_count * head_dim)
            .map(|idx| idx as f32 * 0.017 - 0.25)
            .collect();
        let k_cache: Vec<f32> = (0..context_length * cache_kv_width)
            .map(|idx| 0.5 - idx as f32 * 0.009)
            .collect();
        let v_cache: Vec<f32> = (0..context_length * cache_kv_width)
            .map(|idx| idx as f32 * 0.013 - 0.4)
            .collect();

        let mut serial_out = vec![0.0; head_count * head_dim];
        let mut serial_scores = vec![0.0; context_length];
        apply_attention_heads(
            &mut serial_out,
            &mut serial_scores,
            AttentionInput {
                q: &q,
                k_cache: KvCacheSlice::F32(&k_cache),
                v_cache: KvCacheSlice::F32(&v_cache),
                pos,
                head_count,
                kv_head_count,
                head_dim,
                cache_kv_width,
                context_length,
                scale,
            },
            false,
        );

        let mut parallel_out = vec![0.0; head_count * head_dim];
        let mut parallel_scores = vec![0.0; head_count * context_length];
        apply_attention_heads(
            &mut parallel_out,
            &mut parallel_scores,
            AttentionInput {
                q: &q,
                k_cache: KvCacheSlice::F32(&k_cache),
                v_cache: KvCacheSlice::F32(&v_cache),
                pos,
                head_count,
                kv_head_count,
                head_dim,
                cache_kv_width,
                context_length,
                scale,
            },
            true,
        );

        for (idx, (&candidate, &expected)) in parallel_out.iter().zip(&serial_out).enumerate() {
            assert!(
                (candidate - expected).abs() < 1e-6,
                "idx {idx} candidate {candidate} expected {expected}"
            );
        }
    }

    #[test]
    fn fused_silu_mul_matches_two_pass_reference() {
        let gate: Vec<f32> = (0..257).map(|idx| idx as f32 * 0.025 - 3.0).collect();
        let up: Vec<f32> = (0..257).map(|idx| 2.0 - idx as f32 * 0.011).collect();
        let mut expected_gate = gate.clone();
        let mut expected = vec![0.0; gate.len()];
        silu(&mut expected_gate);
        for idx in 0..expected.len() {
            expected[idx] = expected_gate[idx] * up[idx];
        }

        let mut candidate = vec![0.0; gate.len()];
        fused_silu_mul(&mut candidate, &gate, &up);

        for (idx, (&candidate, &expected)) in candidate.iter().zip(&expected).enumerate() {
            assert!(
                (candidate - expected).abs() < 1e-6,
                "idx {idx} candidate {candidate} expected {expected}"
            );
        }
    }

    #[test]
    fn f16_attention_matches_explicitly_decoded_cache() {
        let head_count = 4;
        let kv_head_count = 2;
        let head_dim = 8;
        let context_length = 5;
        let cache_kv_width = kv_head_count * head_dim;
        let pos = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q: Vec<f32> = (0..head_count * head_dim)
            .map(|idx| idx as f32 * 0.017 - 0.25)
            .collect();
        let k_cache_f16: Vec<u16> = (0..context_length * cache_kv_width)
            .map(|idx| crate::q8::fast_f32_to_f16(0.5 - idx as f32 * 0.009))
            .collect();
        let v_cache_f16: Vec<u16> = (0..context_length * cache_kv_width)
            .map(|idx| crate::q8::fast_f32_to_f16(idx as f32 * 0.013 - 0.4))
            .collect();
        let k_cache_decoded: Vec<f32> = k_cache_f16
            .iter()
            .map(|&bits| crate::q8::fast_f16_to_f32(bits))
            .collect();
        let v_cache_decoded: Vec<f32> = v_cache_f16
            .iter()
            .map(|&bits| crate::q8::fast_f16_to_f32(bits))
            .collect();

        let mut expected = vec![0.0; head_count * head_dim];
        let mut expected_scores = vec![0.0; context_length];
        apply_attention_heads(
            &mut expected,
            &mut expected_scores,
            AttentionInput {
                q: &q,
                k_cache: KvCacheSlice::F32(&k_cache_decoded),
                v_cache: KvCacheSlice::F32(&v_cache_decoded),
                pos,
                head_count,
                kv_head_count,
                head_dim,
                cache_kv_width,
                context_length,
                scale,
            },
            false,
        );

        let mut candidate = vec![0.0; head_count * head_dim];
        let mut candidate_scores = vec![0.0; head_count * context_length];
        apply_attention_heads(
            &mut candidate,
            &mut candidate_scores,
            AttentionInput {
                q: &q,
                k_cache: KvCacheSlice::F16(&k_cache_f16),
                v_cache: KvCacheSlice::F16(&v_cache_f16),
                pos,
                head_count,
                kv_head_count,
                head_dim,
                cache_kv_width,
                context_length,
                scale,
            },
            true,
        );

        for (idx, (&candidate, &expected)) in candidate.iter().zip(&expected).enumerate() {
            assert!(
                (candidate - expected).abs() < 1e-6,
                "idx {idx} candidate {candidate} expected {expected}"
            );
        }
    }

    #[test]
    fn matmul_q8_selected_kernels_match_scalar_reference() {
        let rows = 3;
        let cols = 64;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales = [0.25, 0.5];
        let weights = [
            q8_block(0x3800, 1),
            q8_block(0x3c00, 2),
            q8_block(0x4000, 3),
            q8_block(0x4200, 4),
            q8_block(0x4400, 5),
            q8_block(0x4600, 6),
        ];
        let mut scalar = vec![0.0; rows];
        matmul_q8_0(
            &mut scalar,
            &x_i8,
            &x_scales,
            &weights,
            rows,
            cols,
            selector(Q8DotKernel::Scalar),
        );

        for kernel in [Q8DotKernel::Neon, Q8DotKernel::Sdot] {
            let mut candidate = vec![0.0; rows];
            matmul_q8_0(
                &mut candidate,
                &x_i8,
                &x_scales,
                &weights,
                rows,
                cols,
                selector(kernel),
            );
            assert_eq!(candidate, scalar, "{kernel:?} matmul diverged");
        }
    }

    #[test]
    fn matmul_q4_matches_dequantized_reference() {
        let rows = 3;
        let cols = 64;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales = [0.25, 0.5];
        let weights = [
            q4_block(0x3800, 1),
            q4_block(0x3c00, 2),
            q4_block(0x4000, 3),
            q4_block(0x4200, 4),
            q4_block(0x4400, 5),
            q4_block(0x4600, 6),
        ];
        let mut candidate = vec![0.0; rows];
        matmul_q4_0(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            rows,
            cols,
            selector(Q8DotKernel::Scalar),
        );

        let blocks_per_row = cols / Q8_BLOCK_SIZE;
        let mut expected = vec![0.0; rows];
        for r in 0..rows {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block = &weights[r * blocks_per_row + b];
                let scale = block.scale_f32() * x_scales[b];
                let unpacked = block.unpack_values();
                for i in 0..Q8_BLOCK_SIZE {
                    sum += scale * unpacked[i] as f32 * x_i8[b * Q8_BLOCK_SIZE + i] as f32;
                }
            }
            expected[r] = sum;
        }

        for r in 0..rows {
            assert!(
                (candidate[r] - expected[r]).abs() < 1e-5,
                "row {r} candidate {} expected {}",
                candidate[r],
                expected[r]
            );
        }
    }

    #[test]
    fn matmul_q6_k_matches_dequantized_reference() {
        let rows = 2;
        let cols = QK_K_BLOCK_SIZE * 2;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales: Vec<f32> = (0..cols / Q8_BLOCK_SIZE)
            .map(|idx| 0.125 + idx as f32 * 0.03125)
            .collect();
        let weights = [
            q6_k_block(0x3800, 1),
            q6_k_block(0x3c00, 2),
            q6_k_block(0x4000, 3),
            q6_k_block(0x4200, 4),
        ];
        let mut candidate = vec![0.0; rows];
        matmul_q6_k(&mut candidate, &x_i8, &x_scales, &weights, rows, cols);

        let blocks_per_row = cols / QK_K_BLOCK_SIZE;
        let mut expected = vec![0.0; rows];
        for r in 0..rows {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block = &weights[r * blocks_per_row + b];
                let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
                block.dequantize(&mut values);
                let x_block_start = b * QK_K_BLOCK_SIZE;
                for (i, value) in values.iter().enumerate() {
                    let x_idx = x_block_start + i;
                    sum += *value * x_i8[x_idx] as f32 * x_scales[x_idx / Q8_BLOCK_SIZE];
                }
            }
            expected[r] = sum;
        }

        for r in 0..rows {
            assert!(
                (candidate[r] - expected[r]).abs() < 1e-4,
                "row {r} candidate {} expected {}",
                candidate[r],
                expected[r]
            );
        }
    }

    #[test]
    fn matmul_q4_selected_kernels_match_scalar_reference() {
        let rows = 3;
        let cols = 64;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales = [0.25, 0.5];
        let weights = [
            q4_block(0x3800, 1),
            q4_block(0x3c00, 2),
            q4_block(0x4000, 3),
            q4_block(0x4200, 4),
            q4_block(0x4400, 5),
            q4_block(0x4600, 6),
        ];
        let mut scalar = vec![0.0; rows];
        matmul_q4_0(
            &mut scalar,
            &x_i8,
            &x_scales,
            &weights,
            rows,
            cols,
            selector(Q8DotKernel::Scalar),
        );

        for kernel in [Q8DotKernel::Neon, Q8DotKernel::Sdot] {
            let mut candidate = vec![0.0; rows];
            matmul_q4_0(
                &mut candidate,
                &x_i8,
                &x_scales,
                &weights,
                rows,
                cols,
                selector(kernel),
            );
            assert_eq!(candidate, scalar, "{kernel:?} Q4 matmul diverged");
        }
    }

    #[test]
    fn matmul_q4_swizzled_1x4_matches_row_major_reference() {
        let rows = 4;
        let cols = 64;
        let x_i8: Vec<i8> = (0..cols)
            .map(|idx| ((idx as i16 * 11).rem_euclid(83) - 41) as i8)
            .collect();
        let x_scales = [0.125, 0.375];
        let weights: Vec<Q4_0Block> = (0..rows * (cols / Q8_BLOCK_SIZE))
            .map(|idx| q4_block(0x3800 + idx as u16, idx as i16 + 1))
            .collect();
        let swizzled = swizzle_q4_0_1x4(&weights, rows, cols / Q8_BLOCK_SIZE);

        let mut expected = vec![0.0; rows];
        matmul_q4_0(
            &mut expected,
            &x_i8,
            &x_scales,
            &weights,
            rows,
            cols,
            selector(Q8DotKernel::Scalar),
        );

        let mut candidate = vec![0.0; rows];
        let matrix = crate::model::Q4_0Swizzled1x4Matrix {
            swizzled_1x4: swizzled.clone(),
            page_aligned_1x4: None,
            rows,
            cols,
        };
        matmul_q4_0_swizzled_1x4(
            &mut candidate,
            &x_i8,
            &x_scales,
            &matrix,
            cols,
            selector(Q8DotKernel::Sdot),
        );

        assert_eq!(candidate, expected);

        let mut aligned_candidate = vec![0.0; rows];
        let aligned_matrix = crate::model::Q4_0Swizzled1x4Matrix {
            page_aligned_1x4: Some(crate::model::PageAlignedQ4_0Swizzled1x4::from_swizzled(
                &swizzled,
                rows,
                cols / Q8_BLOCK_SIZE,
            )),
            swizzled_1x4: swizzled,
            rows,
            cols,
        };
        matmul_q4_0_swizzled_1x4(
            &mut aligned_candidate,
            &x_i8,
            &x_scales,
            &aligned_matrix,
            cols,
            selector(Q8DotKernel::Sdot),
        );

        assert_eq!(aligned_candidate, expected);
    }

    #[test]
    fn rms_norm_batch_matches_single_token_reference() {
        let batch_size = 3;
        let dim = 64;
        let input: Vec<f32> = (0..batch_size * dim)
            .map(|idx| idx as f32 * 0.013 - 1.75)
            .collect();
        let weight: Vec<f32> = (0..dim).map(|idx| 0.5 + idx as f32 * 0.007).collect();
        let mut candidate = vec![0.0; batch_size * dim];
        let mut expected = vec![0.0; batch_size * dim];

        rms_norm_batch(&mut candidate, &input, &weight, 1e-5, batch_size, dim);
        for token_idx in 0..batch_size {
            let start = token_idx * dim;
            rms_norm(
                &mut expected[start..start + dim],
                &input[start..start + dim],
                &weight,
                1e-5,
            );
        }

        for (idx, (&candidate, &expected)) in candidate.iter().zip(&expected).enumerate() {
            assert!(
                (candidate - expected).abs() < 1e-6,
                "idx {idx} candidate {candidate} expected {expected}"
            );
        }
    }

    #[test]
    fn quantize_batch_matches_single_token_reference() {
        let batch_size = 3;
        let dim = 64;
        let input: Vec<f32> = (0..batch_size * dim)
            .map(|idx| (idx as f32 * 0.03125).sin() * 3.0)
            .collect();
        let mut candidate_i8 = vec![0; batch_size * dim];
        let mut expected_i8 = vec![0; batch_size * dim];
        let mut candidate_scales = vec![0.0; batch_size * (dim / Q8_BLOCK_SIZE)];
        let mut expected_scales = vec![0.0; batch_size * (dim / Q8_BLOCK_SIZE)];

        quantize_f32_to_q8_0_batch(
            &input,
            &mut candidate_i8,
            &mut candidate_scales,
            batch_size,
            dim,
        );
        for token_idx in 0..batch_size {
            let value_start = token_idx * dim;
            let scale_start = token_idx * (dim / Q8_BLOCK_SIZE);
            quantize_f32_to_q8_0(
                &input[value_start..value_start + dim],
                &mut expected_i8[value_start..value_start + dim],
                &mut expected_scales[scale_start..scale_start + (dim / Q8_BLOCK_SIZE)],
            );
        }

        assert_eq!(candidate_i8, expected_i8);
        assert_eq!(candidate_scales, expected_scales);
    }

    #[test]
    fn matmul_q8_batch_matches_single_token_reference() {
        let batch_size = 3;
        let rows = 3;
        let cols = 64;
        let x_i8: Vec<i8> = (0..batch_size * cols)
            .map(|idx| ((idx * 5) % 61) as i8 - 30)
            .collect();
        let x_scales: Vec<f32> = (0..batch_size * (cols / Q8_BLOCK_SIZE))
            .map(|idx| 0.125 + idx as f32 * 0.03125)
            .collect();
        let weights: Vec<Q8_0Block> = (0..rows * (cols / Q8_BLOCK_SIZE))
            .map(|idx| q8_block(0x3800 + idx as u16, idx as i16 + 1))
            .collect();

        let mut candidate = vec![0.0; batch_size * rows];
        matmul_q8_0_batch(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            BatchMatmulShape {
                batch_size,
                rows,
                cols,
            },
            selector(Q8DotKernel::Scalar),
        );

        for token_idx in 0..batch_size {
            let mut expected = vec![0.0; rows];
            let value_start = token_idx * cols;
            let scale_start = token_idx * (cols / Q8_BLOCK_SIZE);
            matmul_q8_0(
                &mut expected,
                &x_i8[value_start..value_start + cols],
                &x_scales[scale_start..scale_start + (cols / Q8_BLOCK_SIZE)],
                &weights,
                rows,
                cols,
                selector(Q8DotKernel::Scalar),
            );
            assert_eq!(
                &candidate[token_idx * rows..(token_idx + 1) * rows],
                expected.as_slice()
            );
        }
    }

    #[test]
    fn matmul_q4_batch_matches_single_token_reference() {
        let batch_size = 3;
        let rows = 4;
        let cols = 64;
        let x_i8: Vec<i8> = (0..batch_size * cols)
            .map(|idx| ((idx as i16 * 11).rem_euclid(83) - 41) as i8)
            .collect();
        let x_scales: Vec<f32> = (0..batch_size * (cols / Q8_BLOCK_SIZE))
            .map(|idx| 0.125 + idx as f32 * 0.0625)
            .collect();
        let weights: Vec<Q4_0Block> = (0..rows * (cols / Q8_BLOCK_SIZE))
            .map(|idx| q4_block(0x3800 + idx as u16, idx as i16 + 1))
            .collect();

        let mut candidate = vec![0.0; batch_size * rows];
        matmul_q4_0_batch(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            BatchMatmulShape {
                batch_size,
                rows,
                cols,
            },
            selector(Q8DotKernel::Scalar),
        );

        for token_idx in 0..batch_size {
            let mut expected = vec![0.0; rows];
            let value_start = token_idx * cols;
            let scale_start = token_idx * (cols / Q8_BLOCK_SIZE);
            matmul_q4_0(
                &mut expected,
                &x_i8[value_start..value_start + cols],
                &x_scales[scale_start..scale_start + (cols / Q8_BLOCK_SIZE)],
                &weights,
                rows,
                cols,
                selector(Q8DotKernel::Scalar),
            );
            assert_eq!(
                &candidate[token_idx * rows..(token_idx + 1) * rows],
                expected.as_slice()
            );
        }
    }

    #[test]
    fn matmul_q6_k_batch_matches_single_token_reference() {
        let batch_size = 3;
        let rows = 2;
        let cols = QK_K_BLOCK_SIZE * 2;
        let x_i8: Vec<i8> = (0..batch_size * cols)
            .map(|idx| ((idx * 5) % 61) as i8 - 30)
            .collect();
        let x_scales: Vec<f32> = (0..batch_size * (cols / Q8_BLOCK_SIZE))
            .map(|idx| 0.125 + idx as f32 * 0.03125)
            .collect();
        let weights: Vec<Q6KBlock> = (0..rows * (cols / QK_K_BLOCK_SIZE))
            .map(|idx| q6_k_block(0x3800 + idx as u16, idx as i16 + 1))
            .collect();

        let mut candidate = vec![0.0; batch_size * rows];
        matmul_q6_k_batch(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            BatchMatmulShape {
                batch_size,
                rows,
                cols,
            },
        );

        for token_idx in 0..batch_size {
            let mut expected = vec![0.0; rows];
            let value_start = token_idx * cols;
            let scale_start = token_idx * (cols / Q8_BLOCK_SIZE);
            matmul_q6_k(
                &mut expected,
                &x_i8[value_start..value_start + cols],
                &x_scales[scale_start..scale_start + (cols / Q8_BLOCK_SIZE)],
                &weights,
                rows,
                cols,
            );
            assert_eq!(
                &candidate[token_idx * rows..(token_idx + 1) * rows],
                expected.as_slice()
            );
        }
    }

    #[test]
    fn apply_rope_respects_partial_rope_dimension_count() {
        let mut data = vec![1.0, 0.0, 10.0, 20.0];
        apply_rope(&mut data, 1, 1, 4, 2, 10000.0, RopeScaling::default());

        assert!((data[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((data[1] - 1.0_f32.sin()).abs() < 1e-6);
        assert_eq!(data[2], 10.0);
        assert_eq!(data[3], 20.0);
    }
}
