use crate::model::{LlamaModelConfig, LlamaWeights, QuantizedMatrix};
use crate::q8::{
    Q4_0Block, Q6KBlock, Q8_0Block, Q8DotKernel, Q8DotKernelSelector, QK_K_BLOCK_SIZE,
};
use rayon::prelude::*;
use std::{env, sync::OnceLock};

pub const MATMUL_MIN_ROWS_ENV: &str = "NANOCAMELID_MATMUL_MIN_ROWS";
pub const Q4_1X4_SDOT_ENV: &str = "NANOCAMELID_Q4_1X4_SDOT";
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

pub struct LlamaKvCache {
    pub k_cache: Vec<f32>, // block_count * context_length * kv_width
    pub v_cache: Vec<f32>, // block_count * context_length * kv_width
    pub kv_width: usize,
    pub max_seq_len: usize,
}

impl LlamaKvCache {
    pub fn new(block_count: usize, max_seq_len: usize, kv_width: usize) -> Self {
        let size = block_count * max_seq_len * kv_width;
        Self {
            k_cache: vec![0.0; size],
            v_cache: vec![0.0; size],
            kv_width,
            max_seq_len,
        }
    }

    pub fn store_kv(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let layer_offset = layer * self.max_seq_len * self.kv_width;
        let pos_offset = pos * self.kv_width;
        let start = layer_offset + pos_offset;

        self.k_cache[start..start + self.kv_width].copy_from_slice(k);
        self.v_cache[start..start + self.kv_width].copy_from_slice(v);
    }

    pub fn get_k_cache(&self, layer: usize) -> &[f32] {
        let layer_offset = layer * self.max_seq_len * self.kv_width;
        &self.k_cache[layer_offset..layer_offset + self.max_seq_len * self.kv_width]
    }

    pub fn get_v_cache(&self, layer: usize) -> &[f32] {
        let layer_offset = layer * self.max_seq_len * self.kv_width;
        &self.v_cache[layer_offset..layer_offset + self.max_seq_len * self.kv_width]
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

impl LlamaWorkspace {
    pub fn new(config: &LlamaModelConfig) -> Self {
        let max_size = config
            .vocab_size
            .max(config.feed_forward_length)
            .max(config.embedding_length);
        Self {
            hidden: vec![0.0; config.embedding_length],
            residual: vec![0.0; config.embedding_length],
            norm_x: vec![0.0; config.embedding_length],
            q: vec![0.0; config.embedding_length],
            k: vec![0.0; config.kv_width],
            v: vec![0.0; config.kv_width],
            attn_output: vec![0.0; config.embedding_length],
            attn_scores: vec![0.0; config.context_length],
            ffn_gate: vec![0.0; config.feed_forward_length],
            ffn_up: vec![0.0; config.feed_forward_length],
            ffn_gate_up: vec![0.0; config.feed_forward_length],
            x_i8: vec![0; max_size],
            x_scales: vec![0.0; max_size / 32 + 1],
            logits: vec![0.0; config.vocab_size],
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
            if selector.selected == Q8DotKernel::Sdot
                && q4_1x4_sdot_enabled()
                && matrix.rows == rows
                && matrix.cols == cols
            {
                matmul_q4_0_sdot_1x4_swizzled(out, x_i8, x_scales, &matrix.swizzled_1x4, cols);
            } else {
                matmul_q4_0(out, x_i8, x_scales, &matrix.row_major, rows, cols, selector);
            }
        }
        QuantizedMatrix::Q6K(blocks) => matmul_q6_k(out, x_i8, x_scales, blocks, rows, cols),
    }
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

fn matmul_q4_0_sdot_1x4_swizzled(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_0Block],
    cols: usize,
) {
    let blocks_per_row = cols / 32;
    debug_assert_eq!(w.len(), out.len() * blocks_per_row);
    debug_assert!(out.len().is_multiple_of(4));

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
        let dots = dot_q4_0_q8_0_1x4_sdot([row0, row1, row2, row3], x_block_vals);
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

pub fn matmul_q6_k(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q6KBlock],
    _rows: usize,
    cols: usize,
) {
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    for_each_matmul_row(out, |r, out_val| {
        let mut sum = 0.0_f32;
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for (block_idx, w_block) in w_row.iter().enumerate() {
            let x_block_start = block_idx * QK_K_BLOCK_SIZE;
            let x_scale_start = x_block_start / 32;
            sum += w_block.dot_q8_scaled(
                &x_i8[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                &x_scales[x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / 32)],
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
        let kv_mul = config.attention_head_count / config.attention_head_count_kv;

        for h in 0..config.attention_head_count {
            let kv_h = h / kv_mul;
            let q_head = &ws.q[h * config.head_dim..(h + 1) * config.head_dim];

            // Compute scores against all positions p = 0..=pos
            for p in 0..=pos {
                let k_head = &k_cache[p * cache.kv_width + kv_h * config.head_dim
                    ..p * cache.kv_width + (kv_h + 1) * config.head_dim];

                ws.attn_scores[p] = dot_f32(q_head, k_head) * scale;
            }

            // Softmax
            softmax(&mut ws.attn_scores[0..=pos]);

            // Weighted sum of V
            let out_head = &mut ws.attn_output[h * config.head_dim..(h + 1) * config.head_dim];
            for p in 0..=pos {
                let v_head = &v_cache[p * cache.kv_width + kv_h * config.head_dim
                    ..p * cache.kv_width + (kv_h + 1) * config.head_dim];
                let weight = ws.attn_scores[p];
                add_weighted_f32(out_head, v_head, weight);
            }
        }

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

        // SiLU activation on Gate
        silu(&mut ws.ffn_gate);

        // Element-wise product of Gate and Up
        for i in 0..config.feed_forward_length {
            ws.ffn_gate_up[i] = ws.ffn_gate[i] * ws.ffn_up[i];
        }

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
        RopeScaling, add_weighted_f32, add_weighted_f32_scalar, apply_rope, dot_f32,
        dot_f32_scalar, matmul_q4_0, matmul_q4_0_sdot_1x4_swizzled, matmul_q6_k, matmul_q8_0,
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
        matmul_q4_0_sdot_1x4_swizzled(&mut candidate, &x_i8, &x_scales, &swizzled, cols);

        assert_eq!(candidate, expected);
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
