use std::env;
use std::time::{Duration, Instant};
use rayon::prelude::*;

pub const Q4_BLOCK_SIZE: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 2 + 16; // 18 bytes
pub const Q4_DOT_KERNEL_ENV: &str = "NANOCAMELID_Q4_DOT_KERNEL";
pub const Q4_SDOT_CANDIDATE_ENV: &str = "NANOCAMELID_Q4_DOT_SDOT";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q4DotKernel {
    Scalar,
    Neon,
    Sdot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q4DotKernelSelector {
    pub requested: Option<Q4DotKernel>,
    pub selected: Q4DotKernel,
    pub fallback_reason: Option<&'static str>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Q4_0Block {
    pub scale_bits: u16, // f16 scale
    pub qs: [u8; 16],    // 32 quantized 4-bit values (nibbles)
}

impl Q4_0Block {
    pub fn from_bytes(bytes: &[u8; Q4_0_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let mut qs = [0_u8; 16];
        qs.copy_from_slice(&bytes[2..18]);
        Self { scale_bits, qs }
    }

    pub fn scale_f32(&self) -> f32 {
        f16_bits_to_f32(self.scale_bits)
    }

    pub fn dequantize(&self, out: &mut [f32; 32]) {
        let scale = self.scale_f32();
        for j in 0..16 {
            let x0 = (self.qs[j] & 0x0F) as i16 - 8;
            let x1 = (self.qs[j] >> 4) as i16 - 8;
            out[j] = x0 as f32 * scale;
            out[j + 16] = x1 as f32 * scale;
        }
    }
}

pub fn decode_q4_0_blocks(bytes: &[u8]) -> Result<Vec<Q4_0Block>, String> {
    if bytes.len() % Q4_0_BLOCK_BYTES != 0 {
        return Err(format!(
            "Q4_0 byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q4_0_BLOCK_BYTES
        ));
    }

    Ok(bytes
        .chunks_exact(Q4_0_BLOCK_BYTES)
        .map(|chunk| {
            let bytes: &[u8; Q4_0_BLOCK_BYTES] = chunk
                .try_into()
                .expect("chunks_exact guarantees Q4_0 block length");
            Q4_0Block::from_bytes(bytes)
        })
        .collect())
}

pub fn dot_q4_0_q8_0_scalar(w: &Q4_0Block, x: &[i8; 32]) -> i32 {
    let mut sum = 0_i32;
    for j in 0..16 {
        let v0 = (w.qs[j] & 0x0F) as i32 - 8;
        let v1 = (w.qs[j] >> 4) as i32 - 8;
        sum += v0 * x[j] as i32 + v1 * x[j + 16] as i32;
    }
    sum
}

pub fn dot_q4_0_q8_0_neon(w: &Q4_0Block, x: &[i8; 32]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { dot_q4_0_q8_0_neon_aarch64(w, x) };
        }
    }
    dot_q4_0_q8_0_scalar(w, x)
}

pub fn dot_q4_0_q8_0_sdot(w: &Q4_0Block, x: &[i8; 32]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { dot_q4_0_q8_0_sdot_aarch64(w, x) };
        }
    }
    dot_q4_0_q8_0_neon(w, x)
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_q4_0_q8_0_neon_aarch64(w: &Q4_0Block, x_i8: &[i8; 32]) -> i32 {
    use std::arch::aarch64::{
        vld1q_u8, vld1q_s8, vandq_u8, vshrq_n_u8, vdupq_n_u8, vdupq_n_s8, vsubq_s8,
        vreinterpretq_s8_u8, vmull_s8, vget_low_s8, vget_high_s8, vpaddlq_s16, vaddq_s32, vaddvq_s32
    };

    unsafe {
        let qs_vec = vld1q_u8(w.qs.as_ptr());
        let low_mask = vdupq_n_u8(0x0F);
        let low_nibbles = vandq_u8(qs_vec, low_mask);
        let high_nibbles = vshrq_n_u8(qs_vec, 4);

        let offset = vdupq_n_s8(8);
        let w0 = vsubq_s8(vreinterpretq_s8_u8(low_nibbles), offset);
        let w1 = vsubq_s8(vreinterpretq_s8_u8(high_nibbles), offset);

        let x0 = vld1q_s8(x_i8.as_ptr());
        let x1 = vld1q_s8(x_i8.as_ptr().add(16));

        let prod0_low = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
        let prod0_high = vmull_s8(vget_high_s8(w0), vget_high_s8(x0));
        let prod1_low = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
        let prod1_high = vmull_s8(vget_high_s8(w1), vget_high_s8(x1));

        let sum0 = vaddq_s32(vpaddlq_s16(prod0_low), vpaddlq_s16(prod0_high));
        let sum1 = vaddq_s32(vpaddlq_s16(prod1_low), vpaddlq_s16(prod1_high));

        vaddvq_s32(vaddq_s32(sum0, sum1))
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_q4_0_q8_0_sdot_aarch64(w: &Q4_0Block, x_i8: &[i8; 32]) -> i32 {
    use std::arch::aarch64::{
        vld1q_u8, vld1q_s8, vandq_u8, vshrq_n_u8, vdupq_n_u8, vdupq_n_s8, vsubq_s8,
        vreinterpretq_s8_u8, vdupq_n_s32, vaddvq_s32
    };
    use std::arch::asm;

    unsafe {
        let qs_vec = vld1q_u8(w.qs.as_ptr());
        let low_mask = vdupq_n_u8(0x0F);
        let low_nibbles = vandq_u8(qs_vec, low_mask);
        let high_nibbles = vshrq_n_u8(qs_vec, 4);

        let offset = vdupq_n_s8(8);
        let w0 = vsubq_s8(vreinterpretq_s8_u8(low_nibbles), offset);
        let w1 = vsubq_s8(vreinterpretq_s8_u8(high_nibbles), offset);

        let x0 = vld1q_s8(x_i8.as_ptr());
        let x1 = vld1q_s8(x_i8.as_ptr().add(16));

        let mut acc = vdupq_n_s32(0);

        asm!(
            ".arch_extension dotprod",
            "sdot {acc:v}.4s, {w0:v}.16b, {x0:v}.16b",
            "sdot {acc:v}.4s, {w1:v}.16b, {x1:v}.16b",
            acc = inout(vreg) acc,
            w0 = in(vreg) w0,
            x0 = in(vreg) x0,
            w1 = in(vreg) w1,
            x1 = in(vreg) x1,
            options(nostack, preserves_flags),
        );

        vaddvq_s32(acc)
    }
}

pub fn matmul_q4_0(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_0Block],
    _rows: usize,
    cols: usize,
    selector: Q4DotKernelSelector,
) {
    let blocks_per_row = cols / 32;
    match selector.selected {
        Q4DotKernel::Scalar => {
            out.par_iter_mut().enumerate().for_each(|(r, out_val)| {
                let mut sum = 0.0_f32;
                let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
                for b in 0..blocks_per_row {
                    let w_block = &w_row[b];
                    let x_block_vals: &[i8; 32] = (&x_i8[b * 32..(b + 1) * 32]).try_into().unwrap();
                    let dot_val = dot_q4_0_q8_0_scalar(w_block, x_block_vals);
                    sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
                }
                *out_val = sum;
            });
        }
        Q4DotKernel::Neon => {
            out.par_iter_mut().enumerate().for_each(|(r, out_val)| {
                let mut sum = 0.0_f32;
                let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
                for b in 0..blocks_per_row {
                    let w_block = &w_row[b];
                    let x_block_vals: &[i8; 32] = (&x_i8[b * 32..(b + 1) * 32]).try_into().unwrap();
                    let dot_val = dot_q4_0_q8_0_neon(w_block, x_block_vals);
                    sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
                }
                *out_val = sum;
            });
        }
        Q4DotKernel::Sdot => {
            out.par_iter_mut().enumerate().for_each(|(r, out_val)| {
                let mut sum = 0.0_f32;
                let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
                for b in 0..blocks_per_row {
                    let w_block = &w_row[b];
                    let x_block_vals: &[i8; 32] = (&x_i8[b * 32..(b + 1) * 32]).try_into().unwrap();
                    let dot_val = dot_q4_0_q8_0_sdot(w_block, x_block_vals);
                    sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
                }
                *out_val = sum;
            });
        }
    }
}

impl Q4DotKernel {
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
            Self::Sdot => "sdot",
        }
    }
}

impl Q4DotKernelSelector {
    pub fn from_env() -> Self {
        let requested = env::var(Q4_DOT_KERNEL_ENV)
            .ok()
            .as_deref()
            .and_then(parse_requested_kernel);

        let neon = neon_available();
        let dotprod = dotprod_available();
        let sdot_requested = sdot_candidate_requested();

        Self::for_request(requested, neon, dotprod, sdot_requested)
    }

    /// Like [`from_env`], but auto-selects the fastest available kernel (SDOT > NEON >
    /// scalar) when none is requested. All kernels are bit-identical, so output is
    /// unchanged. An explicit `NANOCAMELID_Q4_DOT_KERNEL` request takes precedence.
    pub fn from_env_or_auto() -> Self {
        let requested = env::var(Q4_DOT_KERNEL_ENV)
            .ok()
            .as_deref()
            .and_then(parse_requested_kernel);

        let neon = neon_available();
        let dotprod = dotprod_available();

        if requested.is_none() {
            let (selected, reason) = if dotprod {
                (Q4DotKernel::Sdot, "auto_sdot")
            } else if neon {
                (Q4DotKernel::Neon, "auto_neon")
            } else {
                (Q4DotKernel::Scalar, "auto_scalar")
            };
            return Self {
                requested: None,
                selected,
                fallback_reason: Some(reason),
            };
        }

        Self::for_request(requested, neon, dotprod, sdot_candidate_requested())
    }

    fn for_request(
        requested: Option<Q4DotKernel>,
        neon: bool,
        dotprod: bool,
        sdot_requested: bool,
    ) -> Self {
        match requested {
            None | Some(Q4DotKernel::Scalar) => Self {
                requested,
                selected: Q4DotKernel::Scalar,
                fallback_reason: None,
            },
            Some(Q4DotKernel::Neon) if neon => Self {
                requested,
                selected: Q4DotKernel::Neon,
                fallback_reason: None,
            },
            Some(Q4DotKernel::Neon) => Self {
                requested,
                selected: Q4DotKernel::Scalar,
                fallback_reason: Some("neon_unavailable"),
            },
            Some(Q4DotKernel::Sdot) if !sdot_requested => Self {
                requested,
                selected: Q4DotKernel::Scalar,
                fallback_reason: Some("sdot_candidate_not_enabled"),
            },
            Some(Q4DotKernel::Sdot) if dotprod => Self {
                requested,
                selected: Q4DotKernel::Sdot,
                fallback_reason: None,
            },
            Some(Q4DotKernel::Sdot) => Self {
                requested,
                selected: Q4DotKernel::Scalar,
                fallback_reason: Some("dotprod_unavailable"),
            },
        }
    }
}

fn parse_requested_kernel(value: &str) -> Option<Q4DotKernel> {
    match value {
        "scalar" | "SCALAR" => Some(Q4DotKernel::Scalar),
        "neon" | "NEON" => Some(Q4DotKernel::Neon),
        "sdot" | "SDOT" => Some(Q4DotKernel::Sdot),
        _ => None,
    }
}

pub fn neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

pub fn dotprod_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

pub fn sdot_candidate_requested() -> bool {
    env::var(Q4_SDOT_CANDIDATE_ENV)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

pub fn sdot_candidate_enabled() -> bool {
    sdot_candidate_requested() && dotprod_available()
}

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
    use super::*;

    #[test]
    fn test_q4_scalar_dot_product() {
        let block = Q4_0Block {
            scale_bits: 0x3c00, // f16(1.0)
            qs: [0x55; 16],     // all nibbles are 5
        };
        // 5 - 8 = -3
        let x = [2_i8; 32];
        let res = dot_q4_0_q8_0_scalar(&block, &x);
        // sum over 32 elements: -3 * 2 = -6, -6 * 32 = -192
        assert_eq!(res, -192);
    }

    #[test]
    fn test_q4_neon_matches_scalar() {
        let block = Q4_0Block {
            scale_bits: 0x3c00,
            qs: [0x73, 0x1a, 0xbf, 0x48, 0x92, 0xc0, 0x5e, 0x36,
                 0x81, 0x0a, 0xfe, 0x57, 0xd3, 0x64, 0xb9, 0x2e],
        };
        let mut x = [0_i8; 32];
        for i in 0..32 {
            x[i] = (i as i32 * 3 - 25) as i8;
        }
        let res_scalar = dot_q4_0_q8_0_scalar(&block, &x);
        let res_neon = dot_q4_0_q8_0_neon(&block, &x);
        assert_eq!(res_neon, res_scalar);
    }

    #[test]
    fn test_q4_sdot_matches_scalar() {
        let block = Q4_0Block {
            scale_bits: 0x3c00,
            qs: [0x73, 0x1a, 0xbf, 0x48, 0x92, 0xc0, 0x5e, 0x36,
                 0x81, 0x0a, 0xfe, 0x57, 0xd3, 0x64, 0xb9, 0x2e],
        };
        let mut x = [0_i8; 32];
        for i in 0..32 {
            x[i] = (i as i32 * 7 - 45) as i8;
        }
        let res_scalar = dot_q4_0_q8_0_scalar(&block, &x);
        let res_sdot = dot_q4_0_q8_0_sdot(&block, &x);
        assert_eq!(res_sdot, res_scalar);
    }
}

#[derive(Debug)]
pub struct Q4DotBenchmarkReport {
    pub iterations: usize,
    pub runs: usize,
    pub blocks_per_iteration: usize,
    pub elements_per_iteration: usize,
    pub kernel_selector: Q4DotKernelSelector,
    pub selected: Q4DotTimingSummary,
    pub scalar: Q4DotTimingSummary,
    pub neon: Option<Q4DotTimingSummary>,
    pub sdot: Option<Q4DotTimingSummary>,
}

#[derive(Debug, Clone)]
pub struct Q4DotTimingSummary {
    pub checksum: i64,
    pub elapsed_runs: Vec<Duration>,
}

impl Q4DotTimingSummary {
    pub fn total_elapsed(&self) -> Duration {
        self.elapsed_runs.iter().sum()
    }

    pub fn min_elapsed(&self) -> Duration {
        self.elapsed_runs.iter().copied().min().unwrap_or_default()
    }

    pub fn median_elapsed(&self) -> Duration {
        let mut runs = self.elapsed_runs.clone();
        runs.sort_unstable();
        runs.get(runs.len() / 2).copied().unwrap_or_default()
    }
}

impl Q4DotBenchmarkReport {
    pub fn scalar_min_ns_per_block(&self) -> f64 {
        ns_per_block(self.scalar.min_elapsed(), self.iterations, self.blocks_per_iteration)
    }

    pub fn scalar_median_ns_per_block(&self) -> f64 {
        ns_per_block(self.scalar.median_elapsed(), self.iterations, self.blocks_per_iteration)
    }

    pub fn neon_min_ns_per_block(&self) -> Option<f64> {
        self.neon.as_ref().map(|n| ns_per_block(n.min_elapsed(), self.iterations, self.blocks_per_iteration))
    }

    pub fn neon_median_ns_per_block(&self) -> Option<f64> {
        self.neon.as_ref().map(|n| ns_per_block(n.median_elapsed(), self.iterations, self.blocks_per_iteration))
    }

    pub fn sdot_min_ns_per_block(&self) -> Option<f64> {
        self.sdot.as_ref().map(|s| ns_per_block(s.min_elapsed(), self.iterations, self.blocks_per_iteration))
    }

    pub fn sdot_median_ns_per_block(&self) -> Option<f64> {
        self.sdot.as_ref().map(|s| ns_per_block(s.median_elapsed(), self.iterations, self.blocks_per_iteration))
    }

    pub fn neon_min_speedup(&self) -> Option<f64> {
        self.neon_min_ns_per_block().map(|neon_ns| self.scalar_min_ns_per_block() / neon_ns)
    }

    pub fn neon_median_speedup(&self) -> Option<f64> {
        self.neon_median_ns_per_block().map(|neon_ns| self.scalar_median_ns_per_block() / neon_ns)
    }

    pub fn sdot_min_speedup(&self) -> Option<f64> {
        self.sdot_min_ns_per_block().map(|sdot_ns| self.scalar_min_ns_per_block() / sdot_ns)
    }

    pub fn sdot_median_speedup(&self) -> Option<f64> {
        self.sdot_median_ns_per_block().map(|sdot_ns| self.scalar_median_ns_per_block() / sdot_ns)
    }

    pub fn sdot_vs_neon_min_speedup(&self) -> Option<f64> {
        self.neon_min_ns_per_block().zip(self.sdot_min_ns_per_block()).map(|(n, s)| n / s)
    }

    pub fn sdot_vs_neon_median_speedup(&self) -> Option<f64> {
        self.neon_median_ns_per_block().zip(self.sdot_median_ns_per_block()).map(|(n, s)| n / s)
    }
}

pub fn bench_dot_runs(iterations: usize, runs: usize) -> Q4DotBenchmarkReport {
    let num_blocks = 1024;
    let mut w = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let mut qs = [0_u8; 16];
        for j in 0..16 {
            qs[j] = ((i * 17 + j * 7) % 256) as u8;
        }
        w.push(Q4_0Block {
            scale_bits: 0x3c00, // 1.0
            qs,
        });
    }

    let mut x = vec![0_i8; num_blocks * 32];
    for i in 0..num_blocks * 32 {
        x[i] = ((i * 31 + 13) % 127) as i8 - 63;
    }

    let selector = Q4DotKernelSelector::from_env();

    let time_kernel = |kernel_fn: unsafe fn(&Q4_0Block, &[i8; 32]) -> i32| {
        let mut elapsed_runs = Vec::with_capacity(runs);
        let mut final_checksum = 0_i64;
        for _ in 0..runs {
            let start = Instant::now();
            let mut checksum = 0_i64;
            for _ in 0..iterations {
                let mut sum = 0_i32;
                for b in 0..num_blocks {
                    let w_block = &w[b];
                    let x_block: &[i8; 32] = (&x[b * 32..(b + 1) * 32]).try_into().unwrap();
                    sum = sum.wrapping_add(unsafe { kernel_fn(w_block, x_block) });
                }
                checksum = checksum.wrapping_add(sum as i64);
            }
            elapsed_runs.push(start.elapsed());
            final_checksum = checksum;
        }
        Q4DotTimingSummary {
            checksum: final_checksum,
            elapsed_runs,
        }
    };

    // Cast the safe wrapper functions as unsafe to match closure type signature
    let scalar_fn: unsafe fn(&Q4_0Block, &[i8; 32]) -> i32 = |w, x| dot_q4_0_q8_0_scalar(w, x);
    let neon_fn: unsafe fn(&Q4_0Block, &[i8; 32]) -> i32 = |w, x| dot_q4_0_q8_0_neon(w, x);
    let sdot_fn: unsafe fn(&Q4_0Block, &[i8; 32]) -> i32 = |w, x| dot_q4_0_q8_0_sdot(w, x);

    let scalar = time_kernel(scalar_fn);
    let selected = match selector.selected {
        Q4DotKernel::Scalar => scalar.clone(),
        Q4DotKernel::Neon => time_kernel(neon_fn),
        Q4DotKernel::Sdot => time_kernel(sdot_fn),
    };

    let neon = neon_available().then(|| time_kernel(neon_fn));
    let sdot = dotprod_available().then(|| time_kernel(sdot_fn));

    Q4DotBenchmarkReport {
        iterations,
        runs,
        blocks_per_iteration: num_blocks,
        elements_per_iteration: num_blocks * 32,
        kernel_selector: selector,
        selected,
        scalar,
        neon,
        sdot,
    }
}

fn ns_per_block(elapsed: Duration, iterations: usize, blocks_per_iteration: usize) -> f64 {
    elapsed.as_secs_f64() * 1_000_000_000.0 / (iterations * blocks_per_iteration) as f64
}

