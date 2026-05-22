use std::{
    hint::black_box,
    time::{Duration, Instant},
};

pub const DEFAULT_DOT_BENCH_ITERATIONS: usize = 2_000;

const Q8_BLOCK_SIZE: usize = 32;
const BENCH_BLOCKS: usize = 1_024;
const BENCH_ELEMENTS: usize = Q8_BLOCK_SIZE * BENCH_BLOCKS;

#[derive(Debug)]
pub struct DotBenchmarkReport {
    pub iterations: usize,
    pub blocks_per_iteration: usize,
    pub elements_per_iteration: usize,
    pub scalar: TimedDot,
    pub neon: Option<TimedDot>,
}

#[derive(Debug)]
pub struct TimedDot {
    pub checksum: i64,
    pub elapsed: Duration,
}

impl DotBenchmarkReport {
    pub fn scalar_ns_per_block(&self) -> f64 {
        ns_per_block(
            self.scalar.elapsed,
            self.iterations,
            self.blocks_per_iteration,
        )
    }

    pub fn neon_ns_per_block(&self) -> Option<f64> {
        self.neon
            .as_ref()
            .map(|neon| ns_per_block(neon.elapsed, self.iterations, self.blocks_per_iteration))
    }

    pub fn neon_speedup(&self) -> Option<f64> {
        self.neon_ns_per_block()
            .map(|neon_ns| self.scalar_ns_per_block() / neon_ns)
    }
}

pub fn bench_dot(iterations: usize) -> DotBenchmarkReport {
    let lhs = deterministic_q8_values(BENCH_ELEMENTS, 17);
    let rhs = deterministic_q8_values(BENCH_ELEMENTS, 91);

    let scalar = time_dot(iterations, || {
        dot_i8_scalar(black_box(&lhs), black_box(&rhs))
    });
    let neon = neon_available()
        .then(|| time_dot(iterations, || dot_i8_neon(black_box(&lhs), black_box(&rhs))));

    DotBenchmarkReport {
        iterations,
        blocks_per_iteration: BENCH_BLOCKS,
        elements_per_iteration: BENCH_ELEMENTS,
        scalar,
        neon,
    }
}

pub fn dot_i8_scalar(lhs: &[i8], rhs: &[i8]) -> i32 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(&left, &right)| i32::from(left) * i32::from(right))
        .sum()
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

pub fn dot_i8_neon(lhs: &[i8], rhs: &[i8]) -> i32 {
    assert_eq!(lhs.len(), rhs.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: runtime feature detection confirms NEON support.
            return unsafe { dot_i8_neon_aarch64(lhs, rhs) };
        }
    }

    dot_i8_scalar(lhs, rhs)
}

fn time_dot(iterations: usize, mut dot: impl FnMut() -> i32) -> TimedDot {
    let started = Instant::now();
    let mut checksum = 0_i64;
    for _ in 0..iterations {
        checksum = checksum.wrapping_add(i64::from(black_box(dot())));
    }
    TimedDot {
        checksum: black_box(checksum),
        elapsed: started.elapsed(),
    }
}

fn deterministic_q8_values(len: usize, salt: u32) -> Vec<i8> {
    (0..len)
        .map(|idx| {
            let value = ((idx as u32)
                .wrapping_mul(37)
                .wrapping_add(salt.wrapping_mul(19))
                % 127) as i16
                - 63;
            value as i8
        })
        .collect()
}

fn ns_per_block(elapsed: Duration, iterations: usize, blocks_per_iteration: usize) -> f64 {
    elapsed.as_secs_f64() * 1_000_000_000.0 / (iterations * blocks_per_iteration) as f64
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_i8_neon_aarch64(lhs: &[i8], rhs: &[i8]) -> i32 {
    use std::arch::aarch64::{
        vaddq_s32, vaddvq_s32, vdupq_n_s32, vget_high_s8, vget_low_s8, vld1q_s8, vmull_s8,
        vpaddlq_s16,
    };

    let mut acc = unsafe { vdupq_n_s32(0) };
    let mut idx = 0;
    while idx + 16 <= lhs.len() {
        // SAFETY: the loop bound guarantees 16 readable i8 lanes from both slices.
        let left = unsafe { vld1q_s8(lhs.as_ptr().add(idx)) };
        let right = unsafe { vld1q_s8(rhs.as_ptr().add(idx)) };
        let low_products = unsafe { vmull_s8(vget_low_s8(left), vget_low_s8(right)) };
        let high_products = unsafe { vmull_s8(vget_high_s8(left), vget_high_s8(right)) };
        acc = unsafe { vaddq_s32(acc, vpaddlq_s16(low_products)) };
        acc = unsafe { vaddq_s32(acc, vpaddlq_s16(high_products)) };
        idx += 16;
    }

    let mut sum = unsafe { vaddvq_s32(acc) };
    while idx < lhs.len() {
        sum += i32::from(lhs[idx]) * i32::from(rhs[idx]);
        idx += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::{bench_dot, dot_i8_neon, dot_i8_scalar};

    #[test]
    fn scalar_dot_handles_signed_q8_values() {
        let lhs = [-3, -2, -1, 0, 1, 2, 3];
        let rhs = [4, -5, 6, -7, 8, -9, 10];

        assert_eq!(dot_i8_scalar(&lhs, &rhs), 12);
    }

    #[test]
    fn neon_path_matches_scalar() {
        let lhs: Vec<i8> = (0..257).map(|idx| ((idx * 7) % 127) as i8 - 63).collect();
        let rhs: Vec<i8> = (0..257).map(|idx| ((idx * 11) % 127) as i8 - 63).collect();

        assert_eq!(dot_i8_neon(&lhs, &rhs), dot_i8_scalar(&lhs, &rhs));
    }

    #[test]
    fn q8_dot_benchmark_preserves_checksum_parity() {
        let report = bench_dot(2);

        if let Some(neon) = report.neon {
            assert_eq!(neon.checksum, report.scalar.checksum);
        }
    }
}
