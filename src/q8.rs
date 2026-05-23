use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

pub const DEFAULT_DOT_BENCH_ITERATIONS: usize = 2_000;
pub const DEFAULT_DOT_BENCH_RUNS: usize = 5;
pub const DOT_KERNEL_ENV: &str = "NANOCAMELID_Q8_DOT_KERNEL";
pub const SDOT_CANDIDATE_ENV: &str = "NANOCAMELID_Q8_DOT_SDOT";

const Q8_BLOCK_SIZE: usize = 32;
const BENCH_BLOCKS: usize = 1_024;
const BENCH_ELEMENTS: usize = Q8_BLOCK_SIZE * BENCH_BLOCKS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q8DotKernel {
    Scalar,
    Neon,
    Sdot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q8DotKernelSelector {
    pub requested: Option<Q8DotKernel>,
    pub selected: Q8DotKernel,
    pub fallback_reason: Option<&'static str>,
}

#[derive(Debug)]
pub struct DotBenchmarkReport {
    pub iterations: usize,
    pub runs: usize,
    pub blocks_per_iteration: usize,
    pub elements_per_iteration: usize,
    pub kernel_selector: Q8DotKernelSelector,
    pub selected: DotTimingSummary,
    pub scalar: DotTimingSummary,
    pub neon: Option<DotTimingSummary>,
    pub sdot: Option<DotTimingSummary>,
}

#[derive(Debug)]
pub struct TimedDot {
    pub checksum: i64,
    pub elapsed: Duration,
}

#[derive(Debug)]
pub struct DotTimingSummary {
    pub checksum: i64,
    pub elapsed_runs: Vec<Duration>,
}

impl DotBenchmarkReport {
    pub fn scalar_min_ns_per_block(&self) -> f64 {
        ns_per_block(
            self.scalar.min_elapsed(),
            self.iterations,
            self.blocks_per_iteration,
        )
    }

    pub fn scalar_median_ns_per_block(&self) -> f64 {
        ns_per_block(
            self.scalar.median_elapsed(),
            self.iterations,
            self.blocks_per_iteration,
        )
    }

    pub fn neon_min_ns_per_block(&self) -> Option<f64> {
        self.neon.as_ref().map(|neon| {
            ns_per_block(
                neon.min_elapsed(),
                self.iterations,
                self.blocks_per_iteration,
            )
        })
    }

    pub fn neon_median_ns_per_block(&self) -> Option<f64> {
        self.neon.as_ref().map(|neon| {
            ns_per_block(
                neon.median_elapsed(),
                self.iterations,
                self.blocks_per_iteration,
            )
        })
    }

    pub fn neon_min_speedup(&self) -> Option<f64> {
        self.neon_min_ns_per_block()
            .map(|neon_ns| self.scalar_min_ns_per_block() / neon_ns)
    }

    pub fn neon_median_speedup(&self) -> Option<f64> {
        self.neon_median_ns_per_block()
            .map(|neon_ns| self.scalar_median_ns_per_block() / neon_ns)
    }

    pub fn sdot_min_ns_per_block(&self) -> Option<f64> {
        self.sdot.as_ref().map(|sdot| {
            ns_per_block(
                sdot.min_elapsed(),
                self.iterations,
                self.blocks_per_iteration,
            )
        })
    }

    pub fn sdot_median_ns_per_block(&self) -> Option<f64> {
        self.sdot.as_ref().map(|sdot| {
            ns_per_block(
                sdot.median_elapsed(),
                self.iterations,
                self.blocks_per_iteration,
            )
        })
    }

    pub fn sdot_min_speedup(&self) -> Option<f64> {
        self.sdot_min_ns_per_block()
            .map(|sdot_ns| self.scalar_min_ns_per_block() / sdot_ns)
    }

    pub fn sdot_median_speedup(&self) -> Option<f64> {
        self.sdot_median_ns_per_block()
            .map(|sdot_ns| self.scalar_median_ns_per_block() / sdot_ns)
    }

    pub fn sdot_vs_neon_min_speedup(&self) -> Option<f64> {
        self.neon_min_ns_per_block()
            .zip(self.sdot_min_ns_per_block())
            .map(|(neon_ns, sdot_ns)| neon_ns / sdot_ns)
    }

    pub fn sdot_vs_neon_median_speedup(&self) -> Option<f64> {
        self.neon_median_ns_per_block()
            .zip(self.sdot_median_ns_per_block())
            .map(|(neon_ns, sdot_ns)| neon_ns / sdot_ns)
    }
}

pub fn bench_dot_runs(iterations: usize, runs: usize) -> DotBenchmarkReport {
    let lhs = deterministic_q8_values(BENCH_ELEMENTS, 17);
    let rhs = deterministic_q8_values(BENCH_ELEMENTS, 91);
    let kernel_selector = Q8DotKernelSelector::from_env();

    let scalar = time_dot_runs(iterations, runs, || {
        dot_i8_scalar(black_box(&lhs), black_box(&rhs))
    });
    let selected = time_dot_runs(iterations, runs, || {
        dot_i8_with_selector(black_box(&lhs), black_box(&rhs), kernel_selector)
    });
    let neon = neon_available().then(|| {
        time_dot_runs(iterations, runs, || {
            dot_i8_neon(black_box(&lhs), black_box(&rhs))
        })
    });
    let sdot = sdot_candidate_enabled().then(|| {
        time_dot_runs(iterations, runs, || {
            dot_i8_sdot(black_box(&lhs), black_box(&rhs))
        })
    });

    DotBenchmarkReport {
        iterations,
        runs,
        blocks_per_iteration: BENCH_BLOCKS,
        elements_per_iteration: BENCH_ELEMENTS,
        kernel_selector,
        selected,
        scalar,
        neon,
        sdot,
    }
}

impl Q8DotKernel {
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
            Self::Sdot => "sdot",
        }
    }
}

impl Q8DotKernelSelector {
    pub fn from_env() -> Self {
        let requested = env::var(DOT_KERNEL_ENV)
            .ok()
            .as_deref()
            .and_then(parse_requested_kernel);

        Self::for_request(
            requested,
            RuntimeFeatures::detect(),
            sdot_candidate_requested(),
        )
    }

    fn for_request(
        requested: Option<Q8DotKernel>,
        features: RuntimeFeatures,
        sdot_candidate_enabled: bool,
    ) -> Self {
        match requested {
            None | Some(Q8DotKernel::Scalar) => Self {
                requested,
                selected: Q8DotKernel::Scalar,
                fallback_reason: None,
            },
            Some(Q8DotKernel::Neon) if features.neon => Self {
                requested,
                selected: Q8DotKernel::Neon,
                fallback_reason: None,
            },
            Some(Q8DotKernel::Neon) => Self {
                requested,
                selected: Q8DotKernel::Scalar,
                fallback_reason: Some("neon_unavailable"),
            },
            Some(Q8DotKernel::Sdot) if !sdot_candidate_enabled => Self {
                requested,
                selected: Q8DotKernel::Scalar,
                fallback_reason: Some("sdot_candidate_not_enabled"),
            },
            Some(Q8DotKernel::Sdot) if features.dotprod => Self {
                requested,
                selected: Q8DotKernel::Sdot,
                fallback_reason: None,
            },
            Some(Q8DotKernel::Sdot) => Self {
                requested,
                selected: Q8DotKernel::Scalar,
                fallback_reason: Some("dotprod_unavailable"),
            },
        }
    }
}

#[derive(Clone, Copy)]
struct RuntimeFeatures {
    neon: bool,
    dotprod: bool,
}

impl RuntimeFeatures {
    fn detect() -> Self {
        Self {
            neon: neon_available(),
            dotprod: dotprod_available(),
        }
    }
}

fn parse_requested_kernel(value: &str) -> Option<Q8DotKernel> {
    match value {
        "scalar" | "SCALAR" => Some(Q8DotKernel::Scalar),
        "neon" | "NEON" => Some(Q8DotKernel::Neon),
        "sdot" | "SDOT" => Some(Q8DotKernel::Sdot),
        _ => None,
    }
}

pub fn dot_i8_scalar(lhs: &[i8], rhs: &[i8]) -> i32 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(&left, &right)| i32::from(left) * i32::from(right))
        .sum()
}

pub fn dot_i8_with_selector(lhs: &[i8], rhs: &[i8], selector: Q8DotKernelSelector) -> i32 {
    match selector.selected {
        Q8DotKernel::Scalar => dot_i8_scalar(lhs, rhs),
        Q8DotKernel::Neon => dot_i8_neon(lhs, rhs),
        Q8DotKernel::Sdot => dot_i8_sdot(lhs, rhs),
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

pub fn sdot_candidate_requested() -> bool {
    env::var(SDOT_CANDIDATE_ENV)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

pub fn sdot_candidate_enabled() -> bool {
    sdot_candidate_requested() && dotprod_available()
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

pub fn dot_i8_sdot(lhs: &[i8], rhs: &[i8]) -> i32 {
    assert_eq!(lhs.len(), rhs.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: runtime feature detection confirms FEAT_DotProd support.
            return unsafe { dot_i8_sdot_aarch64(lhs, rhs) };
        }
    }

    dot_i8_scalar(lhs, rhs)
}

impl DotTimingSummary {
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

fn time_dot_runs(iterations: usize, runs: usize, mut dot: impl FnMut() -> i32) -> DotTimingSummary {
    let mut checksum = None;
    let mut elapsed_runs = Vec::with_capacity(runs);

    for _ in 0..runs {
        let timed = time_dot(iterations, &mut dot);
        match checksum {
            Some(expected) => assert_eq!(timed.checksum, expected),
            None => checksum = Some(timed.checksum),
        }
        elapsed_runs.push(timed.elapsed);
    }

    DotTimingSummary {
        checksum: checksum.unwrap_or_default(),
        elapsed_runs,
    }
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

#[cfg(target_arch = "aarch64")]
unsafe fn dot_i8_sdot_aarch64(lhs: &[i8], rhs: &[i8]) -> i32 {
    use std::arch::{
        aarch64::{vaddvq_s32, vdupq_n_s32, vld1q_s8},
        asm,
    };

    let mut acc = unsafe { vdupq_n_s32(0) };
    let mut idx = 0;
    while idx + 16 <= lhs.len() {
        // SAFETY: the loop bound guarantees 16 readable i8 lanes from both slices.
        let left = unsafe { vld1q_s8(lhs.as_ptr().add(idx)) };
        let right = unsafe { vld1q_s8(rhs.as_ptr().add(idx)) };
        unsafe {
            asm!(
                ".arch_extension dotprod",
                "sdot {acc:v}.4s, {left:v}.16b, {right:v}.16b",
                acc = inout(vreg) acc,
                left = in(vreg) left,
                right = in(vreg) right,
                options(nostack, preserves_flags),
            );
        }
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
    use super::{
        Q8DotKernel, Q8DotKernelSelector, RuntimeFeatures, bench_dot_runs, dot_i8_neon,
        dot_i8_scalar, dot_i8_sdot,
    };

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
    fn sdot_path_matches_scalar() {
        let lhs: Vec<i8> = (0..259).map(|idx| ((idx * 5) % 127) as i8 - 63).collect();
        let rhs: Vec<i8> = (0..259).map(|idx| ((idx * 13) % 127) as i8 - 63).collect();

        assert_eq!(dot_i8_sdot(&lhs, &rhs), dot_i8_scalar(&lhs, &rhs));
    }

    #[test]
    fn q8_kernel_selector_defaults_to_scalar() {
        let selector = Q8DotKernelSelector::for_request(
            None,
            RuntimeFeatures {
                neon: true,
                dotprod: true,
            },
            false,
        );

        assert_eq!(selector.selected, Q8DotKernel::Scalar);
        assert_eq!(selector.fallback_reason, None);
    }

    #[test]
    fn q8_kernel_selector_falls_back_when_neon_is_unavailable() {
        let selector = Q8DotKernelSelector::for_request(
            Some(Q8DotKernel::Neon),
            RuntimeFeatures {
                neon: false,
                dotprod: true,
            },
            false,
        );

        assert_eq!(selector.selected, Q8DotKernel::Scalar);
        assert_eq!(selector.fallback_reason, Some("neon_unavailable"));
    }

    #[test]
    fn q8_kernel_selector_keeps_sdot_default_off() {
        let selector = Q8DotKernelSelector::for_request(
            Some(Q8DotKernel::Sdot),
            RuntimeFeatures {
                neon: true,
                dotprod: true,
            },
            false,
        );

        assert_eq!(selector.selected, Q8DotKernel::Scalar);
        assert_eq!(selector.fallback_reason, Some("sdot_candidate_not_enabled"));
    }

    #[test]
    fn q8_dot_benchmark_preserves_checksum_parity() {
        let report = bench_dot_runs(2, 2);

        assert_eq!(report.selected.checksum, report.scalar.checksum);
        if let Some(neon) = report.neon {
            assert_eq!(neon.checksum, report.scalar.checksum);
        }
        if let Some(sdot) = report.sdot {
            assert_eq!(sdot.checksum, report.scalar.checksum);
        }
    }

    #[test]
    fn simd_candidates_match_scalar_across_q8_shapes() {
        for len in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257] {
            let lhs: Vec<i8> = (0..len)
                .map(|idx| ((idx * 17 + 3) % 127) as i8 - 63)
                .collect();
            let rhs: Vec<i8> = (0..len)
                .map(|idx| ((idx * 19 + 7) % 127) as i8 - 63)
                .collect();
            let scalar = dot_i8_scalar(&lhs, &rhs);

            assert_eq!(dot_i8_neon(&lhs, &rhs), scalar, "len={len}");
            assert_eq!(dot_i8_sdot(&lhs, &rhs), scalar, "len={len}");
        }
    }
}
