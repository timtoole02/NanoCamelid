// Criterion benchmark: Q8_0 i8 dot-product kernels (scalar vs NEON vs SDOT).
//
// "Before/after" comparison for the SIMD work. On the Pi (Cortex-A76) all three run; on a
// non-aarch64 host only `scalar` is meaningful. Run with: `cargo bench --bench bench_q8_dot`.
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use nanocamelid::q8::{dot_i8_neon, dot_i8_scalar, dot_i8_sdot, dotprod_available, neon_available};

fn deterministic_i8(n: usize, seed: u64) -> Vec<i8> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as i8).clamp(-127, 127)
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let elements = 32 * 1024; // 1024 Q8_0 blocks
    let lhs = deterministic_i8(elements, 17);
    let rhs = deterministic_i8(elements, 91);

    let mut group = c.benchmark_group("q8_dot");
    group.bench_function("scalar", |b| {
        b.iter(|| black_box(dot_i8_scalar(black_box(&lhs), black_box(&rhs))))
    });
    if neon_available() {
        group.bench_function("neon", |b| {
            b.iter(|| black_box(dot_i8_neon(black_box(&lhs), black_box(&rhs))))
        });
    }
    if dotprod_available() {
        group.bench_function("sdot", |b| {
            b.iter(|| black_box(dot_i8_sdot(black_box(&lhs), black_box(&rhs))))
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
