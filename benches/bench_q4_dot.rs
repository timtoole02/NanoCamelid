// Criterion benchmark: Q4_0 x Q8_0 dot-product kernels (scalar vs NEON vs SDOT).
// Run with: `cargo bench --bench bench_q4_dot`.
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use nanocamelid::q4::{
    dot_q4_0_q8_0_neon, dot_q4_0_q8_0_scalar, dot_q4_0_q8_0_sdot, dotprod_available, neon_available,
    Q4_0Block, Q4_0_BLOCK_BYTES,
};

fn deterministic_block(seed: u64) -> Q4_0Block {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut bytes = [0u8; Q4_0_BLOCK_BYTES];
    // f16 scale = 1.0 (0x3C00)
    bytes[0] = 0x00;
    bytes[1] = 0x3C;
    for b in bytes.iter_mut().skip(2) {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *b = (state >> 33) as u8;
    }
    Q4_0Block::from_bytes(&bytes)
}

fn deterministic_i8x32(seed: u64) -> [i8; 32] {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut out = [0i8; 32];
    for v in out.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = ((state >> 33) as i8).clamp(-127, 127);
    }
    out
}

fn bench(c: &mut Criterion) {
    let w = deterministic_block(17);
    let x = deterministic_i8x32(91);

    let mut group = c.benchmark_group("q4_dot");
    group.bench_function("scalar", |b| {
        b.iter(|| black_box(dot_q4_0_q8_0_scalar(black_box(&w), black_box(&x))))
    });
    if neon_available() {
        group.bench_function("neon", |b| {
            b.iter(|| black_box(dot_q4_0_q8_0_neon(black_box(&w), black_box(&x))))
        });
    }
    if dotprod_available() {
        group.bench_function("sdot", |b| {
            b.iter(|| black_box(dot_q4_0_q8_0_sdot(black_box(&w), black_box(&x))))
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
