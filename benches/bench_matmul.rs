// Criterion benchmark: Q8_0 matmul throughput, 1 worker vs 4 pinned workers.
//
// Demonstrates the rayon per-node parallelism win. The kernel itself is auto-selected
// (SDOT on A76). Run with: `cargo bench --bench bench_matmul`.
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::ThreadPoolBuilder;

use nanocamelid::inference::matmul_q8_0;
use nanocamelid::q8::{Q8DotKernelSelector, Q8_0Block, Q8_BLOCK_SIZE};

fn make_weights(rows: usize, blocks_per_row: usize, seed: u64) -> Vec<Q8_0Block> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        state
    };
    (0..rows * blocks_per_row)
        .map(|_| {
            let mut values = [0i8; Q8_BLOCK_SIZE];
            for v in values.iter_mut() {
                *v = ((next() >> 33) as i8).clamp(-127, 127);
            }
            Q8_0Block::from_parts(0x3C00, values) // f16 scale = 1.0
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let rows = 2048usize;
    let cols = 2048usize;
    let blocks_per_row = cols / Q8_BLOCK_SIZE;

    let weights = make_weights(rows, blocks_per_row, 17);
    let x_i8: Vec<i8> = (0..cols).map(|i| ((i as i32 % 255) - 127) as i8).collect();
    let x_scales: Vec<f32> = vec![0.05; cols / Q8_BLOCK_SIZE];
    let selector = Q8DotKernelSelector::from_env_or_auto();

    let mut group = c.benchmark_group("matmul_q8_0_2048x2048");
    for threads in [1usize, 4usize] {
        let pool = ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        group.bench_function(format!("{threads}t"), |b| {
            let mut out = vec![0.0f32; rows];
            b.iter(|| {
                pool.install(|| {
                    matmul_q8_0(&mut out, &x_i8, &x_scales, &weights, rows, cols, selector)
                });
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
