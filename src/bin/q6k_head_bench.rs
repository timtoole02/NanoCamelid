// Isolate the Mixtral untied-head cost: time matmul_q6_k at the head's
// exact shape (vocab x hidden) with a synthetic Q6_K matrix, plus the
// rms_norm + quantize steps compute_logits_from_hidden performs around it.
// Usage: q6k_head_bench [rows] [cols] [iters]

use nanocamelid::inference;
use nanocamelid::q8::{Q6KBlock, QK_K_BLOCK_SIZE};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let rows: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(32000);
    let cols: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(4096);
    let iters: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(20);

    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    let w: Vec<Q6KBlock> = (0..rows * blocks_per_row)
        .map(|i| {
            let b = (i % 251) as u8;
            Q6KBlock::from_parts(
                [b; 128],
                [b; 64],
                [(i % 17) as i8; 16],
                1000 + (i % 7) as u16,
            )
        })
        .collect();

    let x: Vec<f32> = (0..cols).map(|i| ((i % 97) as f32 - 48.0) / 48.0).collect();
    let mut x_i8 = vec![0_i8; cols];
    let mut x_scales = vec![0.0_f32; cols / 32];
    let mut out = vec![0.0_f32; rows];

    // Warmup
    inference::quantize_f32_to_q8_0(&x, &mut x_i8, &mut x_scales);
    inference::matmul_q6_k(&mut out, &x_i8, &x_scales, &w, rows, cols);

    let started = Instant::now();
    for _ in 0..iters {
        inference::quantize_f32_to_q8_0(&x, &mut x_i8, &mut x_scales);
        inference::matmul_q6_k(&mut out, &x_i8, &x_scales, &w, rows, cols);
    }
    let avg_ms = started.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let checksum: f32 = out.iter().sum();
    println!(
        "json: {{\"benchmark\":\"q6k-head\",\"rows\":{rows},\"cols\":{cols},\"iters\":{iters},\"avg_ms\":{avg_ms:.3},\"checksum\":{checksum}}}"
    );
}
