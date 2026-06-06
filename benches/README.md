# Criterion benchmarks

Before/after evidence for the Cortex-A76 performance work. **Run these on the Pi** — on a
non-aarch64 host the NEON/SDOT entries are skipped, and on Apple Silicon the numbers do not
transfer to the A76.

| Bench | What it measures |
| --- | --- |
| `bench_q8_dot` | Q8_0 i8 dot kernels: scalar vs NEON (`vmull_s8`) vs SDOT (FEAT_DotProd) |
| `bench_q4_dot` | Q4_0×Q8_0 block dot kernels: scalar vs NEON vs SDOT |
| `bench_matmul` | `matmul_q8_0` 2048×2048: 1 rayon worker vs 4 pinned workers |
| `bench_forward` | One full decode `forward_pass` on a tiny synthetic Q8_0 model |

```bash
cargo bench                       # everything
cargo bench --bench bench_matmul  # one suite
cargo bench -- --quick            # faster, less precise
```

## Capturing before/after

The "after" configuration is the repo default (`.cargo/config.toml` sets
`target-cpu=cortex-a76`; release/bench profiles use `lto = true`, `codegen-units = 1`; the
kernel auto-selector picks SDOT; the matmul bench compares 1 vs 4 workers directly).

To reproduce the "before" baseline:

```bash
# disable A76 tuning + force the old defaults (scalar kernel, default profile)
RUSTFLAGS="" NANOCAMELID_Q8_DOT_KERNEL=scalar NANOCAMELID_Q4_DOT_KERNEL=scalar \
  cargo bench
```

Criterion keeps the previous run as the comparison baseline automatically, so running the
"before" config and then the "after" config prints the regression/improvement deltas.

Note on long-vector dot results: with `target-cpu=cortex-a76` the *scalar* loop may
auto-vectorize (LLVM can emit SDOT itself), so on long contiguous vectors it can rival the
hand-written kernels, which use a single accumulator. The matmul path calls the kernel per
32-element block, where this effect is negligible — trust `bench_matmul`/`bench_forward`
over `bench_*_dot` for end-to-end decisions.
