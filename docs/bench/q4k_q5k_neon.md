# Q4_K / Q5_K AArch64 NEON SDOT kernels

Date: 2026-07-04
Scope: performance — add dotprod SDOT matmul kernels for the Q4_K and Q5_K
super-block quant formats (previously scalar-only).

## Why

NanoCamelid decodes every K-quant but only Q8_0/Q4_0/Q6_K/Q8_K had a NEON
SDOT matmul; Q4_K/Q5_K ran a scalar per-element loop (~1.5–2x slower on a
matmul-bound model). Q4_K_M and Q5_K_M are the most common published quant
mixes, so this touched a large share of real-world models. Identified as the
top performance lever in the Camelid→NanoCamelid gap analysis.

## What

- `q4k_dot_preloaded_neon` / `q5k_dot_preloaded_neon` (`src/q8.rs`): compute
  a super-block row-dot as `Σ_j x_scales[j]·(scale[j]·Σ(q·x) − min[j]·Σx)`.
  Only the four integer inner sums are vectorized (via `qk_pair_sdot`, inline
  `sdot` in the same stable `.arch_extension dotprod` idiom as the Q6_K
  kernel); the f32 scale/min combine is byte-identical to the scalar path.
  The activation sub-block sum `Σx` (Q8_0 activations carry no bsums) is a
  second SDOT against a ones vector. Q5_K bakes the 5th bit into the quant
  before the dot.
- Wired behind `NANOCAMELID_Q4K_SDOT` / `NANOCAMELID_Q5K_SDOT` (default on,
  gated on `is_aarch64_feature_detected!("dotprod")`) in the single and batch
  matmul drivers, mirroring `matmul_q6_k`. Scalar path retained as fallback.

## Correctness (bit-exact gate)

The 4 existing parity tests run through the NEON path on Apple Silicon and
pass: `matmul_q4_k_matches_dequantized_reference`,
`matmul_q5_k_matches_dequantized_reference`, and both
`_batch_matches_single_token_reference`. Both `NANOCAMELID_Q*K_SDOT=0`
(scalar) and default (NEON) are green. On a Raspberry Pi 5, real-model greedy
token streams are **identical** NEON vs scalar:

- qwen2.5-coder-0.5B Q5_K_M: token-identical
- Qwen2.5-0.5B Q4_K_M: token-identical
- Llama-3.2-3B Q4_K_M: token-identical

Because only the exact integer sums are vectorized and the f32 combine is
unchanged, NEON is bit-identical to scalar, not merely within tolerance.

## Speed (Raspberry Pi 5, camelid1)

| model | path | scalar | NEON | speedup |
|---|---|---|---|---|
| Llama-3.2-3B Q4_K_M | decode | 3.86 tok/s | **5.58 tok/s** | **1.45x** |
| Llama-3.2-3B Q4_K_M | prefill (77 tok) | 15.24 s | **8.25 s** | **1.85x** |
| qwen2.5-coder-0.5B Q5_K_M | decode | 8.97 | 9.70 | 1.08x |
| Qwen2.5-0.5B Q4_K_M | decode | 13.33 | 13.50 | 1.01x |

The win scales with how matmul-bound the model is: a 3B decode is
matmul-dominated (1.45x), while a 0.5B decode is overhead-bound so the SDOT
is masked by Amdahl (~1.0–1.1x). Prefill (batched GEMM, unpack amortized
across the batch) gains more (1.85x).

**Bottom line:** Q4_K_M 3B decode goes 3.86 → 5.58 tok/s, ≈ the Q4_0 3B rate
(5.33 tok/s) — the K-quant speed penalty on the most common published quant
is essentially eliminated, with bit-exact parity.
