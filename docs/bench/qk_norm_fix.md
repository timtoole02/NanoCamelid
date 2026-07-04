# Qwen3 / Gemma 3 per-head QK-norm fix

Date: 2026-07-04
Scope: correctness fix for the `qwen3` and `gemma3` architectures.

## The bug

Qwen3 and Gemma 3 apply a per-head RMSNorm to Q and K (the `attn_q_norm` /
`attn_k_norm` tensors) between the QKV projection and RoPE — it is a
defining part of those architectures. NanoCamelid loaded neither tensor and
never applied the norm, so its forward pass did not match the model.

Both families were nonetheless listed **"Supported"** in
`docs/MODEL_CATALOG.md` with "exact scalar-vs-selected parity". That parity
only compares NanoCamelid's own scalar and SIMD kernels against each other —
both skipped QK-norm, so they agreed with each other while both diverging
from the real model. Self-consistent wrong answers passed a self-consistency
gate.

## Evidence (qwen3-0.6B Q8_0, greedy, prompt "The capital of France is")

| build | generated token ids | decoded |
|---|---|---|
| baseline (no QK-norm) | `[1479, 1479, 1479, 1479, 1479, 1479, 1479, 1479, 1479, 1479, 1479, 1479]` | degenerate single-token loop |
| **fixed (QK-norm)** | `[12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722, 315, 279]` | **" Paris. The capital of France is also the capital of the"** |

Not "subtly wrong" — without the norm, qwen3 collapses to a repeated token.
The chat path (thinking mode) is likewise correct after the fix: it emits a
coherent `<think>` reasoning trace about France's capital.

Regression: on Llama 3.2 1B (an arch with no QK-norm tensors) the fixed and
baseline binaries produce **token-identical** output
(`[12366, 13, 578, 469, 3168, 301, 22703, 374, 7559, 304, 12366, 13]`),
confirming the change is a true no-op where the tensors are absent.

## The fix

- `apply_qk_norm` (`src/inference.rs`): RMSNorm each `head_dim`-wide slice of
  Q and K in place, reusing the existing `rms_norm` math. Ported 1:1 from
  Camelid's reference per-head QK-norm (`src/diffusion_gemma.rs`, the
  llama.cpp-parity-verified path).
- Applied to Q and K **before RoPE** in both `run_layer_range` (decode) and
  `run_layer_range_batch` (prefill). Because `forward_pass` and
  `prefill_pass_batch` delegate to those two functions, this covers the
  single-node chat/generate path AND the distributed pipeline workers.
- `attn_q_norm.weight` / `attn_k_norm.weight` are loaded optionally in
  `load_layer_weights`, so archs without them (llama, qwen2, mistral,
  smollm, deepseek-qwen) are unaffected.
- NanoCamelid reads `head_dim` from `attention.key_length`, so qwen3's
  128-wide heads (≠ embedding/head_count) are handled correctly.

## Coverage and boundaries

- **qwen3** (0.6B/1.7B/4B/8B): fixed and verified end-to-end.
- **gemma3**: the `gemma-3-1b-it` GGUF carries the same `attn_q_norm` /
  `attn_k_norm` tensors, so QK-norm is now applied there too. Gemma 3 also
  uses features not audited by this change (e.g. sliding-window attention);
  this fix supplies the QK-norm prerequisite, not a full gemma3 correctness
  claim.
- **qwen2 / DeepSeek-R1-Qwen / mistral / smollm / llama**: no QK-norm
  tensors → no-op, byte-identical output.
- Wire tensor-parallelism (`cluster_tp_node`) does not yet carry QK-norm and
  now **fails closed** on qwen3/gemma3 rather than silently dropping it
  (qwen3 fits one Pi, so this is not a practical limit).

## Provenance

Identified as the single highest-leverage item in a cross-repo gap analysis
of Camelid vs NanoCamelid (32-agent adversarial workflow, 2026-07-04): one
small op, reusing an existing primitive, that corrects two already-shipped
"Supported" architecture families.
