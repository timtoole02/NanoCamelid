# Phi-3-mini support: fused QKV / gate-up split + USER_DEFINED stop tokens

Date: 2026-07-04
Scope: model support — bring up `microsoft/Phi-3-mini-4k-instruct-gguf`
(`Phi-3-mini-4k-instruct-q4.gguf`, `general.architecture = phi3`) end to end.

## Why it was blocked

The cross-repo model-support gap analysis flagged two independent real bugs
that kept Phi-3 out of the supported set:

1. **Fused attention/FFN tensors.** llama.cpp's Phi-3 GGUFs ship a single
   fused `blk.*.attn_qkv.weight` (Q, K, V output rows concatenated) and a
   single fused `blk.*.ffn_up.weight` (gate and up concatenated). NanoCamelid's
   loader required the split `attn_q/k/v` and `ffn_gate/ffn_up` tensors and
   errored on the fused layout.
2. **Chat never stops.** The canonical GGUF sets `eos_token_id = 32000`
   (`<|endoftext|>`), but the Phi-3 chat template ends every turn with
   `<|end|>` = `32007`, which was not in the engine's end-of-generation set,
   so chat ran to `max_tokens`.

A third issue surfaced during bring-up and was the actual reason chat looked
broken even after the eog fix (below).

## What changed

### Fused-tensor split at load (`src/model.rs`)

- `slice_matrix_rows(matrix, cols, r0, r1)` slices a `QuantizedMatrix` on
  whole output-row boundaries. Every output row is a whole number of quant
  blocks (`cols / QK_K_BLOCK_SIZE` for K-quants, `cols / 32` for the legacy
  formats), so the split is block-aligned and needs no sub-block arithmetic.
  It covers Q4_K/Q5_K/Q6_K/Q2_K/Q3_K/Q8_K/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL and
  fails closed on swizzled or f32 layouts.
- `load_layer_weights` is now fused-aware: when `attn_qkv.weight` is present it
  splits into `wq = rows[0, q)`, `wk = rows[q, q+kv)`, `wv = rows[q+kv, q+2kv)`
  with `q = attention_output_width`, `kv = kv_width`; when `ffn_gate.weight` is
  absent it splits the fused `ffn_up` into gate `rows[0, ff)` and up
  `rows[ff, 2ff)`.
- `validate_model_tensors` validates the fused shapes
  (`[embedding_length, attention_output_width + 2*kv_width]` and
  `[embedding_length, 2*feed_forward_length]`) when the fused tensors are used.

### End-of-generation set (`src/tokenizer.rs`)

- `<|end|>` (32007) is added to the eog set alongside eos/eot/eom.

### Phi-3 chat renderer (`src/tokenizer.rs`)

- `render_phi3_prompt` emits `<|{role}|>\n{content}<|end|>\n` per message and a
  trailing `<|assistant|>\n`, detected via `is_phi3_template`. The renderer
  requests `parse_special = true`.

### The real stop bug: USER_DEFINED special tokens (`src/tokenizer.rs`)

Even with the renderer requesting `parse_special = true`, the prompt markers
tokenized as literal characters:

```
<s><|user|>\nName three primary colors.<|end|>\n<|assistant|>\n
-> [1, 529, 29989, 1792, 29989, 29958, 13, ...]   (<, |, user, |, > ...)
```

Phi-3 registers `<|user|>` / `<|end|>` / `<|assistant|>` as **USER_DEFINED**
(token_type 4), not CONTROL (3). The SPM special-token split
(`longest_control_token_at`) matched CONTROL only, so the markers fell through
to the score-based SPM merge — and because added markers carry score 0, that
merge decomposes them into their normal sub-pieces. The model then mimicked the
literal markers and never emitted `<|end|>`, so chat never stopped.

The fix matches llama.cpp, whose special-token cache is built from CONTROL,
USER_DEFINED **and** UNKNOWN tokens. `TokenKind::is_special_split()` now returns
true for all three, and it is used in the split matcher and in the two
detokenize strip sites (so a stray marker never leaks into displayed output).
This only affects `parse_special = true` encodes; SPM chat for models that
render with `parse_special = false` (TinyLlama, Mistral, Gemma) is untouched.

After the fix:

```
<s><|user|>\nName three primary colors.<|end|>\n<|assistant|>\n
-> [1, 32010, 29871, 13, 1170, 2211, 7601, 11955, 29889, 32007, 29871, 13, 32001, 29871, 13]
```

## Verification (M4, aarch64 NEON — same SDOT path the Pi runs)

The Raspberry Pi nodes were offline this session, so the smoke ran on the Mac
aarch64 NEON path (the identical kernel path the Pi uses). Pi smoke is pending.

- **Tokenization:** the three markers collapse to `32010` / `32007` / `32001`
  (shown above) instead of literal-character runs.
- **Chat stops cleanly:** `chat Phi-3-mini-4k-instruct-q4.gguf
  "Name three primary colors." 0.0 200` generated **47** tokens (not the 200
  cap), a coherent "Red / Blue / Yellow" answer, and `generation_status: ok` —
  the model emitted `<|end|>` = 32007 and the eog set halted generation.
  About `15.6 tok/sec` decode; exercises the Q4_K NEON kernel.
- **Raw generation:** `generate ... "The capital of France is"` produces
  `"Paris."`.
- **No regression:** Mistral-7B-Instruct-v0.3 Q8_0 (SPM, exercises the
  broadened matcher on the `parse_special = true` completion path) tokenizes
  normal prose unchanged (`[1, 1183, 6333, 1070, 5611, 1117]`) and generates
  `"Paris, ..."`.

## Tests

- `spm_parse_special_splits_user_defined_marker` (new): an SPM fixture with a
  USER_DEFINED `<|end|>` marker must split to its single special id under
  `parse_special = true` and stay a single token mid-text. Fails on the
  CONTROL-only matcher, passes after the fix.
- Full suite green: 123 lib + 191 bin tests pass (`cargo test --release`).

## Follow-up (2026-07-05): bugs found by the Pi hardware smoke

The first pass was validated on an M4 with a K-quant `Phi-3-mini-4k` GGUF. A
real Pi 5 smoke with the canonical `Phi-3.5-mini-instruct-q4_0.gguf` exposed two
variant-specific bugs the Mac path had dodged, plus one non-phi3 resource issue.

1. **Swizzled fused-tensor split.** Q4_0 tensors load through the 1x4 SDOT
   swizzle by default (`NANOCAMELID_Q4_SWIZZLE_1X4`, default on), so the fused
   `attn_qkv`/`ffn_up` arrived as `Q4_0Swizzled1x4`, which `slice_matrix_rows`
   cannot slice (it errors on swizzled/f32). The Mac model was K-quant, which is
   never swizzled, so it never hit this. Fix: `load_quantized_matrix_unswizzled`
   loads the fused tensors as plain `Q4_0`/`Q8_0` (K-quants defer to the normal
   loader), so the whole-row split works; non-fused tensors keep the default
   swizzle. (The split pieces are plain, i.e. not on the 1x4 fast path — a
   correctness-first choice; re-swizzling the pieces is a possible perf
   follow-up.)

2. **Renderer misroute (phi3 stolen by the TinyLlama matcher).** `render_chat_
   prompt` checked `is_tinyllama_marker_template` before `is_phi3_template`, and
   the TinyLlama matcher only required `<|system|>` + `<|user|>` + `<|assistant|>`
   — all present in the Phi-3.5 template. So Phi-3.5 rendered as `tinyllama_
   marker`, which sets `parse_special = self.chat_prompt_parse_special()` (false
   for SPM) → the markers tokenized as literal characters again and chat never
   stopped. (Phi-3-mini-4k's template lacked the `<|system|>` branch, so it
   still routed to phi3 on the Mac.) Fix: `is_tinyllama_marker_template` now also
   requires the template NOT contain `<|end|>` (phi3's turn terminator;
   TinyLlama/Zephyr use `</s>`), making the two matchers mutually exclusive at
   both dispatch sites. New test `phi3_template_is_not_matched_as_tinyllama_
   marker`.

3. **51 GB allocation (not a phi3 bug).** Phi-3.5-mini has a `131072` context;
   the KV cache is sized to the full context, which OOMs a 16 GB Pi. Bounded
   with `NANOCAMELID_CONTEXT_LIMIT` (e.g. `4096`).

### Pi verification (camelid2, Pi 5, default swizzle-on config)

`NANOCAMELID_CONTEXT_LIMIT=4096 nanocamelid chat phi-3.5-mini-instruct-q4_0.gguf
"Name three primary colors." 0.0 200`:

- weights load (no swizzle-split error), renderer `phi3`;
- prompt markers tokenize to `32010`/`32007`/`32001` (no literal-char runs);
- chat **stops cleanly** at 85 tokens (`generation_status: ok`) with a coherent
  answer, ~4.0 tok/s; raw generate → `"Paris."`.

Local: Phi-3-mini-4k still stops at 47 tok (no regression); 124 lib tests pass.
