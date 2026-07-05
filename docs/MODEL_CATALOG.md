# NanoCamelid Model Catalog

This catalog tracks GGUF model rows that NanoCamelid can honestly claim, rows
that are likely compatible with the current runtime, and model families that
need more runtime work before they should be promoted.

The current runtime supports dense Llama-style, Qwen2-style, Qwen3-style
(with per-head QK-norm, fixed 2026-07-04; see `docs/bench/qk_norm_fix.md`),
SmolLM, and Gemma 3 transformer blocks with GGUF tensor types used by the tested
rows below. Runtime tensor support is broader than the promoted model rows:
`Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, and
`Q6_K`, `Q8_K`, and `IQ4_NL` can load and execute through the current reference
paths, but rows only move to supported after Pi-local smoke/parity evidence.
Long-context GGUFs should be smoke-tested with `NANOCAMELID_CONTEXT_LIMIT`
until full advertised-context memory behavior is validated.

## Default Aliases

The stable `1b` and `3b` aliases are intentionally narrow product defaults, not
general model-family selectors:

| Alias | Resolution |
| --- | --- |
| `1b` | `$NANOCAMELID_MODEL_DIR/Llama-3.2-1B-Instruct-Q4_0.gguf` when present, otherwise `$NANOCAMELID_MODEL_DIR/Llama-3.2-1B-Instruct-Q8_0.gguf` |
| `3b` | `$NANOCAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q4_0.gguf` |

`nanocamelid models list`, `nanocamelid models scan`, and `/v1/models` report
an `aliases` field for rows that currently resolve from those aliases.

## Quantization Compatibility

| Quant family | Runtime status | Promotion status |
| --- | --- | --- |
| `Q8_0` | Supported; scalar/NEON/SDOT paths | Pi-smoked baseline rows exist |
| `Q4_0` | Supported; scalar plus default Pi fast paths | Multiple Pi-smoked rows exist |
| `Q4_1` | Supported; scalar reference path | Pi-smoked through Gemma 3 1B IT tensor mix |
| `Q5_0`, `Q5_1` | Supported; scalar reference path | Q5_1 is covered by the Qwen2.5-Coder 0.5B Q5_K_M row |
| `Q4_K`, `Q5_K` | Supported; scalar plus default Pi SDOT path | Q4_K exercised by the Llama 3.2 3B, Llama 3 8B, and Phi-3-mini rows; Q5_K by the Qwen2.5-Coder 0.5B row (NEON is bit-exact vs scalar) |
| `Q2_K`, `Q3_K` | Supported; scalar reference path | Q2_K/Q3_K still need row-specific claims |
| `Q6_K` | Supported; scalar plus Pi SDOT path | Strand 14B capped-context smoke exists |
| `Q8_K` | Supported; scalar/NEON/SDOT Q8 dot path | Needs row-specific Pi smoke before promotion |
| `IQ4_NL` | Supported; scalar reference path | Needs row-specific Pi smoke before promotion |
| `IQ1_*`, `IQ2_*`, `IQ3_*`, other `IQ4_*` | Deferred | Add only when an exact target row needs it |
| `MXFP4`, `NVFP4`, `TQ1_0`, `TQ2_0` | Deferred | Out of scope until a concrete model row justifies the complexity |

## Supported and Pi-Smoked

These rows have been loaded and generated on Raspberry Pi-class ARM64 hardware.

| Model | GGUF row | Architecture | Status | Evidence |
| --- | --- | --- | --- | --- |
| Qwen2.5 0.5B Instruct | `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, `qwen2.5-0.5b-instruct-q4_0.gguf` | `qwen2` | Supported | `ready`; `qwen_im`; direct and chat generation pass; exact scalar-vs-selected parity; 512/1024/2048/4096/8192 context packs pass |
| Qwen2.5-Coder 0.5B Instruct | `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF`, `qwen2.5-coder-0.5b-instruct-q4_0.gguf` | `qwen2` | Supported | `ready`; `qwen_im`; direct and chat generation pass; exact scalar-vs-selected parity; 512/1024/2048/4096/8192 context packs pass |
| Qwen2.5-Coder 0.5B Instruct | `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF`, `qwen2.5-coder-0.5b-instruct-q5_k_m.gguf` | `qwen2` | Supported end-to-end | `ready`; `qwen_im`; mixed `Q5_1`, `Q5_K`, `Q6_K`, and `Q8_0` tensors load; direct generation produced 8 tokens at `9.98 tok/sec`; chat generation produced 8 tokens at `9.71 tok/sec` |
| Qwen3 0.6B Instruct | `Qwen/Qwen3-0.6B-GGUF`, `qwen3-0.6b-q8_0.gguf` | `qwen3` | Supported | `ready`; `qwen_im`; direct and chat generation pass; exact scalar-vs-selected chat parity; 512/1024/2048/4096/8192 context packs pass; sampled RSS about `1.41 GiB`; per-head QK-norm applied (fixed 2026-07-04) |
| Qwen3 1.7B Instruct | `Qwen/Qwen3-1.7B-GGUF`, `qwen3-1.7b-q8_0.gguf` | `qwen3` | Supported | `ready`; `qwen_im`; direct and chat generation pass; exact scalar-vs-selected chat parity; 512/1024/2048/4096/8192 context packs pass; sampled RSS about `3.56 GiB`; per-head QK-norm applied (fixed 2026-07-04) |
| Qwen3 4B Instruct | `miku552/Qwen3-4B-Q4_0-GGUF`, `qwen3-4b-q4_0.gguf` | `qwen3` | Supported | `ready`; `qwen_im`; direct and chat generation pass; exact scalar-vs-selected chat parity; 512/1024/2048/4096/8192 context packs pass; sampled RSS about `5.50 GiB`; per-head QK-norm applied (fixed 2026-07-04) |
| SmolLM3 3B | `jpohhhh/SmolLM3-3B-Q4_0-GGUF`, `smollm3-3b-q4_0.gguf` | `smollm3` | Supported | `ready`; ChatML token fallback renderer; direct and chat generation pass; exact scalar-vs-selected chat parity; 512/1024/2048/4096/8192 context packs pass; sampled RSS about `3.94 GiB` |
| SmolLM2 1.7B Instruct | `Maites/SmolLM2-1.7B-Instruct-Q4_0-GGUF`, `smollm2-1.7b-instruct-q4_0.gguf` | `llama` | Supported | `ready`; `qwen_im`; chat generation passes; direct prompt ended immediately; exact scalar-vs-selected chat parity; 512/1024/2048/4096/8192 context packs pass; sampled RSS about `2.02 GiB` |
| Gemma 3 1B IT | `unsloth/gemma-3-1b-it-GGUF`, `gemma-3-1b-it-q4_0.gguf` | `gemma3` | Supported | `ready`; `gemma_turn`; Q4_1 feed-forward tensors supported; direct and chat generation pass; exact scalar-vs-selected chat parity; 512/1024/2048/4096/8192 context packs pass; sampled RSS about `2.20 GiB`; per-head QK-norm applied (fixed 2026-07-04) |
| DeepSeek-R1-Distill-Qwen 1.5B | `ggml-org/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF`, `deepseek-r1-distill-qwen-1.5b-q4_0.gguf` | `qwen2` | Supported | `ready`; `deepseek_r1_qwen`; direct and chat generation pass; exact scalar-vs-selected parity; 512/1024/2048/4096/8192 context packs pass |
| Llama 3.2 1B Instruct | `Llama-3.2-1B-Instruct-Q4_0.gguf` | `llama` | Supported | `ready`; `llama3_instruct`; direct and chat generation pass; exact scalar-vs-selected parity; 512/1024/2048/4096/8192 context packs pass |
| Llama 3.2 1B Instruct | `Llama-3.2-1B-Instruct-Q8_0.gguf` | `llama` | Supported end-to-end | `ready`; `llama3_instruct`; `llama32_1b_shape: ok`; forced Pi readiness run passes host probe, inspect, scalar-vs-SDOT chat smoke, and direct chat generation of `"Hello!"` at about `3.20 tok/sec`; 512/1024/2048/4096/8192 context packs pass |
| Llama 3.2 3B Instruct | `Llama-3.2-3B-Instruct-Q4_0.gguf` | `llama` | Supported | `ready`; `llama3_instruct`; direct and chat generation pass; exact scalar-vs-selected parity; capped 8096 TUI launch and 512/1024/2048/4096/8192 context packs pass. Three-Pi weighted tensor parallelism (`cluster_tp_node`, shares 2-3-3) decodes at about `10.21 tok/sec` vs `5.33` single-node, output matching the single-node stream; receipts in `docs/bench/phase4_tp_wire.md`. |
| Mistral 7B Instruct v0.1 | `TheBloke/Mistral-7B-Instruct-v0.1-GGUF`, `mistral-7b-instruct-v0.1.Q4_0.gguf` | reports `llama` in tested GGUF | Supported for tested row | `ready`; GGUF has no metadata chat template, so NanoCamelid uses `mistral_inst_token_fallback`; direct and chat generation pass; exact scalar-vs-selected parity; 512/1024/2048/4096/8192 context packs pass |
| Qwen2.5-Coder 7B Instruct | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`, `qwen2.5-coder-7b-instruct-q4_0.gguf` | `qwen2` | Supported | `ready`; `qwen_im`; direct and chat generation pass; exact scalar-vs-selected parity; 512/1024/2048/4096/8192 context packs pass |
| Strand Rust Coder 14B | `Fortytwo-Network/Strand-Rust-Coder-14B-v1-GGUF`, `Fortytwo_Strand-Rust-Coder-14B-v1-Q6_K.gguf` | `qwen2` | Supported three-Pi cluster chat | Single-Pi capped-context smoke works but is slow, about `0.17 tok/sec`; current-head three-Pi `master-chat` with Q6_K SDOT and pre-unpacked batched Q6_K weights validates the `0..16`, `16..32`, `32..48` split, ingested the prompt in about `8.38s`, and generated 6 tokens at about `1.28 tok/sec` |
| Mixtral 8x7B Instruct v0.1 | `mixtral-8x7b-instruct-v0.1.Q4_0.gguf` | `llama` MoE | Supported three-Pi cluster chat | `inspect` reports `ready`; three-Pi `master-chat` on the `0..11`, `11..22`, `22..32` split decodes at about `0.79 tok/sec` (64 tokens, breakdown-backed; the earlier `1.26 tok/sec` short-run figure does not reproduce at current head — see `docs/bench/phase0_baseline.md`, spin-pool fix included). Single-Pi full generation OOMs on 16 GB Pi RAM. |
| Qwen2.5-Coder 32B Instruct | `qwen2.5-coder-32b-instruct-q4_0.gguf` | `qwen2` | Supported cluster/large-model smoke | Three-Pi smoke produced matching code-text tokens at about `0.56 tok/sec` |
| Llama 3 8B Instruct | `bartowski/Meta-Llama-3-8B-Instruct-GGUF`, `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` | `llama` | Supported single-Pi | `inspect` reports `ready`; untied output head; `llama3_instruct` renderer. Single Pi 5: raw generation of `"Paris, ..."` and chat reply `"Red, Yellow, Blue"` (stops at EOS); about `2.5 tok/sec` decode, peak RSS about `9.1 GiB` (fits 16 GB; exercises the Q4_K NEON kernel). Same `llama` path as the 3B and 70B rows. |
| Llama 3 70B Instruct | `Meta-Llama-3-70B-Instruct.Q4_0.gguf` | `llama` | Supported three-Pi cluster chat | `inspect` reports `ready`; missing GGUF `tokenizer.ggml.pre` is accepted for Llama BPE metadata. Three-Pi weighted tensor parallelism (`cluster_tp_node`, shares 2-3-3, shard-direct loading) decodes at about `0.685 tok/sec` (32 tokens); the pipeline path (`master-chat`, rebalanced `0..22`, `22..52`, `52..80` split) measures `0.160 tok/sec` at current head. Receipts in `docs/bench/phase4_tp_wire.md`. Single-Pi support is not claimed. |
| Phi-3-mini-4k / Phi-3.5-mini Instruct | `microsoft/Phi-3-mini-4k-instruct-gguf` (`Phi-3-mini-4k-instruct-q4.gguf`), `microsoft/Phi-3.5-mini-instruct-gguf` (`phi-3.5-mini-instruct-q4_0.gguf`) | `phi3` | Supported single-node | Fused `blk.*.attn_qkv.weight` / `blk.*.ffn_up.weight` split into `wq`/`wk`/`wv` and gate/up on quant-block rows at load; the fused tensors load **unswizzled** so the split works in the default (swizzle-on) config for `Q4_0` as well as K-quant. `phi3` renderer/template — the TinyLlama marker matcher excludes `<\|end\|>`, so a Phi-3 template carrying a `<\|system\|>` branch (Phi-3.5) is not mis-detected as TinyLlama. `<\|user\|>`/`<\|end\|>`/`<\|assistant\|>` (USER_DEFINED) tokenize to special ids `32010`/`32007`/`32001`, and `<\|end\|>`=`32007` is in the EOG set so chat stops. Verified: Phi-3-mini-4k Q4 on M4 (chat stops at 47 tok, `"Paris."`, ~15.6 tok/s); **Phi-3.5-mini Q4_0 on a Pi 5 (camelid2), default config** — weights load, renderer `phi3`, chat stops cleanly at 85 tok (`generation_status: ok`), `"Paris."`, ~4.0 tok/s. Large-context variants (Phi-3.5 is `131072`) need `NANOCAMELID_CONTEXT_LIMIT` to bound the KV cache. Receipt in `docs/bench/phi3_fused_qkv.md`. |

## Likely Compatible, Test Next

These should be prioritized because they fit the already implemented dense
`llama` or `qwen2` paths. Each still needs a Pi `inspect` plus short generation
smoke before being promoted to supported.

| Candidate | Why it should fit | First test to run |
| --- | --- | --- |
| Qwen2.5 1.5B Instruct Q4_0 | Dense `qwen2`; same tokenizer family as the 0.5B rows | `inspect`, then 8-token `generate` |
| Qwen2.5-Coder 1.5B Instruct Q4_0 | Dense `qwen2`; likely best small coding row | `inspect`, then code prompt smoke |
| Qwen2.5 3B Instruct Q4_0 | Dense `qwen2`; still practical on Pi storage/RAM with context cap | `inspect`, then 8-token `generate` |
| Qwen2.5-Coder 3B Instruct Q4_0 | Dense `qwen2`; likely good balanced coding target | `inspect`, then code prompt smoke |
| DeepSeek-R1-Distill-Qwen 7B Q4_0 | Dense `qwen2`; tokenizer alias now accepted | `inspect`, then short reasoning prompt |
| DeepSeek-R1-Distill-Qwen 14B Q4_0/Q6_K | Dense `qwen2`; likely runs but may be slow | capped-context smoke after 7B |
| DeepSeek-R1-Distill-Llama 8B Q4_0 | Dense `llama`; should resemble Llama 3.x path | `inspect`, then 4-8 token smoke |
| TinyLlama 1.1B Chat Q4_0/Q8_0 | Dense `llama`; useful tiny regression row | `inspect`, then chat smoke |
| Mistral 7B rows that report `general.architecture = "mistral"` | Metadata path is now accepted, but needs a real GGUF that reports `mistral` | `inspect` must show `architecture: mistral`, then generation smoke |

## Blocked or Not Yet Claimable

Do not present these as supported until the listed runtime gaps are closed.

| Family | Current status | Required work |
| --- | --- | --- |
| Mixtral / MoE Mistral broader support | One exact Mixtral Q4_0 row has supported three-Pi cluster chat coverage. Broader Mixtral-family and single-node support are not promoted yet. | Add parity checks against a reference runtime, optimize or lazy-load expert weights for single-node memory pressure, and broaden prompt coverage beyond the current short cluster smoke |
| Broader Gemma family | Gemma 3 1B IT Q4_0 is supported; broader Gemma rows are not claimed yet | Add row-specific smokes for each exact GGUF, especially larger Gemma rows with soft-capping or alternate tensor mixes |
| Broader Phi family | Phi-3-mini-4k (Q4) and Phi-3.5-mini (Q4_0) are supported single-node (see the supported table); Phi-3-medium and other variants are not row-claimed yet | Run `inspect` plus a stop-clean chat smoke on each exact GGUF; the fused-QKV/gate-up split (swizzled and plain), USER_DEFINED special tokens, and phi3-vs-tinyllama template routing are already in place. For `131072`-context variants set `NANOCAMELID_CONTEXT_LIMIT` |
| LFM2 family | LFM2 700M/1.2B/2.6B inspect enough metadata and tokenizer state to identify the rows, but generation is not supported yet because the architecture includes shortconv/hybrid blocks and lacks the dense Llama-style `output_norm.weight` contract | Add LFM2 shortconv/hybrid runtime support, then rerun the full support matrix |

## Promotion Checklist

Promote a row from candidate to supported only after:

1. `nanocamelid inspect <model.gguf>` reports `readiness: ready`.
2. Tensor layouts are `ok`.
3. Tokenizer loads without fallback errors.
4. A short generation smoke completes with exact command and token count recorded.
5. Throughput is recorded for the same hardware class and context cap.
6. Any large-model result clearly states whether it is single-node, clustered,
   prompt-level, or token-level only.
