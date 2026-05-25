# NanoCamelid Model Catalog

This catalog tracks GGUF model rows that NanoCamelid can honestly claim, rows
that are likely compatible with the current runtime, and model families that
need more runtime work before they should be promoted.

The current runtime supports dense Llama-style, Qwen2-style, Qwen3-style,
SmolLM, and Gemma 3 transformer blocks with GGUF tensor types used by the tested
rows below: `Q4_0`, `Q4_1`, `Q8_0`, and `Q6_K` where explicitly noted.
Long-context GGUFs should be smoke-tested with `NANOCAMELID_CONTEXT_LIMIT`
until full advertised-context memory behavior is validated.

## Supported and Pi-Smoked

These rows have been loaded and generated on Raspberry Pi-class ARM64 hardware.

| Model | GGUF row | Architecture | Status | Evidence |
| --- | --- | --- | --- | --- |
| Qwen2.5 0.5B Instruct | `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, `qwen2.5-0.5b-instruct-q4_0.gguf` | `qwen2` | Supported | `ready`; `qwen_im`; direct and chat generation pass; exact scalar-vs-selected parity; 512/1024/2048/4096/8192 context packs pass |
| Qwen2.5-Coder 0.5B Instruct | `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF`, `qwen2.5-coder-0.5b-instruct-q4_0.gguf` | `qwen2` | Supported | `ready`; `qwen_im`; direct and chat generation pass; exact scalar-vs-selected parity; 512/1024/2048/4096/8192 context packs pass |
| Qwen3 0.6B Instruct | `Qwen/Qwen3-0.6B-GGUF`, `qwen3-0.6b-q8_0.gguf` | `qwen3` | Supported | `ready`; `qwen_im`; direct and chat generation pass; exact scalar-vs-selected chat parity; 512/1024/2048/4096/8192 context packs pass; sampled RSS about `1.41 GiB` |
| Qwen3 1.7B Instruct | `Qwen/Qwen3-1.7B-GGUF`, `qwen3-1.7b-q8_0.gguf` | `qwen3` | Supported | `ready`; `qwen_im`; direct and chat generation pass; exact scalar-vs-selected chat parity; 512/1024/2048/4096/8192 context packs pass; sampled RSS about `3.56 GiB` |
| Qwen3 4B Instruct | `miku552/Qwen3-4B-Q4_0-GGUF`, `qwen3-4b-q4_0.gguf` | `qwen3` | Supported | `ready`; `qwen_im`; direct and chat generation pass; exact scalar-vs-selected chat parity; 512/1024/2048/4096/8192 context packs pass; sampled RSS about `5.50 GiB` |
| SmolLM3 3B | `jpohhhh/SmolLM3-3B-Q4_0-GGUF`, `smollm3-3b-q4_0.gguf` | `smollm3` | Supported | `ready`; ChatML token fallback renderer; direct and chat generation pass; exact scalar-vs-selected chat parity; 512/1024/2048/4096/8192 context packs pass; sampled RSS about `3.94 GiB` |
| SmolLM2 1.7B Instruct | `Maites/SmolLM2-1.7B-Instruct-Q4_0-GGUF`, `smollm2-1.7b-instruct-q4_0.gguf` | `llama` | Supported | `ready`; `qwen_im`; chat generation passes; direct prompt ended immediately; exact scalar-vs-selected chat parity; 512/1024/2048/4096/8192 context packs pass; sampled RSS about `2.02 GiB` |
| Gemma 3 1B IT | `unsloth/gemma-3-1b-it-GGUF`, `gemma-3-1b-it-q4_0.gguf` | `gemma3` | Supported | `ready`; `gemma_turn`; Q4_1 feed-forward tensors supported; direct and chat generation pass; exact scalar-vs-selected chat parity; 512/1024/2048/4096/8192 context packs pass; sampled RSS about `2.20 GiB` |
| DeepSeek-R1-Distill-Qwen 1.5B | `ggml-org/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF`, `deepseek-r1-distill-qwen-1.5b-q4_0.gguf` | `qwen2` | Supported | `ready`; `deepseek_r1_qwen`; direct and chat generation pass; exact scalar-vs-selected parity; 512/1024/2048/4096/8192 context packs pass |
| Llama 3.2 1B Instruct | `Llama-3.2-1B-Instruct-Q4_0.gguf` | `llama` | Supported | `ready`; `llama3_instruct`; direct and chat generation pass; exact scalar-vs-selected parity; 512/1024/2048/4096/8192 context packs pass |
| Llama 3.2 1B Instruct | `Llama-3.2-1B-Instruct-Q8_0.gguf` | `llama` | Supported baseline | `ready`; `llama3_instruct`; Q8 baseline for validation and Q4 comparison; exact scalar-vs-selected parity; 512/1024/2048/4096/8192 context packs pass |
| Llama 3.2 3B Instruct | `Llama-3.2-3B-Instruct-Q4_0.gguf` | `llama` | Supported | `ready`; `llama3_instruct`; direct and chat generation pass; exact scalar-vs-selected parity; capped 8096 TUI launch and 512/1024/2048/4096/8192 context packs pass |
| Mistral 7B Instruct v0.1 | `TheBloke/Mistral-7B-Instruct-v0.1-GGUF`, `mistral-7b-instruct-v0.1.Q4_0.gguf` | reports `llama` in tested GGUF | Supported for tested row | `ready`; GGUF has no metadata chat template, so NanoCamelid uses `mistral_inst_token_fallback`; direct and chat generation pass; exact scalar-vs-selected parity; 512/1024/2048/4096/8192 context packs pass |
| Qwen2.5-Coder 7B Instruct | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`, `qwen2.5-coder-7b-instruct-q4_0.gguf` | `qwen2` | Supported | `ready`; `qwen_im`; direct and chat generation pass; exact scalar-vs-selected parity; 512/1024/2048/4096/8192 context packs pass |
| Strand Rust Coder 14B | `Fortytwo-Network/Strand-Rust-Coder-14B-v1-GGUF`, `Fortytwo_Strand-Rust-Coder-14B-v1-Q6_K.gguf` | `qwen2` | Supported but slow | README-documented capped-context smoke, about `0.17 tok/sec` |
| Mixtral 8x7B Instruct v0.1 | `mixtral-8x7b-instruct-v0.1.Q4_0.gguf` | `llama` MoE | Experimental cluster chat smoke | `inspect` reports `ready`; three-Pi `master-chat` rendered the `[INST]` prompt and generated 8 tokens at about `1.12 tok/sec`. Single-Pi full generation OOMs on 16 GB Pi RAM. |
| Qwen2.5-Coder 32B Instruct | `qwen2.5-coder-32b-instruct-q4_0.gguf` | `qwen2` | Supported cluster/large-model smoke | Three-Pi smoke produced matching code-text tokens at about `0.56 tok/sec` |
| Llama 3 70B Instruct | `Meta-Llama-3-70B-Instruct.Q4_0.gguf` | `llama` | Token-level cluster smoke only | Three-Pi token-level smoke generated two tokens at about `0.17 tok/sec`; prompt tokenizer path still needs full support |

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
| Mixtral / MoE Mistral broader support | One exact Mixtral Q4_0 row has experimental three-Pi chat smoke coverage. Broader Mixtral support is not promoted yet. | Add parity checks against a reference runtime, optimize or lazy-load expert weights for single-node memory pressure, and broaden prompt coverage beyond the current short cluster smoke |
| Broader Gemma family | Gemma 3 1B IT Q4_0 is supported; broader Gemma rows are not claimed yet | Add row-specific smokes for each exact GGUF, especially larger Gemma rows with soft-capping or alternate tensor mixes |
| Phi family | Phi-3.5 Mini Instruct Q4_0 inspects, but generation is not supported yet because the tested GGUF uses fused `blk.*.attn_qkv.weight` tensors | Add fused-QKV load/runtime splitting and parity tests before promoting |
| LFM2 family | LFM2 700M/1.2B/2.6B inspect enough metadata and tokenizer state to identify the rows, but generation is not supported yet because the architecture includes shortconv/hybrid blocks and lacks the dense Llama-style `output_norm.weight` contract | Add LFM2 shortconv/hybrid runtime support, then rerun the full support matrix |
| Llama 3 70B prompt-level chat | Partial only | Token-level cluster smoke works; full prompt text path needs tokenizer compatibility for the exact 70B GGUF metadata |

## Promotion Checklist

Promote a row from candidate to supported only after:

1. `nanocamelid inspect <model.gguf>` reports `readiness: ready`.
2. Tensor layouts are `ok`.
3. Tokenizer loads without fallback errors.
4. A short generation smoke completes with exact command and token count recorded.
5. Throughput is recorded for the same hardware class and context cap.
6. Any large-model result clearly states whether it is single-node, clustered,
   prompt-level, or token-level only.
