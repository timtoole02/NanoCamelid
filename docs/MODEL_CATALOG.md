# NanoCamelid Model Catalog

This catalog tracks GGUF model rows that NanoCamelid can honestly claim, rows
that are likely compatible with the current runtime, and model families that
need more runtime work before they should be promoted.

The current runtime supports dense Llama-style and Qwen2-style transformer
blocks with GGUF tensor types used by the tested rows below: `Q4_0`, `Q8_0`,
and `Q6_K` where explicitly noted. Long-context GGUFs should be smoke-tested
with `NANOCAMELID_CONTEXT_LIMIT` until full advertised-context memory behavior
is validated.

## Supported and Pi-Smoked

These rows have been loaded and generated on Raspberry Pi-class ARM64 hardware.

| Model | GGUF row | Architecture | Status | Evidence |
| --- | --- | --- | --- | --- |
| Qwen2.5 0.5B Instruct | `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, `qwen2.5-0.5b-instruct-q4_0.gguf` | `qwen2` | Supported | `ready`; 8-token smoke at about `33.31 tok/sec` |
| Qwen2.5-Coder 0.5B Instruct | `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF`, `qwen2.5-coder-0.5b-instruct-q4_0.gguf` | `qwen2` | Supported | `ready`; 8-token smoke at about `33.28 tok/sec` |
| DeepSeek-R1-Distill-Qwen 1.5B | `ggml-org/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF`, `deepseek-r1-distill-qwen-1.5b-q4_0.gguf` | `qwen2` | Supported | `ready`; 8-token smoke at about `13.25 tok/sec` |
| Llama 3.2 1B Instruct | `Llama-3.2-1B-Instruct-Q4_0.gguf` | `llama` | Supported | README-documented Pi chat/generate smoke at about `4.18 tok/sec` |
| Llama 3.2 1B Instruct | `Llama-3.2-1B-Instruct-Q8_0.gguf` | `llama` | Supported baseline | Existing Q8_0 smoke row; useful for correctness checks |
| Llama 3.2 3B Instruct | `Llama-3.2-3B-Instruct-Q4_0.gguf` | `llama` | Supported | `ready`; Llama 3 chat renderer; smoke chat generated `"Hello!"` with exact scalar-vs-selected logit parity; direct generation at about `2.22 tok/sec` |
| Mistral 7B Instruct v0.1 | `TheBloke/Mistral-7B-Instruct-v0.1-GGUF`, `mistral-7b-instruct-v0.1.Q4_0.gguf` | reports `llama` in tested GGUF | Supported for tested row | `ready`; 4-token smoke at about `3.68 tok/sec` |
| Qwen2.5-Coder 7B Instruct | Q4_0 GGUF row | `qwen2` | Supported smoke row | README-documented scalar-vs-selected logit parity and `"Hello"` smoke output |
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
| Gemma 2 / Gemma family | Not supported yet | Add Gemma metadata mapping, GeLU/GELU-ish activation path, Gemma RMSNorm behavior, and attention logit soft-capping where required |
| Phi family | Not supported yet | Inspect GGUF architecture/tensor names, add architecture-specific config and tokenizer support |
| Qwen3 | Not supported by claim yet | Inspect architecture key and tokenizer metadata; add/validate if it remains compatible with current Qwen2 math |
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
