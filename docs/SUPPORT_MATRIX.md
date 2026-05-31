# NanoCamelid v0.1 Support Matrix

This matrix is the product-facing summary of what NanoCamelid v0.1 can claim
today. Model-row evidence and promotion rules live in
[MODEL_CATALOG.md](MODEL_CATALOG.md).

## Product Surfaces

| Surface | v0.1 status | Validation evidence |
| --- | --- | --- |
| Release installer | Supported for published aarch64 Linux releases | `scripts/install.sh` defaults to versioned GitHub release artifacts and verifies `SHA256SUMS`; dev/source mode is explicit |
| Release package | Supported for aarch64 Linux | GitHub release workflow and `scripts/package-release.sh` package the binary, README, license, release notes, and checksums |
| Version output | Supported | `nanocamelid --version` prints the Cargo package version and is covered by validation |
| Doctor preflight | Supported | `nanocamelid doctor` reports version, host summary, model directory, default 1B/3B paths, and next action without loading a GGUF |
| Model discovery | Supported | `nanocamelid models list` and `nanocamelid models scan` enumerate `.gguf` files, classify filename target/quantization hints, and emit JSON summaries |
| Model inspection | Supported | `nanocamelid inspect` and `nanocamelid models inspect` read GGUF metadata, tensor layouts, tokenizer readiness, and strict 1B shape status |
| Readiness gate | Supported for Llama 3.2 1B aliases | `nanocamelid ready 1b` runs probe, strict shape audit, inspect, smoke, and optional direct chat |
| Single-turn chat | Supported for promoted catalog rows | `nanocamelid chat` renders recognized tokenizer chat templates and prints machine-readable generation status |
| Terminal TUI | Supported for promoted catalog rows | `nanocamelid tui` keeps a model loaded for repeated local chat and exposes `/models`, `/status`, `/history`, and transcript save commands |
| Smoke/evidence bundle | Supported for 1B product gate | `nanocamelid smoke 1b`, `evidence 1b`, and Pi scripts provide bounded dry-run and Pi-local validation paths |
| Local API server | Partial | `serve` defaults to `127.0.0.1`, exposes `/health`, `/v1/models`, `/v1/completions`, `/metrics`, optional bearer-token auth, explicit request/input/output cap settings, dry-run planning, and structured JSON errors; `/v1/completions` resolves model ids/aliases/paths and returns OpenAI-shaped text-completion JSON; `/v1/chat/completions` validates request JSON and caps before returning a deliberate not-implemented JSON error |
| Service mode | Planned | systemd/launch supervisor defaults and service hardening are not yet implemented |

## Model Rows

| Row class | v0.1 status | Evidence source |
| --- | --- | --- |
| Llama 3.2 1B Instruct Q4_0/Q8_0 | Supported product defaults | Strict 1B shape audit, readiness, chat smoke, context packs, and Pi-side evidence in `MODEL_CATALOG.md` |
| Llama 3.2 3B Instruct Q4_0 | Supported promoted row | Pi smoke and context-pack evidence in `MODEL_CATALOG.md` |
| Qwen2/Qwen3/SmolLM/Gemma promoted dense rows | Supported promoted rows | Row-specific Pi smoke/parity evidence in `MODEL_CATALOG.md` |
| Cluster-only large rows | Experimental support | Exact-row three-Pi evidence exists, but single-node product support is not claimed |
| Candidate rows | Not yet supported | Must pass the promotion checklist in `MODEL_CATALOG.md` before product claims change |

## v0.1 Gate Still Open

- Wire validated `/v1/chat/completions` requests to bounded chat-template
  generation and replace approximate request-token caps with tokenizer-backed
  input caps for both completion routes.
- Add service-mode packaging and security defaults after the API completion
  path exists.
- Keep broad performance claims behind row-specific Pi evidence.
