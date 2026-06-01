# NanoCamelid v0.1 Support Matrix

This matrix is the product-facing summary of what NanoCamelid v0.1 can claim
today. Model-row evidence and promotion rules live in
[MODEL_CATALOG.md](MODEL_CATALOG.md).

## Product Surfaces

| Surface | v0.1 status | Validation evidence |
| --- | --- | --- |
| Release installer | Supported for published aarch64 Linux releases | `scripts/install.sh` defaults to versioned GitHub release artifacts and verifies `SHA256SUMS`; dev/source mode is explicit |
| Release package | Supported for aarch64 Linux | GitHub release workflow runs `./scripts/validate.sh` before release packaging, then `scripts/package-release.sh` builds the explicit `aarch64-unknown-linux-gnu` target and packages the binary, README, license, release notes, and checksums |
| Version output | Supported | `nanocamelid --version` prints the Cargo package version and is covered by validation |
| Stable CLI contract | Supported | `docs/CLI_CONTRACT.md` defines the v0.1 product command surface: `doctor`, `probe`, `models list`, `models scan`, `models inspect`, `ready`, `chat`, `tui`, and `serve`; `scripts/validate.sh` runs a stable help sweep for every command in that surface; compatibility commands remain available for validation and lab workflows |
| Doctor preflight | Supported | `nanocamelid doctor` reports version, host summary, model directory, default 1B/3B paths, and next action without loading a GGUF; `--json` mirrors the actionable readiness fields for automation |
| Model discovery | Supported | `nanocamelid models list` and `nanocamelid models scan` enumerate `.gguf` files, classify filename target/quantization hints, report active `1b`/`3b` aliases for default rows, and emit JSON summaries |
| Model inspection | Supported | `nanocamelid models inspect` reads GGUF metadata, tensor layouts, tokenizer readiness, and strict 1B shape status through the stable namespace; top-level `inspect` remains a compatibility command |
| Readiness gate | Supported for Llama 3.2 1B aliases | `nanocamelid ready 1b` runs probe, strict shape audit, inspect, smoke, and optional direct chat |
| Single-turn chat | Supported for promoted catalog rows | `nanocamelid chat` renders recognized tokenizer chat templates and prints machine-readable generation status |
| Terminal TUI | Supported for promoted catalog rows | `nanocamelid tui` keeps a model loaded for repeated local chat and exposes `/models`, `/status`, `/history`, and transcript save commands |
| Smoke/evidence bundle | Supported for 1B product gate | `nanocamelid smoke 1b`, `evidence 1b`, and Pi scripts provide bounded dry-run and Pi-local validation paths |
| Local API server | Supported | `serve` defaults to `127.0.0.1`, exposes `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/metrics`, optional bearer-token auth on loopback, required bearer-token auth for non-loopback binds, browser preflight responses on known API paths, explicit request/input/output cap settings, dry-run planning, and structured JSON errors; `docs/API.md` documents the public endpoint contract; local validation smokes auth, discovery, metrics, CORS preflight, known-path method guards, POST validation errors, output caps, and not-found responses; `/v1/completions` resolves model ids/aliases/paths and returns OpenAI-shaped text-completion JSON; `/v1/chat/completions` renders supported tokenizer chat templates and returns OpenAI-shaped chat-completion JSON; input caps are enforced with tokenizer-backed prompt lengths before weights load |
| Service mode | Supported for systemd user services | `scripts/install-systemd-user-service.sh` writes a local-by-default user service with explicit API caps, optional bearer-token EnvironmentFile, non-loopback auth enforcement, dry-run planning, and basic systemd hardening; loopback units keep `IPAddressAllow=localhost`, while authenticated non-loopback units use `IPAddressAllow=any` so the systemd network policy matches the requested bind; launchd and system services are not claimed |
| Observability | Supported for local API basics | `/metrics` reports total accepted requests, completed response counts by status bucket, uptime, and configured request/input/output caps; the validation smoke checks authenticated metrics output and response-status counters |

## Model Rows

| Row class | v0.1 status | Evidence source |
| --- | --- | --- |
| Llama 3.2 1B Instruct Q4_0/Q8_0 | Supported product defaults | Strict 1B shape audit, readiness, chat smoke, context packs, and Pi-side evidence in `MODEL_CATALOG.md` |
| Llama 3.2 3B Instruct Q4_0 | Supported promoted row | Pi smoke and context-pack evidence in `MODEL_CATALOG.md` |
| Qwen2/Qwen3/SmolLM/Gemma promoted dense rows | Supported promoted rows | Row-specific Pi smoke/parity evidence in `MODEL_CATALOG.md` |
| Cluster-only large rows | Experimental support | Exact-row three-Pi evidence exists, but single-node product support is not claimed |
| Candidate rows | Not yet supported | Must pass the promotion checklist in `MODEL_CATALOG.md` before product claims change |

## v0.1 Gate Still Open

- Add OS-level service variants beyond systemd user services only after
  product installs need them.
- Keep broad performance claims behind row-specific Pi evidence.
