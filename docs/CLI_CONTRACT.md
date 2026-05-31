# NanoCamelid v0.1 CLI Contract

This page defines the stable customer-facing command surface for NanoCamelid
v0.1. Commands outside this list may remain available for compatibility,
validation, and lab workflows, but scripts and docs should prefer the contract
below.

## Stable Commands

| Command | Contract | Primary output |
| --- | --- | --- |
| `nanocamelid --version` | Print the packaged NanoCamelid version and exit without loading model files | Plain version text |
| `nanocamelid doctor` | Check install readiness, host summary, model directory, default model paths, and next action without loading a GGUF | Human-readable preflight; `--json` adds a machine-readable JSON status line |
| `nanocamelid probe` | Print host CPU and runtime feature detection | Human-readable host/runtime feature report |
| `nanocamelid models list` | List `.gguf` files directly under the configured model directory | Human-readable list; `--json` emits JSON lines |
| `nanocamelid models scan` | Recursively find `.gguf` files and classify filename target/quantization hints | Human-readable scan; `--json` emits JSON lines |
| `nanocamelid models inspect <model.gguf\|1b\|3b>` | Inspect GGUF metadata through the stable models namespace | Human-readable model audit; `--dry-run` prints the resolved plan |
| `nanocamelid ready 1b` | Run the bounded 1B readiness gate: probe, strict audit, inspect, smoke, and optional direct chat | Human-readable gate steps plus JSON status lines |
| `nanocamelid chat <model.gguf\|1b\|3b> <prompt>` | Run one chat-template-rendered generation turn | Generated text plus machine-readable status |
| `nanocamelid tui <model.gguf\|1b\|3b>` | Open the terminal chat UI for repeated local chat | Interactive terminal UI |
| `nanocamelid serve` | Run the local HTTP API server on loopback by default | HTTP API: `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/metrics` |

## Shared Inputs

- Model directory defaults to `/mnt/nanocamelid/models`.
- `NANOCAMELID_MODEL_DIR` overrides the model directory for discovery and the
  local API server.
- `NANOCAMELID_WORKSPACE` changes the Pi workspace used to derive default model
  paths when `NANOCAMELID_MODEL_DIR` is unset.
- Explicit `.gguf` paths override aliases and environment defaults for commands
  that accept a model argument.
- `1b` and `3b` aliases resolve to documented Llama 3.2 default rows.

## Exit and Error Behavior

- `--help` is supported for every stable command namespace and prints usage,
  options, and relevant environment variables.
- `--dry-run` commands print the resolved plan without loading model weights or
  binding sockets.
- Missing model directories should fail with an actionable message that names
  the directory and points to `--dir` or `NANOCAMELID_MODEL_DIR`.
- Invalid numeric flags should name the failing flag and accepted range.
- Local API errors are structured JSON with an `error.code` and
  `error.message`.

## Compatibility Commands

The top-level `inspect`, `model`, `generate`, `smoke`, `evidence`, and `bench`
commands remain available because they back validation, Pi evidence, and
performance work. They are compatibility surfaces in v0.1, not the primary
customer command contract. Prefer the stable namespace above when adding public
README examples, service docs, or installer output.
