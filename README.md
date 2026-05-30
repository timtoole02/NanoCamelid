# NanoCamelid

NanoCamelid is a compact Rust inference runtime for running local GGUF chat
models on Raspberry Pi-class ARM64 hardware. It is built for inspectable edge
inference: one binary, local model files, terminal chat, repeatable smoke
checks, and Pi-side evidence for supported model rows.

## Current v0.1 Shape

- Release installer defaults to versioned GitHub releases and verifies
  `SHA256SUMS`.
- `nanocamelid --version` prints the package version.
- GGUF inspection, direct generation, single-turn chat, terminal TUI, 1B
  readiness, and model-backed smoke/evidence paths are available from the CLI.
- The default Pi model directory is `/mnt/nanocamelid/models`.
- Llama, Qwen, ChatML, Mistral, DeepSeek-R1-Qwen, and Gemma prompt rendering is
  available for supported smoke and chat paths.
- Supported model claims live in [docs/MODEL_CATALOG.md](docs/MODEL_CATALOG.md)
  and are backed by Pi-side evidence.

## 5-Minute Quickstart

Install the current release on an ARM64 Linux host:

```bash
curl -fsSL https://raw.githubusercontent.com/timtoole02/NanoCamelid/main/scripts/install.sh | bash
nanocamelid --version
nanocamelid probe
```

Place a GGUF model under `/mnt/nanocamelid/models`, or point commands at an
explicit `.gguf` file. For the default Llama 3.2 1B path, use one of:

```bash
/mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf
/mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf
```

Run a dry-run readiness plan before loading the model:

```bash
nanocamelid ready 1b --dry-run
```

Then run the readiness gate and open chat:

```bash
nanocamelid ready 1b --no-chat
nanocamelid chat 1b "Say hello in one sentence." 0.0 8
nanocamelid tui 1b
```

## Install Modes

The installer uses the published aarch64 Linux release by default:

```bash
curl -fsSL https://raw.githubusercontent.com/timtoole02/NanoCamelid/main/scripts/install.sh | \
  bash -s -- --version v0.1.0
```

Source installs are explicit dev mode:

```bash
curl -fsSL https://raw.githubusercontent.com/timtoole02/NanoCamelid/main/scripts/install.sh | \
  bash -s -- --dev
```

On macOS, dev-mode builds require `CARGO_TARGET_DIR` or
`NANOCAMELID_TARGET_DIR` to point at an external `/Volumes` path so local builds
do not create large Cargo artifacts on the internal disk. Run
`./scripts/install.sh --dry-run` from a checkout to inspect the resolved plan.

## Core CLI

```bash
nanocamelid --version
nanocamelid probe
nanocamelid model 1b --dry-run
nanocamelid inspect 1b --dry-run
nanocamelid inspect /path/to/model.gguf
nanocamelid smoke 1b --dry-run
nanocamelid ready 1b --dry-run
nanocamelid evidence 1b --dry-run
nanocamelid generate /path/to/model.gguf "Hello" 0.0 32
nanocamelid chat /path/to/model.gguf "Say hello in one sentence." 0.0 32
nanocamelid tui /path/to/model.gguf
```

Run `nanocamelid help` or `nanocamelid <command> --help` for command-specific
arguments and environment controls.

## Documentation

- [docs/MODEL_CATALOG.md](docs/MODEL_CATALOG.md): supported model rows and next
  candidates.
- [docs/PRODUCT_HISTORY.md](docs/PRODUCT_HISTORY.md): detailed prototype
  history, Pi evidence, performance notes, and advanced launchers.
- [docs/PI_PORTING.md](docs/PI_PORTING.md): Pi deployment and validation notes.
- [docs/HIGH_PERFORMANCE_INFERENCE_WALKTHROUGH.md](docs/HIGH_PERFORMANCE_INFERENCE_WALKTHROUGH.md):
  architecture and benchmark walkthrough.

## Validation

Use the standard local validation gate from a checkout:

```bash
NANOCAMELID_TARGET_DIR="/Volumes/External/nanocamelid-target" ./scripts/validate.sh
./scripts/validate.sh --dry-run
```

On prepared Pi workspaces, the same script defaults to
`/mnt/nanocamelid/target`.
