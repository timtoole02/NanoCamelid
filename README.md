# NanoCamelid

NanoCamelid is a small Rust inference runtime for running GGUF Llama-style
models on Raspberry Pi-class ARM64 hardware.

The current focus is simple: make local model inspection, Q8_0 validation, and
small-model smoke tests easy to run on a Pi. Performance work is intentionally
gated behind explicit commands and environment variables until it has repeatable
Pi evidence.

## Requirements

- Raspberry Pi 5 or another ARM64 Linux machine
- Rust toolchain
- A local GGUF model file

## Quick Start

```bash
git clone https://github.com/timtoole02/NanoCamelid.git
cd NanoCamelid

cargo run -- probe
cargo run -- inspect /path/to/model.gguf
cargo run --release -- smoke q8-model /path/to/model.gguf "Hello" 1
cargo run --release -- smoke q8-chat /path/to/model.gguf "Say hello in one sentence." 8
NANOCAMELID_MODEL_GGUF=/path/to/model.gguf cargo run -- inspect
NANOCAMELID_MODEL_GGUF=/path/to/model.gguf cargo run --release -- generate "Hello" 0.0 32
NANOCAMELID_MODEL_GGUF=/path/to/model.gguf cargo run --release -- chat "Say hello in one sentence." 0.0 32
NANOCAMELID_MODEL_GGUF=/path/to/model.gguf cargo run --release -- tui 0.0 64
NANOCAMELID_SMOKE_GGUF=/path/to/model.gguf cargo run --release -- smoke q8-model "Hello" 1
NANOCAMELID_SMOKE_GGUF=/path/to/model.gguf cargo run --release -- smoke q8-chat "Say hello in one sentence." 8
```

`probe` prints CPU and SIMD feature information. `inspect` reads GGUF metadata
and tensor layout. `smoke q8-model` loads a Q8_0 model, checks scalar/runtime
logit parity, and runs a short greedy generation path from directly tokenized
prompt text. `smoke q8-chat` runs the same parity/generation validation through
the tokenizer chat template so Llama 3.2 1B Instruct rows can be smoke-tested
through the real instruct prompt path. Set `NANOCAMELID_MODEL_GGUF` to reuse
the same 1B GGUF path across repeated `inspect`, `generate`, and `chat` runs,
or `NANOCAMELID_SMOKE_GGUF` to override that shared default just for smoke
validation. `chat` renders a single-turn user prompt through recognized
tokenizer chat templates, including the Llama 3 instruct header/eot format
used by Llama 3.2 1B Instruct rows.
`tui` opens an interactive terminal chat that keeps the model loaded, shows the
connected model path/name, selected Q8 kernel, chat renderer, and per-turn plus
session token-in/token-out counters, TTFT, and throughput.

![NanoCamelid terminal chat showing model telemetry and token counters](docs/images/nanocamelid-tui.png)

On a prepared Pi workspace with the Llama 3.2 1B Instruct Q8_0 GGUF at the
default model path, start the interactive 1B chat directly:

```bash
./scripts/pi/chat-1b.sh
```

The launcher defaults the Q8 dot-product path to NEON on Pi-class ARM64
hardware and runs a `smoke q8-chat` preflight before opening the TUI, so the
1B instruct path keeps the scalar-vs-selected-kernel parity gate in front of
interactive chat. It still honors `NANOCAMELID_Q8_DOT_KERNEL` if you want to
force a different kernel for comparison.

Optional arguments set temperature and maximum assistant output tokens:

```bash
./scripts/pi/chat-1b.sh 0.0 64
```

For faster local iteration, disable the preflight smoke gate explicitly:

```bash
NANOCAMELID_CHAT_SMOKE=0 ./scripts/pi/chat-1b.sh
```

The preflight smoke defaults to `q8-chat` with a one-token response budget, and
you can override the gate with:

- `NANOCAMELID_CHAT_SMOKE_KIND=q8-model|q8-chat`
- `NANOCAMELID_CHAT_SMOKE_PROMPT="..."`
- `NANOCAMELID_CHAT_SMOKE_TOKENS=1`

## Benchmarks

Run the Q8 dot benchmark on the target Pi:

```bash
cargo run --release -- bench q8-dot 1000 3
```

To test the default-off SDOT candidate when the CPU supports it:

```bash
NANOCAMELID_Q8_DOT_KERNEL=sdot \
NANOCAMELID_Q8_DOT_SDOT=1 \
cargo run --release -- bench q8-dot 1000 3
```

The benchmark prints repeated scalar/NEON timing and a JSON summary line. Treat
benchmark output as specific to the exact Pi, model, build, and configuration
where it was captured.

## Raspberry Pi Deployment

Prepare a Pi workspace:

```bash
./scripts/pi/bootstrap.sh
```

Build and test remotely:

```bash
./scripts/remote_build.sh <pi-host> [ssh-key] [pi-user]
```

To include a model-backed smoke test, point the script at a GGUF path that
already exists on the Pi:

```bash
NANOCAMELID_REMOTE_SMOKE_GGUF=/path/on/pi/model.gguf \
./scripts/remote_build.sh <pi-host> [ssh-key] [pi-user]
```

To run the instruct/chat smoke path instead of the raw prompt smoke:

```bash
NANOCAMELID_REMOTE_SMOKE_KIND=q8-chat \
NANOCAMELID_REMOTE_SMOKE_GGUF=/path/on/pi/model.gguf \
./scripts/remote_build.sh <pi-host> [ssh-key] [pi-user]
```

Deployment defaults to rsync snapshots. Advanced deployment modes are available
in the scripts for development workflows.

## Project Status

- Host feature probing is available.
- GGUF metadata and tensor layout inspection are available.
- Q8_0 scalar, NEON, and default-off SDOT dot-product paths are available.
- Single-turn chat prompt rendering is available for recognized instruct templates.
- Interactive terminal chat is available with model/kernel, token, TTFT, and throughput telemetry.
- The Pi 1B chat launcher defaults to the NEON Q8 dot-product path and preserves scalar-vs-NEON parity through the smoke gate.
- Q8_0 model smoke validation is available for supported Llama-style GGUFs.
- Broader model support and performance claims require Pi-local artifacts.

## More Details

- [Pi porting notes](docs/PI_PORTING.md)
- [Camelid porting map](docs/CAMELID_PORTING_MAP.md)

## License

NanoCamelid is licensed under the MIT License. See [LICENSE](LICENSE).
