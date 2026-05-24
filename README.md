# NanoCamelid

NanoCamelid is a small Rust inference runtime for running GGUF local chat
models on Raspberry Pi-class ARM64 hardware.

The current focus is simple: make local model inspection, Q8_0/Q4_0 validation,
and small-model smoke tests easy to run on a Pi. Performance work is
intentionally gated behind explicit commands and environment variables until it
has repeatable Pi evidence.

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

Inside the TUI, use `/model <path>` to load a different GGUF without restarting
the process. A successful switch resets the conversation and token counters. If
the new model fails to load, the current model stays active.

On a prepared Pi workspace with the Llama 3.2 1B Instruct Q4_0 or Q8_0 GGUF at
the default model path, start the interactive 1B chat directly:

```bash
./scripts/pi/chat-1b.sh
```

For the matching one-command 1B validation path on that same Pi workspace:

```bash
./scripts/pi/smoke-1b.sh
```

The launcher prefers the Pi-local Q4_0 model when present, falls back to Q8_0,
and defaults the block dot path to SDOT on Pi-class ARM64 hardware. It runs a
`smoke q8-chat` preflight before opening the TUI, so the 1B instruct path keeps
the scalar-vs-selected-kernel parity gate in front of interactive chat. It still
honors `NANOCAMELID_MODEL_GGUF` and `NANOCAMELID_Q8_DOT_KERNEL` if you want to
force a different model or kernel for comparison.

Optional arguments set temperature and maximum assistant output tokens:

```bash
./scripts/pi/chat-1b.sh 0.0 64
```

`smoke-1b.sh` uses the same model-selection and kernel defaults, but runs only
the smoke gate and exits. By default it runs the real instruct prompt path with
`q8-chat`, the prompt `Say hello in one sentence.`, and an 8-token response
budget. Optional arguments let you override the smoke kind, prompt, and token
budget directly:

```bash
./scripts/pi/smoke-1b.sh q8-chat "Say hello in one sentence." 8
./scripts/pi/smoke-1b.sh q8-model "Hello" 1
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

For GEMV scheduling experiments, `NANOCAMELID_MATMUL_MIN_ROWS` controls the
minimum output row count required before Q8_0, Q4_0, and Q6_K matmuls enter
Rayon. Smaller matrices run sequentially to avoid Rayon queue overhead. The
default is `128`, which was neutral-to-slightly-positive on the Pi 2 short
prompts tested. Set it to `1` to approximate the old always-parallel behavior.
`NANOCAMELID_RAYON_THREADS` controls the global Rayon worker count; by default
the CLI uses up to four workers and pins them to available CPU cores when the
platform exposes affinity controls.

Decode-time attention now uses ARM NEON helpers for Q/K dot products and V
weighted accumulation when running on `aarch64`, with scalar reference fallbacks
for portability. The TUI reuses matching prompt-prefix KV cache across chat
turns, so repeated conversation prefixes are not re-ingested. New prompt suffixes
are still processed with sequential token-by-token prefill; large new prompts on
Qwen2.5-Coder-7B Q4_0 are therefore prefill-bound today. Batched prefill with
32-64 token chunks is the next required step before claiming 4096-token or
8192-token Pi chat usability.

The activation quantization path uses a NEON max-abs scan on `aarch64`, while
preserving scalar rounding/clamping semantics for parity. Q4_0 and Q8_0 matmul
hot loops read activation blocks through validated raw block pointers to avoid
repeated slice-to-array conversion in the innermost loop.

## Tested Models

These rows reflect models that have been loaded and smoke-tested on Raspberry Pi
hardware with the current GGUF path. They are not broad family claims.

| Model | GGUF quant | Status | Notes |
| --- | --- | --- | --- |
| Llama 3.2 1B Instruct | Q4_0 | Working | Pi smoke passes with scalar-vs-selected-kernel logit parity and interactive TUI chat. |
| Llama 3.2 1B Instruct | Q8_0 | Working | Baseline path for Q8 validation and Q4 comparison. |
| Qwen2.5-Coder-7B-Instruct | Q4_0 | Smoke passing | Official Q4_0 GGUF loads, Qwen chat rendering runs, and Pi smoke/chat generation passes. Fused Q6_K output projection improved the short Qwen prompt from 1.55 to 1.90-1.93 tok/sec on Pi 2. |

## Pi Performance Snapshot

Latest clean Pi 2 serial chat timings from the current validated runs:

| Model | Quant | Prompt path | Result |
| --- | --- | --- | --- |
| Llama 3.2 1B Instruct | Q4_0 | 8-token short chat | Model load ~0.95-0.97s, generation ~1.96-1.97s, ~4.07-4.09 tok/sec. |
| Llama 3.2 1B Instruct | Q8_0 | Same 8-token short chat | Model load ~1.32s, generation ~2.21s, ~3.63 tok/sec. |
| Qwen2.5-Coder-7B-Instruct | Q4_0 | 8-token short chat | Same prompt improved from 1.55 tok/sec at `c6e6d67` to 1.90-1.93 tok/sec after fused Q6_K output projection; NEON attention helpers preserve the same short-prompt range. |
| Qwen2.5-Coder-7B-Instruct | Q4_0 | Repeated ~170-token prompt | Model load ~3.60s, generation 8 tokens in ~4.33s, ~1.85 tok/sec after prefill. Larger ~650-token and ~2500-token stress prompts hit timeout before decode, confirming sequential prefill as the blocker. |

The Q4_0 1B path is faster than Q8_0 on the same prompt, but the measured
end-to-end gain is currently about 1.12x, not the theoretical 1.8-2.0x memory
traffic ceiling. The next performance work is broader hot-path reduction beyond
the Q4/Q8 block dot kernel.

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
- Q4_0 loading and Q4_0 weight x Q8_0 activation matmul paths are available.
- Single-turn chat prompt rendering is available for recognized instruct templates.
- Interactive terminal chat is available with model/kernel, token, TTFT, and throughput telemetry.
- The TUI can switch GGUFs at runtime with `/model <path>`.
- The Pi 1B chat launcher defaults to the SDOT Q8 dot-product path when available and preserves scalar-vs-selected-kernel parity through the smoke gate.
- Q8_0 and Q4_0 model smoke validation is available for the tested GGUF rows above.
- Broader model support and performance claims require Pi-local artifacts and row-specific validation.

## More Details

- [Pi porting notes](docs/PI_PORTING.md)
- [Camelid porting map](docs/CAMELID_PORTING_MAP.md)

## License

NanoCamelid is licensed under the MIT License. See [LICENSE](LICENSE).
