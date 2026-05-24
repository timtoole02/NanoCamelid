# NanoCamelid

NanoCamelid is a compact Rust inference runtime for running GGUF local chat
models on Raspberry Pi-class ARM64 hardware.

It is not a wrapper around a desktop inference stack. The current goal is a
small, inspectable runtime that can load local GGUF files, run model smoke tests,
chat in a terminal, and make every performance claim traceable to Pi-side
evidence.

## Current State

- GGUF metadata and tensor layout inspection are available.
- Q8_0, Q4_0, and Q6_K tensor paths are implemented for the tested rows below.
- Llama and Qwen2 chat-template rendering is available for smoke tests and chat.
- The terminal TUI keeps the model loaded and reuses matching KV-cache prefixes
  across turns.
- Prompt ingestion uses guarded batched prefill by default. The current default
  batch size is `16`.
- Long-context models can be smoke-tested with an explicit
  `NANOCAMELID_CONTEXT_LIMIT` cap to avoid allocating their full advertised KV
  cache.
- Scalar reference paths remain in the test suite. Optimized kernels have to
  prove parity before they are treated as production paths.

## Recent Pi Results

Latest runtime evidence below was captured through `b467d2c`
(`feat(runtime): allow capped smoke context`).

On the Pi 2 benchmark lane, Qwen2.5-Coder-7B-Instruct Q4_0 currently validates
through the smoke path with exact scalar-vs-selected logit parity:

- `max_logit_delta: 0.00000000`
- generated smoke text: `"Hello"`
- default prefill batch: `16`

The main prefill improvement came from loop-inverted batched Q4 prefill:

- batch 1, 145-token Qwen prompt: `48.90s`
- batch 16 before loop inversion: `31.38s`
- batch 16 after loop inversion: `17.04s`
- default batch 16 real-model check: about `17.0s`

Synthetic Q4 prefill tuning on the same Pi showed batch 32 slightly ahead in
the isolated benchmark, but the real Qwen chat path favored batch 16, so `16` is
the production default.

Recent experiments that did not land:

- Register-accumulated attention was correct but did not improve the short Qwen
  decode run (`1.88 tok/sec` baseline vs `1.87 tok/sec` experiment).
- Fully vectorized activation quantization was correctness-safe but slightly
  slower on the real 7B Q4 generate path (`4.05s` prefill and `1.34 tok/sec`
  baseline vs `4.07s` and `1.33 tok/sec` experiment).

Strand Rust Coder 14B Q6_K now inspects and runs with a capped context on the
Pi 2 benchmark lane. It is useful compatibility evidence for Qwen2 + Q6_K, but
it is not a practical Pi target yet:

- model: `Fortytwo-Network/Strand-Rust-Coder-14B-v1-GGUF`
- file: `Fortytwo_Strand-Rust-Coder-14B-v1-Q6_K.gguf`
- size: `12.1 GB`
- metadata: Qwen2, 48 layers, 5120 hidden width, 32k advertised context
- short run with `NANOCAMELID_CONTEXT_LIMIT=128`: load about `39-54s`, one-token
  prompt prefill about `6.6s`, 8-token generation `46.06s` (`0.17 tok/sec`)

## Runtime Design

NanoCamelid keeps the runtime small and explicit:

- Rust CLI only; no Python service dependency and no required C++ build step.
- Bounded Rayon worker setup tuned for small ARM boards.
- Optional CPU affinity when the platform exposes it.
- GGUF tensor bytes are sourced from an mmap-backed view during model loading,
  avoiding one temporary file-read buffer per tensor while preserving owned
  runtime weights.
- NEON/SDOT hot paths guarded by architecture checks and parity tests.
- Opt-in experimental Q4_0 1x4 SDOT and swizzled storage paths for benchmark
  comparison.
- Repeatable smoke and benchmark commands instead of broad model-family claims.

## Requirements

- Raspberry Pi 5 or another ARM64 Linux machine
- Rust toolchain
- A local GGUF model file

## Quick Start

Install the latest release build from GitHub:

```bash
curl -fsSL https://raw.githubusercontent.com/timtoole02/NanoCamelid/main/scripts/install.sh | bash
```

The installer clones NanoCamelid, builds the release binary with Cargo, and
links `nanocamelid` into `~/.local/bin`. Override paths when needed:

```bash
NANOCAMELID_INSTALL_DIR=/mnt/nanocamelid/src/NanoCamelid \
CARGO_TARGET_DIR=/mnt/nanocamelid/target \
curl -fsSL https://raw.githubusercontent.com/timtoole02/NanoCamelid/main/scripts/install.sh | bash
```

Manual checkout still works:

```bash
git clone https://github.com/timtoole02/NanoCamelid.git
cd NanoCamelid

cargo run -- probe
cargo run -- inspect /path/to/model.gguf
cargo run --release -- smoke q8-model /path/to/model.gguf "Hello" 1
cargo run --release -- smoke q8-chat /path/to/model.gguf "Say hello in one sentence." 8
NANOCAMELID_MODEL_GGUF=/path/to/model.gguf cargo run --release -- tui 0.0 64
```

`probe` prints CPU and SIMD feature information. `inspect` reads GGUF metadata
and tensor layout. `smoke q8-model` loads a Q8_0 model, checks scalar/runtime
logit parity, and runs a short greedy generation path from directly tokenized
prompt text. `smoke q8-chat` runs the same parity/generation validation through
the tokenizer chat template so Llama 3.2 1B Instruct rows can be smoke-tested
through the real instruct prompt path. Set `NANOCAMELID_MODEL_GGUF` to reuse
the same GGUF path across repeated `inspect`, `generate`, `chat`, and `tui`
runs, or `NANOCAMELID_SMOKE_GGUF` to override that shared default just for smoke
validation.

Single-turn generation is available through either raw prompt text or a rendered
chat prompt:

```bash
NANOCAMELID_MODEL_GGUF=/path/to/model.gguf \
  cargo run --release -- generate "Hello" 0.0 32

NANOCAMELID_MODEL_GGUF=/path/to/model.gguf \
  cargo run --release -- chat "Say hello in one sentence." 0.0 32
```

`tui` opens an interactive terminal chat that keeps the model loaded, shows the
connected model path/name, selected Q8 kernel, chat renderer, and per-turn plus
session token-in/token-out counters, TTFT, and throughput.

`NANOCAMELID_PREFILL_BATCH` controls how many prompt tokens are ingested at once
before decode begins. The default is `16`. Set it to `1` for the old
single-token reference behavior, or use `bench q4-prefill` to compare candidate
batch sizes on the current host without loading a GGUF model.

For very long-context GGUFs, `NANOCAMELID_CONTEXT_LIMIT` can cap the runtime KV
cache during local smoke tests:

```bash
NANOCAMELID_CONTEXT_LIMIT=128 \
  NANOCAMELID_MODEL_GGUF=/path/to/model.gguf \
  cargo run --release -- generate "Hello" 0.0 8
```

This does not change the model metadata or make broad context-length support
claims; it only bounds memory for short validation runs.

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

Run benchmarks on the target Pi in release mode.

```bash
cargo run --release -- bench q8-dot 1000 3
cargo run --release -- bench q4-layout 32768 3584 3
cargo run --release -- bench q4-prefill 128 16
```

To include the default-off SDOT candidate in Q8 dot reports:

```bash
NANOCAMELID_Q8_DOT_SDOT=1 \
cargo run --release -- bench q8-dot 1000 3
```

Each benchmark prints human-readable timing plus a JSON summary line. Treat
results as specific to the exact Pi, model, build, and environment used.

Useful environment controls:

- `NANOCAMELID_PREFILL_BATCH`: prompt-token batch size; default `16`.
- `NANOCAMELID_CONTEXT_LIMIT`: optional runtime KV-cache context cap for short
  smoke tests of long-context models.
- `NANOCAMELID_RAYON_THREADS`: global Rayon worker count.
- `NANOCAMELID_MATMUL_MIN_ROWS`: row-count threshold before matmuls enter Rayon.
- `NANOCAMELID_Q8_DOT_KERNEL=scalar|neon|sdot`: force the selected Q8 kernel.
- `NANOCAMELID_Q8_DOT_SDOT=1`: enable SDOT candidate benchmarking.
- `NANOCAMELID_Q4_1X4_SDOT=1`: enable the experimental Q4_0 1x4 SDOT path.
- `NANOCAMELID_Q4_SWIZZLE_1X4=1`: load compatible Q4_0 tensors in the swizzled
  1x4 runtime layout.

The swizzled Q4_0 1x4 path is an opt-in performance path. It has shown a real
Pi 2 short-chat win with smoke parity, but it remains explicit until broader
prompts and models confirm the shape.

## Tested Models

These rows reflect models that have been loaded and smoke-tested on Raspberry Pi
hardware with the current GGUF path. They are not broad family claims.

| Model | GGUF quant | Status | Notes |
| --- | --- | --- | --- |
| Llama 3.2 1B Instruct | Q4_0 | Working | Pi smoke passes with scalar-vs-selected-kernel logit parity and interactive TUI chat. |
| Llama 3.2 1B Instruct | Q8_0 | Working | Baseline path for Q8 validation and Q4 comparison. |
| Qwen2.5-Coder-7B-Instruct | Q4_0 | Smoke passing | Official Q4_0 GGUF loads, Qwen chat rendering runs, and Pi smoke/chat generation passes with exact scalar-vs-selected logit parity on the smoke gate. |
| Strand Rust Coder 14B v1 | Q6_K | Experimental | Official Q6_K GGUF inspects and runs with `NANOCAMELID_CONTEXT_LIMIT=128`, but current throughput is too slow for practical Pi use. |

## Pi Performance Snapshot

Current Pi 2 evidence, measured on local release builds:

- Llama 3.2 1B Instruct Q4_0 short chat: about `4.07-4.09 tok/sec`.
- Llama 3.2 1B Instruct Q8_0 short chat: about `3.63 tok/sec`.
- Qwen2.5-Coder-7B-Instruct Q4_0 smoke: exact logit parity,
  `max_logit_delta: 0.00000000`.
- Qwen2.5-Coder-7B-Instruct Q4_0 direct generation, short Rust ownership
  prompt: model load `3.66s`, prefill `4.05s`, generation `14` tokens in
  `10.45s` (`1.34 tok/sec`).
- Qwen2.5-Coder-7B-Instruct Q4_0 145-token chat prompt: prefill improved from
  `48.90s` at batch 1 to about `17.0s` with loop-inverted batch 16 prefill.
- Strand Rust Coder 14B v1 Q6_K capped-context smoke: load about `39-54s`,
  one-token prompt prefill about `6.6s`, 8 generated tokens in `46.06s`
  (`0.17 tok/sec`).
- mmap-backed source reads improve the warm Qwen2.5-Coder-7B-Instruct Q4_0
  load path to `2.63s`, but they do not make large models instant. Strand 14B
  Q6_K still takes about `47s` to load because the current runtime still
  decodes/copies quantized blocks and materializes embedding vectors.
- Q8 SDOT single-block microkernel: split accumulators moved the Pi 2 SDOT
  median from about `1.683 ns/block` to about `1.679 ns/block`.

The Q4_0 1B path is faster than Q8_0 on the same prompt, but the measured
end-to-end gain is still far below the theoretical memory-traffic ceiling. The
next useful performance work should be driven by real prompt/decode timings, not
isolated kernel wins alone.

## Raspberry Pi Deployment

Prepare a Pi workspace:

```bash
./scripts/pi/bootstrap.sh
```

Build and test remotely:

```bash
./scripts/remote_build.sh <pi-host> [ssh-key] [pi-user]
```

On a prepared Pi workspace, `remote_build.sh` now reuses the same default 1B
model selection as `scripts/pi/smoke-1b.sh`: it prefers the Pi-local
`Llama-3.2-1B-Instruct-Q4_0.gguf`, falls back to `...Q8_0.gguf`, and runs the
real instruct/chat smoke by default. Disable that model-backed gate explicitly
with:

```bash
NANOCAMELID_REMOTE_SMOKE=0 ./scripts/remote_build.sh <pi-host> [ssh-key] [pi-user]
```

To force a specific GGUF path that already exists on the Pi:

```bash
NANOCAMELID_REMOTE_SMOKE_GGUF=/path/on/pi/model.gguf \
./scripts/remote_build.sh <pi-host> [ssh-key] [pi-user]
```

To override the default chat smoke kind, prompt, or token budget:

```bash
NANOCAMELID_REMOTE_SMOKE_KIND=q8-model \
NANOCAMELID_SMOKE_PROMPT="Hello" \
NANOCAMELID_SMOKE_TOKENS=1 \
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
