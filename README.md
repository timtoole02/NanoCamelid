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
- On AArch64 boards with dot-product support, NanoCamelid now auto-selects the
  SDOT Q8 kernel, Q4_0 1x4 swizzled layout, Q4_0/Q6_K SDOT matmuls, and
  head-parallel attention by default. Scalar and forced-kernel modes remain
  available for comparison.
- Scalar reference paths remain in the test suite. Optimized kernels are kept
  tied to parity tests and Pi-side smoke evidence.
- The working model catalog lives in
  [`docs/MODEL_CATALOG.md`](docs/MODEL_CATALOG.md). It separates Pi-smoked
  supported rows from likely-compatible candidates and blocked runtime families
  such as Mixtral/MoE.

Quick 1B readiness check on a Pi workspace:

```bash
CARGO_TARGET_DIR=/mnt/nanocamelid/target cargo run -- inspect 1b
./scripts/pi/smoke-1b.sh
```

`inspect 1b` resolves `NANOCAMELID_SMOKE_GGUF` or `NANOCAMELID_MODEL_GGUF`
first, then the Pi-local `Llama-3.2-1B-Instruct-Q4_0.gguf` or Q8_0 fallback
under `${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}/models`.

## High-Performance Architecture

NanoCamelid is tuned around the Raspberry Pi 5's Cortex-A76 cores rather than
being a general desktop inference wrapper. The current fast path is built from a
small set of explicit runtime choices:

- **Auto-detected SDOT kernels.** When `NANOCAMELID_Q8_DOT_KERNEL` is unset,
  the runtime probes CPU features and selects SDOT on AArch64 systems with
  dot-product support, NEON when SDOT is unavailable, and scalar otherwise.
- **Q4_0 1x4 swizzled storage.** Compatible Q4_0 matrices are swizzled at load
  time so four adjacent rows can be streamed together in cache-friendly chunks.
- **Q4_0 and Q6_K SDOT matmuls.** The ARM dot-product paths are enabled by
  default on supported CPUs, with scalar references retained for tests and
  diagnostics.
- **Vectorized activation quantization.** AArch64 builds use NEON rounding and
  saturating-narrowing instructions for Q8 activation blocks, avoiding the
  scalar per-element round/clamp loop in the hot path.
- **Head-parallel attention.** Attention heads can be evaluated across Rayon
  workers using per-head scratch storage. This is most useful on longer prompts;
  very short prompts are still dominated by matmul work.
- **Governor telemetry.** `probe` and the TUI surface CPU governor information
  and recommend the non-overclock `performance` governor when Linux reports
  `ondemand`.

The implementation uses stable Rust with targeted `unsafe` AArch64 intrinsics
inside hot kernels. The goal is not a huge abstraction stack; it is an
inspectable Pi runtime where each optimization has a fallback, a test, or a
smoke path.

## Recent Pi Results

Latest runtime evidence below was captured through `59e374d`
(`perf(neon): vectorize activation Q8 block quantization using rounding and
saturating narrowing instructions`).

On the Pi 2 benchmark lane, the default Q8 dot benchmark now auto-selects SDOT
with no speed environment variables set:

- selected kernel: `sdot`
- scalar median: about `3.18 ns/block`
- NEON median: about `2.11 ns/block`
- SDOT median: about `1.69 ns/block`
- SDOT speedup: about `1.88x` over the scalar run in that benchmark sample and
  about `1.25x` over NEON

The isolated Q4 layout benchmark for a Qwen-sized shape also shows the memory
layout win:

- row-major Q4: `90.536ms`
- swizzled 1x4 Q4: `70.648ms` (`1.28x`)
- page-aligned swizzled 1x4: `68.337ms` (`1.32x` vs row-major, `1.034x` vs
  contiguous swizzled)

Page alignment remains opt-in because the incremental real-model gain has not
justified the duplicate chunk storage.

Llama 3.2 1B Instruct Q4_0 now passes the Pi-local chat smoke path and direct
generation check with the default fast profile:

- `smoke q8-chat` generated text: `"Hello!"`
- `max_logit_delta: 0.00000000`
- direct generation prompt: `Say hello in one sentence.`
- model load: about `0.90s`
- prompt ingest: about `0.38s`
- generated text: `"Hello, how are you?" is`
- throughput: `8` tokens in `1.91s` (`4.18 tok/sec`)

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

Recent experiments and narrow wins:

- Register-accumulated attention was correct but did not improve the short Qwen
  decode run (`1.88 tok/sec` baseline vs `1.87 tok/sec` experiment).
- f16 KV-cache storage preserved short Qwen smoke output but did not improve the
  short 16-token prompt (`1.83 tok/sec` vs `1.88 tok/sec` for f32 cache), so it
  remains an opt-in memory-pressure mode.
- Vectorized activation quantization is now landed after Pi-side smoke passed;
  the short 1B run moved from `4.16` to `4.18 tok/sec`, which is a small
  positive result but still within normal run noise.

Strand Rust Coder 14B Q6_K now inspects and runs with a capped context on the
Pi 2 benchmark lane. It is useful compatibility evidence for Qwen2 + Q6_K, but
it is not a practical Pi target yet:

- model: `Fortytwo-Network/Strand-Rust-Coder-14B-v1-GGUF`
- file: `Fortytwo_Strand-Rust-Coder-14B-v1-Q6_K.gguf`
- size: `12.1 GB`
- metadata: Qwen2, 48 layers, 5120 hidden width, 32k advertised context
- short run with `NANOCAMELID_CONTEXT_LIMIT=128`: load about `39-54s`, one-token
  prompt prefill about `6.6s`, 8-token generation `46.06s` (`0.17 tok/sec`)
- Q6_K SDOT preserved the initial smoke output and reduced a capped one-token
  Strand run from about `78s` to about `54s`.

Additional small-model catalog rows now validate on the Pi smoke lane:

- Qwen2.5 0.5B Instruct Q4_0: `ready`, 8-token generation at about
  `33.31 tok/sec`
- Qwen2.5-Coder 0.5B Instruct Q4_0: `ready`, 8-token generation at about
  `33.28 tok/sec`
- DeepSeek-R1-Distill-Qwen 1.5B Q4_0: `ready`, 8-token generation at about
  `13.25 tok/sec`
- Mistral 7B Instruct v0.1 Q4_0: `ready`, 4-token generation at about
  `3.68 tok/sec`

Mixtral is intentionally not listed as supported yet. It is a routed MoE model
family and needs expert tensor loading, router logits, and expert FFN execution
before NanoCamelid can claim it honestly.

## Runtime Design

NanoCamelid keeps the runtime small and explicit:

- Rust CLI only; no Python service dependency and no required C++ build step.
- Bounded Rayon worker setup tuned for small ARM boards.
- Optional CPU affinity when the platform exposes it.
- GGUF tensor bytes are sourced from an mmap-backed view during model loading,
  avoiding one temporary file-read buffer per tensor while preserving owned
  runtime weights.
- NEON/SDOT hot paths guarded by architecture checks and parity tests.
- Default fast-path Q4_0/Q6_K SDOT, Q4_0 1x4 swizzled storage, Q8 SDOT
  auto-selection, and head-parallel attention on supported Pi-class ARM64
  hardware.
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
`smoke 1b chat` preflight before opening the TUI, so the 1B instruct path keeps
the scalar-vs-selected-kernel parity gate in front of interactive chat. It still
honors `NANOCAMELID_MODEL_GGUF` and `NANOCAMELID_Q8_DOT_KERNEL` if you want to
force a different model or kernel for comparison. When the helper needs to build
through Cargo, it uses `/mnt/nanocamelid/target` by default, or an explicit
`CARGO_TARGET_DIR` or `NANOCAMELID_TARGET_DIR` override.

Optional arguments set temperature and maximum assistant output tokens:

```bash
./scripts/pi/chat-1b.sh 0.0 64
```

`smoke-1b.sh` uses the same kernel defaults, but runs only the smoke gate and
exits. Its model-selection precedence is `NANOCAMELID_SMOKE_GGUF`,
`NANOCAMELID_MODEL_GGUF`, Pi-local Q4_0, then Pi-local Q8_0. By default it
runs the real instruct prompt path with `chat`, the prompt
`Say hello in one sentence.`, and an 8-token response budget. Optional
arguments let you override the smoke kind, prompt, and token budget directly:

```bash
./scripts/pi/smoke-1b.sh chat "Say hello in one sentence." 8
./scripts/pi/smoke-1b.sh model "Hello" 1
```

For faster local iteration, disable the preflight smoke gate explicitly:

```bash
NANOCAMELID_CHAT_SMOKE=0 ./scripts/pi/chat-1b.sh
```

The preflight smoke defaults to `chat` with a one-token response budget, and
you can override the gate with:

- `NANOCAMELID_CHAT_SMOKE_KIND=model|chat`
- `NANOCAMELID_CHAT_SMOKE_PROMPT="..."`
- `NANOCAMELID_CHAT_SMOKE_TOKENS=1`

## Benchmarks

Run benchmarks on the target Pi in release mode.

```bash
cargo run --release -- bench q8-dot 1000 3
cargo run --release -- bench q4-layout 32768 3584 3
cargo run --release -- bench q4-prefill 128 16
```

Each benchmark prints human-readable timing plus a JSON summary line. Treat
results as specific to the exact Pi, model, build, and environment used.

Useful environment controls:

- `NANOCAMELID_PREFILL_BATCH`: prompt-token batch size; default `16`.
- `NANOCAMELID_CONTEXT_LIMIT`: optional runtime KV-cache context cap for short
  smoke tests of long-context models.
- `NANOCAMELID_RAYON_THREADS`: global Rayon worker count.
- `NANOCAMELID_WORKER_CORES=1,2,3`: pin Rayon workers to a CPU list. If this is
  unset and Linux reports isolated CPUs in `/sys/devices/system/cpu/isolated`,
  NanoCamelid uses that isolated set automatically.
- `NANOCAMELID_MATMUL_MIN_ROWS`: row-count threshold before matmuls enter Rayon.
- `NANOCAMELID_Q8_DOT_KERNEL=scalar|neon|sdot`: force the selected Q8 kernel.
- `NANOCAMELID_Q8_DOT_SDOT=0`: disable SDOT candidate selection for comparison.
- `NANOCAMELID_Q4_1X4_SDOT=0`: disable the Q4_0 1x4 SDOT path for comparison.
- `NANOCAMELID_Q4_SWIZZLE_1X4=0`: disable compatible Q4_0 tensor swizzling and
  use the row-major Q4 path for comparison.
- `NANOCAMELID_Q4_PAGE_ALIGN_1X4=1`: when the swizzled Q4_0 path is enabled,
  also keep an opt-in page-aligned copy of each 1x4 row chunk. This costs extra
  memory and is not the default.
- `NANOCAMELID_Q6K_SDOT=0`: disable the AArch64 SDOT path for Q6_K-by-Q8
  matmuls when comparing against the scalar route.
- `NANOCAMELID_ATTENTION_HEAD_PARALLEL=0`: disable Rayon head-parallel
  attention for comparison. This uses per-head score scratch space and is most
  visible on longer prompts.
- `NANOCAMELID_KV_CACHE_F16=1`: store KV-cache entries as f16 and decode them
  during attention. This halves KV-cache storage and bandwidth for cached keys
  and values, but it is lossy and remains opt-in until real-model parity and
  long-context speed evidence justify broader use.

The swizzled Q4_0 1x4 path and SDOT kernels are the default fast profile on
supported Pi-class ARM64 hosts. The environment variables above are primarily
for diagnostics and before/after benchmark runs.

The page-aligned Q4_0 1x4 path is narrower: the Pi 2 layout microbenchmark
showed a small gain over contiguous swizzled storage, but it duplicates the
swizzled matrix chunks and should be treated as a measurement switch until
real-model runs justify making it broader.

The f16 KV-cache path is also opt-in. It intentionally compares against an
explicitly decoded f16 reference rather than the full-f32 cache, because
half-precision cache storage is a lossy runtime mode.

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

- Llama 3.2 1B Instruct Q4_0 short generation, default fast profile:
  `4.18 tok/sec`.
- Llama 3.2 1B Instruct Q8_0 short chat: about `3.63 tok/sec`.
- Q8 dot microbenchmark, default-selected SDOT: about `1.69 ns/block`.
- Q4 layout microbenchmark: row-major `90.536ms`, swizzled 1x4 `70.648ms`,
  page-aligned swizzled `68.337ms`.
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
- Experimental Q6_K SDOT on Pi 2 preserved the Strand 14B one-token smoke output
  and reduced a capped one-token wall-clock run from `78s` to `54s`.
- Q4_0 page-aligned 1x4 swizzled storage improved the isolated Pi 2 layout
  microbenchmark from `99.716ms` to `96.445ms` over 7 runs, about `1.034x`
  versus contiguous swizzled storage. The same Qwen prompt stayed essentially
  flat end-to-end, so this remains opt-in because the win is small and requires
  duplicate swizzled chunks.
- Experimental f16 KV-cache storage preserved the Qwen2.5-Coder-7B-Instruct
  Q4_0 4-token smoke output with `max_logit_delta: 0.00000000`, but the short
  16-token Rust prompt was slightly slower than f32 cache (`1.83 tok/sec` vs
  `1.88 tok/sec`). Treat it as a memory-pressure option until longer-context
  runs prove a speed win.
- mmap-backed source reads improve the warm Qwen2.5-Coder-7B-Instruct Q4_0
  load path to `2.63s`, but they do not make large models instant. Strand 14B
  Q6_K still takes about `47s` to load because the current runtime still
  decodes/copies quantized blocks and materializes embedding vectors.
- Q8 SDOT single-block microkernel: split accumulators moved the Pi 2 SDOT
  median from about `1.683 ns/block` to about `1.679 ns/block`.
- Vectorized NEON activation quantization preserved the 1B smoke path and moved
  a short default 1B run from `4.16` to `4.18 tok/sec`. Treat this as a safe
  kernel cleanup, not a proven end-to-end breakthrough.

The Q4_0 1B path is faster than Q8_0 on the same prompt, but the measured
end-to-end gain is still far below the theoretical memory-traffic ceiling. The
next useful performance work should be driven by real prompt/decode timings, not
isolated kernel wins alone.

Use `nanocamelid probe` on Raspberry Pi hosts to inspect CPU max frequency,
governor, isolated CPU state, selected worker-core policy, and SIMD support. The
tool reports telemetry only; boot parameters and overclock settings remain an
operator decision outside NanoCamelid. When Linux reports the `ondemand`
governor, `probe` and the TUI banner recommend the safe non-overclock command
for repeatable low-latency decode:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

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
NANOCAMELID_REMOTE_SMOKE_KIND=model \
NANOCAMELID_SMOKE_PROMPT="Hello" \
NANOCAMELID_SMOKE_TOKENS=1 \
./scripts/remote_build.sh <pi-host> [ssh-key] [pi-user]
```

Deployment defaults to rsync snapshots. Advanced deployment modes are available
in the scripts for development workflows.

## Project Status

- Host feature probing is available.
- GGUF metadata and tensor layout inspection are available.
- Q8_0 scalar, NEON, and auto-selected SDOT dot-product paths are available.
- Q4_0 loading and Q4_0 weight x Q8_0 activation matmul paths are available.
- Single-turn chat prompt rendering is available for recognized instruct templates.
- Interactive terminal chat is available with model/kernel, token, TTFT, and throughput telemetry.
- The TUI can switch GGUFs at runtime with `/model <path>`.
- The default Pi fast profile enables SDOT, Q4 swizzling, Q4/Q6 SDOT matmuls,
  head-parallel attention, and NEON activation quantization when the host
  supports them.
- The Pi 1B chat launcher preserves scalar-vs-selected-kernel parity through
  the smoke gate.
- Q8_0 and Q4_0 model smoke validation is available for the tested GGUF rows above.
- Broader model support and performance claims require Pi-local artifacts and row-specific validation.

## More Details

- [Pi porting notes](docs/PI_PORTING.md)
- [Camelid porting map](docs/CAMELID_PORTING_MAP.md)

## License

NanoCamelid is licensed under the MIT License. See [LICENSE](LICENSE).
