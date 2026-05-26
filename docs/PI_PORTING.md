# Raspberry Pi Porting Plan

NanoCamelid is not a direct copy of Camelid. It is a Raspberry Pi focused extraction that keeps the proven model-loading and correctness lessons, then rebuilds the hot path around small models, ARM64 memory pressure, and CPU features available on Pi-class hardware.

## Product Boundaries

- Target small local models first, starting with 1B-class GGUF rows.
- Keep the first interface minimal: one CLI path for model inspection, one CLI path for generation, and a compact local UI only after the runtime is useful.
- Do not inherit Camelid's full server, desktop WebUI, broad model matrix, or x86/macOS optimization lanes.
- Keep public docs and examples free of private paths, local hostnames, credentials, keys, personal IPs, and machine-specific operator notes.

## First Vertical Slice

1. Host probe: confirm architecture, OS, CPU model, NEON, and dot-product availability.
2. GGUF metadata reader: port only the parser surface required to identify architecture, tensors, quantization type, tokenizer metadata, and context limits.
3. Tokenizer and prompt path: bring over the smallest proven Llama tokenizer flow needed for one supported row.
4. One-token inference: produce one deterministic token for a small model and compare against a known-good baseline.
5. Tight loop benchmark: measure prompt ingestion, first token latency, and decode token latency on Pi 5.

This vertical slice is no longer the active milestone. The current public
baseline is the Llama 3.2 1B readiness path in the CLI and Pi scripts:
`inspect 1b`, `smoke 1b`, `ready 1b`, and `scripts/pi/ready-1b.sh`.

## Runtime Direction

- Prefer static, explicit ARM64 dispatch over broad runtime abstraction.
- Keep Q8_0 as the baseline correctness target, with Q4_0 as the practical
  Llama 3.2 1B fast path. Additional quantized formats should advance only
  through exact model-row smoke evidence.
- Use packed runtime storage when it demonstrably saves memory bandwidth.
- Gate specialized kernels behind feature checks and keep a scalar/reference path for tests.
- Treat benchmark claims as unsupported until the command, model, hardware class, and result are reproducible.

## Validation Gates

- `cargo fmt -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`
- `cargo run -- model 1b --dry-run`
- `cargo run -- ready 1b --dry-run`
- `./scripts/pi/model-1b.sh --dry-run`
- `./scripts/pi/ready-1b.sh --dry-run`
- `cargo run -- probe` on Raspberry Pi 5
- `cargo run -- inspect <model.gguf>` against a local small-model GGUF
- `cargo run --release -- bench q8-dot` on Raspberry Pi 5 for repeated scalar vs NEON Q8 dot timing and JSON output
- `NANOCAMELID_Q8_DOT_SDOT=1 cargo run --release -- bench q8-dot` on Raspberry Pi 5; use the SDOT-vs-NEON ratios as the kernel decision signal
- `cargo run -- smoke 1b chat "Say hello in one sentence." 8` against a
  Pi-local Llama 3.2 1B GGUF before refreshing 1B claims
- Q8_0 block layout tests must keep 34-byte blocks, f16 scale expansion, signed
  i8 payload decoding, and scalar scaled-dot behavior stable
- One exact model-row parity artifact before any support claim

## Pi Workspace Layout

Suggested development layout on a dedicated Linux filesystem:

```text
<nanocamelid-workspace>/
  src/          # repositories
  models/       # local GGUF files, never committed
  benchmarks/   # benchmark outputs and notes
  target/       # Cargo target directory
```
