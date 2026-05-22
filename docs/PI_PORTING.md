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

## Runtime Direction

- Prefer static, explicit ARM64 dispatch over broad runtime abstraction.
- Keep Q8_0 as the first correctness target and add Q4_K_M only after the first row is stable.
- Use packed runtime storage when it demonstrably saves memory bandwidth.
- Gate specialized kernels behind feature checks and keep a scalar/reference path for tests.
- Treat benchmark claims as unsupported until the command, model, hardware class, and result are reproducible.

## Validation Gates

- `cargo fmt -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`
- `cargo run -- probe` on Raspberry Pi 5
- One exact model-row parity artifact before any support claim

## Pi Workspace Layout

Suggested development layout on a dedicated Linux filesystem:

```text
/mnt/nanocamelid/
  src/          # repositories
  models/       # local GGUF files, never committed
  benchmarks/   # benchmark outputs and notes
  target/       # Cargo target directory
```
