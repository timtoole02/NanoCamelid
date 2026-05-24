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
```

`probe` prints CPU and SIMD feature information. `inspect` reads GGUF metadata
and tensor layout. `smoke q8-model` loads a Q8_0 model, checks scalar/runtime
logit parity, and runs a short greedy generation path.

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

Deployment defaults to rsync snapshots. For public Pi checkouts that should keep
git metadata current, use clean fast-forward mode:

```bash
NANOCAMELID_DEPLOY_MODE=git-ff ./scripts/deploy.sh <pi-host>
NANOCAMELID_DEPLOY_MODE=git-ff ./scripts/remote_build.sh <pi-host>
```

`git-ff` refuses dirty worktrees, non-fast-forward branches, non-public origin
URLs, and existing non-git target directories.

## Project Status

- Host feature probing is available.
- GGUF metadata and tensor layout inspection are available.
- Q8_0 scalar, NEON, and default-off SDOT dot-product paths are available.
- Q8_0 model smoke validation is available for supported Llama-style GGUFs.
- Broader model support and performance claims require Pi-local artifacts.

The reusable runtime surface lives in the library crate under `nanocamelid::q8`
and `nanocamelid::gguf`.

## More Details

- [Pi porting notes](docs/PI_PORTING.md)
- [Camelid porting map](docs/CAMELID_PORTING_MAP.md)

## License

NanoCamelid is licensed under the MIT License. See [LICENSE](LICENSE).
