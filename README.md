# 🐪 NanoCamelid

**NanoCamelid** is a hyper-optimized, stripped-down version of the [Camelid](https://github.com/timtoole02/Camelid) inference engine, rebuilt specifically for the **Raspberry Pi** and ARM64 architecture.

While the parent project explores the "archaeology" of x86 and AMX, **NanoCamelid** is a pure-performance side project focused on squeezing every possible token per second out of Broadcom BCM2712 (Pi 5) and BCM2711 (Pi 4) silicon.

## 🚀 The Goal: "Pi-Awesome" Performance

We aren't building a general-purpose engine. We are building an **Edge Appliance**.

- **Hardware-Specific:** Zero-dependency Rust kernels targeting **ARM NEON** and **FEAT_DotProd**.
- **Stripped & Fast:** Removed all server-side bloat to focus on ultra-low latency CLI and local API execution.
- **Memory First:** Optimized for the 4GB/8GB constraints of the Pi, utilizing specialized GGUF quantization (Q4_K_M / Q8_0).

## 🛠 Target Architecture

- **Primary Target:** Raspberry Pi 5 (Cortex-A76)
- **Instruction Set:** AArch64 + NEON SIMD
- **Language:** 100% Rust-native (No C++ dependencies)

## 📦 Roadmap

- [x] Add a Raspberry Pi host probe for ARM64/NEON feature discovery.
- [x] Add a Q8_0 block/layout boundary with scalar reference dot math.
- [ ] Implement specialized NEON matrix multiplication kernels.
- [ ] Port core Camelid logic with ARM-specific static dispatch.
- [ ] Optimize 1B parameter models (Llama 3.2) for "instant" response times.
- [ ] Benchmarking vs. llama.cpp on Pi 5.

## 🤝 Getting Started

```bash
cargo run -- probe
cargo run -- inspect /path/to/model.gguf
cargo run --release -- bench q8-dot [iterations] [runs]
NANOCAMELID_Q8_DOT_SDOT=1 cargo run --release -- bench q8-dot [iterations] [runs]
NANOCAMELID_Q8_DOT_KERNEL=sdot NANOCAMELID_Q8_DOT_SDOT=1 cargo run --release -- bench q8-dot [iterations] [runs]
cargo run --release -- smoke q8-model /path/to/model.gguf "Hello" 1
```

The Q8 dot benchmark prints repeated scalar/NEON timing, a JSON summary line, and
when the default-off SDOT candidate is enabled, direct SDOT-vs-NEON ratios for
retaining or rejecting the kernel on the target Pi. `NANOCAMELID_Q8_DOT_KERNEL`
selects the dispatch path under measurement and defaults to scalar unless an
explicit requested kernel passes runtime feature checks.

The model smoke loads a GGUF and compares scalar Q8_0 matmul logits against the
selected runtime kernel before checking a short greedy generation path.

## Raspberry Pi Deployment

```bash
./scripts/pi/bootstrap.sh
./scripts/remote_build.sh <pi-host> [ssh-key] [pi-user]
```

Deployment defaults to rsync snapshots. For public Pi checkouts that should keep
git metadata current, use clean fast-forward mode:

```bash
NANOCAMELID_DEPLOY_MODE=git-ff ./scripts/deploy.sh <pi-host>
NANOCAMELID_DEPLOY_MODE=git-ff ./scripts/remote_build.sh <pi-host>
```

`git-ff` refuses dirty worktrees, non-fast-forward branches, non-public origin
URLs, and existing non-git target directories. Set
`NANOCAMELID_REMOTE_SMOKE_GGUF` to a Pi-local GGUF path when running
`remote_build.sh` to include the model-backed Q8_0 parity smoke.

The reusable runtime surface now lives in the library crate (`nanocamelid::q8`
and `nanocamelid::gguf`) so Pi runners can share the GGUF/Q8 boundaries instead
of coupling everything to the CLI binary.

For the porting sequence, see [`docs/PI_PORTING.md`](docs/PI_PORTING.md).
For the Camelid-derived implementation map, see
[`docs/CAMELID_PORTING_MAP.md`](docs/CAMELID_PORTING_MAP.md).

## License

NanoCamelid is licensed under the MIT License. See [`LICENSE`](LICENSE).

*This project is currently in its early 'Nano' phase. Benchmark output is hardware-local and should be treated as evidence for the specific Pi/configuration where it was captured.*
