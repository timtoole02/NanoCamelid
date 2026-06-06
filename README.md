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
```

The Q8 dot benchmark prints repeated scalar/NEON timing, a JSON summary line, and
when the default-off SDOT candidate is enabled, direct SDOT-vs-NEON ratios for
retaining or rejecting the kernel on the target Pi. `NANOCAMELID_Q8_DOT_KERNEL`
selects the dispatch path under measurement and defaults to scalar unless an
explicit requested kernel passes runtime feature checks.

The reusable runtime surface now lives in the library crate (`nanocamelid::q8`
and `nanocamelid::gguf`) so Pi runners can share the GGUF/Q8 boundaries instead
of coupling everything to the CLI binary.

For the porting sequence, see [`docs/PI_PORTING.md`](docs/PI_PORTING.md).
For the Camelid-derived implementation map, see
[`docs/CAMELID_PORTING_MAP.md`](docs/CAMELID_PORTING_MAP.md).

## ⚡ Cortex-A76 build tuning

The repo ships tuned defaults — no flags needed:

- `.cargo/config.toml` sets `target-cpu=cortex-a76` for `aarch64-unknown-linux-gnu`
  (NEON + FEAT_DotProd codegen, A76 scheduling).
- Release/bench profiles use `lto = true`, `codegen-units = 1`.
- `generate` builds a rayon pool with one worker per core, **pinned** to the 4 A76 cores
  (`NANOCAMELID_PIN=0` to disable, `NANOCAMELID_THREADS=<n>` to resize).
- The Q8/Q4 dot kernel **auto-selects** SDOT → NEON → scalar at runtime; explicit
  `NANOCAMELID_Q8_DOT_KERNEL` / `NANOCAMELID_Q4_DOT_KERNEL` still take precedence.
- A thermal watchdog logs a warning when the SoC passes 80 °C or the firmware reports
  throttling (`vcgencmd get_throttled`); `probe` prints the current temperature/flags.
- Criterion benchmarks live in [`benches/`](benches/README.md) (run them on the Pi).

All of this changes scheduling and codegen only — **output is bit-identical** (the i8 dot
kernels are exact integer math; float accumulation order is never reordered).

## 🌐 3-node pipeline cluster

NanoCamelid can split a model's transformer layers across three Pi 5s
(pipeline parallelism), pooling 48 GB of RAM to run models up to ~3× what a single Pi
holds. Each node loads only its layer shard + KV cache; the f32 hidden state crosses the
gigabit link as raw little-endian frames over plain TCP (tokio), ~8–16 KB twice per token.
Sampling stays on the head node, so distributed output is **bit-identical** to a
single-node run (proven by `tests/pipeline_equivalence.rs`).

```bash
# 1. Put the 3 Pis' addresses in config/nodes.toml (order = stage order), then:
./scripts/launch_cluster.sh                      # deploy + build + start stage servers
./scripts/launch_cluster.sh "Tell me a story"    # ...and generate from the head node

# Manually:
nanocamelid serve-stage config/nodes.toml node1            # on node1
nanocamelid serve-stage config/nodes.toml node2            # on node2
nanocamelid generate-distributed config/nodes.toml "Hi"    # on node0 (head)

./scripts/stop_cluster.sh                        # tear down the stage servers
```

Layers auto-split evenly (remainder to earlier nodes); pin an explicit split with
`layers = [start, end]` per node in `nodes.toml`. Note: pipeline parallelism adds capacity
and multi-request throughput — single-stream tokens/sec is bounded by total compute plus
two small network hops per token, so for models that already fit on one Pi, plain
`generate` remains the latency-optimal path.

## License

NanoCamelid is licensed under the MIT License. See [`LICENSE`](LICENSE).

*This project is currently in its early 'Nano' phase. Benchmark output is hardware-local and should be treated as evidence for the specific Pi/configuration where it was captured.*
