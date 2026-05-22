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
- [ ] Implement specialized NEON matrix multiplication kernels.
- [ ] Port core Camelid logic with ARM-specific static dispatch.
- [ ] Optimize 1B parameter models (Llama 3.2) for "instant" response times.
- [ ] Benchmarking vs. llama.cpp on Pi 5.

## 🤝 Getting Started

```bash
cargo run -- probe
```

For the porting sequence, see [`docs/PI_PORTING.md`](docs/PI_PORTING.md).

*This project is currently in its early 'Nano' phase. Stay tuned for the first benchmark results.*
