<div align="center">

# 🦙 NanoCamelid

**A compact Rust inference engine for Raspberry Pi — one Pi chats with 1B–8B models; three Pis run Llama 3 70B.**

[![CI][ci-badge]][ci-workflow]
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Language: Rust](https://img.shields.io/badge/language-Rust-orange.svg)
![Platform: Raspberry Pi 5 · ARM64 Linux](https://img.shields.io/badge/platform-Raspberry%20Pi%205%20·%20ARM64-c51a4a.svg)

</div>

NanoCamelid loads GGUF models directly and runs them on Pi-class ARM64
hardware: one static binary, local model files, terminal chat, a web UI, an
OpenAI-style API, and a cluster mode that spreads one model across several
Pis. The GGUF loader, tokenizers, chat templates, and NEON SDOT kernels are
all implemented in this repository — no Python, no wrapper around another
runtime. Every supported model row and every performance number is backed by
committed Pi-side evidence.

![NanoCamelid terminal chat on a Raspberry Pi 5](docs/images/nanocamelid-tui.png)

<div align="center"><sub>The terminal chat surface (<code>nanocamelid tui</code>) on a Raspberry Pi 5.</sub></div>

---

## The cluster scoreboard

Three Raspberry Pi 5s (16GB) on ordinary gigabit Ethernet. Two cluster
lanes, both gated on token parity with the single-node engine:

| model | one Pi 5 | three Pi 5s | lane |
|---|---|---|---|
| Llama 3.2 1B Q4_0 | 13.4 tok/s | **20.4 tok/s** | tensor parallel |
| Llama 3.2 3B Q4_0 | 5.3 tok/s | **10.2 tok/s** | tensor parallel |
| Mixtral 8x7B Q4_0 | *does not fit* | **0.8 tok/s** | pipeline |
| **Llama 3 70B Q4_0** | *does not fit* | **0.7 tok/s** | tensor parallel |

- **Pipeline** (`cluster_tcp_smoke`) splits contiguous layer ranges across
  nodes — the *capacity* lane that fits models no single Pi can hold.
- **Tensor parallel** (`cluster_tp_node`) has every node compute every layer
  on a weighted shard of the weights — the *speed* lane. Shards load
  directly from the GGUF (a node only ever holds its slice), and per-node
  weight shares let mismatched hardware pull its own weight.

Every number comes from a committed, breakdown-backed receipt:
[docs/bench/CAPABILITY_TABLE.md](docs/bench/CAPABILITY_TABLE.md).

## Chat with a 70B on three Pis

Each node loads only its shard (~9–14GB). The master serves the built-in web
UI and an OpenAI-style completions endpoint, so any browser on your LAN can
talk to the cluster:

```bash
# node B and node C — workers, shard indexes 1 and 2 of shares 2,3,3
cluster_tp_node worker /path/to/Meta-Llama-3-70B-Instruct.Q4_0.gguf 0.0.0.0:5921 1 2,3,3
cluster_tp_node worker /path/to/Meta-Llama-3-70B-Instruct.Q4_0.gguf 0.0.0.0:5921 2 2,3,3

# node A — master: small shard, web UI, and API on :8090
cluster_tp_node master-serve /path/to/Meta-Llama-3-70B-Instruct.Q4_0.gguf \
  nodeB:5921,nodeC:5921 2,3,3 8090
```

Open `http://nodeA:8090/` and chat. The share list is per-node KV-head
counts — give a slower node a smaller number and the cluster stays balanced.

## 5-minute quickstart (one Pi)

```bash
curl -fsSL https://raw.githubusercontent.com/timtoole02/NanoCamelid/main/scripts/install.sh | bash
nanocamelid doctor && nanocamelid probe
```

Drop a GGUF under `/mnt/nanocamelid/models` (the `1b` / `3b` aliases resolve
Llama 3.2 Instruct files), then:

```bash
nanocamelid models list
nanocamelid ready 1b --no-chat          # readiness gate with evidence output
nanocamelid chat 1b "Say hello in one sentence." 0.0 32
nanocamelid tui 1b                      # full-screen terminal chat
nanocamelid serve                       # OpenAI-style API + web UI
```

The installer verifies `SHA256SUMS` against the versioned GitHub release
(pin one with `... | bash -s -- --version v0.1.0`; source installs use
`--dev`). Prompt rendering covers Llama, Qwen, ChatML, Mistral,
DeepSeek-R1-Qwen, and Gemma templates on supported rows.

## Core CLI

```bash
nanocamelid models list | scan | inspect 1b
nanocamelid ready 1b --dry-run
nanocamelid chat /path/to/model.gguf "prompt" 0.0 32
nanocamelid tui /path/to/model.gguf
nanocamelid serve --max-input-tokens 2048 --max-output-tokens 256
nanocamelid webui --host 0.0.0.0 --port 8080 --model-dir /path/to/models
```

`nanocamelid help` covers command arguments and environment controls; the
stable v0.1 command contract lives in
[docs/CLI_CONTRACT.md](docs/CLI_CONTRACT.md).

## How claims are made

The engine's rule, applied to every lane: **parity before performance.**
Optimized kernels, cluster splits, and speculative decoding all gate on
token-for-token agreement with the reference path before any speed number is
promoted; negative results get committed alongside the wins (see
[docs/bench/](docs/bench/) for the receipts, including the ones that say
"this didn't pay and here is exactly why").

- Supported model rows: [docs/MODEL_CATALOG.md](docs/MODEL_CATALOG.md)
- Product surface status: [docs/SUPPORT_MATRIX.md](docs/SUPPORT_MATRIX.md)
- HTTP API shapes and caps: [docs/API.md](docs/API.md)
- Run-as-a-service notes: [docs/SERVICE_MODE.md](docs/SERVICE_MODE.md)
- Architecture and benchmark walkthrough:
  [docs/HIGH_PERFORMANCE_INFERENCE_WALKTHROUGH.md](docs/HIGH_PERFORMANCE_INFERENCE_WALKTHROUGH.md)
- Full prototype history with Pi evidence:
  [docs/PRODUCT_HISTORY.md](docs/PRODUCT_HISTORY.md)
- Pi deployment and validation: [docs/PI_PORTING.md](docs/PI_PORTING.md)

## Validation and releases

```bash
./scripts/validate.sh              # fmt, clippy, cargo test --all-targets
./scripts/release-preflight.sh --dry-run
```

Release preflight, artifacts, checksums, and the installer smoke are
documented in [docs/RELEASE_PROCESS.md](docs/RELEASE_PROCESS.md).

[ci-badge]: https://github.com/timtoole02/NanoCamelid/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/timtoole02/NanoCamelid/actions/workflows/ci.yml
