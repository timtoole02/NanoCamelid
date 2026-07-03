# Phase 0 baseline — cluster speedup campaign

Date: 2026-07-03
Commit: c4a0f14 (perf/cluster-speedup; base 84850ed origin/main)
Build: `cargo build --release` on each Pi, `/mnt/nanocamelid/target-cluster-speedup`

## Environment

| node | role | host | ip | hw | governor |
|---|---|---|---|---|---|
| camelid1 | master | camelid1 | 192.168.86.30 | Pi 5 16GB, NVMe | ondemand |
| camelid2 | middle | camelid2 | 192.168.86.48 | Pi 5 16GB, NVMe | ondemand |
| camelid3 | final | camelid3 | 192.168.86.44 | Pi 5 16GB, NVMe | ondemand |

GbE via home-LAN switch (192.168.86.0/24, eth0 on all nodes). NOT a dedicated
switch; runs were done with the LAN otherwise quiet.

DEVIATION from the campaign matrix: governor is `ondemand` on all nodes —
setting `performance` requires sudo, which is password-gated on these Pis.
Every A/B in this phase ran under the same governor on both sides. Revisit
once passwordless `cpupower` access exists.

DEVIATION: camelid2's NVMe (/mnt/nanocamelid) is 100% full; the 3B GGUF on
camelid2 lives at `/home/tooleman/models/` (SD card) instead. Affects load
time only (page cache holds the 1.8G model after first touch).

Model SHA256 (verified identical on every node that holds the file):

- Llama-3.2-1B-Instruct-Q4_0.gguf: `eadfd8fd4e29d48e720eb87fc8242d3a8d4d2dacd52c722adc8e69e48c668efc`
- Llama-3.2-3B-Instruct-Q4_0.gguf: `506d311f2f8802991344f7186badffda9c6a6b4cb50aa7a759ba3d939544df44`
- mixtral-8x7b-instruct-v0.1.Q4_0.gguf: `0c57465507f21bed4364fca37efd310bee92e25a4ce4f5678ef9b44e95830e4e`

DEVIATION: camelid1 (master) browns out under sustained multi-core load —
the decode spin pool (default-on for aarch64, N−1 tight-spin workers) keeps
~3 cores at 100% through all pipeline idle time, and camelid1's supply
cannot sustain that for multi-minute PP runs (it crashed twice today; short
single-node runs survive). All camelid1 cluster processes in this campaign
run under `taskset -c 0,1 NANOCAMELID_RAYON_THREADS=2` (the proven-stable
config; the spin pool sizes itself from the affinity mask). This roughly
doubles camelid1's stage time and is visible in every breakdown below.
Fixing camelid1's power supply would recover ~2x on its stage.

Prompts: `docs/bench/PROMPTS.md` (PROMPT_SHORT / PROMPT_LONG), chat path,
greedy (temp 0), 64 max tokens, `NANOCAMELID_CLUSTER_CONTEXT_LIMIT=512`.

## P0.1 network characterization

`cluster_bench` echo round trip (payload each way), `NANOCAMELID_BENCH_NODELAY`
controls Nagle. Hop map: master(c1) → middle(c2) → final(c3).

| pair | payload each way | avg ms | p50 ms | p95 ms | max ms |
|---|---|---|---|---|---|
| c1↔c2 | 16 KiB | 0.400 | 0.402 | 0.408 | 0.417 |
| c2↔c3 | 16 KiB | 0.374 | 0.372 | 0.390 | 0.414 |
| c1↔c3 | 16 KiB | 0.378 | 0.373 | 0.397 | 0.404 |
| c1↔c2 | 64 KiB | 1.236 | 1.236 | 1.250 | 1.256 |
| c2↔c3 | 64 KiB | 1.237 | 1.236 | 1.252 | 1.315 |
| c1↔c3 | 64 KiB | 1.234 | 1.233 | 1.248 | 1.428 |
| c1↔c2 | 256 KiB | 4.607 | 4.608 | 4.625 | 4.636 |
| c2↔c3 | 256 KiB | 4.596 | 4.596 | 4.611 | 4.621 |
| c1↔c3 | 256 KiB | 4.659 | 4.609 | 4.862 | 4.883 |
| c1↔c2 | 4 MiB | 71.930 | 71.833 | 71.964 | 74.701 |
| c2↔c3 | 4 MiB | 71.840 | 71.836 | 71.895 | 71.911 |
| c1↔c3 | 4 MiB | 71.832 | 71.833 | 71.850 | 71.855 |

Sustained bandwidth from the 4 MiB row: 8 MiB round trip / 71.8 ms ≈
117 MiB/s — GbE wire rate on every pair. No slow link.

Nagle A/B (16 KiB, `NANOCAMELID_BENCH_NODELAY=0`):

| pair | avg ms | p50 ms | max ms |
|---|---|---|---|
| c1↔c2 | 0.396 | 0.402 | 0.414 |
| c2↔c3 | 5.128 | 0.380 | **2100.351** |

With Nagle enabled the c2↔c3 run hit a 2.1-second stall (avg 13x worse
than p50). The production cluster path already sets TCP_NODELAY on every
socket; this receipt is why it must stay.

## P0.2 per-token decode breakdown

Instrumentation: `cluster-decode-breakdown` JSON lines from
`cluster_tcp_smoke` (commit c4a0f14) — decode-step-only (seq_len == 1)
stage timings per role: master compute / activation send / feedback wait;
middle upstream wait / compute / downstream send / downstream wait; final
upstream wait / compute / feedback send. Raw JSON lines below per run.

## P0.3 baseline receipts

All runs: PROMPT_SHORT, chat template, temp 0, max 64 tokens. Single-node
via `nanocamelid chat` (`NANOCAMELID_CONTEXT_LIMIT=512`); cluster via
`cluster_tcp_smoke` master-chat with `NANOCAMELID_CLUSTER_CONTEXT_LIMIT=512`.

### Single node (camelid1, 4 cores — short runs don't trip the fault)

| row | decode tok/s | notes |
|---|---|---|
| 1B Q4_0 | **13.20** | README's 4.18 predates the tied-LM-head + KV perf merges |
| 3B Q4_0 | **5.33** | README's 2.22 likewise stale |

The campaign's success criteria re-base to these numbers, not the README's.

### 3B Q4_0 three-node PP (split 0..10 / 10..19 / 19..28)

**2.991 tok/s** (53 tokens, 17.72s; prompt ingest 7.02s). Per-token
breakdown (avg ms):

| stage | ms |
|---|---|
| master compute (c1, 2-core cap, 10 layers) | 141.6 |
| master activation send | 0.06 |
| middle compute (c2, 9 layers) | 69.0 |
| middle downstream send | 0.06 |
| final compute (c3, 9 layers + LM head + logits) | 122.8 |
| **network residual** (master wait − downstream stages) | **~0.7** |

PP overhead on a model that fits one node: 2.991 vs 5.33 tok/s = 0.56x —
entirely compute serialization (two of three nodes idle at any instant),
not wire time. With an uncapped master (~71ms for 10 layers) the pipeline
would sum to ~264ms ≈ 3.8 tok/s = 0.71x: still strictly worse than one
node, as the conductor predicted for PP.

Raw: master `{"compute_avg_ms":141.558,"send_avg_ms":0.058,"recv_wait_avg_ms":192.605}`,
middle `{"upstream_wait_avg_ms":141.953,"compute_avg_ms":69.028,"downstream_send_avg_ms":0.060,"downstream_wait_avg_ms":123.266}`,
final `{"upstream_wait_avg_ms":211.281,"compute_avg_ms":122.786,"feedback_send_avg_ms":0.034}`.

### Mixtral 8x7B Q4_0 three-node PP (split 0..11 / 11..22 / 22..32)

Run 1 — default flags: **0.502 tok/s** (39 tokens, 77.7s; ingest 30.1s).

| stage | ms |
|---|---|
| master compute (c1, 2-core cap, 11 layers) | 641.7 |
| middle compute (c2, 11 layers) | 332.2 |
| final compute (c3, 10 layers + LM head) | **1017.8** |
| network residual | ~0.8 |

The final node burns ~700ms/token more than its layer share: this GGUF's
`output.weight` is the file's single Q6_K tensor (tensor_types: Q6_K: 1)
and the Q6_K matmul path is scalar unless `NANOCAMELID_Q6K_SDOT=1` — the
same opt-in the historic Strand three-Pi receipts recorded. The README's
1.12 tok/s row is only reachable with that flag (and an uncapped master).

Run 2 — `NANOCAMELID_Q6K_SDOT=1`, same commit/split/prompt: PENDING.

## H0 verdict

**Network is <5% — measured at ~0.2% (3B) and ~0.04% (Mixtral) of
per-token wall time.** GbE at wire rate on every pair; 16 KiB activation
hops cost ~0.2–0.4ms against 300–2000ms compute tokens. Phase 1 therefore
shrinks to framing hygiene only (TCP_NODELAY is already set everywhere —
and must stay: the Nagle-off A/B recorded a 2.1s stall). The levers that
matter are compute-side: Phase 2/3 (speculation fills pipeline bubbles)
and Phase 4 (TP divides the per-token weight read).
