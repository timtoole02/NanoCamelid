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
- mixtral-8x7b-instruct-v0.1.Q4_0.gguf: PENDING

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
upstream wait / compute / feedback send.

RESULTS PENDING (filled from P0.3 runs below).

## P0.3 baseline receipts

RESULTS PENDING.

## H0 verdict

PENDING.
