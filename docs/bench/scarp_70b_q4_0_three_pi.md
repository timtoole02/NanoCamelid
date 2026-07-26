# Llama-3-70B Q4_0 on three Raspberry Pi 5s — 0.700 tok/s

Date: 2026-07-26. Stock clocks, `ondemand`, no overclocking.

Companion to `docs/bench/scarp_phase0_baseline.md` (the bandwidth/compute model),
`docs/bench/scarp_fast_tier.md` (the small-model tier), and
`docs/bench/phase4_tp_wire.md` (the original weighted-TP receipt this supersedes).

---

## Result

**Meta-Llama-3-70B-Instruct.Q4_0, tensor-parallel over three Pi 5s: 0.700 tok/s**,
coherent, served OpenAI-compatible from the head.

```
json: {"shards":3,"shares":[2,3,3],"prompt_tokens":20,"generated":24,
       "decode_tokens_per_sec":0.700,
       "local_compute_avg_ms":874.42,"sync_avg_ms":597.32,"logits_avg_ms":32.12}
```

> *"The sky appears blue because of a phenomenon called Rayleigh scattering, in
> which shorter (blue) wavelengths of light are scattered..."*

That is **2x** the Q2_K configuration it replaces (~0.35 tok/s), on a
strictly higher-quality quantization.

| | 70B Q2_K | **70B Q4_0** |
|---|---|---|
| decode | ~0.35 tok/s | **0.700 tok/s** |
| quantization | 2-bit | **4-bit** |
| disk per node | 26.4 GB | 40.0 GB |
| bound by | compute (K-quant unpack) | memory bandwidth |

## Why Q4_0 wins, and why it is not obvious

Weight *count* is fixed by the model, not the quantization — re-quantizing changes
bytes, never the number of multiply-accumulates. So a lower-bit format only helps
if you are bandwidth-bound. The 70B is not:

| format | weights/byte | achieved GB/s | achieved weights/s |
|---|---|---|---|
| Q4_0 | 1.78 | 12.1 | **21.5 G/s** |
| Q6_K | 1.22 | 12.2 | 15.1 G/s |
| Q2_K + Q3_K | 2.75 | 5.6 | 15.2 G/s |

Q2_K packs 3.05 weights into every byte, so saturating the ~11-12 GB/s memory wall
would demand **33.6 G weights/s** — far beyond any kernel here. It runs at 5.6 GB/s
not because of layout but because its unpack is expensive; the halve-the-cores test
(local compute 2247 -> 4345 ms, **1.93x**) proves compute, not bandwidth, is the
binder. Q4_0's single nibble shift is cheap enough to sit at the memory wall
instead, which is why it is nearly 2x faster despite moving 1.5x more bytes.

**Corollary worth keeping:** in a compute-bound regime the right format has *fewer*
weights per byte — the opposite of the bandwidth-bound intuition.

## Configuration, and why the master takes the small share

Shares `[2,3,3]`, master = shard 0 = the **2**:

| node | share | resident | board |
|---|---|---|---|
| camelid2 (master) | 2/8 | 9.72 GiB | 15.6 GiB |
| camelid3 | 3/8 | 13.75 GiB | 15.6 GiB |
| camelid1 | 3/8 | 13.75 GiB | 15.6 GiB |

The master additionally holds the 591 MB embedding table, the HTTP server and its
workspaces, so it gets the smallest slice. A master at 3/8 would need 14.30 GiB and
does not leave room for KV plus workspaces. **A 4/8 share is not merely slow, it is
unloadable** — 19.9 GB = 18.5 GiB on a 15.6 GiB board. With three nodes and a
maximum feasible share of 3, `[2,3,3]` is the *only* legal split. There is no
planning freedom here.

Related defect, not fixed by this work: `cluster_up::plan_shares` uses RAM only as
a planner *weight*, never as a constraint, so `up` will happily plan `[4,3,1]` and
hand a node a shard that cannot load. `probe_node` already reads MemTotal — the
gate is a few lines away.

Both workers also needed their idle `nanocamelid-stage` user units stopped (0.41
GiB on camelid1) to leave headroom; the original receipt hit the same wall and
stopped camelid3's webui for the same reason.

## Sync is now the bottleneck, and it is a structural one

`sync_avg_ms` of **597 ms is not network** — true wire cost is ~50 ms for a 32 KB
partial. It is the master, done in 874 ms, waiting on the share-3 workers, who need
~1311 ms. Compare the original receipt (`phase4_tp_wire.md:113`): local 1402.7 ms,
sync 93.3 ms, with a 2-core-capped master carrying 2/8. Replacing that board's
supply cut the master's compute by 1.6x and simply moved the bottleneck onto the
workers.

**Eight KV heads across three nodes cannot balance.** `[2,3,3]` puts 3/8 = 0.375 on
the busiest node against an ideal 0.333 — a structural 12.5% tax that no
configuration removes.

Two consequences:
- Moving the master to a share of 3 (and a worker to 2) removes most of the idle
  wait, for perhaps **~0.73 tok/s**. A ~4% gain for a full shard reload.
- **A fourth Pi is the real lever.** `[2,2,2,2]` puts 2/8 on every node — 9.6 GB
  each, comfortably under 10 GiB — for roughly **1.12 tok/s**. That single addition
  beats both software features costed for this campaign (pre-sharded shard files,
  which buy disk and load time but no tok/s; and speculative decode, which the
  design pass put at a wash-to-small-loss at the measured batch ratio).

## What this supersedes

`phase4_tp_wire.md:110` recorded 70B Q4_0 TP-3 at **0.685 tok/s** with a 2-core
master. This run measures **0.700** with three healthy boards. That receipt also
says "a healthy PSU directly buys ~2x here" — **that is wrong**, and this run is the
disproof: it assumed the capped board stayed the straggler. At share 2 it stops
being the straggler and the share-3 workers take over, so the supply is worth
~1.02x on this row. Replace a supply for the **3B fast tier** (9.51 -> 11.54
tok/s), not for the 70B.

## Reproducing

Requires 40.0 GB free per node. Verify the byte count after copying — a truncated
stream is silent.

```bash
# workers first (camelid3 shard 1, camelid1 shard 2), all four cores
env NANOCAMELID_CONTEXT_LIMIT=2048 NANOCAMELID_SPIN_POOL=0 \
    NANOCAMELID_Q4_SWIZZLE_1X4=0 NANOCAMELID_Q8_SWIZZLE_1X4=0 \
  cluster_tp_node worker <70b-q4_0.gguf> 0.0.0.0:5921 <1|2> 2,3,3

# head last, taking the small share -- same env prefix
  cluster_tp_node master-serve <70b-q4_0.gguf> <w1>:5921,<w2>:5921 2,3,3 8090
```

Shard loads take roughly ten minutes per worker from a USB SSD. Workers are
multi-session, so a head can reconnect to warm workers without reloading them.
