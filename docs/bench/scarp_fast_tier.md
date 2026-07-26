# Fast tier — a conversational-speed endpoint on the Pi cluster

Date: 2026-07-25. Goal set by the campaign owner: *"fast tier, whatever it takes,
but no overclocking."* No clock, voltage, or `config.txt` change was made; every
number here is at stock clocks on the `ondemand` governor.

Companion to `docs/bench/scarp_phase0_baseline.md` (the bandwidth/compute model this
is built on) and `docs/bench/scarp_g2a_q4_head_default.md` (the Q4_0 head).

---

## What is running

**Llama-3.2-3B-Instruct-Q4_0, tensor-parallel over two Pi 5s, ~9.5 tok/s**, served
as an OpenAI-compatible endpoint on the head node's port 8090.

```
head    camelid2  shard 0, kv_heads 4   +  serves /v1/chat/completions, /v1/health, /api/cluster, /topology
worker  camelid3  shard 1, kv_heads 4
```

Verified end to end: a `/v1/chat/completions` request returned a correct 48-token
answer in 7.3 s wall clock including prefill and HTTP.

---

## Scaling, measured

Interleaved 1-node/2-node passes, 32 greedy tokens, `PROMPT_SHORT`, ctx cap 4096,
`ondemand`, stock clocks. Two to three repeats each; run-to-run spread was under
1%.

| model | 1 node | **2 nodes** | speedup | sync/token |
|---|---|---|---|---|
| Llama-3.2-1B Q4_0 | 16.15 | **23.19** | 1.44× | 9.8 ms |
| Llama-3.2-3B Q4_0 | 6.36 | **9.51** | 1.50× | 21.9 ms |
| Qwen2.5-Coder-7B Q4_0 | 2.83 | **5.08** | 1.80× | 23.0 ms |

Speedup rises with model size because the per-token sync cost is roughly fixed
while compute halves. On the 3B row, local compute drops 157 → 76.8 ms — a clean
2.04× — and the shortfall from 2× is entirely the 21.9 ms of sync.

`sync` is latency, not bandwidth. A 28-layer model does 56 reductions per token at
~0.39 ms each; the hidden state is only 12.3 KB, which is ~0.1 ms of wire on GbE.
The rest is round-trip and syscall overhead, close to the floor for TCP between two
Pis. **Do not expect a wire-format change to help**; the earlier 70B work reached
the same conclusion from the other direction (sync collapsed 116 → 12 ms when one
node was slowed, proving the figure was wait-for-peer, not transfer).

## Pick your point on the curve

All three are the same command with a different model path:

| you want | model | 2-node tok/s |
|---|---|---|
| snappiest | Llama-3.2-1B Q4_0 | **23.2** |
| balanced (**default**) | Llama-3.2-3B Q4_0 | **9.5** |
| most capable | Qwen2.5-Coder-7B Q4_0 | **5.1** |

## What made it faster

1. **Q4_0 tied head, default-on** (`scarp_g2a_q4_head_default.md`) — but it did not
   reach the cluster until the fix below.
2. **TP head shards now honour `NANOCAMELID_Q4_HEAD`.** Both construction sites in
   `src/tp.rs` hardcoded a Q8_0-swizzled head, so a clustered run silently used a
   larger, slower, differently-rounded head than the same model on one board.
   Measured on the 3B row, 2 nodes, interleaved: **8.913 → 9.557 tok/s, +7.2%**,
   with `local_compute` unchanged at 76.7 ms — the entire delta is head time, which
   is the correct attribution for a head-only change.

## Two things that are NOT wrong, and cost me time

- **`cluster_tp_node reference` is a slow parity path**, not a baseline. It reports
  1.715 tok/s where the production single-node binary does 5.21 on the same model.
  Use `nanocamelid generate` for a single-node number.
- **The TP env prefix must not be applied to a single-node run.** TP sets
  `NANOCAMELID_Q4_SWIZZLE_1X4=0` because the shard splitter needs row-major input,
  and re-applies the swizzle after sharding via `tp::swizzle_shards`. A *single*
  node needs the swizzle at load; measuring it with the TP prefix understates it by
  **2.5×** (6.36 → 2.54 tok/s). An early draft of this table had exactly that bug
  and appeared to show TP being no faster than one board.

Also: a first 3B measurement of 5.27 tok/s failed to reproduce (9.5 on three later
runs). It was taken immediately after a 1.9 GB file copy, with a cold page cache.
**Do not benchmark a model you just transferred.**

## Constraints found

- **Wire TP refuses QK-norm architectures.** `error: wire tensor parallelism does
  not yet support QK-norm architectures (qwen3/gemma3)`. It fails closed with a
  clear message, which is right, but it means **Qwen3 and Gemma 3 cannot use the
  cluster at all** — including Qwen3-4B (5.21 tok/s single-node) and Gemma-3-1B
  (17.12), both otherwise strong fast-tier candidates. Lifting this is the single
  biggest expansion of what the cluster can serve.
- **Qwen2.5-Coder-7B has only 4 KV heads**, so a 3-node split would have to be
  `2,1,1` — badly unbalanced. Two nodes at `2,2` is the natural fit.
- **camelid1 is excluded.** Its 5V rail reads **4.896 V at idle** against 5.136 /
  5.131 on the other two, and it hard-rebooted under load with
  `throttled=0x50000` (under-voltage occurred). It needs a PSU or cable, not a
  config change.

## Why the third board would not help anyway, at 2 cores

TP time is set by the slowest shard. camelid1 is capped to 2 cores for safety, and
the 70B core-scaling test measured a 2-core node at **1.93×** the time of a 4-core
one. Normalising to "share ÷ throughput":

| split (c2, c3, c1) | c2 | c3 | c1 @2 cores | max |
|---|---|---|---|---|
| 3,3,2 | 0.75 | 0.75 | **0.97** | 0.97 |
| 4,3,1 | **1.00** | 0.75 | 0.48 | 1.00 |
| 4,4 (two nodes) | **1.00** | 1.00 | — | 1.00 |

The best 3-node split beats two nodes by ~3%, on a board that browns out under
load. **A healthy camelid1 at 4 cores is a different story** — it would take the
3B row to roughly 13 tok/s — which is the return on replacing that supply.

## Reproducing

```bash
# worker (camelid3)
env NANOCAMELID_CONTEXT_LIMIT=4096 NANOCAMELID_SPIN_POOL=0 \
    NANOCAMELID_Q4_SWIZZLE_1X4=0 NANOCAMELID_Q8_SWIZZLE_1X4=0 \
  cluster_tp_node worker <model.gguf> 0.0.0.0:5921 1 4,4

# head (camelid2) -- same env prefix
  cluster_tp_node master-serve <model.gguf> camelid3.local:5921 4,4 8090
```

Workers first, head last; the head blocks until every worker accepts. Copy models
between boards with netcat rather than relaying through a workstation — measured
**60 MB/s vs ~4 MB/s** — and verify the hash afterwards, because a truncated
stream is silent (one copy landed 33 bytes and reported success).
