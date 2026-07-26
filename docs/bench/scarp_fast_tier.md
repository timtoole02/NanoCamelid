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
- **camelid1 was excluded when this was first measured** (5V rail 4.896 V at idle
  against 5.136 / 5.131, and a hard reboot under load with `throttled=0x50000`).
  **Its supply was replaced on 2026-07-26 and it is now a full member** — see the
  update below.

## UPDATE 2026-07-26 — camelid1's supply was replaced; three nodes now

The section that used to sit here argued the third board could not help. That was
true of a board capped to 2 cores, and it is now obsolete.

**Verified fixed, not assumed.** Idle EXT5V went **4.896-4.931 V -> 5.124 V**
(siblings 5.147 / 5.136), and the board then survived the *exact* workload that
used to hard-reboot it inside two minutes — Strand-14B-Q6_K, all four cores,
unrestricted: peak load 3.82, rail held **5.097 V minimum**, `get_throttled`
stayed **0x0** for the whole run, output coherent at 1.01 tok/s.

Cap removed, split rebalanced `[4,3,1]` -> `[3,3,2]`, all three at four cores:

| model | 2 nodes | 3 nodes, capped `[4,3,1]` | **3 nodes, healthy `[3,3,2]`** |
|---|---|---|---|
| Llama-3.2-3B Q4_0 | 9.51 | — | **11.54 tok/s** |
| Qwen2.5-Coder-32B Q4_0 | — | 1.220 | **1.468 tok/s** |

Both ~1.20x, and the 32B's local compute fell 775 -> 596 ms, exactly the
3/8-vs-4/8 ratio. The one-line fleet health check that found this:
`vcgencmd pmic_read_adc | grep EXT5V_V` — a board 200+ mV below its siblings at
idle has a supply fault, because a load transient then takes it under the
~4.6-4.8 V trip. Idle voltage alone is not proof; confirm under sustained
four-core load, and read `get_throttled` *after* the run because the flag clears
on reboot.

**Sync is now the fast tier's ceiling, not compute.** On the 3B at three nodes:
local compute 60.5 ms, sync 21.7 ms, logits 7.1 ms. Sync is 24% of the token and
did not shrink when compute did (it was 21.9 ms at two nodes), because it is
round-trip latency, not transfer — the hidden state is only 12.3 KB. Eight KV
heads across three nodes also cannot balance: `[3,3,2]` puts 3/8 = 0.375 on the
busiest node against an ideal 0.333, a structural 12.5% tax.

## Reproducing

```bash
# worker (camelid3)
env NANOCAMELID_CONTEXT_LIMIT=4096 NANOCAMELID_SPIN_POOL=0 \
    NANOCAMELID_Q4_SWIZZLE_1X4=0 NANOCAMELID_Q8_SWIZZLE_1X4=0 \
  cluster_tp_node worker <model.gguf> 0.0.0.0:5921 1 4,4

# head (camelid2) -- same env prefix
  cluster_tp_node master-serve <model.gguf> <worker-host>:5921 4,4 8090
```

Workers first, head last; the head blocks until every worker accepts.

Copying models: **always verify the byte count afterwards**, because a truncated
stream is silent — one netcat copy landed 33 bytes and reported success. On
throughput, an early note here claimed a workstation relay ran at ~4 MB/s against
netcat's 60 MB/s. That was wrong: the relay was slow because the workstation was
busy compiling at the time. Idle it sustains ~107 MB/s, and a 40 GB copy to a Pi
averages **~39 MB/s**, which is the Pi's USB-SSD write ceiling and decays to
~10 MB/s past roughly 35 GB as the drive's SLC cache fills. The destination disk,
not the path, is the limit — so copying from a peer Pi is no faster than from the
workstation.
