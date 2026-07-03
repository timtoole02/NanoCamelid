# Phase 4 P4.1 — tensor-parallel split parity (single process)

Date: 2026-07-03
Commit: this tree (feat/nano-turbo)
Harness: `cluster_tp_split_smoke <model> [shards] [max_tokens] [prompt]`
(Mac, scalar Q8 kernel pinned, 1x4 swizzle disabled so shards slice
row-major blocks).

## Sharding (src/tp.rs)

Megatron-style, GQA-aware, dense-only: wq/wk/wv row-sliced by head group
(contiguous — GQA groups stay aligned so the q→kv mapping is preserved per
shard), wo column-sliced (row-parallel partial), w1/w3 row-sliced
(column-parallel), w2 column-sliced (row-parallel partial); norms,
embeddings, and head replicated. Two in-process partial-sum reductions per
layer. All slices sit on 32-value quant-block boundaries; per-shard
activation quantization is block-identical to the full path (1024/2048/4096
slice widths are 32-aligned), so the only numeric difference is f32
summation order in the reductions.

## Gate

| row | prompt | steps | decode mismatches | prompt-phase flips | max logit delta |
|---|---|---|---|---|---|
| 1B Q8_0, 2 shards | short | 45 | **0** | 0 | 0.695 |
| 1B Q8_0, 2 shards | long | 180 | **0** | 1 (Δ0.19) | 0.695 |
| 3B Q8_0, 2 shards | short | 45 | **0** | 1 (Δ0.12) | 0.662 |
| 3B Q8_0, 2 shards | long | 180 | **0** | 0 | 0.662 |

`result: PASS` on all four; generated text coherent and identical to the
reference stream.

## Honest caveats

- The reduction-order drift band is ~0.7 logits; near-tie argmax flips are
  possible in principle at decode positions on other prompts (two were
  observed at never-consumed prompt positions here). Strict bit-stability
  for TP requires a pinned-reduction-order kernel lane where the reference
  fold matches the shard tree (the Camelid deterministic-lane precedent).
  Per-row token-identical receipts remain the promotion gate for any wire
  deployment.
- 3-shard splits of these rows are uneven (8 KV heads / 3); the builder
  currently requires even divisibility — uneven sharding is wire-phase
  work.
- MoE rows are rejected by design (expert sharding is its own conductor).

## What Phase 3's findings predict for P4.2 (wire TP)

TP divides the per-token weight READ (each node streams half the bytes) —
it does not depend on batch amortization, so the Phase 3 blocker does not
apply. Expected per-token sync cost on GbE: 2 reduces × 28-32 layers of
~8-16KB messages ≈ 56-64 small round trips ≈ 15-30ms against a ~90-160ms
compute saving on the 3B row. Phase 0's RTT receipts say the wire budget
holds; the open risk is sync jitter, which the P0.2-style breakdown will
measure from day one.
