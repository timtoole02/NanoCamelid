# Phase 4 P4.2 — two-node tensor parallelism over TCP

Date: 2026-07-03
Branch: feat/tp-wire (base c37f5e5)
Nodes: camelid2 (master, shard 0) + camelid3 (worker, shard 1) over GbE.
`cluster_tp_node worker|master-chat|reference`; both nodes load the GGUF
and keep their shard; embeddings replicated, head on the master; context
capped 512, PROMPT_SHORT, greedy.

Design: `tp_forward_token_reduced` calls a reducer at the two per-layer
reduction points; the wire reducer exchanges partials with a fixed
shard0+shard1 summation order, so both nodes hold bit-identical
activations (no cross-node drift by construction — the only numeric
difference vs a single process is the same reduction-order association
P4.1 characterized).

## Parity gate (hard) — PASS on the wire

Token-identical to the like-for-like single-process reference (same
row-major weights, scalar kernel pinned via `NANOCAMELID_TP_PARITY=1`):
- Mac loopback 1B Q8: TOKEN_IDENTICAL
- camelid2+camelid3 GbE, 1B Q4: TOKEN_IDENTICAL (32 tokens)
- camelid2+camelid3 GbE, 3B Q4: TOKEN_IDENTICAL (32 tokens)

## Speed (per-token breakdown from the master JSON)

| row | kernels | ref tok/s | TP-2 tok/s | speedup | local layers ms | sync ms | head ms |
|---|---|---|---|---|---|---|---|
| 1B Q4 | scalar | 6.17 | 7.54 | 1.22x | 42.9 | 9.6 | 84.3 |
| 1B Q4 | default | 6.88 | 7.71 | 1.12x | 32.8 | 14.6 | 84.3 |
| 3B Q4 | scalar | 2.91 | 3.96 | **1.36x** | 113.3 | 21.7 | 126.2 |
| 3B Q4 | default | 3.05 | 4.07 | **1.33x** | 102.8 | 21.7 | 126.2 |

- **Sync cost is exactly what Phase 0 predicted**: 56 reduce round trips ×
  ~12KB on the 3B row = 21.7ms/token measured (predicted 15–30ms). The
  wire is NOT the TP blocker on GbE at 2 nodes.
- The layer compute halves as designed (3B: ~205ms single → 103ms/node).
- **The replicated head dominates**: 126ms/token on 3B (row-major tied
  head, unsharded). Without it the 3B step would be ~125ms ≈ 1.6x.

## Honest gaps (the P4.3 work list)

1. **Kernel class**: TP shards slice row-major blocks, so the fast
   swizzled 1x4 kernels are unavailable — production single-node (5.33
   tok/s on 3B) still beats TP-2 (4.07). Shard-then-swizzle (build the
   shard, then swizzle its matrices) closes this; nothing about sharding
   forbids it.
2. **Head sharding**: split the LM head row-parallel (vocab slices) and
   merge argmax/top-k — turns 126ms into ~63ms + one 8-byte exchange.
3. **Third node**: 8 KV heads / 3 is uneven (3-3-2); the shard builder
   requires even splits today.
4. Prompt ingest is token-by-token through TP (no batch TP path); fine
   for decode receipts, slow for long prompts.

## Verdict vs the conductor

Two-node TP is **real and parity-clean on the wire**: 1.33–1.36x on the
3B row against a like-for-like baseline, with the all-reduce tax measured
at ~10% of the token budget. The ≥1.8x three-node promotion gate is not
yet testable (item 3), and beating the *production* single-node number
requires items 1–2. All three are mechanical, none require new protocol
work, and the sync headroom says 3-node TP stays wire-feasible.
