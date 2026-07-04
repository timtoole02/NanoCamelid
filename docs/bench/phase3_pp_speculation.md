# Phase 3 — speculation over the pipeline

Date: 2026-07-03
Commit: 9dc6cad + this doc (feat/nano-turbo)
Cluster: camelid1 master (2-core cap, hardware power fault) / camelid2
middle / camelid3 final; Mixtral 8x7B Q4_0 split 0..11/11..22/22..32,
`NANOCAMELID_CLUSTER_CONTEXT_LIMIT=512`, temp 0. Spin-pool fix (e02918b)
active — the post-fix plain baseline is 0.79 tok/s.

## Design

- Wire: `ACTIVATION_VERIFY_MAGIC` batch packets (identical layout to the
  existing batched-prefill packets) plus a `BATCH_FEEDBACK` reply carrying
  the target's greedy prediction after EVERY batch position. Middles relay
  both verbatim and compute their layer range with the existing batch path.
- Master (`master-chat-spec`): drafts k continuation tokens with the
  Phase 2 `NGramDrafter`, runs its own layer range over the k+1 batch,
  dispatches one verify traversal, and accepts the longest prefix where
  draft == target prediction. `NANOCAMELID_SPEC_K` (default 4).
- **No `truncate_kv` message** (deviation from the conductor's P3.2, with
  proof): positions are explicit in every packet, and a rejected suffix's
  KV rows sit at positions strictly above the accepted length — the next
  round's batch overwrites them before any later position attends them,
  the same invariant the single-node step relies on. Distributed KV
  rollback is therefore a no-op.

## Parity gate (hard) — PASS

- Mac loopback (1B Q8, 0..6/6..11/11..16, 48 tokens): token-identical to
  plain `master-chat` at k=4 and k=8.
- Mixtral three-Pi: token-identical on PROMPT_SHORT (spec k=4 vs plain)
  and on the repetitive prompt (spec k=8 vs plain).

## Mixtral speed A/B (same commit, capped master) — NEGATIVE, with the why

| run | tok/s | speculation |
|---|---|---|
| plain, PROMPT_SHORT, 64 tok | 0.791 | — |
| spec k=4, PROMPT_SHORT | 0.776 | 75% acc, but only 4 drafts total (coverage) |
| plain, repetitive, 128 tok | 0.803 | — |
| spec k=8, repetitive | **0.739** | **87.8% acc (101/115), 4.7 tok/round — and still slower** |

**MoE breaks batch-verify economics.** For a dense model a k+1-row verify
batch re-reads the same weights once (~1.15x a single token). For Mixtral,
each row routes to its own 2-of-8 experts, so a 9-row batch touches ~all 8
experts per layer — ~4x the weight traffic of a single decode step. The
verify traversal costs ~4-5x a plain token while yielding ~4.7 tokens:
break-even at best, a loss after the per-row logits and master batch
costs. High acceptance cannot rescue a batch that amortizes nothing.
(Same reason the Mixtral batched PROMPT_LONG prefill took 205s in Phase 0.)

This is the conductor's own caveat made concrete: speculation "partially
rescues the existing PP path" only where batched verification amortizes —
i.e. dense rows. Gate (≥1.4x on Mixtral): **NOT MET — committed as a
negative result with the mechanism identified.**

## Dense rows

3B Q4_0, split 0..10/10..19/19..28, capped master, token-identical parity:

| run | tok/s | speculation |
|---|---|---|
| plain, PROMPT_SHORT (Phase 0) | 2.991 | — |
| spec k=4, PROMPT_SHORT | 3.080 | 75% acc, 4 drafts in 50 rounds (coverage) |
| plain, repetitive, 128 tok | 3.136 | — |
| spec k=8, repetitive | 2.514 | 78.8% acc (82/104), ~4.7 tok/round — still 0.80x |

Round decomposition from the master breakdown (repetitive row): plain =
139.1ms master + 179.6ms downstream ≈ 319ms/token. Spec round (batch of
9) = **740.9ms master + 864.6ms downstream ≈ 1606ms** for ~4.74 tokens ≈
339ms/token. The verify batch costs ~5x a single token end to end:

- the 2-core-capped master pays 5.3x for a 9-row batch (139 → 741ms);
- the final node runs 9 sequential per-row norm+head+argmax passes;
- the engine's batched decode matmuls amortize ~9 rows for ~5x a single
  token's weight read — real but far from the ~1.2x ideal. (Single-node
  Phase 2 won 1.78x because an uncapped 4-core rayon batch runs ~3.2x and
  no pipeline stages stack.)

A ~4.7-token yield cannot beat a ~5x round cost. The blocker is not
speculation and not the wire — it is **small-batch weight-reuse
efficiency in the batched decode path**, plus the capped master hardware.

## 70B capacity row — memory-infeasible on current main

The historic 0..27/27..54/54..80 split OOM-killed the master (anon RSS
15.3G on a 16GB Pi): `load_distributed` materializes token embeddings as
f32, turning the ~0.6G Q4_0 `token_embd` into a 4.2G table. 80 layers ≈
36.8G of shards + 4.2G embeddings + head leaves no feasible 3×16G split.
Unblock: quantized embedding lookup (dequantize one row per token) —
filed as follow-up work.

## Verdict

- Parity gate: **PASS** (loopback + Mixtral short + Mixtral repetitive,
  all token-identical; 3B rows likewise).
- Speed gate (≥1.4x on Mixtral): **NOT MET — negative result committed.**
  Two independent mechanisms, both now measured: MoE routing destroys
  batch amortization structurally (each row picks its own experts), and
  even dense batch verification only amortizes ~9:5 in the current
  kernels, which a ~4.7-token acceptance yield cannot overcome.
- What promotes anyway: the verify-batch protocol and `master-chat-spec`
  are correct (token-identical), default-off, and become immediately
  useful the moment the batched decode path amortizes properly — that
  kernel work is the highest-leverage next item this campaign has
  identified, ahead of TP (whose per-token all-reduces face the same
  small-message compute-bound regime but whose ~3x ceiling divides the
  weight read itself).
- Env: `NANOCAMELID_SPEC_K` (default 4), mode `master-chat-spec`.
