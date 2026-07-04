# Cluster decode capability table

The cluster tokens/sec campaign's final scoreboard (Phase 5 of
`CLUSTER_SPEEDUP_CONDUCTOR`). Every number has a committed receipt; the
per-phase documents in this directory carry the raw JSON breakdowns.
Cluster: 3× Raspberry Pi 5 16GB over GbE; camelid1 runs 2-core-capped
(hardware power fault) and holds the small tensor-parallel share.

| row | single node | 3-Pi pipeline (PP) | tensor parallel | best config |
|---|---|---|---|---|
| Llama 3.2 1B Q4_0 | 13.42 tok/s | — (fits one node) | **20.44** (TP-2, even split) | TP-2, 1.52x |
| Llama 3.2 3B Q4_0 | 5.33 tok/s | 2.99 (capacity only) | **10.21** (TP-3, shares 2-3-3) | TP-3, **1.91x** |
| Mixtral 8x7B Q4_0 | OOM (16GB) | **0.79–0.82** (post spin-pool fix) | MoE not sharded (own conductor) | PP |
| Llama 3 70B Q4_0 | OOM (16GB) | 0.160 | **0.685** (TP-3, shares 2-3-3) | TP-3, **4.3x** |

Receipts: `phase0_baseline.md` (baselines, network, the spin-pool find),
`phase2_spec_decode.md`, `phase3_pp_speculation.md`, `phase4_tp_smoke.md`,
`phase4_tp_wire.md`.

Campaign verdicts against the conductor's falsification targets:

- "3B: 2.22 → 5+ tok/s" → **10.21** (and the single node alone now does 5.33).
- "1B: 4.18 → 8+ tok/s" → **20.44**.
- "Mixtral: 1.12 → 1.6+ via PP+spec" → **NOT MET, mechanism identified**:
  MoE routing destroys batch-verify amortization (each verify row picks its
  own experts), and the batched decode kernels amortize ~9 rows for ~5x one
  token's cost even on dense rows. Speculation (`master-chat-spec`,
  `NANOCAMELID_SPEC_DECODE`) is token-identical and default-off; it pays
  today only on repetitive text (1.78x single-node) and becomes generally
  useful when the batch path learns true weight reuse — the top engine
  lever this campaign leaves behind.
- The ≥1.8x three-node TP gate: **1.91x** on the 3B row, with a capped
  straggler.

Known limits of the TP lane: dense models only, greedy only; per-row
token-identity is the promotion gate for any new row (f32 reduction-order
near-ties can flip argmax — observed only at never-consumed prompt
positions so far); prompt ingest is token-by-token (no batch TP path).
