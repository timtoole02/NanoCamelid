# Phase 2 — nano-turbo speculative decoding (single node)

Date: 2026-07-03
Node: camelid2 (Pi 5 16GB, governor `ondemand` — holds 2.4GHz under load,
measured during the tied-LM-head campaign)
Commit: 1e8f2c6 (feat/nano-turbo; base 84850ed origin/main)
Models:
- Llama-3.2-3B-Instruct-Q4_0 sha256 `506d311f2f8802991344f7186badffda9c6a6b4cb50aa7a759ba3d939544df44`
- Llama-3.2-1B-Instruct-Q4_0 sha256 `eadfd8fd4e29d48e720eb87fc8242d3a8d4d2dacd52c722adc8e69e48c668efc`

Prompts: docs/bench/PROMPTS.md (PROMPT_SHORT / PROMPT_LONG), chat path,
temp 0, 64 max tokens, `NANOCAMELID_CONTEXT_LIMIT=512`.

## Why n-gram and not a draft model (measured)

3B target + 1B Llama draft model (`NANOCAMELID_DRAFT_GGUF`, k=4), short
prompt: 5.20 tok/s plain → 5.10 tok/s spec-on **despite 89.6% acceptance
(43/48)**. On a Pi the 1B draft is itself memory-bandwidth bound (~76 ms per
drafted token vs ~190 ms per 3B target token), so sequential drafting eats
the entire batched-verify win. A useful draft source must be near-zero-cost.

The conductor's "port camelid-turbo (n-gram MLP draft)" is not portable:
no camelid-turbo crate exists in any repo on this machine — that campaign
was parked before the crate landed (only "parked turbo vocab-check hunks"
survive in memory/notes). What was ported instead is the prompt-lookup
n-gram drafter design, which needs no training and no weights.

## Design

- `NGramDrafter`: longest recent n-gram suffix match (3..4-gram), proposes
  the historical continuation; zero weights, O(history) per step.
- `ngram_speculative_step`: samples the target's own next token t0 from the
  live logits, drafts k continuations, verifies t0+drafts in ONE batched
  forward pass (`prefill_pass_batch`), accepts the longest matching prefix.
  Rejected-suffix KV rows are overwritten in place (positions past the
  current pos are never attended). Stop tokens are neither emitted nor
  consumed — plain-loop semantics.
- Env: `NANOCAMELID_SPEC_DECODE=1`, `NANOCAMELID_SPEC_K` (default 4),
  `NANOCAMELID_SPEC_DRAFT=ngram|<draft-gguf>`. Default off.
- Fixed in passing: the model-draft loop counted stop tokens as generated
  (plain loop does not), inflating counts by one and breaking token-stream
  comparison.

## Parity gate (hard) — PASS

Token-id streams (`NANOCAMELID_EMIT_TOKEN_IDS=1`) compared against plain
greedy decode: **11/11 TOKEN_IDENTICAL** (3B short × k∈{2,4,6,8}, 3B long ×
k∈{2,4,6,8}, 1B short k=4, 1B long k=4, 3B repetitive k=8).

## k sweep (3B row, tok/s and acceptance)

| run | plain | k=2 | k=4 | k=6 | k=8 |
|---|---|---|---|---|---|
| short | 5.64 | 5.79 (50%, 2/4) | **5.85** (75%, 3/4) | 5.75 (50%, 3/6) | 5.65 (37.5%, 3/8) |
| long | 5.35 | 5.48 (100%, 4/4) | 5.40 (50%, 4/8) | **5.48** (83%, 5/6) | 5.44 (62.5%, 5/8) |

1B row: short 13.15 → 13.60 (k=4, 75%); long 12.91 → 12.59 (k=4, 16.7%).

The limiting factor is **coverage, not acceptance**: across a 64-token
generation the drafter only proposed 4–12 tokens total on these prompts
(novel prose has few repeating 3-grams), so ≥90% of steps ran plain.

## Upside case: repetitive/structured text

Prompt "Repeat the sentence: The quick brown fox... ten times, one per
line.", 3B, 128 max tokens, k=8:

- plain: 5.57 tok/s
- ngram k=8: **9.91 tok/s = 1.78x**, 85.4% acceptance (82/96), TOKEN_IDENTICAL

## Verdict

- Parity gate: **PASS** everywhere (that was the "full stop" requirement).
- Promotion threshold (≥1.3x on the 3B row, canonical prompts): **NOT MET**
  (~1.01–1.04x). Committed as a negative result per the campaign's standing
  rules. The mechanism works — 1.78x where coverage exists — but prompt
  lookup alone cannot cover novel prose, and the trained draft crate the
  conductor assumed does not exist to port.
- `NANOCAMELID_SPEC_DECODE` stays **default-off**; no README/catalog lane
  changes.
- What Phase 3 keeps regardless: the batched-verify step machinery is
  proven token-identical, and batch-k verification is exactly the piece
  that fills pipeline bubbles in PP decode. Phase 3's win does not require
  high draft coverage to be measurable on the Mixtral row, but its size
  scales with coverage, so expectations should be set by these numbers.
