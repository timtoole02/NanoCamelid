# SCARP G2a — Q4_0 tied LM head becomes the default

Phase: **G2a** — a default flip, not an implementation. The Q4_0-swizzled tied head
was written and merged in PR #4 and has been default-off since. SCARP G0 measured it
(`docs/bench/scarp_phase0_baseline.md` §7) and the campaign owner authorised the flip
on 2026-07-25.

Per R3 this is a **separate commit from any implementation change** — the only
behavioural edit is `q4_head_enabled()`'s default in `src/model.rs`.

---

## Why

Pi 5 decode is bandwidth-bound at ~11 GB/s (G0 §6), so tok/s is that constant divided
by bytes-per-token. The tied LM head is the largest single tensor a small model
streams. Building it Q4_0-swizzled instead of Q8_0-swizzled moves **147.8 MB/token
instead of 279.1 MB** on a 128k-vocab model — a straight bytes reduction, which on
this hardware converts almost 1:1 into speed.

## Result

camelid2, `scripts/pi/scarp-guard.sh` against the recorded G0 baseline
(`/mnt/nanocamelid/scarp/g0-baseline-v2`), `PROMPT_SHORT`, 64 greedy tokens,
`ondemand`, no taskset, rayon default.

| row | before | after | delta | decode RSS before → after |
|---|---|---|---|---|
| Gemma-3-1B-IT Q4_0 | 14.22 | **17.12** | **+20.4%** | 1925 → 1780 MiB |
| Llama-3.2-1B Q4_0 | 13.52 | **15.66** | **+15.8%** | 1974 → 1849 MiB |
| Qwen3-0.6B Q8_0 | 16.35 | **18.31** | **+12.0%** | 1375 → 1301 MiB |
| Llama-3.2-1B Q8_0 | 8.81 | **9.69** | **+10.0%** | 2462 → 2337 MiB |
| Llama-3-8B Q4_K_M **(untied control)** | 2.41 | 2.37 | −1.7% | 6580 → 6580 MiB |

`json: {"guard":"scarp","regressions":0,"allowed_drifts":4}`

The RSS drop is the head-size delta and nothing else: 279.1 − 147.8 = 131.3 MB on the
128k-vocab rows (measured 125 MiB), 320.9 − 169.9 = 151.0 MB on Gemma's 262k vocab
(measured 145 MiB).

**The untied control moved −1.7%, i.e. not at all.** That is the check that matters:
the flip touches only models that take the synthesized-tied-head path, and
Llama-3-8B, which ships a real `output.weight`, is provably unaffected.

`decode.head` share falls correspondingly: Gemma 37.4% → 23.5%, Llama-3.2-1B Q4_0
30.6% → 18.5%, Qwen3 22.0% → 13.2%, Llama-3.2-1B Q8_0 19.8% → 11.4%.

## Parity posture — output changes, and here is the exact bound

Four of five rows changed their greedy token stream. That is expected and it is
reported, not buried:

| row | Q8_0-head sha256 (prefix) | Q4_0-head sha256 (prefix) |
|---|---|---|
| Llama-3.2-1B Q4_0 | `83af152f5c8a480e` | `15a266908f7127a6` |
| Llama-3.2-1B Q8_0 | `f0bba13cf681c449` | `baf3768bbe02cae4` |
| Gemma-3-1B-IT Q4_0 | `a0d521e128a8b8f7` | `ca7b4048444c91ff` |
| Qwen3-0.6B Q8_0 | `604ad49dc91e12e4` | `738ad4a8afa5f927` |
| Llama-3-8B Q4_K_M | `d751990a06cdbdf0` | `d751990a06cdbdf0` **(unchanged)** |

Three things bound the risk:

1. **This is a change of degree, not of kind.** Both heads are re-quantizations of
   the model's own `token_embd` table — which on Llama-3.2-1B-Q4_0 is Q6_K on disk.
   Neither head has ever been the file's own numerics.
2. **The head is the last op in the forward pass.** Per the sister repo's hardest
   lesson (`Camelid/SIROCCO_LANEK.md:89`), a token-parity change is only defensible
   for a *final* projection, because a per-layer change compounds across layers and
   positions. The LM head is exactly that final projection — the narrow case where
   this posture is arguable.
3. **The escape hatch is bit-exact and verified.** `NANOCAMELID_Q4_HEAD=0` restores
   the Q8_0 head. Cross-checked on a different host and OS: on macOS the flag-off
   binary reproduces `f0bba13cf681c449`, the *same* hash the Pi baseline recorded for
   Llama-3.2-1B Q8_0. The restore path is not approximate.

Because the flag itself is the compatibility switch, no separate compat flag was
added — G7.3 can still retire the Q8_0 head only if that switch goes with it, which
it should not.

## Gate

Decode ≥ +10% on every tied row, no regression on the untied control, escape hatch
bit-exact, `scripts/validate.sh` green → **GO, default-on.**

## What this does not do

- Nothing for untied models (Llama-3-8B, Mistral-7B, the Qwen2.5-Coder rows). They
  take `output.weight` and are unaffected by construction.
- Nothing for the f32 `token_embd` table, which is still fully materialized and is
  still the largest single line item in resident memory (G0 §5). That remains the
  open half of G2.
