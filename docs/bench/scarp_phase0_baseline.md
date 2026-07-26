# SCARP Phase G0 — instrument the head, baseline, regression guard

Campaign: **SCARP** (bytes-moved-per-token on a Pi 5).
Phase: **G0** — "this phase cannot fail; it can only inform or kill."
Date: 2026-07-25. Base commit: `af80471` ("Performance updates"), plus the G0.1
instrumentation and the G0.5 guard landed by this phase.

---

## 0. Gate G0 verdict — GO, but the campaign must be re-scoped

**`decode.head` is 19–37% of decode-forward time on the tied rows and 8.1% on the
untied control.** Two tied rows clear 25% (Llama-3.2-1B Q4_0 at 30.6%, Gemma-3-1B-IT
at 37.4%), so by the letter of the G0 gate (`≥ 25%` on **any** tied row) that is
**GO to G1, with G2 confirmed as the priority phase.**

But the reason the head is expensive is **not** the reason the conductor document
gives, and the fix the document specifies for G2 **is already merged and default-on**.

| SCARP §2 claim | verdict |
|---|---|
| "on tied-embedding models the LM head streams the token-embedding matrix in **f32** once per token" | **REFUTED.** The tied head runs a Q8_0-swizzled SDOT matmul. The f32 path is dead by default. |
| "…and is currently the single largest consumer of memory bandwidth in the decode step" | **REFUTED as stated.** Layer weights are the largest consumer on every row (55–91% of bytes/token). The head is the largest *single tensor*. |
| The f32 embedding table is resident and shows up in RSS | **CONFIRMED, and it is worse than §2 says** — the f32 table is resident *in addition to* a second, quantized copy of the same matrix. |
| §2's byte arithmetic (1.05 GB f32 vs 279 MB Q8_0 for Llama-3.2-1B) | **CONFIRMED exactly.** Only the claim about which one decode reads is wrong. |

The campaign survives, with a different headline: **decode on a Pi 5 runs at a
near-constant ~11 GB/s wall, and tok/s is that constant divided by bytes-per-token.**
Three independent estimates of that wall land between 10.0 and 11.5 GB/s across a
0.6B and an 8B model (§6, §7), and a pure byte-count model predicts a measured
speedup to within **1.2%** (§7). Every phase that removes bytes converts ~1:1 into
tok/s; every phase that chases *achieved* bandwidth has ~35% of theoretical headroom
at best, and a measured null in the sister repo.

See §8 for the re-scoped phase list and §9 for the corrections the conductor
document needs.

---

## 1. Environment (R2)

| | |
|---|---|
| host | `camelid2` — Raspberry Pi 5 Model B Rev 1.1, 4× Cortex-A76, 16 GB LPDDR4X |
| kernel | Linux 6.12.75+rpt-rpi-2712 aarch64 (Debian trixie) |
| governor | `ondemand` on both sides of every A/B (R5; `performance` still sudo-gated — see `phase0_baseline.md`) |
| throttling | `vcgencmd get_throttled` = `0x0` before and after every run |
| threads | rayon default (4 workers), **no** `taskset`, `NANOCAMELID_RAYON_THREADS` unset |
| cross-host | `camelid3`, identical hardware/OS/governor, used for replication and for the A/Bs in §6 and §7 |
| excluded | `camelid1` — the brown-out node (R4). All three boards were online for this phase; camelid1 was deliberately kept out of every timing measurement and used only to read a GGUF header. |
| toolchain | rustc 1.96.1 (31fca3adb 2026-06-26), `cargo build --release`, no profile overrides (G1 has not run) |
| model dir | `/mnt/nanocamelid/models` |
| prompts | `PROMPT_SHORT` and `PROMPT_LONG`, verbatim from `docs/bench/PROMPTS.md` |
| decode | greedy (`temp 0.0`), 64 tokens, prefill batch 16 (default) |
| repeats | 2 interleaved passes; rows are interleaved *within* a pass, not run to completion one at a time |

Binaries. Rust release builds are not byte-reproducible across different
absolute build paths, so each host's binary is hashed separately:

| build | host | sha256 |
|---|---|---|
| `af80471` + G0.1 instrumentation | camelid2 | `9f402a5d4aed69c673c8f2ba89f0dd44311d996ae9a533ae1b2e9cea5b498e21` |
| `af80471` + G0.1 instrumentation | camelid3 | `82a72f4a3600cc09858fc46f48500a21774fab14c132b63fa860eee8c33f0224` |
| `af80471` pristine (overhead A/B only) | camelid3 | `7362ec23f42b70b6b6ff29032d173d3c57bb930487db61d2058cf68e854c232f` |

Models: see §10 for SHA256 and the GGUF tensor facts each row depends on.

---

## 2. G0.1 — what was instrumented

`compute_logits_from_hidden` had zero `trace_record` calls. All 37 pre-existing
stages sit inside `run_layer_range` / `run_layer_range_batch`, so the most
expensive single operation in the decode step was invisible to the profiler.
Six stages were added:

| stage | wraps | note |
|---|---|---|
| `decode.head_total` | all of `compute_logits_from_hidden` | mirrors `decode.layer_total` |
| `decode.head_norm` | the final `rms_norm` | |
| `decode.head_quant` | `quantize_f32_to_q8_0` of the hidden state | fires **only** on the quantized-head branch — its `calls` count is the branch witness |
| `decode.head` | the head matmul itself, either branch | the campaign's numerator |
| `decode.embed_lookup` | `embed_token` | |
| `batch.embed_lookup` | the prefill embedding gather | |

Plus one instrument repair in `print_runtime_trace_summary`: it printed
`rows.take(24)` while the codebase already had 37 stages, so the table was
**already silently truncated** before SCARP added anything. It now prints every
stage.

### Deviations from the G0.1 spec, and why

- **`batch.head` was not added: it does not exist.** `prefill_pass_batch`
  (`src/inference.rs:3507`) ends at `run_layer_range_batch` and never calls
  `compute_logits_from_hidden`. Prefill does not compute logits. Two consequences
  worth writing down: the head is a **decode-only** cost, so any prefill-side head
  lever is vacuous; and `decode.head*` is structurally immune to prefill
  contamination, which the layer stages are not.
- **`decode.head_total`, `decode.head_quant` and `decode.forward_total` were added
  beyond the spec.** The G0 gate is defined as a *percentage of decode token time*
  and the trace had no denominator for that — `generation_sec` from the JSON line
  excludes the first forward pass. `decode.forward_total` wraps `forward_pass_inner`
  and gives the gate an exact, in-trace denominator.

### The instrumentation is free, and output-identical

Six interleaved runs per row on `camelid3`, `NANOCAMELID_TRACE` **unset**, pristine
`af80471` vs the instrumented build:

| row | pristine mean tok/s | instrumented mean tok/s | delta | completion sha256 |
|---|---|---|---|---|
| Llama-3.2-1B Q4_0 | 13.343 | 13.333 | **−0.07%** | identical (`83af152f5c8a480e`) |
| Llama-3.2-1B Q8_0 | 8.513 | 8.503 | **−0.12%** | identical (`f0bba13cf681c449`) |

Both deltas are inside run-to-run noise, and the generated token streams are
byte-identical. The added `Instant::now()` pairs cost ~15–30 µs/token against a
~75 ms token; a prior estimate that this could matter for a 5% gate is wrong by
two orders of magnitude.

---

## 3. The finding that re-scopes the campaign

`LlamaWeights::load` (`src/model.rs:471-532`) does **not** leave a tied model on the
f32 path. When the GGUF has no `output.weight` tensor it *synthesizes* one:

```rust
let output_projection = if gguf.tensors.iter().any(|t| t.name == "output.weight") {
    Some(load_quantized_matrix(&mmap, gguf, "output.weight")?)
} else {
    // Tied embeddings: ... re-quantize the already-dequantized f32
    // `token_embeddings` table ... to a Q8_0-swizzled matrix.
```

The gate is `vocab % 4 == 0 && n_embd % 32 == 0 && q8_swizzle_1x4_enabled()`, and
`q8_swizzle_1x4_enabled()` is `.unwrap_or(true)` (`src/model.rs:1349`). So
`compute_logits_from_hidden` takes the `Some(out_proj)` arm and runs
`matmul_q8_0_swizzled_1x4`. **All four tied rows in the G0.2 matrix hit this path;
`matmul_f32` never ran once in this entire phase.** The `decode.head_quant` call
counts in §4 are the direct witness — that stage exists only on the quantized branch,
and it fired 64 times (once per token) on every row.

This is not new work waiting to be done. It is `perf/fast-tied-lm-head`, already
merged, and the repo's own `docs/bench/phase0_baseline.md:203` already says so in
passing: *"the tied Q8-swizzled head runs on rayon"*. SCARP §2 restates a closed
finding as an open one.

**What survives, and it is substantial.** The f32 table is still fully materialized
(`load_f32_or_f16` → `Vec<f32>`, `src/model.rs:468`) and kept resident for the sole
purpose of `embed_token`'s **one-row-per-token** lookup. So a tied model now pays for
the embedding matrix *twice, simultaneously*: 1.05 GB of f32 plus 279 MB of Q8_0 on
Llama-3.2-1B. On the **untied** 8B control it is worse in kind — 2.10 GB of f32 that
the head never touches at all, because the head has its own `output.weight`. That is
a pure-RSS defect, it is real, and nothing in the tree addresses it.

---

## 4. G0.2 — `decode.head` as a share of decode

camelid2, mean of 2 interleaved repeats, 64 greedy tokens. Denominator is
`decode.forward_total` (the per-token model forward), which is exact: its `calls`
equals the generated-token count on every run, and `decode.layer_total` equals
`layers × tokens`, so there is no single-token-prefill contamination in either.

| row | head path taken | prompt | tok/s | `decode.head` ms/tok | `decode.forward_total` ms/tok | **head % of decode** |
|---|---|---|---|---|---|---|
| Llama-3.2-1B Q4_0 | quantized (Q8_0-swizzled tied) | short | 13.28 | 23.33 | 76.16 | **30.6%** |
|  |  | long | 12.63 | 23.39 | 80.05 | **29.2%** |
| Llama-3.2-1B Q8_0 | quantized (Q8_0-swizzled tied) | short | 8.66 | 23.20 | 117.02 | **19.8%** |
|  |  | long | 8.37 | 23.44 | 121.12 | **19.4%** |
| Gemma-3-1B-IT Q4_0 | quantized (Q8_0-swizzled tied) | short | 14.05 | 26.87 | 71.81 | **37.4%** |
|  |  | long | 13.72 | 27.47 | 73.52 | **37.4%** |
| Qwen3-0.6B Q8_0 | quantized (Q8_0-swizzled tied) | short | 16.13 | 13.78 | 62.57 | **22.0%** |
|  |  | long | 14.15 | 14.12 | 71.31 | **19.8%** |
| Llama-3-8B Q4_K_M **(untied control)** | quantized (`output.weight` Q6_K) | short | 2.40 | 34.46 | 422.95 | **8.1%** |
|  |  | long | 2.32 | 36.47 | 437.40 | **8.3%** |

**The untied control behaves exactly as the gate demands**: 8.1–8.3% against
19.4–37.4% on the tied rows. The instrument is validated by its own control.

`decode.head_quant` fired 64 times (once per token) on **all five rows**, including
the four tied ones. That stage exists only inside the `Some(out_proj)` branch, so it
is direct proof that no row ran `matmul_f32`. The head_norm and embed_lookup stages
are noise-floor everywhere (`decode.head_norm` ≈ 0.05 ms *total* over 64 tokens;
`decode.embed_lookup` ≈ 0.1 ms total), so `decode.head` ≈ `decode.head_total`.

The 1B Q4_0 short number reproduces `docs/bench/phase0_baseline.md:203`'s
*"idle camelid2: 1B 13.42"* to within 1%, three weeks and ~25 commits later.

### Cross-host replication (camelid3, same script, same governor)

| row | camelid2 tok/s | camelid3 tok/s | delta | camelid2 head% | camelid3 head% |
|---|---|---|---|---|---|
| Llama-3.2-1B Q4_0 | 13.28 | 12.92 | −2.7% | 30.6% | 30.3% |
| Llama-3.2-1B Q8_0 | 8.66 | 8.38 | −3.2% | 19.8% | 19.6% |
| Llama-3-8B Q4_K_M | 2.40 | 2.19 | −9.0% | 8.1% | 7.8% |

Generated token streams are **byte-identical across the two hosts** on all three
rows. camelid3 is a few percent slower in absolute terms (it had been up 23 days
with a warm page cache and an idle `serve-stage` process resident; camelid2 was
freshly rebooted) — but the **head share is host-independent to within 0.3 points**,
which is the quantity the gate is defined on.

---

## 5. G0.3 — RSS reconciliation

Two different numbers, and the conductor document conflates them. `VmHWM` is the
process high-water mark and it peaks *during weight load*, when the mmap pages and
the decoded `Vec` copies are both resident. What matters for a running server is
what is resident while tokens are produced. The guard records both: `VmHWM`, and
the median `VmRSS` over the generation window.

| row | file on disk | f32 `token_embd` table | quantized head | peak RSS (VmHWM) | **decode RSS** | f32 table as % of decode RSS |
|---|---|---|---|---|---|---|
| Llama-3.2-1B Q4_0 | 763 MB | 1051 MB | 279 MB | 2690 MiB | 1974 MiB | **52%** |
| Llama-3.2-1B Q8_0 | 1313 MB | 1051 MB | 279 MB | 3703 MiB | 2462 MiB | **42%** |
| Gemma-3-1B-IT Q4_0 | 715 MB | 1208 MB | 321 MB | 2602 MiB | 1925 MiB | **61%** |
| Qwen3-0.6B Q8_0 | 633 MB | 622 MB | 165 MB | 1963 MiB | 1375 MiB | **44%** |
| Llama-3-8B Q4_K_M **(untied control)** | 4913 MB | 2101 MB | 431 MB | 11240 MiB | 6580 MiB | **31%** |

The reconciliation closes. Llama-3.2-1B Q4_0: layer weights 548 + head 279 +
f32 table 1051 = 1878 MB against a measured 1974 MiB (2070 MB) — a ~190 MB
remainder for KV cache, workspace, logits and allocator slack. Llama-3-8B:
4186 + 431 + 2101 = 6718 MB against 6580 MiB (6900 MB).

**§2's RSS claim is confirmed and it is the campaign's surviving thesis.** The
`Vec<f32>` copy of `token_embd` is 31–61% of everything resident during decode, and
its only consumer is `embed_token`, which reads **one row of it per token**
(`decode.embed_lookup`: 0.1 ms across 64 tokens). On the untied 8B row it is
2.10 GB that the LM head never touches at all.

Two corrections to §2's framing, both making the case stronger, not weaker:

- The catalog's sampled-RSS column is a decode-style sample, not a high-water mark.
  Qwen3-0.6B Q8_0 reads 1.41 GiB in `docs/MODEL_CATALOG.md`; this phase measures
  1375 MiB (1.34 GiB) decode RSS against a 1963 MiB peak. Quoting VmHWM against the
  catalog would compare two different quantities.
- A tied model does not hold *one* copy of the embedding matrix, it holds **two** —
  the f32 table plus the synthesized Q8_0-swizzled head built from it. §2 assumed
  the f32 copy *was* the head.

---

## 6. G0.4 — achieved bandwidth

Bytes-touched per decode token is derived from the GGUF tensor table: every layer
tensor at its on-disk quantization (the loader copies blocks into `Vec`s at the same
size), plus the head at its resident representation. `token_embd` is **excluded** —
it is dequantized to f32 at load and only one row is read per token. KV-cache reads
are excluded too; at these contexts they are 0.4–1.1% of the total and do not move
any conclusion.

| row | layer bytes/tok | head bytes/tok | total bytes/tok | tok/s | **achieved GB/s** | % of 17.07 GB/s peak |
|---|---|---|---|---|---|---|
| Llama-3.2-1B Q4_0 | 548 MB | 279 MB | 827 MB | 13.28 | **10.98** | 64% |
| Llama-3.2-1B Q8_0 | 1034 MB | 279 MB | 1313 MB | 8.66 | **11.37** | 67% |
| Gemma-3-1B-IT Q4_0 | 395 MB | 321 MB | 715 MB | 14.05 | **10.05** | 59% |
| Qwen3-0.6B Q8_0 | 468 MB | 165 MB | 633 MB | 16.13 | **10.22** | 60% |
| Llama-3-8B Q4_K_M | 4186 MB | 431 MB | 4617 MB | 2.40 | **11.08** | 65% |

**10.05–11.37 GB/s across a 0.6B and an 8B model — a 7.3× span in model size and a
13% span in achieved bandwidth.** Ceiling reference: Pi 5 LPDDR4X-4267 on a 32-bit
bus = 17.07 GB/s theoretical. There is no STREAM-class tool installed on these
boards, so "% of peak" is against the theoretical figure, not a measured triad; the
marginal measurement in §7 is the better estimate of the practical wall.

The head's *time* share tracks its *byte* share, slightly under it — the head kernel
is a little more bandwidth-efficient than the layer kernels, which is what you would
expect from one large contiguous streaming matmul versus seven smaller ones:

| row | head % of bytes/tok (predicted) | head % of decode time (measured) |
|---|---|---|
| Llama-3.2-1B Q4_0 | 33.8% | 30.6% |
| Llama-3.2-1B Q8_0 | 21.3% | 19.8% |
| Gemma-3-1B-IT Q4_0 | 44.9% | 37.4% |
| Qwen3-0.6B Q8_0 | 26.1% | 22.0% |
| Llama-3-8B Q4_K_M | 9.3% | 8.1% |

**This is the number that ranks the rest of the campaign.** Decode is not partly
bandwidth-bound; it is bandwidth-bound to within ~13% across the whole catalog, at
~65% of theoretical DRAM peak. Removing bytes converts to tok/s almost 1:1
(quantified in §7). Chasing achieved bandwidth has ~35% of theoretical headroom in
principle, and the sister repo already measured a software-prefetch attempt at that
headroom as **−0.8%** (`Camelid/docs/perf-deep-dive/KQUANT_RECON.md:181-192`,
default-off, reason given: "the HW prefetcher already covers the sequential stream").

---

## 7. Measured: the one remaining head lever is already in the tree, default-off

`NANOCAMELID_Q4_HEAD=1` (`src/model.rs:1423`, default **false**) builds the tied head
as a Q4_0-swizzled matrix instead of Q8_0 — 147.8 MB/token instead of 279.1 MB/token.
It is implemented, it is untested by any receipt, and it is the whole of what remains
of SCARP G2's bandwidth ambition. Measured here because Gate G0 has to be re-cut
against it.

camelid3, four interleaved passes A/B/A/B, 64 greedy tokens:

| row | prompt | A: default (Q8_0 head) | B: `Q4_HEAD=1` | **delta** |
|---|---|---|---|---|
| Llama-3.2-1B Q4_0 | short | 12.56 | 15.11 | **+20.3%** |
| Llama-3.2-1B Q4_0 | long | 12.44 | 14.62 | **+17.6%** |
| Llama-3.2-1B Q8_0 | short | 8.34 | 9.37 | **+12.3%** |
| Llama-3.2-1B Q8_0 | long | 8.24 | 9.06 | **+10.0%** |

`decode.head` per token: 24.08 → 12.19 ms (Q4_0 row), 23.92 → 12.50 ms (Q8_0 row) —
almost exactly halved, as halving the head bytes predicts. Peak RSS drops by
128,208 KB = **131.3 MB**, which is the head-size delta (279.1 − 147.8 = 131.3 MB)
to the byte.

### The byte model predicts the measurement to ~1%

This is the strongest evidence in the phase, because the two sides differ in exactly
one quantity:

| row | predicted speedup (bytes ratio) | measured speedup | error |
|---|---|---|---|
| Llama-3.2-1B Q4_0 | 826.7 / 695.4 = **1.189×** | 15.11 / 12.56 = **1.203×** | +1.2% |
| Llama-3.2-1B Q8_0 | 1313.3 / 1181.9 = **1.111×** | 9.37 / 8.34 = **1.123×** | +1.1% |

Both errors are in the same direction — measured slightly beats predicted — which is
what the §6 byte-share-vs-time-share table already implies: the head kernel streams a
little faster per byte than the layer kernels, so removing head bytes buys marginally
more than strictly proportional. The model also omits KV reads. Neither correction is
worth applying at this precision.

And the marginal bandwidth — Δbytes ÷ Δtime, free of every fixed overhead — is
**11.05 GB/s** (Q4_0 row) and **11.50 GB/s** (Q8_0 row). That agrees with the
whole-decode figures in §6 and is the best estimate this phase has of the practical
memory wall: **≈11 GB/s, ≈65% of theoretical peak.**

### Parity posture

Token streams **change** (A `83af152f…` → B `15a…` on the Q4_0 row), which is
expected: the Q4_0 head is a re-quantization of an already-dequantized table, so it
is strictly lossier than the Q8_0 head. Both sides are internally deterministic —
each reproduced its own hash across repeats. This lever therefore **cannot** be gated
on token-identity, and per the sister repo's hardest-won lesson
(`Camelid/SIROCCO_LANEK.md:89`) token-parity is only defensible for a **final**
projection — which the LM head is, exactly. That is the narrow case where it is
arguable. It is Tim's call (D-S1/D-S2), not this phase's.

**Nothing here was flipped on.** `NANOCAMELID_Q4_HEAD` remains default-off; this
section is measurement, not implementation.

---

## 8. Re-scoped campaign

Recommendation, in priority order. Every phase below is re-cut against the ~11 GB/s
wall: **a phase is worth doing in proportion to the bytes/token it removes.**

### Keep, but lower the expectation — **G1 (build configuration)**

Premises verified: `Cargo.toml` has no `[profile.release]`, there is no
`.cargo/config.toml`, `build.rs` sets no rustflags, and the only ISA escape hatch is
`.arch_extension dotprod` inside inline asm — so every autovectorized loop outside
the hand-written kernels is compiled for baseline ARMv8.0-A. Still the cheapest phase
in the campaign and bit-identical by construction, so **run it first**. But two
priors say not to expect much from the `target-cpu` half:

- The `perf/fast-tied-lm-head` campaign already measured `target-cpu=native` on these
  boards as **no help**, reasoning that the SDOT kernels are runtime-selected and
  therefore codegen-independent.
- §6 says decode is bandwidth-bound, and the scalar loops `target-cpu` would actually
  improve (`rms_norm`, `softmax`, the norms) are together under 0.3% of decode token
  time.

Where it plausibly *does* pay is off the decode path: the `model.rs` dequant loops
(weight load is 20.6 s for a 763 MB Q4_0 file) and prefill. G1.4 should therefore
report **load time and prefill** alongside decode, or the phase will read as a null
that wasn't. `lto = "fat"` + `codegen-units = 1` is untested by any prior and is the
half worth being hopeful about.

### Re-scope — **G2, from "quantize the tied head" to "stop materializing the f32 embedding table"**

The bandwidth half of G2 is merged (§3). What is left is the RSS half, and it is
worth more than the document credits:

1. **Drop the f32 `token_embd` table.** 31–61% of decode-resident memory (§5), used
   for a one-row-per-token lookup. Fix = the G2.2 `dequantize_row` helper the
   document already specifies. **Do not invent it**: `tp::load_embeddings_f32` /
   `load_rows_direct` (`src/tp.rs:1037-1057`) already dispatches Q4_0/Q8_0/Q6_K to a
   block-level slice of the mmap and dequantizes from there, and
   `tp::split_tied_head_shards` (`src/tp.rs:485-510`) already builds Q8-swizzled tied
   heads. Port from `tp.rs`.
   - Blocker to design around: the synthesized head is currently built *from* the f32
     table (`src/model.rs:508-512`). Removing the table means quantizing the head
     straight from the on-disk blocks.
   - Bar: **bit-identical** if the head is built to the same bits; a pure RSS and
     load-time win with no numerics change. This is the rare phase that is both large
     and free.
   - Secondary prize: weight load is 20.6 s for a 763 MB Q4_0 file, and the
     dequantize-then-requantize round trip is most of it.
2. **Then decide `NANOCAMELID_Q4_HEAD` on its own merits** (§7): +10–20% decode,
   already implemented, changes output. This is a product decision (D-S1/D-S2), not
   an engineering one. Note it is not new evidence either — PR #4 already measured
   this flag at +16.6% (13.41 → 15.63 tok/s) on the same row and board class. §7
   reproduces that independently and adds the byte-model explanation for it.

### Demote and re-gate — **G3 (prefetch)**

The document says G3 ports "STAMPEDE P2.1". **STAMPEDE Phase 2 was never executed** —
its ledger row (`Camelid/docs/perf-deep-dive/STAMPEDE_CONDUCTOR.md:163`) reads
`pending`, and `{2,4,8,16}` is its *design sweep list*, not a result. There is no
winning distance to port.

The nearest real receipt is the opposite result: x86 weight-stream prefetch was built
and measured **NULL, −0.8%, byte-identical, shipped default-off**
(`Camelid/docs/perf-deep-dive/KQUANT_RECON.md:181-192`), explicitly because decode is
DRAM-bound and the hardware prefetcher already covers a sequential stream. §6 says
NanoCamelid is in the same regime. Note also that Camelid's only live `prfm` is
**macOS-gated**, so on aarch64-Linux it has no prefetch at all.

The two defects G3 names are still real (three of four streams unprefetched; distance
expressed in 18-byte blocks, inside one cache line) and the fix is correct regardless
of whether it is measurable. **Keep G3 as a correctness-of-intent cleanup with a
null-expected prior, not as a perf phase.** Its existing "+0–4% → ship anyway, note
the null result honestly" branch is the one to plan for.

### Re-scope — **G4, from "loop inversion" to "register/row blocking"**

The premise is wrong per-type. With defaults on a Pi 5 (`Q4_SWIZZLE_1X4` /
`Q8_SWIZZLE_1X4` true, `dotprod` present), **every batch kernel a real
Llama/Qwen/Gemma GGUF reaches is already block-outer with `unpack()` hoisted**:
Q4_0-swizzled (`src/inference.rs:1610`), Q8_0-swizzled (`:4940`), Q4_K (`:1982`),
Q6_K SDOT (`:1858`), Q2_K/Q3_K/Q5_K/Q8_K/IQ4_NL. The row-major token-outer kernels
the document cites (`matmul_q8_0_batch:1319`, `matmul_q4_0_batch:1361`, …) are the
**non-swizzled fallbacks**. The single true unpack-per-token survivor is the Q6_K
**scalar** fallback, i.e. non-dotprod hardware, which is not a Pi 5.

What does port from STAMPEDE Phase 3 is the part G4.5 deferred — register/row
blocking over the token dimension. That is where Camelid's wins actually came from:
Q4_K unpack-hoist alone +29%, but the 8-row repack reached **2.82× vs off**. Two
concrete, cheap starting points found during this phase:

- `src/inference.rs:4951` and `:5100`: the Q8_0 prefill chunk kernels do
  `let mut token_sums = vec![[0.0_f32; 4]; batch_size];` **unconditionally, per 4-row
  chunk** — `rows/4` heap allocations per matmul, inside a rayon closure. The Q4_0
  twins (`:1596-1607`, `:1768-1779`) already branch on `MAX_STACK_BATCH_SUMS` and use
  a stack array. Copying the Q4_0 shape is free and bit-exact.
- Register-blocking the token dimension, per STAMPEDE's repack-N.

### Confirm scope — **G5 (flash prefill)**

The precedent is real, and richer than the document says, but three corrections:

- **M1 is in `Camelid/SIROCCO_PHASE_P.md`, not `SIROCCO_COMPUTE.md`** (which covers
  only M-C0/M-C3/M-C4). Reading the wrong file gets you the unroll and none of the
  flash design or the parity gate.
- **The 11.5× is contaminated** — past ~11k the baseline also hits a shared-memory
  scores ceiling and degrades super-quadratically. The clean bandwidth-reuse number
  is **3.08× at ctx 8802**. The "~8× KV traffic cut" is an inference, not a
  measurement. Camelid's own M0a guidance: divide your paper traffic ratio by ~5
  before promising anything.
- **Four measured rejections should be read before starting**: GQA-axis reuse
  (0.69–0.92×), larger `Bq` (K32 = 0.92–0.95×), full-block AV (0.81–0.89× — the AV
  phase was *reuse*-sensitive, not occupancy-bound), tensor cores (dead). And M1
  ships **opt-in forever** because byte-identity was unreachable.

G5 remains the largest effort with the least certain gate. This phase has no new
evidence that changes that, so **D-S4 stands as a genuine open question for Tim.**

### Promote — **G6.1 (huge pages) is partly already built**

`NANOCAMELID_Q8_PAGE_ALIGN_1X4` and `NANOCAMELID_Q4_PAGE_ALIGN_1X4`
(`src/model.rs:1359`, `:1413`) already allocate the swizzled head into a page-aligned
arena, and both default **false**. Also note `af80471` itself — the commit SCARP
baselines against — is the commit that added the `mmap.advise(HugePage)` call G6.1
critiques. G6.1 should start by A/B-ing the two flags that exist before writing
anything new.

### Demote — **G6.5 (governor)**

The conductor document treats `ondemand` as a variance source worth sudo access to
remove. The `perf/fast-tied-lm-head` campaign already measured the governor as **not
a lever on these boards** — `ondemand` holds 2.4 GHz under sustained load. Keeping
both sides of every A/B on the same governor (R5) is still right; chasing
`performance` is not urgent, and per R5 switching would force a G0 re-baseline.

### Kill — **G6.4 (NEON `rms_norm` / `apply_qk_norm`)**

`decode.attn_norm` + `decode.ffn_norm` + `decode.head_norm` together are under 0.3%
of decode token time on every row measured here. Even a perfect kernel is invisible,
and it would cost token-parity. Write it down and stop re-litigating it.

---

## 9. Corrections the conductor document needs

| § | claim | correction |
|---|---|---|
| §2 | tied head runs `matmul_f32` over the f32 table | Dead by default since `perf/fast-tied-lm-head`. `src/model.rs:471-532` synthesizes a Q8_0-swizzled head; `q8_swizzle_1x4_enabled()` is `.unwrap_or(true)`. `matmul_f32` is reachable only via `NANOCAMELID_Q8_SWIZZLE_1X4=0` or a vocab/embd shape that fails divisibility. |
| §2 | head is "the single largest consumer of memory bandwidth" | Layer weights are, on every row (55–91% of bytes/token). The head is the largest single *tensor*. |
| §2 table | Llama-3.2-1B "native-quant equivalent 148 MB (Q4_0)" | That file's `token_embd` is **Q6_K, 215.5 MB** on disk, not Q4_0. The f32 sizes in the table (1.05 GB / 1.21 GB / 622 MB) are all exactly right. |
| §2 | `matmul_f32` is "a serial FMA dependency chain, latency-bound" | It is **row-parallel via rayon** (`src/inference.rs:3141-3143`); only the inner dot is scalar, deliberately, to preserve bit-identity. The "single-threaded" comment at `:3140` is past tense. |
| §2 | `dot_f32_neon` at `src/inference.rs:3354` | It is at **`:3374`**; `:3354` is the `dot_f32` dispatcher. It is genuinely unused by the head — but wiring it in would reassociate the reduction and break `matmul_f32`'s stated bit-identity contract, so it is not a free lever. |
| §2 | "no `trace_record` in `compute_logits_from_hidden`" | **Correct**, and now fixed. The printer was also truncating at 24 rows against 37 stages. |
| G0.1 | add `batch.head` | No such call site: `prefill_pass_batch` never computes logits. The head is decode-only. |
| G2.1 | `load_f32_or_f16` dequantizes "every on-disk type" | Q8_K and IQ4_NL hit the catch-all error (`src/model.rs:1226`), so a Q8_K/IQ4_NL `token_embd` fails to load outright. |
| G1.2 | `target-cpu=cortex-a76` presented as untried | `target-cpu=native` was already measured as no help on these boards during `perf/fast-tied-lm-head` (SDOT kernels are runtime-selected, so codegen-independent). Still worth running — but measure load time and prefill, not only decode. |
| G3 | ports "STAMPEDE P2.1" | STAMPEDE Phase 2 is `pending` — never executed. The nearest real prefetch receipt measured **−0.8%** and shipped default-off. |
| G3 | "prefetch hints are free and correct regardless" | Also already considered and declined once: `perf/fast-tied-lm-head` concluded no manual prefetch was needed, on the grounds that contiguous access already feeds the hardware prefetcher. |
| G4 | batch kernels re-unpack per token | False for every kernel a Pi 5 actually reaches; all are already block-outer with `unpack()` hoisted. Only the Q6_K **scalar** fallback isn't. |
| G5 | "SIROCCO M1 + M-C3", `Camelid/SIROCCO_COMPUTE.md` | M1 is in `Camelid/SIROCCO_PHASE_P.md`. Both docs are at `Documents/GitHub/Camelid`, not the T7 path in the prerequisite list. |
| G5 | "up to 11.5× on the attention stage" | Contaminated by a removed shared-memory ceiling. Clean reuse win is **3.08× at ctx 8802**. |
| G5.3 | "Camelid's M-C3 const-generic monomorphization" | M-C3 is a CUDA compile-time constant + `#pragma unroll`. Rust const generics are a fair port but not what Camelid did. |
| G6.1 | huge pages "not built" | Page-aligned arenas for the swizzled head already exist behind two default-off flags (`src/model.rs:1359`, `:1413`). |
| G6.5 | `performance` governor worth chasing | Already measured as not a lever: `ondemand` holds 2.4 GHz under sustained load on these boards. R5 (same governor both sides) still applies. |
| §Risk | "camelid1's power supply contaminates a receipt" | Held: camelid1 was online for this whole phase and was used only to read a GGUF header. No timing number in this receipt came from it. |

One meta-correction. The user memory row *"NanoCamelid tied LM head perf — tied LM
head was 81% of decode; parallel matmul + Q8_0-swizzled head = 3.7× on Pi 5;
`perf/fast-tied-lm-head`"* describes the **pre-fix** state. SCARP §2 reads as if that
campaign had not happened. It had, it is merged, and this phase re-measured its
result: the head went from 81% of decode to 19.8–37.4%.

---

## 10. Models

| row | file | sha256 | `token_embd` on disk | head streamed per token |
|---|---|---|---|---|
| llama32-1b-q4_0 | `Llama-3.2-1B-Instruct-Q4_0.gguf` | `eadfd8fd4e29d48e720eb87fc8242d3a8d4d2dacd52c722adc8e69e48c668efc` | Q6_K, 215.5 MB | tied → Q8_0-swizzled, 279.1 MB |
| llama32-1b-q8_0 | `Llama-3.2-1B-Instruct-Q8_0.gguf` | `432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3` | Q8_0, 279.1 MB | tied → Q8_0-swizzled, 279.1 MB |
| gemma3-1b-it-q4_0 | `gemma-3-1b-it-q4_0.gguf` | `27ee88e03be02e9ba73def9a819d570d8ad73716e50769e87f374ae394b0276e` | Q8_0, 320.9 MB | tied → Q8_0-swizzled, 320.9 MB |
| qwen3-0.6b-q8_0 | `qwen3-0.6b-q8_0.gguf` | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | Q8_0, 165.3 MB | tied → Q8_0-swizzled, 165.3 MB |
| llama3-8b-q4_k_m | `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` | `8ba9baf3a7345f705a11878397500fb25174034f0fd784e83aa4a96aaa47735f` | Q4_K, 295.5 MB | **untied** → `output.weight` Q6_K, 430.9 MB |

Shapes: Llama-3.2-1B vocab 128256 × 2048, 16 layers. Gemma-3-1B-IT vocab 262144 ×
1152, 26 layers. Qwen3-0.6B vocab 151936 × 1024, 28 layers. Llama-3-8B vocab
128256 × 4096, 32 layers.

---

## 11. G0.5 — the regression guard

`scripts/pi/scarp-guard.sh`. Runs the matrix above, one log per run, and writes
`summary.tsv` with tok/s, prefill/generation seconds, peak and decode RSS, the
`decode.forward_total` / `decode.layer_total` / `decode.head` / `decode.head_total`
totals, the `decode.head_quant` call count (the head-path witness), the computed
head percentage, and a **sha256 of the generated token stream**.

With `--baseline DIR` it diffs a new run against a recorded one and exits 5 on a
regression: token-stream change is a hard failure unless `--allow-text-drift` is
passed (for a phase whose bar is token-parity, and the receipt must then say so), and
tok/s is checked against `--tolerance` (default 3%).

```bash
NANOCAMELID_BIN=/mnt/nanocamelid/target-scarp/release/nanocamelid \
  ./scripts/pi/scarp-guard.sh --out /mnt/nanocamelid/scarp/g1 \
  --baseline /mnt/nanocamelid/scarp/g0-baseline-v2 --label "G1 build flags"
```

Rows are interleaved *within* each repeat rather than run to completion one at a
time, so thermal drift spreads across rows instead of loading onto whichever row runs
last. The script is covered by `scripts/validate.sh` (step 33a: dry-run, `--list`,
and four argument-validation failures).

The recorded G0 baseline lives on camelid2 at
`/mnt/nanocamelid/scarp/g0-baseline-v2` (`summary.tsv`, `host.txt`, 20 run logs and
20 RSS time series). Every later SCARP phase attaches a guard run against it.

### Known gaps this phase did not close

- **No STREAM-class memory benchmark on these boards.** "% of peak" in §6 is against
  the 17.07 GB/s theoretical figure. The marginal measurement in §7 (≈11 GB/s) is the
  better number but it is derived from one kernel. A `nanocamelid bench mem`
  subcommand would make every future bandwidth claim direct rather than inferred.
- **`performance` governor still unavailable** (sudo-gated `cpupower`). Everything
  here is `ondemand` on both sides, per R5. This is a smaller gap than the conductor
  document assumes — see the G6.5 demotion in §8 — but it does mean no number here
  has been cross-checked under `performance`.
- **Context-pack sweep not run.** G0 has no parity bar, so 512/1024/2048/4096/8192
  was out of scope; G1 is the first phase that needs it.
- **`scripts/validate.sh` cannot run on the Pis** — `check_public_hygiene()` shells
  out to `rg`, which is not installed on any of the three boards, and the missing-
  command 127 is re-raised as a gate failure. R6 was satisfied by running the full
  gate on the Mac instead (`CARGO_TARGET_DIR` on an external volume): **exit 0**,
  including the six new guard checks at step 33a. Installing ripgrep on the boards,
  or teaching the scan to fall back to `grep -REn`, would remove a real footgun —
  the same one that kept CI red on every run until PR #15.
