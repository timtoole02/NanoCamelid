# Camelid Porting Map

NanoCamelid should treat Camelid as the architecture reference, not as a code dump. The useful work is to port the proven boundaries and tests into a smaller Raspberry Pi runtime, then replace host-specific kernels with ARM64 NEON/DotProd implementations.

## Reference Areas

Use these Camelid areas as the first reference points:

- `src/gguf/reader.rs`: full GGUF metadata and tensor descriptor parsing.
- `src/model.rs`: typed LLaMA metadata extraction, tensor binding, GQA defaults, RoPE settings, tied output projection handling, and shape guardrails.
- `src/tensor/mod.rs`: Q8_0 block constants, f16 scale decoding, file-backed Q8 storage, runtime-owned packed storage, and row/shape validation.
- `src/inference/q8_runtime.rs`: environment-gated runtime plan pattern with default-off experimental routes.
- `src/inference/q8_block_reader.rs`: minimal Q8_0 block reader shape for streaming blocks without materializing full f32 tensors.
- `src/inference/rope.rs`: RoPE behavior, including metadata-derived frequency handling.
- `src/tokenizer/mod.rs` and `tests/tokenizer.rs`: LLaMA tokenizer and chat-template behavior needed before real chat can be honest.
- `scripts/*parity*` and `tests/*`: the parity-first validation model that should be kept for Nano instead of relying on benchmark-only confidence.

## Porting Order

1. **Upgrade GGUF descriptors**
   - Keep the existing `inspect` command, but extend the parser toward Camelid-style tensor descriptors with names, dimensions, tensor type, and absolute offsets.
   - Preserve Q8_0 layout constants: 34 bytes per block, 32 weights per block, 2-byte f16 scale followed by 32 i8 weights.

2. **Add typed model binding**
   - Port the LLaMA metadata boundary before adding generation.
   - Resolve attention heads, KV heads, embedding width, block count, FFN width, RMSNorm epsilon, RoPE base, and optional `rope_freqs.weight`.
   - Preserve the tied-output and output-projection shape lessons from Camelid; output rows must match the runtime consumer layout, not just the descriptor shape.

3. **Build a Pi-specific Q8 runtime storage boundary**
   - Keep scalar Q8 as the reference path.
   - Keep the current NEON and SDOT micro-kernels as benchmarked candidates.
   - Add runtime-owned packed storage only behind explicit Nano flags until parity and model smoke pass.
   - Avoid duplicate row-major plus packed storage as the final performance design; use duplicate storage only for debug/parity staging.

4. **Factor kernel dispatch before wiring inference**
   - Use a small selector similar to Camelid's runtime-plan pattern.
   - Default to scalar/reference unless an explicit flag and runtime feature detection both pass.
   - Require shape-specific parity tests for scalar, NEON, and SDOT before using a kernel in a model path.

5. **Port the minimal dense forward path**
   - Token embedding lookup.
   - RMSNorm.
   - Q/K/V projection.
   - RoPE.
   - Attention output.
   - Gate/up/down FFN.
   - Final norm and output projection.
   - Keep one-token first, then short deterministic prompt packs.

6. **Bring over validation discipline**
   - Every retained kernel needs checksum parity against scalar and a Pi benchmark.
   - Every model path needs prompt/token or generated-token parity against a known-good reference where possible.
   - Public docs should claim only exact evidence-backed rows.

## Current Nano Baseline

NanoCamelid currently has:

- host feature probe
- GGUF metadata/tensor-type inspection
- Q8 i8 dot scalar benchmark
- ARM64 NEON benchmark path
- default-off ARM64 SDOT benchmark path
- repeated-run JSON benchmark output
- reusable library modules for GGUF/Q8 code
- Q8_0 block constants, 34-byte block decoding, f16 scale expansion, and scalar
  scaled block-dot reference tests

The next durable slice should wire the Q8_0 block boundary into GGUF tensor
descriptor reads, then add shape-specific scalar/NEON/SDOT parity tests before
using an optimized kernel in a model path.
