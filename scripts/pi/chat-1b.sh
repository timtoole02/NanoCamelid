#!/usr/bin/env bash
# Start NanoCamelid's terminal chat against the default Pi-local 1B Q8_0 model.
set -euo pipefail

WORKSPACE="${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}"
REPO="${NANOCAMELID_REPO:-$WORKSPACE/src/NanoCamelid}"
MODEL="${NANOCAMELID_MODEL_GGUF:-$WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf}"
TEMP="${1:-${NANOCAMELID_TEMP:-0.0}}"
MAX_TOKENS="${2:-${NANOCAMELID_MAX_TOKENS:-64}}"
BINARY="${NANOCAMELID_BIN:-$WORKSPACE/target/release/nanocamelid}"
export NANOCAMELID_Q8_DOT_KERNEL="${NANOCAMELID_Q8_DOT_KERNEL:-neon}"

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Set NANOCAMELID_MODEL_GGUF=/path/to/model.gguf or place the 1B Q8_0 GGUF at the default path." >&2
  exit 2
fi

if [[ -x "$BINARY" ]]; then
  exec "$BINARY" tui "$MODEL" "$TEMP" "$MAX_TOKENS"
fi

if command -v cargo >/dev/null 2>&1; then
  cd "$REPO"
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORKSPACE/target}"
  exec cargo run --release -- tui "$MODEL" "$TEMP" "$MAX_TOKENS"
fi

echo "NanoCamelid release binary not found and cargo is not on PATH." >&2
echo "Expected binary: $BINARY" >&2
exit 3
