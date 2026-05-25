#!/usr/bin/env bash
# Start NanoCamelid's terminal chat against the default Pi-local 1B model.
set -euo pipefail

WORKSPACE="${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}"
REPO="${NANOCAMELID_REPO:-$WORKSPACE/src/NanoCamelid}"
TARGET_DIR="${CARGO_TARGET_DIR:-${NANOCAMELID_TARGET_DIR:-/mnt/nanocamelid/target}}"
Q4_MODEL="$WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf"
Q8_MODEL="$WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
if [[ -n "${NANOCAMELID_MODEL_GGUF:-}" ]]; then
  MODEL="$NANOCAMELID_MODEL_GGUF"
elif [[ -f "$Q4_MODEL" ]]; then
  MODEL="$Q4_MODEL"
else
  MODEL="$Q8_MODEL"
fi
TEMP="${1:-${NANOCAMELID_TEMP:-0.0}}"
MAX_TOKENS="${2:-${NANOCAMELID_MAX_TOKENS:-64}}"
BINARY="${NANOCAMELID_BIN:-$TARGET_DIR/release/nanocamelid}"
SMOKE_ENABLED="${NANOCAMELID_CHAT_SMOKE:-1}"
SMOKE_KIND="${NANOCAMELID_CHAT_SMOKE_KIND:-chat}"
SMOKE_PROMPT="${NANOCAMELID_CHAT_SMOKE_PROMPT:-Say hello in one sentence.}"
SMOKE_TOKENS="${NANOCAMELID_CHAT_SMOKE_TOKENS:-1}"
export NANOCAMELID_Q8_DOT_SDOT="${NANOCAMELID_Q8_DOT_SDOT:-1}"
export NANOCAMELID_Q8_DOT_KERNEL="${NANOCAMELID_Q8_DOT_KERNEL:-sdot}"

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Set NANOCAMELID_MODEL_GGUF=/path/to/model.gguf or place the 1B Q4_0 or Q8_0 GGUF at the default path." >&2
  exit 2
fi

if [[ -x "$BINARY" ]]; then
  launcher_mode="binary"
elif command -v cargo >/dev/null 2>&1; then
  launcher_mode="cargo"
else
  echo "NanoCamelid release binary not found and cargo is not on PATH." >&2
  echo "Expected binary: $BINARY" >&2
  exit 3
fi

run_nanocamelid() {
  if [[ "$launcher_mode" == "binary" ]]; then
    "$BINARY" "$@"
    return
  fi

  cd "$REPO"
  export CARGO_TARGET_DIR="$TARGET_DIR"
  cargo run --release -- "$@"
}

exec_nanocamelid() {
  if [[ "$launcher_mode" == "binary" ]]; then
    exec "$BINARY" "$@"
  fi

  cd "$REPO"
  export CARGO_TARGET_DIR="$TARGET_DIR"
  exec cargo run --release -- "$@"
}

if [[ "$SMOKE_ENABLED" != "0" ]]; then
  echo "Running $SMOKE_KIND smoke gate before launching 1B chat..."
  run_nanocamelid smoke 1b "$MODEL" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS"
fi

exec_nanocamelid tui "$MODEL" "$TEMP" "$MAX_TOKENS"
