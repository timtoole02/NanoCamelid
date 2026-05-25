#!/usr/bin/env bash
# Run the default Pi-local 1B readiness gate: inspect, smoke, then one chat turn.
set -euo pipefail

WORKSPACE="${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}"
REPO="${NANOCAMELID_REPO:-$WORKSPACE/src/NanoCamelid}"
TARGET_DIR="${CARGO_TARGET_DIR:-${NANOCAMELID_TARGET_DIR:-/mnt/nanocamelid/target}}"
Q4_MODEL="$WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf"
Q8_MODEL="$WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
if [[ -n "${NANOCAMELID_SMOKE_GGUF:-}" ]]; then
  MODEL="$NANOCAMELID_SMOKE_GGUF"
elif [[ -n "${NANOCAMELID_MODEL_GGUF:-}" ]]; then
  MODEL="$NANOCAMELID_MODEL_GGUF"
elif [[ -f "$Q4_MODEL" ]]; then
  MODEL="$Q4_MODEL"
else
  MODEL="$Q8_MODEL"
fi
SMOKE_KIND="${NANOCAMELID_READY_SMOKE_KIND:-${NANOCAMELID_SMOKE_KIND:-chat}}"
SMOKE_PROMPT="${NANOCAMELID_READY_SMOKE_PROMPT:-${NANOCAMELID_SMOKE_PROMPT:-Say hello in one sentence.}}"
SMOKE_TOKENS="${NANOCAMELID_READY_SMOKE_TOKENS:-${NANOCAMELID_SMOKE_TOKENS:-8}}"
CHAT_PROMPT="${1:-${NANOCAMELID_READY_PROMPT:-Say hello in one sentence.}}"
CHAT_TOKENS="${2:-${NANOCAMELID_READY_TOKENS:-8}}"
CHAT_TEMP="${NANOCAMELID_READY_TEMP:-0.0}"
BINARY="${NANOCAMELID_BIN:-$TARGET_DIR/release/nanocamelid}"
export NANOCAMELID_Q8_DOT_SDOT="${NANOCAMELID_Q8_DOT_SDOT:-1}"
export NANOCAMELID_Q8_DOT_KERNEL="${NANOCAMELID_Q8_DOT_KERNEL:-sdot}"

if [[ "$SMOKE_KIND" != "model" && "$SMOKE_KIND" != "chat" && "$SMOKE_KIND" != "q8-model" && "$SMOKE_KIND" != "q8-chat" ]]; then
  echo "Unknown smoke kind: $SMOKE_KIND" >&2
  echo "Expected model, chat, q8-model, or q8-chat." >&2
  exit 2
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Set NANOCAMELID_MODEL_GGUF=/path/to/model.gguf, set NANOCAMELID_SMOKE_GGUF=/path/to/model.gguf, or place the 1B Q4_0 or Q8_0 GGUF at the default path." >&2
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

echo "==> Inspecting 1B model: $MODEL"
run_nanocamelid inspect "$MODEL"

echo "==> Running 1B $SMOKE_KIND smoke gate"
run_nanocamelid smoke 1b "$MODEL" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS"

echo "==> Running direct 1B chat turn"
run_nanocamelid chat "$MODEL" "$CHAT_PROMPT" "$CHAT_TEMP" "$CHAT_TOKENS"
