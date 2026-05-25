#!/usr/bin/env bash
# Run the default Pi-local 1B readiness gate: inspect, smoke, then one chat turn.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ready-1b.sh [model.gguf] [chat_prompt] [chat_tokens]

Runs NanoCamelid's Pi-local Llama 3.2 1B readiness gate:
  1. inspect the selected GGUF
  2. run scalar-vs-selected smoke validation
  3. run one direct chat turn

Model resolution:
  1. explicit model.gguf argument
  2. NANOCAMELID_SMOKE_GGUF
  3. NANOCAMELID_MODEL_GGUF
  4. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf
  5. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf

Useful env:
  NANOCAMELID_WORKSPACE            Pi workspace, default /mnt/nanocamelid
  CARGO_TARGET_DIR                 Cargo output dir, default /mnt/nanocamelid/target
  NANOCAMELID_READY_SMOKE_KIND     Smoke kind, default chat
  NANOCAMELID_READY_SMOKE_PROMPT   Smoke prompt
  NANOCAMELID_READY_SMOKE_TOKENS   Smoke generated token count
  NANOCAMELID_READY_PROMPT         Direct chat prompt
  NANOCAMELID_READY_TOKENS         Direct chat generated token count
  NANOCAMELID_READY_TEMP           Direct chat temperature, default 0.0
  NANOCAMELID_READY_CHAT=0         Stop after inspect and smoke
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

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
if [[ "${1:-}" == *.gguf ]]; then
  MODEL="$1"
  shift
fi
SMOKE_KIND="${NANOCAMELID_READY_SMOKE_KIND:-${NANOCAMELID_SMOKE_KIND:-chat}}"
SMOKE_PROMPT="${NANOCAMELID_READY_SMOKE_PROMPT:-${NANOCAMELID_SMOKE_PROMPT:-Say hello in one sentence.}}"
SMOKE_TOKENS="${NANOCAMELID_READY_SMOKE_TOKENS:-${NANOCAMELID_SMOKE_TOKENS:-8}}"
CHAT_PROMPT="${1:-${NANOCAMELID_READY_PROMPT:-$SMOKE_PROMPT}}"
CHAT_TOKENS="${2:-${NANOCAMELID_READY_TOKENS:-$SMOKE_TOKENS}}"
CHAT_TEMP="${NANOCAMELID_READY_TEMP:-0.0}"
CHAT_ENABLED="${NANOCAMELID_READY_CHAT:-1}"
CHAT_ENABLED_LOWER="$(printf '%s' "$CHAT_ENABLED" | tr '[:upper:]' '[:lower:]')"
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

case "$CHAT_ENABLED_LOWER" in
0 | false | no)
  echo "==> Skipping direct 1B chat turn; NANOCAMELID_READY_CHAT=$CHAT_ENABLED"
  exit 0
  ;;
esac

echo "==> Running direct 1B chat turn"
run_nanocamelid chat "$MODEL" "$CHAT_PROMPT" "$CHAT_TEMP" "$CHAT_TOKENS"
