#!/usr/bin/env bash
# Run NanoCamelid's default Pi-local 1B smoke validation.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: smoke-1b.sh [model.gguf] [chat|model|q8-chat|q8-model] [prompt] [max_tokens]

Runs NanoCamelid's Pi-local Llama 3.2 1B smoke validation.

Model resolution:
  1. explicit model.gguf argument
  2. NANOCAMELID_SMOKE_GGUF
  3. NANOCAMELID_MODEL_GGUF
  4. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf
  5. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf

Useful env:
  NANOCAMELID_WORKSPACE     Pi workspace, default /mnt/nanocamelid
  CARGO_TARGET_DIR          Cargo output dir, default /mnt/nanocamelid/target
  NANOCAMELID_SMOKE_KIND    Default smoke kind, default chat
  NANOCAMELID_SMOKE_PROMPT  Default prompt
  NANOCAMELID_SMOKE_TOKENS  Default generated token count
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

looks_like_gguf_path() {
  case "${1:-}" in
    *.[gG][gG][uU][fF] | *.[gG][gG][uU][fF]/) return 0 ;;
    *) return 1 ;;
  esac
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}"
REPO="${NANOCAMELID_REPO:-$REPO_ROOT}"
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
if looks_like_gguf_path "${1:-}"; then
  MODEL="$1"
  shift
fi
SMOKE_KIND="${1:-${NANOCAMELID_SMOKE_KIND:-chat}}"
SMOKE_PROMPT="${2:-${NANOCAMELID_SMOKE_PROMPT:-Say hello in one sentence.}}"
SMOKE_TOKENS="${3:-${NANOCAMELID_SMOKE_TOKENS:-8}}"
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
  echo "Set NANOCAMELID_SMOKE_GGUF=/path/to/model.gguf, set NANOCAMELID_MODEL_GGUF=/path/to/model.gguf, or place the 1B Q4_0 or Q8_0 GGUF at the default path." >&2
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

echo "Running $SMOKE_KIND against $MODEL"
run_nanocamelid smoke 1b "$MODEL" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS"
