#!/usr/bin/env bash
# Start NanoCamelid's terminal chat against the default Pi-local 1B model.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: chat-1b.sh [model.gguf] [temp] [max_tokens] [--dry-run]

Starts NanoCamelid's Pi-local Llama 3.2 1B terminal chat. By default it runs a
short chat smoke gate before launching the TUI.

Model resolution:
  1. explicit model.gguf argument
  2. NANOCAMELID_MODEL_GGUF
  3. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf
  4. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf

Useful env:
  NANOCAMELID_WORKSPACE          Pi workspace, default /mnt/nanocamelid
  CARGO_TARGET_DIR               Cargo output dir, default /mnt/nanocamelid/target
  NANOCAMELID_CHAT_SMOKE=0       Skip the pre-chat smoke gate; false/no are also accepted
  NANOCAMELID_CHAT_SMOKE_KIND    Smoke kind: chat, model, q8-chat, or q8-model; default chat
  NANOCAMELID_TEMP               Chat temperature, default 0.0
  NANOCAMELID_MAX_TOKENS         Max tokens per assistant turn, default 64

Options:
  --dry-run                      Print the resolved smoke/TUI launch plan without loading the model
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

is_positive_integer() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

require_positive_integer() {
  local label="$1"
  local value="$2"

  if ! is_positive_integer "$value"; then
    echo "$label must be a positive integer: $value" >&2
    exit 2
  fi
}

shell_command() {
  printf '%q' "$1"
  shift
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

DRY_RUN=0
POSITIONAL_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      POSITIONAL_ARGS+=("$arg")
      ;;
  esac
done
if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
  set -- "${POSITIONAL_ARGS[@]}"
else
  set --
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}"
REPO="${NANOCAMELID_REPO:-$REPO_ROOT}"
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
if looks_like_gguf_path "${1:-}"; then
  MODEL="$1"
  shift
fi
TEMP="${1:-${NANOCAMELID_TEMP:-0.0}}"
MAX_TOKENS="${2:-${NANOCAMELID_MAX_TOKENS:-64}}"
BINARY="${NANOCAMELID_BIN:-$TARGET_DIR/release/nanocamelid}"
SMOKE_ENABLED="${NANOCAMELID_CHAT_SMOKE:-1}"
SMOKE_ENABLED_LOWER="$(printf '%s' "$SMOKE_ENABLED" | tr '[:upper:]' '[:lower:]')"
SMOKE_KIND="${NANOCAMELID_CHAT_SMOKE_KIND:-chat}"
SMOKE_PROMPT="${NANOCAMELID_CHAT_SMOKE_PROMPT:-Say hello in one sentence.}"
SMOKE_TOKENS="${NANOCAMELID_CHAT_SMOKE_TOKENS:-1}"
require_positive_integer "Max token count" "$MAX_TOKENS"
require_positive_integer "Smoke token count" "$SMOKE_TOKENS"
export NANOCAMELID_Q8_DOT_SDOT="${NANOCAMELID_Q8_DOT_SDOT:-1}"
export NANOCAMELID_Q8_DOT_KERNEL="${NANOCAMELID_Q8_DOT_KERNEL:-sdot}"

if [[ "$SMOKE_KIND" != "model" && "$SMOKE_KIND" != "chat" && "$SMOKE_KIND" != "q8-model" && "$SMOKE_KIND" != "q8-chat" ]]; then
  echo "Unknown smoke kind: $SMOKE_KIND" >&2
  echo "Expected model, chat, q8-model, or q8-chat." >&2
  exit 2
fi

if [[ -x "$BINARY" ]]; then
  launcher_mode="binary"
elif command -v cargo >/dev/null 2>&1; then
  launcher_mode="cargo"
else
  launcher_mode="unavailable"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "NanoCamelid Llama 3.2 1B chat launch dry run"
  echo "repo: $REPO"
  echo "cargo_target_dir: $TARGET_DIR"
  echo "launcher_mode: $launcher_mode"
  echo "binary: $BINARY"
  echo "model: $MODEL"
  echo "model_exists: $([[ -f "$MODEL" ]] && echo true || echo false)"
  echo "temp: $TEMP"
  echo "max_tokens: $MAX_TOKENS"
  echo "smoke_enabled: $SMOKE_ENABLED"
  echo "smoke_kind: $SMOKE_KIND"
  echo "smoke_prompt: $SMOKE_PROMPT"
  echo "smoke_tokens: $SMOKE_TOKENS"
  case "$SMOKE_ENABLED_LOWER" in
    0 | false | no)
      echo "smoke_command: skipped"
      ;;
    *)
      printf 'smoke_command: '
      shell_command nanocamelid smoke 1b "$MODEL" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS"
      ;;
  esac
  printf 'tui_command: '
  shell_command nanocamelid tui "$MODEL" "$TEMP" "$MAX_TOKENS"
  exit 0
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Set NANOCAMELID_MODEL_GGUF=/path/to/model.gguf or place the 1B Q4_0 or Q8_0 GGUF at the default path." >&2
  exit 2
fi

if [[ "$launcher_mode" == "unavailable" ]]; then
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

case "$SMOKE_ENABLED_LOWER" in
  0 | false | no) ;;
  *)
    echo "Running $SMOKE_KIND smoke gate before launching 1B chat..."
    run_nanocamelid smoke 1b "$MODEL" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS"
    ;;
esac

exec_nanocamelid tui "$MODEL" "$TEMP" "$MAX_TOKENS"
