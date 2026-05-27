#!/usr/bin/env bash
# Run the default Pi-local 1B readiness gate: inspect, smoke, then one chat turn.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ready-1b.sh [model.gguf] [chat|model|q8-chat|q8-model] [prompt] [max_tokens] [--no-chat|--smoke-only|--chat|--dry-run]

Runs NanoCamelid's Pi-local Llama 3.2 1B readiness gate:
  1. audit the selected GGUF's Llama 3.2 1B shape
  2. inspect the selected GGUF
  3. run scalar-vs-selected smoke validation
  4. run one direct chat turn

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
  NANOCAMELID_READY_CHAT=0         Stop after audit, inspect, and smoke
  --no-chat, --smoke-only          Stop after audit, inspect, and smoke; positionals override the smoke prompt
  --chat                           Force the direct chat turn even when NANOCAMELID_READY_CHAT=0
  --dry-run                        Print the resolved readiness plan without loading the model
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

is_non_negative_float() {
  [[ "${1:-}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]
}

require_non_negative_float() {
  local label="$1"
  local value="$2"

  if ! is_non_negative_float "$value"; then
    echo "$label must be a non-negative number: $value" >&2
    exit 2
  fi
}

CHAT_ENABLED_OVERRIDE=""
DRY_RUN=0
POSITIONAL_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --no-chat | --smoke-only)
      CHAT_ENABLED_OVERRIDE="0"
      ;;
    --chat)
      CHAT_ENABLED_OVERRIDE="1"
      ;;
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
source "$SCRIPT_DIR/common.sh"
require_optional_context_limit
require_optional_prefill_batch
WORKSPACE="${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}"
REPO="${NANOCAMELID_REPO:-$REPO_ROOT}"
TARGET_DIR="${CARGO_TARGET_DIR:-${NANOCAMELID_TARGET_DIR:-/mnt/nanocamelid/target}}"
Q4_MODEL="$WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf"
Q8_MODEL="$WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
if [[ -n "${NANOCAMELID_SMOKE_GGUF:-}" ]]; then
  MODEL="$NANOCAMELID_SMOKE_GGUF"
  MODEL_SOURCE="NANOCAMELID_SMOKE_GGUF"
elif [[ -n "${NANOCAMELID_MODEL_GGUF:-}" ]]; then
  MODEL="$NANOCAMELID_MODEL_GGUF"
  MODEL_SOURCE="NANOCAMELID_MODEL_GGUF"
elif [[ -f "$Q4_MODEL" ]]; then
  MODEL="$Q4_MODEL"
  MODEL_SOURCE="workspace Q4_0 default"
else
  MODEL="$Q8_MODEL"
  MODEL_SOURCE="workspace Q8_0 fallback"
fi
if looks_like_gguf_path "${1:-}"; then
  MODEL="$1"
  MODEL_SOURCE="explicit argument"
  shift
fi
case "$MODEL_SOURCE" in
  NANOCAMELID_SMOKE_GGUF | NANOCAMELID_MODEL_GGUF)
    require_gguf_model_path "$MODEL_SOURCE" "$MODEL"
    ;;
esac
SMOKE_KIND="${NANOCAMELID_READY_SMOKE_KIND:-${NANOCAMELID_SMOKE_KIND:-chat}}"
case "${1:-}" in
chat | model | q8-chat | q8-model)
  SMOKE_KIND="$1"
  shift
  ;;
q8-*)
  echo "Unknown smoke kind: $1" >&2
  echo "Expected model, chat, q8-model, or q8-chat." >&2
  exit 2
  ;;
esac
if [[ $# -gt 2 ]]; then
  echo "Unexpected extra readiness argument: ${3}" >&2
  usage >&2
  exit 2
fi
SMOKE_PROMPT="${NANOCAMELID_READY_SMOKE_PROMPT:-${NANOCAMELID_SMOKE_PROMPT:-Say hello in one sentence.}}"
SMOKE_TOKENS="${NANOCAMELID_READY_SMOKE_TOKENS:-${NANOCAMELID_SMOKE_TOKENS:-8}}"
CHAT_TEMP="${NANOCAMELID_READY_TEMP:-0.0}"
CHAT_ENABLED="${CHAT_ENABLED_OVERRIDE:-${NANOCAMELID_READY_CHAT:-1}}"
CHAT_ENABLED_LOWER="$(printf '%s' "$CHAT_ENABLED" | tr '[:upper:]' '[:lower:]')"
case "$CHAT_ENABLED_LOWER" in
  0 | false | no)
    SMOKE_PROMPT="${1:-$SMOKE_PROMPT}"
    SMOKE_TOKENS="${2:-$SMOKE_TOKENS}"
    CHAT_PROMPT="$SMOKE_PROMPT"
    CHAT_TOKENS="$SMOKE_TOKENS"
    ;;
  *)
    CHAT_PROMPT="${1:-${NANOCAMELID_READY_PROMPT:-$SMOKE_PROMPT}}"
    CHAT_TOKENS="${2:-${NANOCAMELID_READY_TOKENS:-$SMOKE_TOKENS}}"
    ;;
esac
require_positive_integer "Smoke token count" "$SMOKE_TOKENS"
case "$CHAT_ENABLED_LOWER" in
  0 | false | no)
    ;;
  *)
    require_positive_integer "Direct chat token count" "$CHAT_TOKENS"
    require_non_negative_float "Direct chat temperature" "$CHAT_TEMP"
    ;;
esac
BINARY="${NANOCAMELID_BIN:-$TARGET_DIR/release/nanocamelid}"
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

shell_command() {
  printf '%q' "$1"
  shift
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

context_env_prefix() {
  if [[ -n "${NANOCAMELID_CONTEXT_LIMIT:-}" ]]; then
    printf 'NANOCAMELID_CONTEXT_LIMIT=%q ' "$NANOCAMELID_CONTEXT_LIMIT"
  fi
}

json_string() {
  local value="$1"
  local out='"'
  local i ch

  for ((i = 0; i < ${#value}; i++)); do
    ch="${value:i:1}"
    case "$ch" in
      '"') out+='\"' ;;
      "\\") out+='\\' ;;
      $'\n') out+='\n' ;;
      $'\r') out+='\r' ;;
      $'\t') out+='\t' ;;
      *) out+="$ch" ;;
    esac
  done

  out+='"'
  printf '%s' "$out"
}

context_limit_plan_value() {
  if [[ -n "${NANOCAMELID_CONTEXT_LIMIT:-}" ]]; then
    printf '%s' "$NANOCAMELID_CONTEXT_LIMIT"
  else
    printf 'unset'
  fi
}

ready_1b_status_json() {
  local direct_chat="$1"
  local chat_tokens="$2"
  local chat_tokens_json="null"

  if [[ -n "$chat_tokens" ]]; then
    chat_tokens_json="$chat_tokens"
  fi

  printf '{"target":"llama32-1b","status":"ok","model":%s,"selected_source":%s,"probe":true,"shape":"llama32_1b","shape_ready":true,"context_limit":%s,"smoke_kind":"%s","smoke_tokens":%s,"direct_chat":%s,"chat_tokens":%s}\n' \
    "$(json_string "$MODEL")" \
    "$(json_string "$MODEL_SOURCE")" \
    "$(json_string "$(context_limit_plan_value)")" \
    "$SMOKE_KIND" \
    "$SMOKE_TOKENS" \
    "$direct_chat" \
    "$chat_tokens_json"
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "NanoCamelid Llama 3.2 1B readiness launcher dry run"
  echo "repo: $REPO"
  echo "cargo_target_dir: $TARGET_DIR"
  echo "launcher_mode: $launcher_mode"
  echo "binary: $BINARY"
  echo "workspace: $WORKSPACE"
  echo "q4_model: $Q4_MODEL"
  echo "q4_exists: $([[ -f "$Q4_MODEL" ]] && echo true || echo false)"
  echo "q8_model: $Q8_MODEL"
  echo "q8_exists: $([[ -f "$Q8_MODEL" ]] && echo true || echo false)"
  echo "selected_source: $MODEL_SOURCE"
  echo "model: $MODEL"
  echo "model_exists: $([[ -f "$MODEL" ]] && echo true || echo false)"
  echo "context_limit: ${NANOCAMELID_CONTEXT_LIMIT:-unset}"
  echo "shape_audit: enabled"
  echo "smoke_kind: $SMOKE_KIND"
  echo "smoke_prompt: $SMOKE_PROMPT"
  echo "smoke_tokens: $SMOKE_TOKENS"
  echo "prefill_batch: $(prefill_batch_plan_value)"
  echo "direct_chat: $([[ "$CHAT_ENABLED_LOWER" == "0" || "$CHAT_ENABLED_LOWER" == "false" || "$CHAT_ENABLED_LOWER" == "no" ]] && echo disabled || echo enabled)"
  echo "status_on_success: ready_1b_status: ok"
  case "$CHAT_ENABLED_LOWER" in
    0 | false | no)
      echo "json_on_success: $(ready_1b_status_json false "")"
      ;;
    *)
      echo "json_on_success: $(ready_1b_status_json true "$CHAT_TOKENS")"
      ;;
  esac
  printf 'probe_command: '
  shell_command nanocamelid probe
  printf 'model_command: '
  shell_command nanocamelid model 1b "$MODEL"
  printf 'inspect_command: '
  shell_command nanocamelid inspect "$MODEL"
  printf 'smoke_command: '
  context_env_prefix
  shell_command nanocamelid smoke 1b "$MODEL" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS"
  case "$CHAT_ENABLED_LOWER" in
    0 | false | no)
      echo "chat_command: skipped"
      ;;
    *)
      echo "chat_prompt: $CHAT_PROMPT"
      echo "chat_temp: $CHAT_TEMP"
      echo "chat_tokens: $CHAT_TOKENS"
      printf 'chat_command: '
      context_env_prefix
      shell_command nanocamelid chat "$MODEL" "$CHAT_PROMPT" "$CHAT_TEMP" "$CHAT_TOKENS"
      ;;
  esac
  exit 0
fi

if [[ "$launcher_mode" == "unavailable" ]]; then
  echo "NanoCamelid release binary not found and cargo is not on PATH." >&2
  echo "Expected binary: $BINARY" >&2
  exit 3
fi
if [[ "$launcher_mode" == "cargo" || -z "${NANOCAMELID_BIN:-}" ]]; then
  require_safe_cargo_target_dir "$TARGET_DIR" "$REPO"
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

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Set NANOCAMELID_MODEL_GGUF=/path/to/model.gguf, set NANOCAMELID_SMOKE_GGUF=/path/to/model.gguf, or place the 1B Q4_0 or Q8_0 GGUF at the default path." >&2
  exit 2
fi

echo "==> Probing host fast-path support"
run_nanocamelid probe

echo "==> Auditing 1B model shape: $MODEL"
run_nanocamelid model 1b "$MODEL"

echo "==> Inspecting 1B model: $MODEL"
run_nanocamelid inspect "$MODEL"

echo "==> Running 1B $SMOKE_KIND smoke gate"
run_nanocamelid smoke 1b "$MODEL" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS"

case "$CHAT_ENABLED_LOWER" in
0 | false | no)
  echo "==> Skipping direct 1B chat turn; NANOCAMELID_READY_CHAT=$CHAT_ENABLED"
  echo "ready_1b_status: ok"
  echo "json: $(ready_1b_status_json false "")"
  exit 0
  ;;
esac

echo "==> Running direct 1B chat turn"
run_nanocamelid chat "$MODEL" "$CHAT_PROMPT" "$CHAT_TEMP" "$CHAT_TOKENS"
echo "ready_1b_status: ok"
echo "json: $(ready_1b_status_json true "$CHAT_TOKENS")"
