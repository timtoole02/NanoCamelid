#!/usr/bin/env bash
# Run the Pi-local Llama 3.2 1B evidence bundle.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: evidence-1b.sh [model.gguf] [--dry-run]

Runs the bounded Pi-local evidence bundle for Llama 3.2 1B:
  1. strict 1B model shape audit
  2. readiness gate without the final direct chat turn
  3. context-pack smoke gate
  4. prefill batch sweep

Model resolution is delegated to the existing 1B launchers:
  1. explicit model.gguf argument
  2. NANOCAMELID_SMOKE_GGUF
  3. NANOCAMELID_MODEL_GGUF
  4. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf
  5. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf

Useful env:
  NANOCAMELID_WORKSPACE          Pi workspace, default /mnt/nanocamelid
  CARGO_TARGET_DIR               Cargo output dir, default /mnt/nanocamelid/target
  NANOCAMELID_SMOKE_KIND         Smoke kind for readiness/context pack, default chat
  NANOCAMELID_SMOKE_PROMPT       Smoke prompt
  NANOCAMELID_SMOKE_TOKENS       Smoke generated token count
  NANOCAMELID_CONTEXT_PACKS      Context caps for context-pack-1b.sh
  NANOCAMELID_PREFILL_BATCHES    Batch list for bench-1b-prefill.sh
  --dry-run                      Print the resolved bundle plan without loading the model
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

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

looks_like_gguf_path() {
  case "${1:-}" in
    *.[gG][gG][uU][fF] | *.[gG][gG][uU][fF]/) return 0 ;;
    *) return 1 ;;
  esac
}

shell_quote() {
  printf '%q' "$1"
}

shell_command() {
  printf '%q' "$1"
  shift
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
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

json_integer_array() {
  local first=1
  local value

  printf '['
  for value in "$@"; do
    if [[ "$first" == "0" ]]; then
      printf ','
    fi
    first=0
    printf '%s' "$value"
  done
  printf ']'
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL_ARGS=()
if [[ $# -gt 1 ]]; then
  echo "Unexpected extra evidence argument: ${2}" >&2
  usage >&2
  exit 2
fi
if looks_like_non_gguf_model_path "${1:-}"; then
  echo "1B evidence model argument must be a .gguf path: $1" >&2
  exit 2
fi
if looks_like_gguf_path "${1:-}"; then
  MODEL_ARGS=("$1")
fi

WORKSPACE="${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}"
REPO="${NANOCAMELID_REPO:-$REPO_ROOT}"
TARGET_DIR="${CARGO_TARGET_DIR:-${NANOCAMELID_TARGET_DIR:-/mnt/nanocamelid/target}}"
SMOKE_KIND="${NANOCAMELID_SMOKE_KIND:-chat}"
SMOKE_PROMPT="${NANOCAMELID_SMOKE_PROMPT:-Say hello in one sentence.}"
SMOKE_TOKENS="${NANOCAMELID_SMOKE_TOKENS:-8}"
CONTEXT_PACKS_RAW="${NANOCAMELID_CONTEXT_PACKS:-512,1024,2048,4096,8192}"
PREFILL_BATCHES_RAW="${NANOCAMELID_PREFILL_BATCHES:-1,16,32,64}"
Q4_MODEL="$WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf"
Q8_MODEL="$WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf"

if [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
  MODEL="${MODEL_ARGS[0]}"
  MODEL_SOURCE="explicit argument"
elif [[ -n "${NANOCAMELID_SMOKE_GGUF:-}" ]]; then
  require_gguf_model_path "NANOCAMELID_SMOKE_GGUF" "$NANOCAMELID_SMOKE_GGUF"
  MODEL="$NANOCAMELID_SMOKE_GGUF"
  MODEL_SOURCE="NANOCAMELID_SMOKE_GGUF"
elif [[ -n "${NANOCAMELID_MODEL_GGUF:-}" ]]; then
  require_gguf_model_path "NANOCAMELID_MODEL_GGUF" "$NANOCAMELID_MODEL_GGUF"
  MODEL="$NANOCAMELID_MODEL_GGUF"
  MODEL_SOURCE="NANOCAMELID_MODEL_GGUF"
elif [[ -f "$Q4_MODEL" ]]; then
  MODEL="$Q4_MODEL"
  MODEL_SOURCE="workspace Q4_0 default"
else
  MODEL="$Q8_MODEL"
  MODEL_SOURCE="workspace Q8_0 fallback"
fi

case "$SMOKE_KIND" in
  chat | model | q8-chat | q8-model) ;;
  *)
    echo "Unknown smoke kind: $SMOKE_KIND" >&2
    echo "Expected chat, model, q8-chat, or q8-model." >&2
    exit 2
    ;;
esac
require_positive_integer() {
  local label="$1"
  local value="$2"

  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$label must be a positive integer: $value" >&2
    exit 2
  fi
}
require_positive_integer "Smoke token count" "$SMOKE_TOKENS"
CONTEXT_PACKS=($(parse_unique_positive_integer_list "context cap" "$CONTEXT_PACKS_RAW"))
PREFILL_BATCHES=($(parse_unique_positive_integer_list "prefill batch size" "$PREFILL_BATCHES_RAW"))

evidence_1b_status_json() {
  printf '{"target":"llama32-1b","status":"ok","model":%s,"selected_source":%s,"quantization":%s,"shape":"llama32_1b","shape_ready":true,"ready_no_chat":true,"context_pack":true,"prefill_bench":true,"smoke_kind":"%s","smoke_tokens":%s,"context_pack_caps":%s,"prefill_batches":%s}\n' \
    "$(json_string "$MODEL")" \
    "$(json_string "$MODEL_SOURCE")" \
    "$(json_string "$(llama32_1b_quantization_for_path "$MODEL")")" \
    "$SMOKE_KIND" \
    "$SMOKE_TOKENS" \
    "$(json_integer_array "${CONTEXT_PACKS[@]}")" \
    "$(json_integer_array "${PREFILL_BATCHES[@]}")"
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "NanoCamelid Llama 3.2 1B evidence bundle dry run"
  echo "repo: $REPO"
  echo "workspace: $WORKSPACE"
  echo "cargo_target_dir: $TARGET_DIR"
  echo "selected_source: $MODEL_SOURCE"
  echo "model: $MODEL"
  echo "model_exists: $([[ -f "$MODEL" ]] && echo true || echo false)"
  echo "quantization: $(llama32_1b_quantization_for_path "$MODEL")"
  echo "smoke_kind: $SMOKE_KIND"
  echo "smoke_prompt: $SMOKE_PROMPT"
  echo "smoke_tokens: $SMOKE_TOKENS"
  echo "context_pack_caps: ${CONTEXT_PACKS[*]}"
  echo "prefill_batches: ${PREFILL_BATCHES[*]}"
  echo "status_on_success: evidence_1b_status: ok"
  echo "json_on_success: $(evidence_1b_status_json)"
  printf 'model_command: '
  if [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
    shell_command ./scripts/pi/model-1b.sh "${MODEL_ARGS[@]}"
  else
    shell_command ./scripts/pi/model-1b.sh
  fi
  printf 'ready_command: '
  if [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
    shell_command ./scripts/pi/ready-1b.sh "${MODEL_ARGS[@]}" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS" --no-chat
  else
    shell_command ./scripts/pi/ready-1b.sh "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS" --no-chat
  fi
  printf 'context_pack_command: '
  if [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
    shell_command ./scripts/pi/context-pack-1b.sh "${MODEL_ARGS[@]}" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS"
  else
    shell_command ./scripts/pi/context-pack-1b.sh "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS"
  fi
  printf 'prefill_bench_command: '
  if [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
    shell_command ./scripts/pi/bench-1b-prefill.sh "${MODEL_ARGS[@]}"
  else
    shell_command ./scripts/pi/bench-1b-prefill.sh
  fi
  exit 0
fi

echo "NanoCamelid Llama 3.2 1B evidence bundle"
echo "workspace: $WORKSPACE"
echo "cargo_target_dir: $TARGET_DIR"
echo "selected_source: $MODEL_SOURCE"
echo "model: $MODEL"
echo "quantization: $(llama32_1b_quantization_for_path "$MODEL")"
echo "smoke_kind: $SMOKE_KIND"
echo "smoke_prompt: $SMOKE_PROMPT"
echo "smoke_tokens: $SMOKE_TOKENS"

cd "$REPO"

echo "==> Auditing selected 1B model"
if [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
  "$SCRIPT_DIR/model-1b.sh" "${MODEL_ARGS[@]}"
else
  "$SCRIPT_DIR/model-1b.sh"
fi

echo
echo "==> Running readiness gate without final direct chat"
if [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
  "$SCRIPT_DIR/ready-1b.sh" "${MODEL_ARGS[@]}" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS" --no-chat
else
  "$SCRIPT_DIR/ready-1b.sh" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS" --no-chat
fi

echo
echo "==> Running context-pack smoke gate"
if [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
  "$SCRIPT_DIR/context-pack-1b.sh" "${MODEL_ARGS[@]}" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS"
else
  "$SCRIPT_DIR/context-pack-1b.sh" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS"
fi

echo
echo "==> Running prefill batch sweep"
if [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
  "$SCRIPT_DIR/bench-1b-prefill.sh" "${MODEL_ARGS[@]}"
else
  "$SCRIPT_DIR/bench-1b-prefill.sh"
fi

echo "evidence_1b_status: ok"
echo "json: $(evidence_1b_status_json)"
