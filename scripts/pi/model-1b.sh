#!/usr/bin/env bash
# Audit NanoCamelid's Pi-local Llama 3.2 1B model resolution.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: model-1b.sh [model.gguf] [--q4|--q8] [--dry-run]

Prints the Pi-local Llama 3.2 1B model selection plan and verifies that the
selected GGUF exists, then runs the strict Llama 3.2 1B shape audit unless
--dry-run is used.

Model resolution:
  1. explicit model.gguf argument
  2. NANOCAMELID_SMOKE_GGUF
  3. NANOCAMELID_MODEL_GGUF
  4. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf
  5. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf

Useful env:
  NANOCAMELID_WORKSPACE     Pi workspace, default /mnt/nanocamelid
  CARGO_TARGET_DIR          Cargo output dir, default /mnt/nanocamelid/target
  --q4, --q8                Select the Pi-local Q4_0 or Q8_0 default row
  --dry-run                 Print the model audit without failing when missing
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

require_gguf_model_path() {
  local label="$1"
  local path="$2"

  if ! looks_like_gguf_path "$path"; then
    echo "$label must be a .gguf path: $path" >&2
    exit 2
  fi
}

DRY_RUN=0
QUANT_MODEL=""
POSITIONAL_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    --q4)
      if [[ -n "$QUANT_MODEL" ]]; then
        echo "Only one 1B quantization selector may be provided." >&2
        exit 2
      fi
      QUANT_MODEL="q4"
      ;;
    --q8)
      if [[ -n "$QUANT_MODEL" ]]; then
        echo "Only one 1B quantization selector may be provided." >&2
        exit 2
      fi
      QUANT_MODEL="q8"
      ;;
    -*)
      echo "Unknown model audit option: $arg" >&2
      usage >&2
      exit 2
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

if [[ $# -gt 1 ]]; then
  echo "Unexpected extra model audit argument: ${2}" >&2
  usage >&2
  exit 2
fi
if [[ $# -eq 1 ]] && ! looks_like_gguf_path "$1"; then
  echo "Model audit argument must be a .gguf path: $1" >&2
  usage >&2
  exit 2
fi

WORKSPACE="${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}"
REPO="${NANOCAMELID_REPO:-$REPO_ROOT}"
TARGET_DIR="${CARGO_TARGET_DIR:-${NANOCAMELID_TARGET_DIR:-/mnt/nanocamelid/target}}"
Q4_MODEL="$WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf"
Q8_MODEL="$WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
MODEL_SOURCE=""

if looks_like_gguf_path "${1:-}"; then
  MODEL="$1"
  MODEL_SOURCE="explicit argument"
elif [[ "$QUANT_MODEL" == "q4" ]]; then
  MODEL="$Q4_MODEL"
  MODEL_SOURCE="workspace Q4_0 requested"
elif [[ "$QUANT_MODEL" == "q8" ]]; then
  MODEL="$Q8_MODEL"
  MODEL_SOURCE="workspace Q8_0 requested"
elif [[ -n "${NANOCAMELID_SMOKE_GGUF:-}" ]]; then
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

case "$MODEL_SOURCE" in
  NANOCAMELID_SMOKE_GGUF | NANOCAMELID_MODEL_GGUF)
    require_gguf_model_path "$MODEL_SOURCE" "$MODEL"
    ;;
esac

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

model_status_json() {
  printf '{"target":"llama32-1b","status":"ok","model":%s,"selected_source":%s,"quantization":%s,"shape":"llama32_1b","shape_ready":true}\n' \
    "$(json_string "$MODEL")" \
    "$(json_string "$MODEL_SOURCE")" \
    "$(json_string "$(llama32_1b_quantization_for_path "$MODEL")")"
}

BINARY="${NANOCAMELID_BIN:-$TARGET_DIR/release/nanocamelid}"
if [[ -x "$BINARY" ]]; then
  launcher_mode="binary"
elif command -v cargo >/dev/null 2>&1; then
  launcher_mode="cargo"
else
  launcher_mode="unavailable"
fi

echo "NanoCamelid Llama 3.2 1B model audit"
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
echo "selected_model: $MODEL"
echo "selected_exists: $([[ -f "$MODEL" ]] && echo true || echo false)"
echo "quantization: $(llama32_1b_quantization_for_path "$MODEL")"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "shape_audit: enabled"
  echo "status_on_success: model_1b_status: ok"
  echo "json_on_success: $(model_status_json)"
  printf 'model_command: '
  shell_command nanocamelid model 1b "$MODEL"
  printf 'inspect_command: '
  shell_command nanocamelid inspect "$MODEL"
  printf 'smoke_command: '
  shell_command nanocamelid smoke 1b "$MODEL" chat "Say hello in one sentence." 8
  printf 'ready_command: '
  shell_command nanocamelid ready 1b "$MODEL"
  printf 'evidence_command: '
  shell_command nanocamelid evidence 1b "$MODEL"
  exit 0
fi

if [[ -f "$MODEL" ]]; then
  if [[ "$launcher_mode" == "unavailable" ]]; then
    echo "NanoCamelid release binary not found and cargo is not on PATH." >&2
    echo "Expected binary: $BINARY" >&2
    exit 3
  fi
  if [[ "$launcher_mode" == "cargo" || -z "${NANOCAMELID_BIN:-}" ]]; then
    require_safe_cargo_target_dir "$TARGET_DIR" "$REPO"
  fi

  if [[ "$launcher_mode" == "binary" ]]; then
    "$BINARY" model 1b "$MODEL"
  else
    cd "$REPO"
    export CARGO_TARGET_DIR="$TARGET_DIR"
    cargo run --release -- model 1b "$MODEL"
  fi
  exit $?
fi

echo "Selected 1B model is missing: $MODEL" >&2
echo "Set NANOCAMELID_SMOKE_GGUF, set NANOCAMELID_MODEL_GGUF, pass an explicit .gguf path, or place the 1B Q4_0 or Q8_0 GGUF under $WORKSPACE/models." >&2
exit 2
