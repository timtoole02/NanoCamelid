#!/usr/bin/env bash
# Audit NanoCamelid's Pi-local Llama 3.2 1B model resolution.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: model-1b.sh [model.gguf] [--dry-run]

Prints the Pi-local Llama 3.2 1B model selection plan and verifies that the
selected GGUF exists unless --dry-run is used.

Model resolution:
  1. explicit model.gguf argument
  2. NANOCAMELID_SMOKE_GGUF
  3. NANOCAMELID_MODEL_GGUF
  4. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf
  5. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf

Useful env:
  NANOCAMELID_WORKSPACE     Pi workspace, default /mnt/nanocamelid
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
Q4_MODEL="$WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf"
Q8_MODEL="$WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
MODEL_SOURCE=""

if looks_like_gguf_path "${1:-}"; then
  MODEL="$1"
  MODEL_SOURCE="explicit argument"
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

echo "NanoCamelid Llama 3.2 1B model audit"
echo "workspace: $WORKSPACE"
echo "q4_model: $Q4_MODEL"
echo "q4_exists: $([[ -f "$Q4_MODEL" ]] && echo true || echo false)"
echo "q8_model: $Q8_MODEL"
echo "q8_exists: $([[ -f "$Q8_MODEL" ]] && echo true || echo false)"
echo "selected_source: $MODEL_SOURCE"
echo "selected_model: $MODEL"
echo "selected_exists: $([[ -f "$MODEL" ]] && echo true || echo false)"

if [[ -f "$MODEL" || "$DRY_RUN" == "1" ]]; then
  exit 0
fi

echo "Selected 1B model is missing: $MODEL" >&2
echo "Set NANOCAMELID_SMOKE_GGUF, set NANOCAMELID_MODEL_GGUF, pass an explicit .gguf path, or place the 1B Q4_0 or Q8_0 GGUF under $WORKSPACE/models." >&2
exit 2
