#!/usr/bin/env bash
# Sweep real 1B chat prefill batch sizes on a Pi workspace.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bench-1b-prefill.sh [model.gguf] [prompt] [max_tokens] [temp] [batches] [--dry-run]

Runs the Pi-local Llama 3.2 1B chat path repeatedly with different
NANOCAMELID_PREFILL_BATCH values. Each run prints NanoCamelid's normal
"Prompt ingested" and generation timing lines, followed by a JSON summary line.

Model resolution:
  1. explicit model.gguf argument
  2. NANOCAMELID_MODEL_GGUF
  3. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf
  4. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf

Useful env:
  NANOCAMELID_WORKSPACE          Pi workspace, default /mnt/nanocamelid
  CARGO_TARGET_DIR               Cargo output dir, default /mnt/nanocamelid/target
  NANOCAMELID_PREFILL_BATCHES    Batch list, default "1,16,32,64"
  NANOCAMELID_PREFILL_PROMPT     Prompt override
  NANOCAMELID_PREFILL_TOKENS     Generated token count, default 2
  NANOCAMELID_PREFILL_TEMP       Temperature, default 0.0
  --dry-run                      Print the resolved sweep plan without loading the model
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

shell_quote() {
  printf '%q' "$1"
}

json_number_or_null() {
  if [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf '%s' "$1"
  else
    printf 'null'
  fi
}

json_integer_or_null() {
  if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    printf '%s' "$1"
  else
    printf 'null'
  fi
}

print_batch_json() {
  local batch="$1"
  local exit_status="$2"
  local run_log="$3"
  local status="ok"
  local prefill_sec=""
  local generated_tokens=""
  local generation_sec=""
  local tokens_per_sec=""

  if [[ "$exit_status" -ne 0 ]]; then
    status="failed"
  fi

  prefill_sec="$(
    sed -nE 's/^Prompt ingested in ([0-9.]+)s with prefill batch [0-9]+$/\1/p' "$run_log" \
      | tail -n 1
  )"
  read -r generated_tokens generation_sec tokens_per_sec < <(
    sed -nE 's/^Generated ([0-9]+) tokens in ([0-9.]+)s \(([0-9.]+) tokens\/sec\)$/\1 \2 \3/p' "$run_log" \
      | tail -n 1
  ) || true

  printf 'json: {"benchmark":"llama32-1b-prefill","batch_size":%s,"status":"%s","exit_status":%s,"prefill_sec":%s,"generated_tokens":%s,"generation_sec":%s,"tokens_per_sec":%s}\n' \
    "$batch" \
    "$status" \
    "$exit_status" \
    "$(json_number_or_null "$prefill_sec")" \
    "$(json_integer_or_null "$generated_tokens")" \
    "$(json_number_or_null "$generation_sec")" \
    "$(json_number_or_null "$tokens_per_sec")"
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
source "$SCRIPT_DIR/common.sh"
WORKSPACE="${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}"
REPO="${NANOCAMELID_REPO:-$REPO_ROOT}"
TARGET_DIR="${CARGO_TARGET_DIR:-${NANOCAMELID_TARGET_DIR:-/mnt/nanocamelid/target}}"
Q4_MODEL="$WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf"
Q8_MODEL="$WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
if [[ -n "${NANOCAMELID_MODEL_GGUF:-}" ]]; then
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
if [[ $# -gt 4 ]]; then
  echo "Unexpected extra prefill benchmark argument: ${5}" >&2
  usage >&2
  exit 2
fi

PROMPT="${1:-${NANOCAMELID_PREFILL_PROMPT:-Explain one practical Raspberry Pi inference bottleneck in two short sentences.}}"
MAX_TOKENS="${2:-${NANOCAMELID_PREFILL_TOKENS:-2}}"
TEMP="${3:-${NANOCAMELID_PREFILL_TEMP:-0.0}}"
BATCHES_RAW="${4:-${NANOCAMELID_PREFILL_BATCHES:-1,16,32,64}}"
require_positive_integer "Generated token count" "$MAX_TOKENS"
require_non_negative_float "Temperature" "$TEMP"
BINARY="${NANOCAMELID_BIN:-$TARGET_DIR/release/nanocamelid}"
export NANOCAMELID_Q8_DOT_SDOT="${NANOCAMELID_Q8_DOT_SDOT:-1}"
export NANOCAMELID_Q8_DOT_KERNEL="${NANOCAMELID_Q8_DOT_KERNEL:-sdot}"

BATCHES=()
for batch in ${BATCHES_RAW//,/ }; do
  if [[ ! "$batch" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid prefill batch size: $batch" >&2
    exit 2
  fi
  BATCHES+=("$batch")
done
if [[ ${#BATCHES[@]} -eq 0 ]]; then
  echo "No prefill batch sizes were provided." >&2
  exit 2
fi

if [[ -x "$BINARY" ]]; then
  launcher_mode="binary"
elif command -v cargo >/dev/null 2>&1; then
  launcher_mode="cargo"
else
  launcher_mode="unavailable"
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

if [[ "$DRY_RUN" == "1" ]]; then
  echo "NanoCamelid Llama 3.2 1B prefill sweep dry run"
  echo "repo: $REPO"
  echo "cargo_target_dir: $TARGET_DIR"
  echo "launcher_mode: $launcher_mode"
  echo "binary: $BINARY"
  echo "selected_source: $MODEL_SOURCE"
  echo "model: $MODEL"
  echo "model_exists: $([[ -f "$MODEL" ]] && echo true || echo false)"
  echo "prompt: $PROMPT"
  echo "max_tokens: $MAX_TOKENS"
  echo "temp: $TEMP"
  echo "batches: ${BATCHES[*]}"
  for batch in "${BATCHES[@]}"; do
    printf 'batch_%s_command: NANOCAMELID_Q8_DOT_SDOT=%s NANOCAMELID_Q8_DOT_KERNEL=%s NANOCAMELID_PREFILL_BATCH=%s nanocamelid chat %s %s %s %s\n' \
      "$batch" \
      "$(shell_quote "$NANOCAMELID_Q8_DOT_SDOT")" \
      "$(shell_quote "$NANOCAMELID_Q8_DOT_KERNEL")" \
      "$(shell_quote "$batch")" \
      "$(shell_quote "$MODEL")" \
      "$(shell_quote "$PROMPT")" \
      "$(shell_quote "$TEMP")" \
      "$(shell_quote "$MAX_TOKENS")"
  done
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
if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Set NANOCAMELID_MODEL_GGUF=/path/to/model.gguf or place the 1B Q4_0 or Q8_0 GGUF at the default path." >&2
  exit 2
fi

echo "NanoCamelid Llama 3.2 1B prefill sweep"
echo "model: $MODEL"
echo "prompt: $PROMPT"
echo "max_tokens: $MAX_TOKENS"
echo "temp: $TEMP"
echo "batches: ${BATCHES[*]}"

EXIT_STATUS=0
RUN_LOG="$(mktemp "${TMPDIR:-/tmp}/nanocamelid-prefill.XXXXXX")"
trap 'rm -f "$RUN_LOG"' EXIT

for batch in "${BATCHES[@]}"; do
  echo
  echo "==> Running with NANOCAMELID_PREFILL_BATCH=$batch"
  : >"$RUN_LOG"
  set +e
  NANOCAMELID_PREFILL_BATCH="$batch" run_nanocamelid chat "$MODEL" "$PROMPT" "$TEMP" "$MAX_TOKENS" 2>&1 | tee "$RUN_LOG"
  batch_status=${PIPESTATUS[0]}
  set -e
  print_batch_json "$batch" "$batch_status" "$RUN_LOG"
  if [[ "$batch_status" -ne 0 ]]; then
    EXIT_STATUS="$batch_status"
    break
  fi
done

exit "$EXIT_STATUS"
