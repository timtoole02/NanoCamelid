#!/usr/bin/env bash
# Sweep real 1B chat prefill batch sizes on a Pi workspace.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bench-1b-prefill.sh [model.gguf] [prompt] [max_tokens] [temp] [batches] [--dry-run]

Probes the host, runs the strict Llama 3.2 1B shape audit, then runs the
Pi-local 1B chat path repeatedly with different NANOCAMELID_PREFILL_BATCH
values. Each run prints NanoCamelid's normal "Prompt ingested" and generation
timing lines, followed by a per-batch JSON summary line. Successful sweeps
finish with `prefill_bench_1b_status: ok` and a compact JSON summary that
records the host probe and best observed prefill and decode batches.

Model resolution:
  1. explicit model.gguf argument
  2. NANOCAMELID_SMOKE_GGUF
  3. NANOCAMELID_MODEL_GGUF
  4. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf
  5. $NANOCAMELID_WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf

Useful env:
  NANOCAMELID_WORKSPACE          Pi workspace, default /mnt/nanocamelid
  CARGO_TARGET_DIR               Cargo output dir, default /mnt/nanocamelid/target
  NANOCAMELID_SMOKE_GGUF         Smoke-specific 1B GGUF override
  NANOCAMELID_MODEL_GGUF         Shared 1B GGUF override
  NANOCAMELID_PREFILL_BATCHES    Batch list, default "1,16,32,64"
  NANOCAMELID_PREFILL_PROMPT     Prompt override
  NANOCAMELID_PREFILL_TOKENS     Generated token count, default 2
  NANOCAMELID_PREFILL_TEMP       Temperature, default 0.0
  NANOCAMELID_CONTEXT_LIMIT      Optional runtime context cap
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

kernel_env_prefix() {
  printf 'NANOCAMELID_Q8_DOT_SDOT=%s NANOCAMELID_Q8_DOT_KERNEL=%s ' \
    "$(shell_quote "$NANOCAMELID_Q8_DOT_SDOT")" \
    "$(shell_quote "$NANOCAMELID_Q8_DOT_KERNEL")"
}

json_number_or_null() {
  if [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf '%s' "$1"
  elif [[ "${1:-}" =~ ^[.][0-9]+$ ]]; then
    printf '0%s' "$1"
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

json_array_from_batches() {
  local out="["
  local first=1
  local batch

  for batch in "$@"; do
    if [[ "$first" == "1" ]]; then
      first=0
    else
      out+=","
    fi
    out+="$batch"
  done

  out+="]"
  printf '%s' "$out"
}

extract_batch_metrics() {
  local run_log="$1"

  BATCH_PROMPT_TOKENS_PER_SEC=""
  BATCH_PROMPT_TOKENS="$(
    sed -nE 's/^json: \{.*"prompt_tokens":([0-9]+),.*$/\1/p' "$run_log" \
      | tail -n 1
  )"
  BATCH_PREFILL_SEC="$(
    sed -nE 's/^Prompt ingested in ([0-9.]+)s with prefill batch [0-9]+$/\1/p' "$run_log" \
      | tail -n 1
  )"
  if [[ -z "$BATCH_PREFILL_SEC" ]]; then
    BATCH_PREFILL_SEC="$(
      sed -nE 's/^json: \{.*"prefill_sec":([0-9.]+),.*$/\1/p' "$run_log" \
        | tail -n 1
    )"
  fi
  read -r BATCH_GENERATED_TOKENS BATCH_GENERATION_SEC BATCH_TOKENS_PER_SEC < <(
    sed -nE 's/^Generated ([0-9]+) tokens in ([0-9.]+)s \(([0-9.]+) tokens\/sec\)$/\1 \2 \3/p' "$run_log" \
      | tail -n 1
  ) || true
  if [[ "$BATCH_PROMPT_TOKENS" =~ ^[0-9]+$ ]] && [[ "$BATCH_PREFILL_SEC" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    BATCH_PROMPT_TOKENS_PER_SEC="$(awk "BEGIN { if ($BATCH_PREFILL_SEC > 0) printf \"%.6f\", $BATCH_PROMPT_TOKENS / $BATCH_PREFILL_SEC }")"
  fi
}

print_batch_json() {
  local batch="$1"
  local exit_status="$2"
  local status="ok"

  if [[ "$exit_status" -ne 0 ]]; then
    status="failed"
  fi

  printf 'json: {"benchmark":"llama32-1b-prefill","batch_size":%s,"status":"%s","exit_status":%s,"prompt_tokens":%s,"prefill_sec":%s,"prompt_tokens_per_sec":%s,"generated_tokens":%s,"generation_sec":%s,"tokens_per_sec":%s}\n' \
    "$batch" \
    "$status" \
    "$exit_status" \
    "$(json_integer_or_null "$BATCH_PROMPT_TOKENS")" \
    "$(json_number_or_null "$BATCH_PREFILL_SEC")" \
    "$(json_number_or_null "$BATCH_PROMPT_TOKENS_PER_SEC")" \
    "$(json_integer_or_null "$BATCH_GENERATED_TOKENS")" \
    "$(json_number_or_null "$BATCH_GENERATION_SEC")" \
    "$(json_number_or_null "$BATCH_TOKENS_PER_SEC")"
}

context_limit_plan_value() {
  if [[ -n "${NANOCAMELID_CONTEXT_LIMIT:-}" ]]; then
    printf '%s' "$NANOCAMELID_CONTEXT_LIMIT"
  else
    printf 'unset'
  fi
}

prefill_summary_json() {
  local best_prefill_batch="$1"
  local best_prefill_sec="$2"
  local best_prefill_prompt_tokens_per_sec="$3"
  local best_decode_batch="$4"
  local best_tokens_per_sec="$5"

  printf '{"benchmark":"llama32-1b-prefill","target":"llama32-1b","status":"ok","model":%s,"selected_source":%s,"quantization":%s,"probe":true,"shape":"llama32_1b","shape_ready":true,"context_limit":%s,"max_tokens":%s,"temp":%s,"batches":%s,"best_prefill_batch":%s,"best_prefill_sec":%s,"best_prefill_prompt_tokens_per_sec":%s,"best_decode_batch":%s,"best_tokens_per_sec":%s}\n' \
    "$(json_string "$MODEL")" \
    "$(json_string "$MODEL_SOURCE")" \
    "$(json_string "$(llama32_1b_quantization_for_path "$MODEL")")" \
    "$(json_string "$(context_limit_plan_value)")" \
    "$MAX_TOKENS" \
    "$(json_number_or_null "$TEMP")" \
    "$(json_array_from_batches "${BATCHES[@]}")" \
    "$(json_integer_or_null "$best_prefill_batch")" \
    "$(json_number_or_null "$best_prefill_sec")" \
    "$(json_number_or_null "$best_prefill_prompt_tokens_per_sec")" \
    "$(json_integer_or_null "$best_decode_batch")" \
    "$(json_number_or_null "$best_tokens_per_sec")"
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
require_optional_context_limit
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
if looks_like_non_gguf_model_path "${1:-}"; then
  echo "1B prefill benchmark model argument must be a .gguf path: $1" >&2
  exit 2
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
SEEN_BATCHES=" "
for batch in ${BATCHES_RAW//,/ }; do
  if [[ ! "$batch" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid prefill batch size: $batch" >&2
    exit 2
  fi
  case "$SEEN_BATCHES" in
    *" $batch "*)
      echo "Duplicate prefill batch size: $batch" >&2
      exit 2
      ;;
  esac
  BATCHES+=("$batch")
  SEEN_BATCHES+="$batch "
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
  echo "workspace: $WORKSPACE"
  echo "q4_model: $Q4_MODEL"
  echo "q4_exists: $([[ -f "$Q4_MODEL" ]] && echo true || echo false)"
  echo "q8_model: $Q8_MODEL"
  echo "q8_exists: $([[ -f "$Q8_MODEL" ]] && echo true || echo false)"
  echo "selected_source: $MODEL_SOURCE"
  echo "model: $MODEL"
  echo "model_exists: $([[ -f "$MODEL" ]] && echo true || echo false)"
  echo "quantization: $(llama32_1b_quantization_for_path "$MODEL")"
  echo "prompt: $PROMPT"
  echo "max_tokens: $MAX_TOKENS"
  echo "temp: $TEMP"
  echo "context_limit: ${NANOCAMELID_CONTEXT_LIMIT:-unset}"
  echo "probe: enabled"
  echo "shape_audit: enabled"
  echo "smoke_gate: enabled"
  echo "batches: ${BATCHES[*]}"
  echo "status_on_success: prefill_bench_1b_status: ok"
  echo "json_on_success: $(prefill_summary_json "" "" "" "" "")"
  printf 'probe_command: '
  shell_command nanocamelid probe
  printf 'model_command: '
  shell_command nanocamelid model 1b "$MODEL"
  printf 'inspect_command: '
  shell_command nanocamelid inspect "$MODEL"
  printf 'smoke_command: '
  context_env_prefix
  kernel_env_prefix
  shell_command nanocamelid smoke 1b "$MODEL" chat "$PROMPT" "$MAX_TOKENS"
  for batch in "${BATCHES[@]}"; do
    printf 'batch_%s_command: ' "$batch"
    context_env_prefix
    kernel_env_prefix
    printf 'NANOCAMELID_PREFILL_BATCH=%s nanocamelid chat %s %s %s %s\n' \
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
  echo "Set NANOCAMELID_SMOKE_GGUF=/path/to/model.gguf, set NANOCAMELID_MODEL_GGUF=/path/to/model.gguf, or place the 1B Q4_0 or Q8_0 GGUF at the default path." >&2
  exit 2
fi

echo "NanoCamelid Llama 3.2 1B prefill sweep"
echo "model: $MODEL"
echo "prompt: $PROMPT"
echo "max_tokens: $MAX_TOKENS"
echo "temp: $TEMP"
echo "context_limit: ${NANOCAMELID_CONTEXT_LIMIT:-unset}"
echo "probe: enabled"
echo "shape_audit: enabled"
echo "smoke_gate: enabled"
echo "batches: ${BATCHES[*]}"

echo "==> Probing host fast-path support"
run_nanocamelid probe

echo "==> Auditing 1B model shape: $MODEL"
run_nanocamelid model 1b "$MODEL"

echo "==> Inspecting 1B model: $MODEL"
run_nanocamelid inspect "$MODEL"

echo "==> Running 1B chat smoke gate"
run_nanocamelid smoke 1b "$MODEL" chat "$PROMPT" "$MAX_TOKENS"

EXIT_STATUS=0
RUN_LOG="$(mktemp "${TMPDIR:-/tmp}/nanocamelid-prefill.XXXXXX")"
trap 'rm -f "$RUN_LOG"' EXIT
BEST_PREFILL_BATCH=""
BEST_PREFILL_SEC=""
BEST_PREFILL_PROMPT_TOKENS_PER_SEC=""
BEST_DECODE_BATCH=""
BEST_TOKENS_PER_SEC=""

for batch in "${BATCHES[@]}"; do
  echo
  echo "==> Running with NANOCAMELID_PREFILL_BATCH=$batch"
  : >"$RUN_LOG"
  set +e
  NANOCAMELID_PREFILL_BATCH="$batch" run_nanocamelid chat "$MODEL" "$PROMPT" "$TEMP" "$MAX_TOKENS" 2>&1 | tee "$RUN_LOG"
  batch_status=${PIPESTATUS[0]}
  set -e
  extract_batch_metrics "$RUN_LOG"
  print_batch_json "$batch" "$batch_status"
  if [[ "$batch_status" -ne 0 ]]; then
    EXIT_STATUS="$batch_status"
    break
  fi
  if [[ -n "$BATCH_PREFILL_SEC" ]] \
    && { [[ -z "$BEST_PREFILL_SEC" ]] || awk "BEGIN { exit !($BATCH_PREFILL_SEC < $BEST_PREFILL_SEC) }"; }; then
    BEST_PREFILL_BATCH="$batch"
    BEST_PREFILL_SEC="$BATCH_PREFILL_SEC"
    BEST_PREFILL_PROMPT_TOKENS_PER_SEC="$BATCH_PROMPT_TOKENS_PER_SEC"
  fi
  if [[ -n "$BATCH_TOKENS_PER_SEC" ]] \
    && { [[ -z "$BEST_TOKENS_PER_SEC" ]] || awk "BEGIN { exit !($BATCH_TOKENS_PER_SEC > $BEST_TOKENS_PER_SEC) }"; }; then
    BEST_DECODE_BATCH="$batch"
    BEST_TOKENS_PER_SEC="$BATCH_TOKENS_PER_SEC"
  fi
done

if [[ "$EXIT_STATUS" -eq 0 ]]; then
  echo "prefill_bench_1b_status: ok"
  echo "json: $(prefill_summary_json "$BEST_PREFILL_BATCH" "$BEST_PREFILL_SEC" "$BEST_PREFILL_PROMPT_TOKENS_PER_SEC" "$BEST_DECODE_BATCH" "$BEST_TOKENS_PER_SEC")"
fi

exit "$EXIT_STATUS"
