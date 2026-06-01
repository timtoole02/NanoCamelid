#!/usr/bin/env bash
# scripts/remote_build.sh - Compile and run tests/benchmarks on the Raspberry Pi remotely
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage: remote_build.sh <pi-ip-or-hostname> [ssh-key-path] [pi-username] [rsync|git-ff] [--dry-run]

Builds and validates NanoCamelid on a Raspberry Pi workspace.

Options:
  --dry-run   Print the resolved deploy/build/readiness plan without SSH or deploy

Useful env:
  NANOCAMELID_REMOTE_CONTEXT_LIMIT Optional single context cap for readiness and prefill sweep
  NANOCAMELID_REMOTE_CONTEXT_PACKS  Optional comma-separated 1B context caps to run after readiness
  NANOCAMELID_REMOTE_PREFILL_BATCH   Optional prompt prefill batch for remote readiness/smoke gates
  NANOCAMELID_REMOTE_TARGET_DIR      Optional absolute Cargo target dir; defaults to <remote-workspace>/target
  NANOCAMELID_REMOTE_MIN_FREE_KB     Optional minimum free KiB required before deploy; defaults to 262144
  NANOCAMELID_REMOTE_DIRTY_POLICY    fail/archive policy for dirty git-ff Pi checkouts; defaults to fail
  NANOCAMELID_REMOTE_1B_QUANT       Optional q4/q8 selector for Pi-local default 1B rows
  NANOCAMELID_REMOTE_PREFILL_BENCH  Set to 1 to run the 1B prefill batch sweep after readiness; 0/false/no/off disables it
  NANOCAMELID_REMOTE_PREFILL_BATCHES Optional comma-separated prefill batches for the remote sweep
  NANOCAMELID_REMOTE_EVIDENCE       Set to 1 to run the 1B evidence bundle after the core build; 0/false/no/off disables it
  NANOCAMELID_REMOTE_READY_CHAT     Set to 0/false/no/off to skip the direct readiness chat turn
  NANOCAMELID_REMOTE_SMOKE=0        Also accepts false/no/off to skip model-backed 1B gates
USAGE
}

DRY_RUN=0
POSITIONAL_ARGS=()
for arg in "$@"; do
  case "$arg" in
    -h | --help)
      usage
      exit 0
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

PI_HOST="${1:-}"
SSH_KEY="${2:-${NANOCAMELID_SSH_KEY:-}}"
PI_USER="${3:-${NANOCAMELID_PI_USER:-$USER}}"
DEPLOY_MODE="${4:-${NANOCAMELID_DEPLOY_MODE:-git-ff}}"
PI_WORKSPACE="${NANOCAMELID_REMOTE_WORKSPACE:-/mnt/nanocamelid}"
PI_TARGET_DIR="${NANOCAMELID_REMOTE_TARGET_DIR:-$PI_WORKSPACE/target}"
PI_REPO="$PI_WORKSPACE/src/NanoCamelid"
REMOTE_MIN_FREE_KB="${NANOCAMELID_REMOTE_MIN_FREE_KB:-262144}"
REMOTE_DIRTY_POLICY="${NANOCAMELID_REMOTE_DIRTY_POLICY:-fail}"
REMOTE_SMOKE_ENABLED="${NANOCAMELID_REMOTE_SMOKE:-1}"
REMOTE_SMOKE_ENABLED_LOWER="$(printf '%s' "$REMOTE_SMOKE_ENABLED" | tr '[:upper:]' '[:lower:]')"
REMOTE_SMOKE_GGUF="${NANOCAMELID_REMOTE_SMOKE_GGUF:-}"
REMOTE_SMOKE_KIND="${NANOCAMELID_REMOTE_SMOKE_KIND:-chat}"
SMOKE_PROMPT="${NANOCAMELID_SMOKE_PROMPT:-Say hello in one sentence.}"
SMOKE_TOKENS="${NANOCAMELID_SMOKE_TOKENS:-8}"
READY_CHAT="${NANOCAMELID_REMOTE_READY_CHAT:-${NANOCAMELID_READY_CHAT:-1}}"
READY_PROMPT="${NANOCAMELID_READY_PROMPT:-$SMOKE_PROMPT}"
READY_TOKENS="${NANOCAMELID_READY_TOKENS:-$SMOKE_TOKENS}"
READY_TEMP="${NANOCAMELID_READY_TEMP:-0.0}"
READY_CHAT_LOWER="$(printf '%s' "$READY_CHAT" | tr '[:upper:]' '[:lower:]')"
case "$READY_CHAT_LOWER" in
  "" | 0 | 1 | false | true | no | yes | off | on) ;;
  *)
    echo "NANOCAMELID_REMOTE_READY_CHAT/NANOCAMELID_READY_CHAT must be 0, 1, false, true, no, yes, off, or on: $READY_CHAT" >&2
    exit 2
    ;;
esac
REMOTE_CONTEXT_LIMIT="${NANOCAMELID_REMOTE_CONTEXT_LIMIT:-${NANOCAMELID_CONTEXT_LIMIT:-}}"
REMOTE_CONTEXT_PACKS="${NANOCAMELID_REMOTE_CONTEXT_PACKS:-}"
REMOTE_PREFILL_BATCH="${NANOCAMELID_REMOTE_PREFILL_BATCH:-${NANOCAMELID_PREFILL_BATCH:-}}"
REMOTE_1B_QUANT="${NANOCAMELID_REMOTE_1B_QUANT:-}"
REMOTE_1B_QUANT_LOWER="$(printf '%s' "$REMOTE_1B_QUANT" | tr '[:upper:]' '[:lower:]')"
REMOTE_PREFILL_BENCH="${NANOCAMELID_REMOTE_PREFILL_BENCH:-0}"
REMOTE_PREFILL_BENCH_LOWER="$(printf '%s' "$REMOTE_PREFILL_BENCH" | tr '[:upper:]' '[:lower:]')"
REMOTE_EVIDENCE="${NANOCAMELID_REMOTE_EVIDENCE:-0}"
REMOTE_EVIDENCE_LOWER="$(printf '%s' "$REMOTE_EVIDENCE" | tr '[:upper:]' '[:lower:]')"
DEFAULT_PREFILL_PROMPT="Explain one practical Raspberry Pi inference bottleneck in two short sentences."
PREFILL_PROMPT="${NANOCAMELID_PREFILL_PROMPT:-$DEFAULT_PREFILL_PROMPT}"
PREFILL_TOKENS="${NANOCAMELID_PREFILL_TOKENS:-2}"
PREFILL_TEMP="${NANOCAMELID_PREFILL_TEMP:-0.0}"
PREFILL_BATCHES="${NANOCAMELID_REMOTE_PREFILL_BATCHES:-${NANOCAMELID_PREFILL_BATCHES:-1,16,32,64}}"
SSH_CONNECT_TIMEOUT="${NANOCAMELID_SSH_CONNECT_TIMEOUT:-10}"

if [[ -z "$PI_HOST" ]]; then
  usage
  exit 1
fi

case "$DEPLOY_MODE" in
  rsync | git-ff) ;;
  *)
    echo "Unknown deploy mode: $DEPLOY_MODE" >&2
    echo "Expected rsync or git-ff." >&2
    exit 2
    ;;
esac

if [[ "$REMOTE_SMOKE_ENABLED_LOWER" != "0" && "$REMOTE_SMOKE_ENABLED_LOWER" != "false" && "$REMOTE_SMOKE_ENABLED_LOWER" != "no" && "$REMOTE_SMOKE_ENABLED_LOWER" != "off" ]]; then
  case "$REMOTE_SMOKE_KIND" in
    chat | model | q8-chat | q8-model) ;;
    *)
      echo "Unknown smoke kind: $REMOTE_SMOKE_KIND" >&2
      echo "Expected chat, model, q8-chat, or q8-model." >&2
      exit 2
      ;;
  esac
fi

if [[ ! -f "$SSH_KEY" ]]; then
  SSH_OPTS=(
    -o BatchMode=yes
    -o ConnectTimeout="$SSH_CONNECT_TIMEOUT"
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=1
  )
else
  SSH_OPTS=(
    -i "$SSH_KEY"
    -o BatchMode=yes
    -o ConnectTimeout="$SSH_CONNECT_TIMEOUT"
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=1
  )
fi

shell_quote() {
  printf '%q' "$1"
}

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

require_non_negative_integer() {
  local label="$1"
  local value="$2"

  if [[ ! "${value:-}" =~ ^[0-9]+$ ]]; then
    echo "$label must be a non-negative integer: $value" >&2
    exit 2
  fi
}

require_absolute_path() {
  local label="$1"
  local value="$2"

  case "$value" in
    /*) ;;
    *)
      echo "$label must be an absolute path: $value" >&2
      exit 2
      ;;
  esac
}

require_remote_cargo_target_dir() {
  case "$PI_TARGET_DIR" in
    "$PI_REPO"/target | "$PI_REPO"/target/*)
      echo "Refusing to use a remote repo-local Cargo target dir: $PI_TARGET_DIR" >&2
      echo "Set NANOCAMELID_REMOTE_TARGET_DIR outside the remote checkout." >&2
      exit 2
      ;;
  esac
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

require_context_caps() {
  local raw_caps="$1"
  local cap
  local cap_count=0
  local seen_caps=" "

  if [[ "$raw_caps" =~ (^|,)[[:space:]]*(,|$) ]]; then
    echo "Context cap must be a positive integer: empty value" >&2
    exit 2
  fi

  for cap in ${raw_caps//,/ }; do
    cap_count=$((cap_count + 1))
    if [[ ! "$cap" =~ ^[1-9][0-9]*$ ]]; then
      echo "Context cap must be a positive integer: $cap" >&2
      exit 2
    fi
    case "$seen_caps" in
      *" $cap "*)
        echo "Context caps must be unique: $cap" >&2
        exit 2
        ;;
    esac
    seen_caps+="$cap "
  done
  if [[ "$cap_count" -eq 0 ]]; then
    echo "Context caps must include at least one positive integer." >&2
    exit 2
  fi
}

require_prefill_batches() {
  local raw_batches="$1"
  local batch
  local batch_count=0
  local seen_batches=" "

  if [[ "$raw_batches" =~ (^|,)[[:space:]]*(,|$) ]]; then
    echo "Prefill batch size must be a positive integer: empty value" >&2
    exit 2
  fi

  for batch in ${raw_batches//,/ }; do
    batch_count=$((batch_count + 1))
    if [[ ! "$batch" =~ ^[1-9][0-9]*$ ]]; then
      echo "Prefill batch size must be a positive integer: $batch" >&2
      exit 2
    fi
    case "$seen_batches" in
      *" $batch "*)
        echo "Prefill batch sizes must be unique: $batch" >&2
        exit 2
        ;;
    esac
    seen_batches+="$batch "
  done
  if [[ "$batch_count" -eq 0 ]]; then
    echo "Prefill batches must include at least one positive integer." >&2
    exit 2
  fi
}

redacted_deploy_key_label() {
  if [[ -n "$SSH_KEY" ]]; then
    echo "<ssh-key-path>"
  else
    echo "<ssh-agent>"
  fi
}

ready_chat_disabled() {
  is_disabled_toggle "$READY_CHAT_LOWER"
}

is_disabled_toggle() {
  [[ "$1" == "0" || "$1" == "false" || "$1" == "no" || "$1" == "off" ]]
}

is_enabled_toggle() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" || "$1" == "on" ]]
}

require_toggle() {
  local label="$1"
  local value="$2"
  local lower_value

  lower_value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
  case "$lower_value" in
    "" | 0 | 1 | false | true | no | yes | off | on) ;;
    *)
      echo "$label must be 0, 1, false, true, no, yes, off, or on: $value" >&2
      exit 2
      ;;
  esac
}

print_readiness_command() {
  local model_arg="${1:-}"

  printf 'readiness_command:'
  if [[ -n "$REMOTE_CONTEXT_LIMIT" ]]; then
    printf ' NANOCAMELID_CONTEXT_LIMIT=%s' "$(shell_quote "$REMOTE_CONTEXT_LIMIT")"
  fi
  if [[ -n "$REMOTE_PREFILL_BATCH" ]]; then
    printf ' NANOCAMELID_PREFILL_BATCH=%s' "$(shell_quote "$REMOTE_PREFILL_BATCH")"
  fi
  printf ' NANOCAMELID_READY_CHAT=%s NANOCAMELID_READY_SMOKE_KIND=%s NANOCAMELID_READY_SMOKE_PROMPT=%s NANOCAMELID_READY_SMOKE_TOKENS=%s' \
    "$(shell_quote "$READY_CHAT")" \
    "$(shell_quote "$REMOTE_SMOKE_KIND")" \
    "$(shell_quote "$SMOKE_PROMPT")" \
    "$(shell_quote "$SMOKE_TOKENS")"
  if ! ready_chat_disabled; then
    printf ' NANOCAMELID_READY_PROMPT=%s NANOCAMELID_READY_TOKENS=%s NANOCAMELID_READY_TEMP=%s' \
      "$(shell_quote "$READY_PROMPT")" \
      "$(shell_quote "$READY_TOKENS")" \
      "$(shell_quote "$READY_TEMP")"
  fi
  printf ' ./scripts/pi/ready-1b.sh'
  if [[ -n "$model_arg" ]]; then
    printf ' %s' "$(shell_quote "$model_arg")"
  fi
  printf '\n'
}

prefill_bench_enabled() {
  is_enabled_toggle "$REMOTE_PREFILL_BENCH_LOWER"
}

evidence_enabled() {
  is_enabled_toggle "$REMOTE_EVIDENCE_LOWER"
}

remote_smoke_disabled() {
  is_disabled_toggle "$REMOTE_SMOKE_ENABLED_LOWER"
}

remote_quant_flag() {
  case "$REMOTE_1B_QUANT_LOWER" in
    q4 | q4_0) printf '%s\n' "--q4" ;;
    q8 | q8_0) printf '%s\n' "--q8" ;;
  esac
}

print_prefill_bench_command() {
  local model_arg="${1:-}"
  local quant_flag
  quant_flag="$(remote_quant_flag)"

  printf 'prefill_bench_command:'
  if [[ -n "$REMOTE_CONTEXT_LIMIT" ]]; then
    printf ' NANOCAMELID_CONTEXT_LIMIT=%s' "$(shell_quote "$REMOTE_CONTEXT_LIMIT")"
  fi
  if [[ -n "$REMOTE_PREFILL_BATCH" ]]; then
    printf ' NANOCAMELID_PREFILL_BATCH=%s' "$(shell_quote "$REMOTE_PREFILL_BATCH")"
  fi
  printf ' NANOCAMELID_PREFILL_PROMPT=%s NANOCAMELID_PREFILL_TOKENS=%s NANOCAMELID_PREFILL_TEMP=%s NANOCAMELID_PREFILL_BATCHES=%s ./scripts/pi/bench-1b-prefill.sh' \
    "$(shell_quote "$PREFILL_PROMPT")" \
    "$(shell_quote "$PREFILL_TOKENS")" \
    "$(shell_quote "$PREFILL_TEMP")" \
    "$(shell_quote "$PREFILL_BATCHES")"
  if [[ -n "$model_arg" ]]; then
    printf ' %s' "$(shell_quote "$model_arg")"
  elif [[ -n "$quant_flag" ]]; then
    printf ' %s' "$(shell_quote "$quant_flag")"
  fi
  printf '\n'
}

print_evidence_command() {
  local model_arg="${1:-}"
  local quant_flag
  quant_flag="$(remote_quant_flag)"

  printf 'evidence_command:'
  if [[ -n "$REMOTE_CONTEXT_LIMIT" ]]; then
    printf ' NANOCAMELID_CONTEXT_LIMIT=%s' "$(shell_quote "$REMOTE_CONTEXT_LIMIT")"
  fi
  if [[ -n "$REMOTE_PREFILL_BATCH" ]]; then
    printf ' NANOCAMELID_PREFILL_BATCH=%s' "$(shell_quote "$REMOTE_PREFILL_BATCH")"
  fi
  printf ' NANOCAMELID_SMOKE_KIND=%s NANOCAMELID_SMOKE_PROMPT=%s NANOCAMELID_SMOKE_TOKENS=%s NANOCAMELID_CONTEXT_PACKS=%s NANOCAMELID_PREFILL_PROMPT=%s NANOCAMELID_PREFILL_TOKENS=%s NANOCAMELID_PREFILL_TEMP=%s NANOCAMELID_PREFILL_BATCHES=%s ./scripts/pi/evidence-1b.sh' \
    "$(shell_quote "$REMOTE_SMOKE_KIND")" \
    "$(shell_quote "$SMOKE_PROMPT")" \
    "$(shell_quote "$SMOKE_TOKENS")" \
    "$(shell_quote "${REMOTE_CONTEXT_PACKS:-512,1024,2048,4096,8192}")" \
    "$(shell_quote "$PREFILL_PROMPT")" \
    "$(shell_quote "$PREFILL_TOKENS")" \
    "$(shell_quote "$PREFILL_TEMP")" \
    "$(shell_quote "$PREFILL_BATCHES")"
  if [[ -n "$model_arg" ]]; then
    printf ' %s' "$(shell_quote "$model_arg")"
  elif [[ -n "$quant_flag" ]]; then
    printf ' %s' "$(shell_quote "$quant_flag")"
  fi
  printf '\n'
}

require_toggle "NANOCAMELID_REMOTE_SMOKE" "$REMOTE_SMOKE_ENABLED"
require_toggle "NANOCAMELID_REMOTE_PREFILL_BENCH" "$REMOTE_PREFILL_BENCH"
require_toggle "NANOCAMELID_REMOTE_EVIDENCE" "$REMOTE_EVIDENCE"
require_absolute_path "NANOCAMELID_REMOTE_WORKSPACE" "$PI_WORKSPACE"
require_absolute_path "NANOCAMELID_REMOTE_TARGET_DIR" "$PI_TARGET_DIR"
require_remote_cargo_target_dir
require_non_negative_integer "NANOCAMELID_REMOTE_MIN_FREE_KB" "$REMOTE_MIN_FREE_KB"
case "$REMOTE_DIRTY_POLICY" in
  fail | archive) ;;
  *)
    echo "NANOCAMELID_REMOTE_DIRTY_POLICY must be fail or archive: $REMOTE_DIRTY_POLICY" >&2
    exit 2
    ;;
esac

if evidence_enabled && remote_smoke_disabled; then
  echo "NANOCAMELID_REMOTE_EVIDENCE requires NANOCAMELID_REMOTE_SMOKE to be enabled." >&2
  exit 2
fi

if [[ -n "$REMOTE_CONTEXT_LIMIT" ]]; then
  require_positive_integer "Remote context limit" "$REMOTE_CONTEXT_LIMIT"
fi
if [[ -n "$REMOTE_PREFILL_BATCH" ]]; then
  require_positive_integer "Remote prefill batch" "$REMOTE_PREFILL_BATCH"
fi
if [[ -n "$REMOTE_CONTEXT_PACKS" ]]; then
  require_context_caps "$REMOTE_CONTEXT_PACKS"
fi

if ! remote_smoke_disabled; then
  case "$REMOTE_1B_QUANT_LOWER" in
    "" | q4 | q4_0 | q8 | q8_0) ;;
    *)
      echo "NANOCAMELID_REMOTE_1B_QUANT must be q4 or q8: $REMOTE_1B_QUANT" >&2
      exit 2
      ;;
  esac
  if [[ -n "$REMOTE_1B_QUANT" && -n "$REMOTE_SMOKE_GGUF" ]]; then
    echo "NANOCAMELID_REMOTE_1B_QUANT cannot be combined with NANOCAMELID_REMOTE_SMOKE_GGUF; use one exact model selector." >&2
    exit 2
  fi
  if [[ -n "$REMOTE_SMOKE_GGUF" ]] && ! looks_like_gguf_path "$REMOTE_SMOKE_GGUF"; then
    echo "NANOCAMELID_REMOTE_SMOKE_GGUF must be a .gguf path: $REMOTE_SMOKE_GGUF" >&2
    exit 2
  fi
  require_positive_integer "Smoke token count" "$SMOKE_TOKENS"
  if [[ "$READY_CHAT_LOWER" != "0" && "$READY_CHAT_LOWER" != "false" && "$READY_CHAT_LOWER" != "no" && "$READY_CHAT_LOWER" != "off" ]]; then
    require_positive_integer "Readiness token count" "$READY_TOKENS"
    require_non_negative_float "Readiness temperature" "$READY_TEMP"
  fi
  if prefill_bench_enabled || evidence_enabled; then
    require_positive_integer "Prefill token count" "$PREFILL_TOKENS"
    require_non_negative_float "Prefill temperature" "$PREFILL_TEMP"
    require_prefill_batches "$PREFILL_BATCHES"
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "NanoCamelid remote build dry run"
  echo "target: <pi-user>@<pi-host>"
  echo "target_redacted: true"
  echo "deploy_mode: $DEPLOY_MODE"
  echo "ssh_batch_mode: yes"
  echo "ssh_connect_timeout_sec: $SSH_CONNECT_TIMEOUT"
  echo "remote_workspace: $PI_WORKSPACE"
  echo "remote_repo: $PI_REPO"
  echo "cargo_target_dir: $PI_TARGET_DIR"
  echo "remote_min_free_kb: $REMOTE_MIN_FREE_KB"
  echo "remote_dirty_policy: $REMOTE_DIRTY_POLICY"
  echo "remote_smoke_enabled: $REMOTE_SMOKE_ENABLED"
  echo "remote_smoke_kind: $REMOTE_SMOKE_KIND"
  echo "smoke_prompt: $SMOKE_PROMPT"
  echo "smoke_tokens: $SMOKE_TOKENS"
  echo "ready_chat: $READY_CHAT"
  echo "ready_prompt: $READY_PROMPT"
  echo "ready_tokens: $READY_TOKENS"
  echo "ready_temp: $READY_TEMP"
  echo "context_limit: ${REMOTE_CONTEXT_LIMIT:-unset}"
  echo "prefill_batch: ${REMOTE_PREFILL_BATCH:-default}"
  echo "remote_1b_quant: ${REMOTE_1B_QUANT:-auto}"
  echo "context_pack_caps: ${REMOTE_CONTEXT_PACKS:-skipped}"
  echo "prefill_bench_enabled: $REMOTE_PREFILL_BENCH"
  echo "evidence_enabled: $REMOTE_EVIDENCE"
  echo "prefill_prompt: $PREFILL_PROMPT"
  echo "prefill_tokens: $PREFILL_TOKENS"
  echo "prefill_temp: $PREFILL_TEMP"
  echo "prefill_batches: $PREFILL_BATCHES"
  printf 'deploy_command: scripts/deploy.sh %s %s %s %s\n' \
    "<pi-host>" \
    "$(redacted_deploy_key_label)" \
    "<pi-user>" \
    "$(shell_quote "$DEPLOY_MODE")"
  echo "remote_steps: cargo fmt -- --check; cargo test; cargo clippy --all-targets -- -D warnings; cargo check; cargo build --release; probe; model 1b --dry-run; ready 1b --dry-run; bench q8-dot 1000 3"
  if remote_smoke_disabled; then
    echo "readiness_shape_audit: skipped"
    echo "readiness_command: skipped"
    echo "evidence_command: skipped"
  elif evidence_enabled && [[ -n "$REMOTE_SMOKE_GGUF" ]]; then
    echo "readiness_shape_audit: delegated_to_evidence_bundle"
    echo "readiness_command: skipped"
    print_evidence_command "$REMOTE_SMOKE_GGUF"
  elif evidence_enabled; then
    echo "readiness_shape_audit: delegated_to_evidence_bundle"
    echo "readiness_command: skipped"
    print_evidence_command
  elif [[ -n "$REMOTE_SMOKE_GGUF" ]]; then
    echo "readiness_shape_audit: enabled"
    echo "evidence_command: skipped"
    print_readiness_command "$REMOTE_SMOKE_GGUF"
  else
    echo "readiness_shape_audit: enabled"
    echo "evidence_command: skipped"
    print_readiness_command "$(remote_quant_flag)"
  fi
  if remote_smoke_disabled || evidence_enabled || [[ -z "$REMOTE_CONTEXT_PACKS" ]]; then
    echo "context_pack_command: skipped"
  elif [[ -n "$REMOTE_SMOKE_GGUF" ]]; then
    printf 'context_pack_command:'
    if [[ -n "$REMOTE_PREFILL_BATCH" ]]; then
      printf ' NANOCAMELID_PREFILL_BATCH=%s' "$(shell_quote "$REMOTE_PREFILL_BATCH")"
    fi
    printf ' NANOCAMELID_CONTEXT_PACKS=%s ./scripts/pi/context-pack-1b.sh %s %s %s %s\n' \
      "$(shell_quote "$REMOTE_CONTEXT_PACKS")" \
      "$(shell_quote "$REMOTE_SMOKE_GGUF")" \
      "$(shell_quote "$REMOTE_SMOKE_KIND")" \
      "$(shell_quote "$SMOKE_PROMPT")" \
      "$(shell_quote "$SMOKE_TOKENS")"
  else
    printf 'context_pack_command:'
    if [[ -n "$REMOTE_PREFILL_BATCH" ]]; then
      printf ' NANOCAMELID_PREFILL_BATCH=%s' "$(shell_quote "$REMOTE_PREFILL_BATCH")"
    fi
    printf ' NANOCAMELID_CONTEXT_PACKS=%s ./scripts/pi/context-pack-1b.sh' \
      "$(shell_quote "$REMOTE_CONTEXT_PACKS")"
    quant_flag="$(remote_quant_flag)"
    if [[ -n "$quant_flag" ]]; then
      printf ' %s' "$(shell_quote "$quant_flag")"
    fi
    printf ' %s %s %s\n' \
      "$(shell_quote "$REMOTE_SMOKE_KIND")" \
      "$(shell_quote "$SMOKE_PROMPT")" \
      "$(shell_quote "$SMOKE_TOKENS")"
  fi
  if remote_smoke_disabled || evidence_enabled || ! prefill_bench_enabled; then
    echo "prefill_bench_command: skipped"
  elif [[ -n "$REMOTE_SMOKE_GGUF" ]]; then
    print_prefill_bench_command "$REMOTE_SMOKE_GGUF"
  else
    print_prefill_bench_command
  fi
  exit 0
fi

echo "Deploying latest changes first..."
"$(dirname "$0")/deploy.sh" "$PI_HOST" "$SSH_KEY" "$PI_USER" "$DEPLOY_MODE"

echo "Building NanoCamelid on target Pi..."
printf -v REMOTE_PI_WORKSPACE '%q' "$PI_WORKSPACE"
printf -v REMOTE_PI_TARGET_DIR '%q' "$PI_TARGET_DIR"
printf -v REMOTE_PI_REPO '%q' "$PI_REPO"
printf -v REMOTE_SMOKE_ENABLED_ARG '%q' "$REMOTE_SMOKE_ENABLED"
printf -v REMOTE_SMOKE_ENABLED_LOWER_ARG '%q' "$REMOTE_SMOKE_ENABLED_LOWER"
printf -v REMOTE_SMOKE_GGUF_ARG '%q' "$REMOTE_SMOKE_GGUF"
printf -v REMOTE_SMOKE_KIND_ARG '%q' "$REMOTE_SMOKE_KIND"
printf -v REMOTE_SMOKE_PROMPT_ARG '%q' "$SMOKE_PROMPT"
printf -v REMOTE_SMOKE_TOKENS_ARG '%q' "$SMOKE_TOKENS"
printf -v READY_CHAT_ARG '%q' "$READY_CHAT"
printf -v READY_PROMPT_ARG '%q' "$READY_PROMPT"
printf -v READY_TOKENS_ARG '%q' "$READY_TOKENS"
printf -v READY_TEMP_ARG '%q' "$READY_TEMP"
printf -v REMOTE_CONTEXT_LIMIT_ARG '%q' "$REMOTE_CONTEXT_LIMIT"
printf -v REMOTE_CONTEXT_PACKS_ARG '%q' "$REMOTE_CONTEXT_PACKS"
printf -v REMOTE_PREFILL_BATCH_ARG '%q' "$REMOTE_PREFILL_BATCH"
printf -v REMOTE_1B_QUANT_ARG '%q' "$REMOTE_1B_QUANT"
printf -v REMOTE_1B_QUANT_LOWER_ARG '%q' "$REMOTE_1B_QUANT_LOWER"
printf -v REMOTE_PREFILL_BENCH_ARG '%q' "$REMOTE_PREFILL_BENCH"
printf -v REMOTE_PREFILL_BENCH_LOWER_ARG '%q' "$REMOTE_PREFILL_BENCH_LOWER"
printf -v REMOTE_EVIDENCE_ARG '%q' "$REMOTE_EVIDENCE"
printf -v REMOTE_EVIDENCE_LOWER_ARG '%q' "$REMOTE_EVIDENCE_LOWER"
printf -v PREFILL_PROMPT_ARG '%q' "$PREFILL_PROMPT"
printf -v PREFILL_TOKENS_ARG '%q' "$PREFILL_TOKENS"
printf -v PREFILL_TEMP_ARG '%q' "$PREFILL_TEMP"
printf -v PREFILL_BATCHES_ARG '%q' "$PREFILL_BATCHES"
ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "${PI_USER}@${PI_HOST}" \
  "PI_WORKSPACE=$REMOTE_PI_WORKSPACE PI_TARGET_DIR=$REMOTE_PI_TARGET_DIR PI_REPO=$REMOTE_PI_REPO REMOTE_SMOKE_ENABLED=$REMOTE_SMOKE_ENABLED_ARG REMOTE_SMOKE_ENABLED_LOWER=$REMOTE_SMOKE_ENABLED_LOWER_ARG REMOTE_SMOKE_GGUF=$REMOTE_SMOKE_GGUF_ARG REMOTE_SMOKE_KIND=$REMOTE_SMOKE_KIND_ARG SMOKE_PROMPT=$REMOTE_SMOKE_PROMPT_ARG SMOKE_TOKENS=$REMOTE_SMOKE_TOKENS_ARG READY_CHAT=$READY_CHAT_ARG READY_PROMPT=$READY_PROMPT_ARG READY_TOKENS=$READY_TOKENS_ARG READY_TEMP=$READY_TEMP_ARG REMOTE_CONTEXT_LIMIT=$REMOTE_CONTEXT_LIMIT_ARG REMOTE_CONTEXT_PACKS=$REMOTE_CONTEXT_PACKS_ARG REMOTE_PREFILL_BATCH=$REMOTE_PREFILL_BATCH_ARG REMOTE_1B_QUANT=$REMOTE_1B_QUANT_ARG REMOTE_1B_QUANT_LOWER=$REMOTE_1B_QUANT_LOWER_ARG REMOTE_PREFILL_BENCH=$REMOTE_PREFILL_BENCH_ARG REMOTE_PREFILL_BENCH_LOWER=$REMOTE_PREFILL_BENCH_LOWER_ARG REMOTE_EVIDENCE=$REMOTE_EVIDENCE_ARG REMOTE_EVIDENCE_LOWER=$REMOTE_EVIDENCE_LOWER_ARG PREFILL_PROMPT=$PREFILL_PROMPT_ARG PREFILL_TOKENS=$PREFILL_TOKENS_ARG PREFILL_TEMP=$PREFILL_TEMP_ARG PREFILL_BATCHES=$PREFILL_BATCHES_ARG bash" << 'EOF'
  # Export Cargo path to make sure cargo commands work in non-interactive shells
  export PATH="$HOME/.cargo/bin:$PATH"
  if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
  fi

  # Source environment variables if they exist
  if [ -f "$PI_WORKSPACE/env.sh" ]; then
    source "$PI_WORKSPACE/env.sh"
  fi
  export CARGO_TARGET_DIR="${PI_TARGET_DIR:-/mnt/nanocamelid/target}"
  mkdir -p "$CARGO_TARGET_DIR"

  cd "$PI_REPO"

  # If bootstrap has not been run, run it to prepare workspace directories
  if [ ! -d "$PI_WORKSPACE/benchmarks" ] || [ ! -d "$CARGO_TARGET_DIR" ]; then
    chmod +x ./scripts/pi/bootstrap.sh
    NANOCAMELID_WORKSPACE="$PI_WORKSPACE" ./scripts/pi/bootstrap.sh
    if [ -f "$PI_WORKSPACE/env.sh" ]; then
      source "$PI_WORKSPACE/env.sh"
    fi
    export CARGO_TARGET_DIR="${PI_TARGET_DIR:-/mnt/nanocamelid/target}"
  fi

  echo "==> Cargo target dir: $CARGO_TARGET_DIR"

  echo "==> Checking format..."
  cargo fmt -- --check

  echo "==> Running tests..."
  cargo test

  echo "==> Running clippy..."
  cargo clippy --all-targets -- -D warnings

  echo "==> Running cargo check..."
  cargo check

  echo "==> Building release..."
  cargo build --release

  echo "==> Host CPU / feature probe:"
  cargo run -- probe

  echo "==> Checking 1B model audit dry-run:"
  cargo run -- model 1b --dry-run

  echo "==> Checking 1B readiness dry-run:"
  cargo run -- ready 1b --dry-run

  echo "==> Running benchmark (Q8 matrix dot product NEON/SDOT):"
  NANOCAMELID_Q8_DOT_SDOT=1 cargo run --release -- bench q8-dot 1000 3

  default_q4_model="$PI_WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf"
  default_q8_model="$PI_WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf"

  remote_quant_flag() {
    case "$REMOTE_1B_QUANT_LOWER" in
      q4 | q4_0) printf '%s\n' "--q4" ;;
      q8 | q8_0) printf '%s\n' "--q8" ;;
    esac
  }

  default_1b_selector_available() {
    [ -n "$(remote_quant_flag)" ] || [ -n "${NANOCAMELID_MODEL_GGUF:-}" ] || [ -f "$default_q4_model" ] || [ -f "$default_q8_model" ]
  }

  run_ready_1b() {
    ready_chat_lower="$(printf '%s' "$READY_CHAT" | tr '[:upper:]' '[:lower:]')"
    env_args=(
      "NANOCAMELID_WORKSPACE=$PI_WORKSPACE"
      "NANOCAMELID_REPO=$PI_REPO"
      "NANOCAMELID_READY_CHAT=$READY_CHAT"
      "NANOCAMELID_READY_SMOKE_KIND=$REMOTE_SMOKE_KIND"
      "NANOCAMELID_READY_SMOKE_PROMPT=$SMOKE_PROMPT"
      "NANOCAMELID_READY_SMOKE_TOKENS=$SMOKE_TOKENS"
    )
    if [ -n "$REMOTE_CONTEXT_LIMIT" ]; then
      env_args+=("NANOCAMELID_CONTEXT_LIMIT=$REMOTE_CONTEXT_LIMIT")
    fi
    if [ -n "$REMOTE_PREFILL_BATCH" ]; then
      env_args+=("NANOCAMELID_PREFILL_BATCH=$REMOTE_PREFILL_BATCH")
    fi
    case "$ready_chat_lower" in
      0 | false | no | off) ;;
      *)
        env_args+=(
          "NANOCAMELID_READY_PROMPT=$READY_PROMPT"
          "NANOCAMELID_READY_TOKENS=$READY_TOKENS"
          "NANOCAMELID_READY_TEMP=$READY_TEMP"
        )
        ;;
    esac

    if [ $# -gt 0 ]; then
      env "${env_args[@]}" ./scripts/pi/ready-1b.sh "$1"
    else
      env "${env_args[@]}" ./scripts/pi/ready-1b.sh
    fi
  }

  run_prefill_bench() {
    env_args=(
      "NANOCAMELID_WORKSPACE=$PI_WORKSPACE"
      "NANOCAMELID_REPO=$PI_REPO"
      "NANOCAMELID_PREFILL_PROMPT=$PREFILL_PROMPT"
      "NANOCAMELID_PREFILL_TOKENS=$PREFILL_TOKENS"
      "NANOCAMELID_PREFILL_TEMP=$PREFILL_TEMP"
      "NANOCAMELID_PREFILL_BATCHES=$PREFILL_BATCHES"
    )
    if [ -n "$REMOTE_CONTEXT_LIMIT" ]; then
      env_args+=("NANOCAMELID_CONTEXT_LIMIT=$REMOTE_CONTEXT_LIMIT")
    fi
    if [ -n "$REMOTE_PREFILL_BATCH" ]; then
      env_args+=("NANOCAMELID_PREFILL_BATCH=$REMOTE_PREFILL_BATCH")
    fi

    if [ $# -gt 0 ]; then
      env "${env_args[@]}" ./scripts/pi/bench-1b-prefill.sh "$1"
    else
      env "${env_args[@]}" ./scripts/pi/bench-1b-prefill.sh
    fi
  }

  run_context_pack() {
    env_args=(
      "NANOCAMELID_WORKSPACE=$PI_WORKSPACE"
      "NANOCAMELID_REPO=$PI_REPO"
      "NANOCAMELID_CONTEXT_PACKS=$REMOTE_CONTEXT_PACKS"
    )
    if [ -n "$REMOTE_PREFILL_BATCH" ]; then
      env_args+=("NANOCAMELID_PREFILL_BATCH=$REMOTE_PREFILL_BATCH")
    fi

    if [ $# -gt 0 ]; then
      env "${env_args[@]}" ./scripts/pi/context-pack-1b.sh "$1" "$REMOTE_SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS"
    else
      env "${env_args[@]}" ./scripts/pi/context-pack-1b.sh "$REMOTE_SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS"
    fi
  }

  run_evidence_bundle() {
    env_args=(
      "NANOCAMELID_WORKSPACE=$PI_WORKSPACE"
      "NANOCAMELID_REPO=$PI_REPO"
      "NANOCAMELID_SMOKE_KIND=$REMOTE_SMOKE_KIND"
      "NANOCAMELID_SMOKE_PROMPT=$SMOKE_PROMPT"
      "NANOCAMELID_SMOKE_TOKENS=$SMOKE_TOKENS"
      "NANOCAMELID_CONTEXT_PACKS=${REMOTE_CONTEXT_PACKS:-512,1024,2048,4096,8192}"
      "NANOCAMELID_PREFILL_PROMPT=$PREFILL_PROMPT"
      "NANOCAMELID_PREFILL_TOKENS=$PREFILL_TOKENS"
      "NANOCAMELID_PREFILL_TEMP=$PREFILL_TEMP"
      "NANOCAMELID_PREFILL_BATCHES=$PREFILL_BATCHES"
    )
    if [ -n "$REMOTE_CONTEXT_LIMIT" ]; then
      env_args+=("NANOCAMELID_CONTEXT_LIMIT=$REMOTE_CONTEXT_LIMIT")
    fi
    if [ -n "$REMOTE_PREFILL_BATCH" ]; then
      env_args+=("NANOCAMELID_PREFILL_BATCH=$REMOTE_PREFILL_BATCH")
    fi

    if [ $# -gt 0 ]; then
      env "${env_args[@]}" ./scripts/pi/evidence-1b.sh "$1"
    else
      env "${env_args[@]}" ./scripts/pi/evidence-1b.sh
    fi
  }

  is_disabled_toggle() {
    [ "$1" = "0" ] || [ "$1" = "false" ] || [ "$1" = "no" ] || [ "$1" = "off" ]
  }

  is_enabled_toggle() {
    [ "$1" = "1" ] || [ "$1" = "true" ] || [ "$1" = "yes" ] || [ "$1" = "on" ]
  }

  if is_disabled_toggle "$REMOTE_SMOKE_ENABLED_LOWER"; then
    echo "==> Skipping model-backed 1B readiness; NANOCAMELID_REMOTE_SMOKE=$REMOTE_SMOKE_ENABLED"
  elif is_enabled_toggle "$REMOTE_EVIDENCE_LOWER" && [ -n "$REMOTE_SMOKE_GGUF" ]; then
    echo "==> Running explicit 1B evidence bundle: $REMOTE_SMOKE_KIND"
    run_evidence_bundle "$REMOTE_SMOKE_GGUF"
  elif is_enabled_toggle "$REMOTE_EVIDENCE_LOWER" && default_1b_selector_available; then
    echo "==> Running default Pi-local 1B evidence bundle: $REMOTE_SMOKE_KIND"
    quant_flag="$(remote_quant_flag)"
    if [ -n "$quant_flag" ]; then
      run_evidence_bundle "$quant_flag"
    else
      run_evidence_bundle
    fi
  elif is_enabled_toggle "$REMOTE_EVIDENCE_LOWER"; then
    echo "==> Skipping model-backed 1B evidence bundle; no explicit GGUF path was set and no default Pi-local 1B model was found."
  elif [ -n "$REMOTE_SMOKE_GGUF" ]; then
    echo "==> Running explicit 1B readiness gate: $REMOTE_SMOKE_KIND"
    run_ready_1b "$REMOTE_SMOKE_GGUF"
    if [ -n "$REMOTE_CONTEXT_PACKS" ]; then
      echo "==> Running explicit 1B context-pack smoke gate: $REMOTE_CONTEXT_PACKS"
      run_context_pack "$REMOTE_SMOKE_GGUF"
    fi
    if is_enabled_toggle "$REMOTE_PREFILL_BENCH_LOWER"; then
      echo "==> Running explicit 1B prefill batch sweep: $PREFILL_BATCHES"
      run_prefill_bench "$REMOTE_SMOKE_GGUF"
    fi
  elif default_1b_selector_available; then
    echo "==> Running default Pi-local 1B readiness gate: $REMOTE_SMOKE_KIND"
    quant_flag="$(remote_quant_flag)"
    if [ -n "$quant_flag" ]; then
      run_ready_1b "$quant_flag"
    else
      run_ready_1b
    fi
    if [ -n "$REMOTE_CONTEXT_PACKS" ]; then
      echo "==> Running default Pi-local 1B context-pack smoke gate: $REMOTE_CONTEXT_PACKS"
      if [ -n "$quant_flag" ]; then
        run_context_pack "$quant_flag"
      else
        run_context_pack
      fi
    fi
    if is_enabled_toggle "$REMOTE_PREFILL_BENCH_LOWER"; then
      echo "==> Running default Pi-local 1B prefill batch sweep: $PREFILL_BATCHES"
      if [ -n "$quant_flag" ]; then
        run_prefill_bench "$quant_flag"
      else
        run_prefill_bench
      fi
    fi
  else
    echo "==> Skipping model-backed 1B readiness; no explicit GGUF path was set and no default Pi-local 1B model was found."
  fi
EOF
