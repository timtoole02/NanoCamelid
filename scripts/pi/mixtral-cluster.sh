#!/usr/bin/env bash
# Launch or print NanoCamelid's supported Mixtral three-Pi cluster plan.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: mixtral-cluster.sh plan|final|middle|master [model.gguf] [--dry-run]

Runs the exact Mixtral 8x7B Instruct v0.1 Q4_0 three-Pi cluster path.

Layer split:
  master: 0..11
  middle worker: 11..22
  final worker: 22..32

Useful env:
  NANOCAMELID_WORKSPACE                 Pi workspace, default /mnt/nanocamelid
  CARGO_TARGET_DIR                      Cargo output dir, default /mnt/nanocamelid/target
  NANOCAMELID_MIXTRAL_GGUF              Explicit Mixtral GGUF path
  NANOCAMELID_CLUSTER_FINAL_BIND        Final worker bind addr, default 0.0.0.0:5007
  NANOCAMELID_CLUSTER_MIDDLE_BIND       Middle worker bind addr, default 0.0.0.0:5006
  NANOCAMELID_CLUSTER_FINAL_ADDR        Final worker host:port, required by middle
  NANOCAMELID_CLUSTER_MIDDLE_ADDR       Middle worker host:port, required by master
  NANOCAMELID_CLUSTER_PROMPT            Master chat prompt
  NANOCAMELID_CLUSTER_TOKENS            Master generated token count, default 8
  NANOCAMELID_CLUSTER_CONTEXT_LIMIT     Runtime context cap, default 128
  --dry-run                             Print commands without running
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

MODE="${1:-plan}"
case "$MODE" in
  plan | final | middle | master)
    shift || true
    ;;
  *)
    echo "Unknown Mixtral cluster mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}"
REPO="${NANOCAMELID_REPO:-$REPO_ROOT}"
TARGET_DIR="${CARGO_TARGET_DIR:-${NANOCAMELID_TARGET_DIR:-/mnt/nanocamelid/target}}"
MODEL="${NANOCAMELID_MIXTRAL_GGUF:-$WORKSPACE/models/mixtral-8x7b-instruct-v0.1.Q4_0.gguf}"
if looks_like_gguf_path "${1:-}"; then
  MODEL="$1"
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Unexpected extra Mixtral cluster argument: $1" >&2
  usage >&2
  exit 2
fi

FINAL_BIND="${NANOCAMELID_CLUSTER_FINAL_BIND:-0.0.0.0:5007}"
MIDDLE_BIND="${NANOCAMELID_CLUSTER_MIDDLE_BIND:-0.0.0.0:5006}"
FINAL_ADDR="${NANOCAMELID_CLUSTER_FINAL_ADDR:-}"
MIDDLE_ADDR="${NANOCAMELID_CLUSTER_MIDDLE_ADDR:-}"
PROMPT="${NANOCAMELID_CLUSTER_PROMPT:-Write one short sentence about Raspberry Pi clusters.}"
TOKENS="${NANOCAMELID_CLUSTER_TOKENS:-8}"
CONTEXT_LIMIT="${NANOCAMELID_CLUSTER_CONTEXT_LIMIT:-128}"
BINARY="${NANOCAMELID_BIN:-$TARGET_DIR/release/cluster_tcp_smoke}"
export NANOCAMELID_CLUSTER_CONTEXT_LIMIT="$CONTEXT_LIMIT"

if ! is_positive_integer "$TOKENS"; then
  echo "NANOCAMELID_CLUSTER_TOKENS must be a positive integer: $TOKENS" >&2
  exit 2
fi
if ! is_positive_integer "$CONTEXT_LIMIT"; then
  echo "NANOCAMELID_CLUSTER_CONTEXT_LIMIT must be a positive integer: $CONTEXT_LIMIT" >&2
  exit 2
fi

if [[ -x "$BINARY" ]]; then
  launcher_mode="binary"
elif command -v cargo >/dev/null 2>&1; then
  launcher_mode="cargo"
else
  launcher_mode="unavailable"
fi

run_cluster() {
  if [[ "$launcher_mode" == "binary" ]]; then
    exec "$BINARY" "$@"
  fi

  cd "$REPO"
  export CARGO_TARGET_DIR="$TARGET_DIR"
  exec cargo run --release --bin cluster_tcp_smoke -- "$@"
}

print_plan() {
  echo "NanoCamelid Mixtral cluster plan"
  echo "repo: $REPO"
  echo "cargo_target_dir: $TARGET_DIR"
  echo "launcher_mode: $launcher_mode"
  echo "binary: $BINARY"
  echo "model: $MODEL"
  echo "model_exists: $([[ -f "$MODEL" ]] && echo true || echo false)"
  echo "context_limit: $CONTEXT_LIMIT"
  echo "split: master 0..11 | middle 11..22 | final 22..32"
  echo "final_bind: $FINAL_BIND"
  echo "middle_bind: $MIDDLE_BIND"
  echo "final_addr: ${FINAL_ADDR:-<required-for-middle>}"
  echo "middle_addr: ${MIDDLE_ADDR:-<required-for-master>}"
  echo "prompt: $PROMPT"
  echo "tokens: $TOKENS"
  printf 'final_command: '
  shell_command "$0" final "$MODEL"
  printf 'middle_command: '
  shell_command "$0" middle "$MODEL"
  printf 'master_command: '
  shell_command "$0" master "$MODEL"
}

if [[ "$DRY_RUN" == "1" || "$MODE" == "plan" ]]; then
  print_plan
  exit 0
fi

if [[ "$launcher_mode" == "unavailable" ]]; then
  echo "cluster_tcp_smoke release binary not found and cargo is not on PATH." >&2
  echo "Expected binary: $BINARY" >&2
  exit 3
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Mixtral model not found: $MODEL" >&2
  echo "Set NANOCAMELID_MIXTRAL_GGUF=/path/to/mixtral.gguf or place the GGUF at $WORKSPACE/models/mixtral-8x7b-instruct-v0.1.Q4_0.gguf." >&2
  exit 2
fi

case "$MODE" in
  final)
    run_cluster worker "$MODEL" "$FINAL_BIND" 22
    ;;
  middle)
    if [[ -z "$FINAL_ADDR" ]]; then
      echo "NANOCAMELID_CLUSTER_FINAL_ADDR is required for middle mode." >&2
      exit 2
    fi
    run_cluster middle-worker "$MODEL" "$MIDDLE_BIND" "$FINAL_ADDR" 11 22
    ;;
  master)
    if [[ -z "$MIDDLE_ADDR" ]]; then
      echo "NANOCAMELID_CLUSTER_MIDDLE_ADDR is required for master mode." >&2
      exit 2
    fi
    run_cluster master-chat "$MODEL" "$MIDDLE_ADDR" "$PROMPT" 11 "$TOKENS"
    ;;
esac
