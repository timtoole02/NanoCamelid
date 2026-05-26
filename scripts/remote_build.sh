#!/usr/bin/env bash
# scripts/remote_build.sh - Compile and run tests/benchmarks on the Raspberry Pi remotely
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage: remote_build.sh <pi-ip-or-hostname> [ssh-key-path] [pi-username] [rsync|git-ff] [--dry-run]

Builds and validates NanoCamelid on a Raspberry Pi workspace.

Options:
  --dry-run   Print the resolved deploy/build/readiness plan without SSH or deploy
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
PI_TARGET_DIR="/mnt/nanocamelid/target"
PI_REPO="$PI_WORKSPACE/src/NanoCamelid"
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

if [[ -z "$PI_HOST" ]]; then
  usage
  exit 1
fi

case "$REMOTE_SMOKE_KIND" in
  chat | model | q8-chat | q8-model) ;;
  *)
    echo "Unknown smoke kind: $REMOTE_SMOKE_KIND" >&2
    echo "Expected chat, model, q8-chat, or q8-model." >&2
    exit 2
    ;;
esac

if [[ ! -f "$SSH_KEY" ]]; then
  SSH_OPTS=()
else
  SSH_OPTS=(-i "$SSH_KEY")
fi

shell_quote() {
  printf '%q' "$1"
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

redacted_deploy_key_label() {
  if [[ -n "$SSH_KEY" ]]; then
    echo "<ssh-key-path>"
  else
    echo "<ssh-agent>"
  fi
}

if [[ "$REMOTE_SMOKE_ENABLED_LOWER" != "0" && "$REMOTE_SMOKE_ENABLED_LOWER" != "false" && "$REMOTE_SMOKE_ENABLED_LOWER" != "no" ]]; then
  require_positive_integer "Smoke token count" "$SMOKE_TOKENS"
  if [[ "$READY_CHAT_LOWER" != "0" && "$READY_CHAT_LOWER" != "false" && "$READY_CHAT_LOWER" != "no" ]]; then
    require_positive_integer "Readiness token count" "$READY_TOKENS"
    require_non_negative_float "Readiness temperature" "$READY_TEMP"
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "NanoCamelid remote build dry run"
  echo "target: <pi-user>@<pi-host>"
  echo "target_redacted: true"
  echo "deploy_mode: $DEPLOY_MODE"
  echo "remote_workspace: $PI_WORKSPACE"
  echo "remote_repo: $PI_REPO"
  echo "cargo_target_dir: $PI_TARGET_DIR"
  echo "remote_smoke_enabled: $REMOTE_SMOKE_ENABLED"
  echo "remote_smoke_kind: $REMOTE_SMOKE_KIND"
  echo "smoke_prompt: $SMOKE_PROMPT"
  echo "smoke_tokens: $SMOKE_TOKENS"
  echo "ready_chat: $READY_CHAT"
  echo "ready_prompt: $READY_PROMPT"
  echo "ready_tokens: $READY_TOKENS"
  echo "ready_temp: $READY_TEMP"
  printf 'deploy_command: scripts/deploy.sh %s %s %s %s\n' \
    "<pi-host>" \
    "$(redacted_deploy_key_label)" \
    "<pi-user>" \
    "$(shell_quote "$DEPLOY_MODE")"
  echo "remote_steps: cargo fmt -- --check; cargo test; cargo clippy --all-targets -- -D warnings; cargo check; cargo build --release; probe; bench q8-dot 1000 3"
  if [[ "$REMOTE_SMOKE_ENABLED_LOWER" == "0" || "$REMOTE_SMOKE_ENABLED_LOWER" == "false" || "$REMOTE_SMOKE_ENABLED_LOWER" == "no" ]]; then
    echo "readiness_command: skipped"
  elif [[ -n "$REMOTE_SMOKE_GGUF" ]]; then
    printf 'readiness_command: NANOCAMELID_READY_CHAT=%s NANOCAMELID_READY_SMOKE_KIND=%s NANOCAMELID_READY_SMOKE_PROMPT=%s NANOCAMELID_READY_SMOKE_TOKENS=%s NANOCAMELID_READY_PROMPT=%s NANOCAMELID_READY_TOKENS=%s NANOCAMELID_READY_TEMP=%s ./scripts/pi/ready-1b.sh %s\n' \
      "$(shell_quote "$READY_CHAT")" \
      "$(shell_quote "$REMOTE_SMOKE_KIND")" \
      "$(shell_quote "$SMOKE_PROMPT")" \
      "$(shell_quote "$SMOKE_TOKENS")" \
      "$(shell_quote "$READY_PROMPT")" \
      "$(shell_quote "$READY_TOKENS")" \
      "$(shell_quote "$READY_TEMP")" \
      "$(shell_quote "$REMOTE_SMOKE_GGUF")"
  else
    printf 'readiness_command: NANOCAMELID_READY_CHAT=%s NANOCAMELID_READY_SMOKE_KIND=%s NANOCAMELID_READY_SMOKE_PROMPT=%s NANOCAMELID_READY_SMOKE_TOKENS=%s NANOCAMELID_READY_PROMPT=%s NANOCAMELID_READY_TOKENS=%s NANOCAMELID_READY_TEMP=%s ./scripts/pi/ready-1b.sh\n' \
      "$(shell_quote "$READY_CHAT")" \
      "$(shell_quote "$REMOTE_SMOKE_KIND")" \
      "$(shell_quote "$SMOKE_PROMPT")" \
      "$(shell_quote "$SMOKE_TOKENS")" \
      "$(shell_quote "$READY_PROMPT")" \
      "$(shell_quote "$READY_TOKENS")" \
      "$(shell_quote "$READY_TEMP")"
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
ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "${PI_USER}@${PI_HOST}" \
  "PI_WORKSPACE=$REMOTE_PI_WORKSPACE PI_TARGET_DIR=$REMOTE_PI_TARGET_DIR PI_REPO=$REMOTE_PI_REPO REMOTE_SMOKE_ENABLED=$REMOTE_SMOKE_ENABLED_ARG REMOTE_SMOKE_ENABLED_LOWER=$REMOTE_SMOKE_ENABLED_LOWER_ARG REMOTE_SMOKE_GGUF=$REMOTE_SMOKE_GGUF_ARG REMOTE_SMOKE_KIND=$REMOTE_SMOKE_KIND_ARG SMOKE_PROMPT=$REMOTE_SMOKE_PROMPT_ARG SMOKE_TOKENS=$REMOTE_SMOKE_TOKENS_ARG READY_CHAT=$READY_CHAT_ARG READY_PROMPT=$READY_PROMPT_ARG READY_TOKENS=$READY_TOKENS_ARG READY_TEMP=$READY_TEMP_ARG bash" << 'EOF'
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

  echo "==> Running benchmark (Q8 matrix dot product NEON/SDOT):"
  NANOCAMELID_Q8_DOT_SDOT=1 cargo run --release -- bench q8-dot 1000 3

  default_q4_model="$PI_WORKSPACE/models/Llama-3.2-1B-Instruct-Q4_0.gguf"
  default_q8_model="$PI_WORKSPACE/models/Llama-3.2-1B-Instruct-Q8_0.gguf"

  if [ "$REMOTE_SMOKE_ENABLED_LOWER" = "0" ] || [ "$REMOTE_SMOKE_ENABLED_LOWER" = "false" ] || [ "$REMOTE_SMOKE_ENABLED_LOWER" = "no" ]; then
    echo "==> Skipping model-backed 1B readiness; NANOCAMELID_REMOTE_SMOKE=$REMOTE_SMOKE_ENABLED"
  elif [ -n "$REMOTE_SMOKE_GGUF" ]; then
    echo "==> Running explicit 1B readiness gate: $REMOTE_SMOKE_KIND"
    NANOCAMELID_WORKSPACE="$PI_WORKSPACE" \
      NANOCAMELID_REPO="$PI_REPO" \
      NANOCAMELID_READY_CHAT="$READY_CHAT" \
      NANOCAMELID_READY_SMOKE_KIND="$REMOTE_SMOKE_KIND" \
      NANOCAMELID_READY_SMOKE_PROMPT="$SMOKE_PROMPT" \
      NANOCAMELID_READY_SMOKE_TOKENS="$SMOKE_TOKENS" \
      NANOCAMELID_READY_PROMPT="$READY_PROMPT" \
      NANOCAMELID_READY_TOKENS="$READY_TOKENS" \
      NANOCAMELID_READY_TEMP="$READY_TEMP" \
      ./scripts/pi/ready-1b.sh "$REMOTE_SMOKE_GGUF"
  elif [ -n "${NANOCAMELID_MODEL_GGUF:-}" ] || [ -f "$default_q4_model" ] || [ -f "$default_q8_model" ]; then
    echo "==> Running default Pi-local 1B readiness gate: $REMOTE_SMOKE_KIND"
    NANOCAMELID_WORKSPACE="$PI_WORKSPACE" \
      NANOCAMELID_REPO="$PI_REPO" \
      NANOCAMELID_READY_CHAT="$READY_CHAT" \
      NANOCAMELID_READY_SMOKE_KIND="$REMOTE_SMOKE_KIND" \
      NANOCAMELID_READY_SMOKE_PROMPT="$SMOKE_PROMPT" \
      NANOCAMELID_READY_SMOKE_TOKENS="$SMOKE_TOKENS" \
      NANOCAMELID_READY_PROMPT="$READY_PROMPT" \
      NANOCAMELID_READY_TOKENS="$READY_TOKENS" \
      NANOCAMELID_READY_TEMP="$READY_TEMP" \
      ./scripts/pi/ready-1b.sh
  else
    echo "==> Skipping model-backed 1B readiness; no explicit GGUF path was set and no default Pi-local 1B model was found."
  fi
EOF
