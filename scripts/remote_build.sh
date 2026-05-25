#!/usr/bin/env bash
# scripts/remote_build.sh - Compile and run tests/benchmarks on the Raspberry Pi remotely
set -euo pipefail

PI_HOST="${1:-}"
SSH_KEY="${2:-${NANOCAMELID_SSH_KEY:-}}"
PI_USER="${3:-${NANOCAMELID_PI_USER:-$USER}}"
DEPLOY_MODE="${4:-${NANOCAMELID_DEPLOY_MODE:-git-ff}}"
PI_WORKSPACE="${NANOCAMELID_REMOTE_WORKSPACE:-/mnt/nanocamelid}"
PI_REPO="$PI_WORKSPACE/src/NanoCamelid"
REMOTE_SMOKE_ENABLED="${NANOCAMELID_REMOTE_SMOKE:-1}"
REMOTE_SMOKE_GGUF="${NANOCAMELID_REMOTE_SMOKE_GGUF:-}"
REMOTE_SMOKE_KIND="${NANOCAMELID_REMOTE_SMOKE_KIND:-chat}"
SMOKE_PROMPT="${NANOCAMELID_SMOKE_PROMPT:-Say hello in one sentence.}"
SMOKE_TOKENS="${NANOCAMELID_SMOKE_TOKENS:-8}"

if [[ -z "$PI_HOST" ]]; then
  echo "Usage: $0 <pi-ip-or-hostname> [ssh-key-path] [pi-username]" >&2
  exit 1
fi

if [[ ! -f "$SSH_KEY" ]]; then
  SSH_OPTS=()
else
  SSH_OPTS=(-i "$SSH_KEY")
fi

echo "Deploying latest changes first..."
"$(dirname "$0")/deploy.sh" "$PI_HOST" "$SSH_KEY" "$PI_USER" "$DEPLOY_MODE"

echo "Building NanoCamelid on ${PI_USER}@${PI_HOST}..."
printf -v REMOTE_PI_WORKSPACE '%q' "$PI_WORKSPACE"
printf -v REMOTE_PI_REPO '%q' "$PI_REPO"
printf -v REMOTE_SMOKE_ENABLED_ARG '%q' "$REMOTE_SMOKE_ENABLED"
printf -v REMOTE_SMOKE_GGUF_ARG '%q' "$REMOTE_SMOKE_GGUF"
printf -v REMOTE_SMOKE_KIND_ARG '%q' "$REMOTE_SMOKE_KIND"
printf -v REMOTE_SMOKE_PROMPT_ARG '%q' "$SMOKE_PROMPT"
printf -v REMOTE_SMOKE_TOKENS_ARG '%q' "$SMOKE_TOKENS"
ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "${PI_USER}@${PI_HOST}" \
  "PI_WORKSPACE=$REMOTE_PI_WORKSPACE PI_REPO=$REMOTE_PI_REPO REMOTE_SMOKE_ENABLED=$REMOTE_SMOKE_ENABLED_ARG REMOTE_SMOKE_GGUF=$REMOTE_SMOKE_GGUF_ARG REMOTE_SMOKE_KIND=$REMOTE_SMOKE_KIND_ARG SMOKE_PROMPT=$REMOTE_SMOKE_PROMPT_ARG SMOKE_TOKENS=$REMOTE_SMOKE_TOKENS_ARG bash" << 'EOF'
  # Export Cargo path to make sure cargo commands work in non-interactive shells
  export PATH="$HOME/.cargo/bin:$PATH"
  if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
  fi

  # Source environment variables if they exist
  if [ -f "$PI_WORKSPACE/env.sh" ]; then
    source "$PI_WORKSPACE/env.sh"
  fi
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/mnt/nanocamelid/target}"
  mkdir -p "$CARGO_TARGET_DIR"

  cd "$PI_REPO"

  # If bootstrap has not been run, run it to prepare workspace directories
  if [ ! -d "$PI_WORKSPACE/benchmarks" ] || [ ! -d "$CARGO_TARGET_DIR" ]; then
    chmod +x ./scripts/pi/bootstrap.sh
    NANOCAMELID_WORKSPACE="$PI_WORKSPACE" ./scripts/pi/bootstrap.sh
    if [ -f "$PI_WORKSPACE/env.sh" ]; then
      source "$PI_WORKSPACE/env.sh"
    fi
    export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/mnt/nanocamelid/target}"
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

  if [ -n "$REMOTE_SMOKE_GGUF" ]; then
    echo "==> Inspecting explicit 1B smoke model:"
    cargo run --release -- inspect "$REMOTE_SMOKE_GGUF"
  elif [ -n "${NANOCAMELID_MODEL_GGUF:-}" ] || [ -f "$default_q4_model" ] || [ -f "$default_q8_model" ]; then
    echo "==> Inspecting default Pi-local 1B model:"
    NANOCAMELID_WORKSPACE="$PI_WORKSPACE" cargo run --release -- inspect 1b
  else
    echo "==> Skipping 1B inspect; no explicit GGUF path was set and no default Pi-local 1B model was found."
  fi

  case "$REMOTE_SMOKE_KIND" in
    chat) generic_smoke_kind="q8-chat" ;;
    model) generic_smoke_kind="q8-model" ;;
    q8-chat|q8-model) generic_smoke_kind="$REMOTE_SMOKE_KIND" ;;
    *)
      echo "Unknown smoke kind: $REMOTE_SMOKE_KIND" >&2
      echo "Expected chat, model, q8-chat, or q8-model." >&2
      exit 2
      ;;
  esac

  if [ "$REMOTE_SMOKE_ENABLED" = "0" ]; then
    echo "==> Skipping model-backed smoke; NANOCAMELID_REMOTE_SMOKE=0"
  elif [ -n "$REMOTE_SMOKE_GGUF" ]; then
    echo "==> Running model-backed smoke: $REMOTE_SMOKE_KIND"
    NANOCAMELID_Q8_DOT_KERNEL="${NANOCAMELID_Q8_DOT_KERNEL:-neon}" \
      NANOCAMELID_Q8_DOT_SDOT="${NANOCAMELID_Q8_DOT_SDOT:-1}" \
      cargo run --release -- smoke "$generic_smoke_kind" "$REMOTE_SMOKE_GGUF" "$SMOKE_PROMPT" "$SMOKE_TOKENS"
  elif [ -n "${NANOCAMELID_MODEL_GGUF:-}" ] || [ -f "$default_q4_model" ] || [ -f "$default_q8_model" ]; then
    echo "==> Running default Pi-local 1B smoke: $REMOTE_SMOKE_KIND"
    NANOCAMELID_WORKSPACE="$PI_WORKSPACE" \
      NANOCAMELID_REPO="$PI_REPO" \
      NANOCAMELID_SMOKE_KIND="$REMOTE_SMOKE_KIND" \
      NANOCAMELID_SMOKE_PROMPT="$SMOKE_PROMPT" \
      NANOCAMELID_SMOKE_TOKENS="$SMOKE_TOKENS" \
      ./scripts/pi/smoke-1b.sh
  else
    echo "==> Skipping model-backed smoke; no explicit GGUF path was set and no default Pi-local 1B model was found."
  fi
EOF
