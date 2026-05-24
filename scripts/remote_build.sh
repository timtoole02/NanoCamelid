#!/usr/bin/env bash
# scripts/remote_build.sh - Compile and run tests/benchmarks on the Raspberry Pi remotely
set -euo pipefail

PI_HOST="${1:-}"
SSH_KEY="${2:-${NANOCAMELID_SSH_KEY:-}}"
PI_USER="${3:-${NANOCAMELID_PI_USER:-$USER}}"
DEPLOY_MODE="${4:-${NANOCAMELID_DEPLOY_MODE:-rsync}}"
PI_WORKSPACE="${NANOCAMELID_REMOTE_WORKSPACE:-/mnt/nanocamelid}"
PI_REPO="$PI_WORKSPACE/src/NanoCamelid"
REMOTE_SMOKE_GGUF="${NANOCAMELID_REMOTE_SMOKE_GGUF:-}"
SMOKE_PROMPT="${NANOCAMELID_SMOKE_PROMPT:-Hello}"
SMOKE_TOKENS="${NANOCAMELID_SMOKE_TOKENS:-1}"

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
printf -v REMOTE_SMOKE_GGUF_ARG '%q' "$REMOTE_SMOKE_GGUF"
printf -v REMOTE_SMOKE_PROMPT_ARG '%q' "$SMOKE_PROMPT"
printf -v REMOTE_SMOKE_TOKENS_ARG '%q' "$SMOKE_TOKENS"
ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "${PI_USER}@${PI_HOST}" \
  "PI_WORKSPACE=$REMOTE_PI_WORKSPACE PI_REPO=$REMOTE_PI_REPO REMOTE_SMOKE_GGUF=$REMOTE_SMOKE_GGUF_ARG SMOKE_PROMPT=$REMOTE_SMOKE_PROMPT_ARG SMOKE_TOKENS=$REMOTE_SMOKE_TOKENS_ARG bash" << 'EOF'
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

  if [ -n "$REMOTE_SMOKE_GGUF" ]; then
    echo "==> Running model-backed Q8_0 smoke:"
    NANOCAMELID_Q8_DOT_KERNEL="${NANOCAMELID_Q8_DOT_KERNEL:-neon}" \
      NANOCAMELID_Q8_DOT_SDOT="${NANOCAMELID_Q8_DOT_SDOT:-1}" \
      cargo run --release -- smoke q8-model "$REMOTE_SMOKE_GGUF" "$SMOKE_PROMPT" "$SMOKE_TOKENS"
  else
    echo "==> Skipping model-backed Q8_0 smoke; set NANOCAMELID_REMOTE_SMOKE_GGUF to a Pi-local GGUF path."
  fi
EOF
