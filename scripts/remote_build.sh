#!/usr/bin/env bash
# scripts/remote_build.sh - Compile and run tests/benchmarks on the Raspberry Pi remotely
set -euo pipefail

PI_HOST="${1:-}"
SSH_KEY="${2:-${NANOCAMELID_SSH_KEY:-}}"
PI_USER="${3:-${NANOCAMELID_PI_USER:-$USER}}"
PI_WORKSPACE="${NANOCAMELID_REMOTE_WORKSPACE:-/mnt/nanocamelid}"
PI_REPO="$PI_WORKSPACE/src/NanoCamelid"

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
"$(dirname "$0")/deploy.sh" "$PI_HOST" "$SSH_KEY" "$PI_USER"

echo "Building NanoCamelid on ${PI_USER}@${PI_HOST}..."
ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "${PI_USER}@${PI_HOST}" \
  "PI_WORKSPACE='$PI_WORKSPACE' PI_REPO='$PI_REPO' bash" << 'EOF'
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
    ./scripts/pi/bootstrap.sh
    source "$PI_WORKSPACE/env.sh"
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
EOF
