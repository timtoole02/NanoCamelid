#!/usr/bin/env bash
# scripts/remote_build.sh - Compile and run tests/benchmarks on the Raspberry Pi remotely
set -euo pipefail

PI_HOST="${1:-}"
_SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$_SD/../config/cluster.env" ]] && source "$_SD/../config/cluster.env"
SSH_KEY="${2:-${SSH_KEY:-$HOME/.ssh/id_ed25519}}"
PI_USER="${3:-${PI_USER:-pi}}"

if [[ -z "$PI_HOST" ]]; then
  echo "Usage: $0 <pi-ip-or-hostname> [ssh-key-path] [pi-username]" >&2
  exit 1
fi

if [[ ! -f "$SSH_KEY" ]]; then
  SSH_OPT=""
else
  SSH_OPT="-o IdentitiesOnly=yes -i $SSH_KEY"
fi

echo "Deploying latest changes first..."
"$(dirname "$0")/deploy.sh" "$PI_HOST" "$SSH_KEY" "$PI_USER"

echo "Building NanoCamelid on ${PI_USER}@${PI_HOST}..."
ssh ${SSH_OPT} "${PI_USER}@${PI_HOST}" bash << 'EOF'
  # Export Cargo path to make sure cargo commands work in non-interactive shells
  export PATH="$HOME/.cargo/bin:$PATH"
  if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
  fi

  # Source environment variables if they exist
  if [ -f ~/nanocamelid/env.sh ]; then
    source ~/nanocamelid/env.sh
  fi

  cd ~/nanocamelid/src/NanoCamelid

  # If bootstrap has not been run, run it to prepare workspace directories
  if [ ! -d ../../benchmarks ]; then
    chmod +x ./scripts/pi/bootstrap.sh
    ./scripts/pi/bootstrap.sh
    source ~/nanocamelid/env.sh
  fi

  echo "==> Running cargo check..."
  cargo check

  echo "==> Building release..."
  cargo build --release

  echo "==> Host CPU / feature probe:"
  cargo run -- probe

  echo "==> Running benchmark (Q8 matrix dot product NEON/SDOT):"
  NANOCAMELID_Q8_DOT_SDOT=1 cargo run --release -- bench q8-dot 1000 3
EOF
