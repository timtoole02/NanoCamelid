#!/usr/bin/env bash
# scripts/deploy.sh - Deploy NanoCamelid codebase to Raspberry Pi
set -euo pipefail

PI_HOST="${1:-}"
# Optional local credentials (gitignored): config/cluster.env may set PI_USER / SSH_KEY.
_SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$_SD/../config/cluster.env" ]] && source "$_SD/../config/cluster.env"
SSH_KEY="${2:-${SSH_KEY:-$HOME/.ssh/id_ed25519}}"
PI_USER="${3:-${PI_USER:-pi}}"

if [[ -z "$PI_HOST" ]]; then
  echo "Usage: $0 <pi-ip-or-hostname> [ssh-key-path] [pi-username]" >&2
  echo "Example: $0 192.168.1.100" >&2
  echo "         $0 pi5.local" >&2
  exit 1
fi

if [[ ! -f "$SSH_KEY" ]]; then
  echo "Warning: SSH private key not found at $SSH_KEY" >&2
  echo "Will attempt connection using default ssh key agent." >&2
  SSH_OPT=""
else
  SSH_OPT="-o IdentitiesOnly=yes -i $SSH_KEY"
fi

# Derive repo root relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Connecting to ${PI_USER}@${PI_HOST} to check/create directories..."
ssh ${SSH_OPT} -o ConnectTimeout=5 "${PI_USER}@${PI_HOST}" "mkdir -p ~/nanocamelid/src"

echo "Syncing NanoCamelid folder via rsync from $REPO_ROOT..."
rsync -avz \
  --exclude 'target/' \
  --exclude '.git/' \
  --exclude 'models/' \
  --exclude 'config/cluster.env' \
  ${SSH_OPT:+-e "ssh $SSH_OPT"} \
  "$REPO_ROOT/" \
  "${PI_USER}@${PI_HOST}:~/nanocamelid/src/NanoCamelid"

echo "Synchronization complete!"
echo "Source files deployed to: ~/nanocamelid/src/NanoCamelid on the Pi"
