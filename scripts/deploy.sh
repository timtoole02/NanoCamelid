#!/usr/bin/env bash
# scripts/deploy.sh - Deploy NanoCamelid codebase to Raspberry Pi
set -euo pipefail

PI_HOST="${1:-}"
SSH_KEY="${2:-${NANOCAMELID_SSH_KEY:-}}"
PI_USER="${3:-${NANOCAMELID_PI_USER:-$USER}}"
PI_WORKSPACE="${NANOCAMELID_REMOTE_WORKSPACE:-/mnt/nanocamelid}"
PI_REPO="$PI_WORKSPACE/src/NanoCamelid"

if [[ -z "$PI_HOST" ]]; then
  echo "Usage: $0 <pi-ip-or-hostname> [ssh-key-path] [pi-username]" >&2
  echo "Example: $0 192.168.1.100" >&2
  echo "         $0 pi5.local" >&2
  exit 1
fi

if [[ ! -f "$SSH_KEY" ]]; then
  if [[ -n "$SSH_KEY" ]]; then
    echo "Warning: configured SSH private key was not found; using default ssh agent." >&2
  fi
  SSH_OPTS=()
else
  SSH_OPTS=(-i "$SSH_KEY")
fi

# Derive repo root relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Connecting to ${PI_USER}@${PI_HOST} to check/create directories..."
ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} -o ConnectTimeout=5 "${PI_USER}@${PI_HOST}" "mkdir -p '$PI_REPO'"

echo "Syncing NanoCamelid folder via rsync from $REPO_ROOT..."
RSYNC_SSH=()
if [[ ${#SSH_OPTS[@]} -gt 0 ]]; then
  RSYNC_SSH=(-e "ssh -i $SSH_KEY")
fi
rsync -avz \
  --exclude 'target/' \
  --exclude '.git/' \
  --exclude '.cargo/' \
  --exclude '.openclaw/' \
  --exclude 'models/' \
  --exclude 'AGENTS.md' \
  --exclude 'HEARTBEAT.md' \
  --exclude 'IDENTITY.md' \
  --exclude 'SOUL.md' \
  --exclude 'TOOLS.md' \
  --exclude 'USER.md' \
  ${RSYNC_SSH[@]+"${RSYNC_SSH[@]}"} \
  "$REPO_ROOT/" \
  "${PI_USER}@${PI_HOST}:$PI_REPO"

echo "Synchronization complete!"
echo "Source files deployed to: $PI_REPO on the Pi"
