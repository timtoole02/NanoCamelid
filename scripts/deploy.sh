#!/usr/bin/env bash
# scripts/deploy.sh - Deploy NanoCamelid codebase to Raspberry Pi
set -euo pipefail

PI_HOST="${1:-}"
SSH_KEY="${2:-${NANOCAMELID_SSH_KEY:-}}"
PI_USER="${3:-${NANOCAMELID_PI_USER:-$USER}}"
DEPLOY_MODE="${4:-${NANOCAMELID_DEPLOY_MODE:-rsync}}"
PI_WORKSPACE="${NANOCAMELID_REMOTE_WORKSPACE:-/mnt/nanocamelid}"
PI_REPO="$PI_WORKSPACE/src/NanoCamelid"
PUBLIC_REPO_URL="${NANOCAMELID_PUBLIC_REPO_URL:-https://github.com/timtoole02/NanoCamelid.git}"
PI_BRANCH="${NANOCAMELID_REMOTE_BRANCH:-main}"

if [[ -z "$PI_HOST" ]]; then
  echo "Usage: $0 <pi-ip-or-hostname> [ssh-key-path] [pi-username] [rsync|git-ff]" >&2
  echo "Example: $0 192.168.1.100" >&2
  echo "         $0 pi5.local" >&2
  echo "         NANOCAMELID_DEPLOY_MODE=git-ff $0 pi5.local" >&2
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

echo "Connecting to target Pi to check/create directories..."
ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} -o ConnectTimeout=5 "${PI_USER}@${PI_HOST}" "mkdir -p '$PI_REPO'"

if [[ "$DEPLOY_MODE" == "git-ff" ]]; then
  echo "Updating Pi checkout with clean git fast-forward mode..."
  printf -v REMOTE_PI_REPO '%q' "$PI_REPO"
  printf -v REMOTE_PUBLIC_REPO_URL '%q' "$PUBLIC_REPO_URL"
  printf -v REMOTE_PI_BRANCH '%q' "$PI_BRANCH"
  ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} -o ConnectTimeout=5 "${PI_USER}@${PI_HOST}" \
    "PI_REPO=$REMOTE_PI_REPO PUBLIC_REPO_URL=$REMOTE_PUBLIC_REPO_URL PI_BRANCH=$REMOTE_PI_BRANCH bash" <<'EOF'
set -euo pipefail

repo_parent="$(dirname "$PI_REPO")"
mkdir -p "$repo_parent"

if [[ ! -d "$PI_REPO/.git" ]]; then
  if [[ -d "$PI_REPO" ]] && [[ -n "$(find "$PI_REPO" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing git-ff update: $PI_REPO exists but is not a git checkout." >&2
    echo "Move it aside, empty it, or use rsync mode for snapshot deployment." >&2
    exit 3
  fi

  rm -rf "$PI_REPO"
  git clone --branch "$PI_BRANCH" "$PUBLIC_REPO_URL" "$PI_REPO"
  cd "$PI_REPO"
  git status --short --branch
  exit 0
fi

cd "$PI_REPO"

if [[ "$(git config --get remote.origin.url || true)" != "$PUBLIC_REPO_URL" ]]; then
  echo "Refusing git-ff update: remote.origin.url does not match public repo URL." >&2
  exit 4
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing git-ff update: checkout has local changes or untracked files." >&2
  git status --short
  exit 5
fi

git fetch --prune origin "$PI_BRANCH"

if git show-ref --verify --quiet "refs/heads/$PI_BRANCH"; then
  git checkout "$PI_BRANCH"
else
  git checkout -b "$PI_BRANCH" "origin/$PI_BRANCH"
fi

local_head="$(git rev-parse HEAD)"
remote_head="$(git rev-parse "origin/$PI_BRANCH")"
merge_base="$(git merge-base HEAD "origin/$PI_BRANCH")"

if [[ "$local_head" != "$merge_base" ]]; then
  echo "Refusing git-ff update: local branch is not an ancestor of origin/$PI_BRANCH." >&2
  exit 6
fi

if [[ "$local_head" != "$remote_head" ]]; then
  git merge --ff-only "origin/$PI_BRANCH"
fi

git status --short --branch
EOF
  echo "Git fast-forward update complete: $PI_REPO on the Pi"
  exit 0
elif [[ "$DEPLOY_MODE" != "rsync" ]]; then
  echo "Unknown deploy mode: $DEPLOY_MODE" >&2
  echo "Expected rsync or git-ff." >&2
  exit 2
fi

echo "Syncing NanoCamelid folder via rsync from $REPO_ROOT to target Pi..."
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
echo "Source files deployed to Pi workspace."
