#!/usr/bin/env bash
# scripts/stop_cluster.sh - Stop all NanoCamelid pipeline stage servers.
# Usage: ./scripts/stop_cluster.sh
# Env overrides: NANOCAMELID_CLUSTER_CONFIG, PI_USER, SSH_KEY
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

[[ -f "$REPO_ROOT/config/cluster.env" ]] && source "$REPO_ROOT/config/cluster.env"
if [[ -n "${NANOCAMELID_CLUSTER_CONFIG:-}" ]]; then
  CONFIG="$NANOCAMELID_CLUSTER_CONFIG"
elif [[ -f "$REPO_ROOT/config/nodes.local.toml" ]]; then
  CONFIG="$REPO_ROOT/config/nodes.local.toml"
else
  CONFIG="$REPO_ROOT/config/nodes.toml"
fi
PI_USER="${PI_USER:-pi}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

names=($(awk -F'"' '/^name[[:space:]]*=/ {print $2}' "$CONFIG"))
hosts=($(awk -F'"' '/^host[[:space:]]*=/ {print $2}' "$CONFIG"))

if [[ -f "$SSH_KEY" ]]; then
  SSH_OPT="-o IdentitiesOnly=yes -i $SSH_KEY"
else
  SSH_OPT=""
fi

for i in "${!names[@]}"; do
  [[ "$i" == "0" ]] && continue
  echo "--- stopping serve-stage on ${names[$i]} (${hosts[$i]}) ---"
  ssh ${SSH_OPT} "${PI_USER}@${hosts[$i]}" \
    'pkill -f "nanocamelid serve-stage" && echo stopped || echo "nothing running"'
done
