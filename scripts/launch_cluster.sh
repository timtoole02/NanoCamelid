#!/usr/bin/env bash
# scripts/launch_cluster.sh - Deploy, build, and launch the 3-node NanoCamelid pipeline.
#
# Reads the node list from config/nodes.toml (order = stage order). For each node it
# deploys the source (scripts/deploy.sh) and builds in release mode; then it starts
# `serve-stage` on every node after node0, and finally (if a prompt was given) runs
# `generate-distributed` on node0.
#
# Usage:
#   ./scripts/launch_cluster.sh                          # deploy+build+start stages only
#   ./scripts/launch_cluster.sh "Tell me a story" 0.0 64 # ...then generate on node0
#
# Env overrides: NANOCAMELID_CLUSTER_CONFIG, PI_USER, SSH_KEY
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Optional local credentials (gitignored).
[[ -f "$REPO_ROOT/config/cluster.env" ]] && source "$REPO_ROOT/config/cluster.env"
# Prefer the gitignored local config (real IPs) over the committed example.
if [[ -n "${NANOCAMELID_CLUSTER_CONFIG:-}" ]]; then
  CONFIG="$NANOCAMELID_CLUSTER_CONFIG"
elif [[ -f "$REPO_ROOT/config/nodes.local.toml" ]]; then
  CONFIG="$REPO_ROOT/config/nodes.local.toml"
else
  CONFIG="$REPO_ROOT/config/nodes.toml"
fi
PI_USER="${PI_USER:-pi}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

PROMPT="${1:-}"
TEMP="${2:-0.0}"
MAX_TOKENS="${3:-128}"

REMOTE_REPO="~/nanocamelid/src/NanoCamelid"
REMOTE_CONFIG="$REMOTE_REPO/config/$(basename "$CONFIG")"
STAGE_LOG="~/nanocamelid/stage.log"

if [[ ! -f "$CONFIG" ]]; then
  echo "Cluster config not found: $CONFIG" >&2
  exit 1
fi

# Pull the node names/hosts out of nodes.toml, preserving order.
names=($(awk -F'"' '/^name[[:space:]]*=/ {print $2}' "$CONFIG"))
hosts=($(awk -F'"' '/^host[[:space:]]*=/ {print $2}' "$CONFIG"))

if [[ "${#names[@]}" -lt 2 || "${#names[@]}" -ne "${#hosts[@]}" ]]; then
  echo "Could not parse at least 2 nodes from $CONFIG (got ${#names[@]} names, ${#hosts[@]} hosts)" >&2
  exit 1
fi

if [[ -f "$SSH_KEY" ]]; then
  SSH_OPT="-o IdentitiesOnly=yes -i $SSH_KEY"
else
  SSH_OPT=""
fi

# The remote build places the binary according to env.sh's CARGO_TARGET_DIR
# (~/nanocamelid/target); fall back to the in-repo target dir.
REMOTE_RUN='
  export PATH="$HOME/.cargo/bin:$PATH"
  [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
  [ -f ~/nanocamelid/env.sh ] && source ~/nanocamelid/env.sh
  BIN="${CARGO_TARGET_DIR:-$HOME/nanocamelid/src/NanoCamelid/target}/release/nanocamelid"
'

echo "==> Deploying + building on all ${#names[@]} nodes..."
for i in "${!names[@]}"; do
  echo "--- ${names[$i]} (${hosts[$i]}) ---"
  "$SCRIPT_DIR/deploy.sh" "${hosts[$i]}" "$SSH_KEY" "$PI_USER"
  ssh ${SSH_OPT} "${PI_USER}@${hosts[$i]}" bash <<EOF
  export PATH="\$HOME/.cargo/bin:\$PATH"
  [ -f "\$HOME/.cargo/env" ] && source "\$HOME/.cargo/env"
  if [ ! -d ~/nanocamelid/benchmarks ]; then
    chmod +x $REMOTE_REPO/scripts/pi/bootstrap.sh
    $REMOTE_REPO/scripts/pi/bootstrap.sh
  fi
  [ -f ~/nanocamelid/env.sh ] && source ~/nanocamelid/env.sh
  cd $REMOTE_REPO
  cargo build --release
EOF
done

echo "==> Starting pipeline stages on ${names[@]:1}..."
for i in "${!names[@]}"; do
  [[ "$i" == "0" ]] && continue
  node="${names[$i]}"
  host="${hosts[$i]}"
  echo "--- starting serve-stage on $node ($host) ---"
  ssh ${SSH_OPT} "${PI_USER}@${host}" bash <<EOF
$REMOTE_RUN
  pkill -f "nanocamelid serve-stage" 2>/dev/null || true
  sleep 0.5
  nohup "\$BIN" serve-stage "$REMOTE_CONFIG" "$node" > $STAGE_LOG 2>&1 &
  sleep 1
  if pgrep -f "nanocamelid serve-stage" >/dev/null; then
    echo "$node: serve-stage running (log: $STAGE_LOG)"
  else
    echo "$node: FAILED to start serve-stage; last log lines:" >&2
    tail -5 $STAGE_LOG >&2
    exit 1
  fi
EOF
done

if [[ -z "$PROMPT" ]]; then
  echo
  echo "Stages are up. To generate, run on ${names[0]} (${hosts[0]}):"
  echo "  nanocamelid generate-distributed $REMOTE_CONFIG \"<prompt>\" [temp] [max_tokens]"
  echo "or re-run this script with a prompt argument."
  exit 0
fi

echo "==> Running generate-distributed on head ${names[0]} (${hosts[0]})..."
ssh ${SSH_OPT} "${PI_USER}@${hosts[0]}" bash <<EOF
$REMOTE_RUN
  "\$BIN" generate-distributed "$REMOTE_CONFIG" "$PROMPT" "$TEMP" "$MAX_TOKENS"
EOF
