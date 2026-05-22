#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "NanoCamelid Pi bootstrap expects an aarch64 Linux host." >&2
  exit 1
fi

if ! command -v rustc >/dev/null 2>&1; then
  echo "Install Rust first: https://rustup.rs/" >&2
  exit 1
fi

workspace="${NANOCAMELID_WORKSPACE:-$HOME/nanocamelid}"
mkdir -p "$workspace"/{src,models,benchmarks,target}

cat > "$workspace/env.sh" <<EOF
export NANOCAMELID_WORKSPACE="$workspace"
export CARGO_TARGET_DIR="$workspace/target"
EOF

echo "NanoCamelid workspace ready: $workspace"
echo "Load it with: source $workspace/env.sh"
