#!/usr/bin/env bash
set -euo pipefail

repo_url="${NANOCAMELID_REPO_URL:-https://github.com/timtoole02/NanoCamelid.git}"
repo_ref="${NANOCAMELID_REF:-main}"
install_dir="${NANOCAMELID_INSTALL_DIR:-$HOME/.local/share/nanocamelid/NanoCamelid}"
bin_dir="${NANOCAMELID_BIN_DIR:-$HOME/.local/bin}"
target_dir="${CARGO_TARGET_DIR:-$HOME/.cache/nanocamelid/target}"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need git
need cargo

case "$(uname -s)" in
  Linux) ;;
  *)
    echo "NanoCamelid is intended for Linux ARM64 targets; continuing on $(uname -s)." >&2
    ;;
esac

case "$(uname -m)" in
  aarch64|arm64) ;;
  *)
    echo "NanoCamelid is tuned for ARM64; continuing on $(uname -m)." >&2
    ;;
esac

mkdir -p "$(dirname "$install_dir")" "$bin_dir" "$target_dir"

if [[ -d "$install_dir/.git" ]]; then
  echo "Updating NanoCamelid in $install_dir"
  git -C "$install_dir" fetch --quiet origin "$repo_ref"
  git -C "$install_dir" checkout --quiet "$repo_ref"
  git -C "$install_dir" pull --ff-only --quiet origin "$repo_ref"
elif [[ -e "$install_dir" ]]; then
  echo "Install directory exists but is not a Git checkout: $install_dir" >&2
  echo "Set NANOCAMELID_INSTALL_DIR to another path or move the existing directory." >&2
  exit 1
else
  echo "Cloning NanoCamelid into $install_dir"
  git clone --quiet --branch "$repo_ref" "$repo_url" "$install_dir"
fi

echo "Building NanoCamelid release binary"
CARGO_TARGET_DIR="$target_dir" cargo build --release --manifest-path "$install_dir/Cargo.toml"

binary="$target_dir/release/nanocamelid"
if [[ ! -x "$binary" ]]; then
  echo "Build finished, but binary was not found at $binary" >&2
  exit 1
fi

ln -sf "$binary" "$bin_dir/nanocamelid"

echo "NanoCamelid installed:"
echo "  repo:   $install_dir"
echo "  binary: $bin_dir/nanocamelid"
echo
if [[ ":$PATH:" != *":$bin_dir:"* ]]; then
  echo "Add this to your shell profile if nanocamelid is not on PATH:"
  echo "  export PATH=\"$bin_dir:\$PATH\""
else
  echo "Try it:"
  echo "  nanocamelid --help"
fi
