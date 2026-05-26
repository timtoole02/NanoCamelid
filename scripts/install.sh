#!/usr/bin/env bash
set -euo pipefail

repo_url="${NANOCAMELID_REPO_URL:-https://github.com/timtoole02/NanoCamelid.git}"
repo_ref="${NANOCAMELID_REF:-main}"
install_dir="${NANOCAMELID_INSTALL_DIR:-$HOME/.local/share/nanocamelid/NanoCamelid}"
bin_dir="${NANOCAMELID_BIN_DIR:-$HOME/.local/bin}"

usage() {
  cat <<'USAGE'
Usage: install.sh [--dry-run]

Installs the latest NanoCamelid release binary.

Options:
  --dry-run   Print the resolved install/build plan without cloning or building

Target-dir resolution:
  1. CARGO_TARGET_DIR
  2. NANOCAMELID_TARGET_DIR
  3. /mnt/nanocamelid/target when /mnt/nanocamelid exists
  4. $HOME/.cache/nanocamelid/target on non-macOS hosts

On macOS, set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to an external
/Volumes path that resolves back under /Volumes, so install builds do not create
Cargo artifacts on the internal disk.
USAGE
}

default_target_dir() {
  if [[ -d /mnt/nanocamelid || -e /mnt/nanocamelid ]]; then
    echo "/mnt/nanocamelid/target"
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    return 1
  else
    echo "$HOME/.cache/nanocamelid/target"
  fi
}

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    -h | --help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

resolve_existing_prefix() {
  local path="$1"
  local suffix=""
  local parent

  while [[ ! -e "$path" ]]; do
    parent="$(dirname "$path")"
    if [[ "$parent" == "$path" ]]; then
      printf '%s%s\n' "$path" "$suffix"
      return
    fi
    suffix="/$(basename "$path")$suffix"
    path="$parent"
  done

  if [[ -d "$path" ]]; then
    (
      cd -P -- "$path"
      local resolved_pwd="$PWD"
      if [[ "$resolved_pwd" == //* ]]; then
        resolved_pwd="/${resolved_pwd#//}"
      fi
      if [[ "$resolved_pwd" == "/" ]]; then
        printf '/%s\n' "${suffix#/}"
      else
        printf '%s%s\n' "$resolved_pwd" "$suffix"
      fi
    )
  else
    printf '%s%s\n' "$path" "$suffix"
  fi
}

validate_target_dir() {
  local target_dir="$1"

  case "$target_dir" in
    target|target/*|./target|./target/*)
      echo "Refusing to use a relative repo-local Cargo target dir: $target_dir" >&2
      echo "Set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to an external path." >&2
      exit 2
      ;;
  esac

  if [[ "$(uname -s)" == "Darwin" ]]; then
    case "$target_dir" in
      /Volumes/*) ;;
      *)
        echo "Refusing to use a non-external Cargo target dir on macOS: $target_dir" >&2
        echo "Set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to a /Volumes path." >&2
        exit 2
        ;;
    esac

    local resolved_target_dir
    resolved_target_dir="$(resolve_existing_prefix "$target_dir")"
    case "$resolved_target_dir" in
      /Volumes/*) ;;
      *)
        echo "Refusing to use a Cargo target dir that resolves outside /Volumes on macOS: $target_dir -> $resolved_target_dir" >&2
        echo "Set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to a real external drive path." >&2
        exit 2
        ;;
    esac
  fi
}

target_dir="${CARGO_TARGET_DIR:-${NANOCAMELID_TARGET_DIR:-}}"
if [[ -z "$target_dir" ]]; then
  if ! target_dir="$(default_target_dir)"; then
    echo "Refusing to guess a Cargo target dir on macOS." >&2
    echo "Set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to an external path." >&2
    exit 2
  fi
fi
validate_target_dir "$target_dir"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "NanoCamelid install dry run"
  echo "repo_url: $repo_url"
  echo "repo_ref: $repo_ref"
  echo "install_dir: $install_dir"
  echo "bin_dir: $bin_dir"
  echo "cargo_target_dir: $target_dir"
  echo "steps: ensure git and cargo; clone/update repo; cargo build --release; link nanocamelid"
  exit 0
fi

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
