#!/usr/bin/env bash
set -euo pipefail

repo_url="${NANOCAMELID_REPO_URL:-https://github.com/timtoole02/NanoCamelid.git}"
version="${NANOCAMELID_VERSION:-v0.1.0}"
repo_ref="${NANOCAMELID_REF:-main}"
install_mode="${NANOCAMELID_INSTALL_MODE:-release}"
install_dir="${NANOCAMELID_INSTALL_DIR:-$HOME/.local/share/nanocamelid/NanoCamelid}"
bin_dir="${NANOCAMELID_BIN_DIR:-$HOME/.local/bin}"
release_base_url="${NANOCAMELID_RELEASE_BASE_URL:-https://github.com/timtoole02/NanoCamelid/releases/download}"
release_target="${NANOCAMELID_RELEASE_TARGET:-aarch64-unknown-linux-gnu}"

usage() {
  cat <<'USAGE'
Usage: install.sh [--version v0.1.0] [--dev] [--dry-run]

Installs NanoCamelid from a versioned GitHub release by default.

Options:
  --version   Release tag to install, default v0.1.0
  --dev       Explicitly install from source/main with Cargo
  --dry-run   Print the resolved install plan without downloading or building

Target-dir resolution:
  1. CARGO_TARGET_DIR
  2. NANOCAMELID_TARGET_DIR
  3. /mnt/nanocamelid/target when /mnt/nanocamelid exists
  4. $HOME/.cache/nanocamelid/target on non-macOS hosts

On macOS, set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to an external
/Volumes path that resolves back under /Volumes, so install builds do not create
Cargo artifacts on the internal disk.

Dev/source mode env:
  NANOCAMELID_INSTALL_MODE=source
  NANOCAMELID_REF=main
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
while [[ "$#" -gt 0 ]]; do
  arg="$1"
  case "$arg" in
    -h | --help)
      usage
      exit 0
      ;;
    --version)
      if [[ -z "${2:-}" ]]; then
        echo "--version requires a release tag, for example v0.1.0" >&2
        exit 2
      fi
      version="$2"
      shift
      ;;
    --dev)
      install_mode="source"
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
  shift
done

version="${version#v}"
version="v$version"

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
if [[ "$install_mode" == "source" || "$install_mode" == "dev" ]]; then
  if [[ -z "$target_dir" ]]; then
    if ! target_dir="$(default_target_dir)"; then
      echo "Refusing to guess a Cargo target dir on macOS." >&2
      echo "Set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to an external path." >&2
      exit 2
    fi
  fi
  validate_target_dir "$target_dir"
else
  target_dir="not used for release install"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "NanoCamelid install dry run"
  echo "install_mode: $install_mode"
  echo "version: $version"
  echo "release_target: $release_target"
  echo "release_url: $release_base_url/$version/nanocamelid-$version-$release_target.tar.gz"
  echo "repo_url: $repo_url"
  echo "repo_ref: $repo_ref"
  echo "install_dir: $install_dir"
  echo "bin_dir: $bin_dir"
  echo "cargo_target_dir: $target_dir"
  if [[ "$install_mode" == "source" || "$install_mode" == "dev" ]]; then
    echo "steps: ensure git and cargo; clone/update repo; cargo build --release; link nanocamelid"
  else
    echo "steps: ensure curl and tar; download release tarball and SHA256SUMS; verify checksum; install nanocamelid"
  fi
  exit 0
fi

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

mkdir -p "$(dirname "$install_dir")" "$bin_dir"

if [[ "$install_mode" == "source" || "$install_mode" == "dev" ]]; then
  need git
  need cargo
  mkdir -p "$target_dir"

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
else
  need curl
  need tar

  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT

  archive="nanocamelid-$version-$release_target.tar.gz"
  release_url="$release_base_url/$version/$archive"
  sums_url="$release_base_url/$version/SHA256SUMS"

  echo "Downloading NanoCamelid $version release"
  curl -fsSL "$release_url" -o "$tmp_dir/$archive"
  curl -fsSL "$sums_url" -o "$tmp_dir/SHA256SUMS"

  (
    cd "$tmp_dir"
    if command -v sha256sum >/dev/null 2>&1; then
      grep -F "  $archive" SHA256SUMS | sha256sum -c -
    else
      expected="$(awk -v file="$archive" '$2 == file { print $1 }' SHA256SUMS)"
      if [[ -z "$expected" ]]; then
        echo "Checksum entry missing for $archive" >&2
        exit 1
      fi
      actual="$(shasum -a 256 "$archive" | awk '{ print $1 }')"
      if [[ "$actual" != "$expected" ]]; then
        echo "Checksum mismatch for $archive" >&2
        exit 1
      fi
    fi
    tar -xzf "$archive"
  )

  extracted_binary="$(find "$tmp_dir" -type f -name nanocamelid -perm -111 | head -n 1)"
  if [[ -z "$extracted_binary" ]]; then
    echo "Release archive did not contain an executable nanocamelid binary" >&2
    exit 1
  fi

  install -m 0755 "$extracted_binary" "$bin_dir/nanocamelid"
fi

echo "NanoCamelid installed:"
echo "  binary: $bin_dir/nanocamelid"
if [[ "$install_mode" == "source" || "$install_mode" == "dev" ]]; then
  echo "  repo:   $install_dir"
else
  echo "  release: $version"
fi
echo
if [[ ":$PATH:" != *":$bin_dir:"* ]]; then
  echo "Add this to your shell profile if nanocamelid is not on PATH:"
  echo "  export PATH=\"$bin_dir:\$PATH\""
else
  echo "Try it:"
  echo "  nanocamelid --help"
fi
