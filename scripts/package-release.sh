#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cargo_version="$(grep -m1 '^version = ' "$repo_root/Cargo.toml" | sed 's/version = "\(.*\)"/\1/')"
raw_version="${NANOCAMELID_VERSION:-$cargo_version}"
version="${raw_version#v}"
version_tag="v$version"
target_triple="${NANOCAMELID_RELEASE_TARGET:-aarch64-unknown-linux-gnu}"
package_name="nanocamelid-${version_tag}-${target_triple}"
dist_dir="${NANOCAMELID_DIST_DIR:-$repo_root/dist}"
stage_dir="$dist_dir/$package_name"
target_dir="${CARGO_TARGET_DIR:-${NANOCAMELID_TARGET_DIR:-}}"

usage() {
  cat <<'USAGE'
Usage: package-release.sh [--dry-run]

Builds the release binary and creates:
  dist/nanocamelid-v<version>-aarch64-unknown-linux-gnu.tar.gz
  dist/SHA256SUMS

Env:
  NANOCAMELID_VERSION         Override package version; default Cargo.toml version
  NANOCAMELID_RELEASE_TARGET  Override artifact target name; default aarch64-unknown-linux-gnu
  NANOCAMELID_DIST_DIR        Output directory; default ./dist
  CARGO_TARGET_DIR            Cargo target directory; otherwise Cargo metadata is used

On macOS, set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to an external
/Volumes path that resolves back under /Volumes, so release packaging does not
create Cargo artifacts on the internal disk.
USAGE
}

dry_run=0
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      dry_run=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$version" != "$cargo_version" ]]; then
  echo "Release version $version_tag does not match Cargo.toml version $cargo_version." >&2
  echo "Update Cargo.toml before packaging, or set NANOCAMELID_VERSION=v$cargo_version." >&2
  exit 2
fi

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
  local dir="$1"

  case "$dir" in
    target|target/*|./target|./target/*)
      echo "Refusing to use a relative repo-local Cargo target dir: $dir" >&2
      echo "Set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to an external path." >&2
      exit 2
      ;;
  esac

  if [[ "$(uname -s)" == "Darwin" ]]; then
    case "$dir" in
      /Volumes/*) ;;
      *)
        echo "Refusing to use a non-external Cargo target dir on macOS: $dir" >&2
        echo "Set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to a /Volumes path." >&2
        exit 2
        ;;
    esac

    local resolved_target_dir
    resolved_target_dir="$(resolve_existing_prefix "$dir")"
    case "$resolved_target_dir" in
      /Volumes/*) ;;
      *)
        echo "Refusing to use a Cargo target dir that resolves outside /Volumes on macOS: $dir -> $resolved_target_dir" >&2
        echo "Set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to a real external drive path." >&2
        exit 2
        ;;
    esac
  fi
}

resolve_cargo_metadata_target_dir() {
  local metadata
  metadata="$(cargo metadata --no-deps --format-version 1)"
  printf '%s\n' "$metadata" | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p' | head -n 1
}

if [[ -z "$target_dir" ]]; then
  if ! target_dir="$(resolve_cargo_metadata_target_dir)" || [[ -z "$target_dir" ]]; then
    echo "Could not resolve Cargo target directory." >&2
    echo "Set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR explicitly." >&2
    exit 2
  fi
fi

validate_target_dir "$target_dir"
expected_version_output="nanocamelid $version"

if [[ "$dry_run" == "1" ]]; then
  echo "NanoCamelid release package dry run"
  echo "version: $version_tag"
  echo "target_triple: $target_triple"
  echo "dist_dir: $dist_dir"
  echo "cargo_target_dir: $target_dir"
  echo "artifact: $dist_dir/$package_name.tar.gz"
  echo "version_manifest: $stage_dir/VERSION"
  echo "cargo_command: cargo build --release --bins --target $target_triple"
  echo "binary: $target_dir/$target_triple/release/nanocamelid"
  echo "version_check: nanocamelid --version == $expected_version_output"
  echo "steps: cargo build --release --bins --target $target_triple; verify nanocamelid --version; stage binary VERSION README docs LICENSE CHANGELOG RELEASE_NOTES service installer; tar; sha256"
  exit 0
fi

rm -rf "$stage_dir"
mkdir -p "$stage_dir" "$dist_dir"

CARGO_TARGET_DIR="$target_dir" cargo build --release --bins --target "$target_triple"

binary="$target_dir/$target_triple/release/nanocamelid"
if [[ ! -x "$binary" ]]; then
  echo "Release binary not found at $binary" >&2
  exit 1
fi
actual_version_output="$("$binary" --version 2>/dev/null || true)"
if [[ "$actual_version_output" != "$expected_version_output" ]]; then
  echo "Release binary version mismatch: expected '$expected_version_output', got '${actual_version_output:-<no output>}'" >&2
  exit 1
fi

cp "$binary" "$stage_dir/nanocamelid"
printf '%s\n' "$version_tag" > "$stage_dir/VERSION"
cp "$repo_root/README.md" "$stage_dir/README.md"
cp -R "$repo_root/docs" "$stage_dir/docs"
cp "$repo_root/LICENSE" "$stage_dir/LICENSE"
cp "$repo_root/CHANGELOG.md" "$stage_dir/CHANGELOG.md"
cp "$repo_root/RELEASE_NOTES.md" "$stage_dir/RELEASE_NOTES.md"
mkdir -p "$stage_dir/scripts"
cp "$repo_root/scripts/install-systemd-user-service.sh" "$stage_dir/scripts/install-systemd-user-service.sh"

(
  cd "$dist_dir"
  tar -czf "$package_name.tar.gz" "$package_name"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$package_name.tar.gz" > SHA256SUMS
  else
    shasum -a 256 "$package_name.tar.gz" > SHA256SUMS
  fi
)

echo "Created $dist_dir/$package_name.tar.gz"
echo "Created $dist_dir/SHA256SUMS"
