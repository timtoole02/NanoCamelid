#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: validate.sh [--dry-run]

Runs NanoCamelid's standard local validation gate:
  1. cargo fmt -- --check
  2. cargo test
  3. cargo clippy --all-targets -- -D warnings
  4. cargo run -- ready 1b --dry-run
  5. ./scripts/pi/smoke-1b.sh --dry-run
  6. ./scripts/pi/ready-1b.sh --dry-run
  7. ./scripts/pi/chat-1b.sh --dry-run
  8. ./scripts/pi/bench-1b-prefill.sh --dry-run
  9. ./scripts/remote_build.sh <redacted-pi-host> --dry-run
  10. ./scripts/install.sh --dry-run

Target-dir resolution:
  1. CARGO_TARGET_DIR
  2. NANOCAMELID_TARGET_DIR
  3. /mnt/nanocamelid/target when /mnt/nanocamelid exists
  4. $HOME/.cache/nanocamelid/target on non-macOS hosts

On macOS, set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to an external
/Volumes path that resolves back under /Volumes, so validation does not create
build artifacts on the internal disk. Dry runs print the resolved commands
without creating the target dir.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

default_target_dir() {
  if [[ -d /mnt/nanocamelid || -e /mnt/nanocamelid ]]; then
    echo "/mnt/nanocamelid/target"
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    return 1
  else
    echo "$HOME/.cache/nanocamelid/target"
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
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

TARGET_DIR="${CARGO_TARGET_DIR:-${NANOCAMELID_TARGET_DIR:-}}"
if [[ -z "$TARGET_DIR" ]]; then
  if ! TARGET_DIR="$(default_target_dir)"; then
    echo "Refusing to guess a Cargo target dir on macOS." >&2
    echo "Set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to an external path." >&2
    exit 2
  fi
fi

validate_target_dir() {
  local target_dir="$1"

  case "$target_dir" in
    "$REPO_ROOT"/target|"$REPO_ROOT"/target/*|target|target/*|./target|./target/*)
      echo "Refusing to use a repo-local Cargo target dir: $target_dir" >&2
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

export CARGO_TARGET_DIR="$TARGET_DIR"
validate_target_dir "$CARGO_TARGET_DIR"

incremental_reason=""
if [[ "$CARGO_TARGET_DIR" == /Volumes/* && -z "${CARGO_INCREMENTAL:-}" ]]; then
  export CARGO_INCREMENTAL=0
  incremental_reason="disabled for /Volumes target dir to avoid Cargo hard-link cache warnings"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "NanoCamelid validation dry run"
  echo "cargo_target_dir: $CARGO_TARGET_DIR"
  if [[ -n "$incremental_reason" ]]; then
    echo "cargo_incremental: 0 ($incremental_reason)"
  else
    echo "cargo_incremental: ${CARGO_INCREMENTAL:-default}"
  fi
  echo "steps: cargo fmt -- --check; cargo test; cargo clippy --all-targets -- -D warnings; cargo run -- ready 1b --dry-run; ./scripts/pi/smoke-1b.sh --dry-run; ./scripts/pi/ready-1b.sh --dry-run; ./scripts/pi/chat-1b.sh --dry-run; ./scripts/pi/bench-1b-prefill.sh --dry-run; ./scripts/remote_build.sh <redacted-pi-host> --dry-run; ./scripts/install.sh --dry-run"
  exit 0
fi

expect_failure() {
  local description="$1"
  shift

  if "$@"; then
    echo "Expected failure but command passed: $description" >&2
    exit 1
  fi
}

mkdir -p "$CARGO_TARGET_DIR"

echo "==> Cargo target dir: $CARGO_TARGET_DIR"
if [[ -n "$incremental_reason" ]]; then
  echo "==> CARGO_INCREMENTAL=0 ($incremental_reason)"
fi

echo "==> Checking format..."
cargo fmt -- --check

echo "==> Running tests..."
cargo test

echo "==> Running clippy..."
cargo clippy --all-targets -- -D warnings

echo "==> Checking 1B readiness CLI dry run..."
cargo run -- ready 1b --dry-run

echo "==> Checking 1B Pi smoke launcher dry run..."
./scripts/pi/smoke-1b.sh --dry-run

echo "==> Checking 1B Pi readiness launcher dry run..."
./scripts/pi/ready-1b.sh --dry-run

echo "==> Checking 1B Pi chat launcher dry run..."
./scripts/pi/chat-1b.sh --dry-run

echo "==> Checking 1B Pi chat launcher rejects invalid temperature..."
expect_failure "chat-1b invalid temperature" env NANOCAMELID_TEMP=bad ./scripts/pi/chat-1b.sh --dry-run

echo "==> Checking 1B Pi prefill benchmark launcher dry run..."
./scripts/pi/bench-1b-prefill.sh --dry-run

echo "==> Checking 1B Pi readiness launcher rejects invalid temperature..."
expect_failure "ready-1b invalid temperature" env NANOCAMELID_READY_TEMP=bad ./scripts/pi/ready-1b.sh --dry-run

echo "==> Checking remote Pi build launcher dry run..."
./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher rejects invalid smoke token count..."
expect_failure "remote_build invalid smoke token count" env NANOCAMELID_SMOKE_TOKENS=bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher rejects invalid readiness temperature..."
expect_failure "remote_build invalid readiness temperature" env NANOCAMELID_READY_TEMP=bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking installer dry run target-dir safety..."
./scripts/install.sh --dry-run
