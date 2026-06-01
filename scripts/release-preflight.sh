#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cargo_version="$(grep -m1 '^version = ' "$repo_root/Cargo.toml" | sed 's/version = "\(.*\)"/\1/')"
raw_version="${NANOCAMELID_VERSION:-$cargo_version}"
version="${raw_version#v}"
version_tag="v$version"
release_target="${NANOCAMELID_RELEASE_TARGET:-aarch64-unknown-linux-gnu}"
remote="${NANOCAMELID_RELEASE_REMOTE:-origin}"
dry_run=0
check_remote=0
require_unpublished=0

usage() {
  cat <<'USAGE'
Usage: release-preflight.sh [--dry-run] [--check-remote] [--require-unpublished]

Checks that the local checkout is ready to publish a versioned NanoCamelid
release tag. The default is local-only so normal validation does not depend on
network access.

Options:
  --dry-run              Print the preflight plan and release next action
  --check-remote         Check whether the release tag already exists on origin
  --require-unpublished  Fail if the local or remote release tag already exists

Env:
  NANOCAMELID_VERSION         Release tag/version, default Cargo.toml version
  NANOCAMELID_RELEASE_TARGET  Release target, default aarch64-unknown-linux-gnu
  NANOCAMELID_RELEASE_REMOTE  Git remote to inspect, default origin
USAGE
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      dry_run=1
      ;;
    --check-remote)
      check_remote=1
      ;;
    --require-unpublished)
      require_unpublished=1
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

need git

if [[ "$version" != "$cargo_version" ]]; then
  echo "Release version $version_tag does not match Cargo.toml version $cargo_version." >&2
  echo "Update Cargo.toml before release preflight, or set NANOCAMELID_VERSION=v$cargo_version." >&2
  exit 2
fi

cd "$repo_root"

working_tree="clean"
if ! git diff --quiet -- . || ! git diff --cached --quiet -- .; then
  working_tree="dirty"
fi
if [[ "$working_tree" != "clean" && "$dry_run" != "1" ]]; then
  echo "Release preflight requires a clean working tree." >&2
  echo "Commit or stash local changes before publishing $version_tag." >&2
  exit 2
fi

head_sha="$(git rev-parse --short HEAD)"
branch="$(git symbolic-ref --quiet --short HEAD || echo detached)"
local_tag_status="absent"
local_tag_commit=""
if git rev-parse -q --verify "refs/tags/$version_tag" >/dev/null; then
  local_tag_commit="$(git rev-list -n 1 "$version_tag")"
  if [[ "$local_tag_commit" == "$(git rev-parse HEAD)" ]]; then
    local_tag_status="exists_at_head"
  else
    local_tag_status="exists_elsewhere"
  fi
fi

if [[ "$require_unpublished" == "1" && "$local_tag_status" != "absent" ]]; then
  echo "Release tag $version_tag already exists locally ($local_tag_status)." >&2
  exit 2
fi
if [[ "$local_tag_status" == "exists_elsewhere" ]]; then
  echo "Release tag $version_tag exists locally but does not point at HEAD." >&2
  echo "Move to the tagged commit or choose the matching NANOCAMELID_VERSION." >&2
  exit 2
fi

remote_tag_status="not_checked"
if [[ "$check_remote" == "1" ]]; then
  set +e
  git ls-remote --exit-code --tags "$remote" "refs/tags/$version_tag" >/dev/null 2>&1
  remote_status=$?
  set -e

  if [[ "$remote_status" == "0" ]]; then
    remote_tag_status="exists"
  elif [[ "$remote_status" == "2" ]]; then
    remote_tag_status="absent"
  else
    echo "Could not check release tag $version_tag on remote $remote." >&2
    exit "$remote_status"
  fi
  if [[ "$require_unpublished" == "1" && "$remote_tag_status" == "exists" ]]; then
    echo "Release tag $version_tag already exists on $remote." >&2
    exit 2
  fi
fi

next_action="git tag -a $version_tag -m 'NanoCamelid $version_tag' && git push $remote $version_tag"
if [[ "$local_tag_status" == "exists_at_head" ]]; then
  next_action="git push $remote $version_tag"
fi
if [[ "$remote_tag_status" == "exists" ]]; then
  next_action="inspect the GitHub release workflow for $version_tag"
fi

echo "NanoCamelid release preflight"
echo "version: $version_tag"
echo "cargo_version: $cargo_version"
echo "release_target: $release_target"
echo "branch: $branch"
echo "head: $head_sha"
echo "working_tree: $working_tree"
echo "local_tag: $local_tag_status"
echo "remote: $remote"
echo "remote_tag: $remote_tag_status"
echo "checks: ./scripts/validate.sh; ./scripts/package-release.sh --dry-run; ./scripts/install.sh --dry-run"
echo "next_action: $next_action"

if [[ "$dry_run" == "1" ]]; then
  exit 0
fi

./scripts/validate.sh
./scripts/package-release.sh --dry-run
./scripts/install.sh --dry-run
