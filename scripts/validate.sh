#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: validate.sh [--dry-run]

Runs NanoCamelid's standard local validation gate:
  1. public doc/example hygiene scan
  2. cargo fmt -- --check
  3. cargo test
  4. cargo clippy --all-targets -- -D warnings
  5. stable v0.1 CLI help contract sweep
  6. cargo run -- smoke --help
  7. cargo run -- doctor --dry-run --json
  8. cargo run -- serve --help
  9. cargo run -- serve --dry-run
  10. local serve HTTP smoke on 127.0.0.1, including request/output caps
  11. cargo run -- models --help
  11a. cargo run -- models list --help
  11b. cargo run -- models scan --help
  11c. cargo run -- models inspect --help
  12. cargo run -- models list --dry-run --dir /mnt/nanocamelid/models --json
  13. cargo run -- models scan --dry-run --dir /mnt/nanocamelid/models --json
  14. cargo run -- models inspect 1b --dry-run
  14a. cargo run -- models inspect 3b --dry-run
  14b. cargo run -- models inspect /models/custom.gguf --dry-run
  15. cargo run -- model 1b --dry-run
  16. cargo run -- inspect 1b --dry-run
  17. cargo run -- generate 1b --dry-run
  18. cargo run -- chat 1b --dry-run
  19. cargo run -- smoke 1b --dry-run
  20. cargo run -- ready 1b --dry-run
  21. cargo run -- evidence 1b --dry-run
  22. cargo run -- tui 1b --dry-run
  23. cargo run -- bench 1b --dry-run
  24. cargo run -- bench 1b --help
  25. ./scripts/pi/model-1b.sh --dry-run
  26. ./scripts/pi/smoke-1b.sh --dry-run
  27. ./scripts/pi/ready-1b.sh --dry-run
  28. ./scripts/pi/chat-1b.sh --dry-run
  29. ./scripts/pi/bench-1b-prefill.sh --dry-run
  30. ./scripts/pi/context-pack-1b.sh --dry-run
  31. ./scripts/pi/evidence-1b.sh --dry-run
  32. ./scripts/pi/strand-cluster.sh --dry-run
  33. ./scripts/pi/mixtral-cluster.sh --dry-run
  34. ./scripts/remote_build.sh <redacted-pi-host> --dry-run
  35. NANOCAMELID_REMOTE_CONTEXT_PACKS=512,1024 ./scripts/remote_build.sh <redacted-pi-host> --dry-run
  36. NANOCAMELID_REMOTE_PREFILL_BENCH=1 ./scripts/remote_build.sh <redacted-pi-host> --dry-run
  37. NANOCAMELID_REMOTE_EVIDENCE=1 ./scripts/remote_build.sh <redacted-pi-host> --dry-run
  38. NANOCAMELID_REMOTE_1B_QUANT=q4 ./scripts/remote_build.sh <redacted-pi-host> --dry-run
  39. NANOCAMELID_REMOTE_1B_QUANT=q8 ./scripts/remote_build.sh <redacted-pi-host> --dry-run
  40. cargo run -- --version
  41. ./scripts/install.sh --dry-run
  42. ./scripts/package-release.sh --dry-run
  43. ./scripts/install-systemd-user-service.sh --dry-run

Target-dir resolution:
  1. CARGO_TARGET_DIR
  2. NANOCAMELID_TARGET_DIR
  3. /mnt/nanocamelid/target when /mnt/nanocamelid exists
  4. a single existing /Volumes/*/nanocamelid-target on macOS hosts
  5. $HOME/.cache/nanocamelid/target on non-macOS hosts

On macOS, set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to an external
/Volumes path that resolves back under /Volumes, or create one
/Volumes/<drive>/nanocamelid-target directory. Validation refuses to create or
guess an internal-disk target dir. Dry runs print the resolved commands without
creating the target dir.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

find_macos_external_target_dir() {
  local candidates=()
  local candidate

  for candidate in /Volumes/*/nanocamelid-target; do
    if [[ -d "$candidate" ]]; then
      candidates+=("$candidate")
    fi
  done

  if [[ "${#candidates[@]}" -eq 1 ]]; then
    echo "${candidates[0]}"
    return 0
  fi

  return 1
}

default_target_dir() {
  if [[ -d /mnt/nanocamelid || -e /mnt/nanocamelid ]]; then
    echo "/mnt/nanocamelid/target"
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    find_macos_external_target_dir
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
    echo "Set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to an external path, or create one /Volumes/<drive>/nanocamelid-target directory." >&2
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
export NANOCAMELID_SMOKE_GGUF="${NANOCAMELID_SMOKE_GGUF:-/mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf}"

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
  echo "steps: public doc/example hygiene scan; cargo fmt -- --check; cargo test; cargo clippy --all-targets -- -D warnings; stable v0.1 CLI help contract sweep; cargo run -- smoke --help; cargo run -- doctor --dry-run --json; cargo run -- serve --help; cargo run -- serve --dry-run; local serve HTTP smoke on 127.0.0.1 with request/output caps; cargo run -- models --help; cargo run -- models list --help; cargo run -- models scan --help; cargo run -- models inspect --help; cargo run -- models list --dry-run --dir /mnt/nanocamelid/models --json; cargo run -- models scan --dry-run --dir /mnt/nanocamelid/models --json; cargo run -- models inspect 1b --dry-run; cargo run -- models inspect 3b --dry-run; cargo run -- models inspect /models/custom.gguf --dry-run; cargo run -- model 1b --dry-run; cargo run -- inspect 1b --dry-run; cargo run -- generate 1b --dry-run; cargo run -- chat 1b --dry-run; cargo run -- smoke 1b --dry-run; cargo run -- ready 1b --dry-run; cargo run -- evidence 1b --dry-run; cargo run -- tui 1b --dry-run; cargo run -- bench 1b --dry-run; cargo run -- bench 1b --help; ./scripts/pi/model-1b.sh --dry-run; ./scripts/pi/smoke-1b.sh --dry-run; ./scripts/pi/ready-1b.sh --dry-run; ./scripts/pi/chat-1b.sh --dry-run; ./scripts/pi/bench-1b-prefill.sh --dry-run; ./scripts/pi/context-pack-1b.sh --dry-run; ./scripts/pi/evidence-1b.sh --dry-run; ./scripts/pi/strand-cluster.sh --dry-run; ./scripts/pi/mixtral-cluster.sh --dry-run; ./scripts/remote_build.sh <redacted-pi-host> --dry-run; NANOCAMELID_REMOTE_CONTEXT_PACKS=512,1024 ./scripts/remote_build.sh <redacted-pi-host> --dry-run; NANOCAMELID_REMOTE_PREFILL_BENCH=1 ./scripts/remote_build.sh <redacted-pi-host> --dry-run; NANOCAMELID_REMOTE_EVIDENCE=1 ./scripts/remote_build.sh <redacted-pi-host> --dry-run; NANOCAMELID_REMOTE_1B_QUANT=q4 ./scripts/remote_build.sh <redacted-pi-host> --dry-run; NANOCAMELID_REMOTE_1B_QUANT=q8 ./scripts/remote_build.sh <redacted-pi-host> --dry-run; ./scripts/install.sh --dry-run; release installer companion-file dry-run checks; ./scripts/package-release.sh --dry-run; ./scripts/install-systemd-user-service.sh --dry-run"
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

expect_failure_output() {
  local description="$1"
  local expected="$2"
  shift 2
  local output
  local status

  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e

  if [[ "$status" -eq 0 ]]; then
    echo "Expected failure but command passed: $description" >&2
    exit 1
  fi
  if ! grep -F -- "$expected" <<<"$output" >/dev/null; then
    echo "Expected failure output missing for $description: $expected" >&2
    exit 1
  fi
}

expect_output_order() {
  local description="$1"
  local first="$2"
  local second="$3"
  shift 3

  if ! "$@" | awk -v first="$first" -v second="$second" '
    index($0, first) && first_line == 0 { first_line = NR }
    index($0, second) && second_line == 0 { second_line = NR }
    END { exit !(first_line > 0 && second_line > 0 && first_line < second_line) }
  '; then
    echo "Expected output order missing for $description: $first before $second" >&2
    exit 1
  fi
}

expect_output() {
  local description="$1"
  local expected="$2"
  shift 2
  local output_file

  output_file="$(mktemp "${TMPDIR:-/tmp}/nanocamelid-validate-output.XXXXXX")"
  if ! "$@" >"$output_file" 2>&1; then
    echo "Command failed for $description" >&2
    cat "$output_file" >&2
    rm -f "$output_file"
    exit 1
  fi
  if ! grep -F -- "$expected" "$output_file" >/dev/null; then
    echo "Expected output missing for $description: $expected" >&2
    cat "$output_file" >&2
    rm -f "$output_file"
    exit 1
  fi
  rm -f "$output_file"
}

expect_no_output() {
  local description="$1"
  local unexpected="$2"
  shift 2
  local output_file

  output_file="$(mktemp "${TMPDIR:-/tmp}/nanocamelid-validate-output.XXXXXX")"
  if ! "$@" >"$output_file" 2>&1; then
    echo "Command failed for $description" >&2
    cat "$output_file" >&2
    rm -f "$output_file"
    exit 1
  fi
  if grep -F "$unexpected" "$output_file" >/dev/null; then
    echo "Unexpected output found for $description: $unexpected" >&2
    cat "$output_file" >&2
    rm -f "$output_file"
    exit 1
  fi
  rm -f "$output_file"
}

expect_output_count() {
  local description="$1"
  local expected="$2"
  local expected_count="$3"
  shift 3

  local actual_count
  actual_count="$("$@" | awk -v expected="$expected" 'index($0, expected) { count++ } END { print count + 0 }')"
  if [[ "$actual_count" != "$expected_count" ]]; then
    echo "Expected $expected_count occurrences for $description but found $actual_count: $expected" >&2
    exit 1
  fi
}

check_stable_cli_help() {
  expect_output "top-level help documents command usage" "nanocamelid <command> [args]" cargo run -- --help
  expect_output "top-level help separates stable commands" "Stable v0.1 commands:" cargo run -- --help
  expect_output "top-level help separates compatibility commands" "Compatibility and lab commands:" cargo run -- --help
  expect_output "doctor help documents usage" "nanocamelid doctor [--json] [--dry-run]" cargo run -- doctor --help
  expect_output "probe help documents usage" "nanocamelid probe" cargo run -- probe --help
  expect_output "models namespace help documents list" "nanocamelid models list" cargo run -- models --help
  expect_output "models list help documents usage" "nanocamelid models list" cargo run -- models list --help
  expect_output "help models list documents usage" "NanoCamelid models list" cargo run -- help models list
  expect_output "models scan help documents usage" "nanocamelid models scan" cargo run -- models scan --help
  expect_output "help models scan documents usage" "NanoCamelid models scan" cargo run -- help models scan
  expect_output "models inspect help documents usage" "nanocamelid models inspect <model.gguf|1b|3b>" cargo run -- models inspect --help
  expect_output "help models inspect documents usage" "NanoCamelid models inspect" cargo run -- help models inspect
  expect_output "ready help documents 1b gate" "nanocamelid ready 1b" cargo run -- ready --help
  expect_output "chat help documents explicit model usage" "nanocamelid chat <model.gguf> <prompt>" cargo run -- chat --help
  expect_output "chat help documents 1b alias usage" "nanocamelid chat 1b <prompt>" cargo run -- chat --help
  expect_output "tui help documents explicit model usage" "nanocamelid tui <model.gguf> [temp] [max_tokens]" cargo run -- tui --help
  expect_output "tui help documents 1b alias usage" "nanocamelid tui 1b [temp] [max_tokens]" cargo run -- tui --help
  expect_output "serve help documents usage" "nanocamelid serve [--host <addr>] [--port <port>]" cargo run -- serve --help
}

check_public_hygiene() {
  local public_paths=(README.md docs scripts/deploy.sh)
  local public_doc_paths=(README.md docs)
  local matches
  local doc_matches
  local status

  set +e
  matches="$(
    rg -n \
      -e '/Users/' \
      -e 'file://' \
      -e '(^|[^0-9])(10|192\.168)\.[0-9]{1,3}\.[0-9]{1,3}([^0-9]|$)' \
      -e '(^|[^0-9])172\.(1[6-9]|2[0-9]|3[0-1])\.[0-9]{1,3}\.[0-9]{1,3}([^0-9]|$)' \
      "${public_paths[@]}"
  )"
  status=$?
  set -e

  if [[ "$status" -eq 0 ]]; then
    echo "Public docs/examples contain private paths or RFC1918 IP examples:" >&2
    echo "$matches" >&2
    exit 1
  fi
  if [[ "$status" -gt 1 ]]; then
    echo "Public hygiene scan failed." >&2
    echo "$matches" >&2
    exit "$status"
  fi

  set +e
  doc_matches="$(
    rg -n \
      -e '\b[A-Za-z0-9._-]+\.local\b' \
      -e '(^|[^A-Za-z0-9_-])raspberrypi([^A-Za-z0-9_-]|$)' \
      -e '(^|[[:space:]])ssh[[:space:]]+[^<[:space:]]' \
      -e '\.ssh/' \
      -e 'BEGIN [A-Z ]*PRIVATE KEY' \
      -e 'PRIVATE KEY' \
      -e 'ssh-rsa[[:space:]]' \
      -e 'ssh-ed25519[[:space:]]' \
      "${public_doc_paths[@]}"
  )"
  status=$?
  set -e

  if [[ "$status" -eq 0 ]]; then
    echo "Public docs contain local hostnames, raw SSH commands, or key material:" >&2
    echo "$doc_matches" >&2
    exit 1
  fi
  if [[ "$status" -gt 1 ]]; then
    echo "Public docs hostname/key hygiene scan failed." >&2
    echo "$doc_matches" >&2
    exit "$status"
  fi
}

expect_file_contains() {
  local description="$1"
  local expected="$2"
  local file="$3"

  if ! grep -F -- "$expected" "$file" >/dev/null; then
    echo "Expected response body missing for $description: $expected" >&2
    echo "Response body:" >&2
    cat "$file" >&2
    exit 1
  fi
}

expect_http_status() {
  local description="$1"
  local expected="$2"
  local actual="$3"
  local file="$4"

  if [[ "$actual" != "$expected" ]]; then
    echo "Expected HTTP $expected for $description but got $actual" >&2
    echo "Response body:" >&2
    cat "$file" >&2
    exit 1
  fi
}

validate_api_smoke_cleanup() {
  if [[ -n "${api_smoke_pid:-}" ]]; then
    kill "$api_smoke_pid" >/dev/null 2>&1 || true
    wait "$api_smoke_pid" >/dev/null 2>&1 || true
  fi
  if [[ -n "${api_smoke_model_dir:-}" ]]; then
    rm -rf "$api_smoke_model_dir"
  fi
  if [[ -n "${api_smoke_log:-}" ]]; then
    rm -f "$api_smoke_log"
  fi
  if [[ -n "${api_smoke_body:-}" ]]; then
    rm -f "$api_smoke_body"
  fi
}

check_local_api_smoke() {
  if ! command -v curl >/dev/null 2>&1; then
    echo "Missing required command for local API smoke: curl" >&2
    exit 1
  fi

  local port="${NANOCAMELID_VALIDATE_API_PORT:-18080}"
  local base_url="http://127.0.0.1:$port"
  local api_key="redacted-validation-token"
  local status
  local attempt

  api_smoke_pid=""
  api_smoke_model_dir="$(mktemp -d "${TMPDIR:-/tmp}/nanocamelid-validate-models.XXXXXX")"
  api_smoke_log="$(mktemp "${TMPDIR:-/tmp}/nanocamelid-validate-api.XXXXXX.log")"
  api_smoke_body="$(mktemp "${TMPDIR:-/tmp}/nanocamelid-validate-api.XXXXXX.body")"
  trap validate_api_smoke_cleanup EXIT

  cargo run -- serve \
    --host 127.0.0.1 \
    --port "$port" \
    --model-dir "$api_smoke_model_dir" \
    --api-key "$api_key" \
    --max-request-bytes 4096 \
    --max-input-tokens 64 \
    --max-output-tokens 8 \
    >"$api_smoke_log" 2>&1 &
  api_smoke_pid=$!

  for attempt in {1..40}; do
    if ! kill -0 "$api_smoke_pid" >/dev/null 2>&1; then
      echo "Local API smoke server exited before readiness." >&2
      cat "$api_smoke_log" >&2
      exit 1
    fi
    status="$(curl -sS -o "$api_smoke_body" -w "%{http_code}" \
      -H "Authorization: Bearer $api_key" "$base_url/health" || true)"
    if [[ "$status" == "200" ]]; then
      break
    fi
    sleep 0.25
  done
  expect_http_status "authenticated health readiness" "200" "$status" "$api_smoke_body"
  expect_file_contains "authenticated health readiness" "\"status\":\"ok\"" "$api_smoke_body"

  status="$(curl -sS -o "$api_smoke_body" -w "%{http_code}" "$base_url/health" || true)"
  expect_http_status "unauthenticated health rejection" "401" "$status" "$api_smoke_body"
  expect_file_contains "unauthenticated health rejection" "\"code\":\"unauthorized\"" "$api_smoke_body"

  status="$(curl -sS -o "$api_smoke_body" -w "%{http_code}" \
    -X OPTIONS \
    -H "Origin: http://127.0.0.1" \
    -H "Access-Control-Request-Method: POST" \
    -H "Access-Control-Request-Headers: authorization,content-type" \
    "$base_url/v1/chat/completions" || true)"
  expect_http_status "unauthenticated browser preflight" "204" "$status" "$api_smoke_body"

  status="$(curl -sS -o "$api_smoke_body" -w "%{http_code}" \
    -H "Authorization: Bearer $api_key" "$base_url/v1/models" || true)"
  expect_http_status "model list endpoint" "200" "$status" "$api_smoke_body"
  expect_file_contains "model list endpoint" "\"object\":\"list\"" "$api_smoke_body"
  expect_file_contains "model list endpoint" "\"data\":[]" "$api_smoke_body"

  status="$(curl -sS -o "$api_smoke_body" -w "%{http_code}" \
    -H "Authorization: Bearer $api_key" "$base_url/metrics" || true)"
  expect_http_status "metrics endpoint" "200" "$status" "$api_smoke_body"
  expect_file_contains "metrics endpoint" "nanocamelid_requests_total" "$api_smoke_body"
  expect_file_contains "metrics endpoint" "nanocamelid_responses_total{status=\"200\"}" "$api_smoke_body"
  expect_file_contains "metrics endpoint" "nanocamelid_responses_total{status=\"401\"} 1" "$api_smoke_body"
  expect_file_contains "metrics endpoint" "nanocamelid_max_output_tokens 8" "$api_smoke_body"

  status="$(curl -sS -o "$api_smoke_body" -w "%{http_code}" \
    -H "Authorization: Bearer $api_key" "$base_url/v1/completions" || true)"
  expect_http_status "completion method guard" "405" "$status" "$api_smoke_body"
  expect_file_contains "completion method guard" "\"code\":\"method_not_allowed\"" "$api_smoke_body"

  status="$(curl -sS -o "$api_smoke_body" -w "%{http_code}" \
    -X POST \
    -H "Authorization: Bearer $api_key" \
    -H "Content-Type: application/json" \
    "$base_url/v1/completions" || true)"
  expect_http_status "completion missing body" "400" "$status" "$api_smoke_body"
  expect_file_contains "completion missing body" "\"code\":\"missing_body\"" "$api_smoke_body"

  status="$(
    printf '{"model":"1b","prompt":"%05000d","max_tokens":1}' 0 | \
      curl -sS -o "$api_smoke_body" -w "%{http_code}" \
        -X POST \
        -H "Authorization: Bearer $api_key" \
        -H "Content-Type: application/json" \
        --data-binary @- \
        "$base_url/v1/completions" || true
  )"
  expect_http_status "completion request byte cap" "413" "$status" "$api_smoke_body"
  expect_file_contains "completion request byte cap" "\"code\":\"request_too_large\"" "$api_smoke_body"

  status="$(curl -sS -o "$api_smoke_body" -w "%{http_code}" \
    -X POST \
    -H "Authorization: Bearer $api_key" \
    -H "Content-Type: application/json" \
    -d '{"model":"1b","prompt":"hello","max_tokens":9}' \
    "$base_url/v1/completions" || true)"
  expect_http_status "completion output cap" "400" "$status" "$api_smoke_body"
  expect_file_contains "completion output cap" "\"code\":\"output_tokens_exceeded\"" "$api_smoke_body"

  status="$(curl -sS -o "$api_smoke_body" -w "%{http_code}" \
    -X POST \
    -H "Authorization: Bearer $api_key" \
    -H "Content-Type: application/json" \
    -d '{"model":"1b","messages":[{"role":"tool","content":"hello"}],"max_tokens":1}' \
    "$base_url/v1/chat/completions" || true)"
  expect_http_status "chat invalid role structured error" "400" "$status" "$api_smoke_body"
  expect_file_contains "chat invalid role structured error" "\"code\":\"invalid_messages\"" "$api_smoke_body"

  status="$(curl -sS -o "$api_smoke_body" -w "%{http_code}" \
    -H "Authorization: Bearer $api_key" "$base_url/not-found" || true)"
  expect_http_status "unknown endpoint structured error" "404" "$status" "$api_smoke_body"
  expect_file_contains "unknown endpoint structured error" "\"code\":\"not_found\"" "$api_smoke_body"

  validate_api_smoke_cleanup
  trap - EXIT
}

mkdir -p "$CARGO_TARGET_DIR"

echo "==> Cargo target dir: $CARGO_TARGET_DIR"
if [[ -n "$incremental_reason" ]]; then
  echo "==> CARGO_INCREMENTAL=0 ($incremental_reason)"
fi

echo "==> Checking public docs and examples for private paths/IPs..."
check_public_hygiene
expect_file_contains "release workflow standard validation gate" "run: ./scripts/validate.sh" .github/workflows/release.yml

echo "==> Checking format..."
cargo fmt -- --check

echo "==> Running tests..."
cargo test

echo "==> Running clippy..."
cargo clippy --all-targets -- -D warnings

echo "==> Checking stable v0.1 CLI help contract..."
check_stable_cli_help

echo "==> Checking smoke CLI help defaults..."
expect_output "smoke help q8 default prompt" "q8-* [prompt]                             Prompt text, default \"Hello\"" cargo run -- smoke --help
expect_output "smoke help 1b default prompt" "1b/3b [prompt]                            Prompt text, default \"Say hello in one sentence.\"" cargo run -- smoke --help
expect_output "smoke help 1b default tokens" "1b/3b [max_tokens]                        Greedy tokens to generate after parity, default 8" cargo run -- smoke --help
expect_output "smoke help 1b quant selectors" "--q4" cargo run -- smoke --help

echo "==> Checking doctor, serve, and models CLI contract..."
expect_output "probe help documents no-arg usage" "nanocamelid probe" cargo run -- probe --help
expect_failure_output "probe rejects extra argument" "unexpected probe argument" cargo run -- probe extra
expect_failure_output "probe rejects unknown option" "unknown probe option" cargo run -- probe --json
expect_output "doctor dry-run status" "NanoCamelid doctor" cargo run -- doctor --dry-run
expect_output "doctor json output" "\"command\":\"doctor\"" cargo run -- doctor --dry-run --json
expect_output "doctor json default model selection" "\"default_1b_selected\"" cargo run -- doctor --dry-run --json
expect_output "doctor json next action" "\"next_action\":" cargo run -- doctor --dry-run --json
expect_output "serve help documents default loopback" "default bind address is 127.0.0.1:8080" cargo run -- serve --help
expect_output "serve help documents non-loopback auth" "Non-loopback binds require bearer-token auth" cargo run -- serve --help
expect_output "serve help documents health endpoint" "GET  /health" cargo run -- serve --help
expect_output "serve dry-run status" "NanoCamelid serve dry run" cargo run -- serve --dry-run
expect_output "serve dry-run default listen" "listen: http://127.0.0.1:8080" cargo run -- serve --dry-run
expect_output "serve dry-run endpoint contract" "endpoints: /health /v1/models /v1/completions /v1/chat/completions /metrics" cargo run -- serve --dry-run
expect_output "serve dry-run request byte cap" "max_request_bytes: 65536" cargo run -- serve --dry-run
expect_output "serve dry-run caps" "max_output_tokens: 256" cargo run -- serve --dry-run
expect_output "serve dry-run explicit command" "serve_command: nanocamelid serve --host 127.0.0.1 --port 8080 --model-dir /mnt/nanocamelid/models --max-request-bytes 65536 --max-input-tokens 2048 --max-output-tokens 256" cargo run -- serve --dry-run
expect_output "serve dry-run api key required from env" "api_key_required: true" env NANOCAMELID_API_KEY=redacted-test-key cargo run -- serve --dry-run
expect_output "serve dry-run custom port" "listen: http://127.0.0.1:9090" cargo run -- serve --port 9090 --dry-run
expect_failure_output "serve rejects bad port" "serve --port must be an integer from 1 to 65535" cargo run -- serve --port 0 --dry-run
expect_failure_output "serve rejects unauthenticated network bind" "serve --host outside loopback requires --api-key or NANOCAMELID_API_KEY" cargo run -- serve --host 0.0.0.0 --dry-run
expect_output "serve allows authenticated network bind" "listen: http://0.0.0.0:8080" env NANOCAMELID_API_KEY=redacted-test-key cargo run -- serve --host 0.0.0.0 --dry-run

echo "==> Checking local API server HTTP smoke..."
check_local_api_smoke
expect_file_contains "API docs missing body error" '| `400` | `missing_body` |' docs/API.md
expect_file_contains "API docs invalid content length error" '| `400` | `invalid_content_length` |' docs/API.md
expect_file_contains "API docs request byte cap error" '| `413` | `request_too_large` |' docs/API.md

expect_output "models help lists scan" "nanocamelid models scan" cargo run -- models --help
expect_output "models list dry-run command" "list_command: nanocamelid models list --dir /mnt/nanocamelid/models" cargo run -- models list --dry-run --dir /mnt/nanocamelid/models --json
expect_output "models list json dry run" "\"command\":\"models list\"" cargo run -- models list --dry-run --dir /mnt/nanocamelid/models --json
expect_output "models scan dry-run command" "scan_command: nanocamelid models scan --dir /mnt/nanocamelid/models" cargo run -- models scan --dry-run --dir /mnt/nanocamelid/models --json
expect_output "models scan json dry run" "\"command\":\"models scan\"" cargo run -- models scan --dry-run --dir /mnt/nanocamelid/models --json
expect_output "models inspect 1b dry-run" "inspect_command: nanocamelid inspect 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- models inspect 1b --dry-run
expect_output "models inspect 3b dry-run" "inspect_command: nanocamelid inspect /mnt/nanocamelid/models/Llama-3.2-3B-Instruct-Q4_0.gguf" env -u NANOCAMELID_SMOKE_GGUF cargo run -- models inspect 3b --dry-run
expect_output "models inspect explicit GGUF dry-run" "inspect_command: nanocamelid inspect /models/custom.gguf" cargo run -- models inspect /models/custom.gguf --dry-run
expect_failure_output "models missing dir" "models directory not found" cargo run -- models list --dir /definitely/missing/nanocamelid-models
expect_failure_output "models missing command" "missing models command" cargo run -- models
expect_failure_output "models inspect missing model" "models inspect requires <model.gguf|1b|3b>" cargo run -- models inspect --dry-run
expect_failure_output "models inspect rejects unknown alias" "models inspect model argument must be a .gguf path or 1b/3b alias" cargo run -- models inspect badalias --dry-run
expect_failure_output "models inspect rejects non-GGUF path" "models inspect model argument must be a .gguf path or 1b/3b alias" cargo run -- models inspect /models/not-a-gguf --dry-run

echo "==> Checking 1B model audit CLI dry run..."
cargo run -- model 1b --dry-run
expect_output "model 1b shape audit dry run" "shape_audit: enabled" cargo run -- model 1b --dry-run
expect_output "model 1b success marker dry run" "status_on_success: model_1b_status: ok" cargo run -- model 1b --dry-run
expect_output "model 1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" cargo run -- model 1b --dry-run
expect_output "model 1b selected quantization dry run" "quantization: q8_0" cargo run -- model 1b --dry-run
expect_output "model 1b forced q4 source" "selected_source: workspace Q4_0 requested" cargo run -- model 1b --q4 --dry-run
expect_output "model 1b forced q4 path" "selected_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" cargo run -- model 1b --q4 --dry-run
expect_output "model 1b forced q8 source" "selected_source: workspace Q8_0 requested" cargo run -- model 1b --q8 --dry-run
expect_failure "model 1b conflicting quant selectors" cargo run -- model 1b --q4 --q8 --dry-run
expect_output "model 1b json records quantization" "\"quantization\":\"q8_0\"" cargo run -- model 1b --dry-run
expect_output "model 1b shape json marker dry run" "\"shape\":\"llama32_1b\",\"shape_ready\":true" cargo run -- model 1b --dry-run
expect_output "model 1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- model 1b --dry-run
expect_output "model 1b inspect follow-up command" "inspect_command: nanocamelid inspect 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- model 1b --dry-run
expect_output "model 1b smoke follow-up command" "smoke_command: nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello in one sentence.' 8" cargo run -- model 1b --dry-run
expect_output "model 1b ready follow-up command" "ready_command: nanocamelid ready 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- model 1b --dry-run
expect_output "model 1b evidence follow-up command" "evidence_command: nanocamelid evidence 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- model 1b --dry-run

echo "==> Checking 1B model audit CLI rejects non-GGUF model args..."
expect_failure "model 1b invalid model arg" cargo run -- model 1b not-a-model --dry-run
expect_failure "model 1b invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- model 1b --dry-run

echo "==> Checking 1B inspect CLI dry run..."
cargo run -- inspect 1b --dry-run
expect_output "inspect 1b q4 model audit" "q4_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" cargo run -- inspect 1b --dry-run
expect_output "inspect 1b q8 model audit" "q8_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- inspect 1b --dry-run
expect_output "inspect 1b selected source" "selected_source: " cargo run -- inspect 1b --dry-run
expect_output "inspect 1b shape audit dry run" "shape_audit: enabled" cargo run -- inspect 1b --dry-run
expect_output "inspect 1b success marker dry run" "status_on_success: inspect_1b_status: ok" cargo run -- inspect 1b --dry-run
expect_output "inspect 1b json success marker dry run" "\"target\":\"llama32-1b\",\"command\":\"inspect\",\"status\":\"ok\"" cargo run -- inspect 1b --dry-run
expect_output "inspect 1b json records quantization" "\"quantization\":\"q8_0\"" cargo run -- inspect 1b --dry-run
expect_output "inspect 1b shape json marker dry run" "\"shape\":\"llama32_1b\",\"shape_ready\":true" cargo run -- inspect 1b --dry-run
expect_output "inspect 1b explicit model path" "model: /models/custom.gguf" cargo run -- inspect 1b /models/custom.gguf --dry-run
expect_output "inspect 1b explicit command" "inspect_command: nanocamelid inspect 1b /models/custom.gguf" cargo run -- inspect 1b /models/custom.gguf --dry-run
expect_output "inspect 1b forced q4 source" "selected_source: workspace Q4_0 requested" cargo run -- inspect 1b --q4 --dry-run
expect_output "inspect 1b forced q4 path" "model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" cargo run -- inspect 1b --q4 --dry-run
expect_failure "inspect 1b invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- inspect 1b --dry-run
expect_failure "inspect 1b invalid explicit model path" cargo run -- inspect 1b /models/not-a-gguf --dry-run
expect_failure "inspect 1b conflicting quant selectors" cargo run -- inspect 1b --q4 --q8 --dry-run
expect_failure "inspect 1b extra argument" cargo run -- inspect 1b /models/custom.gguf extra --dry-run

echo "==> Checking 1B generate CLI dry run..."
cargo run -- generate 1b --dry-run
expect_output "generate 1b selected source" "selected_source: " cargo run -- generate 1b --dry-run
expect_output "generate 1b env selected source" "selected_source: NANOCAMELID_MODEL_GGUF" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=/models/custom.gguf cargo run -- generate 1b --dry-run
expect_output "generate 1b smoke env selected source" "selected_source: NANOCAMELID_SMOKE_GGUF" env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf NANOCAMELID_MODEL_GGUF=/models/custom.gguf cargo run -- generate 1b --dry-run
expect_output "generate 1b smoke env model" "model: /models/smoke.gguf" env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf NANOCAMELID_MODEL_GGUF=/models/custom.gguf cargo run -- generate 1b --dry-run
expect_output "generate 1b default model" "model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- generate 1b --dry-run
expect_output "generate 1b placeholder prompt" "prompt: <prompt>" cargo run -- generate 1b --dry-run
expect_output "generate 1b command" "generate_command: nanocamelid generate /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf '<prompt>' 0 128" cargo run -- generate 1b --dry-run
expect_output "generate 1b shape audit dry run" "shape_audit: enabled" cargo run -- generate 1b --dry-run
expect_output "generate 1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- generate 1b --dry-run
expect_output "generate 1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- generate 1b --dry-run
expect_output "generate 1b context-limited command" "generate_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid generate /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf '<prompt>' 0 128" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- generate 1b --dry-run
expect_output "generate 1b prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_PREFILL_BATCH=32 cargo run -- generate 1b --dry-run
expect_failure "generate 1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad cargo run -- generate 1b --dry-run
expect_failure "generate 1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad cargo run -- generate 1b --dry-run
expect_failure "generate 1b invalid smoke env model path" env NANOCAMELID_SMOKE_GGUF=not-a-model cargo run -- generate 1b --dry-run
expect_failure "generate 1b invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- generate 1b --dry-run
expect_failure "generate 1b invalid alias model path" cargo run -- generate 1b /models/not-a-gguf --dry-run
expect_failure_output "generate invalid explicit model path" "model argument must be a .gguf path" cargo run -- generate /models/not-a-gguf hi --dry-run
expect_failure_output "generate prompt without env needs model" "missing GGUF model path" cargo run -- generate "Say hello" --dry-run

echo "==> Checking 1B chat CLI dry run..."
cargo run -- chat 1b --dry-run
expect_output "chat 1b selected source" "selected_source: " cargo run -- chat 1b --dry-run
expect_output "chat 1b env selected source" "selected_source: NANOCAMELID_MODEL_GGUF" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=/models/custom.gguf cargo run -- chat 1b --dry-run
expect_output "chat 1b smoke env selected source" "selected_source: NANOCAMELID_SMOKE_GGUF" env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf NANOCAMELID_MODEL_GGUF=/models/custom.gguf cargo run -- chat 1b --dry-run
expect_output "chat 1b smoke env model" "model: /models/smoke.gguf" env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf NANOCAMELID_MODEL_GGUF=/models/custom.gguf cargo run -- chat 1b --dry-run
expect_output "chat 1b default model" "model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- chat 1b --dry-run
expect_output "chat 1b placeholder prompt" "prompt: <prompt>" cargo run -- chat 1b --dry-run
expect_output "chat 1b command" "chat_command: nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf '<prompt>' 0 128" cargo run -- chat 1b --dry-run
expect_output "chat 1b shape audit dry run" "shape_audit: enabled" cargo run -- chat 1b --dry-run
expect_output "chat 1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- chat 1b --dry-run
expect_output "chat 1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- chat 1b --dry-run
expect_output "chat 1b context-limited command" "chat_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf '<prompt>' 0 128" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- chat 1b --dry-run
expect_output "chat 1b prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_PREFILL_BATCH=32 cargo run -- chat 1b --dry-run
expect_failure "chat 1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad cargo run -- chat 1b --dry-run
expect_failure "chat 1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad cargo run -- chat 1b --dry-run
expect_failure "chat 1b invalid smoke env model path" env NANOCAMELID_SMOKE_GGUF=not-a-model cargo run -- chat 1b --dry-run
expect_failure "chat 1b invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- chat 1b --dry-run
expect_failure "chat 1b invalid alias model path" cargo run -- chat 1b /models/not-a-gguf --dry-run
expect_failure_output "chat invalid explicit model path" "model argument must be a .gguf path" cargo run -- chat /models/not-a-gguf hi --dry-run
expect_failure_output "chat prompt without env needs model" "missing GGUF model path" cargo run -- chat "Say hello" --dry-run

echo "==> Checking 1B smoke CLI dry run..."
cargo run -- smoke 1b --dry-run
expect_output "smoke 1b q4 model audit" "q4_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b q8 model audit" "q8_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b selected source" "selected_source: " cargo run -- smoke 1b --dry-run
expect_output "smoke 1b selected quantization" "quantization: q8_0" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b forced q4 source" "selected_source: workspace Q4_0 requested" cargo run -- smoke 1b --q4 --dry-run
expect_output "smoke 1b forced q4 path" "model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" cargo run -- smoke 1b --q4 --dry-run
expect_output "smoke 1b forced q8 source" "selected_source: workspace Q8_0 requested" cargo run -- smoke 1b --q8 --dry-run
expect_failure "smoke 1b conflicting quant selectors" cargo run -- smoke 1b --q4 --q8 --dry-run
expect_output "smoke 1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- smoke 1b --dry-run
expect_output "smoke 1b context-limited command" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello in one sentence.' 8" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- smoke 1b --dry-run
expect_output "smoke 1b shape audit dry run" "shape_audit: enabled" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b success marker dry run" "status_on_success: smoke_1b_status: ok" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b json records quantization" "\"quantization\":\"q8_0\"" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b json records shape audit" "\"shape\":\"llama32_1b\",\"shape_ready\":true" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b json records prompt" "\"smoke_prompt\":\"Say hello in one sentence.\"" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- smoke 1b --dry-run
expect_output_order "smoke 1b model audit before smoke" "model_command: nanocamelid model 1b" "smoke_command: nanocamelid smoke 1b" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_PREFILL_BATCH=32 cargo run -- smoke 1b --dry-run
expect_output "smoke 1b json records prefill batch" "\"prefill_batch\":32" env NANOCAMELID_PREFILL_BATCH=32 cargo run -- smoke 1b --dry-run
expect_output "smoke 1b command carries prefill batch" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello in one sentence.' 8" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 cargo run -- smoke 1b --dry-run
expect_failure "smoke 1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad cargo run -- smoke 1b --dry-run
expect_failure "smoke 1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad cargo run -- smoke 1b --dry-run
expect_failure "smoke 1b invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- smoke 1b --dry-run
expect_failure "smoke 1b invalid explicit model path" cargo run -- smoke 1b /models/not-a-gguf --dry-run

echo "==> Checking 1B smoke CLI rejects invalid token count..."
expect_failure "smoke 1b invalid token count" cargo run -- smoke 1b chat "Say hello in one sentence." 0 --dry-run

echo "==> Checking 1B readiness CLI dry run..."
cargo run -- ready 1b --dry-run
expect_output "ready 1b help documents prefill batch" "NANOCAMELID_PREFILL_BATCH" cargo run -- ready 1b --help
expect_output "ready 1b q4 model audit" "q4_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" cargo run -- ready 1b --dry-run
expect_output "ready 1b q8 model audit" "q8_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- ready 1b --dry-run
expect_output "ready 1b shape audit dry run" "shape_audit: enabled" cargo run -- ready 1b --dry-run
expect_output "ready 1b success marker dry run" "status_on_success: ready_1b_status: ok" cargo run -- ready 1b --dry-run
expect_output "ready 1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" cargo run -- ready 1b --dry-run
expect_output "ready 1b selected quantization" "quantization: q8_0" cargo run -- ready 1b --dry-run
expect_output "ready 1b forced q4 source" "selected_source: workspace Q4_0 requested" cargo run -- ready 1b --q4 --dry-run
expect_output "ready 1b forced q4 path" "model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" cargo run -- ready 1b --q4 --dry-run
expect_output "ready 1b forced q8 source" "selected_source: workspace Q8_0 requested" cargo run -- ready 1b --q8 --dry-run
expect_failure "ready 1b conflicting quant selectors" cargo run -- ready 1b --q4 --q8 --dry-run
expect_output "ready 1b json records quantization" "\"quantization\":\"q8_0\"" cargo run -- ready 1b --dry-run
expect_output "ready 1b json records probe" "\"probe\":true" cargo run -- ready 1b --dry-run
expect_output "ready 1b json records shape audit" "\"shape\":\"llama32_1b\",\"shape_ready\":true" cargo run -- ready 1b --dry-run
expect_output "ready 1b json records smoke prompt" "\"smoke_prompt\":\"Say hello in one sentence.\"" cargo run -- ready 1b --dry-run
expect_output "ready 1b json records chat prompt" "\"chat_prompt\":\"Say hello in one sentence.\"" cargo run -- ready 1b --dry-run
expect_output "ready 1b json records chat temperature" "\"chat_temp\":0" cargo run -- ready 1b --dry-run
expect_output "ready 1b no-chat json success marker dry run" "\"direct_chat\":false,\"chat_prompt\":null,\"chat_tokens\":null,\"chat_temp\":null" cargo run -- ready 1b --no-chat --dry-run
expect_output "ready 1b prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_PREFILL_BATCH=32 cargo run -- ready 1b --dry-run
expect_output "ready 1b json records prefill batch" "\"prefill_batch\":32" env NANOCAMELID_PREFILL_BATCH=32 cargo run -- ready 1b --dry-run
expect_output "ready 1b smoke command carries prefill batch" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello in one sentence.' 8" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 cargo run -- ready 1b --dry-run
expect_output "ready 1b chat command carries prefill batch" "chat_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 'Say hello in one sentence.' 0 8" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 cargo run -- ready 1b --dry-run
expect_output "ready 1b probe command" "probe_command: nanocamelid probe" cargo run -- ready 1b --dry-run
expect_output "ready 1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- ready 1b --dry-run
expect_output_order "ready 1b probe before inspect" "probe_command: nanocamelid probe" "inspect_command: nanocamelid inspect 1b" cargo run -- ready 1b --dry-run
expect_output_order "ready 1b model audit before inspect" "model_command: nanocamelid model 1b" "inspect_command: nanocamelid inspect 1b" cargo run -- ready 1b --dry-run
expect_output "ready 1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- ready 1b --dry-run
expect_output "ready 1b context-limited smoke command" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello in one sentence.' 8" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- ready 1b --dry-run
expect_output "ready 1b context-limited chat command" "chat_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 'Say hello in one sentence.' 0 8" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- ready 1b --dry-run
expect_failure "ready 1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad cargo run -- ready 1b --dry-run
expect_failure "ready 1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad cargo run -- ready 1b --dry-run
expect_failure "ready 1b invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- ready 1b --dry-run
expect_failure "ready 1b invalid explicit model path" cargo run -- ready 1b /models/not-a-gguf --dry-run
expect_failure "ready 1b invalid direct chat toggle" env NANOCAMELID_READY_CHAT=flase cargo run -- ready 1b --dry-run

echo "==> Checking 1B readiness CLI rejects invalid direct chat env..."
expect_failure "ready 1b invalid direct chat temperature" env NANOCAMELID_READY_TEMP=bad cargo run -- ready 1b --dry-run
expect_failure "ready 1b invalid direct chat token count" env NANOCAMELID_READY_TOKENS=0 cargo run -- ready 1b --dry-run

echo "==> Checking 1B readiness CLI ignores direct chat env when chat is disabled..."
env NANOCAMELID_READY_CHAT=flase cargo run -- ready 1b --no-chat --dry-run
env NANOCAMELID_READY_TEMP=bad NANOCAMELID_READY_TOKENS=0 cargo run -- ready 1b --no-chat --dry-run

echo "==> Checking 1B evidence CLI dry run..."
cargo run -- evidence 1b --dry-run
expect_output "evidence 1b help documents context packs" "NANOCAMELID_CONTEXT_PACKS" cargo run -- evidence 1b --help
expect_output "evidence 1b q4 model audit" "q4_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b q8 model audit" "q8_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b selected source" "selected_source: " cargo run -- evidence 1b --dry-run
expect_output "evidence 1b selected quantization" "quantization: q8_0" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b shape audit dry run" "shape_audit: enabled" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b success marker dry run" "status_on_success: evidence_1b_status: ok" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b json records quantization" "\"quantization\":\"q8_0\"" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b json records shape audit" "\"shape\":\"llama32_1b\",\"shape_ready\":true" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b json records smoke prompt" "\"smoke_prompt\":\"Say hello in one sentence.\"" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b json records prefill batch" "\"prefill_batch\":16" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b json records context caps" "\"context_pack_caps\":[512,1024,2048,4096,8192]" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b json records prefill batches" "\"prefill_batches\":[1,16,32,64]" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- evidence 1b --dry-run
expect_output "evidence 1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b ready no-chat command" "ready_command: nanocamelid ready 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello in one sentence.' 8 --no-chat" cargo run -- evidence 1b --dry-run
expect_output_order "evidence 1b model audit before ready" "model_command: nanocamelid model 1b" "ready_command: nanocamelid ready 1b" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b context command" "context_512_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello in one sentence.' 8" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b prefill command" "prefill_bench_command: nanocamelid bench 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 'Explain one practical Raspberry Pi inference bottleneck in two short sentences.' 2 0.0 '1,16,32,64'" cargo run -- evidence 1b --dry-run
expect_output "evidence 1b context-limited ready command" "ready_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid ready 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello in one sentence.' 8 --no-chat" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- evidence 1b --dry-run
expect_output "evidence 1b context-limited prefill command" "prefill_bench_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid bench 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 'Explain one practical Raspberry Pi inference bottleneck in two short sentences.' 2 0.0 '1,16,32,64'" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- evidence 1b --dry-run
expect_output "evidence 1b command carries prefill batch" "ready_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid ready 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello in one sentence.' 8 --no-chat" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 cargo run -- evidence 1b --dry-run
expect_output "evidence 1b prefill command carries prefill batch" "prefill_bench_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid bench 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 'Explain one practical Raspberry Pi inference bottleneck in two short sentences.' 2 0.0 '1,16,32,64'" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 cargo run -- evidence 1b --dry-run
expect_output "evidence 1b explicit model command" "ready_command: nanocamelid ready 1b /models/custom.gguf chat 'Say hello in one sentence.' 8 --no-chat" cargo run -- evidence 1b /models/custom.gguf --dry-run
expect_failure "evidence 1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad cargo run -- evidence 1b --dry-run
expect_failure "evidence 1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad cargo run -- evidence 1b --dry-run
expect_failure "evidence 1b invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- evidence 1b --dry-run
expect_failure "evidence 1b invalid explicit model path" cargo run -- evidence 1b /models/not-a-gguf --dry-run
expect_failure "evidence 1b invalid context pack" env NANOCAMELID_CONTEXT_PACKS=512,bad cargo run -- evidence 1b --dry-run
expect_failure "evidence 1b empty context pack" env NANOCAMELID_CONTEXT_PACKS=512,,1024 cargo run -- evidence 1b --dry-run
expect_failure "evidence 1b duplicate context pack" env NANOCAMELID_CONTEXT_PACKS=512,512 cargo run -- evidence 1b --dry-run
expect_failure "evidence 1b invalid prefill batches" env NANOCAMELID_PREFILL_BATCHES=1,bad cargo run -- evidence 1b --dry-run
expect_failure "evidence 1b empty prefill batch" env NANOCAMELID_PREFILL_BATCHES=1,,16 cargo run -- evidence 1b --dry-run

echo "==> Checking 1B TUI CLI dry run..."
cargo run -- tui 1b --dry-run
expect_output "tui 1b selected source" "selected_source: " cargo run -- tui 1b --dry-run
expect_output "tui 1b env selected source" "selected_source: NANOCAMELID_MODEL_GGUF" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=/models/custom.gguf cargo run -- tui 1b --dry-run
expect_output "tui 1b smoke env selected source" "selected_source: NANOCAMELID_SMOKE_GGUF" env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf NANOCAMELID_MODEL_GGUF=/models/custom.gguf cargo run -- tui 1b --dry-run
expect_output "tui 1b smoke env model" "model: /models/smoke.gguf" env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf NANOCAMELID_MODEL_GGUF=/models/custom.gguf cargo run -- tui 1b --dry-run
expect_output "tui 1b dry-run command" "tui_command: nanocamelid tui /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 0 128" cargo run -- tui 1b --dry-run
expect_output "tui 1b shape audit dry run" "shape_audit: enabled" cargo run -- tui 1b --dry-run
expect_output "tui 1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- tui 1b --dry-run
expect_output "tui 1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- tui 1b --dry-run
expect_output "tui 1b context-limited command" "tui_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid tui /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 0 128" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- tui 1b --dry-run
expect_output "tui 1b prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_PREFILL_BATCH=32 cargo run -- tui 1b --dry-run
expect_output "tui 1b command carries prefill batch" "tui_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid tui /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 0 128" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 cargo run -- tui 1b --dry-run
expect_failure "tui 1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad cargo run -- tui 1b --dry-run
expect_failure "tui 1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad cargo run -- tui 1b --dry-run
expect_failure "tui 1b invalid smoke env model path" env NANOCAMELID_SMOKE_GGUF=not-a-model cargo run -- tui 1b --dry-run
expect_failure "tui 1b invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- tui 1b --dry-run
expect_failure "tui 1b invalid alias model path" cargo run -- tui 1b /models/not-a-gguf --dry-run
expect_failure_output "tui invalid explicit model path" "model argument must be a .gguf path" cargo run -- tui /models/not-a-gguf --dry-run

echo "==> Checking 1B prefill benchmark CLI dry run..."
cargo run -- bench 1b --dry-run
expect_output "bench 1b nested help" "bench 1b [model.gguf]" cargo run -- bench 1b --help
expect_output "bench 1b q4 model audit" "q4_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" cargo run -- bench 1b --dry-run
expect_output "bench 1b q8 model audit" "q8_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- bench 1b --dry-run
expect_output "bench 1b selected source" "selected_source: " cargo run -- bench 1b --dry-run
expect_output "bench 1b selected quantization" "quantization: q8_0" cargo run -- bench 1b --dry-run
expect_output "bench 1b smoke env override" "selected_source: NANOCAMELID_SMOKE_GGUF" env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf cargo run -- bench 1b --dry-run
expect_output "bench 1b smoke env model" "model: /models/smoke.gguf" env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf cargo run -- bench 1b --dry-run
expect_output "bench 1b shape audit dry run" "shape_audit: enabled" cargo run -- bench 1b --dry-run
expect_output "bench 1b smoke gate dry run" "smoke_gate: enabled" cargo run -- bench 1b --dry-run
expect_output "bench 1b success marker dry run" "status_on_success: prefill_bench_1b_status: ok" cargo run -- bench 1b --dry-run
expect_output "bench 1b json success marker dry run" "\"benchmark\":\"llama32-1b-prefill\",\"target\":\"llama32-1b\",\"status\":\"ok\"" cargo run -- bench 1b --dry-run
expect_output "bench 1b json records quantization" "\"quantization\":\"q8_0\"" cargo run -- bench 1b --dry-run
expect_output "bench 1b json records probe" "\"probe\":true" cargo run -- bench 1b --dry-run
expect_output "bench 1b json records shape audit" "\"shape\":\"llama32_1b\",\"shape_ready\":true" cargo run -- bench 1b --dry-run
expect_output "bench 1b json records prompt" "\"prompt\":\"Explain one practical Raspberry Pi inference bottleneck in two short sentences.\"" cargo run -- bench 1b --dry-run
expect_output "bench 1b default batches dry run" "batches: 1 16 32 64" cargo run -- bench 1b --dry-run
expect_output "bench 1b probe command" "probe_command: nanocamelid probe" cargo run -- bench 1b --dry-run
expect_output "bench 1b smoke command" "smoke_command: NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Explain one practical Raspberry Pi inference bottleneck in two short sentences.' 2" cargo run -- bench 1b --dry-run
expect_output "bench 1b batch command" "batch_16_command: NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot NANOCAMELID_PREFILL_BATCH=16 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 'Explain one practical Raspberry Pi inference bottleneck in two short sentences.' 0.0 2" cargo run -- bench 1b --dry-run
expect_output "bench 1b inspect command" "inspect_command: nanocamelid inspect 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- bench 1b --dry-run
expect_output_order "bench 1b probe before model audit" "probe_command: nanocamelid probe" "model_command: nanocamelid model 1b" cargo run -- bench 1b --dry-run
expect_output_order "bench 1b inspect before smoke" "inspect_command: nanocamelid inspect 1b" "smoke_command: NANOCAMELID_Q8_DOT_SDOT=1" cargo run -- bench 1b --dry-run
expect_output_order "bench 1b smoke before batch" "smoke_command: NANOCAMELID_Q8_DOT_SDOT=1" "batch_16_command: NANOCAMELID_Q8_DOT_SDOT=1" cargo run -- bench 1b --dry-run
expect_output "bench 1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- bench 1b --dry-run
expect_output "bench 1b context-limited smoke command" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Explain one practical Raspberry Pi inference bottleneck in two short sentences.' 2" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- bench 1b --dry-run
expect_output "bench 1b context-limited batch command" "batch_16_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot NANOCAMELID_PREFILL_BATCH=16 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 'Explain one practical Raspberry Pi inference bottleneck in two short sentences.' 0.0 2" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- bench 1b --dry-run
expect_failure_output "bench 1b missing model dry-run hint" "nanocamelid bench 1b --dry-run" env NANOCAMELID_SMOKE_GGUF=/tmp/nanocamelid-missing-validation.gguf cargo run -- bench 1b
expect_failure "bench 1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad cargo run -- bench 1b --dry-run
expect_failure "bench 1b invalid token count" cargo run -- bench 1b prompt 0 --dry-run
expect_failure "bench 1b invalid temp" cargo run -- bench 1b prompt 1 bad --dry-run
expect_failure "bench 1b invalid batch" cargo run -- bench 1b prompt 1 0.0 0 --dry-run
expect_failure "bench 1b duplicate batch" cargo run -- bench 1b prompt 1 0.0 16,32,16 --dry-run
expect_failure "bench 1b invalid smoke env model path" env NANOCAMELID_SMOKE_GGUF=not-a-model cargo run -- bench 1b --dry-run
expect_failure "bench 1b invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- bench 1b --dry-run
expect_failure "bench 1b invalid explicit model path" cargo run -- bench 1b /models/not-a-gguf --dry-run
expect_failure "bench 1b unknown option" cargo run -- bench 1b --oops --dry-run

echo "==> Checking 1B model audit dry run..."
./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b shape audit dry run" "shape_audit: enabled" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b success marker dry run" "status_on_success: model_1b_status: ok" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b selected quantization dry run" "quantization: q8_0" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b forced q4 source" "selected_source: workspace Q4_0 requested" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/model-1b.sh --q4 --dry-run
expect_output "model-1b forced q4 path" "selected_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/model-1b.sh --q4 --dry-run
expect_output "model-1b forced q8 source" "selected_source: workspace Q8_0 requested" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/model-1b.sh --q8 --dry-run
expect_failure "model-1b conflicting quant selectors" ./scripts/pi/model-1b.sh --q4 --q8 --dry-run
expect_failure_output "model-1b quant conflicts with explicit model" "1B model audit quantization selector cannot be combined with an explicit model path." ./scripts/pi/model-1b.sh --q4 /models/custom.gguf --dry-run
expect_failure_output "model-1b quant conflicts with env model" "1B model audit quantization selector cannot be combined with NANOCAMELID_SMOKE_GGUF or NANOCAMELID_MODEL_GGUF." env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf ./scripts/pi/model-1b.sh --q8 --dry-run
expect_output "model-1b json records quantization" "\"quantization\":\"q8_0\"" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b json shape marker dry run" "\"shape\":\"llama32_1b\",\"shape_ready\":true" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b inspect follow-up command" "inspect_command: nanocamelid inspect 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b smoke follow-up command" "smoke_command: nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b ready follow-up command" "ready_command: nanocamelid ready 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b evidence follow-up command" "evidence_command: nanocamelid evidence 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/model-1b.sh --dry-run

echo "==> Checking 1B model audit rejects non-GGUF model args..."
expect_failure "model-1b invalid model arg" ./scripts/pi/model-1b.sh not-a-model --dry-run
expect_failure "model-1b invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model ./scripts/pi/model-1b.sh --dry-run
expect_failure "model-1b repo-local target dir" bash -c 'tmp="$(mktemp "${TMPDIR:-/tmp}/nanocamelid-model-1b.XXXXXX").gguf"; : >"$tmp"; trap "rm -f \"$tmp\"" EXIT; CARGO_TARGET_DIR=target ./scripts/pi/model-1b.sh "$tmp"'

echo "==> Checking 1B Pi smoke launcher dry run..."
./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b help documents prefill batch" "NANOCAMELID_PREFILL_BATCH" ./scripts/pi/smoke-1b.sh --help
expect_output "smoke-1b help documents quant selectors" "--q4, --q8" ./scripts/pi/smoke-1b.sh --help
expect_output "smoke-1b q4 model audit" "q4_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b q8 model audit" "q8_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b selected source" "selected_source: " ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b selected quantization" "quantization: q8_0" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b forced q4 source" "selected_source: workspace Q4_0 requested" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/smoke-1b.sh --q4 --dry-run
expect_output "smoke-1b forced q4 path" "model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/smoke-1b.sh --q4 --dry-run
expect_output "smoke-1b forced q8 source" "selected_source: workspace Q8_0 requested" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/smoke-1b.sh --q8 --dry-run
expect_failure_output "smoke-1b conflicting quant selectors" "Only one 1B smoke quantization selector may be provided." ./scripts/pi/smoke-1b.sh --q4 --q8 --dry-run
expect_failure_output "smoke-1b quant conflicts with explicit model" "1B smoke quantization selector cannot be combined with an explicit model path." ./scripts/pi/smoke-1b.sh --q4 /models/custom.gguf --dry-run
expect_failure_output "smoke-1b quant conflicts with env model" "1B smoke quantization selector cannot be combined with NANOCAMELID_SMOKE_GGUF or NANOCAMELID_MODEL_GGUF." env NANOCAMELID_MODEL_GGUF=/models/model.gguf ./scripts/pi/smoke-1b.sh --q8 --dry-run
expect_output "smoke-1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b context-limited command" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b shape audit dry run" "shape_audit: enabled" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b success marker dry run" "status_on_success: smoke_1b_status: ok" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b json records quantization" "\"quantization\":\"q8_0\"" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b json records shape audit" "\"shape\":\"llama32_1b\",\"shape_ready\":true" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b json records prompt" "\"smoke_prompt\":\"Say hello in one sentence.\"" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/smoke-1b.sh --dry-run
expect_output_order "smoke-1b model audit before smoke" "model_command: nanocamelid model 1b" "smoke_command: nanocamelid smoke 1b" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b json records prefill batch" "\"prefill_batch\":32" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b command carries prefill batch" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/smoke-1b.sh --dry-run
expect_failure "smoke-1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad ./scripts/pi/smoke-1b.sh --dry-run
expect_failure "smoke-1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b prompt without explicit kind" "smoke_kind: chat" ./scripts/pi/smoke-1b.sh "Say hello in one sentence." 3 --dry-run
expect_output "smoke-1b token override without explicit kind" "smoke_tokens: 3" ./scripts/pi/smoke-1b.sh "Say hello in one sentence." 3 --dry-run
expect_failure "smoke-1b invalid q8 kind" ./scripts/pi/smoke-1b.sh q8-broken --dry-run
expect_failure "smoke-1b invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model ./scripts/pi/smoke-1b.sh --dry-run
expect_failure "smoke-1b invalid explicit model path" ./scripts/pi/smoke-1b.sh /models/not-a-gguf --dry-run
expect_failure "smoke-1b repo-local target dir" env CARGO_TARGET_DIR=target ./scripts/pi/smoke-1b.sh

echo "==> Checking 1B Pi readiness launcher dry run..."
./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b help documents prefill batch" "NANOCAMELID_PREFILL_BATCH" ./scripts/pi/ready-1b.sh --help
expect_output_count "ready-1b smoke prompt printed once" "smoke_prompt:" 1 ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b q4 model audit" "q4_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b q8 model audit" "q8_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b shape audit dry run" "shape_audit: enabled" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b success marker dry run" "status_on_success: ready_1b_status: ok" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b selected quantization" "quantization: q8_0" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b forced q4 source" "selected_source: workspace Q4_0 requested" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/ready-1b.sh --q4 --dry-run
expect_output "ready-1b forced q4 path" "model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/ready-1b.sh --q4 --dry-run
expect_output "ready-1b forced q8 source" "selected_source: workspace Q8_0 requested" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/ready-1b.sh --q8 --dry-run
expect_failure "ready-1b conflicting quant selectors" ./scripts/pi/ready-1b.sh --q4 --q8 --dry-run
expect_output "ready-1b json records quantization" "\"quantization\":\"q8_0\"" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b json records probe" "\"probe\":true" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b json records shape audit" "\"shape\":\"llama32_1b\",\"shape_ready\":true" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b json records smoke prompt" "\"smoke_prompt\":\"Say hello in one sentence.\"" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b json records chat prompt" "\"chat_prompt\":\"Say hello in one sentence.\"" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b json records chat temperature" "\"chat_temp\":0.0" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b no-chat json success marker dry run" "\"direct_chat\":false,\"chat_prompt\":null,\"chat_tokens\":null,\"chat_temp\":null" ./scripts/pi/ready-1b.sh --no-chat --dry-run
expect_output "ready-1b selected source" "selected_source: " ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b json records prefill batch" "\"prefill_batch\":32" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b smoke command carries prefill batch" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b chat command carries prefill batch" "chat_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf Say\\ hello\\ in\\ one\\ sentence. 0.0 8" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b probe command" "probe_command: nanocamelid probe" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/ready-1b.sh --dry-run
expect_output_order "ready-1b probe before inspect" "probe_command: nanocamelid probe" "inspect_command: nanocamelid inspect 1b" ./scripts/pi/ready-1b.sh --dry-run
expect_output_order "ready-1b model audit before inspect" "model_command: nanocamelid model 1b" "inspect_command: nanocamelid inspect 1b" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b context-limited smoke command" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b context-limited chat command" "chat_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf Say\\ hello\\ in\\ one\\ sentence. 0.0 8" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/ready-1b.sh --dry-run
expect_failure "ready-1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad ./scripts/pi/ready-1b.sh --dry-run
expect_failure "ready-1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad ./scripts/pi/ready-1b.sh --dry-run
expect_failure "ready-1b invalid q8 kind" ./scripts/pi/ready-1b.sh q8-broken --dry-run
expect_failure "ready-1b invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model ./scripts/pi/ready-1b.sh --dry-run
expect_failure "ready-1b invalid explicit model path" ./scripts/pi/ready-1b.sh /models/not-a-gguf --dry-run
expect_failure "ready-1b repo-local target dir" env CARGO_TARGET_DIR=target ./scripts/pi/ready-1b.sh --no-chat

echo "==> Checking 1B Pi chat launcher dry run..."
./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b help documents prefill batch" "NANOCAMELID_PREFILL_BATCH" ./scripts/pi/chat-1b.sh --help
expect_output "chat-1b selected source" "selected_source: " ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b smoke env override" "selected_source: NANOCAMELID_SMOKE_GGUF" env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b smoke env model" "model: /models/smoke.gguf" env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b shape audit dry run" "shape_audit: enabled" ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b smoke-covered model audit" "model_command: covered by smoke_command" ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b context-limited smoke command" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 1" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b context-limited tui command" "tui_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid tui /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 0.0 64" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b smoke command carries prefill batch" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 1" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b tui command carries prefill batch" "tui_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid tui /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 0.0 64" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/chat-1b.sh --dry-run
expect_failure "chat-1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad ./scripts/pi/chat-1b.sh --dry-run
expect_failure "chat-1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad ./scripts/pi/chat-1b.sh --dry-run
expect_failure "chat-1b invalid smoke env model path" env NANOCAMELID_SMOKE_GGUF=not-a-model ./scripts/pi/chat-1b.sh --dry-run
expect_failure "chat-1b invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model ./scripts/pi/chat-1b.sh --dry-run
expect_failure "chat-1b invalid explicit model path" ./scripts/pi/chat-1b.sh /models/not-a-gguf --dry-run
expect_failure "chat-1b repo-local target dir" env CARGO_TARGET_DIR=target ./scripts/pi/chat-1b.sh

echo "==> Checking 1B Pi chat launcher rejects invalid temperature..."
expect_failure "chat-1b invalid temperature" env NANOCAMELID_TEMP=bad ./scripts/pi/chat-1b.sh --dry-run
expect_failure "chat-1b invalid smoke toggle" env NANOCAMELID_CHAT_SMOKE=flase ./scripts/pi/chat-1b.sh --dry-run

echo "==> Checking 1B Pi chat launcher ignores smoke env when smoke is disabled..."
env NANOCAMELID_CHAT_SMOKE=0 NANOCAMELID_CHAT_SMOKE_KIND=bad NANOCAMELID_CHAT_SMOKE_TOKENS=bad ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b disabled smoke keeps model audit" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" env NANOCAMELID_CHAT_SMOKE=0 ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b disabled smoke skips smoke command" "smoke_command: skipped" env NANOCAMELID_CHAT_SMOKE=0 ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b smoke off keeps model audit" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" env NANOCAMELID_CHAT_SMOKE=off ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b smoke off skips smoke command" "smoke_command: skipped" env NANOCAMELID_CHAT_SMOKE=off ./scripts/pi/chat-1b.sh --dry-run

echo "==> Checking 1B Pi prefill benchmark launcher dry run..."
./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill q4 model audit" "q4_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill q8 model audit" "q8_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill selected source" "selected_source: " ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill selected quantization" "quantization: q8_0" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill q4 selector source" "selected_source: workspace Q4_0 requested" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/bench-1b-prefill.sh --q4 --dry-run
expect_output "bench-1b-prefill q4 selector model" "model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/bench-1b-prefill.sh --q4 --dry-run
expect_failure_output "bench-1b-prefill conflicting quant selectors" "Only one 1B prefill benchmark quantization selector may be provided." ./scripts/pi/bench-1b-prefill.sh --q4 --q8 --dry-run
expect_failure_output "bench-1b-prefill quant conflicts with explicit model" "1B prefill benchmark quantization selector cannot be combined with an explicit model path." ./scripts/pi/bench-1b-prefill.sh --q4 /models/custom.gguf --dry-run
expect_failure_output "bench-1b-prefill quant conflicts with env model" "1B prefill benchmark quantization selector cannot be combined with NANOCAMELID_SMOKE_GGUF or NANOCAMELID_MODEL_GGUF." env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf ./scripts/pi/bench-1b-prefill.sh --q8 --dry-run
expect_output "bench-1b-prefill smoke env override" "selected_source: NANOCAMELID_SMOKE_GGUF" env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill smoke env model" "model: /models/smoke.gguf" env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill shape audit dry run" "shape_audit: enabled" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill smoke gate dry run" "smoke_gate: enabled" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill probe command" "probe_command: nanocamelid probe" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill inspect command" "inspect_command: nanocamelid inspect 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill smoke command" "smoke_command: NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. 2" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output_order "bench-1b-prefill probe before model audit" "probe_command: nanocamelid probe" "model_command: nanocamelid model 1b" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output_order "bench-1b-prefill inspect before smoke" "inspect_command: nanocamelid inspect 1b" "smoke_command: NANOCAMELID_Q8_DOT_SDOT=1" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output_order "bench-1b-prefill smoke before batch" "smoke_command: NANOCAMELID_Q8_DOT_SDOT=1" "batch_16_command: NANOCAMELID_Q8_DOT_SDOT=1" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill context-limited smoke command" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. 2" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill context-limited batch command" "batch_16_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot NANOCAMELID_PREFILL_BATCH=16 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. 0.0 2" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill success marker dry run" "status_on_success: prefill_bench_1b_status: ok" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill json success marker dry run" "\"benchmark\":\"llama32-1b-prefill\",\"target\":\"llama32-1b\",\"status\":\"ok\"" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill json records quantization" "\"quantization\":\"q8_0\"" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill json records probe" "\"probe\":true" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill json records shape audit" "\"shape\":\"llama32_1b\",\"shape_ready\":true" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill json records prompt" "\"prompt\":\"Explain one practical Raspberry Pi inference bottleneck in two short sentences.\"" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill json batches dry run" "\"batches\":[1,16,32,64]" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_failure "bench-1b-prefill invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_failure "bench-1b-prefill invalid smoke env model path" env NANOCAMELID_SMOKE_GGUF=not-a-model ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_failure "bench-1b-prefill invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_failure "bench-1b-prefill invalid explicit model path" ./scripts/pi/bench-1b-prefill.sh /models/not-a-gguf --dry-run
expect_failure "bench-1b-prefill unknown option" ./scripts/pi/bench-1b-prefill.sh --oops --dry-run
expect_failure "bench-1b-prefill repo-local target dir" env CARGO_TARGET_DIR=target ./scripts/pi/bench-1b-prefill.sh

echo "==> Checking 1B Pi prefill benchmark launcher rejects invalid generated token count..."
expect_failure "bench-1b-prefill invalid generated token count" env NANOCAMELID_PREFILL_TOKENS=0 ./scripts/pi/bench-1b-prefill.sh --dry-run

echo "==> Checking 1B Pi prefill benchmark launcher rejects invalid temperature..."
expect_failure "bench-1b-prefill invalid temperature" env NANOCAMELID_PREFILL_TEMP=bad ./scripts/pi/bench-1b-prefill.sh --dry-run

echo "==> Checking 1B Pi prefill benchmark launcher rejects invalid batch size..."
expect_failure "bench-1b-prefill invalid batch size" env NANOCAMELID_PREFILL_BATCHES=1,bad,32 ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_failure_output "bench-1b-prefill empty batch size" "Invalid prefill batch size: empty value" env NANOCAMELID_PREFILL_BATCHES=1,,32 ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_failure "bench-1b-prefill duplicate batch size" env NANOCAMELID_PREFILL_BATCHES=16,32,16 ./scripts/pi/bench-1b-prefill.sh --dry-run

echo "==> Checking 1B Pi context-pack launcher dry run..."
./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b help documents prefill batch" "NANOCAMELID_PREFILL_BATCH" ./scripts/pi/context-pack-1b.sh --help
expect_output "context-pack-1b help documents quant selectors" "--q4, --q8" ./scripts/pi/context-pack-1b.sh --help
expect_output "context-pack-1b selected source" "selected_source: " ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b selected quantization" "quantization: q8_0" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b forced q4 source" "selected_source: workspace Q4_0 requested" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/context-pack-1b.sh --q4 --dry-run
expect_output "context-pack-1b forced q4 path" "model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/context-pack-1b.sh --q4 --dry-run
expect_output "context-pack-1b forced q8 source" "selected_source: workspace Q8_0 requested" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/context-pack-1b.sh --q8 --dry-run
expect_failure_output "context-pack-1b conflicting quant selectors" "Only one 1B context-pack quantization selector may be provided." ./scripts/pi/context-pack-1b.sh --q4 --q8 --dry-run
expect_failure_output "context-pack-1b quant conflicts with explicit model" "1B context-pack quantization selector cannot be combined with an explicit model path." ./scripts/pi/context-pack-1b.sh --q4 /models/custom.gguf --dry-run
expect_failure_output "context-pack-1b quant conflicts with env model" "1B context-pack quantization selector cannot be combined with NANOCAMELID_SMOKE_GGUF or NANOCAMELID_MODEL_GGUF." env NANOCAMELID_MODEL_GGUF=/models/model.gguf ./scripts/pi/context-pack-1b.sh --q8 --dry-run
expect_output "context-pack-1b shape audit dry run" "shape_audit: enabled" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b success marker dry run" "status_on_success: context_pack_1b_status: ok" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b json records quantization" "\"quantization\":\"q8_0\"" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b json records shape audit" "\"shape\":\"llama32_1b\",\"shape_ready\":true" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b json records prompt" "\"smoke_prompt\":\"Say hello in one sentence.\"" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b json caps dry run" "\"context_caps\":[512,1024,2048,4096,8192]" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b json records prefill batch" "\"prefill_batch\":32" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b command carries prefill batch" "context_512_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b prompt without explicit kind" "smoke_kind: chat" ./scripts/pi/context-pack-1b.sh "Say hello in one sentence." 3 512,1024 --dry-run
expect_output "context-pack-1b caps without explicit kind" "context_caps: 512 1024" ./scripts/pi/context-pack-1b.sh "Say hello in one sentence." 3 512,1024 --dry-run
expect_failure "context-pack-1b invalid q8 kind" ./scripts/pi/context-pack-1b.sh q8-broken --dry-run
expect_failure "context-pack-1b invalid env model path" env -u NANOCAMELID_SMOKE_GGUF NANOCAMELID_MODEL_GGUF=not-a-model ./scripts/pi/context-pack-1b.sh --dry-run
expect_failure "context-pack-1b invalid explicit model path" ./scripts/pi/context-pack-1b.sh /models/not-a-gguf --dry-run
expect_failure "context-pack-1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad ./scripts/pi/context-pack-1b.sh --dry-run
expect_failure "context-pack-1b repo-local target dir" env CARGO_TARGET_DIR=target ./scripts/pi/context-pack-1b.sh

echo "==> Checking 1B Pi context-pack launcher rejects invalid context cap..."
expect_failure "context-pack-1b invalid context cap" env NANOCAMELID_CONTEXT_PACKS=512,bad,2048 ./scripts/pi/context-pack-1b.sh --dry-run
expect_failure_output "context-pack-1b empty context cap" "Invalid context cap: empty value" env NANOCAMELID_CONTEXT_PACKS=512,,2048 ./scripts/pi/context-pack-1b.sh --dry-run
expect_failure_output "context-pack-1b duplicate context cap" "Duplicate context cap: 512" env NANOCAMELID_CONTEXT_PACKS=512,512 ./scripts/pi/context-pack-1b.sh --dry-run

echo "==> Checking 1B Pi evidence bundle dry run..."
./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b help documents context packs" "NANOCAMELID_CONTEXT_PACKS" ./scripts/pi/evidence-1b.sh --help
expect_output "evidence-1b q4 model audit" "q4_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b q8 model audit" "q8_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b q4 existence check" "q4_exists: " ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b q8 existence check" "q8_exists: " ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b help documents quant selectors" "--q4, --q8" ./scripts/pi/evidence-1b.sh --help
expect_output "evidence-1b q4 selector source" "selected_source: workspace Q4_0 requested" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/evidence-1b.sh --q4 --dry-run
expect_output "evidence-1b q4 selector child command" "model_command: ./scripts/pi/model-1b.sh /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" env -u NANOCAMELID_SMOKE_GGUF -u NANOCAMELID_MODEL_GGUF ./scripts/pi/evidence-1b.sh --q4 --dry-run
expect_failure_output "evidence-1b conflicting quant selectors" "Only one 1B evidence quantization selector may be provided." ./scripts/pi/evidence-1b.sh --q4 --q8 --dry-run
expect_failure_output "evidence-1b quant conflicts with explicit model" "1B evidence quantization selector cannot be combined with an explicit model path." ./scripts/pi/evidence-1b.sh --q4 /models/custom.gguf --dry-run
expect_failure_output "evidence-1b quant conflicts with env model" "1B evidence quantization selector cannot be combined with NANOCAMELID_SMOKE_GGUF or NANOCAMELID_MODEL_GGUF." env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf ./scripts/pi/evidence-1b.sh --q8 --dry-run
expect_output "evidence-1b success marker dry run" "status_on_success: evidence_1b_status: ok" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b shape audit dry run" "shape_audit: enabled" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b json records quantization" "\"quantization\":\"q8_0\"" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b json records smoke prompt" "\"smoke_prompt\":\"Say hello in one sentence.\"" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b json records prefill batch" "\"prefill_batch\":16" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b json records prefill prompt" "\"prefill_prompt\":\"Explain one practical Raspberry Pi inference bottleneck in two short sentences.\"" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b help documents prefill batch" "NANOCAMELID_PREFILL_BATCH" ./scripts/pi/evidence-1b.sh --help
expect_output "evidence-1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b json records context limit" "\"context_limit\":\"512\"" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b json records context caps" "\"context_pack_caps\":[512,1024,2048,4096,8192]" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b json records prefill batches" "\"prefill_batches\":[1,16,32,64]" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b model command" "model_command: ./scripts/pi/model-1b.sh" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b ready no-chat command" "ready_command: ./scripts/pi/ready-1b.sh chat Say\\ hello\\ in\\ one\\ sentence. 8 --no-chat" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b ready command carries context limit" "ready_command: NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/ready-1b.sh chat Say\\ hello\\ in\\ one\\ sentence. 8 --no-chat" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b ready command carries prefill batch" "ready_command: NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/ready-1b.sh chat Say\\ hello\\ in\\ one\\ sentence. 8 --no-chat" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b context-pack command" "context_pack_command: NANOCAMELID_CONTEXT_PACKS=512\\,1024\\,2048\\,4096\\,8192 ./scripts/pi/context-pack-1b.sh chat Say\\ hello\\ in\\ one\\ sentence. 8" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b context-pack command carries context limit" "context_pack_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_CONTEXT_PACKS=512\\,1024\\,2048\\,4096\\,8192 ./scripts/pi/context-pack-1b.sh chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b per-context command" "context_1024_command: NANOCAMELID_CONTEXT_LIMIT=1024 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b per-context command carries prefill batch" "context_512_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b prefill command" "prefill_bench_command: NANOCAMELID_PREFILL_PROMPT=Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. NANOCAMELID_PREFILL_TOKENS=2 NANOCAMELID_PREFILL_TEMP=0.0 NANOCAMELID_PREFILL_BATCHES=1\\,16\\,32\\,64 ./scripts/pi/bench-1b-prefill.sh" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b prefill command carries context limit" "prefill_bench_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_PROMPT=Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b explicit model ready command" "ready_command: ./scripts/pi/ready-1b.sh /models/custom.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8 --no-chat" ./scripts/pi/evidence-1b.sh /models/custom.gguf --dry-run
expect_failure "evidence-1b invalid explicit model path" ./scripts/pi/evidence-1b.sh /models/not-a-gguf --dry-run
expect_failure "evidence-1b invalid smoke kind" env NANOCAMELID_SMOKE_KIND=bad ./scripts/pi/evidence-1b.sh --dry-run
expect_failure "evidence-1b invalid smoke token count" env NANOCAMELID_SMOKE_TOKENS=0 ./scripts/pi/evidence-1b.sh --dry-run
expect_failure "evidence-1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad ./scripts/pi/evidence-1b.sh --dry-run
expect_failure_output "evidence-1b invalid prefill batch" "NANOCAMELID_PREFILL_BATCH must be a positive integer: bad" env NANOCAMELID_PREFILL_BATCH=bad ./scripts/pi/evidence-1b.sh --dry-run
expect_failure "evidence-1b invalid context caps" env NANOCAMELID_CONTEXT_PACKS=512,bad ./scripts/pi/evidence-1b.sh --dry-run
expect_failure "evidence-1b invalid prefill batches" env NANOCAMELID_PREFILL_BATCHES=1,bad ./scripts/pi/evidence-1b.sh --dry-run
expect_failure_output "evidence-1b empty context cap" "Invalid context cap: empty value" env NANOCAMELID_CONTEXT_PACKS=512,,1024 ./scripts/pi/evidence-1b.sh --dry-run
expect_failure_output "evidence-1b empty prefill batch" "Invalid prefill batch size: empty value" env NANOCAMELID_PREFILL_BATCHES=1,,16 ./scripts/pi/evidence-1b.sh --dry-run
expect_failure_output "evidence-1b duplicate context cap" "Duplicate context cap: 512" env NANOCAMELID_CONTEXT_PACKS=512,1024,512 ./scripts/pi/evidence-1b.sh --dry-run
expect_failure_output "evidence-1b duplicate prefill batch" "Duplicate prefill batch size: 16" env NANOCAMELID_PREFILL_BATCHES=1,16,16 ./scripts/pi/evidence-1b.sh --dry-run
expect_failure_output "evidence-1b actual log records context caps before model load" "context_pack_caps: 512 1024" env NANOCAMELID_CONTEXT_PACKS=512,1024 NANOCAMELID_SMOKE_GGUF=/tmp/nanocamelid-missing-validation.gguf ./scripts/pi/evidence-1b.sh
expect_failure "evidence-1b repo-local target dir" env CARGO_TARGET_DIR=target ./scripts/pi/evidence-1b.sh

echo "==> Checking Strand cluster launcher dry run..."
./scripts/pi/strand-cluster.sh --dry-run
expect_failure "strand-cluster repo-local target dir" env CARGO_TARGET_DIR=target ./scripts/pi/strand-cluster.sh final

echo "==> Checking Mixtral cluster launcher dry run..."
./scripts/pi/mixtral-cluster.sh --dry-run
expect_failure "mixtral-cluster repo-local target dir" env CARGO_TARGET_DIR=target ./scripts/pi/mixtral-cluster.sh final

echo "==> Checking Mixtral cluster launcher rejects invalid token count..."
expect_failure "mixtral-cluster invalid token count" env NANOCAMELID_CLUSTER_TOKENS=0 ./scripts/pi/mixtral-cluster.sh --dry-run

echo "==> Checking Mixtral cluster launcher rejects invalid context cap..."
expect_failure "mixtral-cluster invalid context cap" env NANOCAMELID_CLUSTER_CONTEXT_LIMIT=bad ./scripts/pi/mixtral-cluster.sh --dry-run

echo "==> Checking 1B Pi readiness launcher rejects invalid temperature..."
expect_failure "ready-1b invalid temperature" env NANOCAMELID_READY_TEMP=bad ./scripts/pi/ready-1b.sh --dry-run
expect_failure "ready-1b invalid direct chat toggle" env NANOCAMELID_READY_CHAT=flase ./scripts/pi/ready-1b.sh --dry-run
expect_failure_output "ready-1b quant conflicts with explicit model" "1B readiness quantization selector cannot be combined with an explicit model path." ./scripts/pi/ready-1b.sh --q4 /models/custom.gguf --dry-run
expect_failure_output "ready-1b quant conflicts with env model" "1B readiness quantization selector cannot be combined with NANOCAMELID_SMOKE_GGUF or NANOCAMELID_MODEL_GGUF." env NANOCAMELID_MODEL_GGUF=/models/model.gguf ./scripts/pi/ready-1b.sh --q8 --dry-run

echo "==> Checking 1B Pi readiness launcher ignores direct chat env when chat is disabled..."
env NANOCAMELID_READY_CHAT=flase ./scripts/pi/ready-1b.sh --no-chat --dry-run
env NANOCAMELID_READY_TEMP=bad NANOCAMELID_READY_TOKENS=0 ./scripts/pi/ready-1b.sh --no-chat --dry-run

echo "==> Checking remote Pi build launcher dry run..."
./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build help documents remote prefill batch" "NANOCAMELID_REMOTE_PREFILL_BATCH" bash -c './scripts/remote_build.sh --help 2>&1'
expect_output "remote_build help documents remote target dir" "NANOCAMELID_REMOTE_TARGET_DIR" bash -c './scripts/remote_build.sh --help 2>&1'
expect_output "remote_build help documents remote 1b quant selector" "NANOCAMELID_REMOTE_1B_QUANT" bash -c './scripts/remote_build.sh --help 2>&1'
expect_output "remote_build help documents dirty policy" "NANOCAMELID_REMOTE_DIRTY_POLICY" bash -c './scripts/remote_build.sh --help 2>&1'
expect_output "remote_build prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_REMOTE_PREFILL_BATCH=32 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build derives target dir from workspace" "cargo_target_dir: /tmp/nanocamelid-alt/target" env NANOCAMELID_REMOTE_WORKSPACE=/tmp/nanocamelid-alt ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build target dir override" "cargo_target_dir: /tmp/nanocamelid-target-alt" env NANOCAMELID_REMOTE_TARGET_DIR=/tmp/nanocamelid-target-alt ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build dirty archive policy dry run" "remote_dirty_policy: archive" env NANOCAMELID_REMOTE_DIRTY_POLICY=archive ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build readiness command carries prefill batch" "readiness_command: NANOCAMELID_PREFILL_BATCH=32 NANOCAMELID_READY_CHAT=1" env NANOCAMELID_REMOTE_PREFILL_BATCH=32 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build q4 quant dry run" "remote_1b_quant: q4" env NANOCAMELID_REMOTE_1B_QUANT=q4 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build q4 readiness selector" "./scripts/pi/ready-1b.sh --q4" env NANOCAMELID_REMOTE_1B_QUANT=q4 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build q8 quant dry run" "remote_1b_quant: q8" env NANOCAMELID_REMOTE_1B_QUANT=q8 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build q8 readiness selector" "./scripts/pi/ready-1b.sh --q8" env NANOCAMELID_REMOTE_1B_QUANT=q8 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build q8 context selector" "context_pack_command: NANOCAMELID_CONTEXT_PACKS=512\\,1024 ./scripts/pi/context-pack-1b.sh --q8 chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_REMOTE_1B_QUANT=q8 NANOCAMELID_REMOTE_CONTEXT_PACKS=512,1024 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build q8 prefill selector" "prefill_bench_command: NANOCAMELID_PREFILL_PROMPT=Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. NANOCAMELID_PREFILL_TOKENS=2 NANOCAMELID_PREFILL_TEMP=0.0 NANOCAMELID_PREFILL_BATCHES=1\\,16\\,32\\,64 ./scripts/pi/bench-1b-prefill.sh --q8" env NANOCAMELID_REMOTE_1B_QUANT=q8 NANOCAMELID_REMOTE_PREFILL_BENCH=1 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build q4 evidence selector" "evidence_command: NANOCAMELID_SMOKE_KIND=chat NANOCAMELID_SMOKE_PROMPT=Say\\ hello\\ in\\ one\\ sentence. NANOCAMELID_SMOKE_TOKENS=8 NANOCAMELID_CONTEXT_PACKS=512\\,1024\\,2048\\,4096\\,8192 NANOCAMELID_PREFILL_PROMPT=Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. NANOCAMELID_PREFILL_TOKENS=2 NANOCAMELID_PREFILL_TEMP=0.0 NANOCAMELID_PREFILL_BATCHES=1\\,16\\,32\\,64 ./scripts/pi/evidence-1b.sh --q4" env NANOCAMELID_REMOTE_1B_QUANT=q4 NANOCAMELID_REMOTE_EVIDENCE=1 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure_output "remote_build invalid remote 1b quant selector" "NANOCAMELID_REMOTE_1B_QUANT must be q4 or q8" env NANOCAMELID_REMOTE_1B_QUANT=q5 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure_output "remote_build remote 1b quant conflicts with explicit model" "NANOCAMELID_REMOTE_1B_QUANT cannot be combined with NANOCAMELID_REMOTE_SMOKE_GGUF" env NANOCAMELID_REMOTE_1B_QUANT=q4 NANOCAMELID_REMOTE_SMOKE_GGUF=/models/custom.gguf ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher rejects invalid deploy mode..."
expect_failure "remote_build invalid deploy mode" ./scripts/remote_build.sh "<redacted-pi-host>" "" "" bad-mode --dry-run
expect_failure_output "remote_build invalid dirty policy" "NANOCAMELID_REMOTE_DIRTY_POLICY must be fail or archive" env NANOCAMELID_REMOTE_DIRTY_POLICY=reset ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher rejects invalid smoke token count..."
expect_failure "remote_build invalid smoke token count" env NANOCAMELID_SMOKE_TOKENS=bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build invalid explicit model path" env NANOCAMELID_REMOTE_SMOKE_GGUF=not-a-model ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher rejects invalid readiness temperature..."
expect_failure "remote_build invalid readiness temperature" env NANOCAMELID_READY_TEMP=bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build invalid readiness chat toggle" env NANOCAMELID_READY_CHAT=flase ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure_output "remote_build invalid remote readiness chat toggle" "NANOCAMELID_REMOTE_READY_CHAT/NANOCAMELID_READY_CHAT must be" env NANOCAMELID_REMOTE_READY_CHAT=flase ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher ignores smoke env when remote smoke is disabled..."
env NANOCAMELID_REMOTE_SMOKE=0 NANOCAMELID_SMOKE_TOKENS=bad NANOCAMELID_READY_TEMP=bad NANOCAMELID_READY_TOKENS=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
env NANOCAMELID_REMOTE_SMOKE=0 NANOCAMELID_REMOTE_SMOKE_KIND=bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
env NANOCAMELID_REMOTE_SMOKE=off NANOCAMELID_REMOTE_SMOKE_KIND=bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build disabled smoke skips shape audit" "readiness_shape_audit: skipped" env NANOCAMELID_REMOTE_SMOKE=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build smoke off skips shape audit" "readiness_shape_audit: skipped" env NANOCAMELID_REMOTE_SMOKE=off ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher ignores direct chat env when readiness chat is disabled..."
env NANOCAMELID_READY_CHAT=0 NANOCAMELID_READY_TEMP=bad NANOCAMELID_READY_TOKENS=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build shape audit dry run" "readiness_shape_audit: enabled" ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build readiness context limit dry run" "context_limit: 512" env NANOCAMELID_REMOTE_CONTEXT_LIMIT=512 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build readiness command carries context limit" "readiness_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_READY_CHAT=1" env NANOCAMELID_REMOTE_CONTEXT_LIMIT=512 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build invalid readiness context limit" env NANOCAMELID_REMOTE_CONTEXT_LIMIT=bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build invalid prefill batch" env NANOCAMELID_REMOTE_PREFILL_BATCH=bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build no-chat readiness command" "readiness_command: NANOCAMELID_READY_CHAT=0 NANOCAMELID_READY_SMOKE_KIND=chat" env NANOCAMELID_READY_CHAT=0 NANOCAMELID_READY_TEMP=bad NANOCAMELID_READY_TOKENS=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_no_output "remote_build no-chat omits direct chat token env" "NANOCAMELID_READY_TOKENS=0" env NANOCAMELID_READY_CHAT=0 NANOCAMELID_READY_TEMP=bad NANOCAMELID_READY_TOKENS=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_no_output "remote_build no-chat omits direct chat temp env" "NANOCAMELID_READY_TEMP=bad" env NANOCAMELID_READY_CHAT=0 NANOCAMELID_READY_TEMP=bad NANOCAMELID_READY_TOKENS=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher plans optional context packs..."
expect_output "remote_build context-pack dry run" "context_pack_command: NANOCAMELID_CONTEXT_PACKS=512\\,1024 ./scripts/pi/context-pack-1b.sh chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_REMOTE_CONTEXT_PACKS=512,1024 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build context-pack command carries prefill batch" "context_pack_command: NANOCAMELID_PREFILL_BATCH=32 NANOCAMELID_CONTEXT_PACKS=512\\,1024 ./scripts/pi/context-pack-1b.sh chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_REMOTE_CONTEXT_PACKS=512,1024 NANOCAMELID_REMOTE_PREFILL_BATCH=32 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher rejects invalid context packs..."
expect_failure "remote_build invalid context cap" env NANOCAMELID_REMOTE_CONTEXT_PACKS=512,bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure_output "remote_build empty context cap" "Context cap must be a positive integer: empty value" env NANOCAMELID_REMOTE_CONTEXT_PACKS=512,,1024 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build duplicate context cap" env NANOCAMELID_REMOTE_CONTEXT_PACKS=512,1024,512 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher plans optional 1B prefill sweep..."
expect_output "remote_build prefill dry run" "prefill_bench_command: NANOCAMELID_PREFILL_PROMPT=Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. NANOCAMELID_PREFILL_TOKENS=2 NANOCAMELID_PREFILL_TEMP=0.0 NANOCAMELID_PREFILL_BATCHES=1\\,16\\,32\\,64 ./scripts/pi/bench-1b-prefill.sh" env NANOCAMELID_REMOTE_PREFILL_BENCH=1 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build prefill command carries context limit" "prefill_bench_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_PROMPT=Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck" env NANOCAMELID_REMOTE_CONTEXT_LIMIT=512 NANOCAMELID_REMOTE_PREFILL_BENCH=1 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build prefill command carries preflight prefill batch" "prefill_bench_command: NANOCAMELID_PREFILL_BATCH=32 NANOCAMELID_PREFILL_PROMPT=Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck" env NANOCAMELID_REMOTE_PREFILL_BENCH=1 NANOCAMELID_REMOTE_PREFILL_BATCH=32 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build prefill prompt override" "prefill_bench_command: NANOCAMELID_PREFILL_PROMPT=Custom\\ prefill" env NANOCAMELID_REMOTE_PREFILL_BENCH=1 NANOCAMELID_PREFILL_PROMPT="Custom prefill" ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build prefill off skips sweep" "prefill_bench_command: skipped" env NANOCAMELID_REMOTE_PREFILL_BENCH=off ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build empty prefill toggle skips sweep" "prefill_bench_command: skipped" env NANOCAMELID_REMOTE_PREFILL_BENCH= ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher plans optional 1B evidence bundle..."
expect_output "remote_build evidence dry run" "evidence_command: NANOCAMELID_SMOKE_KIND=chat NANOCAMELID_SMOKE_PROMPT=Say\\ hello\\ in\\ one\\ sentence. NANOCAMELID_SMOKE_TOKENS=8 NANOCAMELID_CONTEXT_PACKS=512\\,1024\\,2048\\,4096\\,8192 NANOCAMELID_PREFILL_PROMPT=Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. NANOCAMELID_PREFILL_TOKENS=2 NANOCAMELID_PREFILL_TEMP=0.0 NANOCAMELID_PREFILL_BATCHES=1\\,16\\,32\\,64 ./scripts/pi/evidence-1b.sh" env NANOCAMELID_REMOTE_EVIDENCE=1 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build evidence delegates shape audit" "readiness_shape_audit: delegated_to_evidence_bundle" env NANOCAMELID_REMOTE_EVIDENCE=1 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build evidence skips composed readiness" "readiness_command: skipped" env NANOCAMELID_REMOTE_EVIDENCE=1 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build evidence command carries context limit" "evidence_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_SMOKE_KIND=chat" env NANOCAMELID_REMOTE_EVIDENCE=1 NANOCAMELID_REMOTE_CONTEXT_LIMIT=512 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build evidence command carries prefill batch" "evidence_command: NANOCAMELID_PREFILL_BATCH=32 NANOCAMELID_SMOKE_KIND=chat" env NANOCAMELID_REMOTE_EVIDENCE=1 NANOCAMELID_REMOTE_PREFILL_BATCH=32 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build evidence explicit model" "evidence_command: NANOCAMELID_SMOKE_KIND=chat NANOCAMELID_SMOKE_PROMPT=Say\\ hello\\ in\\ one\\ sentence. NANOCAMELID_SMOKE_TOKENS=8 NANOCAMELID_CONTEXT_PACKS=512\\,1024\\,2048\\,4096\\,8192 NANOCAMELID_PREFILL_PROMPT=Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. NANOCAMELID_PREFILL_TOKENS=2 NANOCAMELID_PREFILL_TEMP=0.0 NANOCAMELID_PREFILL_BATCHES=1\\,16\\,32\\,64 ./scripts/pi/evidence-1b.sh /models/custom.gguf" env NANOCAMELID_REMOTE_EVIDENCE=1 NANOCAMELID_REMOTE_SMOKE_GGUF=/models/custom.gguf ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build evidence requires smoke" env NANOCAMELID_REMOTE_EVIDENCE=1 NANOCAMELID_REMOTE_SMOKE=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build evidence off uses readiness" "evidence_command: skipped" env NANOCAMELID_REMOTE_EVIDENCE=off ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build empty evidence toggle uses readiness" "evidence_command: skipped" env NANOCAMELID_REMOTE_EVIDENCE= ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher rejects invalid prefill sweep settings..."
expect_failure "remote_build invalid prefill batch" env NANOCAMELID_REMOTE_PREFILL_BENCH=1 NANOCAMELID_REMOTE_PREFILL_BATCHES=1,bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure_output "remote_build empty prefill batch" "Prefill batch size must be a positive integer: empty value" env NANOCAMELID_REMOTE_PREFILL_BENCH=1 NANOCAMELID_REMOTE_PREFILL_BATCHES=1,,16 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build duplicate prefill batch" env NANOCAMELID_REMOTE_PREFILL_BENCH=1 NANOCAMELID_REMOTE_PREFILL_BATCHES=16,32,16 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build invalid prefill token count" env NANOCAMELID_REMOTE_PREFILL_BENCH=1 NANOCAMELID_PREFILL_TOKENS=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build invalid prefill temperature" env NANOCAMELID_REMOTE_PREFILL_BENCH=1 NANOCAMELID_PREFILL_TEMP=bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking release version output..."
expect_output "top-level version" "nanocamelid 0.1.0" cargo run -- --version

echo "==> Checking installer dry run target-dir safety..."
./scripts/install.sh --dry-run
expect_output "installer release target override" "release_target: aarch64-unknown-linux-gnu" env NANOCAMELID_RELEASE_TARGET=aarch64-unknown-linux-gnu ./scripts/install.sh --dry-run
expect_output "installer release companion dir" "release_install_dir: " ./scripts/install.sh --dry-run
expect_output "installer release companion dir includes version and target" "releases/v0.1.0-aarch64-unknown-linux-gnu" ./scripts/install.sh --dry-run
expect_output "installer release verifies version manifest" "verify VERSION manifest and nanocamelid --version" ./scripts/install.sh --dry-run
expect_output "installer release installs bundled files" "install bundled README docs changelog release notes service script and nanocamelid" ./scripts/install.sh --dry-run
expect_failure_output "installer rejects unsafe release companion dir" "Refusing unsafe release install dir" env NANOCAMELID_RELEASE_INSTALL_DIR=/ ./scripts/install.sh --dry-run
expect_output "installer dev mode skips release URL" "release_url: not used" ./scripts/install.sh --dev --dry-run
expect_output "installer dev mode skips companion dir" "release_install_dir: not used" ./scripts/install.sh --dev --dry-run

echo "==> Checking release package dry run target contract..."
./scripts/package-release.sh --dry-run
expect_output "release package target" "target_triple: aarch64-unknown-linux-gnu" ./scripts/package-release.sh --dry-run
expect_output "release package version manifest" "version_manifest: " ./scripts/package-release.sh --dry-run
expect_output "release package explicit cargo target" "cargo_command: cargo build --release --bins --target aarch64-unknown-linux-gnu" ./scripts/package-release.sh --dry-run
expect_output "release package target-scoped binary" "binary: " ./scripts/package-release.sh --dry-run
expect_output "release package target-scoped binary path" "/aarch64-unknown-linux-gnu/release/nanocamelid" ./scripts/package-release.sh --dry-run
expect_output "release package version check" "version_check: nanocamelid --version == nanocamelid 0.1.0" ./scripts/package-release.sh --dry-run
expect_output "release package stages changelog" "stage binary VERSION README docs LICENSE CHANGELOG RELEASE_NOTES service installer" ./scripts/package-release.sh --dry-run
expect_failure_output "release package rejects version mismatch" "does not match Cargo.toml version" env NANOCAMELID_VERSION=v9.9.9 ./scripts/package-release.sh --dry-run
expect_failure_output "release package rejects relative target dir" "Refusing to use a relative repo-local Cargo target dir" env CARGO_TARGET_DIR=target ./scripts/package-release.sh --dry-run

echo "==> Checking systemd user service installer dry run..."
./scripts/install-systemd-user-service.sh --dry-run
expect_output "service installer dry-run status" "NanoCamelid systemd user service dry run" ./scripts/install-systemd-user-service.sh --dry-run
expect_output "service installer default listen" "listen: http://127.0.0.1:8080" ./scripts/install-systemd-user-service.sh --dry-run
expect_output "service installer hardening" "NoNewPrivileges=true" ./scripts/install-systemd-user-service.sh --dry-run
expect_output "service installer localhost allowlist" "IPAddressAllow=localhost" ./scripts/install-systemd-user-service.sh --dry-run
expect_output "service installer api key redacted state" "api_key_required: true" env NANOCAMELID_API_KEY=redacted-test-key ./scripts/install-systemd-user-service.sh --dry-run
expect_no_output "service installer does not print api key" "redacted-test-key" env NANOCAMELID_API_KEY=redacted-test-key ./scripts/install-systemd-user-service.sh --dry-run
expect_failure_output "service installer rejects bad port" "--port must be an integer from 1 to 65535" ./scripts/install-systemd-user-service.sh --port 0 --dry-run
expect_failure_output "service installer rejects unauthenticated network bind" "--host outside loopback requires --api-key or NANOCAMELID_API_KEY" ./scripts/install-systemd-user-service.sh --host 0.0.0.0 --dry-run
expect_output "service installer allows authenticated network bind" "listen: http://0.0.0.0:8080" env NANOCAMELID_API_KEY=redacted-test-key ./scripts/install-systemd-user-service.sh --host 0.0.0.0 --dry-run
expect_output "service installer network bind allowlist" "IPAddressAllow=any" env NANOCAMELID_API_KEY=redacted-test-key ./scripts/install-systemd-user-service.sh --host 0.0.0.0 --dry-run
