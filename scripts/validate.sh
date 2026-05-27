#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: validate.sh [--dry-run]

Runs NanoCamelid's standard local validation gate:
  1. cargo fmt -- --check
  2. cargo test
  3. cargo clippy --all-targets -- -D warnings
  4. cargo run -- model 1b --dry-run
  5. cargo run -- inspect 1b --dry-run
  6. cargo run -- generate 1b --dry-run
  7. cargo run -- chat 1b --dry-run
  8. cargo run -- smoke 1b --dry-run
  9. cargo run -- ready 1b --dry-run
  10. cargo run -- tui 1b --dry-run
  11. cargo run -- bench 1b --dry-run
  12. cargo run -- bench 1b --help
  13. ./scripts/pi/model-1b.sh --dry-run
  14. ./scripts/pi/smoke-1b.sh --dry-run
  15. ./scripts/pi/ready-1b.sh --dry-run
  16. ./scripts/pi/chat-1b.sh --dry-run
  17. ./scripts/pi/bench-1b-prefill.sh --dry-run
  18. ./scripts/pi/context-pack-1b.sh --dry-run
  19. ./scripts/pi/evidence-1b.sh --dry-run
  20. ./scripts/pi/strand-cluster.sh --dry-run
  21. ./scripts/pi/mixtral-cluster.sh --dry-run
  22. ./scripts/remote_build.sh <redacted-pi-host> --dry-run
  23. NANOCAMELID_REMOTE_CONTEXT_PACKS=512,1024 ./scripts/remote_build.sh <redacted-pi-host> --dry-run
  24. NANOCAMELID_REMOTE_PREFILL_BENCH=1 ./scripts/remote_build.sh <redacted-pi-host> --dry-run
  25. ./scripts/install.sh --dry-run

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
  echo "steps: cargo fmt -- --check; cargo test; cargo clippy --all-targets -- -D warnings; cargo run -- model 1b --dry-run; cargo run -- inspect 1b --dry-run; cargo run -- generate 1b --dry-run; cargo run -- chat 1b --dry-run; cargo run -- smoke 1b --dry-run; cargo run -- ready 1b --dry-run; cargo run -- tui 1b --dry-run; cargo run -- bench 1b --dry-run; cargo run -- bench 1b --help; ./scripts/pi/model-1b.sh --dry-run; ./scripts/pi/smoke-1b.sh --dry-run; ./scripts/pi/ready-1b.sh --dry-run; ./scripts/pi/chat-1b.sh --dry-run; ./scripts/pi/bench-1b-prefill.sh --dry-run; ./scripts/pi/context-pack-1b.sh --dry-run; ./scripts/pi/evidence-1b.sh --dry-run; ./scripts/pi/strand-cluster.sh --dry-run; ./scripts/pi/mixtral-cluster.sh --dry-run; ./scripts/remote_build.sh <redacted-pi-host> --dry-run; NANOCAMELID_REMOTE_CONTEXT_PACKS=512,1024 ./scripts/remote_build.sh <redacted-pi-host> --dry-run; NANOCAMELID_REMOTE_PREFILL_BENCH=1 ./scripts/remote_build.sh <redacted-pi-host> --dry-run; ./scripts/install.sh --dry-run"
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
  if ! grep -F "$expected" <<<"$output" >/dev/null; then
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

  if ! "$@" | grep -F "$expected" >/dev/null; then
    echo "Expected output missing for $description: $expected" >&2
    exit 1
  fi
}

expect_no_output() {
  local description="$1"
  local unexpected="$2"
  shift 2

  if "$@" | grep -F "$unexpected" >/dev/null; then
    echo "Unexpected output found for $description: $unexpected" >&2
    exit 1
  fi
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

echo "==> Checking 1B model audit CLI dry run..."
cargo run -- model 1b --dry-run
expect_output "model 1b shape audit dry run" "shape_audit: enabled" cargo run -- model 1b --dry-run
expect_output "model 1b success marker dry run" "status_on_success: model_1b_status: ok" cargo run -- model 1b --dry-run
expect_output "model 1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" cargo run -- model 1b --dry-run
expect_output "model 1b selected quantization dry run" "quantization: q8_0" cargo run -- model 1b --dry-run
expect_output "model 1b json records quantization" "\"quantization\":\"q8_0\"" cargo run -- model 1b --dry-run
expect_output "model 1b shape json marker dry run" "\"shape\":\"llama32_1b\",\"shape_ready\":true" cargo run -- model 1b --dry-run
expect_output "model 1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- model 1b --dry-run
expect_output "model 1b inspect follow-up command" "inspect_command: nanocamelid inspect /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- model 1b --dry-run
expect_output "model 1b smoke follow-up command" "smoke_command: nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello in one sentence.' 8" cargo run -- model 1b --dry-run
expect_output "model 1b ready follow-up command" "ready_command: nanocamelid ready 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- model 1b --dry-run

echo "==> Checking 1B model audit CLI rejects non-GGUF model args..."
expect_failure "model 1b invalid model arg" cargo run -- model 1b not-a-model --dry-run
expect_failure "model 1b invalid env model path" env NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- model 1b --dry-run

echo "==> Checking 1B inspect CLI dry run..."
cargo run -- inspect 1b --dry-run
expect_output "inspect 1b q4 model audit" "q4_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" cargo run -- inspect 1b --dry-run
expect_output "inspect 1b q8 model audit" "q8_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- inspect 1b --dry-run
expect_output "inspect 1b selected source" "selected_source: " cargo run -- inspect 1b --dry-run
expect_output "inspect 1b shape audit dry run" "shape_audit: enabled" cargo run -- inspect 1b --dry-run
expect_output "inspect 1b success marker dry run" "status_on_success: inspect_1b_status: ok" cargo run -- inspect 1b --dry-run
expect_failure "inspect 1b invalid env model path" env NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- inspect 1b --dry-run
expect_failure "inspect 1b extra argument" cargo run -- inspect 1b extra --dry-run

echo "==> Checking 1B generate CLI dry run..."
cargo run -- generate 1b --dry-run
expect_output "generate 1b selected source" "selected_source: workspace Q8_0 fallback" cargo run -- generate 1b --dry-run
expect_output "generate 1b env selected source" "selected_source: NANOCAMELID_MODEL_GGUF" env NANOCAMELID_MODEL_GGUF=/models/custom.gguf cargo run -- generate 1b --dry-run
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
expect_failure "generate 1b invalid env model path" env NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- generate 1b --dry-run

echo "==> Checking 1B chat CLI dry run..."
cargo run -- chat 1b --dry-run
expect_output "chat 1b selected source" "selected_source: workspace Q8_0 fallback" cargo run -- chat 1b --dry-run
expect_output "chat 1b env selected source" "selected_source: NANOCAMELID_MODEL_GGUF" env NANOCAMELID_MODEL_GGUF=/models/custom.gguf cargo run -- chat 1b --dry-run
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
expect_failure "chat 1b invalid env model path" env NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- chat 1b --dry-run

echo "==> Checking 1B smoke CLI dry run..."
cargo run -- smoke 1b --dry-run
expect_output "smoke 1b q4 model audit" "q4_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b q8 model audit" "q8_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b selected source" "selected_source: " cargo run -- smoke 1b --dry-run
expect_output "smoke 1b selected quantization" "quantization: q8_0" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- smoke 1b --dry-run
expect_output "smoke 1b context-limited command" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello in one sentence.' 8" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- smoke 1b --dry-run
expect_output "smoke 1b shape audit dry run" "shape_audit: enabled" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b success marker dry run" "status_on_success: smoke_1b_status: ok" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b json records quantization" "\"quantization\":\"q8_0\"" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b json records shape audit" "\"shape\":\"llama32_1b\",\"shape_ready\":true" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- smoke 1b --dry-run
expect_output_order "smoke 1b model audit before smoke" "model_command: nanocamelid model 1b" "smoke_command: nanocamelid smoke 1b" cargo run -- smoke 1b --dry-run
expect_output "smoke 1b prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_PREFILL_BATCH=32 cargo run -- smoke 1b --dry-run
expect_output "smoke 1b json records prefill batch" "\"prefill_batch\":32" env NANOCAMELID_PREFILL_BATCH=32 cargo run -- smoke 1b --dry-run
expect_output "smoke 1b command carries prefill batch" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello in one sentence.' 8" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 cargo run -- smoke 1b --dry-run
expect_failure "smoke 1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad cargo run -- smoke 1b --dry-run
expect_failure "smoke 1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad cargo run -- smoke 1b --dry-run
expect_failure "smoke 1b invalid env model path" env NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- smoke 1b --dry-run
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
expect_output "ready 1b json records quantization" "\"quantization\":\"q8_0\"" cargo run -- ready 1b --dry-run
expect_output "ready 1b json records probe" "\"probe\":true" cargo run -- ready 1b --dry-run
expect_output "ready 1b json records shape audit" "\"shape\":\"llama32_1b\",\"shape_ready\":true" cargo run -- ready 1b --dry-run
expect_output "ready 1b no-chat json success marker dry run" "\"direct_chat\":false,\"chat_tokens\":null" cargo run -- ready 1b --no-chat --dry-run
expect_output "ready 1b prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_PREFILL_BATCH=32 cargo run -- ready 1b --dry-run
expect_output "ready 1b json records prefill batch" "\"prefill_batch\":32" env NANOCAMELID_PREFILL_BATCH=32 cargo run -- ready 1b --dry-run
expect_output "ready 1b smoke command carries prefill batch" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello in one sentence.' 8" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 cargo run -- ready 1b --dry-run
expect_output "ready 1b chat command carries prefill batch" "chat_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 'Say hello in one sentence.' 0 8" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 cargo run -- ready 1b --dry-run
expect_output "ready 1b probe command" "probe_command: nanocamelid probe" cargo run -- ready 1b --dry-run
expect_output "ready 1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- ready 1b --dry-run
expect_output_order "ready 1b probe before inspect" "probe_command: nanocamelid probe" "inspect_command: nanocamelid inspect" cargo run -- ready 1b --dry-run
expect_output_order "ready 1b model audit before inspect" "model_command: nanocamelid model 1b" "inspect_command: nanocamelid inspect" cargo run -- ready 1b --dry-run
expect_output "ready 1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- ready 1b --dry-run
expect_output "ready 1b context-limited smoke command" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Say hello in one sentence.' 8" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- ready 1b --dry-run
expect_output "ready 1b context-limited chat command" "chat_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 'Say hello in one sentence.' 0 8" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- ready 1b --dry-run
expect_failure "ready 1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad cargo run -- ready 1b --dry-run
expect_failure "ready 1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad cargo run -- ready 1b --dry-run
expect_failure "ready 1b invalid env model path" env NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- ready 1b --dry-run
expect_failure "ready 1b invalid explicit model path" cargo run -- ready 1b /models/not-a-gguf --dry-run

echo "==> Checking 1B readiness CLI rejects invalid direct chat env..."
expect_failure "ready 1b invalid direct chat temperature" env NANOCAMELID_READY_TEMP=bad cargo run -- ready 1b --dry-run
expect_failure "ready 1b invalid direct chat token count" env NANOCAMELID_READY_TOKENS=0 cargo run -- ready 1b --dry-run

echo "==> Checking 1B readiness CLI ignores direct chat env when chat is disabled..."
env NANOCAMELID_READY_TEMP=bad NANOCAMELID_READY_TOKENS=0 cargo run -- ready 1b --no-chat --dry-run

echo "==> Checking 1B TUI CLI dry run..."
cargo run -- tui 1b --dry-run
expect_output "tui 1b selected source" "selected_source: workspace Q8_0 fallback" cargo run -- tui 1b --dry-run
expect_output "tui 1b env selected source" "selected_source: NANOCAMELID_MODEL_GGUF" env NANOCAMELID_MODEL_GGUF=/models/custom.gguf cargo run -- tui 1b --dry-run
expect_output "tui 1b dry-run command" "tui_command: nanocamelid tui /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 0 128" cargo run -- tui 1b --dry-run
expect_output "tui 1b shape audit dry run" "shape_audit: enabled" cargo run -- tui 1b --dry-run
expect_output "tui 1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- tui 1b --dry-run
expect_output "tui 1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- tui 1b --dry-run
expect_output "tui 1b context-limited command" "tui_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid tui /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 0 128" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- tui 1b --dry-run
expect_output "tui 1b prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_PREFILL_BATCH=32 cargo run -- tui 1b --dry-run
expect_output "tui 1b command carries prefill batch" "tui_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid tui /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 0 128" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 cargo run -- tui 1b --dry-run
expect_failure "tui 1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad cargo run -- tui 1b --dry-run
expect_failure "tui 1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad cargo run -- tui 1b --dry-run
expect_failure "tui 1b invalid env model path" env NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- tui 1b --dry-run

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
expect_output "bench 1b default batches dry run" "batches: 1 16 32 64" cargo run -- bench 1b --dry-run
expect_output "bench 1b probe command" "probe_command: nanocamelid probe" cargo run -- bench 1b --dry-run
expect_output "bench 1b smoke command" "smoke_command: NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Explain one practical Raspberry Pi inference bottleneck in two short sentences.' 2" cargo run -- bench 1b --dry-run
expect_output "bench 1b batch command" "batch_16_command: NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot NANOCAMELID_PREFILL_BATCH=16 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 'Explain one practical Raspberry Pi inference bottleneck in two short sentences.' 0.0 2" cargo run -- bench 1b --dry-run
expect_output "bench 1b inspect command" "inspect_command: nanocamelid inspect /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" cargo run -- bench 1b --dry-run
expect_output_order "bench 1b probe before model audit" "probe_command: nanocamelid probe" "model_command: nanocamelid model 1b" cargo run -- bench 1b --dry-run
expect_output_order "bench 1b inspect before smoke" "inspect_command: nanocamelid inspect" "smoke_command: NANOCAMELID_Q8_DOT_SDOT=1" cargo run -- bench 1b --dry-run
expect_output_order "bench 1b smoke before batch" "smoke_command: NANOCAMELID_Q8_DOT_SDOT=1" "batch_16_command: NANOCAMELID_Q8_DOT_SDOT=1" cargo run -- bench 1b --dry-run
expect_output "bench 1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- bench 1b --dry-run
expect_output "bench 1b context-limited smoke command" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat 'Explain one practical Raspberry Pi inference bottleneck in two short sentences.' 2" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- bench 1b --dry-run
expect_output "bench 1b context-limited batch command" "batch_16_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot NANOCAMELID_PREFILL_BATCH=16 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf 'Explain one practical Raspberry Pi inference bottleneck in two short sentences.' 0.0 2" env NANOCAMELID_CONTEXT_LIMIT=512 cargo run -- bench 1b --dry-run
expect_failure_output "bench 1b missing model dry-run hint" "nanocamelid bench 1b --dry-run" cargo run -- bench 1b
expect_failure "bench 1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad cargo run -- bench 1b --dry-run
expect_failure "bench 1b invalid token count" cargo run -- bench 1b prompt 0 --dry-run
expect_failure "bench 1b invalid temp" cargo run -- bench 1b prompt 1 bad --dry-run
expect_failure "bench 1b invalid batch" cargo run -- bench 1b prompt 1 0.0 0 --dry-run
expect_failure "bench 1b duplicate batch" cargo run -- bench 1b prompt 1 0.0 16,32,16 --dry-run
expect_failure "bench 1b invalid smoke env model path" env NANOCAMELID_SMOKE_GGUF=not-a-model cargo run -- bench 1b --dry-run
expect_failure "bench 1b invalid env model path" env NANOCAMELID_MODEL_GGUF=not-a-model cargo run -- bench 1b --dry-run
expect_failure "bench 1b invalid explicit model path" cargo run -- bench 1b /models/not-a-gguf --dry-run

echo "==> Checking 1B model audit dry run..."
./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b shape audit dry run" "shape_audit: enabled" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b success marker dry run" "status_on_success: model_1b_status: ok" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b selected quantization dry run" "quantization: q8_0" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b json records quantization" "\"quantization\":\"q8_0\"" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b json shape marker dry run" "\"shape\":\"llama32_1b\",\"shape_ready\":true" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b inspect follow-up command" "inspect_command: nanocamelid inspect /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b smoke follow-up command" "smoke_command: nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8" ./scripts/pi/model-1b.sh --dry-run
expect_output "model-1b ready follow-up command" "ready_command: nanocamelid ready 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/model-1b.sh --dry-run

echo "==> Checking 1B model audit rejects non-GGUF model args..."
expect_failure "model-1b invalid model arg" ./scripts/pi/model-1b.sh not-a-model --dry-run
expect_failure "model-1b invalid env model path" env NANOCAMELID_MODEL_GGUF=not-a-model ./scripts/pi/model-1b.sh --dry-run
expect_failure "model-1b repo-local target dir" bash -c 'tmp="$(mktemp "${TMPDIR:-/tmp}/nanocamelid-model-1b.XXXXXX").gguf"; : >"$tmp"; trap "rm -f \"$tmp\"" EXIT; CARGO_TARGET_DIR=target ./scripts/pi/model-1b.sh "$tmp"'

echo "==> Checking 1B Pi smoke launcher dry run..."
./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b help documents prefill batch" "NANOCAMELID_PREFILL_BATCH" ./scripts/pi/smoke-1b.sh --help
expect_output "smoke-1b q4 model audit" "q4_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b q8 model audit" "q8_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b selected source" "selected_source: " ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b selected quantization" "quantization: q8_0" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b context-limited command" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b shape audit dry run" "shape_audit: enabled" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b success marker dry run" "status_on_success: smoke_1b_status: ok" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b json records quantization" "\"quantization\":\"q8_0\"" ./scripts/pi/smoke-1b.sh --dry-run
expect_output "smoke-1b json records shape audit" "\"shape\":\"llama32_1b\",\"shape_ready\":true" ./scripts/pi/smoke-1b.sh --dry-run
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
expect_failure "smoke-1b invalid env model path" env NANOCAMELID_MODEL_GGUF=not-a-model ./scripts/pi/smoke-1b.sh --dry-run
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
expect_output "ready-1b json records quantization" "\"quantization\":\"q8_0\"" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b json records probe" "\"probe\":true" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b json records shape audit" "\"shape\":\"llama32_1b\",\"shape_ready\":true" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b no-chat json success marker dry run" "\"direct_chat\":false,\"chat_tokens\":null" ./scripts/pi/ready-1b.sh --no-chat --dry-run
expect_output "ready-1b selected source" "selected_source: " ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b json records prefill batch" "\"prefill_batch\":32" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b smoke command carries prefill batch" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b chat command carries prefill batch" "chat_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf Say\\ hello\\ in\\ one\\ sentence. 0.0 8" env NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b probe command" "probe_command: nanocamelid probe" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/ready-1b.sh --dry-run
expect_output_order "ready-1b probe before inspect" "probe_command: nanocamelid probe" "inspect_command: nanocamelid inspect" ./scripts/pi/ready-1b.sh --dry-run
expect_output_order "ready-1b model audit before inspect" "model_command: nanocamelid model 1b" "inspect_command: nanocamelid inspect" ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b context-limited smoke command" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/ready-1b.sh --dry-run
expect_output "ready-1b context-limited chat command" "chat_command: NANOCAMELID_CONTEXT_LIMIT=512 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf Say\\ hello\\ in\\ one\\ sentence. 0.0 8" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/ready-1b.sh --dry-run
expect_failure "ready-1b invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad ./scripts/pi/ready-1b.sh --dry-run
expect_failure "ready-1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad ./scripts/pi/ready-1b.sh --dry-run
expect_failure "ready-1b invalid q8 kind" ./scripts/pi/ready-1b.sh q8-broken --dry-run
expect_failure "ready-1b invalid env model path" env NANOCAMELID_MODEL_GGUF=not-a-model ./scripts/pi/ready-1b.sh --dry-run
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
expect_failure "chat-1b invalid env model path" env NANOCAMELID_MODEL_GGUF=not-a-model ./scripts/pi/chat-1b.sh --dry-run
expect_failure "chat-1b repo-local target dir" env CARGO_TARGET_DIR=target ./scripts/pi/chat-1b.sh

echo "==> Checking 1B Pi chat launcher rejects invalid temperature..."
expect_failure "chat-1b invalid temperature" env NANOCAMELID_TEMP=bad ./scripts/pi/chat-1b.sh --dry-run

echo "==> Checking 1B Pi chat launcher ignores smoke env when smoke is disabled..."
env NANOCAMELID_CHAT_SMOKE=0 NANOCAMELID_CHAT_SMOKE_KIND=bad NANOCAMELID_CHAT_SMOKE_TOKENS=bad ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b disabled smoke keeps model audit" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" env NANOCAMELID_CHAT_SMOKE=0 ./scripts/pi/chat-1b.sh --dry-run
expect_output "chat-1b disabled smoke skips smoke command" "smoke_command: skipped" env NANOCAMELID_CHAT_SMOKE=0 ./scripts/pi/chat-1b.sh --dry-run

echo "==> Checking 1B Pi prefill benchmark launcher dry run..."
./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill q4 model audit" "q4_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill q8 model audit" "q8_model: /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill selected source" "selected_source: " ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill selected quantization" "quantization: q8_0" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill smoke env override" "selected_source: NANOCAMELID_SMOKE_GGUF" env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill smoke env model" "model: /models/smoke.gguf" env NANOCAMELID_SMOKE_GGUF=/models/smoke.gguf ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill context limit dry run" "context_limit: 512" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill shape audit dry run" "shape_audit: enabled" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill smoke gate dry run" "smoke_gate: enabled" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill probe command" "probe_command: nanocamelid probe" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill model audit command" "model_command: nanocamelid model 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill inspect command" "inspect_command: nanocamelid inspect /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill smoke command" "smoke_command: NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. 2" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output_order "bench-1b-prefill probe before model audit" "probe_command: nanocamelid probe" "model_command: nanocamelid model 1b" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output_order "bench-1b-prefill inspect before smoke" "inspect_command: nanocamelid inspect" "smoke_command: NANOCAMELID_Q8_DOT_SDOT=1" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output_order "bench-1b-prefill smoke before batch" "smoke_command: NANOCAMELID_Q8_DOT_SDOT=1" "batch_16_command: NANOCAMELID_Q8_DOT_SDOT=1" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill context-limited smoke command" "smoke_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. 2" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill context-limited batch command" "batch_16_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_Q8_DOT_SDOT=1 NANOCAMELID_Q8_DOT_KERNEL=sdot NANOCAMELID_PREFILL_BATCH=16 nanocamelid chat /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. 0.0 2" env NANOCAMELID_CONTEXT_LIMIT=512 ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill success marker dry run" "status_on_success: prefill_bench_1b_status: ok" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill json success marker dry run" "\"benchmark\":\"llama32-1b-prefill\",\"target\":\"llama32-1b\",\"status\":\"ok\"" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill json records quantization" "\"quantization\":\"q8_0\"" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill json records probe" "\"probe\":true" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill json records shape audit" "\"shape\":\"llama32_1b\",\"shape_ready\":true" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_output "bench-1b-prefill json batches dry run" "\"batches\":[1,16,32,64]" ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_failure "bench-1b-prefill invalid context limit" env NANOCAMELID_CONTEXT_LIMIT=bad ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_failure "bench-1b-prefill invalid smoke env model path" env NANOCAMELID_SMOKE_GGUF=not-a-model ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_failure "bench-1b-prefill invalid env model path" env NANOCAMELID_MODEL_GGUF=not-a-model ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_failure "bench-1b-prefill invalid explicit model path" ./scripts/pi/bench-1b-prefill.sh /models/not-a-gguf --dry-run
expect_failure "bench-1b-prefill repo-local target dir" env CARGO_TARGET_DIR=target ./scripts/pi/bench-1b-prefill.sh

echo "==> Checking 1B Pi prefill benchmark launcher rejects invalid generated token count..."
expect_failure "bench-1b-prefill invalid generated token count" env NANOCAMELID_PREFILL_TOKENS=0 ./scripts/pi/bench-1b-prefill.sh --dry-run

echo "==> Checking 1B Pi prefill benchmark launcher rejects invalid temperature..."
expect_failure "bench-1b-prefill invalid temperature" env NANOCAMELID_PREFILL_TEMP=bad ./scripts/pi/bench-1b-prefill.sh --dry-run

echo "==> Checking 1B Pi prefill benchmark launcher rejects invalid batch size..."
expect_failure "bench-1b-prefill invalid batch size" env NANOCAMELID_PREFILL_BATCHES=1,bad,32 ./scripts/pi/bench-1b-prefill.sh --dry-run
expect_failure "bench-1b-prefill duplicate batch size" env NANOCAMELID_PREFILL_BATCHES=16,32,16 ./scripts/pi/bench-1b-prefill.sh --dry-run

echo "==> Checking 1B Pi context-pack launcher dry run..."
./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b help documents prefill batch" "NANOCAMELID_PREFILL_BATCH" ./scripts/pi/context-pack-1b.sh --help
expect_output "context-pack-1b selected source" "selected_source: " ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b selected quantization" "quantization: q8_0" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b shape audit dry run" "shape_audit: enabled" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b success marker dry run" "status_on_success: context_pack_1b_status: ok" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b json records quantization" "\"quantization\":\"q8_0\"" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b json records shape audit" "\"shape\":\"llama32_1b\",\"shape_ready\":true" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b json caps dry run" "\"context_caps\":[512,1024,2048,4096,8192]" ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b prefill batch dry run" "prefill_batch: 32" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b json records prefill batch" "\"prefill_batch\":32" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b command carries prefill batch" "context_512_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_BATCH=32 nanocamelid smoke 1b /mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_PREFILL_BATCH=32 ./scripts/pi/context-pack-1b.sh --dry-run
expect_output "context-pack-1b prompt without explicit kind" "smoke_kind: chat" ./scripts/pi/context-pack-1b.sh "Say hello in one sentence." 3 512,1024 --dry-run
expect_output "context-pack-1b caps without explicit kind" "context_caps: 512 1024" ./scripts/pi/context-pack-1b.sh "Say hello in one sentence." 3 512,1024 --dry-run
expect_failure "context-pack-1b invalid q8 kind" ./scripts/pi/context-pack-1b.sh q8-broken --dry-run
expect_failure "context-pack-1b invalid env model path" env NANOCAMELID_MODEL_GGUF=not-a-model ./scripts/pi/context-pack-1b.sh --dry-run
expect_failure "context-pack-1b invalid explicit model path" ./scripts/pi/context-pack-1b.sh /models/not-a-gguf --dry-run
expect_failure "context-pack-1b invalid prefill batch" env NANOCAMELID_PREFILL_BATCH=bad ./scripts/pi/context-pack-1b.sh --dry-run
expect_failure "context-pack-1b repo-local target dir" env CARGO_TARGET_DIR=target ./scripts/pi/context-pack-1b.sh

echo "==> Checking 1B Pi context-pack launcher rejects invalid context cap..."
expect_failure "context-pack-1b invalid context cap" env NANOCAMELID_CONTEXT_PACKS=512,bad,2048 ./scripts/pi/context-pack-1b.sh --dry-run

echo "==> Checking 1B Pi evidence bundle dry run..."
./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b help documents context packs" "NANOCAMELID_CONTEXT_PACKS" ./scripts/pi/evidence-1b.sh --help
expect_output "evidence-1b success marker dry run" "status_on_success: evidence_1b_status: ok" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b json success marker dry run" "\"target\":\"llama32-1b\",\"status\":\"ok\"" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b json records quantization" "\"quantization\":\"q8_0\"" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b json records context caps" "\"context_pack_caps\":[512,1024,2048,4096,8192]" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b json records prefill batches" "\"prefill_batches\":[1,16,32,64]" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b model command" "model_command: ./scripts/pi/model-1b.sh" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b ready no-chat command" "ready_command: ./scripts/pi/ready-1b.sh chat Say\\ hello\\ in\\ one\\ sentence. 8 --no-chat" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b context-pack command" "context_pack_command: ./scripts/pi/context-pack-1b.sh chat Say\\ hello\\ in\\ one\\ sentence. 8" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b prefill command" "prefill_bench_command: ./scripts/pi/bench-1b-prefill.sh" ./scripts/pi/evidence-1b.sh --dry-run
expect_output "evidence-1b explicit model ready command" "ready_command: ./scripts/pi/ready-1b.sh /models/custom.gguf chat Say\\ hello\\ in\\ one\\ sentence. 8 --no-chat" ./scripts/pi/evidence-1b.sh /models/custom.gguf --dry-run
expect_failure "evidence-1b invalid explicit model path" ./scripts/pi/evidence-1b.sh /models/not-a-gguf --dry-run
expect_failure "evidence-1b invalid smoke kind" env NANOCAMELID_SMOKE_KIND=bad ./scripts/pi/evidence-1b.sh --dry-run
expect_failure "evidence-1b invalid smoke token count" env NANOCAMELID_SMOKE_TOKENS=0 ./scripts/pi/evidence-1b.sh --dry-run
expect_failure "evidence-1b invalid context caps" env NANOCAMELID_CONTEXT_PACKS=512,bad ./scripts/pi/evidence-1b.sh --dry-run
expect_failure "evidence-1b invalid prefill batches" env NANOCAMELID_PREFILL_BATCHES=1,bad ./scripts/pi/evidence-1b.sh --dry-run

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

echo "==> Checking 1B Pi readiness launcher ignores direct chat env when chat is disabled..."
env NANOCAMELID_READY_TEMP=bad NANOCAMELID_READY_TOKENS=0 ./scripts/pi/ready-1b.sh --no-chat --dry-run

echo "==> Checking remote Pi build launcher dry run..."
./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher rejects invalid deploy mode..."
expect_failure "remote_build invalid deploy mode" ./scripts/remote_build.sh "<redacted-pi-host>" "" "" bad-mode --dry-run

echo "==> Checking remote Pi build launcher rejects invalid smoke token count..."
expect_failure "remote_build invalid smoke token count" env NANOCAMELID_SMOKE_TOKENS=bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build invalid explicit model path" env NANOCAMELID_REMOTE_SMOKE_GGUF=not-a-model ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher rejects invalid readiness temperature..."
expect_failure "remote_build invalid readiness temperature" env NANOCAMELID_READY_TEMP=bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher ignores smoke env when remote smoke is disabled..."
env NANOCAMELID_REMOTE_SMOKE=0 NANOCAMELID_SMOKE_TOKENS=bad NANOCAMELID_READY_TEMP=bad NANOCAMELID_READY_TOKENS=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
env NANOCAMELID_REMOTE_SMOKE=0 NANOCAMELID_REMOTE_SMOKE_KIND=bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build disabled smoke skips shape audit" "readiness_shape_audit: skipped" env NANOCAMELID_REMOTE_SMOKE=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher ignores direct chat env when readiness chat is disabled..."
env NANOCAMELID_READY_CHAT=0 NANOCAMELID_READY_TEMP=bad NANOCAMELID_READY_TOKENS=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build shape audit dry run" "readiness_shape_audit: enabled" ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build readiness context limit dry run" "context_limit: 512" env NANOCAMELID_REMOTE_CONTEXT_LIMIT=512 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build readiness command carries context limit" "readiness_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_READY_CHAT=1" env NANOCAMELID_REMOTE_CONTEXT_LIMIT=512 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build invalid readiness context limit" env NANOCAMELID_REMOTE_CONTEXT_LIMIT=bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build no-chat readiness command" "readiness_command: NANOCAMELID_READY_CHAT=0 NANOCAMELID_READY_SMOKE_KIND=chat" env NANOCAMELID_READY_CHAT=0 NANOCAMELID_READY_TEMP=bad NANOCAMELID_READY_TOKENS=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_no_output "remote_build no-chat omits direct chat token env" "NANOCAMELID_READY_TOKENS=0" env NANOCAMELID_READY_CHAT=0 NANOCAMELID_READY_TEMP=bad NANOCAMELID_READY_TOKENS=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_no_output "remote_build no-chat omits direct chat temp env" "NANOCAMELID_READY_TEMP=bad" env NANOCAMELID_READY_CHAT=0 NANOCAMELID_READY_TEMP=bad NANOCAMELID_READY_TOKENS=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher plans optional context packs..."
expect_output "remote_build context-pack dry run" "context_pack_command: NANOCAMELID_CONTEXT_PACKS=512\\,1024 ./scripts/pi/context-pack-1b.sh chat Say\\ hello\\ in\\ one\\ sentence. 8" env NANOCAMELID_REMOTE_CONTEXT_PACKS=512,1024 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher rejects invalid context packs..."
expect_failure "remote_build invalid context cap" env NANOCAMELID_REMOTE_CONTEXT_PACKS=512,bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher plans optional 1B prefill sweep..."
expect_output "remote_build prefill dry run" "prefill_bench_command: NANOCAMELID_PREFILL_PROMPT=Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. NANOCAMELID_PREFILL_TOKENS=2 NANOCAMELID_PREFILL_TEMP=0.0 NANOCAMELID_PREFILL_BATCHES=1\\,16\\,32\\,64 ./scripts/pi/bench-1b-prefill.sh" env NANOCAMELID_REMOTE_PREFILL_BENCH=1 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build prefill command carries context limit" "prefill_bench_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_PREFILL_PROMPT=Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck" env NANOCAMELID_REMOTE_CONTEXT_LIMIT=512 NANOCAMELID_REMOTE_PREFILL_BENCH=1 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build prefill prompt override" "prefill_bench_command: NANOCAMELID_PREFILL_PROMPT=Custom\\ prefill" env NANOCAMELID_REMOTE_PREFILL_BENCH=1 NANOCAMELID_PREFILL_PROMPT="Custom prefill" ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher plans optional 1B evidence bundle..."
expect_output "remote_build evidence dry run" "evidence_command: NANOCAMELID_SMOKE_KIND=chat NANOCAMELID_SMOKE_PROMPT=Say\\ hello\\ in\\ one\\ sentence. NANOCAMELID_SMOKE_TOKENS=8 NANOCAMELID_CONTEXT_PACKS=512\\,1024\\,2048\\,4096\\,8192 NANOCAMELID_PREFILL_PROMPT=Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. NANOCAMELID_PREFILL_TOKENS=2 NANOCAMELID_PREFILL_TEMP=0.0 NANOCAMELID_PREFILL_BATCHES=1\\,16\\,32\\,64 ./scripts/pi/evidence-1b.sh" env NANOCAMELID_REMOTE_EVIDENCE=1 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build evidence delegates shape audit" "readiness_shape_audit: delegated_to_evidence_bundle" env NANOCAMELID_REMOTE_EVIDENCE=1 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build evidence skips composed readiness" "readiness_command: skipped" env NANOCAMELID_REMOTE_EVIDENCE=1 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build evidence command carries context limit" "evidence_command: NANOCAMELID_CONTEXT_LIMIT=512 NANOCAMELID_SMOKE_KIND=chat" env NANOCAMELID_REMOTE_EVIDENCE=1 NANOCAMELID_REMOTE_CONTEXT_LIMIT=512 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_output "remote_build evidence explicit model" "evidence_command: NANOCAMELID_SMOKE_KIND=chat NANOCAMELID_SMOKE_PROMPT=Say\\ hello\\ in\\ one\\ sentence. NANOCAMELID_SMOKE_TOKENS=8 NANOCAMELID_CONTEXT_PACKS=512\\,1024\\,2048\\,4096\\,8192 NANOCAMELID_PREFILL_PROMPT=Explain\\ one\\ practical\\ Raspberry\\ Pi\\ inference\\ bottleneck\\ in\\ two\\ short\\ sentences. NANOCAMELID_PREFILL_TOKENS=2 NANOCAMELID_PREFILL_TEMP=0.0 NANOCAMELID_PREFILL_BATCHES=1\\,16\\,32\\,64 ./scripts/pi/evidence-1b.sh /models/custom.gguf" env NANOCAMELID_REMOTE_EVIDENCE=1 NANOCAMELID_REMOTE_SMOKE_GGUF=/models/custom.gguf ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build evidence requires smoke" env NANOCAMELID_REMOTE_EVIDENCE=1 NANOCAMELID_REMOTE_SMOKE=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking remote Pi build launcher rejects invalid prefill sweep settings..."
expect_failure "remote_build invalid prefill batch" env NANOCAMELID_REMOTE_PREFILL_BENCH=1 NANOCAMELID_REMOTE_PREFILL_BATCHES=1,bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build duplicate prefill batch" env NANOCAMELID_REMOTE_PREFILL_BENCH=1 NANOCAMELID_REMOTE_PREFILL_BATCHES=16,32,16 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build invalid prefill token count" env NANOCAMELID_REMOTE_PREFILL_BENCH=1 NANOCAMELID_PREFILL_TOKENS=0 ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run
expect_failure "remote_build invalid prefill temperature" env NANOCAMELID_REMOTE_PREFILL_BENCH=1 NANOCAMELID_PREFILL_TEMP=bad ./scripts/remote_build.sh "<redacted-pi-host>" --dry-run

echo "==> Checking installer dry run target-dir safety..."
./scripts/install.sh --dry-run
