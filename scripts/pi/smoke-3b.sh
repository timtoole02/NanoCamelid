#!/usr/bin/env bash
# Run NanoCamelid's default Pi-local 3B smoke validation.
set -euo pipefail

WORKSPACE="${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}"
REPO="${NANOCAMELID_REPO:-$WORKSPACE/src/NanoCamelid}"
TARGET_DIR="${CARGO_TARGET_DIR:-${NANOCAMELID_TARGET_DIR:-/mnt/nanocamelid/target}}"
MODEL="${NANOCAMELID_SMOKE_GGUF:-${NANOCAMELID_MODEL_GGUF:-$WORKSPACE/models/Llama-3.2-3B-Instruct-Q4_0.gguf}}"
DRY_RUN=0
POSITIONAL_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      POSITIONAL_ARGS+=("$arg")
      ;;
  esac
done
if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
  set -- "${POSITIONAL_ARGS[@]}"
else
  set --
fi
SMOKE_KIND="${1:-${NANOCAMELID_SMOKE_KIND:-chat}}"
SMOKE_PROMPT="${2:-${NANOCAMELID_SMOKE_PROMPT:-Say hello in one sentence.}}"
SMOKE_TOKENS="${3:-${NANOCAMELID_SMOKE_TOKENS:-8}}"
BINARY="${NANOCAMELID_BIN:-$TARGET_DIR/release/nanocamelid}"
export NANOCAMELID_Q8_DOT_SDOT="${NANOCAMELID_Q8_DOT_SDOT:-1}"
export NANOCAMELID_Q8_DOT_KERNEL="${NANOCAMELID_Q8_DOT_KERNEL:-sdot}"

if [[ "$SMOKE_KIND" != "model" && "$SMOKE_KIND" != "chat" && "$SMOKE_KIND" != "q8-model" && "$SMOKE_KIND" != "q8-chat" ]]; then
  echo "Unknown smoke kind: $SMOKE_KIND" >&2
  echo "Expected model, chat, q8-model, or q8-chat." >&2
  exit 2
fi

if [[ "$DRY_RUN" == "1" ]] && command -v cargo >/dev/null 2>&1; then
  launcher_mode="cargo"
elif [[ -x "$BINARY" ]]; then
  launcher_mode="binary"
elif command -v cargo >/dev/null 2>&1; then
  launcher_mode="cargo"
else
  echo "NanoCamelid release binary not found and cargo is not on PATH." >&2
  echo "Expected binary: $BINARY" >&2
  exit 3
fi

run_nanocamelid() {
  if [[ "$launcher_mode" == "binary" ]]; then
    "$BINARY" "$@"
    return
  fi

  cd "$REPO"
  export CARGO_TARGET_DIR="$TARGET_DIR"
  cargo run --release -- "$@"
}

if [[ "$DRY_RUN" == "1" ]]; then
  run_nanocamelid smoke 3b "$MODEL" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS" --dry-run
  exit 0
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Set NANOCAMELID_SMOKE_GGUF=/path/to/model.gguf, set NANOCAMELID_MODEL_GGUF=/path/to/model.gguf, or place the 3B Q4_0 GGUF at the default path." >&2
  exit 2
fi

echo "Running $SMOKE_KIND against $MODEL"
run_nanocamelid smoke 3b "$MODEL" "$SMOKE_KIND" "$SMOKE_PROMPT" "$SMOKE_TOKENS"
