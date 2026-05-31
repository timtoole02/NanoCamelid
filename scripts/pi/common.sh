#!/usr/bin/env bash

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

require_safe_cargo_target_dir() {
  local target_dir="$1"
  local repo_root="${2:-}"

  case "$target_dir" in
    "" | target | target/* | ./target | ./target/*)
      echo "Refusing to use a repo-local Cargo target dir: ${target_dir:-<empty>}" >&2
      echo "Set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to an external path." >&2
      exit 2
      ;;
  esac

  if [[ -n "$repo_root" ]]; then
    case "$target_dir" in
      "$repo_root"/target | "$repo_root"/target/*)
        echo "Refusing to use a repo-local Cargo target dir: $target_dir" >&2
        echo "Set CARGO_TARGET_DIR or NANOCAMELID_TARGET_DIR to an external path." >&2
        exit 2
        ;;
    esac
  fi

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

looks_like_gguf_path() {
  case "${1:-}" in
    *.[gG][gG][uU][fF] | *.[gG][gG][uU][fF]/) return 0 ;;
    *) return 1 ;;
  esac
}

looks_like_non_gguf_model_path() {
  local value="${1:-}"
  value="${value%/}"

  if looks_like_gguf_path "$value"; then
    return 1
  fi

  case "$value" in
    */* | *\\* | "~"*) return 0 ;;
    *) return 1 ;;
  esac
}

require_gguf_model_path() {
  local label="$1"
  local path="$2"

  if ! looks_like_gguf_path "$path"; then
    echo "$label must be a .gguf path: $path" >&2
    exit 2
  fi
}

require_optional_context_limit() {
  local value="${NANOCAMELID_CONTEXT_LIMIT:-}"

  if [[ -z "$value" ]]; then
    return
  fi

  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "NANOCAMELID_CONTEXT_LIMIT must be a positive integer: $value" >&2
    exit 2
  fi
}

require_optional_prefill_batch() {
  local value="${NANOCAMELID_PREFILL_BATCH:-}"

  if [[ -z "$value" ]]; then
    return
  fi

  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "NANOCAMELID_PREFILL_BATCH must be a positive integer: $value" >&2
    exit 2
  fi
}

prefill_batch_plan_value() {
  if [[ -n "${NANOCAMELID_PREFILL_BATCH:-}" ]]; then
    printf '%s' "$NANOCAMELID_PREFILL_BATCH"
  else
    printf '16'
  fi
}

parse_unique_positive_integer_list() {
  local label="$1"
  local value="$2"
  local item
  local parsed=()
  local seen=" "

  if [[ "$value" =~ (^|,)[[:space:]]*(,|$) ]]; then
    echo "Invalid $label: empty value" >&2
    exit 2
  fi

  for item in ${value//,/ }; do
    if [[ ! "$item" =~ ^[1-9][0-9]*$ ]]; then
      echo "Invalid $label: $item" >&2
      exit 2
    fi
    case "$seen" in
      *" $item "*)
        echo "Duplicate $label: $item" >&2
        exit 2
        ;;
    esac
    parsed+=("$item")
    seen+="$item "
  done
  if [[ ${#parsed[@]} -eq 0 ]]; then
    echo "No $label values were provided." >&2
    exit 2
  fi

  printf '%s\n' "${parsed[@]}"
}

llama32_1b_quantization_for_path() {
  case "$(basename "${1:-}")" in
    Llama-3.2-1B-Instruct-Q4_0.gguf) printf 'q4_0' ;;
    Llama-3.2-1B-Instruct-Q8_0.gguf) printf 'q8_0' ;;
    *) printf 'unknown' ;;
  esac
}

require_unambiguous_1b_quant_selector() {
  local label="$1"
  local quant_selector="$2"
  local explicit_model="${3:-}"

  if [[ -z "$quant_selector" ]]; then
    return
  fi

  if looks_like_gguf_path "$explicit_model"; then
    echo "$label quantization selector cannot be combined with an explicit model path." >&2
    exit 2
  fi

  if [[ -n "${NANOCAMELID_SMOKE_GGUF:-}" || -n "${NANOCAMELID_MODEL_GGUF:-}" ]]; then
    echo "$label quantization selector cannot be combined with NANOCAMELID_SMOKE_GGUF or NANOCAMELID_MODEL_GGUF." >&2
    exit 2
  fi
}
