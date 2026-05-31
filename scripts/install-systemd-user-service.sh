#!/usr/bin/env bash
set -euo pipefail

service_name="${NANOCAMELID_SERVICE_NAME:-nanocamelid}"
binary="${NANOCAMELID_BIN:-}"
host="${NANOCAMELID_SERVICE_HOST:-127.0.0.1}"
port="${NANOCAMELID_SERVICE_PORT:-8080}"
model_dir="${NANOCAMELID_MODEL_DIR:-/mnt/nanocamelid/models}"
max_request_bytes="${NANOCAMELID_MAX_REQUEST_BYTES:-65536}"
max_input_tokens="${NANOCAMELID_MAX_INPUT_TOKENS:-2048}"
max_output_tokens="${NANOCAMELID_MAX_OUTPUT_TOKENS:-256}"
api_key="${NANOCAMELID_API_KEY:-}"
unit_dir="${NANOCAMELID_SYSTEMD_USER_DIR:-$HOME/.config/systemd/user}"
config_dir="${NANOCAMELID_CONFIG_DIR:-$HOME/.config/nanocamelid}"
enable_now=0
dry_run=0

usage() {
  cat <<'USAGE'
Usage: install-systemd-user-service.sh [options]

Installs a systemd user service for the NanoCamelid local API server.
The service binds to 127.0.0.1:8080 by default and uses the default Pi model
directory at /mnt/nanocamelid/models.

Options:
  --binary <path>              nanocamelid binary, default NANOCAMELID_BIN or PATH lookup
  --model-dir <path>           Model directory, default /mnt/nanocamelid/models
  --host <addr>                Bind address, default 127.0.0.1
  --port <port>                Bind port, default 8080
  --api-key <token>            Store bearer token in a 0600 EnvironmentFile
  --max-request-bytes <count>  HTTP request byte cap, default 65536
  --max-input-tokens <count>   Request input token cap, default 2048
  --max-output-tokens <count>  Response token cap, default 256
  --service-name <name>        systemd unit name without .service, default nanocamelid
  --enable-now                 Run systemctl --user enable --now after writing the unit
  --dry-run                    Print the resolved unit and commands without writing files

Env:
  NANOCAMELID_BIN
  NANOCAMELID_MODEL_DIR
  NANOCAMELID_API_KEY
  NANOCAMELID_SERVICE_HOST
  NANOCAMELID_SERVICE_PORT
  NANOCAMELID_MAX_REQUEST_BYTES
  NANOCAMELID_MAX_INPUT_TOKENS
  NANOCAMELID_MAX_OUTPUT_TOKENS
  NANOCAMELID_SERVICE_NAME
USAGE
}

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

validate_name() {
  if [[ ! "$1" =~ ^[A-Za-z0-9_.@-]+$ ]]; then
    echo "Invalid service name: $1" >&2
    echo "Use only letters, digits, dots, underscores, @, or hyphens." >&2
    exit 2
  fi
}

validate_no_newline() {
  local label="$1"
  local value="$2"
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "$label must not contain newlines" >&2
    exit 2
  fi
}

systemd_quote() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

env_quote() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

while [[ "$#" -gt 0 ]]; do
  arg="$1"
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --binary)
      if [[ -z "${2:-}" ]]; then
        echo "--binary requires a path" >&2
        exit 2
      fi
      binary="$2"
      shift
      ;;
    --model-dir)
      if [[ -z "${2:-}" ]]; then
        echo "--model-dir requires a path" >&2
        exit 2
      fi
      model_dir="$2"
      shift
      ;;
    --host)
      if [[ -z "${2:-}" ]]; then
        echo "--host requires an address" >&2
        exit 2
      fi
      host="$2"
      shift
      ;;
    --port)
      if [[ -z "${2:-}" ]]; then
        echo "--port requires a port" >&2
        exit 2
      fi
      port="$2"
      shift
      ;;
    --api-key)
      if [[ -z "${2:-}" ]]; then
        echo "--api-key requires a token" >&2
        exit 2
      fi
      api_key="$2"
      shift
      ;;
    --max-request-bytes)
      if [[ -z "${2:-}" ]]; then
        echo "--max-request-bytes requires a count" >&2
        exit 2
      fi
      max_request_bytes="$2"
      shift
      ;;
    --max-input-tokens)
      if [[ -z "${2:-}" ]]; then
        echo "--max-input-tokens requires a count" >&2
        exit 2
      fi
      max_input_tokens="$2"
      shift
      ;;
    --max-output-tokens)
      if [[ -z "${2:-}" ]]; then
        echo "--max-output-tokens requires a count" >&2
        exit 2
      fi
      max_output_tokens="$2"
      shift
      ;;
    --service-name)
      if [[ -z "${2:-}" ]]; then
        echo "--service-name requires a name" >&2
        exit 2
      fi
      service_name="$2"
      shift
      ;;
    --enable-now)
      enable_now=1
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
  shift
done

validate_name "$service_name"
validate_no_newline "binary" "$binary"
validate_no_newline "model-dir" "$model_dir"
validate_no_newline "host" "$host"
validate_no_newline "api-key" "$api_key"

if ! is_positive_int "$port" || (( port > 65535 )); then
  echo "--port must be an integer from 1 to 65535" >&2
  exit 2
fi
if ! is_positive_int "$max_request_bytes"; then
  echo "--max-request-bytes must be a positive integer" >&2
  exit 2
fi
if ! is_positive_int "$max_input_tokens"; then
  echo "--max-input-tokens must be a positive integer" >&2
  exit 2
fi
if ! is_positive_int "$max_output_tokens"; then
  echo "--max-output-tokens must be a positive integer" >&2
  exit 2
fi

if [[ -z "$binary" ]]; then
  if command -v nanocamelid >/dev/null 2>&1; then
    binary="$(command -v nanocamelid)"
  else
    binary="$HOME/.local/bin/nanocamelid"
  fi
fi

case "$binary" in
  /*) ;;
  *)
    echo "--binary must be an absolute path: $binary" >&2
    exit 2
    ;;
esac

unit_path="$unit_dir/$service_name.service"
env_path="$config_dir/$service_name.env"
api_key_required=false
if [[ -n "$api_key" ]]; then
  api_key_required=true
fi

exec_start="$(systemd_quote "$binary") serve --host $(systemd_quote "$host") --port $(systemd_quote "$port") --model-dir $(systemd_quote "$model_dir") --max-request-bytes $(systemd_quote "$max_request_bytes") --max-input-tokens $(systemd_quote "$max_input_tokens") --max-output-tokens $(systemd_quote "$max_output_tokens")"
readonly_model_path="-$(systemd_quote "$model_dir")"
env_file_path="-$(systemd_quote "$env_path")"

printf -v unit_content '%s\n' \
  "[Unit]" \
  "Description=NanoCamelid local API server" \
  "Documentation=https://github.com/timtoole02/NanoCamelid" \
  "After=network.target" \
  "" \
  "[Service]" \
  "Type=simple" \
  "EnvironmentFile=$env_file_path" \
  "ExecStart=$exec_start" \
  "Restart=on-failure" \
  "RestartSec=2s" \
  "NoNewPrivileges=true" \
  "PrivateTmp=true" \
  "ProtectSystem=strict" \
  "ProtectHome=read-only" \
  "ReadOnlyPaths=$readonly_model_path" \
  "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6" \
  "IPAddressDeny=any" \
  "IPAddressAllow=localhost" \
  "" \
  "[Install]" \
  "WantedBy=default.target"

printf -v env_content '%s\n' \
  "# NanoCamelid service environment." \
  "# Set NANOCAMELID_API_KEY to require Authorization: Bearer <token>."
if [[ -n "$api_key" ]]; then
  env_content="${env_content}NANOCAMELID_API_KEY=$(env_quote "$api_key")
"
else
  env_content="${env_content}# NANOCAMELID_API_KEY=\"replace-me\"
"
fi

if [[ "$dry_run" == "1" ]]; then
  echo "NanoCamelid systemd user service dry run"
  echo "service_name: $service_name"
  echo "unit_path: $unit_path"
  echo "env_path: $env_path"
  echo "listen: http://$host:$port"
  echo "model_dir: $model_dir"
  echo "api_key_required: $api_key_required"
  echo "enable_now: $enable_now"
  echo "exec_start: $exec_start"
  echo "unit:"
  printf '%s\n' "$unit_content"
  echo "commands: install -m 0600 env file; install -m 0644 unit file; systemctl --user daemon-reload"
  if [[ "$enable_now" == "1" ]]; then
    echo "enable_command: systemctl --user enable --now $service_name.service"
  else
    echo "start_command: systemctl --user start $service_name.service"
  fi
  exit 0
fi

need install
need systemctl

if [[ ! -x "$binary" ]]; then
  echo "NanoCamelid binary not executable: $binary" >&2
  echo "Install NanoCamelid first or pass --binary <path>." >&2
  exit 1
fi

mkdir -p "$unit_dir" "$config_dir"
printf '%s\n' "$env_content" | install -m 0600 /dev/stdin "$env_path"
printf '%s\n' "$unit_content" | install -m 0644 /dev/stdin "$unit_path"
systemctl --user daemon-reload

echo "NanoCamelid systemd user service installed:"
echo "  unit: $unit_path"
echo "  env:  $env_path"
echo "  listen: http://$host:$port"
echo "  api_key_required: $api_key_required"

if [[ "$enable_now" == "1" ]]; then
  systemctl --user enable --now "$service_name.service"
  echo "  status: enabled and started"
else
  echo
  echo "Start it with:"
  echo "  systemctl --user start $service_name.service"
  echo "Enable it at login with:"
  echo "  systemctl --user enable $service_name.service"
fi
