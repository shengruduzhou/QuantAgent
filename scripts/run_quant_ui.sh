#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="${QUANTAGENT_HOME:-${repo_root}/runtime}"
host="${QUANT_UI_HOST:-127.0.0.1}"
port="${QUANT_UI_PORT:-8000}"
skip_build="${QUANT_UI_SKIP_BUILD:-false}"
reload="${QUANT_UI_RELOAD:-false}"
python_bin="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' \
    "Usage: ./scripts/run_quant_ui.sh [options]" \
    "" \
    "Options:" \
    "  --runtime PATH   Runtime root (default: QUANTAGENT_HOME or ./runtime)" \
    "  --host HOST      Bind host (default: 127.0.0.1)" \
    "  --port PORT      Bind port (default: 8000)" \
    "  --reload         Enable backend auto-reload" \
    "  --skip-build     Reuse the existing frontend dist" \
    "  -h, --help       Show this help"
}

require_value() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    printf 'Missing value for %s\n' "$1" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime)
      require_value "$@"
      runtime="$2"
      shift 2
      ;;
    --host)
      require_value "$@"
      host="$2"
      shift 2
      ;;
    --port)
      require_value "$@"
      port="$2"
      shift 2
      ;;
    --reload)
      reload="true"
      shift
      ;;
    --skip-build)
      skip_build="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
  printf 'Invalid port: %s\n' "$port" >&2
  exit 2
fi

mkdir -p "$runtime"

cd "${repo_root}/apps/quant-ui"
if [[ ! -d node_modules ]]; then
  npm ci
fi

if [[ "$skip_build" != "true" ]]; then
  npm run build
fi

cd "${repo_root}"
printf 'QuantAgent Institutional Workstation\n'
printf '  Runtime: %s\n' "$runtime"
printf '  Web:     http://%s:%s\n' "$host" "$port"
printf '  API:     http://%s:%s/api\n' "$host" "$port"

export QUANTAGENT_HOME="$runtime"
export QUANT_UI_HOST="$host"
export QUANT_UI_PORT="$port"
export QUANT_UI_RELOAD="$reload"

api_args=(--runtime "$runtime" --host "$host" --port "$port")
if [[ "$reload" == "true" ]]; then
  api_args+=(--reload)
else
  api_args+=(--no-reload)
fi

exec "$python_bin" -m services.quant_api "${api_args[@]}"
