#!/usr/bin/env bash
# QuantAgent V7 — daily data-layer health check
#
# Usage:
#   ./scripts/daily_health_check.sh [--lake-root PATH] [--output-root PATH] [--no-write]
#
# Exit codes (forwarded from the Python health checker):
#   0  all gates OK
#   1  at least one WARN, none FAIL
#   2  at least one FAIL  ← systemd OnFailure triggers here
#
# The script resolves paths relative to the repo root so it can be called
# from any working directory (systemd ConditionPathExists needs absolute paths;
# configure WorkingDirectory= in the .service unit instead).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAKE_ROOT="${LAKE_ROOT:-${REPO_ROOT}/runtime/data/v7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runtime/reports/daily_health}"

# Pick the right python: prefer the project venv, fall back to PATH.
if [ -x "${REPO_ROOT}/AI_quant_venv/bin/python" ]; then
    PY="${REPO_ROOT}/AI_quant_venv/bin/python"
elif [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.venv/bin/activate"
    PY="python"
else
    PY="python"
fi

cd "${REPO_ROOT}"

exec "${PY}" -m quantagent.cli \
    health-check-v7 \
    --lake-root "${LAKE_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    "$@"
