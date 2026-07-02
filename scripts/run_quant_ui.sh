#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}/apps/quant-ui"
if [[ ! -d node_modules ]]; then
  npm install
fi

if [[ "${QUANT_UI_SKIP_BUILD:-false}" != "true" ]]; then
  npm run build
fi

cd "${repo_root}"
exec python3 -m services.quant_api
