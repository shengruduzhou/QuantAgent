#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}/apps/quant-ui"

if [[ ! -d node_modules ]]; then
  npm install
fi

exec npm run dev
