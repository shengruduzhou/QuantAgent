#!/bin/bash
# Daily refresh of the ENG_ST30 risk-warning register (cninfo announcements +
# SZSE short-name register) and re-publication of the strict PIT ST table
# runtime/data/u0/pit/st_pit_evidence.parquet that the fresh-window catch-up reads.
#
# Only month windows that were still open when last fetched are fetched again.
# `publish` applies the frozen HOLDOUT/C1-C3 results: an exchange that did not pass
# has no rows and its market rows stay refused. Any failure leaves the previous
# table in place; its as-of date then stops short of new sessions, so the
# catch-up fails closed instead of guessing.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-AI_quant_venv/bin/python3}
C=runtime/data/u0/pit/_st_announcements
END=$(TZ=Asia/Shanghai date +%F)
echo "=== ST register refresh start $(date -Is) end=$END ==="
$PY scripts/u0_st_announcements.py crawl --cache "$C" --end "$END"
$PY scripts/u0_st_announcements.py keyword-crawl --cache "$C" --end "$END"
$PY scripts/u0_st_announcements.py pdf --cache "$C"
$PY scripts/u0_st_announcements.py refresh-szse --cache "$C"
$PY scripts/u0_st_announcements.py build --cache "$C"
$PY scripts/u0_st_announcements.py publish --cache "$C"
echo "=== ST register refresh end $(date -Is) ==="
