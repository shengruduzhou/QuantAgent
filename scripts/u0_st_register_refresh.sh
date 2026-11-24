#!/bin/bash
# Daily refresh of the risk-warning (ST) evidence that the fresh-window catch-up
# reads: runtime/data/u0/pit/st_pit_evidence.parquet.
#
#   SZSE      exchange short-name register (ENG_ST30 publish, unchanged).
#   SSE/BSE   ENG_ST31 forward-anchored register: a new TickFlow full-universe
#             name snapshot (anchor), cninfo announcements and per-company dated
#             short names. Rows are written only if the one-shot validation
#             passed AND today's SZSE monitor holds (ENG_ST31 §6–§7).
#
# Only month windows still open when last fetched are fetched again. A failed
# snapshot or name fetch is tolerated (the register runs on what it has); a
# failed crawl or publish aborts and leaves the previous table in place, whose
# as-of date then stops short of new sessions, so the catch-up fails closed.
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
$PY scripts/u0_st_forward.py snapshot || echo "WARN: instrument snapshot failed; no new anchor today"
$PY scripts/u0_st_forward.py company-names --mode production \
    || echo "WARN: some company name lists failed; older lists are used"
$PY scripts/u0_st_forward.py publish
echo "=== ST register refresh end $(date -Is) ==="
