#!/bin/bash
# H-029 activation: trailing partial-day coverage guard.
#
# Problem it prevents: catchup_panel_chunked's window ends at TODAY, and a
# multi-hour pass can cross the 15:00 CST close mid-run. Symbols fetched before
# the close carry no bar for today; symbols fetched after do. The merge then
# writes a partial day (~40% coverage observed risk on 2026-07-21), which is
# worse than having no day at all: cross-sectional ranks, and therefore that
# date's book decisions, would be computed on a biased half-universe while
# every downstream status still reads OK.
#
# Guard: after the supervisor releases its lock, drop TRAILING dates whose
# coverage is below MIN_COV of the reference count (panel tail backed up
# first), then re-run rescore -> daily runner -> healthcheck so all artifacts
# reflect the corrected panel. Dropped days are NOT lost: the next scheduled
# supervisor run refetches them in a clean post-close window.
#
# Only TRAILING dates are touched (a low-coverage day with complete days after
# it is a historical gap for the repair pass, not a partial-fetch artifact).
set -u
cd /home/shanhefu/QuantAgent
PY=AI_quant_venv/bin/python3
LOG=runtime/paper/fresh_blind/coverage_guard.log
MIN_COV=${MIN_COV:-0.93}   # FRESH_HOLDOUT_FREEZE_MANIFEST QC gate
exec >> "$LOG" 2>&1

echo "=== coverage guard start $(date -Is) ==="

# wait for the catch-up supervisor to release its lock (blocking, bounded by
# the caller's tmux lifetime); prevents mutating the panel mid-merge
LOCK=runtime/paper/fresh_blind/.catchup_supervisor.lock
exec 9>"$LOCK"
echo "waiting for supervisor lock $(date -Is)"
flock 9
echo "lock acquired $(date -Is)"

dropped=$($PY - "$MIN_COV" <<'EOF'
import shutil, sys
import pandas as pd
from pathlib import Path

MIN_COV = float(sys.argv[1])
PANEL = Path("runtime/data/v7/silver/market_panel/market_panel.parquet")
panel = pd.read_parquet(PANEL)
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
cov = panel[panel["trade_date"] >= panel["trade_date"].max() - pd.Timedelta(days=30)] \
        .groupby("trade_date").size().sort_index()
if len(cov) < 4:
    print(""); raise SystemExit(0)
ref = int(cov.iloc[:-3].median()) if len(cov) > 3 else int(cov.median())
bad = []
for d in reversed(cov.index):                 # trailing dates only
    if cov[d] < MIN_COV * ref:
        bad.append(d)
    else:
        break
if not bad:
    print(""); raise SystemExit(0)
backup = PANEL.with_suffix(f".pre_coverage_guard_{pd.Timestamp.now():%Y%m%d_%H%M%S}.parquet")
shutil.copyfile(PANEL, backup)
keep = panel[~panel["trade_date"].isin(bad)]
keep.to_parquet(PANEL, index=False)
print(",".join(str(d.date()) for d in sorted(bad)), file=sys.stderr)
print(",".join(str(d.date()) for d in sorted(bad)))
EOF
)

if [ -n "$dropped" ]; then
  echo "DROPPED partial trailing dates: $dropped (panel backed up; refetched by next supervisor run)"
  echo "--- rescore after correction $(date -Is)"
  rm -f runtime/paper/fresh_blind/daily/composite_forward.parquet \
        runtime/paper/fresh_blind/daily/sleeve_scores.parquet
  timeout 10800 $PY scripts/forward_daily_inference.py \
    --run-dir runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300 \
    --start 2026-05-08 --device cuda \
    --output runtime/paper/fresh_blind/daily/composite_forward.parquet \
    --sleeve-scores-output runtime/paper/fresh_blind/daily/sleeve_scores.parquet
  echo "rescore rc=$?"
  timeout 10800 $PY scripts/fresh_blind_daily.py
  echo "runner rc=$?"
else
  echo "no partial trailing dates -- panel coverage clean, no action"
fi

$PY scripts/fresh_blind_healthcheck.py
echo "healthcheck rc=$?"
echo "=== coverage guard end $(date -Is) ==="
