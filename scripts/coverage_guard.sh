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

# --- value-level partial-bar probe (incident 2026-07-21) -------------------
# Coverage and aggregate-volume checks CANNOT see partial intraday bars: the
# corrupt 07-21 day had 100% coverage and 1.025x aggregate volume. The only
# reliable detector is a direct re-fetch of a few symbols after the close.
echo "--- partial-bar probe on trailing date $(date -Is)"
$PY - <<'EOF'
import sys, time
import pandas as pd
sys.path.insert(0, "scripts")
from update_market_panel_daily import _tf_client
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
p = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "close", "volume"])
p["trade_date"] = pd.to_datetime(p["trade_date"])
last = p["trade_date"].max()
day = p[p["trade_date"] == last].set_index("symbol")
probe = sorted(day.index)[:4] + sorted(day.index)[len(day)//2:len(day)//2 + 2]
tf = _tf_client()
s = int((last - pd.Timedelta(days=5)).timestamp() * 1000)
e = int((last + pd.Timedelta(days=1)).timestamp() * 1000) - 1
bad = 0
for sym in probe:
    try:
        k = pd.DataFrame(tf.klines.get(sym, period="1d", start_time=s, end_time=e, adjust="none"))
        if k.empty:
            continue
        k["d"] = (pd.to_datetime(k["timestamp"], unit="ms", utc=True)
                  .dt.tz_convert("Asia/Shanghai").dt.normalize().dt.tz_localize(None))
        row = k[k["d"] == last]
        if row.empty:
            continue
        vv = float(row.iloc[-1]["volume"]) * 100.0
        pv = float(day.loc[sym, "volume"])
        if vv > 0 and pv / vv < 0.95:
            print(f"PARTIAL_BAR {sym}: panel {pv:,.0f} vs vendor {vv:,.0f} ratio {pv/vv:.3f}")
            bad += 1
    except Exception as ex:
        print(f"probe error {sym}: {str(ex)[:60]}")
    time.sleep(7)   # vendor limit is 10 req/min
print(f"partial_bar_symbols={bad}/{len(probe)} on {last.date()}")
sys.exit(9 if bad else 0)
EOF
probe_rc=$?
if [ $probe_rc -eq 9 ]; then
  echo "PARTIAL_BARS_DETECTED on trailing date -- dropping it (refetched post-close by next run)"
  $PY - <<'EOF'
import shutil
import pandas as pd
from pathlib import Path
PANEL = Path("runtime/data/v7/silver/market_panel/market_panel.parquet")
panel = pd.read_parquet(PANEL)
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
last = panel["trade_date"].max()
shutil.copyfile(PANEL, PANEL.with_suffix(f".pre_partial_drop_{pd.Timestamp.now():%Y%m%d_%H%M%S}.parquet"))
panel[panel["trade_date"] != last].to_parquet(PANEL, index=False)
print(f"dropped {last.date()}")
EOF
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
fi

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

# disk-growth control: every catch-up/drop writes a ~480MB panel backup.
# Keep the 3 most recent; older ones are redundant (git + ledger carry lineage).
KEPT=3
ls -1t runtime/data/v7/silver/market_panel/market_panel.pre_*.parquet 2>/dev/null \
  | tail -n +$((KEPT + 1)) | while read -r old; do
      echo "pruning stale panel backup: $(basename "$old")"; rm -f "$old"
  done
echo "panel backups retained: $(ls -1 runtime/data/v7/silver/market_panel/market_panel.pre_*.parquet 2>/dev/null | wc -l)"
echo "=== coverage guard end $(date -Is) ==="
