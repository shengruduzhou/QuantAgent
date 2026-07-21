#!/bin/bash
# H-029 blind-activation auto-repair pipeline (runs detached in tmux).
# Steps: (1) up to 3 idempotent targeted repair passes for the 2026-07-03..14
# rate-limited symbols; (2) residual-gap audit; (3) deterministic rescore of
# the fresh window (incremental outputs deleted first); (4) daily runner
# re-run to complete the day-1 shadow record; (5) healthcheck.
# All output -> runtime/paper/fresh_blind/auto_repair_20260715.log
set -u
cd /home/shanhefu/QuantAgent
PY=AI_quant_venv/bin/python3
LOG=runtime/paper/fresh_blind/auto_repair_20260715.log
exec >> "$LOG" 2>&1

echo "=== auto-repair start $(date -Is) ==="

# --from-catchup: skip the (already-converged) 07-03..14 repair passes
if [ "${1:-}" = "--from-catchup" ]; then
  echo "skipping repair passes (--from-catchup)"
else
for pass in 1 2 3; do
  echo "--- repair pass $pass $(date -Is)"
  timeout 7200 $PY scripts/repair_window_20260715.py
  echo "pass $pass rc=$?"
  gaps=$($PY - <<'EOF'
import pandas as pd
p = pd.read_parquet("runtime/data/v7/silver/market_panel/market_panel.parquet",
                    columns=["symbol","trade_date"])
p["trade_date"] = pd.to_datetime(p["trade_date"])
seed = set(p.loc[p["trade_date"]=="2026-07-02","symbol"])
win = p[(p["trade_date"]>="2026-07-03")&(p["trade_date"]<="2026-07-14")]
n_dates = win["trade_date"].nunique()
cov = win.groupby("symbol").size()
full = {s for s,c in cov.items() if c>=n_dates}
print(len(seed-full))
EOF
)
  echo "pass $pass residual symbols with gaps: $gaps"
  [ "$gaps" -le 40 ] && break   # 2026-07-04 precedent floor: ~35 delisted/long-suspended
done

fi  # end of repair-pass skip branch

echo "--- chunked panel catch-up (resumable; missing closes since panel max) $(date -Is)"
rc=1
for attempt in 1 2 3; do
  timeout 10800 $PY scripts/catchup_panel_chunked.py
  rc=$?
  echo "catchup attempt $attempt rc=$rc"
  [ $rc -eq 0 ] && break
done
if [ $rc -ne 0 ]; then echo "CATCHUP_FAILED rc=$rc"; echo "=== auto-repair ABORT $(date -Is) ==="; exit 3; fi

echo "--- rescore fresh window $(date -Is)"
rm -f runtime/paper/fresh_blind/daily/composite_forward.parquet \
      runtime/paper/fresh_blind/daily/sleeve_scores.parquet
timeout 10800 $PY scripts/forward_daily_inference.py \
  --run-dir runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300 \
  --start 2026-05-08 --device cuda \
  --output runtime/paper/fresh_blind/daily/composite_forward.parquet \
  --sleeve-scores-output runtime/paper/fresh_blind/daily/sleeve_scores.parquet
rc=$?
if [ $rc -ne 0 ]; then echo "RESCORE_FAILED rc=$rc"; echo "=== auto-repair ABORT $(date -Is) ==="; exit 2; fi

echo "--- daily runner $(date -Is)"
timeout 10800 $PY scripts/fresh_blind_daily.py
echo "runner rc=$?"

echo "--- healthcheck $(date -Is)"
$PY scripts/fresh_blind_healthcheck.py
echo "healthcheck rc=$?"
echo "=== auto-repair end $(date -Is) ==="
