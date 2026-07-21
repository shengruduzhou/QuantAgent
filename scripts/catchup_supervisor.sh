#!/bin/bash
# H-029 activation: persistent catch-up supervisor (runs detached in tmux).
#
# Problem it solves: TickFlow throttling degraded to ~2-3 s/call, so ONE
# full-universe pass (~3.6k symbols) exceeds the 3 h budget the daily runner
# and auto_repair allow. catchup_panel_chunked.py stages progress and resumes,
# but a single bounded attempt per day can never converge -- each cron run
# times out mid-window and the ALERT persists (observed 07-18..07-20).
#
# This supervisor re-enters the resumable catch-up until the panel actually
# reaches the latest AVAILABLE trading close (recomputed every iteration, so a
# session that crosses 15:00 CST picks up the new day), then hands off to the
# full auto-repair pipeline (rescore -> daily runner -> healthcheck).
#
# Bounded: MAX_ITERS iterations, each with its own timeout. No unbounded loop.
# All output -> runtime/paper/fresh_blind/catchup_supervisor.log
set -u
cd /home/shanhefu/QuantAgent
PY=AI_quant_venv/bin/python3
LOG=runtime/paper/fresh_blind/catchup_supervisor.log
MAX_ITERS=${MAX_ITERS:-8}
ITER_TIMEOUT=${ITER_TIMEOUT:-10800}
exec >> "$LOG" 2>&1

# single-instance lock: concurrent catch-ups would race on _staging_catchup/
LOCK=runtime/paper/fresh_blind/.catchup_supervisor.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "=== catchup supervisor SKIP $(date -Is): another instance holds the lock ==="
  exit 0
fi

echo "=== catchup supervisor start $(date -Is) (max_iters=$MAX_ITERS timeout=${ITER_TIMEOUT}s) ==="

target_and_state() {
  # prints: "<panel_max> <target> <state>"  state in {CURRENT, BEHIND}
  $PY - <<'EOF'
import pandas as pd
from datetime import datetime, timedelta, timezone
panel = pd.read_parquet("runtime/data/v7/silver/market_panel/market_panel.parquet",
                        columns=["trade_date"])
pmax = pd.to_datetime(panel["trade_date"]).max().normalize()
# China close 15:00 CST; treat a day's close as available from 15:30 CST
cst = datetime.now(timezone.utc) + timedelta(hours=8)
try:
    import akshare as ak
    cal = pd.to_datetime(ak.tool_trade_date_hist_sina()["trade_date"]).sort_values()
    days = cal[cal <= pd.Timestamp(cst.date())]
    target = days.iloc[-1].normalize()
    if target.date() == cst.date() and (cst.hour * 60 + cst.minute) < 15 * 60 + 30:
        target = days.iloc[-2].normalize()   # today's close not out yet
except Exception:                            # calendar unavailable -> weekday fallback
    d = pd.Timestamp(cst.date())
    if (cst.hour * 60 + cst.minute) < 15 * 60 + 30:
        d -= pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    target = d
print(f"{pmax.date()} {target.date()} {'CURRENT' if pmax >= target else 'BEHIND'}")
EOF
}

for i in $(seq 1 "$MAX_ITERS"); do
  read -r pmax target state <<< "$(target_and_state)"
  echo "--- iter $i $(date -Is): panel_max=$pmax target=$target state=$state"
  if [ "$state" = "CURRENT" ]; then
    echo "panel is current -> catch-up converged"
    break
  fi
  timeout "$ITER_TIMEOUT" $PY scripts/catchup_panel_chunked.py
  rc=$?
  echo "iter $i catchup rc=$rc $(date -Is)"
  if [ $rc -ne 0 ] && [ $rc -ne 124 ]; then
    echo "CATCHUP_HARD_ERROR rc=$rc (not a timeout) -- stopping, staging preserved"
    echo "=== catchup supervisor ABORT $(date -Is) ==="
    exit 3
  fi
  # rc=124 (timeout) is EXPECTED and benign: staging persists, next iter resumes
done

read -r pmax target state <<< "$(target_and_state)"
echo "--- post-loop $(date -Is): panel_max=$pmax target=$target state=$state"
if [ "$state" != "CURRENT" ]; then
  echo "NOT_CONVERGED after $MAX_ITERS iters (panel_max=$pmax target=$target)"
  echo "=== catchup supervisor end-incomplete $(date -Is) ==="
  exit 4
fi

echo "--- handing off to auto-repair (rescore -> runner -> healthcheck) $(date -Is)"
bash scripts/auto_repair_20260715.sh --from-catchup
echo "auto_repair rc=$?"
echo "=== catchup supervisor end $(date -Is) ==="
