#!/usr/bin/env bash
# Intraday 做T (T+0) runner — run DURING trading hours (not pre-open) on the current
# held pool. Pulls TickFlow 1-minute bars → 做T 买卖带 + 盘口防护. Research output only.
#
# Deploy as a cron every ~30 min on trading days, 09:35–14:55 Asia/Shanghai, e.g.:
#   */30 9-14 * * 1-5  /home/shanhefu/QuantAgent/scripts/run_intraday_dot.sh
# (09:25 集合竞价 also worth a run for opening-auction sentiment.)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="AI_quant_venv/bin/python3"
TODAY="$(date +%F)"
LOG_DIR="runtime/logs/intraday"; mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/intraday_${TODAY}.log" 2>&1
echo "===== intraday 做T $TODAY $(date +%T) ====="

# Current held pool = latest weekly chain pool (factor ∪ 产业链). Fall back to newest pool file.
POOL="${INTRADAY_POOL:-}"
if [ -z "$POOL" ]; then
  POOL="$(ls -t runtime/reports/monthly/chain_pool_*.parquet 2>/dev/null | grep -vE 'nonews|scrambled|offline|candidates' | head -1)"
fi
if [ -z "$POOL" ] || [ ! -f "$POOL" ]; then
  echo "no pool found (run the weekly research first)"; exit 1
fi
echo "pool=$POOL"
$PY scripts/intraday_dot_signals.py --pool "$POOL" || echo "WARN intraday_dot_signals"
echo "===== intraday 做T done $(date +%T) ====="
