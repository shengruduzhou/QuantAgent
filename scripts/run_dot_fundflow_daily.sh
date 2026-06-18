#!/usr/bin/env bash
# Standalone daily do-T order-flow accumulation + auto-monitor (post A-share close).
#
# Isolated from run_forward_daily.sh so it does NOT resurrect the full forward
# book pipeline -- it only collects what the order-flow do-T research needs:
#   1) refresh 1-min bars for the held names (last 7d)
#   2) snapshot 东财 per-minute fund flow for the held names (free, ~50 req/day)
#   3) run the monitor: once >= MIN_DAYS forward days accumulate, it auto-trains
#      the EV edge-frontier WITH vs WITHOUT order-flow and records the verdict.
#
# Schedule (system TZ = Asia/Tokyo; A-share close 15:00 CST = 16:00 JST):
#   40 16 * * 1-5  /home/shanhefu/QuantAgent/scripts/run_dot_fundflow_daily.sh
# Idempotent; safe to re-run. Logs to runtime/logs/dot_fundflow/.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="AI_quant_venv/bin/python3"
TODAY="$(date +%F)"
LOG_DIR="runtime/logs/dot_fundflow"; mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/dot_fundflow_${TODAY}.log" 2>&1
echo "===== dot fundflow daily $TODAY $(date +%T %Z) ====="

# weekday guard in CST (the A-share calendar), not JST
dow=$(TZ=Asia/Shanghai date +%u); if [ "$dow" -gt 5 ]; then echo "CST weekend — skip"; exit 0; fi

SYMS=/tmp/dot_ff_syms.txt; rm -f "$SYMS"
for book in A_default B_loose; do
  f="runtime/paper/forward/$book/targets_latest.csv"
  [ -f "$f" ] && tail -n +2 "$f" | cut -d, -f1 >> "$SYMS"
done
# fallback to the cached research universe if no live book
[ -s "$SYMS" ] || cp runtime/tmp/minute_cache_symbols.txt "$SYMS" 2>/dev/null

if [ -s "$SYMS" ]; then
  sort -u "$SYMS" -o "$SYMS"
  echo "held names: $(wc -l < "$SYMS")"
  $PY scripts/fetch_tickflow_minute_history.py --symbols-file "$SYMS" \
      --start "$(date -d '7 days ago' +%F)" --sleep 0.2 || echo "WARN minute refresh"
  $PY scripts/collect_eastmoney_fundflow_minute.py --symbols-file "$SYMS" --sleep 0.4 || echo "WARN fundflow collect"
else
  echo "no symbols available — skip collection"
fi

$PY scripts/intraday_dot_ev_fundflow_monitor.py || echo "WARN monitor"

# 8) DAILY fund-flow factor (push2his, 120d history → backtestable). push2his has
#    intermittent mainland-IP connection risk-control; probe first and only fetch
#    when it responds (avoid hammering a blocked endpoint). One success = full
#    history; then evaluate cross-sectional IC for the main daily model.
if $PY - <<'PROBE' >/dev/null 2>&1
import requests
r=requests.get("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
  params={"secid":"1.600519","fields1":"f1,f2,f3,f7","fields2":"f51,f52,f53,f54,f55,f56,f57","lmt":"120"},
  headers={"User-Agent":"Mozilla/5.0","Referer":"https://quote.eastmoney.com/"},timeout=8)
raise SystemExit(0 if r.json().get("data",{}).get("klines") else 1)
PROBE
then
  echo "push2his responding — fetching daily fund-flow factor"
  $PY scripts/fetch_eastmoney_fundflow_daily.py --symbols-file "${SYMS:-runtime/tmp/minute_cache_symbols.txt}" --sleep 0.5 \
      && $PY scripts/eval_fundflow_daily_factor.py > runtime/reports/intraday_dot_ev_fundflow/daily_factor_ic.json 2>&1 \
      && echo "daily fund-flow IC -> runtime/reports/intraday_dot_ev_fundflow/daily_factor_ic.json"
else
  echo "push2his blocked today — skip daily fund-flow fetch (will retry tomorrow)"
fi
echo "===== dot fundflow daily done $(date +%T) ====="
