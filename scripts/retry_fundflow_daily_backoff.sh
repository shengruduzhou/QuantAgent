#!/usr/bin/env bash
# Gentle intraday backoff retry for the BLOCKED push2his daily fund-flow fetch.
# Probes with ONE lightweight request per cycle (does not hammer / extend the
# ban). The moment push2his responds, it fetches the full 120d history + runs the
# cross-sectional IC eval, then exits. Otherwise backs off until MAX_ATTEMPTS,
# after which the daily cron takes over.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="AI_quant_venv/bin/python3"
SYMS="${1:-runtime/tmp/minute_cache_symbols.txt}"
OUT_IC="runtime/reports/intraday_dot_ev_fundflow/daily_factor_ic.json"
mkdir -p runtime/reports/intraday_dot_ev_fundflow
MAX_ATTEMPTS=40
backoff=120          # start 2 min
backoff_cap=300      # cap 5 min

probe() {  # exit 0 if push2his responds with data
  $PY - <<'PROBE' >/dev/null 2>&1
import requests
r=requests.get("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
  params={"secid":"1.600519","fields1":"f1,f2,f3,f7","fields2":"f51,f52,f53,f54,f55,f56,f57","lmt":"120"},
  headers={"User-Agent":"Mozilla/5.0","Referer":"https://quote.eastmoney.com/"},timeout=8)
raise SystemExit(0 if r.json().get("data",{}).get("klines") else 1)
PROBE
}

for n in $(seq 1 $MAX_ATTEMPTS); do
  if probe; then
    echo "[$(date +%T)] push2his CLEARED on attempt $n — fetching daily fund-flow ($(wc -l < "$SYMS") names)"
    $PY scripts/fetch_eastmoney_fundflow_daily.py --symbols-file "$SYMS" --sleep 0.5
    rc=$?
    if [ $rc -eq 0 ]; then
      $PY scripts/eval_fundflow_daily_factor.py | tee "$OUT_IC"
      echo "[$(date +%T)] DONE — daily fund-flow IC written to $OUT_IC"
      exit 0
    fi
    echo "[$(date +%T)] fetch returned rc=$rc (partial/empty) — will retry"
  else
    echo "[$(date +%T)] attempt $n: still blocked; sleeping ${backoff}s"
  fi
  sleep "$backoff"
  backoff=$(( backoff * 5 / 4 )); [ "$backoff" -gt "$backoff_cap" ] && backoff=$backoff_cap
done
echo "[$(date +%T)] gave up after $MAX_ATTEMPTS attempts — daily cron will retry"
exit 1
