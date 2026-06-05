#!/usr/bin/env bash
# Daily pre-open pipeline: refresh fast-moving evidence (舆情/国家队/债市) and
# emit the daily sentiment brief. Heavy policy crawl is monthly (run_monthly).
# Idempotent; safe to re-run. Logs to runtime/logs/daily/.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="AI_quant_venv/bin/python3"
QA="AI_quant_venv/bin/quantagent"
TODAY="$(date +%F)"
LOG_DIR="runtime/logs/daily"; mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/daily_${TODAY}.log" 2>&1
echo "===== daily pipeline $TODAY $(date +%T) ====="

# 1) 舆情: 每日新闻情绪短线推断 (own universe via 人气榜, robust fallback)
$PY scripts/daily_sentiment_brief.py --as-of "$TODAY" --top-n 60 --lookback-days 3 || echo "WARN daily_sentiment_brief"

# 2) 国家队 (top-10 holder 汇金/证金) — cheap if quarter unchanged
$PY scripts/fetch_state_team.py --output-root runtime/data/v7 --min-events 3 || echo "WARN state_team"

# 3) 债市收益率曲线
$PY scripts/fetch_bond_flows.py --lookback-days 120 || echo "WARN bond_flows"
$QA import-bond-flows-v7 --input runtime/data/v7/raw/bond/bond_yields_raw.csv \
    --source akshare:bond_china_yield --min-days 30 --min-date-continuity 0.90 || echo "WARN bond import"

echo "===== daily pipeline done $(date +%T) ====="
