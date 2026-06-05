#!/usr/bin/env bash
# Monthly pipeline: refresh slow evidence (红头文件 crawl / 投行研报 / LLM 十五五
# 政策研判), fuse all evidence + 舆情, build the LLM+factor hybrid pool, and write
# the monthly research report (选股池参考). Logs to runtime/logs/monthly/.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="AI_quant_venv/bin/python3"
QA="AI_quant_venv/bin/quantagent"
TODAY="$(date +%F)"; YM="$(date +%Y%m)"
LOG_DIR="runtime/logs/monthly"; mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/monthly_${YM}.log" 2>&1
echo "===== monthly pipeline $TODAY $(date +%T) ====="

# 1) 红头文件爬虫 → policy_events silver
$QA ingest-policy --as-of "$TODAY" --themes auto --allow-network --active-discovery --max-per-source 12 || echo "WARN ingest-policy"
$PY - <<'PYEOF' || echo "WARN policy transform"
import pandas as pd, os
src=f"runtime/data/v7/evidence/evidence_{__import__('datetime').date.today()}.csv"
if os.path.exists(src):
    df=pd.read_csv(src)
    raw=pd.DataFrame({"source":df["source_name"],"announced_at":df["published_at"],
        "title":df["title"],"body_summary":df.get("body",""),"url":df.get("url","")})
    os.makedirs("runtime/data/v7/raw/policy",exist_ok=True)
    raw.to_csv("runtime/data/v7/raw/policy/policy_raw.csv",index=False)
PYEOF
$QA import-policy-events-v7 --input runtime/data/v7/raw/policy/policy_raw.csv \
    --source-version "crawl_${YM}" --min-events 3 --min-theme-coverage 0.30 || echo "WARN policy import"

# 2) LLM 十五五 政策方向研判 (authoritative sector direction)
$PY scripts/fetch_llm_policy_priorities.py --as-of "$TODAY" --top-n 10 || echo "WARN llm_priorities"

# 3) 投行研报 + 舆情
$PY scripts/fetch_broker_reports.py --symbols-from runtime/tmp/demo_preds_20260605.parquet --lookback-days 120 --min-events 3 || echo "WARN broker"
$PY scripts/fetch_news_sentiment.py --symbols-from runtime/tmp/demo_preds_20260605.parquet --lookback-days 21 --as-of "$TODAY" || echo "WARN news"

# 4) 融合证据 (政策驱动方向 + 舆情个股)
$PY scripts/build_combined_canonical.py || echo "WARN combine"

# 5) LLM+因子 混合股池 (research-only, no orders)
$QA build-llm-hybrid-stock-pool-v8 \
    --predictions-path runtime/tmp/demo_preds_20260605.parquet \
    --canonical-evidence-path runtime/data/v7/silver/combined_canonical.parquet \
    --as-of-date "$TODAY" --candidate-pool-size 52 --stock-top-n 20 --sector-top-n 12 \
    --capital 200000 --allow-network --allow-fallback \
    --output-dir runtime/reports/v8/llm_hybrid_combined || echo "WARN hybrid"

# 6) 月度研报
$PY scripts/monthly_research_report.py --as-of "$TODAY" || echo "WARN monthly_report"
echo "===== monthly pipeline done $(date +%T) ====="
