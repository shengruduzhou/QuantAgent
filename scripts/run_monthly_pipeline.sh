#!/usr/bin/env bash
# Monthly pipeline: refresh slow evidence (红头文件 crawl / 投行研报 / LLM 十五五
# 政策研判), fuse all evidence + 舆情, build the LLM+factor hybrid pool, and write
# the monthly research report (选股池参考). Logs to runtime/logs/monthly/.
#
# IMPORTANT: the prediction universe is never inferred from a dated demo file.
# Pass it as argv[1] or QUANTAGENT_MONTHLY_PREDICTIONS_PATH. Missing/stale lineage
# is safer than silently researching the wrong universe, so this input fails closed.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="AI_quant_venv/bin/python3"
QA="AI_quant_venv/bin/quantagent"
TODAY="$(date +%F)"; YM="$(date +%Y%m)"
LOG_DIR="runtime/logs/monthly"; mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/monthly_${YM}.log" 2>&1

echo "===== monthly pipeline $TODAY $(date +%T) ====="

PREDICTIONS_PATH="${1:-${QUANTAGENT_MONTHLY_PREDICTIONS_PATH:-}}"
if [[ -z "$PREDICTIONS_PATH" ]]; then
    echo "ERROR monthly pipeline requires an explicit predictions artifact as argv[1] or QUANTAGENT_MONTHLY_PREDICTIONS_PATH" >&2
    exit 2
fi
if [[ ! -f "$PREDICTIONS_PATH" ]]; then
    echo "ERROR predictions artifact does not exist: $PREDICTIONS_PATH" >&2
    exit 2
fi

# Bind every downstream research surface to the same immutable input identity.
# The manifest records path/hash/as-of before any network/data refresh begins.
$PY - "$PREDICTIONS_PATH" "$TODAY" "$YM" <<'PYEOF'
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

prediction_path = Path(sys.argv[1]).resolve()
as_of = sys.argv[2]
year_month = sys.argv[3]
if not prediction_path.is_file():
    raise SystemExit(f"predictions artifact missing: {prediction_path}")

digest = sha256(prediction_path.read_bytes()).hexdigest()
out = Path("runtime/reports/monthly") / f"input_manifest_{year_month}.json"
out.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "schema": "quantagent.monthly-research-input.v1",
    "as_of": as_of,
    "predictions_path": str(prediction_path),
    "predictions_sha256": digest,
    "implicit_demo_fallback": False,
    "research_only": True,
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"monthly predictions path={prediction_path} sha256={digest}")
print(f"monthly input manifest={out}")
PYEOF
if [[ $? -ne 0 ]]; then
    echo "ERROR failed to bind monthly predictions provenance" >&2
    exit 2
fi

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

# 3) 投行研报 + 舆情 — same explicitly bound prediction universe
$PY scripts/fetch_broker_reports.py --symbols-from "$PREDICTIONS_PATH" --lookback-days 120 --min-events 3 || echo "WARN broker"
$PY scripts/fetch_news_sentiment.py --symbols-from "$PREDICTIONS_PATH" --lookback-days 21 --as-of "$TODAY" || echo "WARN news"

# 4) 融合证据 (政策驱动方向 + 舆情个股)
$PY scripts/build_combined_canonical.py || echo "WARN combine"

# 5) LLM+因子 混合股池 (research-only, no orders) — exact same artifact
$QA build-llm-hybrid-stock-pool-v8 \
    --predictions-path "$PREDICTIONS_PATH" \
    --canonical-evidence-path runtime/data/v7/silver/combined_canonical.parquet \
    --as-of-date "$TODAY" --candidate-pool-size 52 --stock-top-n 20 --sector-top-n 12 \
    --capital 200000 --allow-network --allow-fallback \
    --output-dir runtime/reports/v8/llm_hybrid_combined || echo "WARN hybrid"

# 6) 月度研报
$PY scripts/monthly_research_report.py --as-of "$TODAY" || echo "WARN monthly_report"
echo "===== monthly pipeline done $(date +%T) ====="
