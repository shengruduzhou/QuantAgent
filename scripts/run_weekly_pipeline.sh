#!/usr/bin/env bash
# Weekly research pipeline (用户要求"一周出一个研报"): run the产业链深度Agent for the
# latest factor-prediction date in --live mode (richest PIT news) → a detailed
# 舆情全景 + 多产业链 + 财报深挖 + 因子∪产业链 股池 report. Research-only, no orders.
# Logs to runtime/logs/weekly/.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="AI_quant_venv/bin/python3"
TODAY="$(date +%F)"; WK="$(date +%G-W%V)"
LOG_DIR="runtime/logs/weekly"; mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/weekly_${WK}.log" 2>&1
echo "===== weekly pipeline $TODAY ($WK) $(date +%T) ====="

# Prefer a fresh real prediction file if one exists for today, else fall back to the
# latest date in the canonical short_5d predictions parquet (write it to runtime/tmp).
PRED_SRC="${WEEKLY_PRED_SRC:-runtime/reports/v8/deep/v8_full_v3_20260602_051048/short_5d/predictions.parquet}"
ASOF="$($PY - "$PRED_SRC" "$TODAY" <<'PYEOF'
import sys, pandas as pd
src, today = sys.argv[1], sys.argv[2]
p = pd.read_parquet(src); p["trade_date"] = pd.to_datetime(p["trade_date"])
asof = min(p["trade_date"].max(), pd.Timestamp(today))
day = p[p["trade_date"] == p[p["trade_date"] <= asof]["trade_date"].max()]
out = f"runtime/tmp/real_preds_{day['trade_date'].iloc[0].strftime('%Y%m%d')}.parquet"
day[["trade_date","symbol","alpha_score"]].rename(columns={"alpha_score":"prediction"}).to_parquet(out, index=False)
print(day["trade_date"].iloc[0].strftime("%Y-%m-%d"))
PYEOF
)"
echo "as_of=$ASOF preds=runtime/tmp/real_preds_$(echo "$ASOF" | tr -d '-').parquet"

# 产业链深度研报 (live 模式: news_cctv + 最新投行研报). 6 链 / 每环节4只 / 财报深挖40只.
$PY scripts/industry_chain_research.py --as-of "$ASOF" \
    --predictions "runtime/tmp/real_preds_$(echo "$ASOF" | tr -d '-').parquet" \
    --live --n-chains 6 --stocks-per-segment 4 --max-stocks-enrich 40 \
    --n-factor 20 --n-chain 15 || echo "WARN chain_research"

# 向前实盘纸面记录 (唯一真正前视无关的OOS): 冻结本周 因子/链/并集 股池, 供未来用当时
# 还不存在的价格评分; 并结算已到期(forward窗口走完且有价)的历史条目.
PRED_TMP="runtime/tmp/real_preds_$(echo "$ASOF" | tr -d '-').parquet"
CHAIN_POOL="runtime/reports/monthly/chain_pool_${ASOF}.parquet"
$PY scripts/forward_paper_log.py freeze --as-of "$ASOF" \
    --predictions "$PRED_TMP" --chain-pool "$CHAIN_POOL" \
    --n-factor 20 --n-chain 15 --fw 0.6 --fwd-td 5 || echo "WARN forward_freeze"
$PY scripts/forward_paper_log.py settle || echo "WARN forward_settle"

echo "===== weekly pipeline done $(date +%T) ====="
