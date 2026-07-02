#!/usr/bin/env bash
# Stage 10 daily driver — run post-close each trading day (cron 17:30 JST).
# FAIL-SOFT: every step is wrapped so a throttled data source or a single failure
# can NEVER abort the pipeline. Each stage is time-bounded.
cd "$(dirname "$0")/.." || exit 1
PY=AI_quant_venv/bin/python3
export PYTHONUNBUFFERED=1
echo "######## Stage10 daily $(date -Iseconds) ########"

# 1) scan + PIT snapshot (10.1/10.2) — has its own retry+cache fallback
timeout 1800 $PY scripts/stage10_daily_scan.py 2>&1 | grep -vE "it/s|it\]|[0-9]+%\|" | tail -6
echo "--- scan rc=${PIPESTATUS[0]} ---"

# 2) live 公告 + 主营 verification (10.3) — fail-soft, stale fallback on throttle
timeout 1200 $PY scripts/stage10_verify_orders.py --live 2>&1 | grep -vE "it/s|it\]|[0-9]+%\|" | tail -4 || echo "verify failed (non-fatal)"

# 3) paper-trade: generate today's portfolio + roll forward marks (10.4)
timeout 600 $PY scripts/stage10_paper_trade.py --action both 2>&1 | tail -20 || echo "paper-trade failed (non-fatal)"

# 4) daily health check (10.4b)
timeout 120 $PY scripts/stage10_health_check.py || echo "health-check failed (non-fatal)"

echo "######## Stage10 daily done $(date -Iseconds) ########"
