#!/usr/bin/env bash
# Forward daily loop (post-close): panel append -> model inference on the new
# date -> hold-band A/B book update + orders + 反T watchlist -> ledger
# freeze/settle -> minute-bar cache refresh for held names.
#
# Predictions from the frozen v8.8 run end 2026-05-07; this loop keeps the
# paper account live beyond that. Run AFTER the close (e.g. 16:30 CST).
# Idempotent; safe to re-run. Logs to runtime/logs/forward/.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="AI_quant_venv/bin/python3"
TODAY="$(date +%F)"
LOG_DIR="runtime/logs/forward"; mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/forward_${TODAY}.log" 2>&1
echo "===== forward daily $TODAY $(date +%T) ====="

# 0) weekday guard (cron may fire on holidays; the panel append no-ops then)
dow=$(date +%u); if [ "$dow" -gt 5 ]; then echo "weekend — skip"; exit 0; fi

# 1) ST snapshot + daily klines -> panel append (flags derived enrich-style)
$PY scripts/fetch_st_list_tickflow.py || echo "WARN st_list refresh"
$PY scripts/update_market_panel_daily.py || { echo "FATAL panel append"; exit 1; }

# 2) score the new date(s) with the frozen v8.8 sleeves
$PY scripts/forward_daily_inference.py || { echo "FATAL inference"; exit 1; }

# 3) hold-band A/B book update + orders + 反T watchlist for tomorrow
$PY scripts/forward_book_update.py || echo "WARN book update"

# 3.5) RL C-book (paper-only; gated DO_NOT_ENABLE for capital 2026-06-12)
$PY scripts/forward_rl_book.py || echo "WARN rl C-book"

# 4) forward research ledger: freeze today's pool, settle elapsed rows
$PY scripts/forward_paper_log.py freeze --as-of "$TODAY" \
    --predictions runtime/reports/v8/forward/ensemble_forward.parquet || echo "WARN ledger freeze"
$PY scripts/forward_paper_log.py settle || echo "WARN ledger settle"

# 5) refresh minute bars for both books' held names (做T replay/eval data)
for book in A_default B_loose; do
  f="runtime/paper/forward/$book/targets_latest.csv"
  [ -f "$f" ] && tail -n +2 "$f" | cut -d, -f1 >> /tmp/fwd_held_syms.txt
done
if [ -s /tmp/fwd_held_syms.txt ]; then
  sort -u /tmp/fwd_held_syms.txt > /tmp/fwd_held_syms_u.txt
  $PY scripts/fetch_tickflow_minute_history.py --symbols-file /tmp/fwd_held_syms_u.txt \
      --start "$(date -d '7 days ago' +%F)" --sleep 0.2 || echo "WARN minute refresh"
  rm -f /tmp/fwd_held_syms.txt /tmp/fwd_held_syms_u.txt
fi

echo "===== forward daily done $(date +%T) ====="
