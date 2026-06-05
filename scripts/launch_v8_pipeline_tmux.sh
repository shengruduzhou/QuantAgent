#!/usr/bin/env bash
# Launch the v8 training pipeline inside a tmux session with three panes:
#   pane 0 — the training process
#   pane 1 — live log tail
#   pane 2 — every-30s status (router diagnostics, latest backtest metrics)
#
# Usage:
#   bash scripts/launch_v8_pipeline_tmux.sh
#   tmux attach -t qa_v8_pipeline
#
# Override behaviour via env vars (defaults shown below).
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-qa_v8_pipeline}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shanhefu/QuantAgent}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/AI_quant_venv/bin/python}"
RUN_TAG="${RUN_TAG:-v8_$(date +%Y%m%d_%H%M%S)}"

# Training scope (default = top-200 silver panel symbols, 2022-2024 window)
UNIVERSE_FILE="${UNIVERSE_FILE:-$PROJECT_ROOT/runtime/reports/v8/pipeline/universe_top200.txt}"
START_DATE="${START_DATE:-2022-01-04}"
END_DATE="${END_DATE:-2024-12-31}"

HORIZON_CLASS="${HORIZON_CLASS:-short_5d}"
TOP_K="${TOP_K:-20}"
GA_POP="${GA_POP:-24}"
GA_GEN="${GA_GEN:-10}"

SILVER_PANEL="${SILVER_PANEL:-$PROJECT_ROOT/runtime/data/v7/silver/market_panel/market_panel.parquet}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runtime/reports/v8/pipeline/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/runtime/logs/v8_pipeline}"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
LOG_PATH="$LOG_DIR/${RUN_TAG}.log"

# Sanity checks
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "FATAL: python binary not found at $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$SILVER_PANEL" ]]; then
  echo "FATAL: silver panel not found at $SILVER_PANEL" >&2
  exit 1
fi
if [[ ! -f "$UNIVERSE_FILE" ]]; then
  echo "INFO: universe file missing — generating top-200 by liquidity from silver panel"
  "$PYTHON_BIN" - <<PY
import pandas as pd, pathlib
mp = pd.read_parquet('$SILVER_PANEL', columns=['symbol','trade_date','amount'])
cutoff = mp['trade_date'].max() - pd.Timedelta(days=365)
top = mp[mp['trade_date']>=cutoff].groupby('symbol')['amount'].mean().sort_values(ascending=False).head(200)
pathlib.Path('$UNIVERSE_FILE').parent.mkdir(parents=True, exist_ok=True)
open('$UNIVERSE_FILE','w').write(','.join(top.index.astype(str)))
print('wrote', '$UNIVERSE_FILE')
PY
fi

SYMBOLS="$(tr -d '\n' < "$UNIVERSE_FILE")"
N_SYMS=$(($(awk -F',' '{print NF}' "$UNIVERSE_FILE")))

cat <<EOF
============================================================
v8 training pipeline launch
  session     : $SESSION_NAME
  run tag     : $RUN_TAG
  universe    : $N_SYMS symbols (from $UNIVERSE_FILE)
  date range  : $START_DATE → $END_DATE
  horizon     : $HORIZON_CLASS  top-K=$TOP_K
  GA          : pop=$GA_POP gen=$GA_GEN
  output      : $OUTPUT_ROOT
  log         : $LOG_PATH
============================================================
EOF

# Kill an existing session with the same name (idempotent re-launch)
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tearing down stale session $SESSION_NAME"
  tmux kill-session -t "$SESSION_NAME"
fi

# Build the training command. The tee'd log is what monitor pane #1 tails.
TRAIN_CMD="$PYTHON_BIN -m quantagent.cli train-v8-pipeline \
  --symbols '$SYMBOLS' \
  --start-date $START_DATE --end-date $END_DATE \
  --use-silver-panel '$SILVER_PANEL' \
  --horizon-class $HORIZON_CLASS --top-k $TOP_K \
  --ga-population $GA_POP --ga-generations $GA_GEN \
  --output-dir '$OUTPUT_ROOT' 2>&1 | tee -a '$LOG_PATH'"

# pane 0 — training
tmux new-session -d -s "$SESSION_NAME" -n train \
  "cd '$PROJECT_ROOT' && echo '[start] $(date)' >> '$LOG_PATH' && ${TRAIN_CMD}; echo '[end] $(date) — exit=\$?' >> '$LOG_PATH'; echo READY_TO_CLOSE; sleep 60"

# pane 1 — log tail
tmux split-window -t "$SESSION_NAME:0" -h "tail -F '$LOG_PATH'"

# pane 2 — periodic status (router + backtest metrics)
STATUS_CMD="while true; do clear; echo '[\$(date)]' '$RUN_TAG'; echo '-- output dir --'; ls -la '$OUTPUT_ROOT' 2>/dev/null | head -15; echo '-- router --'; cat '$OUTPUT_ROOT/router_diagnostics.json' 2>/dev/null | head -25; echo '-- backtest metrics --'; cat '$OUTPUT_ROOT/backtest/metrics.json' 2>/dev/null; sleep 30; done"
tmux split-window -t "$SESSION_NAME:0.1" -v "$STATUS_CMD"

tmux select-pane -t "$SESSION_NAME:0.0"
tmux select-layout -t "$SESSION_NAME:0" tiled >/dev/null 2>&1 || true

cat <<EOF

[ok] tmux session '$SESSION_NAME' detached.
attach with        : tmux attach -t $SESSION_NAME
log path           : $LOG_PATH
output artifacts   : $OUTPUT_ROOT
list panes         : tmux list-panes -t $SESSION_NAME
kill session       : tmux kill-session -t $SESSION_NAME
EOF
