#!/usr/bin/env bash
# Launch the v8 multi-horizon sweep inside tmux with 3 panes:
#   pane 0 — sweep driver (scripts/run_v8_sweep.py)
#   pane 1 — live log tail
#   pane 2 — periodic status (latest metrics + artifact tree)
#
# Usage:
#   bash scripts/launch_v8_pipeline_sweep_tmux.sh
#   tmux attach -t qa_v8_sweep
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-qa_v8_sweep}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shanhefu/QuantAgent}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/AI_quant_venv/bin/python}"
RUN_TAG="${RUN_TAG:-v8_sweep_$(date +%Y%m%d_%H%M%S)}"

UNIVERSE_FILE="${UNIVERSE_FILE:-$PROJECT_ROOT/runtime/reports/v8/pipeline/universe_top500.txt}"
START_DATE="${START_DATE:-2019-01-02}"
END_DATE="${END_DATE:-2024-12-31}"
TOP_K="${TOP_K:-30}"
GA_POP="${GA_POP:-48}"
GA_GEN="${GA_GEN:-20}"

SILVER_PANEL="${SILVER_PANEL:-$PROJECT_ROOT/runtime/data/v7/silver/market_panel/market_panel.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runtime/reports/v8/pipeline/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/runtime/logs/v8_pipeline}"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
LOG_PATH="$LOG_DIR/${RUN_TAG}.log"

[[ -x "$PYTHON_BIN" ]] || { echo "missing python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$SILVER_PANEL" ]] || { echo "missing silver panel: $SILVER_PANEL" >&2; exit 1; }
[[ -f "$UNIVERSE_FILE" ]] || { echo "missing universe file: $UNIVERSE_FILE" >&2; exit 1; }

N_SYMS=$(($(awk -F',' '{print NF}' "$UNIVERSE_FILE")))

cat <<EOF
============================================================
v8 multi-horizon sweep
  session     : $SESSION_NAME
  run tag     : $RUN_TAG
  universe    : $N_SYMS symbols
  date range  : $START_DATE → $END_DATE
  horizons    : short_5d, mid_5d_30d, long_30d_120d
  top-K=$TOP_K  GA pop=$GA_POP gen=$GA_GEN
  output      : $OUTPUT_ROOT
  log         : $LOG_PATH
============================================================
EOF

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tearing down stale session $SESSION_NAME"
  tmux kill-session -t "$SESSION_NAME"
fi

# pane 0 — sweep driver
DRIVER_CMD="cd '$PROJECT_ROOT' && '$PYTHON_BIN' scripts/run_v8_sweep.py \
  --python-bin '$PYTHON_BIN' \
  --symbols-file '$UNIVERSE_FILE' \
  --silver-panel '$SILVER_PANEL' \
  --start-date $START_DATE --end-date $END_DATE \
  --output-root '$OUTPUT_ROOT' \
  --top-k $TOP_K --ga-population $GA_POP --ga-generations $GA_GEN \
  2>&1 | tee -a '$LOG_PATH'"

tmux new-session -d -s "$SESSION_NAME" -n sweep \
  "echo '[start] '\$(date) >> '$LOG_PATH'; $DRIVER_CMD; echo '[end] '\$(date) >> '$LOG_PATH'; echo READY_TO_CLOSE; sleep 60"

# pane 1 — log tail
tmux split-window -t "$SESSION_NAME:0" -h "tail -F '$LOG_PATH'"

# pane 2 — periodic status
STATUS_CMD="while true; do clear; echo '[' \$(date) '] $RUN_TAG'; echo '-- artifacts --'; find '$OUTPUT_ROOT' -maxdepth 3 -type f 2>/dev/null | head -25; echo '-- metrics per horizon --'; for h in short_5d mid_5d_30d long_30d_120d; do echo \">> \$h\"; cat '$OUTPUT_ROOT/'\$h'/backtest/metrics.json' 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); [print(f\"  {k}: {v}\") for k,v in d.items()]' 2>/dev/null; done; sleep 30; done"
tmux split-window -t "$SESSION_NAME:0.1" -v "$STATUS_CMD"

tmux select-pane -t "$SESSION_NAME:0.0"
tmux select-layout -t "$SESSION_NAME:0" tiled >/dev/null 2>&1 || true

cat <<EOF

[ok] tmux session '$SESSION_NAME' detached.
attach           : tmux attach -t $SESSION_NAME
log              : $LOG_PATH
artifacts        : $OUTPUT_ROOT
inspect panes    : tmux list-panes -t $SESSION_NAME
stop run         : tmux kill-session -t $SESSION_NAME
EOF
