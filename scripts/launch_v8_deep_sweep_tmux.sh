#!/usr/bin/env bash
# Launch the v8 FT-Transformer (GPU) sweep across 3 horizons inside tmux.
#
# Panes (3-up layout):
#   0 — sweep driver (scripts/run_v8_deep_sweep.py)
#   1 — live log tail (filtered)
#   2 — nvidia-smi + per-horizon metrics refresh
#
# Usage:
#   bash scripts/launch_v8_deep_sweep_tmux.sh
#   tmux attach -t qa_v8_deep
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-qa_v8_deep}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shanhefu/QuantAgent}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/AI_quant_venv/bin/python}"
RUN_TAG="${RUN_TAG:-v8_deep_$(date +%Y%m%d_%H%M%S)}"

UNIVERSE_FILE="${UNIVERSE_FILE:-$PROJECT_ROOT/runtime/reports/v8/pipeline/universe_top500.txt}"
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet}"
SILVER_PANEL="${SILVER_PANEL:-$PROJECT_ROOT/runtime/data/v7/silver/market_panel/market_panel.parquet}"

TRAIN_START="${TRAIN_START:-2018-01-02}"
TRAIN_END="${TRAIN_END:-2023-06-30}"
TEST_END="${TEST_END:-2024-12-31}"
EMBARGO="${EMBARGO:-30}"
TOP_K="${TOP_K:-30}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
D_TOKEN="${D_TOKEN:-128}"
N_BLOCKS="${N_BLOCKS:-4}"
N_HEADS="${N_HEADS:-8}"
DATES_PER_STEP="${DATES_PER_STEP:-8}"
TRAIN_MICRO_BATCH="${TRAIN_MICRO_BATCH:-}"
CROSS_SECTIONAL_NORM="${CROSS_SECTIONAL_NORM:-rank}"
LABEL_NORM="${LABEL_NORM:-1}"
ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.10}"
FFN_DROPOUT="${FFN_DROPOUT:-0.10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
LR="${LR:-0.001}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runtime/reports/v8/deep/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/runtime/logs/v8_deep}"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
LOG_PATH="$LOG_DIR/${RUN_TAG}.log"

[[ -x "$PYTHON_BIN" ]] || { echo "missing python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$DATASET_PATH" ]] || { echo "missing dataset: $DATASET_PATH" >&2; exit 1; }
[[ -f "$SILVER_PANEL" ]] || { echo "missing silver panel: $SILVER_PANEL" >&2; exit 1; }
[[ -f "$UNIVERSE_FILE" ]] || { echo "missing universe file: $UNIVERSE_FILE" >&2; exit 1; }

N_SYMS=$(($(awk -F',' '{print NF}' "$UNIVERSE_FILE")))

cat <<EOF
============================================================
v8 deep GPU sweep
  session     : $SESSION_NAME
  run tag     : $RUN_TAG
  universe    : $N_SYMS symbols
  date range  : $TRAIN_START → $TRAIN_END  (train) / $TEST_END (test)
  embargo     : $EMBARGO bdays
  horizons    : short_5d → mid_5d_30d → long_30d_120d
  FT-Trans    : epochs=$MAX_EPOCHS  batch=$BATCH_SIZE  d_token=$D_TOKEN  blocks=$N_BLOCKS  heads=$N_HEADS  lr=$LR
  top-K       : $TOP_K
  GPU         : require GPU=yes (NVIDIA via torch.cuda)
  output      : $OUTPUT_ROOT
  log         : $LOG_PATH
============================================================
EOF

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tearing down stale session $SESSION_NAME"
  tmux kill-session -t "$SESSION_NAME"
fi

DRIVER_CMD="cd '$PROJECT_ROOT' && '$PYTHON_BIN' scripts/run_v8_deep_sweep.py \
  --python-bin '$PYTHON_BIN' \
  --symbols-file '$UNIVERSE_FILE' \
  --dataset-path '$DATASET_PATH' \
  --silver-panel-path '$SILVER_PANEL' \
  --train-start $TRAIN_START --train-end $TRAIN_END --test-end $TEST_END \
  --output-root '$OUTPUT_ROOT' \
  --embargo-days $EMBARGO --top-k $TOP_K \
  --max-epochs $MAX_EPOCHS --batch-size $BATCH_SIZE \
  --d-token $D_TOKEN --n-blocks $N_BLOCKS --n-heads $N_HEADS \
  --dates-per-step $DATES_PER_STEP \
  --cross-sectional-norm $CROSS_SECTIONAL_NORM \
  --label-norm $LABEL_NORM \
  --attention-dropout $ATTENTION_DROPOUT \
  --ffn-dropout $FFN_DROPOUT \
  --weight-decay $WEIGHT_DECAY \
  --early-stopping-patience $EARLY_STOP_PATIENCE \
  ${TRAIN_MICRO_BATCH:+--train-micro-batch $TRAIN_MICRO_BATCH} \
  --learning-rate $LR \
  2>&1 | tee -a '$LOG_PATH'"

tmux new-session -d -s "$SESSION_NAME" -n train \
  "echo '[start] '\$(date) >> '$LOG_PATH'; $DRIVER_CMD; echo '[end] '\$(date) >> '$LOG_PATH'; echo READY_TO_CLOSE; sleep 120"

# Pane 1 — filtered log tail
tmux split-window -t "$SESSION_NAME:0" -h \
  "tail -F '$LOG_PATH' | grep -vE 'FutureWarning|daily = merged|pkg_resources'"

# Pane 2 — nvidia-smi + per-horizon metrics
STATUS_CMD=$(cat <<MON
while true; do
  clear
  echo "[$(date)] $RUN_TAG"
  echo "-- nvidia-smi --"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null
  echo
  echo "-- artifacts --"
  find '$OUTPUT_ROOT' -maxdepth 3 -type f 2>/dev/null | sort | head -30
  echo
  for h in short_5d mid_5d_30d long_30d_120d; do
    f='$OUTPUT_ROOT'/\$h/backtest/metrics.json
    if [[ -f "\$f" ]]; then
      echo ">> \$h"
      cat "\$f" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"  total_return={d[\"total_return\"]:.4f}  sharpe={d[\"sharpe\"]:.3f}  max_dd={d[\"max_drawdown\"]:.4f}  n_trades={d[\"n_trades\"]}")' 2>/dev/null
    else
      echo ">> \$h: pending"
    fi
  done
  sleep 15
done
MON
)
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
