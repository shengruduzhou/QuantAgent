#!/usr/bin/env bash
# v10 ensemble training: 3 seeds × 12 folds, sequentially on single GPU.
# Each seed writes to its own model dir. After all 3 finish, average per-fold
# predictions and run the deployed-sleeve backtest on the ensemble.
#
# Usage:
#   bash scripts/run_v10_ensemble.sh                 # foreground (blocks)
#   nohup bash scripts/run_v10_ensemble.sh > runtime/logs/v10_ensemble.log 2>&1 &
#
# Resume note: each seed checks for completed walk_forward/ and skips
# already-done seeds — safe to re-run if interrupted.

set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS=(1729 4096 8191)
DATASET="runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet"
BASE_OUT="runtime/models/v7_alpha_full_universe_nosynth_v10"
LOG_DIR="runtime/logs"
mkdir -p "$LOG_DIR"

for seed in "${SEEDS[@]}"; do
  out="${BASE_OUT}_seed${seed}"
  marker="$out/_seed_completed.txt"
  if [ -f "$marker" ]; then
    echo "[$(date '+%H:%M:%S')] seed=$seed already completed (marker $marker present) — skipping"
    continue
  fi
  echo "[$(date '+%H:%M:%S')] === starting seed $seed → $out ==="
  QA_TRAINING_DATASET="$DATASET" \
    QA_TRAINING_OUTPUT="$out" \
    QA_MIN_SYNTH_FEATURES=0 \
    QA_N_SPLITS=12 \
    QA_FT_SEED="$seed" \
    QA_SKIP_FINAL_FIT=1 \
    AI_quant_venv/bin/python -u scripts/run_full_universe_train.py \
      > "$LOG_DIR/v10_seed${seed}.log" 2>&1
  echo "$(date '+%Y-%m-%dT%H:%M:%S')" > "$marker"
  echo "[$(date '+%H:%M:%S')] === seed $seed DONE ==="
done

echo "[$(date '+%H:%M:%S')] all seeds done — aggregating ensemble"
AI_quant_venv/bin/python scripts/aggregate_v10_ensemble.py \
  > "$LOG_DIR/v10_ensemble_aggregate.log" 2>&1
echo "[$(date '+%H:%M:%S')] ensemble pipeline complete"
