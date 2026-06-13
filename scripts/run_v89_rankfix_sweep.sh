#!/usr/bin/env bash
# v8.9 retrain on the rank-fixed v88 dataset (batch-local-rank corruption fix).
#
# Mirrors the v8.8 run (runtime/reports/v8/deep/v88_judgment_20260611_2015
# per-sleeve run_config.json) exactly, except:
#   * --dataset-path -> training_dataset_alpha181_exec_v88_rankfix.parquet
#   * long_30d_120d gets --train-micro-batch 1024 up front (v8.8's long sleeve
#     OOM'd in the sweep and had to be retried with it).
# Run inside the GPU tmux session. ~6h total on the RTX 3090.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="AI_quant_venv/bin/python3"
TS="$(date +%Y%m%d_%H%M)"
ROOT="runtime/reports/v8/deep/v89_rankfix_${TS}"
LOG="runtime/logs/v8_deep/v89_rankfix_${TS}.log"
mkdir -p "$ROOT" runtime/logs/v8_deep
exec >>"$LOG" 2>&1
echo "===== v8.9 rankfix sweep start $(date -Is) root=$ROOT ====="

COMMON=(
  --dataset-path runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v88_rankfix.parquet
  --silver-panel-path runtime/data/v7/silver/market_panel/market_panel.parquet
  --symbols-file runtime/data/v7/universe_v88_comma.txt
  --train-start 2018-01-02 --train-end 2024-06-30 --test-end 2026-05-15
  --embargo-days 30 --top-k 30 --max-epochs 80 --batch-size 8192
  --d-token 256 --n-blocks 6 --n-heads 8 --dates-per-step 1
  --cross-sectional-norm rank --label-norm
  --attention-dropout 0.25 --ffn-dropout 0.25 --weight-decay 0.001
  --early-stopping-patience 8 --learning-rate 0.0005
  --feature-policy judgment --require-gpu
)

fail=0
for HZ in short_5d mid_5d_30d long_30d_120d; do
  EXTRA=()
  [ "$HZ" = "long_30d_120d" ] && EXTRA=(--train-micro-batch 1024)
  echo "===== horizon $HZ start $(date -Is) ====="
  $PY -m quantagent.cli train-v8-deep --horizon-class "$HZ" \
      --output-dir "$ROOT/$HZ" "${COMMON[@]}" "${EXTRA[@]}" \
      || { echo "FATAL horizon $HZ failed"; fail=1; }
done

echo "===== blend $(date -Is) ====="
$PY -c "
import sys, pathlib
sys.path.insert(0, 'scripts')
from run_v8_deep_sweep import blend
blend(pathlib.Path('$ROOT'))
"
echo "===== v8.9 rankfix sweep done $(date -Is) fail=$fail ====="
exit "$fail"
