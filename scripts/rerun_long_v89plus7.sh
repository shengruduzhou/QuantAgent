#!/usr/bin/env bash
# Redo ONLY the long_30d_120d sleeve that OOM'd, then re-blend (short+mid already OK).
# OOM fix: batch 8192->4096, micro-batch 512, expandable_segments. GPU-only.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="AI_quant_venv/bin/python3"
export QUANTAGENT_HORIZON_ASSIGNMENT="runtime/reports/v89_closed_loop/horizon_factor_assignment_plus7.json"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT="$(cat runtime/reports/v89_closed_loop/LATEST_RETRAIN_ROOT.txt)"
echo "===== redo long sleeve $(date -Is) root=$ROOT ====="

$PY -m quantagent.cli train-v8-deep --horizon-class long_30d_120d --output-dir "$ROOT/long_30d_120d" \
  --dataset-path runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean.parquet \
  --silver-panel-path runtime/data/v7/silver/market_panel/market_panel.parquet \
  --symbols-file runtime/data/v7/universe_v88_comma.txt \
  --train-start 2018-01-02 --train-end 2024-06-30 --test-end 2026-05-15 \
  --embargo-days 30 --top-k 30 --max-epochs 80 --batch-size 4096 \
  --d-token 256 --n-blocks 6 --n-heads 8 --dates-per-step 1 \
  --cross-sectional-norm rank --label-norm \
  --attention-dropout 0.25 --ffn-dropout 0.25 --weight-decay 0.001 \
  --early-stopping-patience 8 --learning-rate 0.0005 \
  --feature-policy judgment --require-gpu --train-micro-batch 512
rc=$?
if [ $rc -ne 0 ]; then echo "FATAL long redo failed rc=$rc"; touch "$ROOT/_LONG_REDONE_fail"; exit $rc; fi

echo "===== re-blend (all 3 sleeves) $(date -Is) ====="
$PY -c "import sys, pathlib; sys.path.insert(0,'scripts'); from run_v8_deep_sweep import blend; blend(pathlib.Path('$ROOT'))"
touch "$ROOT/_LONG_REDONE_ok"
echo "===== long redo + reblend DONE $(date -Is) ====="
