#!/usr/bin/env bash
# v8.9 + 7 clean LLM factors GPU retrain (3 sleeves + blend).
# Detached, GPU-only. Launch with:
#   nohup setsid bash scripts/run_v89_plus7_retrain.sh > runtime/logs/v89_closed_loop/retrain_plus7_detached.log 2>&1 &
# Monitor: tail -f runtime/logs/v89_closed_loop/retrain_plus7_detached.log
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="AI_quant_venv/bin/python3"
export QUANTAGENT_HORIZON_ASSIGNMENT="runtime/reports/v89_closed_loop/horizon_factor_assignment_plus7.json"
# Reduce CUDA fragmentation (the long sleeve sits near the 24GB ceiling).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS="$(date +%Y%m%d_%H%M)"
ROOT="runtime/reports/v89_closed_loop/retrain_plus7_${TS}"
mkdir -p "$ROOT" runtime/logs/v89_closed_loop
echo "$ROOT" > runtime/reports/v89_closed_loop/LATEST_RETRAIN_ROOT.txt
echo "===== v8.9+7 retrain start $(date -Is) root=$ROOT ====="

COMMON=(
  --dataset-path runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean.parquet
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
  EXTRA=(); [ "$HZ" = "long_30d_120d" ] && EXTRA=(--train-micro-batch 1024)
  echo "===== sleeve $HZ start $(date -Is) ====="
  $PY -m quantagent.cli train-v8-deep --horizon-class "$HZ" --output-dir "$ROOT/$HZ" "${COMMON[@]}" "${EXTRA[@]}" \
    || { echo "FATAL sleeve $HZ failed"; fail=1; }
done

echo "===== blend $(date -Is) ====="
$PY -c "import sys, pathlib; sys.path.insert(0,'scripts'); from run_v8_deep_sweep import blend; blend(pathlib.Path('$ROOT'))" || fail=1
echo "===== v8.9+7 retrain DONE $(date -Is) fail=$fail root=$ROOT ====="
touch "$ROOT/_RETRAIN_COMPLETE_fail${fail}"
