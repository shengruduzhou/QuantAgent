#!/usr/bin/env bash
# Re-run ONLY the long_30d_120d sleeve (OOM'd at batch 8192 / 180 feature-tokens)
# with a small batch + expandable_segments, then re-blend all 3 sleeves.
# short_5d + mid_5d_30d are already trained in $ROOT. GPU-only.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="AI_quant_venv/bin/python3"
export QUANTAGENT_HORIZON_ASSIGNMENT="runtime/reports/v89_closed_loop/horizon_factor_assignment_plus7.json"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Long sleeve OOM root cause = 178 feature-tokens (attention is O(tokens^2)).
# Cap to the top-64 judgment factors by |ICIR| (drops redundant size proxies)
# so attention memory drops ~8x; only affects long (short/mid have <64 anyway).
export QUANTAGENT_JUDGMENT_MAX_FACTORS=64
ROOT="runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300"
echo "===== finish long sleeve $(date -Is) batch=2048 micro=128 max_factors=64 expandable_segments ====="
$PY -m quantagent.cli train-v8-deep --horizon-class long_30d_120d --output-dir "$ROOT/long_30d_120d" \
  --dataset-path runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean.parquet \
  --silver-panel-path runtime/data/v7/silver/market_panel/market_panel.parquet \
  --symbols-file runtime/data/v7/universe_v88_comma.txt \
  --train-start 2018-01-02 --train-end 2024-06-30 --test-end 2026-05-15 \
  --embargo-days 30 --top-k 30 --max-epochs 80 --batch-size 2048 \
  --d-token 256 --n-blocks 6 --n-heads 8 --dates-per-step 1 \
  --cross-sectional-norm rank --label-norm \
  --attention-dropout 0.25 --ffn-dropout 0.25 --weight-decay 0.001 \
  --early-stopping-patience 8 --learning-rate 0.0005 \
  --feature-policy judgment --require-gpu --train-micro-batch 128
LONG_RC=$?
echo "long sleeve rc=$LONG_RC $(date -Is)"
echo "===== blend (all 3 sleeves) $(date -Is) ====="
$PY -c "import sys, pathlib; sys.path.insert(0,'scripts'); from run_v8_deep_sweep import blend; blend(pathlib.Path('$ROOT'))"
echo "===== finish_long DONE $(date -Is) long_rc=$LONG_RC ====="
touch "$ROOT/_LONG_BLEND_COMPLETE_rc${LONG_RC}"
