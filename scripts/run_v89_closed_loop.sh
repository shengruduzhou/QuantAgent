#!/usr/bin/env bash
# v8.9 trustworthy closed loop: discover -> OOS-evaluate -> materialize ->
# GPU retrain -> RL eval -> strict tradable backtest.
#
# This wires the existing entrypoints into ONE reproducible chain with the
# trust guards added in this change set:
#   * factor discovery uses the tradability-aware IC gate (--exclude-st) and a
#     clean OOS cutoff (--train-end) so accepted factors have no phantom edge;
#   * factor generation uses the LLM closed loop + persistent memory;
#   * OOS evaluation re-checks survivors on the untouched post-cutoff window
#     (baseline_protocol-style variant-C honesty) via evaluate_discovered_factors;
#   * the RL eval carries the env-flat guard (book_dispersion_report) so a
#     gross/cash "win" on a flat book is not promoted;
#   * the final number is the trusted baseline_protocol variant-C excess.
#
# PREPARE-ONLY: run this yourself inside the GPU tmux session. It is heavy
# (LLM calls + a multi-hour 3-sleeve transformer retrain on the RTX 3090).
# Use --dry-run first to print the exact commands without executing.
#
# Usage:
#   scripts/run_v89_closed_loop.sh --dry-run
#   scripts/run_v89_closed_loop.sh                 # full run
#   scripts/run_v89_closed_loop.sh --from materialize   # resume from a step
#   scripts/run_v89_closed_loop.sh --only discover      # one step only
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY="AI_quant_venv/bin/python3"

# ---------------------------------------------------------------------------
# EDIT THESE to match your environment (defaults mirror the v8.9 rankfix sweep
# and rl_pit_train_eval.py). The base dataset is the one you retrain on; the
# materialize step writes a +synth copy next to it.
# ---------------------------------------------------------------------------
PANEL="runtime/data/v7/silver/market_panel/market_panel.parquet"
LABELS="runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v88_rankfix.parquet"
BASE_DATASET="runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v88_rankfix.parquet"
SYMBOLS_FILE="runtime/data/v7/universe_v88_comma.txt"
SECTOR="runtime/data/v7/silver/sector_map/sector_map.parquet"

TRAIN_END="2024-07-31"          # clean-OOS cutoff for discovery + evaluation
FACTOR_OOS_END="2025-08-31"     # survivor selection capped here (evaluate --oos-end)
HOLDOUT_START="2025-09-01"      # final-test window start: unseen by selection+model
RETRAIN_TRAIN_END="2024-06-30"  # transformer train window end (sweep parity)
TEST_END="2026-05-15"
DEVICE="cuda"
LLM_MODEL="${QUANTAGENT_LLM_MODEL:-}"   # empty = provider default
ROUNDS=6
CANDS_PER_ROUND=4

TS="$(date +%Y%m%d_%H%M)"
ROOT="runtime/reports/v89_closed_loop/${TS}"
DISC="$ROOT/discover"
EVAL="$ROOT/evaluate"
MATERIALIZED="runtime/data/v7/gold/training_dataset/training_dataset_v89_closed_loop_${TS}.parquet"
RETRAIN="$ROOT/retrain"
MEMORY="runtime/state/factor_loop_memory.jsonl"
BLEND_PRED="$RETRAIN/ensemble_composite.parquet"
RL_OUT="$ROOT/rl_pit"

DRY_RUN=0
ONLY=""
FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --only) ONLY="$2"; shift ;;
    --from) FROM="$2"; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
  shift
done

STEPS=(discover evaluate materialize retrain rl_eval backtest)
_started=0
want_step() {
  [ -n "$ONLY" ] && { [ "$1" = "$ONLY" ] && return 0 || return 1; }
  [ -z "$FROM" ] && return 0
  [ "$1" = "$FROM" ] && _started=1
  [ "$_started" = "1" ] && return 0 || return 1
}

run() {
  echo "+ $*"
  [ "$DRY_RUN" = "1" ] && return 0
  "$@"
}

banner() { echo; echo "===== $* ($(date -Is)) ====="; }

mkdir -p "$ROOT" "$DISC" "$EVAL" "$RETRAIN" "$RL_OUT" "$(dirname "$MEMORY")" runtime/logs/v89_closed_loop

# 1) DISCOVER ---------------------------------------------------------------
if want_step discover; then
  banner "1/6 discover factors (LLM closed loop, tradability-aware, clean OOS)"
  LLM_FLAG=(); [ -n "$LLM_MODEL" ] && LLM_FLAG=(--llm-model "$LLM_MODEL")
  run $PY -m quantagent.cli synthesize-factors-v7 \
    --market-panel "$PANEL" --labels "$LABELS" \
    --rd-agent --use-llm --allow-network \
    --label-column forward_return_5d \
    --rounds "$ROUNDS" --llm-candidates-per-round "$CANDS_PER_ROUND" \
    --train-end "$TRAIN_END" --exclude-st --min-validation-icir 0.2 \
    --memory-path "$MEMORY" \
    --output-dir "$DISC" "${LLM_FLAG[@]}"
fi

# 2) EVALUATE on the untouched post-cutoff window (variant-C honesty) --------
if want_step evaluate; then
  banner "2/6 evaluate survivors on clean OOS"
  run $PY scripts/evaluate_discovered_factors.py \
    --definitions "$DISC/synthesized_definitions.json" \
    --market-panel "$PANEL" --labels "$LABELS" \
    --train-end "$TRAIN_END" --oos-end "$FACTOR_OOS_END" \
    --output-dir "$EVAL"
  echo "  survivors -> $EVAL/accepted_definitions.json (selected on OOS<=$FACTOR_OOS_END; $HOLDOUT_START+ held out)"
fi

# 3) MATERIALIZE survivors into the training dataset ------------------------
# Adds the accepted DSL factors as columns. NOTE: keep the rest of the build
# flags identical to how BASE_DATASET was built so only the synth block changes.
if want_step materialize; then
  banner "3/6 materialize survivors into training dataset"
  run $PY -m quantagent.cli build-training-dataset-v7 \
    --market-panel "$PANEL" \
    --factor-library alpha181 \
    --synthesized-factors "$EVAL/accepted_definitions.json" \
    --output "$MATERIALIZED"
  echo "  materialized dataset -> $MATERIALIZED"
fi

# 4) RETRAIN v8.9 on GPU (3 horizon sleeves + blend) ------------------------
if want_step retrain; then
  banner "4/6 GPU retrain (short/mid/long sleeves + blend)"
  COMMON=(
    --dataset-path "$MATERIALIZED"
    --silver-panel-path "$PANEL"
    --symbols-file "$SYMBOLS_FILE"
    --train-start 2018-01-02 --train-end "$RETRAIN_TRAIN_END" --test-end "$TEST_END"
    --embargo-days 30 --top-k 30 --max-epochs 80 --batch-size 8192
    --d-token 256 --n-blocks 6 --n-heads 8 --dates-per-step 1
    --cross-sectional-norm rank --label-norm
    --attention-dropout 0.25 --ffn-dropout 0.25 --weight-decay 0.001
    --early-stopping-patience 8 --learning-rate 0.0005
    --feature-policy judgment --require-gpu
  )
  for HZ in short_5d mid_5d_30d long_30d_120d; do
    EXTRA=(); [ "$HZ" = "long_30d_120d" ] && EXTRA=(--train-micro-batch 1024)
    banner "  sleeve $HZ"
    run $PY -m quantagent.cli train-v8-deep --horizon-class "$HZ" \
      --output-dir "$RETRAIN/$HZ" "${COMMON[@]}" "${EXTRA[@]}"
  done
  banner "  blend sleeves -> $BLEND_PRED"
  run $PY -c "import sys, pathlib; sys.path.insert(0,'scripts'); from run_v8_deep_sweep import blend; blend(pathlib.Path('$RETRAIN'))"
fi

# 5) RL eval (env-flat guard + strict re-sim vs passive book) ---------------
if want_step rl_eval; then
  banner "5/6 RL PIT train+eval (env-flat guard, strict verdict)"
  run $PY scripts/rl_pit_train_eval.py \
    --predictions "$BLEND_PRED" --score-column composite_score \
    --train-end 2025-12-31 --test-start 2026-01-02 \
    --device "$DEVICE" --output-dir "$RL_OUT"
  echo "  verdict -> $RL_OUT/verdict.json (check env_dispersion.env_can_select)"
fi

# 6) Strict tradable backtest (THE trusted number: variant-C absolute CAGR) --
# Reports absolute CAGR/Calmar (goal = max absolute return, then Calmar). Judge
# on the HELD-OUT window (factor selection used --oos-end before it) so the
# number is contamination-free. Exports a UI-discoverable backtest artifact.
if want_step backtest; then
  banner "6/6 strict tradable backtest (baseline_protocol variant-C, held-out)"
  for K in 10 20; do
    run $PY scripts/baseline_protocol.py \
      --predictions "$BLEND_PRED" --score-column composite_score --top-k $K \
      --start "$HOLDOUT_START" --end "$TEST_END" \
      --variants C_flags_eligible_delay1 \
      --save-backtest-dir "$ROOT/realtest_holdout_top${K}" \
      --output "$ROOT/realtest_holdout_top${K}/baseline.json"
  done
  echo "  trusted absolute CAGR/Calmar -> $ROOT/realtest_holdout_top{10,20}/backtest/metrics.json (UI-visible)"
fi

banner "closed loop complete — artifacts under $ROOT"
