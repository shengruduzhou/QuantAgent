#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-qa_v7_full_ai}"
MISSION_NAME="${MISSION_NAME:-mission_full_ai_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shanhefu/QuantAgent}"
QUANTAGENT_HOME="${QUANTAGENT_HOME:-$PROJECT_ROOT/runtime}"
PROVIDER_URI="${PROVIDER_URI:-$QUANTAGENT_HOME/data/raw/qlib/cn_data}"
AS_OF_DATE="${AS_OF_DATE:-2026-05-15}"
START_DATE="${START_DATE:-2020-01-01}"
END_DATE="${END_DATE:-2026-05-15}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/AI_quant_venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/AI_quant_venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_SYMBOLS="${MAX_SYMBOLS:-0}"
FT_MAX_EPOCHS="${FT_MAX_EPOCHS:-60}"
FT_BATCH_SIZE="${FT_BATCH_SIZE:-8192}"
N_TRIALS="${N_TRIALS:-100}"
GENERATIONS="${GENERATIONS:-30}"
RL_TIMESTEPS="${RL_TIMESTEPS:-5000000}"
SELECTION_MODE="${SELECTION_MODE:-ai_threshold}"
ALPHA_THRESHOLD="${ALPHA_THRESHOLD:-0.0}"
CONFIDENCE_FLOOR="${CONFIDENCE_FLOOR:-0.55}"
SELECTION_TOP_K_MIN="${SELECTION_TOP_K_MIN:-5}"
SELECTION_TOP_K_MAX="${SELECTION_TOP_K_MAX:-100}"
MIN_ORDER_VALUE_YUAN="${MIN_ORDER_VALUE_YUAN:-100}"

REFRESH_AKSHARE_MARKET="${REFRESH_AKSHARE_MARKET:-0}"
REFRESH_FUNDAMENTALS="${REFRESH_FUNDAMENTALS:-0}"
REFRESH_VALUATION="${REFRESH_VALUATION:-0}"
REFRESH_SECTOR_MAP="${REFRESH_SECTOR_MAP:-0}"
ALLOW_NETWORK="${ALLOW_NETWORK:-0}"

mkdir -p "$QUANTAGENT_HOME/logs"
LOG_PATH="$QUANTAGENT_HOME/logs/${MISSION_NAME}.log"

cd "$PROJECT_ROOT"

CMD=(
  "$PYTHON_BIN" -m quantagent.cli
  run-full-ai-quant-v7
  --symbols auto
  --provider-uri "$PROVIDER_URI"
  --max-symbols "$MAX_SYMBOLS"
  --start-date "$START_DATE"
  --end-date "$END_DATE"
  --as-of-date "$AS_OF_DATE"
  --model ft_transformer
  --ft-device cuda
  --require-gpu
  --ft-max-epochs "$FT_MAX_EPOCHS"
  --ft-batch-size "$FT_BATCH_SIZE"
  --horizons 1,5,20,60,120,126
  --primary-horizon 5
  --split-mode rolling
  --purge-days 126
  --embargo-days 5
  --valid-size-days 20
  --min-train-days 756
  --rolling-train-days 1260
  --min-rows 1000
  --min-train-rows 1000
  --min-symbols 50
  --min-dates 252
  --top-k 30
  --top-k-ratio 0.10
  --min-selection-pressure 3.0
  --selection-mode "$SELECTION_MODE"
  --alpha-threshold "$ALPHA_THRESHOLD"
  --confidence-floor "$CONFIDENCE_FLOOR"
  --selection-top-k-min "$SELECTION_TOP_K_MIN"
  --selection-top-k-max "$SELECTION_TOP_K_MAX"
  --max-weight 0.10
  --max-sector 0.30
  --max-turnover 0.40
  --initial-cash 1000000
  --min-order-value-yuan "$MIN_ORDER_VALUE_YUAN"
  --dynamic-top-k
  --timing-gate
  --holding-period-mode soft
  --capital-tier 1000000:0.10,10000000:0.05,100000000:0.02
  --run-autopilot-search
  --n-trials "$N_TRIALS"
  --generations "$GENERATIONS"
  --rl-timesteps "$RL_TIMESTEPS"
)

if [[ "$ALLOW_NETWORK" == "1" ]]; then
  CMD+=(--allow-network)
fi
if [[ "$REFRESH_AKSHARE_MARKET" == "1" ]]; then
  CMD+=(--refresh-akshare-market)
fi
if [[ "$REFRESH_FUNDAMENTALS" == "1" ]]; then
  CMD+=(--refresh-fundamentals)
fi
if [[ "$REFRESH_VALUATION" == "1" ]]; then
  CMD+=(--refresh-valuation)
fi
if [[ "$REFRESH_SECTOR_MAP" == "1" ]]; then
  CMD+=(--refresh-sector-map)
fi

printf 'Launching %s in tmux session %s\n' "$MISSION_NAME" "$SESSION_NAME"
printf 'Log: %s\n' "$LOG_PATH"
printf 'Command:' > "$LOG_PATH"
printf ' %q' "${CMD[@]}" >> "$LOG_PATH"
printf '\n\n' >> "$LOG_PATH"
printf -v CMD_STRING '%q ' "${CMD[@]}"

tmux new-session -d -s "$SESSION_NAME" -n "$MISSION_NAME" \
  "cd '$PROJECT_ROOT' && export QUANTAGENT_HOME='$QUANTAGENT_HOME' CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_NVML_BASED_CUDA_CHECK=1 PYTHONPATH='$PROJECT_ROOT/src' && nvidia-smi && $CMD_STRING 2>&1 | tee -a '$LOG_PATH'"

echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Watch log: tail -f $LOG_PATH"
