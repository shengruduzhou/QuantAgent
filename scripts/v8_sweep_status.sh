#!/usr/bin/env bash
# One-shot status snapshot for the active v8 sweep run.
# Prints session state, latest log lines, per-horizon metrics, ensemble status.
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shanhefu/QuantAgent}"
cd "$PROJECT_ROOT"

echo "==== tmux session ===="
tmux ls 2>/dev/null | grep qa_v8_sweep || echo "  (no qa_v8_sweep session)"

echo
echo "==== sweep processes ===="
pgrep -af 'run_v8_sweep|train-v8-pipeline' 2>/dev/null | head -3 || echo "  (none)"

LOG=$(ls -t "$PROJECT_ROOT/runtime/logs/v8_pipeline/v8_sweep_"*.log 2>/dev/null | head -1 || true)
echo
echo "==== latest log: $LOG ===="
if [[ -n "$LOG" && -f "$LOG" ]]; then
  echo "  bytes: $(wc -c < "$LOG"), lines: $(wc -l < "$LOG")"
  echo
  echo "---- non-warning tail (last 20) ----"
  grep -vE "FutureWarning|daily = merged" "$LOG" | tail -20
fi

RUN=$(ls -td "$PROJECT_ROOT/runtime/reports/v8/pipeline/v8_sweep_"* 2>/dev/null | head -1 || true)
echo
echo "==== artifacts root: $RUN ===="
if [[ -n "$RUN" && -d "$RUN" ]]; then
  find "$RUN" -maxdepth 3 -type f 2>/dev/null | sort | head -40
  echo
  for h in short_5d mid_5d_30d long_30d_120d; do
    f="$RUN/$h/backtest/metrics.json"
    if [[ -f "$f" ]]; then
      echo "---- $h backtest metrics ----"
      cat "$f"
      echo
    else
      echo "---- $h: backtest/metrics.json missing yet ----"
    fi
  done
  if [[ -f "$RUN/ensemble_summary.json" ]]; then
    echo "---- ensemble summary ----"
    cat "$RUN/ensemble_summary.json"
  fi
fi
