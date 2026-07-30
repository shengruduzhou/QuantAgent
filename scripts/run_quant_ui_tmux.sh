#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
session="${QUANT_UI_TMUX_SESSION:-quantagent}"
runtime="${QUANTAGENT_HOME:-${repo_root}/runtime}"
log_dir="${runtime}/logs"
log_path="${log_dir}/quant_ui.log"

if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is required. Install it, then rerun this launcher.\n' >&2
  exit 1
fi

if tmux has-session -t "$session" 2>/dev/null; then
  printf 'QuantAgent is already running in tmux session %s.\n' "$session"
  printf 'Attach: tmux attach -t %s\n' "$session"
  printf 'Log:    %s\n' "$log_path"
  exit 0
fi

mkdir -p "$log_dir"
printf -v launch_command '%q ' "${repo_root}/scripts/run_quant_ui.sh" "$@"
printf -v quoted_log '%q' "$log_path"
printf -v quoted_repo '%q' "$repo_root"

tmux new-session -d -s "$session" -n workstation \
  "cd ${quoted_repo} && ${launch_command} 2>&1 | tee -a ${quoted_log}"
tmux split-window -t "${session}:workstation" -v -p 32 \
  "tail -n 200 -F ${quoted_log}"

if command -v nvidia-smi >/dev/null 2>&1; then
  tmux split-window -t "${session}:workstation" -h -p 42 \
    "watch -n 0.1 nvidia-smi"
else
  tmux split-window -t "${session}:workstation" -h -p 42 \
    "printf 'GPU monitor unavailable: nvidia-smi not found.\\n'; exec bash"
fi

tmux select-layout -t "${session}:workstation" tiled
printf 'QuantAgent tmux session started: %s\n' "$session"
printf 'Attach: tmux attach -t %s\n' "$session"
printf 'Log:    %s\n' "$log_path"
printf 'Panes:  server · live log · GPU watch (0.1s)\n'
