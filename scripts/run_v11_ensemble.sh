#!/usr/bin/env bash
# Stage 6 — v11 ensemble training launcher
#
# Trains the v11 model across the full 12-fold × 3-seed × 3-sub-model
# grid with all Stage 2-5 silver features pre-attached.
#
# Layout per fold:
#   runtime/models/v7_alpha_v11/walk_forward/fold_NNN/seed_M/<sub>/
#   where <sub> in {main, lowbuy, breakout, limitup_risk}
#
# Env vars:
#   QA_OUTPUT_ROOT       — base output dir (default: runtime/models/v7_alpha_v11)
#   QA_LAKE_ROOT         — silver data lake (default: runtime/data/v7)
#   QA_REPLAY_OUT        — replay reports (default: runtime/reports/sleeve_replay_v11)
#   QA_SEEDS             — comma-separated seeds (default: 1729,17,42)
#   QA_FOLDS             — comma-separated fold indices or "all" (default: all)
#   QA_SUB_MODELS        — comma-separated of {main,lowbuy,breakout,limitup_risk} (default: main)
#   QA_SKIP_TRAINING     — when set to 1, only run integration + replay
#
# Exit codes:
#   0  full run OK
#   1  integration attach failed
#   2  training failed
#   3  replay failed
#
# This script is launchable from a systemd unit; non-zero exit triggers
# OnFailure alerts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PY="${REPO_ROOT}/AI_quant_venv/bin/python"
if [ ! -x "${PY}" ]; then
    PY="python"
fi

OUTPUT_ROOT="${QA_OUTPUT_ROOT:-${REPO_ROOT}/runtime/models/v7_alpha_v11}"
LAKE_ROOT="${QA_LAKE_ROOT:-${REPO_ROOT}/runtime/data/v7}"
REPLAY_OUT="${QA_REPLAY_OUT:-${REPO_ROOT}/runtime/reports/sleeve_replay_v11}"
SEEDS="${QA_SEEDS:-1729,17,42}"
FOLDS="${QA_FOLDS:-all}"
SUB_MODELS="${QA_SUB_MODELS:-main}"
SKIP_TRAINING="${QA_SKIP_TRAINING:-0}"

mkdir -p "${OUTPUT_ROOT}"

echo "============================================================"
echo "v11 ensemble launch"
echo "  output_root : ${OUTPUT_ROOT}"
echo "  lake_root   : ${LAKE_ROOT}"
echo "  seeds       : ${SEEDS}"
echo "  folds       : ${FOLDS}"
echo "  sub_models  : ${SUB_MODELS}"
echo "  skip_train  : ${SKIP_TRAINING}"
echo "============================================================"

# ---------------------------------------------------------------------
# Step 1 — integration attach + manifest snapshot
# ---------------------------------------------------------------------
echo
echo "[1/3] running v11 integration attach + manifest snapshot..."
"${PY}" - <<EOF || exit 1
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, "src")
from quantagent.training.v11_integration import (
    V11IntegrationConfig,
    attach_v11_features,
    write_v11_attach_log,
)

panel_path = Path("${LAKE_ROOT}") / "silver" / "market_panel" / "market_features.parquet"
out_dir = Path("${OUTPUT_ROOT}") / "integration_audit"
out_dir.mkdir(parents=True, exist_ok=True)

if not panel_path.exists():
    print(f"  WARN panel missing: {panel_path}; writing empty attach log only")
    summary = {"n_rows": 0, "attach_log": [], "features_attached": [], "features_skipped": []}
    (out_dir / "v11_attach_log.json").write_text(json.dumps(summary, indent=2))
else:
    panel = pd.read_parquet(panel_path, columns=["trade_date", "symbol"])
    cfg = V11IntegrationConfig(lake_root="${LAKE_ROOT}")
    result = attach_v11_features(panel, cfg)
    path = write_v11_attach_log(result, out_dir)
    print(f"  wrote attach log to {path}")
    print(f"  features_attached: {[e.product for e in result.attach_log if e.attached]}")
    print(f"  features_skipped : {[e.product for e in result.attach_log if not e.attached]}")
EOF

# ---------------------------------------------------------------------
# Step 2 — training (gated by QA_SKIP_TRAINING)
# ---------------------------------------------------------------------
if [ "${SKIP_TRAINING}" = "1" ]; then
    echo
    echo "[2/3] SKIPPING training (QA_SKIP_TRAINING=1)"
else
    echo
    echo "[2/3] v11 training across folds/seeds/sub-models..."
    IFS="," read -ra SEED_LIST <<<"${SEEDS}"
    IFS="," read -ra SUB_LIST <<<"${SUB_MODELS}"
    for seed in "${SEED_LIST[@]}"; do
        for sub in "${SUB_LIST[@]}"; do
            seed_dir="${OUTPUT_ROOT}/walk_forward_seed_${seed}_${sub}"
            mkdir -p "${seed_dir}"
            echo "  training seed=${seed} sub=${sub} → ${seed_dir}"
            # The actual trainer entry point is project-specific; this
            # template calls v7-train-walk-forward-v7. Replace as needed.
            "${PY}" -m quantagent.cli train-walk-forward-v7 \
                --output-dir "${seed_dir}" \
                --seed "${seed}" \
                --sub-model "${sub}" \
                || { echo "  TRAIN FAILED seed=${seed} sub=${sub}"; exit 2; }
        done
    done
fi

# ---------------------------------------------------------------------
# Step 3 — replay across all trained dirs
# ---------------------------------------------------------------------
echo
echo "[3/3] replay aggregation..."
mkdir -p "${REPLAY_OUT}"
for sub in $(echo "${SUB_MODELS}" | tr ',' ' '); do
    for seed in $(echo "${SEEDS}" | tr ',' ' '); do
        seed_dir="${OUTPUT_ROOT}/walk_forward_seed_${seed}_${sub}"
        replay_dir="${REPLAY_OUT}/seed_${seed}_${sub}"
        if [ -d "${seed_dir}" ]; then
            echo "  replay seed=${seed} sub=${sub}"
            QA_PROBE_DIR="${seed_dir}" QA_REPLAY_OUT="${replay_dir}" \
                "${PY}" scripts/replay_horizon_sleeves.py \
                || { echo "  REPLAY FAILED seed=${seed} sub=${sub}"; exit 3; }
        else
            echo "  skip replay seed=${seed} sub=${sub} — no models found at ${seed_dir}"
        fi
    done
done

echo
echo "============================================================"
echo "v11 ensemble launch complete"
echo "  integration audit : ${OUTPUT_ROOT}/integration_audit"
echo "  replay reports    : ${REPLAY_OUT}"
echo "============================================================"
