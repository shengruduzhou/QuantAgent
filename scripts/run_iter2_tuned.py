"""Iter2: re-train on cached alpha181 dataset with higher dropout + weight_decay
+ longer training to compress max_drawdown below the 25% gate.

Reuses runtime/data/v7/gold/training_dataset/training_dataset_alpha181_500.parquet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent.training.v7_experiment import V7TrainingConfig, run_v7_training_experiment


def main() -> None:
    dataset_path = Path("runtime/data/v7/gold/training_dataset/training_dataset_alpha181_500.parquet")
    output_dir = Path("runtime/models/v7_alpha_iter2")

    print(f"loading {dataset_path}", flush=True)
    df = pd.read_parquet(dataset_path)
    print(f"  {len(df):,} rows, {len(df.columns)} cols", flush=True)

    config = V7TrainingConfig(
        model="ft_transformer",
        horizons=(1, 5, 20, 60, 120, 126),
        min_train_rows=5000,
        n_splits=4,
        split_mode="rolling",
        valid_size_days=20,
        min_train_days=504,
        rolling_train_days=756,
        embargo_days=5,
        purge_days=126,
        output_dir=str(output_dir),
        # ----- tuned knobs (vs iter1 defaults) -----
        ft_max_epochs=120,                # 80 → 120 (more training + early stopping by val_loss)
        ft_batch_size=8192,
        ft_d_token=128,
        ft_n_blocks=5,
        ft_n_heads=8,
        ft_attention_dropout=0.20,        # 0.10 → 0.20 (more regularisation)
        ft_ffn_dropout=0.20,              # 0.10 → 0.20
        ft_weight_decay=1e-3,             # 1e-4 → 1e-3 (stronger L2)
        ft_use_amp=True,
        ft_device="cuda",
        require_gpu=True,
        run_synth_ablation=False,
        emit_ic_decay_diagnostics=True,
        cost_bps=12.0,
        experiment_name="alpha181_iter2_500syms_tuned",
        registry_root="runtime/models/registry",
    )

    print("starting training...", flush=True)
    result = run_v7_training_experiment(df, config)
    print(f"\n=== DONE: status={result.status} ===", flush=True)
    print(json.dumps({
        "rank_ic_mean": result.metrics.get("rank_ic_mean"),
        "ICIR": result.metrics.get("ICIR"),
        "max_drawdown": result.metrics.get("max_drawdown"),
        "turnover_adjusted_net_return": result.metrics.get("turnover_adjusted_net_return"),
        "hit_rate": result.metrics.get("hit_rate"),
        "adverse_regime_passed": result.metrics.get("adverse_regime_passed"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
