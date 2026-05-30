"""Full-universe training — v2 entry (regime-aware long-only top-K + rich output).

Replaces the previous full-universe script. What changed vs the previous run
(``runtime/models/v7_alpha_full_universe/`` directory layout is the same):

  1. **Strategy is long-only top-K** (A-share executable for retail). The old
     pipeline reported a dollar-neutral H-L portfolio which had zero net
     market exposure → great in bear, awful in bull. Long-only top-K rides
     beta in trends and adds selection alpha on top.
  2. **Strategy is horizon-sleeved**: short/swing/trend sleeves use separate
     horizon mixes and rebalance cadences. A stock can receive a small short
     sleeve allocation, then keep/add exposure only if 20d/60d/120d sleeves also
     agree.
  3. **Annualisation bug fixed** — H-day returns are divided by H before
     compounding 252×. The old run reported 2.8 billion % annualised because
     it compounded 5d returns 252 times.
  4. **Differentiable listwise loss** replaces the broken argsort rank loss
     (gradient was identically zero — the model was trained on Huber MSE
     only). The new loss directly maximises softmax-weighted top-K return,
     same objective as the executable backtest.
  5. **Risk-first controls** target Max DD <10 % through rank weights, per-name
     caps, volatility scaling, earlier regime de-risking, and a portfolio
     drawdown kill switch.
  6. **Rich output**: ``equity_curve.csv`` + ``trade_blotter.csv`` +
     ``monthly_returns.csv`` + ``summary.md`` (initial/ending capital,
     annualised return, Sharpe, max DD, monthly win rate, IR vs benchmark).

Walk-forward expanded to 12 folds × 60d valid = 720 OOS days (~2022-2025).

Output: ``runtime/models/v7_alpha_full_universe/`` (overwritten).
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent.training.v7_experiment import V7TrainingConfig, run_v7_training_experiment


DATASET = Path(os.environ.get(
    "QA_TRAINING_DATASET",
    "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full.parquet",
))
OUTPUT = Path(os.environ.get("QA_TRAINING_OUTPUT", "runtime/models/v7_alpha_full_universe"))


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def validate_feature_schema(feature_cols: list[str]) -> None:
    min_synth_features = env_int("QA_MIN_SYNTH_FEATURES", 50)
    synth_cols = [col for col in feature_cols if col.startswith("synth_")]
    if min_synth_features > 0 and len(synth_cols) < min_synth_features:
        raise RuntimeError(
            "full-universe dataset has only "
            f"{len(synth_cols)} synth_* features; expected at least {min_synth_features}. "
            "Rebuild the training dataset with symbolic GA definitions before training, "
            "or set QA_MIN_SYNTH_FEATURES=0 for an explicit no-GA diagnostic run."
        )


def _label_outlier_caps(label_cols: list[str]) -> dict[str, float]:
    defaults = {
        "forward_return_1d": 0.25,
        "forward_return_5d": 0.80,
        "forward_return_20d": 1.50,
        "forward_return_60d": 3.00,
        "forward_return_120d": 5.00,
        "forward_return_126d": 5.00,
    }
    scale = env_float("QA_LABEL_OUTLIER_SCALE", 1.0)
    return {col: defaults.get(col, 3.0) * scale for col in label_cols}


def select_columns(parquet_path: Path) -> tuple[list[str], list[str], list[str]]:
    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    entity = ["symbol", "trade_date", "available_at"]
    labels: list[str] = []
    features: list[str] = []
    for f in schema:
        name = f.name
        type_str = str(f.type)
        if "label_end_" in name:
            continue
        if name in entity:
            continue
        if name.startswith("forward_return_"):
            labels.append(name)
            continue
        if type_str in {"double", "float", "int64", "int32", "bool"}:
            features.append(name)
    return entity, labels, features


def main() -> None:
    t0 = time.time()
    print(f"[{time.time()-t0:5.1f}s] inspecting parquet schema", flush=True)
    entity_cols, label_cols, feature_cols = select_columns(DATASET)
    validate_feature_schema(feature_cols)
    print(f"  entity={len(entity_cols)} labels={len(label_cols)} features={len(feature_cols)}", flush=True)
    print(f"  labels: {label_cols}", flush=True)

    print(f"[{time.time()-t0:5.1f}s] loading parquet (float32 cast)", flush=True)
    all_cols = entity_cols + label_cols + feature_cols
    df = pd.read_parquet(DATASET, columns=all_cols)
    print(f"  loaded rows={len(df):,} cols={len(df.columns)} "
          f"mem={df.memory_usage(deep=True).sum()/1e9:.2f} GB", flush=True)

    print(f"[{time.time()-t0:5.1f}s] downcasting numeric to float32", flush=True)
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    print(f"  after downcast mem={df.memory_usage(deep=True).sum()/1e9:.2f} GB", flush=True)

    drop_cols = [c for c in feature_cols if df[c].isna().all()]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        feature_cols = [c for c in feature_cols if c not in drop_cols]
        print(f"[{time.time()-t0:5.1f}s] dropped {len(drop_cols)} all-NaN feature cols", flush=True)
        print(f"  surviving features: {len(feature_cols)}", flush=True)

    nan_counts = df[feature_cols].isna().sum().sort_values(ascending=False)
    n_with_nan = int((nan_counts > 0).sum())
    if n_with_nan:
        print(f"[{time.time()-t0:5.1f}s] imputing NaN→0 in {n_with_nan} feature cols", flush=True)
        df[feature_cols] = df[feature_cols].fillna(0.0)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0.0)

    before = len(df)
    df = df.dropna(subset=label_cols, how="all").reset_index(drop=True)
    print(f"[{time.time()-t0:5.1f}s] kept {len(df):,} of {before:,} rows after label-NaN filter",
          flush=True)
    caps = _label_outlier_caps(label_cols)
    mask = pd.Series(True, index=df.index)
    outlier_counts: dict[str, int] = {}
    for col, cap in caps.items():
        if col not in df.columns:
            continue
        bad = df[col].abs() > float(cap)
        outlier_counts[col] = int(bad.sum())
        mask &= ~bad.fillna(False)
    if not mask.all():
        before = len(df)
        df = df.loc[mask].reset_index(drop=True)
        print(
            f"[{time.time()-t0:5.1f}s] dropped {before - len(df):,} rows with impossible label outliers "
            f"{outlier_counts}",
            flush=True,
        )
    gc.collect()

    config = V7TrainingConfig(
        model="ft_transformer",
        # 1d horizon dropped: IC was 0.014 (essentially noise) and it never
        # participates in the deployed sleeves. Removing it frees model
        # capacity for the horizons we actually use. forward_return_1d label
        # is still loaded for daily P&L (kept in dataset columns).
        horizons=(5, 20, 60, 120, 126),
        min_train_rows=10_000,
        # ----- walk-forward: 12 folds × 60d valid = 720d OOS (covers 2022-2025) -----
        n_splits=env_int("QA_N_SPLITS", 12),
        split_mode="rolling",
        valid_size_days=env_int("QA_VALID_SIZE_DAYS", 60),
        min_train_days=env_int("QA_MIN_TRAIN_DAYS", 504),
        rolling_train_days=env_int("QA_ROLLING_TRAIN_DAYS", 1008),
        embargo_days=5,
        purge_days=126,
        cost_bps=12.0,
        output_dir=str(OUTPUT),
        feature_columns=tuple(feature_cols),
        # ----- FT-Transformer -----
        ft_max_epochs=env_int("QA_FT_MAX_EPOCHS", 60),
        ft_learning_rate=env_float("QA_FT_LEARNING_RATE", 3e-4),
        ft_batch_size=env_int("QA_FT_BATCH_SIZE", 8192),
        ft_dates_per_step=env_int("QA_FT_DATES_PER_STEP", 2),
        ft_train_micro_batch=env_int("QA_FT_TRAIN_MICRO_BATCH", 500),
        ft_d_token=env_int("QA_FT_D_TOKEN", 128),
        ft_n_blocks=env_int("QA_FT_N_BLOCKS", 5),
        ft_n_heads=env_int("QA_FT_N_HEADS", 8),
        ft_attention_dropout=env_float("QA_FT_ATTENTION_DROPOUT", 0.10),
        ft_ffn_dropout=env_float("QA_FT_FFN_DROPOUT", 0.10),
        ft_weight_decay=env_float("QA_FT_WEIGHT_DECAY", 1e-4),
        ft_use_amp=env_bool("QA_FT_USE_AMP", True),
        ft_device="cuda",
        require_gpu=True,
        ft_seed=env_int("QA_FT_SEED", 1729),  # ensemble: vary across runs
        skip_final_fit=env_bool("QA_SKIP_FINAL_FIT", True),  # default skip to avoid post-WF OOM
        # ----- executable backtest: horizon sleeves + risk-first gates + rich output -----
        primary_horizon=20,
        top_k=50,
        weighting="rank",
        softmax_temp=0.5,
        initial_capital=1_000_000.0,
        benchmark_label="csi300",
        benchmark_path="runtime/data/v7/raw/akshare/index/equity_index.parquet",
        executable_strategy="horizon_sleeves",
        executable_base_gross=1.00,
        executable_max_weight_per_name=0.035,  # 3-sleeve mix → overlap risk goes up; tighter cap.
        executable_max_turnover=0.30,
        executable_vol_target_annual=0.18,
        target_max_drawdown=0.10,
        drawdown_soft_limit=0.055,
        drawdown_hard_limit=0.070,
        drawdown_kill_limit=0.080,
        regime_gate_enabled=True,
        regime_ret_window=20,
        regime_ret_threshold=-0.05,
        regime_ma_window=200,
        regime_caution_exposure=0.95,
        regime_crisis_exposure=0.10,
        regime_low_exposure=0.30,
        regime_high_exposure=1.00,
        risk_free_rate_annual=0.02,
        # ----- diagnostics -----
        emit_ic_decay_diagnostics=True,
        run_synth_ablation=False,
        experiment_name="alpha181_full_universe_2018_2026_v2_regime_aware",
        registry_root="runtime/models/registry",
    )

    print(f"[{time.time()-t0:5.1f}s] starting training (model=ft_transformer, cuda)", flush=True)
    print(f"  walk-forward: rolling {config.rolling_train_days}d train, "
          f"{config.valid_size_days}d valid × {config.n_splits} splits = "
          f"~{config.valid_size_days * config.n_splits}d OOS", flush=True)
    print(f"  strategy: horizon sleeves ({config.weighting}), "
          f"maxDD target <{config.target_max_drawdown:.0%}, regime/DD/vol gates ON, init capital "
          f"{config.initial_capital:,.0f} RMB", flush=True)

    result = run_v7_training_experiment(df, config)
    print(f"\n[{time.time()-t0:5.1f}s] DONE: status={result.status}", flush=True)
    m = result.metrics
    print("\nheadline metrics:", flush=True)
    for k in ("rank_ic_mean", "ICIR", "hit_rate", "evaluated_days",
              "fold_count", "prediction_rows", "feature_count", "data_range"):
        print(f"  {k:32s} = {m.get(k)}", flush=True)
    exec_block = m.get("executable_backtest") or {}
    if exec_block:
        print("\nexecutable backtest (horizon sleeves + risk gates):", flush=True)
        for k in ("initial_capital", "ending_capital", "total_return_pct",
                  "annualised_return_pct", "annualised_vol_pct", "sharpe",
                  "max_drawdown_pct", "max_drawdown_target_passed", "benchmark_annualised_pct",
                  "excess_annualised_pct", "information_ratio",
                  "hit_vs_benchmark_pct", "monthly_win_months",
                  "monthly_total_months", "n_rebalance_dates",
                  "oos_start", "oos_end"):
            v = exec_block.get(k)
            if isinstance(v, float):
                print(f"  {k:32s} = {v:,.4f}", flush=True)
            else:
                print(f"  {k:32s} = {v}", flush=True)
        for k in ("equity_curve_path", "trade_blotter_path",
                  "monthly_returns_path", "summary_md_path"):
            v = exec_block.get(k)
            if v:
                print(f"  → {v}", flush=True)


if __name__ == "__main__":
    main()
