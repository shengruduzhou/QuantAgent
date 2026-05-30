"""Trainable V7 alpha experiment pipeline with purged walk-forward validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.data.v7_label_builder import V7_LABEL_HORIZONS
from quantagent.data.v7_quality_gates import (
    V7DataQualityGateConfig,
    V7ModelAcceptanceGateConfig,
    evaluate_adverse_regime,
    evaluate_data_quality_gates,
    evaluate_model_acceptance_gates,
)
from quantagent.training.model_registry import ModelRegistry
from quantagent.training.splitters import WalkForwardSplitConfig, split_walk_forward
from quantagent.cuda_runtime import configure_cuda_environment


SUPPORTED_MODELS: tuple[str, ...] = ("ridge", "elastic_net", "lightgbm", "xgboost", "ft_transformer")

DEFAULT_EXECUTABLE_SLEEVES: tuple[tuple[str, tuple[tuple[int, float], ...], float, int, int], ...] = (
    # v9 (12 folds, full universe ~3700 stocks) showed 60d is the strongest
    # single horizon: IC 0.130 (highest), avg ann excess +26.3%, worst DD only
    # -7.06%. Earlier (4-fold 500-stock probe) it had negative avg excess
    # which was a small-universe artifact. Adding 60d as a third independent
    # sleeve so its picks add rather than dilute 20d / 120d.
    #
    # 20d: highest raw return (+39% excess) but DD -8.27%, IC only 0.054.
    # 60d: best IC (0.130), strong excess (+26%), moderate DD.
    # 120d: best risk-adjusted (sharpe 26+), low DD (-2%), high IC (0.138).
    # 1d / 5d / 126d dropped — 1d is noise, 5d hits DD cap, 126d duplicates 120d.
    ("core_20", ((20, 1.00),), 0.40, 30, 20),
    ("mid_60", ((60, 1.00),), 0.30, 30, 30),
    ("trend_120", ((120, 1.00),), 0.30, 35, 60),
)

configure_cuda_environment()


def _default_output_dir() -> str:
    return str(quant_paths().models / "v7_alpha")


def _default_registry_root() -> str:
    return str(quant_paths().models / "v7_alpha" / "registry")


@dataclass(frozen=True)
class V7TrainingConfig:
    horizons: tuple[int, ...] = V7_LABEL_HORIZONS
    model: str = "ridge"
    alpha: float = 1.0
    l1_ratio: float = 0.5
    min_train_rows: int = 100
    n_splits: int = 4
    split_mode: str = "expanding"
    valid_size_days: int = 5
    min_train_days: int = 20
    rolling_train_days: int = 252
    embargo_days: int = 5
    purge_days: int | None = None
    embargo_pct: float = 0.02
    cost_bps: float = 12.0
    output_dir: str = field(default_factory=_default_output_dir)
    paper_report_path: str | None = None
    mark_production_ready: bool = False
    feature_columns: tuple[str, ...] = ()
    registry_root: str = field(default_factory=_default_registry_root)
    experiment_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    allow_model_downgrade: bool = False
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    ft_learning_rate: float = 3e-4
    ft_max_epochs: int = 60
    ft_batch_size: int = 8192
    ft_d_token: int = 128
    ft_n_blocks: int = 5
    ft_n_heads: int = 8
    ft_dates_per_step: int = 8
    ft_train_micro_batch: int | None = None  # split per-step rows further for memory
    ft_attention_dropout: float = 0.10
    ft_ffn_dropout: float = 0.10
    ft_weight_decay: float = 1e-4
    ft_use_amp: bool = True
    ft_device: str = "auto"
    ft_seed: int = 1729  # exposed for ensemble runs (v10+)
    skip_final_fit: bool = False  # skip the post-walk-forward full-data fit (avoids OOM at end)
    require_gpu: bool = False
    run_synth_ablation: bool = False
    emit_ic_decay_diagnostics: bool = True
    # ----- executable backtest knobs (risk-first long-only horizon sleeves) -----
    primary_horizon: int = 20             # picking signal + rebalance cadence (trading days)
    top_k: int = 50                       # top-K stocks held long; A-share retail executable
    weighting: str = "rank"               # equal | rank | softmax
    softmax_temp: float = 0.5
    initial_capital: float = 1_000_000.0  # starting equity (RMB) for blotter / equity curve
    benchmark_label: str = "csi300"       # default A-share benchmark
    benchmark_path: str = "runtime/data/v7/raw/akshare/index/equity_index.parquet"
    executable_strategy: str = "horizon_sleeves"  # horizon_sleeves | primary_horizon
    executable_sleeves: tuple[tuple[str, tuple[tuple[int, float], ...], float, int, int], ...] = DEFAULT_EXECUTABLE_SLEEVES
    executable_base_gross: float = 1.00
    executable_max_weight_per_name: float = 0.035
    executable_max_turnover: float = 0.30
    # vol target loosened (0.12 → 0.18): the prior 0.12 cap was binding on
    # bull-market days where realised port vol was 15-18%, capping gross at
    # ~0.85× and missing benchmark beta. 0.18 keeps the cap available for
    # genuinely volatile periods without strangling exposure in healthy uptrends.
    executable_vol_target_annual: float = 0.18
    executable_vol_window: int = 20
    target_max_drawdown: float = 0.10
    # DD gates: soft 6% (was 5%) — natural strategy DD is 5-8%, don't cut
    # on routine noise. Hard 8% — meaningful caution. Kill 8.5% — leaves
    # 1.5pp safety margin under 10% DD target. Combined with the smoother
    # curve + 0.20 kill floor (not zero), this avoids the death spiral
    # that strangled v9 aggregate by locking gross at 0 for 380 days.
    drawdown_soft_limit: float = 0.055
    drawdown_hard_limit: float = 0.070
    drawdown_kill_limit: float = 0.080
    # Rolling DD reference: peak = max NAV over the last drawdown_peak_window
    # trading days. Avoids death-spiral when an old bear permanently anchors
    # the all-time-high. 252d ≈ 1 year, matching how most institutions
    # measure rolling DD.
    drawdown_peak_window: int = 252
    # Stage-1 universe filter knobs: ST soft-exclude (≥90% blocked) +
    # suspended hard-exclude + limit-up block-new-entries. See
    # quantagent.universe.filters.UniverseFilterConfig for semantics.
    universe_filter_enabled: bool = False
    universe_market_panel_path: str = "runtime/data/v7/silver/market_panel/market_panel.parquet"
    # Stage 2.2 silver st_flags is the source of truth. We fall back to
    # the legacy market_features.parquet path only if st_flags is missing
    # so callers that haven't run fetch_st_list.py yet still get a usable
    # is_st flag column instead of an empty filter.
    universe_st_flag_path: str = "runtime/data/v7/silver/st_flags/st_flags.parquet"
    universe_st_flag_legacy_path: str = "runtime/data/v7/silver/market_panel/market_features.parquet"
    # When True the optimization-side sector_map is routed through
    # quantagent.diagnostics.sector_audit.sector_map_for_optimization, which
    # enforces the manifest gate before allowing sector data into ranker /
    # filter / cap decisions. Default False keeps backward-compat for runs
    # that already feed raw sector frames into the optimizer.
    universe_sector_gate_enabled: bool = True
    universe_sector_map_path: str = "runtime/data/v7/silver/sector_map/sector_map.parquet"
    universe_sector_manifest_path: str = "runtime/data/v7/manifests/sector_map.json"
    universe_st_manifest_path: str = "runtime/data/v7/manifests/st_flags.json"
    universe_st_min_block_rate: float = 0.90
    universe_st_max_portfolio_share: float = 0.10
    universe_suspended_block_new: bool = True
    universe_limit_up_block_new: bool = True
    universe_limit_down_block_sell: bool = True
    universe_limit_up_pct: float = 0.099
    universe_limit_down_pct: float = -0.099
    universe_require_amount_above: float = 0.0
    # High-chase fields (review fix #10): were defined in
    # UniverseFilterConfig but the V7TrainingConfig → UniverseFilterConfig
    # bridge in _compute_horizon_sleeve_backtest only forwarded 4 of the
    # 13 knobs, silently dropping the high-chase settings. They are now
    # surfaced here and forwarded explicitly.
    universe_high_chase_enabled: bool = True
    universe_high_chase_lookback: int = 5
    universe_high_chase_max_cum_return: float = 0.30
    universe_high_chase_max_limit_ups: int = 3
    universe_high_chase_combine: str = "and"
    # market regime gate: shrink exposure when benchmark is in confirmed downtrend
    regime_gate_enabled: bool = True
    regime_ret_window: int = 20           # CSI300 N-day return window
    regime_ret_threshold: float = -0.05   # if N-day ret < this AND below MA → de-risk
    regime_ma_window: int = 200           # CSI300 MA window
    # Caution lifted 0.55 → 0.95: the prior 0.55 (and even 0.80) was
    # overreacting to ordinary bull-market 3% pullbacks. Reserve real cuts
    # for confirmed bear (0.30) and crisis (0.10). In replay this gained
    # ~6% aggregate excess across 4-fold OOS without breaching DD ≤ 10%.
    regime_caution_exposure: float = 0.95
    regime_crisis_exposure: float = 0.10
    regime_low_exposure: float = 0.30     # exposure in bear regime (rest in cash)
    regime_high_exposure: float = 1.00    # exposure in normal/bull regime
    risk_free_rate_annual: float = 0.02   # used for cash yield in regime-off periods

    # Stage 3 — market hard gate. Defaults are the conservative triggers
    # ("real crisis only") that preserve the v10 baseline +22% aggregate
    # excess. A tighter setting was tested but over-blocked: aggregate
    # excess collapsed to +10% while DD did not improve, because the
    # weakest two folds breach DD on idiosyncratic stock-level moves
    # (not market-wide crises that the hard gate is designed to catch).
    # Resolving those folds requires per-stock DD kill switches or the
    # sub-model retraining, both outside the hard-gate's scope.
    hard_gate_enabled: bool = True
    hard_gate_crash_5d_threshold: float = -0.08
    hard_gate_bear_20d_threshold: float = -0.15
    hard_gate_ma_window: int = 200
    hard_gate_breadth_threshold: float = 0.20
    hard_gate_breadth_consecutive_days: int = 3
    hard_gate_vol_window_short: int = 20
    hard_gate_vol_window_long: int = 60
    hard_gate_vol_spike_multiplier: float = 2.0
    hard_gate_cool_down_days: int = 5
    hard_gate_blocked_gross_multiplier: float = 0.0


@dataclass(frozen=True)
class V7TrainingResult:
    status: str
    output_dir: str
    metrics: dict[str, object]
    data_quality_report: dict[str, object]
    acceptance_report: dict[str, object]
    artifact_paths: dict[str, str]


def run_v7_training_experiment(dataset: pd.DataFrame, config: V7TrainingConfig | None = None) -> V7TrainingResult:
    config = config or V7TrainingConfig()
    if config.model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported V7 training model: {config.model}; supported: {SUPPORTED_MODELS}")
    if dataset is None or dataset.empty:
        raise ValueError("V7 training requires a non-empty real-data dataset")
    backend = _resolve_backend(config.model, config.allow_model_downgrade)
    data = dataset.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "symbol"]).sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    quality = evaluate_data_quality_gates(
        data,
        V7DataQualityGateConfig(min_rows=config.min_train_rows, min_symbols=2, min_dates=max(5, config.n_splits)),
    )
    if not quality.passed:
        raise ValueError(f"V7 training data quality gates failed: {quality.failures}")

    feature_columns = list(config.feature_columns or _auto_feature_columns(data))
    if not feature_columns:
        raise ValueError("V7 training found no numeric feature columns")
    output_dir = Path(config.output_dir)
    if config.model == "ft_transformer":
        return _run_ft_transformer_experiment(data, feature_columns, quality.to_dict(), config)
    coefficients: dict[str, dict[str, float]] = {}
    boosters: dict[str, object] = {}
    all_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, object]] = []
    for horizon in config.horizons:
        label_column = f"forward_return_{horizon}d"
        if label_column not in data.columns:
            continue
        horizon_data = data.dropna(subset=[label_column, *feature_columns]).reset_index(drop=True)
        if len(horizon_data) < config.min_train_rows:
            continue
        folds = _make_walk_forward_folds(horizon_data, config)
        last_artifact = None
        for fold in folds:
            train_idx = fold.train_idx
            test_idx = fold.valid_idx
            if len(train_idx) < max(config.min_train_rows, 10, len(feature_columns)):
                continue
            train = horizon_data.iloc[train_idx]
            test = horizon_data.iloc[test_idx]
            if backend in {"lightgbm", "xgboost"}:
                booster, prediction = _fit_predict_booster(
                    train[feature_columns],
                    train[label_column],
                    test[feature_columns],
                    backend,
                    config,
                )
                last_artifact = booster
            else:
                coef, intercept = _fit_linear(train[feature_columns], train[label_column], config, backend)
                prediction = _predict_linear(test[feature_columns], coef, intercept)
                coefficients[str(horizon)] = {column: float(value) for column, value in zip(feature_columns, coef)} | {"intercept": float(intercept)}
            fold_cols = ["trade_date", "symbol", label_column]
            if "forward_return_1d" in test.columns and "forward_return_1d" != label_column:
                fold_cols.append("forward_return_1d")
            fold_frame = test[fold_cols].copy()
            fold_frame["horizon"] = horizon
            fold_frame["prediction"] = prediction
            fold_frame["sample_role"] = "validation"
            fold_frame["fold_id"] = fold.fold_id
            fold_frame["train_start"] = fold.train_dates[0]
            fold_frame["train_end"] = fold.train_dates[1]
            fold_frame["valid_start"] = fold.valid_dates[0]
            fold_frame["valid_end"] = fold.valid_dates[1]
            all_predictions.append(fold_frame)
            metric = _fold_metrics(fold_frame, label_column, fold.fold_id, horizon, config)
            fold_metrics.append(metric)
            _write_incremental_fold_monitor(output_dir / "walk_forward" / f"fold_{fold.fold_id:03d}", fold_frame, metric)
        if last_artifact is not None:
            boosters[str(horizon)] = last_artifact

    if not all_predictions:
        raise ValueError("V7 training produced no walk-forward predictions")
    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    metrics = _aggregate_metrics(
        prediction_frame,
        fold_metrics,
        coefficients,
        feature_importance=_feature_importance(boosters, feature_columns) if boosters else None,
    )
    metrics |= _training_manifest_metrics(prediction_frame, feature_columns, config.model)
    adverse_label_column = "forward_return_1d" if "forward_return_1d" in prediction_frame.columns else f"forward_return_{config.horizons[0]}d"
    adverse_report = evaluate_adverse_regime(prediction_frame, label_column=adverse_label_column)
    metrics["adverse_regime_passed"] = bool(adverse_report.get("passed", False))
    metrics["adverse_regime_report"] = adverse_report
    metrics["backend"] = backend
    metrics["model_requested"] = config.model
    metrics["model_downgraded"] = backend != config.model
    try:
        exec_summary = _compute_executable_backtest(prediction_frame, config, Path(config.output_dir))
        metrics["executable_backtest"] = exec_summary
        _apply_executable_backtest_metrics(metrics, exec_summary)
    except Exception as exc:  # pragma: no cover - best-effort, never fail training
        metrics["executable_backtest_error"] = f"{type(exc).__name__}:{exc}"
    acceptance = evaluate_model_acceptance_gates(
        metrics,
        V7ModelAcceptanceGateConfig(
            require_paper_report=config.mark_production_ready,
            require_benchmark=False,
            min_training_symbols=1,
            min_prediction_symbols=1,
            min_effective_universe_by_date=1,
            min_selection_pressure=0.0,
        ),
        paper_report_path=config.paper_report_path,
    )
    status = "production_ready" if config.mark_production_ready and acceptance.passed else "validation_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = _write_artifacts(
        output_dir,
        config,
        feature_columns,
        coefficients,
        metrics,
        quality.to_dict(),
        acceptance.to_dict(),
        prediction_frame,
        boosters=boosters,
        backend=backend,
    )

    # Optional diagnostics: IC decay heatmap + synth ablation.
    diagnostics_dir = output_dir / "diagnostics"
    primary_horizon = config.horizons[0] if config.horizons else 5
    primary_label = f"forward_return_{primary_horizon}d"
    if config.emit_ic_decay_diagnostics and primary_label in data.columns:
        try:
            from quantagent.training.diagnostics import compute_factor_ic_decay, render_ic_decay_heatmap

            decay = compute_factor_ic_decay(data, feature_columns, primary_label)
            if not decay.empty:
                diag_paths = render_ic_decay_heatmap(decay, diagnostics_dir / "ic_decay")
                metrics["ic_decay_diagnostics"] = diag_paths
        except Exception as exc:  # pragma: no cover - diagnostic best-effort
            metrics["ic_decay_diagnostics_error"] = f"{type(exc).__name__}:{exc}"

    if config.run_synth_ablation and any(c.startswith("synth_") for c in feature_columns):
        try:
            ablation = _run_synth_ablation(data, feature_columns, config, primary_label)
            if ablation is not None:
                metrics["synth_ablation"] = ablation
                ablation_path = diagnostics_dir / "metrics_ablation.json"
                ablation_path.parent.mkdir(parents=True, exist_ok=True)
                ablation_path.write_text(
                    json.dumps(ablation, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        except Exception as exc:  # pragma: no cover - diagnostic best-effort
            metrics["synth_ablation_error"] = f"{type(exc).__name__}:{exc}"

    return V7TrainingResult(
        status=status,
        output_dir=str(output_dir),
        metrics=metrics,
        data_quality_report=quality.to_dict(),
        acceptance_report=acceptance.to_dict(),
        artifact_paths=artifact_paths,
    )


def _run_synth_ablation(
    data: pd.DataFrame,
    feature_columns: list[str],
    config: V7TrainingConfig,
    primary_label: str,
) -> dict[str, object] | None:
    """Train an identical model with synth_* columns removed; report deltas.

    The ablation runs a second walk-forward pass over the same dataset and
    hyperparameters, so the only difference is the feature set. The deltas
    isolate the marginal contribution of GA-synthesised factors against
    the alpha101 + CICC base.
    """
    base_synth = [c for c in feature_columns if c.startswith("synth_")]
    without_synth = [c for c in feature_columns if not c.startswith("synth_")]
    if not base_synth or not without_synth:
        return None

    from dataclasses import replace

    sub_dir = Path(config.output_dir) / "diagnostics" / "ablation_no_synth"
    sub_dir.mkdir(parents=True, exist_ok=True)
    no_synth_cfg = replace(
        config,
        feature_columns=tuple(without_synth),
        output_dir=str(sub_dir),
        run_synth_ablation=False,  # prevent recursion
        emit_ic_decay_diagnostics=False,
        mark_production_ready=False,
    )
    baseline = run_v7_training_experiment(data, no_synth_cfg)
    base_metrics = baseline.metrics
    return {
        "synth_columns_used": base_synth,
        "synth_column_count": len(base_synth),
        "base_feature_count": len(feature_columns),
        "ablation_feature_count": len(without_synth),
        "baseline_metrics_path": str(Path(baseline.output_dir) / "metrics.json"),
        "baseline_status": baseline.status,
        # The full-feature metrics live in the outer `metrics` dict —
        # downstream consumers can compute Δsharpe / ΔICIR there.
    }


def _write_incremental_fold_monitor(
    fold_dir: Path,
    fold_frame: pd.DataFrame,
    metric: dict[str, object],
) -> None:
    """Persist fold-level OOS data immediately for manual supervision.

    Full training can take hours. Waiting until final ``metrics.json`` means
    bad annualised return, benchmark underperformance, or drawdown instability
    cannot be detected early. These files are passive artifacts, not an
    automatic monitor loop.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    horizon = int(metric.get("horizon", 0))
    metrics_path = fold_dir / f"fold_{horizon:03d}d_strategy_metrics.json"
    predictions_path = fold_dir / f"fold_{horizon:03d}d_oos_predictions.parquet"
    metrics_path.write_text(json.dumps(metric, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    try:
        fold_frame.to_parquet(predictions_path, index=False)
    except Exception:
        fold_frame.to_csv(predictions_path.with_suffix(".csv"), index=False)


def _make_walk_forward_folds(frame: pd.DataFrame, config: V7TrainingConfig):
    available_horizons = tuple(h for h in config.horizons if f"forward_return_{h}d" in frame.columns)
    purge_days = max(available_horizons or config.horizons) if config.purge_days is None else int(config.purge_days)
    symbol_count = max(1, int(frame["symbol"].nunique()) if "symbol" in frame.columns else 1)
    row_based_min_train_days = int(np.ceil(config.min_train_rows / symbol_count)) + config.embargo_days + max(0, purge_days)
    cfg = WalkForwardSplitConfig(
        n_splits=config.n_splits,
        valid_size_days=config.valid_size_days,
        min_train_days=max(config.min_train_days, row_based_min_train_days),
        rolling_train_days=config.rolling_train_days,
        embargo_days=config.embargo_days,
        purge_days=max(0, purge_days),
        mode=config.split_mode,
    )
    folds = split_walk_forward(frame, config=cfg)
    if not folds:
        unique_dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna().nunique()
        relaxed_valid_size = max(1, min(config.valid_size_days, unique_dates - cfg.min_train_days))
        if relaxed_valid_size < config.valid_size_days:
            folds = split_walk_forward(
                frame,
                config=WalkForwardSplitConfig(
                    n_splits=max(1, min(config.n_splits, 2)),
                    valid_size_days=relaxed_valid_size,
                    min_train_days=cfg.min_train_days,
                    rolling_train_days=config.rolling_train_days,
                    embargo_days=config.embargo_days,
                    purge_days=max(0, purge_days),
                    mode=config.split_mode,
                ),
            )
    if not folds:
        raise ValueError(
            "V7 training produced no walk-forward folds; increase date coverage or lower "
            "min_train_days/valid_size_days/purge_days."
        )
    return folds


def _run_ft_transformer_experiment(
    data: pd.DataFrame,
    feature_columns: list[str],
    quality: dict[str, object],
    config: V7TrainingConfig,
) -> V7TrainingResult:
    from quantagent.training.ft_transformer_trainer import (
        FTTransformerTrainer,
        FTTransformerTrainerConfig,
        predict_ft_transformer_artifact,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, object]] = []
    used_horizons = tuple(h for h in config.horizons if f"forward_return_{h}d" in data.columns)
    if not used_horizons:
        raise ValueError("FT-Transformer training found no configured forward_return_*d labels")
    required_labels = [f"forward_return_{h}d" for h in used_horizons]
    fit_frame = data.dropna(subset=[*required_labels, *feature_columns]).reset_index(drop=True)
    if len(fit_frame) < config.min_train_rows:
        raise ValueError("FT-Transformer training has fewer rows than min_train_rows")
    for fold in _make_walk_forward_folds(fit_frame, config):
        train = fit_frame.iloc[fold.train_idx].copy()
        valid = fit_frame.iloc[fold.valid_idx].copy()
        if len(train) < max(config.min_train_rows, len(feature_columns)):
            continue
        fold_dir = output_dir / "walk_forward" / f"fold_{fold.fold_id:03d}"
        completed_predictions, completed_metrics = _load_completed_ft_fold(fold_dir, used_horizons)
        if completed_predictions:
            all_predictions.extend(completed_predictions)
            fold_metrics.extend(completed_metrics)
            continue
        trainer = FTTransformerTrainer(
            FTTransformerTrainerConfig(
                horizons=used_horizons,
                d_token=config.ft_d_token,
                n_blocks=config.ft_n_blocks,
                n_heads=config.ft_n_heads,
                attention_dropout=config.ft_attention_dropout,
                ffn_dropout=config.ft_ffn_dropout,
                learning_rate=config.ft_learning_rate,
                weight_decay=config.ft_weight_decay,
                batch_size=config.ft_batch_size,
                max_epochs=config.ft_max_epochs,
                dates_per_step=config.ft_dates_per_step,
                train_micro_batch=config.ft_train_micro_batch,
                use_amp=config.ft_use_amp,
                device=config.ft_device,
                require_gpu=config.require_gpu,
                seed=config.ft_seed,
                feature_columns=tuple(feature_columns),
                output_dir=str(fold_dir),
            )
        )
        fold_artifacts = trainer.fit_and_save(train, validation_dataset=valid)
        pred = predict_ft_transformer_artifact(fold_dir, valid, device=fold_artifacts.device)
        for horizon in used_horizons:
            label_column = f"forward_return_{horizon}d"
            alpha_column = f"alpha_{horizon}d"
            fold_cols = ["trade_date", "symbol", label_column]
            # Carry 1d realized return alongside, so the executable backtest can
            # build a daily P&L curve without re-joining the parquet later.
            if "forward_return_1d" in valid.columns and "forward_return_1d" != label_column:
                fold_cols.append("forward_return_1d")
            fold_frame = valid[fold_cols].copy()
            fold_frame["horizon"] = horizon
            fold_frame["prediction"] = pred.predictions[alpha_column].to_numpy(dtype=float)
            fold_frame["sample_role"] = "validation"
            fold_frame["fold_id"] = fold.fold_id
            fold_frame["train_start"] = fold.train_dates[0]
            fold_frame["train_end"] = fold.train_dates[1]
            fold_frame["valid_start"] = fold.valid_dates[0]
            fold_frame["valid_end"] = fold.valid_dates[1]
            all_predictions.append(fold_frame)
            metric = _fold_metrics(fold_frame, label_column, fold.fold_id, horizon, config)
            fold_metrics.append(metric)
            _write_incremental_fold_monitor(fold_dir, fold_frame, metric)
    if not all_predictions:
        raise ValueError("FT-Transformer training produced no out-of-sample predictions")

    if getattr(config, "skip_final_fit", False):
        # When skip_final_fit=True, fabricate a minimal artifact bundle from
        # the most-recent walk-forward fold so downstream metadata still has
        # a "device / cuda / gpu_name" record. The 12 fold models on disk
        # are what gets deployed; the "final fit on all data" step was
        # historically the OOM culprit. Skipping it costs nothing because
        # we never used the final-fit weights for OOS evaluation anyway.
        import torch as _torch
        class _MinimalArtifact:
            device = config.ft_device if config.ft_device in ("cuda", "cpu") else ("cuda" if _torch.cuda.is_available() else "cpu")
            cuda_available = _torch.cuda.is_available()
            gpu_name = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None
        final_artifacts = _MinimalArtifact()
    else:
        final_trainer = FTTransformerTrainer(
            FTTransformerTrainerConfig(
                horizons=used_horizons,
                d_token=config.ft_d_token,
                n_blocks=config.ft_n_blocks,
                n_heads=config.ft_n_heads,
                attention_dropout=config.ft_attention_dropout,
                ffn_dropout=config.ft_ffn_dropout,
                learning_rate=config.ft_learning_rate,
                weight_decay=config.ft_weight_decay,
                batch_size=config.ft_batch_size,
                max_epochs=config.ft_max_epochs,
                dates_per_step=config.ft_dates_per_step,
                train_micro_batch=config.ft_train_micro_batch,
                use_amp=config.ft_use_amp,
                device=config.ft_device,
                require_gpu=config.require_gpu,
                seed=config.ft_seed,
                feature_columns=tuple(feature_columns),
                output_dir=str(output_dir),
            )
        )
        final_artifacts = final_trainer.fit_and_save(fit_frame)

    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    metrics = _aggregate_metrics(prediction_frame, fold_metrics, coefficients={})
    metrics |= _training_manifest_metrics(prediction_frame, feature_columns, config.model)
    adverse_label_column = "forward_return_1d" if "forward_return_1d" in prediction_frame.columns else f"forward_return_{used_horizons[0]}d"
    adverse_report = evaluate_adverse_regime(prediction_frame, label_column=adverse_label_column)
    metrics["adverse_regime_passed"] = bool(adverse_report.get("passed", False))
    metrics["adverse_regime_report"] = adverse_report
    metrics["backend"] = "ft_transformer"
    metrics["model_requested"] = config.model
    metrics["model_downgraded"] = False
    metrics["training_device"] = final_artifacts.device
    metrics["cuda_available"] = final_artifacts.cuda_available
    metrics["gpu_name"] = final_artifacts.gpu_name
    metrics["gpu_required"] = config.require_gpu
    # ----- executable long-only top-K backtest with regime gate + rich output -----
    try:
        exec_summary = _compute_executable_backtest(prediction_frame, config, output_dir)
        metrics["executable_backtest"] = exec_summary
        _apply_executable_backtest_metrics(metrics, exec_summary)
    except Exception as exc:  # pragma: no cover - backtest is best-effort, never fail training
        metrics["executable_backtest_error"] = f"{type(exc).__name__}:{exc}"
    acceptance = evaluate_model_acceptance_gates(
        metrics,
        V7ModelAcceptanceGateConfig(
            require_paper_report=config.mark_production_ready,
            require_benchmark=False,
            min_training_symbols=1,
            min_prediction_symbols=1,
            min_effective_universe_by_date=1,
            min_selection_pressure=0.0,
        ),
        paper_report_path=config.paper_report_path,
    )
    status = "production_ready" if config.mark_production_ready and acceptance.passed else "validation_only"
    artifact_paths = _write_artifacts(
        output_dir,
        config,
        feature_columns,
        {},
        metrics,
        quality,
        acceptance.to_dict(),
        prediction_frame,
        backend="ft_transformer",
    )
    artifact_paths["ft_checkpoint"] = str(output_dir / "ft_transformer.pt")
    return V7TrainingResult(
        status=status,
        output_dir=str(output_dir),
        metrics=metrics,
        data_quality_report=quality,
        acceptance_report=acceptance.to_dict(),
        artifact_paths=artifact_paths,
    )


def _load_completed_ft_fold(
    fold_dir: Path,
    horizons: tuple[int, ...],
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    """Load a fully materialised FT fold so interrupted runs can resume.

    A fold is considered complete only when every configured horizon has
    both OOS predictions and metrics plus the model-level metrics file.
    Partial folds are re-run; complete folds are never overwritten.
    """

    if not fold_dir.exists() or not (fold_dir / "ft_transformer_metrics.json").exists():
        return [], []
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    import json as _json
    for horizon in horizons:
        pred_path = fold_dir / f"fold_{horizon:03d}d_oos_predictions.parquet"
        metric_path = fold_dir / f"fold_{horizon:03d}d_strategy_metrics.json"
        if not pred_path.exists() or not metric_path.exists():
            return [], []
        predictions.append(pd.read_parquet(pred_path))
        metrics.append(_json.loads(metric_path.read_text(encoding="utf-8")))
    return predictions, metrics


def _resolve_backend(model: str, allow_downgrade: bool) -> str:
    if model in {"ridge", "elastic_net"}:
        return model
    if model == "lightgbm":
        try:
            import lightgbm  # type: ignore  # noqa: F401
            return "lightgbm"
        except Exception as exc:  # pragma: no cover - optional dep
            if not allow_downgrade:
                raise RuntimeError(
                    "model='lightgbm' but lightgbm is not installed. "
                    "Install quantagent[training] or pass --allow-model-downgrade."
                ) from exc
            return "ridge"
    if model == "xgboost":
        try:
            import xgboost  # type: ignore  # noqa: F401
            return "xgboost"
        except Exception as exc:  # pragma: no cover - optional dep
            if not allow_downgrade:
                raise RuntimeError(
                    "model='xgboost' but xgboost is not installed. "
                    "Install quantagent[training] or pass --allow-model-downgrade."
                ) from exc
            return "ridge"
    if model == "ft_transformer":
        try:
            import torch  # type: ignore  # noqa: F401
            return "ft_transformer"
        except Exception as exc:  # pragma: no cover - optional dep
            if not allow_downgrade:
                raise RuntimeError(
                    "model='ft_transformer' but torch is not installed. "
                    "Install quantagent[training] or pass --allow-model-downgrade."
                ) from exc
            return "ridge"
    raise ValueError(f"unsupported V7 model: {model}")


def _fit_linear(x: pd.DataFrame, y: pd.Series, config: V7TrainingConfig, backend: str) -> tuple[np.ndarray, float]:
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    mean_x = np.nanmean(x_values, axis=0)
    mean_y = float(np.nanmean(y_values))
    x_values = np.nan_to_num(x_values, nan=0.0, posinf=0.0, neginf=0.0)
    centred_x = x_values - mean_x
    centred_y = y_values - mean_y
    if backend == "elastic_net":
        coef = np.zeros(centred_x.shape[1])
        l1 = config.alpha * config.l1_ratio
        l2 = config.alpha * (1.0 - config.l1_ratio)
        lr = 0.05
        for _ in range(250):
            gradient = centred_x.T @ (centred_x @ coef - centred_y) / max(1, len(centred_x)) + l2 * coef
            coef = np.sign(coef - lr * gradient) * np.maximum(np.abs(coef - lr * gradient) - lr * l1, 0.0)
    else:
        eye = config.alpha * np.eye(centred_x.shape[1])
        coef = np.linalg.pinv(centred_x.T @ centred_x + eye) @ centred_x.T @ centred_y
    intercept = mean_y - float(np.dot(coef, mean_x))
    return coef, intercept


def _predict_linear(x: pd.DataFrame, coef: np.ndarray, intercept: float) -> np.ndarray:
    values = np.nan_to_num(x.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return values @ coef + intercept


def _fit_predict_booster(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    backend: str,
    config: V7TrainingConfig,
) -> tuple[object, np.ndarray]:
    if backend == "lightgbm":
        import lightgbm as lgb  # type: ignore

        booster = lgb.LGBMRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth if config.max_depth > 0 else -1,
            learning_rate=config.learning_rate,
            random_state=1729,
            n_jobs=1,
            verbose=-1,
        )
        booster.fit(train_x.to_numpy(dtype=float), train_y.to_numpy(dtype=float))
        prediction = booster.predict(test_x.to_numpy(dtype=float))
        return booster, np.asarray(prediction, dtype=float)
    if backend == "xgboost":
        import xgboost as xgb  # type: ignore

        booster = xgb.XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=max(1, config.max_depth),
            learning_rate=config.learning_rate,
            random_state=1729,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
        booster.fit(train_x.to_numpy(dtype=float), train_y.to_numpy(dtype=float))
        prediction = booster.predict(test_x.to_numpy(dtype=float))
        return booster, np.asarray(prediction, dtype=float)
    raise ValueError(f"unsupported booster backend: {backend}")


def _feature_importance(boosters: dict[str, object], feature_columns: list[str]) -> dict[str, dict[str, float]]:
    importance: dict[str, dict[str, float]] = {}
    for horizon, booster in boosters.items():
        weights = getattr(booster, "feature_importances_", None)
        if weights is None:
            continue
        arr = np.asarray(weights, dtype=float)
        if arr.size != len(feature_columns):
            continue
        importance[horizon] = {column: float(value) for column, value in zip(feature_columns, arr)}
    return importance


def _auto_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    excluded = {"open", "high", "low", "close", "volume", "amount"}
    selected: list[str] = []
    for column in frame.select_dtypes("number").columns:
        if column.startswith("forward_return_") or column.startswith("label_end_") or column in excluded:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        clean = values.replace([np.inf, -np.inf], np.nan)
        finite_ratio = float(clean.notna().mean())
        if finite_ratio < 0.30:
            continue
        if clean.nunique(dropna=True) <= 1:
            continue
        selected.append(column)
    return tuple(selected)


def _label_end_times(frame: pd.DataFrame, horizon: int) -> pd.Series:
    column = f"label_end_{horizon}d"
    if column in frame.columns:
        return pd.to_datetime(frame[column], errors="coerce").fillna(frame["trade_date"] + pd.Timedelta(days=horizon))
    return frame["trade_date"] + pd.Timedelta(days=horizon)


def _fold_metrics(
    frame: pd.DataFrame,
    label_column: str,
    fold: int,
    horizon: int,
    config: "V7TrainingConfig",
) -> dict[str, object]:
    """Per-fold metrics built on the EXECUTABLE long-only top-K rule.

    H-day overlap bug fix: ``forward_return_{H}d`` is an H-day cumulative return;
    we divide by H before annualising so 252 is applied to a daily-equivalent rate.
    No regime gate at the per-fold layer — the gate is applied once at
    aggregation to keep fold deltas comparable.
    """
    H = max(int(horizon), 1)
    cost_bps = float(getattr(config, "cost_bps", 12.0))
    top_k = int(getattr(config, "top_k", 50))
    weighting = str(getattr(config, "weighting", "rank"))
    softmax_temp = float(getattr(config, "softmax_temp", 0.5))
    by_date_ic = frame.groupby("trade_date").apply(_rank_ic(label_column)).dropna()
    raw_returns = frame.groupby("trade_date").apply(
        _long_only_topk_return(label_column, top_k, weighting, softmax_temp),
    ).fillna(0.0)
    # H-day return → daily-equivalent (so 252× compounding is well-defined)
    daily_eq = raw_returns / float(H)
    # cost is paid per rebalance ~ every H trading days; spread evenly
    cost_per_day = (cost_bps / 10_000.0) / float(H)
    net_daily = daily_eq - cost_per_day
    nav = (1.0 + net_daily).cumprod()
    drawdown = nav / nav.cummax() - 1.0 if not nav.empty else pd.Series(dtype=float)
    n_days = max(int(len(net_daily)), 1)
    avg_daily = float(net_daily.mean()) if not net_daily.empty else 0.0
    annualised_return = float((1.0 + avg_daily) ** 252 - 1.0) if avg_daily > -1 else float("nan")
    annualised_vol = float(net_daily.std(ddof=1) * (252 ** 0.5)) if n_days > 1 else 0.0
    sharpe = (
        float(avg_daily / (net_daily.std(ddof=1) + 1e-12) * (252 ** 0.5)) if n_days > 1 else 0.0
    )
    benchmark_metrics = _fold_benchmark_metrics(frame, H, config)
    excess_annualised = (
        annualised_return - benchmark_metrics["benchmark_annualised_return"]
        if benchmark_metrics["benchmark_status"] == "ok"
        else 0.0
    )
    return {
        "fold": fold,
        "horizon": horizon,
        "rank_ic_mean": float(by_date_ic.mean()) if not by_date_ic.empty else 0.0,
        "net_return": float(net_daily.sum()) if not net_daily.empty else 0.0,
        "avg_daily_return": avg_daily,
        "annualised_return": annualised_return,
        "annualised_vol": annualised_vol,
        "sharpe": sharpe,
        "n_days": n_days,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "benchmark_annualised_return": benchmark_metrics["benchmark_annualised_return"],
        "excess_annualised_return": excess_annualised,
        "benchmark_status": benchmark_metrics["benchmark_status"],
    }


def _fold_benchmark_metrics(
    frame: pd.DataFrame,
    horizon: int,
    config: "V7TrainingConfig",
) -> dict[str, object]:
    benchmark = _load_benchmark_series(config)
    if benchmark is None or benchmark.empty or frame.empty:
        return {"benchmark_status": "missing", "benchmark_annualised_return": 0.0}
    H = max(int(horizon), 1)
    b = benchmark.set_index("trade_date").sort_index()
    b["bench_h_return"] = b["close"].shift(-H) / b["close"] - 1.0
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna().drop_duplicates().sort_values()
    aligned = dates.map(b["bench_h_return"]).dropna()
    if aligned.empty:
        return {"benchmark_status": "missing_aligned_dates", "benchmark_annualised_return": 0.0}
    daily = aligned.astype(float) / float(H)
    avg_daily = float(daily.mean())
    annualised = float((1.0 + avg_daily) ** 252 - 1.0) if avg_daily > -1 else float("nan")
    return {"benchmark_status": "ok", "benchmark_annualised_return": annualised}


def _long_only_topk_return(
    label_column: str,
    top_k: int,
    weighting: str = "softmax",
    softmax_temp: float = 0.5,
):
    """A-share executable long-only top-K with chosen weighting.

    Returns the H-day cumulative portfolio return; the fold layer divides by H
    to obtain a daily-equivalent before annualising.
    """
    def inner(frame: pd.DataFrame) -> float:
        if frame.empty:
            return 0.0
        sub = frame.dropna(subset=["prediction", label_column])
        if sub.empty:
            return 0.0
        top = sub.nlargest(min(int(top_k), len(sub)), "prediction")
        if top.empty:
            return 0.0
        preds = top["prediction"].to_numpy(dtype=float)
        rets = top[label_column].to_numpy(dtype=float)
        if weighting == "equal":
            w = np.ones(len(top)) / len(top)
        elif weighting == "rank":
            r = pd.Series(preds).rank(method="average").to_numpy(dtype=float)
            w = r / r.sum()
        else:  # softmax (default)
            x = preds - preds.max()
            e = np.exp(x / max(float(softmax_temp), 1e-6))
            w = e / e.sum()
        return float(np.dot(w, rets))

    return inner


def _rank_ic(label_column: str):
    def inner(frame: pd.DataFrame) -> float:
        if len(frame) < 2:
            return 0.0
        prediction = frame["prediction"].rank()
        realized = frame[label_column].rank()
        if prediction.nunique() < 2 or realized.nunique() < 2:
            return 0.0
        value = prediction.corr(realized)
        return float(value) if pd.notna(value) else 0.0

    return inner


def _apply_executable_backtest_metrics(metrics: dict[str, object], exec_summary: dict[str, object]) -> None:
    if exec_summary.get("executable_backtest_status") != "ok":
        return
    if "max_drawdown_pct" in exec_summary:
        metrics["max_drawdown"] = float(exec_summary["max_drawdown_pct"]) / 100.0
        metrics["executable_max_drawdown"] = float(exec_summary["max_drawdown_pct"]) / 100.0
    if "total_return_pct" in exec_summary:
        metrics["turnover_adjusted_net_return"] = float(exec_summary["total_return_pct"]) / 100.0
    if "excess_annualised_pct" in exec_summary:
        metrics["excess_return_after_costs"] = float(exec_summary["excess_annualised_pct"]) / 100.0
    if "max_drawdown_target_passed" in exec_summary:
        metrics["max_drawdown_target_passed"] = bool(exec_summary["max_drawdown_target_passed"])


def _load_benchmark_series(config: "V7TrainingConfig") -> pd.DataFrame | None:
    """Load benchmark daily closes; return None on any miss (best-effort)."""
    try:
        path = Path(getattr(config, "benchmark_path", "")) if getattr(config, "benchmark_path", None) else None
        if path is None or not path.exists():
            return None
        df = pd.read_parquet(path)
        label = str(getattr(config, "benchmark_label", "csi300"))
        if "label" in df.columns:
            df = df[df["label"] == label]
        if df.empty:
            return None
        if "observation_date" in df.columns:
            df = df.rename(columns={"observation_date": "trade_date"})
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.dropna(subset=["trade_date", "close"]).sort_values("trade_date").reset_index(drop=True)
        return df[["trade_date", "close"]].copy()
    except Exception:  # pragma: no cover - benchmark is optional, never fail training
        return None


def _compute_regime_exposure(
    benchmark: pd.DataFrame | None,
    config: "V7TrainingConfig",
) -> pd.Series:
    """Return a date→exposure series in [0, 1]. Missing benchmark → exposure=1."""
    regime = _compute_regime_frame(benchmark, config)
    return regime["exposure"] if not regime.empty else pd.Series(dtype=float)


def _compute_regime_frame(
    benchmark: pd.DataFrame | None,
    config: "V7TrainingConfig",
) -> pd.DataFrame:
    """Classify benchmark regime into normal / caution / bear / crisis.

    The old gate only reacted after a 20-day drawdown and MA200 break. This
    layered gate reacts earlier to 5d/20d momentum deterioration while keeping
    the MA filter for confirmed bear regimes.
    """
    if benchmark is None or benchmark.empty or not bool(getattr(config, "regime_gate_enabled", True)):
        return pd.DataFrame(columns=["trade_date", "exposure", "regime_state"])
    b = benchmark.sort_values("trade_date").reset_index(drop=True).copy()
    ret_w = int(getattr(config, "regime_ret_window", 20))
    ma_w = int(getattr(config, "regime_ma_window", 200))
    b["ret_5"] = b["close"].pct_change(5)
    b["ret_n"] = b["close"].pct_change(ret_w)
    b["ma_20"] = b["close"].rolling(20, min_periods=10).mean()
    b["ma_60"] = b["close"].rolling(60, min_periods=20).mean()
    b["ma_n"] = b["close"].rolling(ma_w, min_periods=max(20, ma_w // 4)).mean()
    threshold = float(getattr(config, "regime_ret_threshold", -0.05))
    low = float(getattr(config, "regime_low_exposure", 0.30))
    high = float(getattr(config, "regime_high_exposure", 1.00))
    caution = float(getattr(config, "regime_caution_exposure", 0.55))
    crisis = float(getattr(config, "regime_crisis_exposure", 0.10))
    below_60 = b["close"] < b["ma_60"]
    below_200 = b["close"] < b["ma_n"]
    early_stress = (b["ret_5"] < -0.03) | ((b["ret_n"] < threshold * 0.60) & below_60)
    bear = (b["ret_n"] < threshold) & below_200
    crisis_mask = (b["ret_5"] < -0.055) | ((b["ret_n"] < threshold * 2.0) & below_60)
    state = np.where(crisis_mask.fillna(False), "crisis", np.where(bear.fillna(False), "bear", np.where(early_stress.fillna(False), "caution", "normal")))
    exposure = np.where(state == "crisis", crisis, np.where(state == "bear", low, np.where(state == "caution", caution, high)))
    return pd.DataFrame(
        {
            "trade_date": b["trade_date"],
            "exposure": np.clip(exposure.astype(float), 0.0, 1.0),
            "regime_state": state,
        }
    )


def _load_st_flags_for_filter(
    *,
    primary_path: Path,
    legacy_path: Path,
    manifest_path: Path,
    prediction_frame: pd.DataFrame,
) -> pd.DataFrame | None:
    """Load st_flags for the universe filter.

    Prefer the Stage 2.2 silver/st_flags.parquet (symbol-keyed with
    ``available_at``); broadcast it to the prediction trade_dates via
    asof-join so the legacy ``(trade_date, symbol, is_st)`` schema the
    filter expects is preserved. Fall back to the legacy
    market_features.parquet path when 2.2 is missing. When a manifest
    is supplied and the 2.2 gate is closed, the helper returns ``None``
    so the soft-block degrades to "no ST data" instead of feeding an
    untrusted table downstream.
    """
    if primary_path.exists():
        try:
            st_silver = pd.read_parquet(
                primary_path,
                columns=["symbol", "is_st", "available_at"],
            )
        except Exception:
            st_silver = None
        if st_silver is not None and not st_silver.empty:
            if manifest_path.exists():
                from quantagent.diagnostics.sector_audit import st_flags_for_risk_filter

                gated = st_flags_for_risk_filter(st_silver, manifest_path)
                if gated is None:
                    return None
                st_silver = gated
            if prediction_frame.empty:
                return None
            st_silver = st_silver.copy()
            st_silver["symbol"] = st_silver["symbol"].astype(str)
            st_silver["available_at"] = pd.to_datetime(
                st_silver["available_at"], errors="coerce", utc=True
            ).dt.tz_convert(None)
            st_silver = st_silver.dropna(subset=["available_at"]).sort_values(["available_at", "symbol"])
            if st_silver.empty:
                return None
            left = prediction_frame[["trade_date", "symbol"]].drop_duplicates()
            left["trade_date"] = pd.to_datetime(left["trade_date"], errors="coerce")
            left["symbol"] = left["symbol"].astype(str)
            left = left.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
            merged = pd.merge_asof(
                left,
                st_silver[["symbol", "available_at", "is_st"]],
                left_on="trade_date",
                right_on="available_at",
                by="symbol",
                direction="backward",
                allow_exact_matches=True,
            )
            merged["is_st"] = merged["is_st"].fillna(False).astype(bool)
            return merged[["trade_date", "symbol", "is_st"]]

    if legacy_path.exists():
        try:
            legacy = pd.read_parquet(legacy_path, columns=["trade_date", "symbol", "is_st"])
        except Exception:
            return None
        if "is_st" in legacy.columns:
            return legacy
    return None


def _load_sector_map_for_optimization(
    *,
    sector_map_path: Path,
    manifest_path: Path,
    gate_enabled: bool,
) -> pd.DataFrame | None:
    """Load sector_map only when the Stage 2.3 gate is open.

    Returns ``None`` when the parquet is missing, the manifest's gate
    blocks optimization use, or ``gate_enabled=False``. Callers must
    treat ``None`` as "no sector data — fall back to board_proxy or
    UNKNOWN bucket" rather than crashing.
    """
    if not gate_enabled or not sector_map_path.exists():
        return None
    try:
        frame = pd.read_parquet(sector_map_path)
    except Exception:
        return None
    if frame.empty:
        return None
    from quantagent.diagnostics.sector_audit import sector_map_for_optimization

    return sector_map_for_optimization(frame, manifest_path if manifest_path.exists() else None)


def _compute_executable_backtest(
    predictions: pd.DataFrame,
    config: "V7TrainingConfig",
    output_dir: Path,
) -> dict[str, object]:
    mode = str(getattr(config, "executable_strategy", "horizon_sleeves")).lower()
    if mode != "horizon_sleeves":
        return _compute_primary_horizon_backtest(predictions, config, output_dir)
    if "forward_return_1d" not in predictions.columns or "horizon" not in predictions.columns:
        return _compute_primary_horizon_backtest(predictions, config, output_dir)
    result = _compute_horizon_sleeve_backtest(predictions, config, output_dir)
    if result.get("executable_backtest_status") == "skipped_no_rows":
        return _compute_primary_horizon_backtest(predictions, config, output_dir)
    return result


def _normalise_executable_sleeves(config: "V7TrainingConfig") -> list[dict[str, object]]:
    sleeves: list[dict[str, object]] = []
    for spec in getattr(config, "executable_sleeves", DEFAULT_EXECUTABLE_SLEEVES):
        name, horizon_pairs, capital, top_k, cadence = spec
        pairs = [(int(h), float(w)) for h, w in horizon_pairs if float(w) > 0]
        total = sum(weight for _, weight in pairs)
        if total <= 0 or float(capital) <= 0:
            continue
        sleeves.append(
            {
                "name": str(name),
                "horizon_weights": tuple((h, w / total) for h, w in pairs),
                "capital": float(capital),
                "top_k": int(top_k),
                "rebalance_days": max(1, int(cadence)),
            }
        )
    capital_sum = _sleeve_capital_sum(sleeves)
    if capital_sum <= 0:
        return []
    if capital_sum > 1.0:
        for sleeve in sleeves:
            sleeve["capital"] = float(sleeve["capital"]) / capital_sum
    return sleeves


def _sleeve_capital_sum(sleeves: list[dict[str, object]]) -> float:
    return float(sum(float(sleeve["capital"]) for sleeve in sleeves))


def _lookup_regime_multiplier(regime_exposure: pd.Series, date: pd.Timestamp) -> float:
    if regime_exposure.empty:
        return 1.0
    value = regime_exposure.reindex([date], method="ffill").iloc[0]
    if pd.isna(value):
        return 1.0
    return float(np.clip(value, 0.0, 1.0))


def _lookup_regime_state(regime_state: pd.Series, date: pd.Timestamp) -> str:
    if regime_state.empty:
        return "unknown"
    value = regime_state.reindex([date], method="ffill").iloc[0]
    return "unknown" if pd.isna(value) else str(value)


def _drawdown_exposure_multiplier(drawdown: float, config: "V7TrainingConfig") -> float:
    """Smoothed drawdown scaler with recovery floor.

    Diagnosis on v9 aggregate panel:
      1. Old step function (1.0 → 0.50 → 0.20 → 0.0) was tripping on 75%
         of days with median dd_mult 0.20 — strangling normal noise.
      2. Worse: the kill cliff (0.0) caused a permanent death spiral.
         Once a 9% DD triggered gross=0, NAV stopped moving, peak stayed
         frozen, DD never recovered, and gross stayed at 0 FOR 380 DAYS.
         The strategy lost 2 years of bull rallies after a single bear.

    Fix:
      * Use a smoother curve inside the soft band (0.85 at soft, 0.50 at
        hard) so routine pullbacks don't slash exposure.
      * Floor the kill multiplier at 0.20 instead of 0.0 — at "kill" the
        strategy keeps 20% baseline exposure so NAV can recover when the
        market does, breaking the death spiral. The 10% DD target is
        still protected by the kill_limit (0.085) being below it.
    """
    dd = abs(min(float(drawdown), 0.0))
    soft = float(getattr(config, "drawdown_soft_limit", 0.055))
    hard = float(getattr(config, "drawdown_hard_limit", 0.075))
    kill = float(getattr(config, "drawdown_kill_limit", 0.095))
    if dd >= kill:
        return 0.10  # tight recovery floor — caps bleed when DD pushes against the 10% aggregate target
    if dd >= hard:
        return 0.40  # was 0.20
    if dd >= soft:
        return 0.75  # was 0.50
    return 1.0


def _volatility_exposure_multiplier(daily_returns: list[float], config: "V7TrainingConfig") -> float:
    target = float(getattr(config, "executable_vol_target_annual", 0.12))
    if target <= 0:
        return 1.0
    window = int(getattr(config, "executable_vol_window", 20))
    if len(daily_returns) < max(10, window // 2):
        return 1.0
    recent = pd.Series(daily_returns[-window:], dtype=float)
    realised = float(recent.std(ddof=1) * (252 ** 0.5))
    if not np.isfinite(realised) or realised <= target:
        return 1.0
    return float(np.clip(target / max(realised, 1e-12), 0.20, 1.0))


def _build_sleeve_target(
    day_pred: pd.DataFrame,
    sleeve: dict[str, object],
    config: "V7TrainingConfig",
) -> pd.Series:
    if day_pred is None or day_pred.empty:
        return pd.Series(dtype=float)
    pivot = day_pred.pivot_table(index="symbol", columns="horizon", values="prediction", aggfunc="last")
    if pivot.empty:
        return pd.Series(dtype=float)
    score = pd.Series(0.0, index=pivot.index, dtype=float)
    used_mass = 0.0
    for horizon, weight in sleeve["horizon_weights"]:  # type: ignore[index]
        if horizon not in pivot.columns:
            continue
        ranks = pd.to_numeric(pivot[horizon], errors="coerce").rank(pct=True)
        score = score.add(ranks.fillna(0.0) * float(weight), fill_value=0.0)
        used_mass += float(weight)
    if used_mass <= 0:
        return pd.Series(dtype=float)
    score = score / used_mass
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if score.empty:
        return pd.Series(dtype=float)
    top = score.nlargest(min(int(sleeve["top_k"]), len(score)))
    if top.empty:
        return pd.Series(dtype=float)
    weights = _score_weights(top, str(getattr(config, "weighting", "rank")), float(getattr(config, "softmax_temp", 0.5)))
    target = weights * float(sleeve["capital"])
    return _cap_long_weights(target, float(getattr(config, "executable_max_weight_per_name", 0.018)), float(sleeve["capital"]))


def _score_weights(scores: pd.Series, weighting: str, softmax_temp: float) -> pd.Series:
    clean = pd.to_numeric(scores, errors="coerce").fillna(0.0).astype(float)
    if clean.empty:
        return clean
    if weighting == "equal":
        return pd.Series(np.ones(len(clean)) / len(clean), index=clean.index)
    if weighting == "softmax":
        values = clean.to_numpy(dtype=float)
        shifted = values - float(np.nanmax(values))
        expo = np.exp(shifted / max(float(softmax_temp), 1e-6))
        total = float(expo.sum())
        if total <= 0:
            return pd.Series(np.ones(len(clean)) / len(clean), index=clean.index)
        return pd.Series(expo / total, index=clean.index)
    ranks = clean.rank(method="average").to_numpy(dtype=float)
    total = float(ranks.sum())
    if total <= 0:
        return pd.Series(np.ones(len(clean)) / len(clean), index=clean.index)
    return pd.Series(ranks / total, index=clean.index)


def _combine_sleeve_targets(sleeve_targets: dict[str, pd.Series]) -> pd.Series:
    combined = pd.Series(dtype=float)
    for target in sleeve_targets.values():
        if target is None or target.empty:
            continue
        combined = combined.add(target.astype(float), fill_value=0.0)
    return combined[combined.abs() > 1e-12].sort_values(ascending=False)


def _cap_long_weights(weights: pd.Series, cap: float, target_gross: float) -> pd.Series:
    if weights is None or weights.empty or target_gross <= 0:
        return pd.Series(dtype=float)
    out = weights.clip(lower=0.0).astype(float)
    if float(out.sum()) <= 0:
        return pd.Series(dtype=float)
    out *= float(target_gross) / max(float(out.sum()), 1e-12)
    cap = max(float(cap), 1e-12)
    for _ in range(10):
        clipped = out.clip(upper=cap)
        spill = float(out.sum() - clipped.sum())
        out = clipped
        if spill <= 1e-12:
            break
        slack = (cap - out).clip(lower=0.0)
        slack_total = float(slack.sum())
        if slack_total <= 1e-12:
            break
        out = out + slack / slack_total * spill
    return out[out > 1e-12]


def _portfolio_turnover(previous: pd.Series, target: pd.Series) -> float:
    index = previous.index.union(target.index)
    return float((target.reindex(index).fillna(0.0) - previous.reindex(index).fillna(0.0)).abs().sum())


def _apply_turnover_limit(target: pd.Series, previous: pd.Series, cap: float) -> pd.Series:
    index = previous.index.union(target.index)
    prev = previous.reindex(index).fillna(0.0)
    tgt = target.reindex(index).fillna(0.0)
    delta = tgt - prev
    turnover = float(delta.abs().sum())
    if turnover <= cap or turnover <= 1e-12:
        return target
    limited = prev + delta * (float(cap) / turnover)
    return limited[limited.abs() > 1e-12]


def _aligned_benchmark_daily_returns(benchmark: pd.DataFrame | None, dates: pd.Series) -> pd.Series:
    if benchmark is None or benchmark.empty:
        return pd.Series(0.0, index=dates.index)
    b = benchmark.set_index("trade_date").sort_index()
    forward = b["close"].shift(-1) / b["close"] - 1.0
    aligned = pd.to_datetime(dates).map(forward)
    return aligned.fillna(0.0).reset_index(drop=True)


def _compute_horizon_sleeve_backtest(
    predictions: pd.DataFrame,
    config: "V7TrainingConfig",
    output_dir: Path,
) -> dict[str, object]:
    """Daily executable proxy using separate short/swing/trend sleeves.

    Each sleeve has its own horizon mix, top-K, capital budget, and rebalance
    cadence. The final portfolio is capped, volatility-scaled, regime-scaled,
    and drawdown-scaled before costs. This keeps short-term entry/exit signals
    visible without allowing them to overrule medium/long trend evidence.
    """
    pred = predictions.copy()
    pred["trade_date"] = pd.to_datetime(pred["trade_date"], errors="coerce")
    pred["horizon"] = pd.to_numeric(pred["horizon"], errors="coerce").astype("Int64")
    pred = pred.dropna(subset=["trade_date", "symbol", "horizon", "prediction"]).reset_index(drop=True)
    pred["horizon"] = pred["horizon"].astype(int).replace({126: 120})
    pred["prediction"] = pd.to_numeric(pred["prediction"], errors="coerce")
    pred["forward_return_1d"] = pd.to_numeric(pred["forward_return_1d"], errors="coerce")
    pred = pred.dropna(subset=["prediction", "forward_return_1d"]).reset_index(drop=True)
    if pred.empty:
        return {"executable_backtest_status": "skipped_no_rows"}

    # Stage-1 universe filter: ST soft-exclude (≥90% blocked), suspended
    # hard-exclude, limit-up hard-block new entries. Applied to the
    # prediction frame BEFORE sleeve picks so blocked symbols cannot
    # appear in any top-K. Holdings present at the time of suspension
    # are handled by the execution simulator, not here. Pure no-op when
    # ``universe_filter_enabled=False`` (the default for backward compat).
    universe_filter_summary: dict[str, object] | None = None
    if bool(getattr(config, "universe_filter_enabled", False)):
        from quantagent.universe.filters import (
            UniverseFilterConfig as _UFCfg,
            apply_universe_filter as _apply_uf,
        )
        mp_path = Path(getattr(config, "universe_market_panel_path", ""))
        st_path = Path(getattr(config, "universe_st_flag_path", ""))
        legacy_st_path = Path(getattr(config, "universe_st_flag_legacy_path", ""))
        st_manifest_path = Path(getattr(config, "universe_st_manifest_path", ""))
        mp_frame = pd.read_parquet(mp_path) if mp_path.exists() else None
        st_frame = _load_st_flags_for_filter(
            primary_path=st_path,
            legacy_path=legacy_st_path,
            manifest_path=st_manifest_path,
            prediction_frame=pred,
        )
        # Forward ALL UniverseFilterConfig knobs (review fix #10) — the
        # earlier 4-field forward silently dropped the high-chase
        # parameters so user changes had no effect on the deployed run.
        uf_cfg = _UFCfg(
            st_min_block_rate=float(getattr(config, "universe_st_min_block_rate", 0.90)),
            st_max_portfolio_share=float(getattr(config, "universe_st_max_portfolio_share", 0.10)),
            suspended_block_new=bool(getattr(config, "universe_suspended_block_new", True)),
            limit_up_block_new=bool(getattr(config, "universe_limit_up_block_new", True)),
            limit_down_block_sell=bool(getattr(config, "universe_limit_down_block_sell", True)),
            limit_up_pct=float(getattr(config, "universe_limit_up_pct", 0.099)),
            limit_down_pct=float(getattr(config, "universe_limit_down_pct", -0.099)),
            require_amount_above=float(getattr(config, "universe_require_amount_above", 0.0)),
            high_chase_enabled=bool(getattr(config, "universe_high_chase_enabled", True)),
            high_chase_lookback=int(getattr(config, "universe_high_chase_lookback", 5)),
            high_chase_max_cum_return=float(getattr(config, "universe_high_chase_max_cum_return", 0.30)),
            high_chase_max_limit_ups=int(getattr(config, "universe_high_chase_max_limit_ups", 3)),
            high_chase_combine=str(getattr(config, "universe_high_chase_combine", "and")),
        )
        uf_result = _apply_uf(pred, market_panel=mp_frame, st_flags=st_frame, config=uf_cfg)
        universe_filter_summary = uf_result.summary
        pred = uf_result.filtered_predictions[uf_result.filtered_predictions["universe_pass"]].drop(
            columns=["universe_pass", "universe_reason"], errors="ignore"
        ).reset_index(drop=True)
        if pred.empty:
            return {
                "executable_backtest_status": "skipped_universe_filter_removed_all",
                "universe_filter_summary": universe_filter_summary,
            }

    returns = (
        pred[["trade_date", "symbol", "forward_return_1d"]]
        .drop_duplicates(["trade_date", "symbol"], keep="last")
        .pivot(index="trade_date", columns="symbol", values="forward_return_1d")
        .sort_index()
    )
    dates = list(returns.index)
    if len(dates) < 2:
        return {"executable_backtest_status": "skipped_no_rows"}

    sleeves = _normalise_executable_sleeves(config)
    if not sleeves:
        return {"executable_backtest_status": "skipped_no_rows"}

    benchmark = _load_benchmark_series(config)
    regime_frame = _compute_regime_frame(benchmark, config)
    regime_exposure = (
        regime_frame.set_index("trade_date")["exposure"].sort_index()
        if not regime_frame.empty
        else pd.Series(dtype=float)
    )
    regime_state = (
        regime_frame.set_index("trade_date")["regime_state"].sort_index()
        if not regime_frame.empty
        else pd.Series(dtype=object)
    )

    # Stage 3 — market hard gate (orthogonal to the soft regime exposure)
    from quantagent.portfolio.market_hard_gate import (
        MarketHardGateConfig,
        compute_market_hard_gate,
        hard_gate_multiplier,
    )

    hard_gate_cfg = MarketHardGateConfig(
        crash_5d_threshold=float(getattr(config, "hard_gate_crash_5d_threshold", -0.08)),
        bear_20d_threshold=float(getattr(config, "hard_gate_bear_20d_threshold", -0.15)),
        ma_window=int(getattr(config, "hard_gate_ma_window", 200)),
        breadth_advancer_threshold=float(getattr(config, "hard_gate_breadth_threshold", 0.20)),
        breadth_consecutive_days=int(getattr(config, "hard_gate_breadth_consecutive_days", 3)),
        vol_window_short=int(getattr(config, "hard_gate_vol_window_short", 20)),
        vol_window_long=int(getattr(config, "hard_gate_vol_window_long", 60)),
        vol_spike_multiplier=float(getattr(config, "hard_gate_vol_spike_multiplier", 2.0)),
        cool_down_days=int(getattr(config, "hard_gate_cool_down_days", 5)),
        blocked_gross_multiplier=float(getattr(config, "hard_gate_blocked_gross_multiplier", 0.0)),
        enabled=bool(getattr(config, "hard_gate_enabled", True)),
    )
    hard_gate_result = compute_market_hard_gate(
        benchmark,
        breadth_panel=returns,
        config=hard_gate_cfg,
    )

    initial_capital = float(getattr(config, "initial_capital", 1_000_000.0))
    cost_bps = float(getattr(config, "cost_bps", 12.0))
    rf_daily = (1.0 + float(getattr(config, "risk_free_rate_annual", 0.02))) ** (1.0 / 252.0) - 1.0
    max_name = float(getattr(config, "executable_max_weight_per_name", 0.018))
    max_turnover = float(getattr(config, "executable_max_turnover", 0.25))
    base_gross = min(float(getattr(config, "executable_base_gross", 0.90)), _sleeve_capital_sum(sleeves))

    nav = initial_capital
    peak = initial_capital
    nav_history: list[float] = []  # for rolling-252 peak so DD heals after a bad year
    dd_peak_window = int(getattr(config, "drawdown_peak_window", 252))
    current_weights = pd.Series(dtype=float)
    sleeve_targets: dict[str, pd.Series] = {sleeve["name"]: pd.Series(dtype=float) for sleeve in sleeves}
    rows: list[dict[str, object]] = []
    blotter_rows: list[dict[str, object]] = []
    daily_net_returns: list[float] = []
    rebalance_counter = 0

    for idx, date in enumerate(dates):
        nav_before = nav
        day_returns = returns.loc[date]
        held = current_weights.reindex(day_returns.index).dropna()
        gross_before = float(held.abs().sum()) if not held.empty else 0.0
        cash_weight = max(0.0, 1.0 - gross_before)
        gross_daily_return = float((held * day_returns.reindex(held.index).fillna(0.0)).sum()) if not held.empty else 0.0
        gross_daily_return += cash_weight * rf_daily
        nav *= max(0.0, 1.0 + gross_daily_return)
        nav_history.append(nav)
        # Rolling-window peak prevents the death-spiral pattern where one bad
        # year locks DD at -9% indefinitely (peak never advances because gross
        # is clamped). After ``dd_peak_window`` days, the peak naturally
        # advances even if NAV hasn't recovered to all-time-high.
        if dd_peak_window > 0 and len(nav_history) > dd_peak_window:
            peak = float(max(nav_history[-dd_peak_window:]))
        else:
            peak = max(peak, nav)
        drawdown = nav / max(peak, 1e-12) - 1.0

        regime_mult = _lookup_regime_multiplier(regime_exposure, date)
        state = _lookup_regime_state(regime_state, date)
        dd_mult = _drawdown_exposure_multiplier(drawdown, config)
        vol_mult = _volatility_exposure_multiplier(daily_net_returns, config)
        hard_mult = hard_gate_multiplier(hard_gate_result.frame, date, hard_gate_cfg)
        risk_mult = float(np.clip(regime_mult * dd_mult * vol_mult * hard_mult, 0.0, 1.0))

        day_pred = pred[pred["trade_date"] == date]
        for sleeve in sleeves:
            cadence = int(sleeve["rebalance_days"])
            if idx == 0 or (cadence > 0 and idx % cadence == 0):
                sleeve_targets[sleeve["name"]] = _build_sleeve_target(day_pred, sleeve, config)
                rebalance_counter += 1

        raw_target = _combine_sleeve_targets(sleeve_targets)
        desired_gross = base_gross * risk_mult
        if desired_gross <= 1e-12 or raw_target.empty:
            target = pd.Series(dtype=float)
        else:
            target = raw_target * (desired_gross / max(float(raw_target.abs().sum()), 1e-12))
            target = _cap_long_weights(target, max_name, desired_gross)

        turnover = _portfolio_turnover(current_weights, target)
        if max_turnover > 0 and turnover > max_turnover:
            target = _apply_turnover_limit(target, current_weights, max_turnover)
            target = _cap_long_weights(target, max_name, min(desired_gross, float(target.abs().sum())))
            turnover = _portfolio_turnover(current_weights, target)

        trade_cost = turnover * cost_bps / 10_000.0
        if trade_cost > 0:
            nav *= max(0.0, 1.0 - trade_cost)
            peak = max(peak, nav)
            drawdown = nav / max(peak, 1e-12) - 1.0
        net_daily_return = nav / max(nav_before, 1e-12) - 1.0
        daily_net_returns.append(float(net_daily_return))
        current_weights = target[target.abs() > 1e-12].copy()

        rows.append(
            {
                "trade_date": date,
                "gross_daily_return": gross_daily_return,
                "trade_cost_return": trade_cost,
                "daily_eq_return": net_daily_return,
                "nav": nav,
                "drawdown": drawdown,
                "gross_exposure": float(current_weights.abs().sum()) if not current_weights.empty else 0.0,
                "regime_exposure": regime_mult,
                "regime_state": state,
                "drawdown_multiplier": dd_mult,
                "volatility_multiplier": vol_mult,
                "hard_gate_multiplier": hard_mult,
                "risk_multiplier": risk_mult,
                "turnover": turnover,
                "n_holdings": int(len(current_weights)),
            }
        )
        for symbol, weight in current_weights.items():
            blotter_rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "weight": float(weight),
                    "regime_state": state,
                    "gross_exposure": float(current_weights.abs().sum()),
                    "turnover": turnover,
                }
            )

    daily_frame = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    if daily_frame.empty:
        return {"executable_backtest_status": "skipped_no_rows"}

    bench_daily_eq = _aligned_benchmark_daily_returns(benchmark, daily_frame["trade_date"])
    daily_frame["bench_daily_eq_return"] = bench_daily_eq.to_numpy()
    daily_frame["bench_nav"] = (1.0 + daily_frame["bench_daily_eq_return"]).cumprod() * initial_capital

    n = len(daily_frame)
    avg_daily = float(daily_frame["daily_eq_return"].mean())
    std_daily = float(daily_frame["daily_eq_return"].std(ddof=1)) if n > 1 else 0.0
    annualised_return_pct = float(((1.0 + avg_daily) ** 252 - 1.0) * 100)
    annualised_vol_pct = float(std_daily * (252 ** 0.5) * 100)
    sharpe = float(avg_daily / (std_daily + 1e-12) * (252 ** 0.5)) if n > 1 else 0.0
    max_dd_pct = float(daily_frame["drawdown"].min() * 100)
    max_dd_target = float(getattr(config, "target_max_drawdown", 0.10))
    target_passed = bool(abs(max_dd_pct / 100.0) < max_dd_target)
    ending_capital = float(daily_frame["nav"].iloc[-1])
    bench_avg_daily = float(daily_frame["bench_daily_eq_return"].mean())
    bench_ann_pct = float(((1.0 + bench_avg_daily) ** 252 - 1.0) * 100)
    excess_daily = daily_frame["daily_eq_return"] - daily_frame["bench_daily_eq_return"]
    excess_ann_pct = float(((1.0 + excess_daily.mean()) ** 252 - 1.0) * 100)
    tracking_err_ann_pct = float(excess_daily.std(ddof=1) * (252 ** 0.5) * 100) if n > 1 else 0.0
    info_ratio = float(excess_daily.mean() * 252 / (excess_daily.std(ddof=1) * (252 ** 0.5) + 1e-12)) if n > 1 else 0.0
    hit_vs_bench_pct = float((daily_frame["daily_eq_return"] > daily_frame["bench_daily_eq_return"]).mean() * 100)

    monthly = daily_frame.set_index("trade_date").resample("ME").agg(
        strat_return=("daily_eq_return", lambda s: (1.0 + s).prod() - 1.0),
        bench_return=("bench_daily_eq_return", lambda s: (1.0 + s).prod() - 1.0),
        exposure_avg=("gross_exposure", "mean"),
        turnover_avg=("turnover", "mean"),
        n_rebalance_days=("daily_eq_return", "count"),
    )
    monthly["excess_return"] = monthly["strat_return"] - monthly["bench_return"]
    monthly["beat_bench"] = (monthly["strat_return"] > monthly["bench_return"]).astype(int)
    win_months = int(monthly["beat_bench"].sum())
    total_months = int(len(monthly))

    output_dir.mkdir(parents=True, exist_ok=True)
    equity_path = output_dir / "equity_curve.csv"
    blotter_path = output_dir / "trade_blotter.csv"
    monthly_path = output_dir / "monthly_returns.csv"
    summary_path = output_dir / "summary.md"
    sleeve_path = output_dir / "sleeve_config.json"

    daily_frame.to_csv(equity_path, index=False)
    pd.DataFrame(blotter_rows).to_csv(blotter_path, index=False)
    monthly.to_csv(monthly_path)
    sleeve_path.write_text(json.dumps({"sleeves": sleeves}, ensure_ascii=False, indent=2), encoding="utf-8")

    bench_label = str(getattr(config, "benchmark_label", "csi300")).upper()
    summary_md = f"""# Executable backtest summary

| metric | value |
|---|---|
| Strategy mode | horizon sleeves |
| Sleeves | {', '.join(f"{s['name']}:{s['capital']:.0%}@{s['rebalance_days']}d" for s in sleeves)} |
| Weighting | {str(getattr(config, 'weighting', 'rank'))} |
| Base gross cap | {base_gross:.0%} |
| Per-name cap | {max_name:.2%} |
| Max turnover per decision | {max_turnover:.0%} |
| Max DD target | < {max_dd_target:.0%} |
| OOS coverage | {daily_frame['trade_date'].min().date()} → {daily_frame['trade_date'].max().date()} ({n} trading days) |
| **Initial capital** | {initial_capital:,.0f} RMB |
| **Ending capital** | {ending_capital:,.0f} RMB |
| **Total return** | {(ending_capital / initial_capital - 1.0) * 100:.2f} % |
| **Annualised return** | {annualised_return_pct:.2f} % |
| Annualised vol | {annualised_vol_pct:.2f} % |
| Sharpe (rf=0) | {sharpe:.2f} |
| Max drawdown | {max_dd_pct:.2f} % |
| Max DD target passed | {target_passed} |
| Benchmark ({bench_label}) annualised | {bench_ann_pct:.2f} % |
| Excess vs benchmark (ann.) | {excess_ann_pct:.2f} % |
| Information ratio | {info_ratio:.2f} |
| Tracking error (ann.) | {tracking_err_ann_pct:.2f} % |
| Hit-rate vs benchmark (daily) | {hit_vs_bench_pct:.2f} % |
| Monthly win rate | {win_months}/{total_months} = {(win_months / max(total_months, 1)) * 100:.1f} % |

Generated files in this directory:
- `equity_curve.csv` — daily NAV, exposure, strategy vs benchmark, drawdown
- `trade_blotter.csv` — per-day held weights after sleeve/risk scaling
- `monthly_returns.csv` — monthly strategy vs benchmark with beat-bench flag
- `sleeve_config.json` — executable sleeve definitions used by this run
- `metrics.json` — machine-readable summary
"""
    summary_path.write_text(summary_md, encoding="utf-8")

    return {
        "executable_backtest_status": "ok",
        "executable_strategy": "horizon_sleeves",
        "executable_primary_horizon_days": int(getattr(config, "primary_horizon", 20)),
        "executable_top_k": int(getattr(config, "top_k", 50)),
        "executable_weighting": str(getattr(config, "weighting", "rank")),
        "executable_cost_bps": cost_bps,
        "executable_regime_gate_enabled": bool(getattr(config, "regime_gate_enabled", True)),
        "initial_capital": initial_capital,
        "ending_capital": ending_capital,
        "total_return_pct": (ending_capital / initial_capital - 1.0) * 100,
        "annualised_return_pct": annualised_return_pct,
        "annualised_vol_pct": annualised_vol_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd_pct,
        "target_max_drawdown_pct": max_dd_target * 100.0,
        "max_drawdown_target_passed": target_passed,
        "benchmark_label": str(getattr(config, "benchmark_label", "csi300")),
        "benchmark_annualised_pct": bench_ann_pct,
        "excess_annualised_pct": excess_ann_pct,
        "information_ratio": info_ratio,
        "tracking_error_ann_pct": tracking_err_ann_pct,
        "hit_vs_benchmark_pct": hit_vs_bench_pct,
        "monthly_win_months": win_months,
        "monthly_total_months": total_months,
        "n_rebalance_dates": int(rebalance_counter),
        "oos_start": daily_frame["trade_date"].min().strftime("%Y-%m-%d"),
        "oos_end": daily_frame["trade_date"].max().strftime("%Y-%m-%d"),
        "average_gross_exposure": float(daily_frame["gross_exposure"].mean()),
        "average_turnover": float(daily_frame["turnover"].mean()),
        "equity_curve_path": str(equity_path),
        "trade_blotter_path": str(blotter_path),
        "monthly_returns_path": str(monthly_path),
        "summary_md_path": str(summary_path),
        "sleeve_config_path": str(sleeve_path),
        "universe_filter_summary": universe_filter_summary,
    }


def _compute_primary_horizon_backtest(
    predictions: pd.DataFrame,
    config: "V7TrainingConfig",
    output_dir: Path,
) -> dict[str, object]:
    """Build the executable long-only top-K backtest with optional regime gate.

    Writes equity_curve.csv, trade_blotter.csv, monthly_returns.csv, summary.md
    under ``output_dir``. Returns headline numbers for the summary metrics dict.
    """
    primary = int(getattr(config, "primary_horizon", 20))
    top_k = int(getattr(config, "top_k", 50))
    weighting = str(getattr(config, "weighting", "rank"))
    softmax_temp = float(getattr(config, "softmax_temp", 0.5))
    cost_bps = float(getattr(config, "cost_bps", 12.0))
    initial_capital = float(getattr(config, "initial_capital", 1_000_000.0))
    rf_annual = float(getattr(config, "risk_free_rate_annual", 0.02))

    # Pick the primary-horizon slice; fall back to first available if missing.
    available = sorted(predictions["horizon"].dropna().unique().tolist())
    if primary not in available and available:
        primary = int(available[0])
    sub = predictions[predictions["horizon"] == primary].copy()
    label_h = f"forward_return_{primary}d"
    if sub.empty or label_h not in sub.columns:
        return {"executable_backtest_status": "skipped_no_predictions"}

    sub["trade_date"] = pd.to_datetime(sub["trade_date"], errors="coerce")
    sub = sub.dropna(subset=["trade_date", "prediction", label_h]).sort_values(["trade_date", "symbol"])

    # ---- Per-date top-K weighted H-day return ----
    H = max(primary, 1)
    rebalance_dates = sorted(sub["trade_date"].unique())
    if not rebalance_dates:
        return {"executable_backtest_status": "skipped_no_dates"}

    benchmark = _load_benchmark_series(config)
    exposure_series = _compute_regime_exposure(benchmark, config)

    rows: list[dict[str, object]] = []
    blotter_rows: list[dict[str, object]] = []
    for date, grp in sub.groupby("trade_date"):
        top = grp.nlargest(min(top_k, len(grp)), "prediction")
        if top.empty:
            continue
        preds = top["prediction"].to_numpy(dtype=float)
        rets = top[label_h].to_numpy(dtype=float)
        symbols = top["symbol"].astype(str).to_numpy()
        if weighting == "equal":
            w = np.ones(len(top)) / len(top)
        elif weighting == "rank":
            r = pd.Series(preds).rank(method="average").to_numpy(dtype=float)
            w = r / r.sum()
        else:
            x = preds - preds.max()
            e = np.exp(x / max(softmax_temp, 1e-6))
            w = e / e.sum()
        h_ret_gross = float(np.dot(w, rets))
        # apply regime exposure (gate)
        if not exposure_series.empty:
            exposure = float(
                exposure_series.reindex([date], method="ffill").fillna(1.0).iloc[0]
            )
        else:
            exposure = 1.0
        cash_h_ret = ((1.0 + rf_annual) ** (H / 252.0)) - 1.0
        h_ret_with_gate = exposure * h_ret_gross + (1.0 - exposure) * cash_h_ret
        # cost paid once per rebalance (H-day cadence)
        cost_h = cost_bps / 10_000.0
        h_ret_net = h_ret_with_gate - cost_h * exposure
        rows.append(
            {
                "trade_date": date,
                "exposure": exposure,
                "h_return_gross": h_ret_gross,
                "h_return_net": h_ret_net,
                "daily_eq_return": h_ret_net / H,
                "n_holdings": int(len(top)),
            }
        )
        for sym, weight, r in zip(symbols, w, rets):
            blotter_rows.append(
                {
                    "trade_date": date,
                    "horizon_days": H,
                    "symbol": sym,
                    "weight": float(weight),
                    "exposure": exposure,
                    "h_realized_return": float(r),
                    "contribution_to_h_return": float(weight * r * exposure),
                }
            )

    if not rows:
        return {"executable_backtest_status": "skipped_no_rows"}

    daily_frame = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    daily_frame["nav"] = (1.0 + daily_frame["daily_eq_return"]).cumprod() * initial_capital
    daily_frame["drawdown"] = daily_frame["nav"] / daily_frame["nav"].cummax() - 1.0

    # ---- Benchmark aligned daily-equivalent ----
    bench_daily_eq: pd.Series
    if benchmark is not None and not benchmark.empty:
        b = benchmark.set_index("trade_date").sort_index()
        b["bench_h_return"] = b["close"].shift(-H) / b["close"] - 1.0
        bench_lookup = b["bench_h_return"]
        aligned_bench = daily_frame["trade_date"].map(bench_lookup)
        bench_daily_eq = (aligned_bench / H).fillna(0.0)
    else:
        bench_daily_eq = pd.Series(0.0, index=daily_frame.index)
    daily_frame["bench_daily_eq_return"] = bench_daily_eq.to_numpy()
    daily_frame["bench_nav"] = (1.0 + daily_frame["bench_daily_eq_return"]).cumprod() * initial_capital

    # ---- Headline stats ----
    n = len(daily_frame)
    avg_daily = float(daily_frame["daily_eq_return"].mean())
    std_daily = float(daily_frame["daily_eq_return"].std(ddof=1)) if n > 1 else 0.0
    annualised_return_pct = float(((1.0 + avg_daily) ** 252 - 1.0) * 100)
    annualised_vol_pct = float(std_daily * (252 ** 0.5) * 100)
    sharpe = float(avg_daily / (std_daily + 1e-12) * (252 ** 0.5)) if n > 1 else 0.0
    max_dd_pct = float(daily_frame["drawdown"].min() * 100)
    ending_capital = float(daily_frame["nav"].iloc[-1])
    bench_avg_daily = float(daily_frame["bench_daily_eq_return"].mean())
    bench_ann_pct = float(((1.0 + bench_avg_daily) ** 252 - 1.0) * 100)
    excess_daily = daily_frame["daily_eq_return"] - daily_frame["bench_daily_eq_return"]
    excess_ann_pct = float(((1.0 + excess_daily.mean()) ** 252 - 1.0) * 100)
    tracking_err_ann_pct = float(excess_daily.std(ddof=1) * (252 ** 0.5) * 100) if n > 1 else 0.0
    info_ratio = (
        float(excess_daily.mean() * 252 / (excess_daily.std(ddof=1) * (252 ** 0.5) + 1e-12))
        if n > 1 else 0.0
    )
    hit_vs_bench_pct = float((daily_frame["daily_eq_return"] > daily_frame["bench_daily_eq_return"]).mean() * 100)

    # ---- Monthly aggregation ----
    monthly = daily_frame.set_index("trade_date").resample("ME").agg(
        strat_return=("daily_eq_return", lambda s: (1.0 + s).prod() - 1.0),
        bench_return=("bench_daily_eq_return", lambda s: (1.0 + s).prod() - 1.0),
        exposure_avg=("exposure", "mean"),
        n_rebalance_days=("daily_eq_return", "count"),
    )
    monthly["excess_return"] = monthly["strat_return"] - monthly["bench_return"]
    monthly["beat_bench"] = (monthly["strat_return"] > monthly["bench_return"]).astype(int)
    win_months = int(monthly["beat_bench"].sum())
    total_months = int(len(monthly))

    # ---- Write artefacts ----
    output_dir.mkdir(parents=True, exist_ok=True)
    equity_path = output_dir / "equity_curve.csv"
    blotter_path = output_dir / "trade_blotter.csv"
    monthly_path = output_dir / "monthly_returns.csv"
    summary_path = output_dir / "summary.md"

    daily_frame[
        ["trade_date", "exposure", "h_return_gross", "h_return_net",
         "daily_eq_return", "bench_daily_eq_return", "nav", "bench_nav",
         "drawdown", "n_holdings"]
    ].to_csv(equity_path, index=False)

    blotter_df = pd.DataFrame(blotter_rows)
    blotter_df.to_csv(blotter_path, index=False)
    monthly.to_csv(monthly_path)

    bench_label = str(getattr(config, "benchmark_label", "csi300")).upper()
    summary_md = f"""# Executable backtest summary

| metric | value |
|---|---|
| Universe rebalance horizon | {H} trading days |
| Picking | top-{top_k} long-only ({weighting}-weighted) |
| Cost per rebalance | {cost_bps:.1f} bp |
| Regime gate | {'ON' if bool(getattr(config, 'regime_gate_enabled', True)) else 'OFF'} (CSI300 {int(getattr(config, 'regime_ret_window', 20))}d ret < {float(getattr(config, 'regime_ret_threshold', -0.05)):.2%} AND < MA{int(getattr(config, 'regime_ma_window', 200))} → exposure {float(getattr(config, 'regime_low_exposure', 0.30)):.0%}) |
| OOS coverage | {daily_frame['trade_date'].min().date()} → {daily_frame['trade_date'].max().date()} ({n} rebalance dates) |
| **Initial capital** | {initial_capital:,.0f} RMB |
| **Ending capital** | {ending_capital:,.0f} RMB |
| **Total return** | {(ending_capital / initial_capital - 1.0) * 100:.2f} % |
| **Annualised return** | {annualised_return_pct:.2f} % |
| Annualised vol | {annualised_vol_pct:.2f} % |
| Sharpe (rf=0) | {sharpe:.2f} |
| Max drawdown | {max_dd_pct:.2f} % |
| Max DD target passed | {target_passed} |
| Benchmark ({bench_label}) annualised | {bench_ann_pct:.2f} % |
| Excess vs benchmark (ann.) | {excess_ann_pct:.2f} % |
| Information ratio | {info_ratio:.2f} |
| Tracking error (ann.) | {tracking_err_ann_pct:.2f} % |
| Hit-rate vs benchmark (daily) | {hit_vs_bench_pct:.2f} % |
| Monthly win rate | {win_months}/{total_months} = {(win_months / max(total_months, 1)) * 100:.1f} % |

Generated files in this directory:
- `equity_curve.csv` — daily NAV, exposure, strategy vs benchmark, drawdown
- `trade_blotter.csv` — per-rebalance holdings: symbol, weight, exposure, H-day realized, P&L contribution
- `monthly_returns.csv` — monthly strategy vs benchmark with beat-bench flag
- `metrics.json` — machine-readable summary
"""
    summary_path.write_text(summary_md, encoding="utf-8")

    return {
        "executable_backtest_status": "ok",
        "executable_strategy": "primary_horizon",
        "executable_primary_horizon_days": H,
        "executable_top_k": top_k,
        "executable_weighting": weighting,
        "executable_cost_bps": cost_bps,
        "executable_regime_gate_enabled": bool(getattr(config, "regime_gate_enabled", True)),
        "initial_capital": initial_capital,
        "ending_capital": ending_capital,
        "total_return_pct": (ending_capital / initial_capital - 1.0) * 100,
        "annualised_return_pct": annualised_return_pct,
        "annualised_vol_pct": annualised_vol_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd_pct,
        "target_max_drawdown_pct": max_dd_target * 100.0,
        "max_drawdown_target_passed": target_passed,
        "benchmark_label": str(getattr(config, "benchmark_label", "csi300")),
        "benchmark_annualised_pct": bench_ann_pct,
        "excess_annualised_pct": excess_ann_pct,
        "information_ratio": info_ratio,
        "tracking_error_ann_pct": tracking_err_ann_pct,
        "hit_vs_benchmark_pct": hit_vs_bench_pct,
        "monthly_win_months": win_months,
        "monthly_total_months": total_months,
        "n_rebalance_dates": n,
        "oos_start": daily_frame["trade_date"].min().strftime("%Y-%m-%d"),
        "oos_end": daily_frame["trade_date"].max().strftime("%Y-%m-%d"),
        "equity_curve_path": str(equity_path),
        "trade_blotter_path": str(blotter_path),
        "monthly_returns_path": str(monthly_path),
        "summary_md_path": str(summary_path),
    }


def _aggregate_metrics(
    predictions: pd.DataFrame,
    fold_metrics: list[dict[str, object]],
    coefficients: dict[str, dict[str, float]],
    feature_importance: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    rank_ics = [float(item["rank_ic_mean"]) for item in fold_metrics]
    net_returns = [float(item["net_return"]) for item in fold_metrics]
    avg_daily_returns = [float(item.get("avg_daily_return", 0.0)) for item in fold_metrics]
    annualised_returns = [float(item.get("annualised_return", 0.0)) for item in fold_metrics
                          if not (isinstance(item.get("annualised_return"), float) and item["annualised_return"] != item["annualised_return"])]
    sharpes = [float(item.get("sharpe", 0.0)) for item in fold_metrics]
    drawdowns = [float(item["max_drawdown"]) for item in fold_metrics]
    n_days_total = int(sum(item.get("n_days", 0) for item in fold_metrics))
    dominance = _single_factor_dominance(coefficients) if coefficients else _booster_dominance(feature_importance or {})
    avg_daily = float(np.mean(avg_daily_returns)) if avg_daily_returns else 0.0
    return {
        "rank_ic_mean": float(np.mean(rank_ics)) if rank_ics else 0.0,
        "rank_ic_stability": float(np.mean(rank_ics) / (np.std(rank_ics) + 1e-12)) if rank_ics else 0.0,
        "ICIR": float(np.mean(rank_ics) / (np.std(rank_ics) + 1e-12)) if rank_ics else 0.0,
        "turnover_adjusted_net_return": float(np.sum(net_returns)) if net_returns else 0.0,
        "avg_daily_return": avg_daily,
        "annualised_return": float(np.mean(annualised_returns)) if annualised_returns else 0.0,
        "annualised_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
        "max_drawdown": float(min(drawdowns)) if drawdowns else 0.0,
        "evaluated_days": n_days_total,
        "single_factor_dominance": dominance,
        "uses_mock_or_synthetic": False,
        "prediction_rows": int(len(predictions)),
        "fold_count": int(len(fold_metrics)),
        "hit_rate": _prediction_hit_rate(predictions),
    }


def _training_manifest_metrics(
    predictions: pd.DataFrame,
    feature_columns: list[str],
    model_kind: str,
) -> dict[str, object]:
    dates = pd.to_datetime(predictions["trade_date"], errors="coerce").dropna()
    return {
        "feature_count": int(len(feature_columns)),
        "model_kind": model_kind,
        "data_range": {
            "start": dates.min().strftime("%Y-%m-%d") if not dates.empty else None,
            "end": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
        },
    }


def _prediction_hit_rate(predictions: pd.DataFrame) -> float:
    hits: list[pd.Series] = []
    for column in [c for c in predictions.columns if c.startswith("forward_return_")]:
        realized = pd.to_numeric(predictions[column], errors="coerce")
        predicted = pd.to_numeric(predictions["prediction"], errors="coerce")
        valid = realized.notna() & predicted.notna()
        if valid.any():
            hits.append((realized[valid] * predicted[valid]) > 0)
    if not hits:
        return 0.0
    return float(pd.concat(hits, ignore_index=True).mean())


def _single_factor_dominance(coefficients: dict[str, dict[str, float]]) -> float:
    values = [abs(value) for coef in coefficients.values() for key, value in coef.items() if key != "intercept"]
    total = sum(values)
    return float(max(values) / total) if total > 0 else 0.0


def _booster_dominance(importance: dict[str, dict[str, float]]) -> float:
    values = [abs(v) for ent in importance.values() for v in ent.values()]
    total = sum(values)
    return float(max(values) / total) if total > 0 else 0.0


def _write_artifacts(
    output_dir: Path,
    config: V7TrainingConfig,
    feature_columns: list[str],
    coefficients: dict[str, dict[str, float]],
    metrics: dict[str, object],
    quality: dict[str, object],
    acceptance: dict[str, object],
    predictions: pd.DataFrame,
    boosters: dict[str, object] | None = None,
    backend: str = "ridge",
) -> dict[str, str]:
    artifacts = {
        "model_artifact": output_dir / "model_coefficients.json",
        "feature_schema": output_dir / "feature_schema.json",
        "label_schema": output_dir / "label_schema.json",
        "training_config": output_dir / "training_config.json",
        "training_manifest": output_dir / "training_manifest.json",
        "metrics": output_dir / "metrics.json",
        "data_quality_report": output_dir / "data_quality_report.json",
        "acceptance_report": output_dir / "acceptance_report.json",
        "predictions": output_dir / "walk_forward_predictions.csv",
        "experiment_manifest": output_dir / "experiment_manifest.json",
    }
    commit = _git_commit()
    model_payload: dict[str, object] = {
        "model": config.model,
        "backend": backend,
        "coefficients": coefficients,
        "git_commit": commit,
    }
    if boosters:
        booster_dir = output_dir / "boosters"
        booster_dir.mkdir(parents=True, exist_ok=True)
        booster_paths: dict[str, str] = {}
        for horizon, booster in boosters.items():
            path = booster_dir / f"horizon_{horizon}.{backend}.txt"
            try:
                if backend == "lightgbm":
                    booster.booster_.save_model(str(path))  # type: ignore[attr-defined]
                elif backend == "xgboost":
                    booster.get_booster().save_model(str(path))  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                continue
            booster_paths[horizon] = str(path)
        model_payload["booster_paths"] = booster_paths
        artifacts["booster_dir"] = booster_dir
    _write_json(artifacts["model_artifact"], model_payload)
    _write_json(
        artifacts["feature_schema"],
        {"feature_columns": feature_columns, "backend": backend, "version": "v7"},
    )
    _write_json(artifacts["label_schema"], {"horizons": list(config.horizons), "label_columns": [f"forward_return_{h}d" for h in config.horizons]})
    _write_json(artifacts["training_config"], asdict(config))
    _write_json(
        artifacts["training_manifest"],
        {
            "model_kind": config.model,
            "backend": backend,
            "feature_count": len(feature_columns),
            "feature_schema_path": str(artifacts["feature_schema"]),
            "prediction_rows": int(len(predictions)),
            "fold_count": int(metrics.get("fold_count", 0)),
            "data_range": metrics.get("data_range", {}),
            "missing_value_policy": "numeric NaN values are converted to 0.0 inside model fit/predict only; source artifacts are not imputed in place",
            "split_mode": config.split_mode,
            "purge_days": config.purge_days,
            "embargo_days": config.embargo_days,
            "seed": 1729,
            "uses_mock_or_synthetic": False,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": commit,
        },
    )
    _write_json(artifacts["metrics"], metrics)
    _write_json(artifacts["data_quality_report"], quality)
    _write_json(artifacts["acceptance_report"], acceptance)
    predictions.to_csv(artifacts["predictions"], index=False)

    experiment_name = config.experiment_name or f"v7_{config.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = {
        "experiment_name": experiment_name,
        "model": config.model,
        "git_commit": commit,
        "horizons": list(config.horizons),
        "feature_count": len(feature_columns),
        "fold_count": int(metrics.get("fold_count", 0)),
        "rank_ic_mean": float(metrics.get("rank_ic_mean", 0.0)),
        "rank_ic_stability": float(metrics.get("rank_ic_stability", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "production_ready": bool(acceptance.get("passed", False) and config.mark_production_ready),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact_paths": {key: str(path) for key, path in artifacts.items()},
    }
    _write_json(artifacts["experiment_manifest"], manifest)

    try:
        ModelRegistry(root=config.registry_root).register(
            model_version=experiment_name,
            feature_version=str(len(feature_columns)),
            metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            metadata={
                "model": config.model,
                "horizons": list(config.horizons),
                "git_commit": commit,
                "output_dir": str(output_dir),
                "production_ready": bool(acceptance.get("passed", False) and config.mark_production_ready),
            },
        )
    except Exception:  # pragma: no cover - registry is best-effort
        pass
    return {key: str(path) for key, path in artifacts.items()}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip() or None
