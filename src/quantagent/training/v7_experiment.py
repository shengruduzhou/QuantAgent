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
    ft_max_epochs: int = 60
    ft_batch_size: int = 8192
    ft_d_token: int = 128
    ft_n_blocks: int = 5
    ft_n_heads: int = 8
    ft_dates_per_step: int = 8
    ft_attention_dropout: float = 0.10
    ft_ffn_dropout: float = 0.10
    ft_weight_decay: float = 1e-4
    ft_use_amp: bool = True
    ft_device: str = "auto"
    require_gpu: bool = False
    run_synth_ablation: bool = False
    emit_ic_decay_diagnostics: bool = True


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
            fold_frame = test[["trade_date", "symbol", label_column]].copy()
            fold_frame["horizon"] = horizon
            fold_frame["prediction"] = prediction
            fold_frame["sample_role"] = "validation"
            fold_frame["fold_id"] = fold.fold_id
            fold_frame["train_start"] = fold.train_dates[0]
            fold_frame["train_end"] = fold.train_dates[1]
            fold_frame["valid_start"] = fold.valid_dates[0]
            fold_frame["valid_end"] = fold.valid_dates[1]
            all_predictions.append(fold_frame)
            fold_metrics.append(_fold_metrics(fold_frame, label_column, fold.fold_id, horizon, config.cost_bps))
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
    output_dir = Path(config.output_dir)
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
        trainer = FTTransformerTrainer(
            FTTransformerTrainerConfig(
                horizons=used_horizons,
                d_token=config.ft_d_token,
                n_blocks=config.ft_n_blocks,
                n_heads=config.ft_n_heads,
                attention_dropout=config.ft_attention_dropout,
                ffn_dropout=config.ft_ffn_dropout,
                learning_rate=config.learning_rate,
                weight_decay=config.ft_weight_decay,
                batch_size=config.ft_batch_size,
                max_epochs=config.ft_max_epochs,
                dates_per_step=config.ft_dates_per_step,
                use_amp=config.ft_use_amp,
                device=config.ft_device,
                require_gpu=config.require_gpu,
                feature_columns=tuple(feature_columns),
                output_dir=str(fold_dir),
            )
        )
        fold_artifacts = trainer.fit_and_save(train, validation_dataset=valid)
        pred = predict_ft_transformer_artifact(fold_dir, valid, device=fold_artifacts.device)
        for horizon in used_horizons:
            label_column = f"forward_return_{horizon}d"
            alpha_column = f"alpha_{horizon}d"
            fold_frame = valid[["trade_date", "symbol", label_column]].copy()
            fold_frame["horizon"] = horizon
            fold_frame["prediction"] = pred.predictions[alpha_column].to_numpy(dtype=float)
            fold_frame["sample_role"] = "validation"
            fold_frame["fold_id"] = fold.fold_id
            fold_frame["train_start"] = fold.train_dates[0]
            fold_frame["train_end"] = fold.train_dates[1]
            fold_frame["valid_start"] = fold.valid_dates[0]
            fold_frame["valid_end"] = fold.valid_dates[1]
            all_predictions.append(fold_frame)
            fold_metrics.append(_fold_metrics(fold_frame, label_column, fold.fold_id, horizon, config.cost_bps))
    if not all_predictions:
        raise ValueError("FT-Transformer training produced no out-of-sample predictions")

    final_trainer = FTTransformerTrainer(
        FTTransformerTrainerConfig(
            horizons=used_horizons,
            d_token=config.ft_d_token,
            n_blocks=config.ft_n_blocks,
            n_heads=config.ft_n_heads,
            attention_dropout=config.ft_attention_dropout,
            ffn_dropout=config.ft_ffn_dropout,
            learning_rate=config.learning_rate,
            weight_decay=config.ft_weight_decay,
            batch_size=config.ft_batch_size,
            max_epochs=config.ft_max_epochs,
            use_amp=config.ft_use_amp,
            device=config.ft_device,
            require_gpu=config.require_gpu,
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


def _fold_metrics(frame: pd.DataFrame, label_column: str, fold: int, horizon: int, cost_bps: float) -> dict[str, object]:
    by_date_ic = frame.groupby("trade_date").apply(_rank_ic(label_column)).dropna()
    returns = frame.groupby("trade_date").apply(_long_short_return(label_column)).fillna(0.0)
    turnover_cost = cost_bps / 10_000.0
    net_returns = returns - turnover_cost
    nav = (1.0 + net_returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0 if not nav.empty else pd.Series(dtype=float)
    n_days = max(int(len(net_returns)), 1)
    avg_daily_return = float(net_returns.mean()) if not net_returns.empty else 0.0
    annualised_return = (
        float((1.0 + avg_daily_return) ** 252 - 1.0) if avg_daily_return > -1 else float("nan")
    )
    annualised_vol = float(net_returns.std(ddof=1) * (252 ** 0.5)) if n_days > 1 else 0.0
    sharpe = float(avg_daily_return / (net_returns.std(ddof=1) + 1e-12) * (252 ** 0.5)) if n_days > 1 else 0.0
    return {
        "fold": fold,
        "horizon": horizon,
        "rank_ic_mean": float(by_date_ic.mean()) if not by_date_ic.empty else 0.0,
        "net_return": float(net_returns.sum()) if not net_returns.empty else 0.0,
        "avg_daily_return": avg_daily_return,
        "annualised_return": annualised_return,
        "annualised_vol": annualised_vol,
        "sharpe": sharpe,
        "n_days": n_days,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
    }


def _long_short_return(label_column: str):
    def inner(frame: pd.DataFrame) -> float:
        if len(frame) < 2:
            return 0.0
        ranks = frame["prediction"].rank(pct=True) - 0.5
        gross = ranks.abs().sum()
        if gross <= 0:
            return 0.0
        weights = ranks / gross
        return float((weights * frame[label_column]).sum())

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
