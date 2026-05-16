"""Trainable V7 alpha experiment pipeline with purged walk-forward validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from quantagent.data.v7_label_builder import V7_LABEL_HORIZONS
from quantagent.data.v7_quality_gates import (
    V7DataQualityGateConfig,
    V7ModelAcceptanceGateConfig,
    evaluate_adverse_regime,
    evaluate_data_quality_gates,
    evaluate_model_acceptance_gates,
)
from quantagent.quant_math.purged_cv import PurgedKFoldConfig, purged_kfold_split
from quantagent.training.model_registry import ModelRegistry


SUPPORTED_MODELS: tuple[str, ...] = ("ridge", "elastic_net", "lightgbm", "xgboost")


@dataclass(frozen=True)
class V7TrainingConfig:
    horizons: tuple[int, ...] = V7_LABEL_HORIZONS
    model: str = "ridge"
    alpha: float = 1.0
    l1_ratio: float = 0.5
    min_train_rows: int = 100
    n_splits: int = 4
    embargo_pct: float = 0.02
    cost_bps: float = 12.0
    output_dir: str = "artifacts/v7_alpha"
    paper_report_path: str | None = None
    mark_production_ready: bool = False
    feature_columns: tuple[str, ...] = ()
    registry_root: str = "artifacts/v7_alpha/registry"
    experiment_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    allow_model_downgrade: bool = False
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05


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
        label_end = _label_end_times(horizon_data, horizon)
        splits = purged_kfold_split(
            horizon_data["trade_date"],
            label_end,
            PurgedKFoldConfig(n_splits=min(config.n_splits, max(2, len(horizon_data) // 2)), embargo_pct=config.embargo_pct),
        )
        last_artifact = None
        for fold, (train_idx, test_idx) in enumerate(splits):
            if len(train_idx) < max(10, len(feature_columns)):
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
            all_predictions.append(fold_frame)
            fold_metrics.append(_fold_metrics(fold_frame, label_column, fold, horizon, config.cost_bps))
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
    adverse_report = evaluate_adverse_regime(prediction_frame, label_column="forward_return_1d")
    metrics["adverse_regime_passed"] = bool(adverse_report.get("passed", False))
    metrics["adverse_regime_report"] = adverse_report
    metrics["backend"] = backend
    metrics["model_requested"] = config.model
    metrics["model_downgraded"] = backend != config.model
    acceptance = evaluate_model_acceptance_gates(
        metrics,
        V7ModelAcceptanceGateConfig(require_paper_report=config.mark_production_ready),
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
    return V7TrainingResult(
        status=status,
        output_dir=str(output_dir),
        metrics=metrics,
        data_quality_report=quality.to_dict(),
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
    return tuple(
        column
        for column in frame.select_dtypes("number").columns
        if not column.startswith("forward_return_") and not column.startswith("label_end_") and column not in excluded
    )


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
    return {
        "fold": fold,
        "horizon": horizon,
        "rank_ic_mean": float(by_date_ic.mean()) if not by_date_ic.empty else 0.0,
        "net_return": float(net_returns.sum()) if not net_returns.empty else 0.0,
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
    drawdowns = [float(item["max_drawdown"]) for item in fold_metrics]
    dominance = _single_factor_dominance(coefficients) if coefficients else _booster_dominance(feature_importance or {})
    return {
        "rank_ic_mean": float(np.mean(rank_ics)) if rank_ics else 0.0,
        "rank_ic_stability": float(np.mean(rank_ics) / (np.std(rank_ics) + 1e-12)) if rank_ics else 0.0,
        "turnover_adjusted_net_return": float(np.sum(net_returns)) if net_returns else 0.0,
        "max_drawdown": float(min(drawdowns)) if drawdowns else 0.0,
        "single_factor_dominance": dominance,
        "uses_mock_or_synthetic": False,
        "prediction_rows": int(len(predictions)),
        "fold_count": int(len(fold_metrics)),
    }


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
