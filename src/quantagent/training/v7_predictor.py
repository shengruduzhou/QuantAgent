"""V7 alpha prediction helpers.

Loads a previously trained V7 alpha artifact directory and emits a
``predictions.parquet/csv`` from a feature frame. Supports both:

* Classical models persisted by :func:`run_v7_training_experiment`
  (``ridge`` / ``elastic_net`` coefficients in ``model_coefficients.json``
  and the optional native booster files under ``boosters/``).
* Deep alpha models persisted by :class:`V7DeepAlphaTrainer`
  (``deep_alpha_state.json`` + ``deep_alpha_config.json``).
* FT-Transformer artifacts persisted as ``ft_transformer.pt``.

The predictor never re-fits anything; it is a deterministic forward
pass meant for ``predict-alpha-v7`` and the prediction-to-target-weights
pipeline. Live trading remains disabled; the output is just a frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class V7PredictionResult:
    predictions: pd.DataFrame
    model_kind: str
    feature_columns: tuple[str, ...]
    horizons: tuple[int, ...]
    artifact_dir: str


def _resolve_artifact_dir(path: str | Path) -> Path:
    artifact = Path(path)
    if artifact.is_file():
        artifact = artifact.parent
    return artifact


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_v7_alpha(
    model_dir: str | Path,
    feature_frame: pd.DataFrame,
    *,
    primary_horizon: int | None = None,
) -> V7PredictionResult:
    """Run inference against the artifact directory at ``model_dir``."""

    if feature_frame is None or feature_frame.empty:
        raise ValueError("predict_v7_alpha: feature frame is empty")
    artifact = _resolve_artifact_dir(model_dir)
    if not artifact.exists():
        raise FileNotFoundError(f"V7 model artifact directory not found: {artifact}")

    deep_state = artifact / "deep_alpha_state.json"
    ft_state = artifact / "ft_transformer.pt"
    classic_state = artifact / "model_coefficients.json"
    if deep_state.exists():
        return _predict_deep(artifact, deep_state, feature_frame, primary_horizon)
    if ft_state.exists():
        return _predict_ft_transformer(artifact, feature_frame, primary_horizon)
    if classic_state.exists():
        return _predict_classic(artifact, classic_state, feature_frame, primary_horizon)
    raise FileNotFoundError(
        f"V7 model artifact directory {artifact} contains neither "
        f"deep_alpha_state.json, ft_transformer.pt nor model_coefficients.json"
    )


def _predict_deep(
    artifact_dir: Path,
    state_path: Path,
    feature_frame: pd.DataFrame,
    primary_horizon: int | None,
) -> V7PredictionResult:
    from quantagent.training.v7_deep_trainer import V7DeepAlphaTrainer, V7DeepAlphaTrainerConfig

    config_payload = _read_json(artifact_dir / "deep_alpha_config.json") if (artifact_dir / "deep_alpha_config.json").exists() else {}
    trainer = V7DeepAlphaTrainer(V7DeepAlphaTrainerConfig(**{k: v for k, v in config_payload.items() if k in V7DeepAlphaTrainerConfig.__dataclass_fields__}))
    trainer.load(state_path)
    pred = trainer.predict(feature_frame)
    horizons = tuple(int(h) for h in trainer.state.horizons)  # type: ignore[union-attr]
    feature_columns = tuple(trainer.state.feature_columns)  # type: ignore[union-attr]
    if "prediction" not in pred.columns:
        primary = primary_horizon if primary_horizon in horizons else horizons[0]
        pred["prediction"] = pred[f"alpha_{primary}d"]
    return V7PredictionResult(
        predictions=pred.reset_index(drop=True),
        model_kind="deep",
        feature_columns=feature_columns,
        horizons=horizons,
        artifact_dir=str(artifact_dir),
    )


def _predict_classic(
    artifact_dir: Path,
    state_path: Path,
    feature_frame: pd.DataFrame,
    primary_horizon: int | None,
) -> V7PredictionResult:
    state = _read_json(state_path)
    backend = str(state.get("backend", state.get("model", "ridge")))
    coefficients: dict[str, dict[str, float]] = state.get("coefficients", {})  # type: ignore[assignment]
    booster_paths: dict[str, str] = state.get("booster_paths", {})  # type: ignore[assignment]
    schema_path = artifact_dir / "feature_schema.json"
    schema = _read_json(schema_path) if schema_path.exists() else {}
    feature_columns = tuple(schema.get("feature_columns", [])) if isinstance(schema, dict) else ()
    if not feature_columns and coefficients:
        # Reconstruct feature columns from the first horizon entry, minus 'intercept'.
        first = next(iter(coefficients.values()))
        feature_columns = tuple(name for name in first.keys() if name != "intercept")
    missing = [c for c in feature_columns if c not in feature_frame.columns]
    if missing:
        raise ValueError(f"predict_v7_alpha: feature frame missing columns {missing}")
    base_columns = [c for c in ("symbol", "trade_date") if c in feature_frame.columns]
    output = feature_frame[base_columns].copy()
    horizons: list[int] = []
    if booster_paths and backend in {"lightgbm", "xgboost"}:
        for horizon_str, path_str in sorted(booster_paths.items(), key=lambda kv: int(kv[0])):
            horizon = int(horizon_str)
            horizons.append(horizon)
            output[f"alpha_{horizon}d"] = _booster_predict(backend, Path(path_str), feature_frame, feature_columns)
    else:
        x_values = np.nan_to_num(feature_frame[list(feature_columns)].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        for horizon_str, entry in sorted(coefficients.items(), key=lambda kv: int(kv[0])):
            horizon = int(horizon_str)
            horizons.append(horizon)
            intercept = float(entry.get("intercept", 0.0))
            coef = np.asarray([float(entry.get(name, 0.0)) for name in feature_columns], dtype=float)
            output[f"alpha_{horizon}d"] = x_values @ coef + intercept
    if not horizons:
        raise ValueError("classic V7 model artifact has no usable horizons")
    primary = primary_horizon if primary_horizon in horizons else horizons[0]
    output["prediction"] = output[f"alpha_{primary}d"]
    return V7PredictionResult(
        predictions=output.reset_index(drop=True),
        model_kind=f"classic:{backend}",
        feature_columns=feature_columns,
        horizons=tuple(horizons),
        artifact_dir=str(artifact_dir),
    )


def _predict_ft_transformer(
    artifact_dir: Path,
    feature_frame: pd.DataFrame,
    primary_horizon: int | None,
) -> V7PredictionResult:
    from quantagent.training.ft_transformer_trainer import predict_ft_transformer_artifact

    result = predict_ft_transformer_artifact(
        artifact_dir,
        feature_frame,
        primary_horizon=primary_horizon,
    )
    return V7PredictionResult(
        predictions=result.predictions,
        model_kind="ft_transformer",
        feature_columns=result.feature_columns,
        horizons=result.horizons,
        artifact_dir=result.artifact_dir,
    )


def _booster_predict(backend: str, path: Path, feature_frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    matrix = np.nan_to_num(feature_frame[list(feature_columns)].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if backend == "lightgbm":
        import lightgbm as lgb  # type: ignore

        booster = lgb.Booster(model_file=str(path))
        return np.asarray(booster.predict(matrix), dtype=float)
    if backend == "xgboost":
        import xgboost as xgb  # type: ignore

        booster = xgb.Booster()
        booster.load_model(str(path))
        return np.asarray(booster.predict(xgb.DMatrix(matrix)), dtype=float)
    raise ValueError(f"unsupported booster backend: {backend}")


__all__ = ["V7PredictionResult", "predict_v7_alpha"]
