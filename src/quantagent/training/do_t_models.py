"""Tabular model training for intraday Do-T EV signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantagent.execution.intraday_ev_engine import IntradayModelSignals


CLASSIFICATION_TARGETS = {
    "model_sell_high_success": "label_sell_high_success",
    "model_buyback_now": "label_buyback_now_success",
    "model_buy_low_success": "label_buy_low_success",
    "model_sell_after_buy_success": "label_sell_after_buy_success",
    "model_failure_risk": "label_sell_high_fail_new_high",
    "model_breakdown_risk": "label_buy_low_fail_breakdown",
    "model_eod_restore": "label_sell_high_eod_restore",
}

REGRESSION_TARGETS = {
    # gross edge (before cost): decide_ev subtracts cost itself, so feeding net
    # edge here would double-count the round-trip cost.
    "model_sell_high_edge": "label_sell_high_gross_edge_bps",
    "model_buy_low_edge": "label_buy_low_gross_edge_bps",
    "model_buyback_edge": "label_buyback_now_edge_bps",
    "model_wait_extra_edge": "label_wait_extra_edge_bps",
    "model_miss_rebound_risk": "label_miss_rebound_risk",
    "model_adverse_after_sell": "label_adverse_excursion_after_sell",
    "model_adverse_after_buy": "label_adverse_excursion_after_buy",
}


@dataclass
class TrainedDoTModels:
    classifiers: dict[str, Any] = field(default_factory=dict)
    regressors: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    backend: str = ""
    calibration_method: str = "isotonic"
    diagnostics: dict[str, Any] = field(default_factory=dict)


def train_do_t_models(
    dataset: pd.DataFrame,
    *,
    feature_columns: list[str],
    backend: str = "lightgbm",
    calibration_method: str = "isotonic",
    allow_sklearn_fallback: bool = True,
    random_state: int = 42,
) -> TrainedDoTModels:
    """Train success, edge, and failure-risk models for the EV engine.

    Preferred backends are LightGBM, CatBoost, or XGBoost.  If they are not
    installed, callers may allow the scikit-learn fallback for local smoke
    tests; production training should install one of the preferred libraries.
    """
    if dataset is None or dataset.empty:
        raise ValueError("dataset is empty")
    missing_features = [c for c in feature_columns if c not in dataset.columns]
    if missing_features:
        raise ValueError(f"missing feature columns: {missing_features}")
    _ensure_sklearn()
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import brier_score_loss
    from sklearn.pipeline import make_pipeline

    X = dataset[feature_columns].replace([np.inf, -np.inf], np.nan)
    models = TrainedDoTModels(feature_columns=list(feature_columns), backend=backend, calibration_method=calibration_method)
    estimator_factory = _estimator_factory(backend, allow_sklearn_fallback, random_state)

    diagnostics: dict[str, Any] = {"classifiers": {}, "regressors": {}}
    for name, target in CLASSIFICATION_TARGETS.items():
        if target not in dataset.columns:
            continue
        y = pd.to_numeric(dataset[target], errors="coerce")
        mask = y.notna()
        yv = y[mask].astype(int)
        if yv.empty:
            continue
        min_class_count = int(yv.value_counts().min()) if not yv.empty else 0
        if yv.nunique() < 2 or len(yv) < 30 or min_class_count < 2:
            clf = make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="prior"))
            clf.fit(X.loc[mask], yv)
            diagnostics["classifiers"][name] = {"rows": int(len(yv)), "positive_rate": float(yv.mean()), "calibrated": False}
        else:
            base = estimator_factory("classifier") or HistGradientBoostingClassifier(random_state=random_state)
            clf = make_pipeline(SimpleImputer(strategy="median"), base)
            calibrated = CalibratedClassifierCV(clf, method=calibration_method, cv=min(3, min_class_count))
            calibrated.fit(X.loc[mask], yv)
            prob = _positive_probability(calibrated, X.loc[mask])
            diagnostics["classifiers"][name] = {
                "rows": int(len(yv)),
                "positive_rate": float(yv.mean()),
                "brier": float(brier_score_loss(yv, prob)),
                "calibrated": True,
            }
            clf = calibrated
        models.classifiers[name] = clf

    for name, target in REGRESSION_TARGETS.items():
        if target not in dataset.columns:
            continue
        y = pd.to_numeric(dataset[target], errors="coerce")
        mask = y.notna()
        yv = y[mask].astype(float)
        if yv.empty:
            continue
        if len(yv) < 30:
            reg = make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="median"))
        else:
            base = estimator_factory("regressor") or HistGradientBoostingRegressor(random_state=random_state)
            reg = make_pipeline(SimpleImputer(strategy="median"), base)
        reg.fit(X.loc[mask], yv)
        pred = reg.predict(X.loc[mask])
        diagnostics["regressors"][name] = {
            "rows": int(len(yv)),
            "mean_target_bps": float(yv.mean()),
            "mae_bps": float(np.mean(np.abs(pred - yv.to_numpy()))),
        }
        models.regressors[name] = reg

    models.diagnostics = diagnostics
    return models


def predict_model_signals(models: TrainedDoTModels, rows: pd.DataFrame) -> list[IntradayModelSignals]:
    if rows is None or rows.empty:
        return []
    X = rows[models.feature_columns].replace([np.inf, -np.inf], np.nan)
    cols: dict[str, np.ndarray] = {}
    for name, clf in models.classifiers.items():
        if hasattr(clf, "predict_proba"):
            cols[name] = _positive_probability(clf, X)
        else:
            cols[name] = clf.predict(X).astype(float)
    for name, reg in models.regressors.items():
        cols[name] = reg.predict(X).astype(float)

    out: list[IntradayModelSignals] = []
    n = len(X)
    zeros = np.zeros(n, dtype=float)

    def pred(name: str, default: float = 0.0) -> np.ndarray:
        if name in cols:
            return np.nan_to_num(cols[name].astype(float), nan=default, posinf=default, neginf=default)
        if default == 0.0:
            return zeros
        return np.full(n, float(default), dtype=float)

    for i in range(n):
        adverse_after_sell = max(0.0, float(pred("model_adverse_after_sell", 20.0)[i]))
        adverse_after_buy = float(pred("model_adverse_after_buy", -20.0)[i])
        out.append(
            IntradayModelSignals(
                p_sell_high_success=float(pred("model_sell_high_success")[i]),
                expected_sell_high_gain_bps=max(0.0, float(pred("model_sell_high_edge")[i])),
                p_fail_new_high=float(pred("model_failure_risk")[i]),
                expected_chase_loss_bps=adverse_after_sell,
                p_buyback_now=float(pred("model_buyback_now")[i]),
                expected_buyback_edge_bps=float(pred("model_buyback_edge")[i]),
                wait_extra_edge_bps=float(pred("model_wait_extra_edge")[i]),
                miss_rebound_risk_bps=max(0.0, float(pred("model_miss_rebound_risk")[i])),
                p_buy_low_success=float(pred("model_buy_low_success")[i]),
                expected_buy_low_gain_bps=max(0.0, float(pred("model_buy_low_edge")[i])),
                p_fail_breakdown=float(pred("model_breakdown_risk")[i]),
                expected_breakdown_loss_bps=max(0.0, -adverse_after_buy),
                p_sell_after_buy_success=float(pred("model_sell_after_buy_success")[i]),
                expected_sell_after_buy_edge_bps=max(0.0, float(pred("model_buy_low_edge")[i])),
                p_eod_restore=float(pred("model_eod_restore")[i]),
                risk_score=max(float(pred("model_failure_risk")[i]), float(pred("model_breakdown_risk")[i])),
                model_version=models.backend,
            )
        )
    return out


def save_models(models: TrainedDoTModels, path: str | Path) -> Path:
    """Persist trained EV models for live/forward inference."""
    import joblib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, p)
    return p


def load_models(path: str | Path) -> TrainedDoTModels:
    import joblib

    return joblib.load(Path(path))


def save_training_diagnostics(models: TrainedDoTModels, output_dir: str | Path) -> Path:
    import json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "do_t_model_diagnostics.json"
    path.write_text(json.dumps(models.diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _estimator_factory(backend: str, allow_sklearn_fallback: bool, random_state: int):
    backend = backend.lower()
    if backend == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor

            return lambda kind: (
                LGBMClassifier(random_state=random_state, n_estimators=200) if kind == "classifier"
                else LGBMRegressor(random_state=random_state, n_estimators=200)
            )
        except ImportError:
            pass
    if backend == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor

            return lambda kind: (
                CatBoostClassifier(random_seed=random_state, verbose=False) if kind == "classifier"
                else CatBoostRegressor(random_seed=random_state, verbose=False)
            )
        except ImportError:
            pass
    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor

            return lambda kind: (
                XGBClassifier(random_state=random_state, eval_metric="logloss") if kind == "classifier"
                else XGBRegressor(random_state=random_state)
            )
        except ImportError:
            pass
    if allow_sklearn_fallback:
        return lambda kind: None
    raise ImportError(
        f"{backend} is not installed. Install lightgbm, catboost, or xgboost, "
        "or pass allow_sklearn_fallback=True for smoke tests."
    )


def _ensure_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError("scikit-learn is required for probability calibration; install quantagent[training].") from exc


def _positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps.get("model"), "classes_", None)
    if classes is not None:
        classes_list = list(classes)
        if 1 in classes_list:
            return np.asarray(proba[:, classes_list.index(1)], dtype=float)
        return np.zeros(len(X), dtype=float)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return np.asarray(proba[:, 1], dtype=float)
    return np.zeros(len(X), dtype=float)


__all__ = [
    "CLASSIFICATION_TARGETS",
    "REGRESSION_TARGETS",
    "TrainedDoTModels",
    "load_models",
    "predict_model_signals",
    "save_models",
    "save_training_diagnostics",
    "train_do_t_models",
]
