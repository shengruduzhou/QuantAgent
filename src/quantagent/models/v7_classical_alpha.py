"""Classical ML baselines for V7 multi-horizon alpha.

The deep multi-horizon model in :mod:`quantagent.models.v7_deep_alpha`
runs an untrained tower MLP by default — useful as a deterministic seed
but not a substitute for a trained learner. This module ships two
explainable baselines that *can* learn from cross-sectional
``forward_return_{horizon}d`` labels at run time:

* :class:`RidgeAlphaModel` — ridge regression per horizon
  (NumPy closed-form solution, no scikit-learn dependency required).
* :class:`ElasticNetAlphaModel` — proximal gradient ElasticNet
  (numpy-only fallback so the model trains without ``sklearn``).

When ``scikit-learn`` is installed both models switch to its
optimised implementations transparently.

The models share the same predict signature as
:func:`predict_v7_deep_alpha` so they are drop-in replacements that
honour the V7 deep alpha contract (per-symbol :class:`MultiHorizonAlpha`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from quantagent.v7.schemas import (
    FactorApplicability,
    MultiHorizonAlpha,
    ThematicUniverseMember,
)


V7_CLASSICAL_HORIZONS: tuple[int, ...] = (1, 5, 20, 60, 120, 126)


@dataclass(frozen=True)
class ClassicalAlphaConfig:
    horizons: tuple[int, ...] = V7_CLASSICAL_HORIZONS
    model: str = "ridge"  # "ridge" | "elastic_net"
    alpha: float = 1.0  # L2 regularisation strength
    l1_ratio: float = 0.5  # ElasticNet mixing (1.0 = lasso, 0.0 = ridge)
    max_iter: int = 200
    learning_rate: float = 0.05
    min_train_rows: int = 30
    feature_columns: tuple[str, ...] = ()
    volatility_floor: float = 0.05
    interval_width_multiplier: float = 1.65
    confidence_floor: float = 0.05
    confidence_ceiling: float = 0.95
    fraud_penalty_weight: float = 0.30


@dataclass
class _LinearWeights:
    columns: tuple[str, ...]
    coef: np.ndarray
    intercept: float


class ClassicalAlphaModel:
    """Cross-sectional, per-horizon ridge / ElasticNet baseline."""

    def __init__(self, config: ClassicalAlphaConfig | None = None) -> None:
        self.config = config or ClassicalAlphaConfig()
        self._weights: dict[int, _LinearWeights] = {}

    def fit(self, feature_frame: pd.DataFrame, feature_columns: Sequence[str] | None = None) -> "ClassicalAlphaModel":
        if feature_frame is None or feature_frame.empty:
            return self
        columns = tuple(feature_columns) if feature_columns is not None else self.config.feature_columns
        if not columns:
            columns = _auto_feature_columns(feature_frame)
        if not columns:
            return self
        data = feature_frame.replace([np.inf, -np.inf], np.nan).copy()
        for horizon in self.config.horizons:
            label_column = f"forward_return_{horizon}d"
            if label_column not in data.columns:
                continue
            train = data[list(columns) + [label_column]].dropna()
            if len(train) < self.config.min_train_rows:
                continue
            x = train[list(columns)].to_numpy(dtype=np.float64)
            y = train[label_column].to_numpy(dtype=np.float64)
            if self.config.model == "elastic_net":
                coef, intercept = _fit_elastic_net(x, y, self.config)
            else:
                coef, intercept = _fit_ridge(x, y, self.config.alpha)
            self._weights[horizon] = _LinearWeights(columns, coef, intercept)
        return self

    def predict(
        self,
        feature_frame: pd.DataFrame,
        universe_members: list[ThematicUniverseMember],
        factor_applicability: Iterable[FactorApplicability] = (),
    ) -> dict[str, MultiHorizonAlpha]:
        if feature_frame is None or feature_frame.empty:
            return {}
        applicability = list(factor_applicability)
        latest = _latest_rows(feature_frame)
        members = {member.symbol: member for member in universe_members}
        output: dict[str, MultiHorizonAlpha] = {}
        for symbol, row in latest.iterrows():
            member = members.get(str(symbol))
            if member is None:
                continue
            horizon_scores: dict[int, float] = {}
            for horizon in self.config.horizons:
                horizon_scores[horizon] = self._predict_horizon(row, horizon, member)
            volatility = max(self.config.volatility_floor, _safe_float(row.get("volatility_20d", 0.20)))
            fraud_penalty = member.fraud_risk_score / 100.0
            risk_penalty = min(
                1.0,
                self.config.fraud_penalty_weight * fraud_penalty + max(0.0, volatility - 0.25),
            )
            expected = float(np.mean(list(horizon_scores.values()))) if horizon_scores else 0.0
            interval = self.config.interval_width_multiplier * volatility / np.sqrt(252.0 / 20.0)
            confidence = float(
                np.clip(
                    member.source_confidence * (1.0 - risk_penalty) * _factor_support(applicability, member),
                    self.config.confidence_floor,
                    self.config.confidence_ceiling,
                )
            )
            output[str(symbol)] = MultiHorizonAlpha(
                symbol=str(symbol),
                alpha_1d=horizon_scores.get(1, 0.0),
                alpha_5d=horizon_scores.get(5, 0.0),
                alpha_20d=horizon_scores.get(20, 0.0),
                alpha_60d=horizon_scores.get(60, 0.0),
                alpha_120d=horizon_scores.get(120, 0.0),
                alpha_126d=horizon_scores.get(126, 0.0),
                expected_return=expected,
                expected_excess_return=expected - _safe_float(row.get("benchmark_expected_return", 0.0)),
                volatility_forecast=volatility,
                downside_risk=max(0.0, volatility * 0.60 + risk_penalty * 0.20),
                confidence=confidence,
                conformal_confidence=float(np.clip(1.0 - interval, self.config.confidence_floor, self.config.confidence_ceiling)),
                prediction_interval_low=expected - interval,
                prediction_interval_high=expected + interval,
                rank_score=float(np.clip(_cross_section_rank(latest, symbol, "classical_alpha_seed") * 100.0, 0.0, 100.0)),
                regime_adjusted_score=float(np.clip(expected * 100.0 * confidence, -100.0, 100.0)),
                factor_contribution=_top_feature_contribution(self._weights, row, horizon_scores),
                evidence_contribution={member.theme: member.exposure_score / 100.0},
                risk_penalty=risk_penalty,
                final_alpha_score=float(np.clip(expected * 100.0 - risk_penalty * 25.0, -100.0, 100.0)),
            )
        return output

    def _predict_horizon(self, row: pd.Series, horizon: int, member: ThematicUniverseMember) -> float:
        weights = self._weights.get(horizon)
        if weights is None:
            return _seed_horizon_score(row, horizon, member)
        x = np.array([_safe_float(row.get(column, 0.0)) for column in weights.columns], dtype=np.float64)
        raw = float(np.dot(weights.coef, x) + weights.intercept)
        return float(np.tanh(raw))


def predict_v7_classical_alpha(
    feature_frame: pd.DataFrame,
    universe_members: list[ThematicUniverseMember],
    factor_applicability: Iterable[FactorApplicability] = (),
    config: ClassicalAlphaConfig | None = None,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, MultiHorizonAlpha]:
    """Walk-forward-friendly entrypoint: fit on all visible rows then predict the latest cross-section."""

    model = ClassicalAlphaModel(config)
    return model.fit(feature_frame, feature_columns).predict(feature_frame, universe_members, factor_applicability)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    try:
        from sklearn.linear_model import Ridge  # type: ignore
    except Exception:  # pragma: no cover - sklearn optional
        Ridge = None  # type: ignore[assignment]
    if Ridge is not None:
        model = Ridge(alpha=alpha)
        model.fit(x, y)
        return np.asarray(model.coef_, dtype=np.float64), float(model.intercept_)
    # Closed-form ridge solution: w = (X^T X + alpha I)^-1 X^T y
    mean_x = x.mean(axis=0)
    mean_y = float(y.mean())
    centred_x = x - mean_x
    centred_y = y - mean_y
    eye = alpha * np.eye(centred_x.shape[1])
    coef = np.linalg.solve(centred_x.T @ centred_x + eye, centred_x.T @ centred_y)
    intercept = mean_y - float(np.dot(coef, mean_x))
    return coef, intercept


def _fit_elastic_net(x: np.ndarray, y: np.ndarray, config: ClassicalAlphaConfig) -> tuple[np.ndarray, float]:
    try:
        from sklearn.linear_model import ElasticNet  # type: ignore
    except Exception:  # pragma: no cover - sklearn optional
        ElasticNet = None  # type: ignore[assignment]
    if ElasticNet is not None:
        model = ElasticNet(alpha=config.alpha, l1_ratio=config.l1_ratio, max_iter=config.max_iter)
        model.fit(x, y)
        return np.asarray(model.coef_, dtype=np.float64), float(model.intercept_)
    # numpy-only proximal-gradient fallback
    mean_x = x.mean(axis=0)
    mean_y = float(y.mean())
    centred_x = x - mean_x
    centred_y = y - mean_y
    coef = np.zeros(centred_x.shape[1])
    l1 = config.alpha * config.l1_ratio
    l2 = config.alpha * (1.0 - config.l1_ratio)
    n = max(1, centred_x.shape[0])
    for _ in range(config.max_iter):
        prediction = centred_x @ coef
        gradient = centred_x.T @ (prediction - centred_y) / n + l2 * coef
        coef = _soft_threshold(coef - config.learning_rate * gradient, config.learning_rate * l1)
    intercept = mean_y - float(np.dot(coef, mean_x))
    return coef, intercept


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _auto_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    excluded = {
        "trade_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "theme",
        "sector",
        "chain_node",
        "watchlist_status",
    }
    numeric = frame.select_dtypes("number").columns
    return tuple(
        column
        for column in numeric
        if column not in excluded and not column.startswith("forward_return_")
    )


def _seed_horizon_score(row: pd.Series, horizon: int, member: ThematicUniverseMember) -> float:
    short = _safe_float(row.get("momentum_20d"))
    medium = _safe_float(row.get("theme_strength")) / 100.0
    long = _safe_float(row.get("fundamental_score")) / 100.0
    fraud_penalty = member.fraud_risk_score / 100.0
    if horizon <= 5:
        raw = 0.55 * short + 0.25 * medium + 0.10 * long
    elif horizon <= 20:
        raw = 0.30 * short + 0.40 * medium + 0.20 * long
    else:
        raw = 0.10 * short + 0.30 * medium + 0.50 * long
    raw -= 0.25 * fraud_penalty
    return float(np.tanh(raw))


def _latest_rows(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "trade_date" in data.columns:
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
        data = data.sort_values(["symbol", "trade_date"]).groupby("symbol", sort=False).tail(1)
    data = data.set_index("symbol", drop=False)
    data["classical_alpha_seed"] = data.apply(_seed_score, axis=1)
    return data


def _seed_score(row: pd.Series) -> float:
    components = (
        _safe_float(row.get("theme_strength")) / 100.0,
        _safe_float(row.get("fundamental_score")) / 100.0,
        _safe_float(row.get("exposure_score")) / 100.0,
        _safe_float(row.get("momentum_20d")),
        _safe_float(row.get("policy_strength")) / 100.0,
    )
    return float(np.mean([np.tanh(value) for value in components]))


def _cross_section_rank(frame: pd.DataFrame, symbol: object, column: str) -> float:
    if column not in frame.columns or symbol not in frame.index:
        return 0.5
    ranks = frame[column].rank(pct=True)
    value = ranks.loc[symbol]
    if isinstance(value, pd.Series):
        value = float(value.iloc[0])
    return float(value) if np.isfinite(value) else 0.5


def _factor_support(applicability: Iterable[FactorApplicability], member: ThematicUniverseMember) -> float:
    matched: list[float] = []
    for item in applicability:
        if item.factor_lifecycle_stage not in {"production", "validation"}:
            continue
        if item.applicable_theme and member.theme not in item.applicable_theme:
            continue
        matched.append(max(0.0, item.rank_icir) + item.hit_rate)
    if not matched:
        return 0.80
    return float(np.clip(0.65 + float(np.mean(matched)) * 0.20, 0.65, 1.10))


def _top_feature_contribution(
    weights: dict[int, _LinearWeights],
    row: pd.Series,
    horizon_scores: dict[int, float],
) -> dict[str, float]:
    if not weights:
        return {"momentum_20d": _safe_float(row.get("momentum_20d")) * 0.5}
    longest_horizon = max(weights.keys())
    horizon_weights = weights[longest_horizon]
    contribution: dict[str, float] = {}
    for column, coef in zip(horizon_weights.columns, horizon_weights.coef):
        contribution[column] = float(coef * _safe_float(row.get(column, 0.0)))
    return contribution


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(numeric) or np.isinf(numeric):
        return default
    return numeric
