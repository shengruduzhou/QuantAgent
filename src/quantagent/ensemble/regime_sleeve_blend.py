"""Regime- and uncertainty-aware blending for short/mid/long alpha sleeves."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Mapping

import numpy as np
import pandas as pd


DEFAULT_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "bull_expansion": {"short": 0.45, "mid": 0.45, "long": 0.10},
    "bull_consolidation": {"short": 0.30, "mid": 0.50, "long": 0.20},
    "normal": {"short": 0.25, "mid": 0.50, "long": 0.25},
    "caution": {"short": 0.15, "mid": 0.45, "long": 0.40},
    "bear_capitulation": {"short": 0.10, "mid": 0.30, "long": 0.60},
    "crisis": {"short": 0.00, "mid": 0.20, "long": 0.80},
}


@dataclass(frozen=True)
class RegimeSleeveBlendConfig:
    date_col: str = "trade_date"
    symbol_col: str = "symbol"
    sleeve_score_columns: Mapping[str, str] = field(
        default_factory=lambda: {
            "short": "short_score",
            "mid": "mid_score",
            "long": "long_score",
        }
    )
    sleeve_uncertainty_columns: Mapping[str, str] = field(
        default_factory=lambda: {
            "short": "short_uncertainty",
            "mid": "mid_uncertainty",
            "long": "long_uncertainty",
        }
    )
    regime_weights: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_REGIME_WEIGHTS.items()}
    )
    uncertainty_penalty: float = 0.75
    max_single_sleeve_weight: float = 0.80
    cash_score_threshold: float = 0.55


def _rank_pct(frame: pd.DataFrame, column: str, date_col: str) -> pd.Series:
    return frame.groupby(date_col)[column].rank(pct=True, method="average")


def _normalise_weights(weights: Mapping[str, float], available: set[str], cap: float) -> dict[str, float]:
    raw = {name: max(0.0, float(value)) for name, value in weights.items() if name in available}
    if not raw:
        return {}
    total = sum(raw.values())
    if total <= 0:
        raw = {name: 1.0 for name in available}
        total = float(len(raw))
    normalised = {name: value / total for name, value in raw.items()}
    # Iterative water-fill cap.  This avoids a single sleeve dominating when
    # one or two sleeves are missing on a date.
    for _ in range(10):
        over = {name: value for name, value in normalised.items() if value > cap}
        if not over:
            break
        fixed = {name: cap for name in over}
        remaining_names = [name for name in normalised if name not in over]
        remaining_mass = max(0.0, 1.0 - cap * len(over))
        remaining_total = sum(normalised[name] for name in remaining_names)
        if remaining_names and remaining_total > 0:
            normalised = {
                **fixed,
                **{
                    name: remaining_mass * normalised[name] / remaining_total
                    for name in remaining_names
                },
            }
        else:
            normalised = fixed
            break
    total = sum(normalised.values())
    return {name: value / total for name, value in normalised.items()} if total > 0 else {}


def blend_sleeves(
    predictions: pd.DataFrame,
    regime_by_date: pd.Series | Mapping[object, str],
    *,
    config: RegimeSleeveBlendConfig | None = None,
) -> pd.DataFrame:
    """Blend per-date cross-sectional sleeve ranks with uncertainty shrinkage."""
    cfg = config or RegimeSleeveBlendConfig()
    required = {cfg.date_col, cfg.symbol_col}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"sleeve predictions missing columns: {sorted(missing)}")
    out = predictions.copy()
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col], errors="coerce")
    out = out.dropna(subset=[cfg.date_col, cfg.symbol_col])
    if out.duplicated([cfg.date_col, cfg.symbol_col]).any():
        raise ValueError("duplicate trade_date/symbol rows in sleeve predictions")

    regime_series = pd.Series(regime_by_date)
    regime_series.index = pd.to_datetime(regime_series.index, errors="coerce")
    out["market_regime"] = out[cfg.date_col].map(regime_series).fillna("normal").astype(str)

    available_sleeves = {
        sleeve for sleeve, column in cfg.sleeve_score_columns.items() if column in out.columns
    }
    if not available_sleeves:
        raise ValueError("no configured sleeve score columns are present")

    for sleeve in available_sleeves:
        score_col = cfg.sleeve_score_columns[sleeve]
        score = pd.to_numeric(out[score_col], errors="coerce")
        out[f"{sleeve}_rank"] = _rank_pct(
            out.assign(**{score_col: score}), score_col, cfg.date_col
        ).fillna(0.5)
        uncertainty_col = cfg.sleeve_uncertainty_columns.get(sleeve, "")
        if uncertainty_col and uncertainty_col in out.columns:
            uncertainty = pd.to_numeric(out[uncertainty_col], errors="coerce").fillna(1.0).clip(0, 1)
        else:
            uncertainty = pd.Series(1.0, index=out.index)
        out[f"{sleeve}_confidence_multiplier"] = (
            1.0 - cfg.uncertainty_penalty * uncertainty
        ).clip(0.0, 1.0)

    composite = pd.Series(0.0, index=out.index, dtype=float)
    effective_weight_sum = pd.Series(0.0, index=out.index, dtype=float)
    for regime, regime_rows in out.groupby("market_regime"):
        base = cfg.regime_weights.get(regime, cfg.regime_weights.get("normal", {}))
        weights = _normalise_weights(base, available_sleeves, cfg.max_single_sleeve_weight)
        idx = regime_rows.index
        for sleeve, weight in weights.items():
            confidence = out.loc[idx, f"{sleeve}_confidence_multiplier"]
            effective = float(weight) * confidence
            composite.loc[idx] += effective * out.loc[idx, f"{sleeve}_rank"]
            effective_weight_sum.loc[idx] += effective
            out.loc[idx, f"effective_weight_{sleeve}"] = effective
    out["composite_score"] = np.where(
        effective_weight_sum > 1e-12,
        composite / effective_weight_sum,
        0.5,
    )
    out["blend_confidence"] = effective_weight_sum.clip(0.0, 1.0)
    out["cash_preferred"] = out["blend_confidence"] < cfg.cash_score_threshold
    return out


def fit_regime_weights_grid(
    training_predictions: pd.DataFrame,
    regime_by_date: pd.Series,
    *,
    forward_return_col: str,
    config: RegimeSleeveBlendConfig | None = None,
    grid_step: float = 0.25,
    top_k: int = 20,
) -> dict[str, dict[str, float]]:
    """Low-dimensional training-only grid fit with worst-subperiod objective.

    This is deliberately small and interpretable.  It must be called inside an
    outer selection protocol; it is not an alternative to nested CV/PBO/DSR.
    """
    cfg = config or RegimeSleeveBlendConfig()
    if forward_return_col not in training_predictions.columns:
        raise ValueError(f"missing {forward_return_col}")
    sleeves = [
        sleeve for sleeve, column in cfg.sleeve_score_columns.items()
        if column in training_predictions.columns
    ]
    if len(sleeves) < 2:
        raise ValueError("at least two sleeves are required to fit blend weights")
    values = np.arange(0.0, 1.0 + 1e-9, grid_step)
    candidates: list[dict[str, float]] = []
    for weights in product(values, repeat=len(sleeves)):
        if abs(sum(weights) - 1.0) > 1e-9:
            continue
        if max(weights) > cfg.max_single_sleeve_weight + 1e-9:
            continue
        candidates.append(dict(zip(sleeves, map(float, weights))))
    if not candidates:
        raise ValueError("weight grid is empty")

    data = training_predictions.copy()
    data[cfg.date_col] = pd.to_datetime(data[cfg.date_col], errors="coerce")
    regimes = pd.Series(regime_by_date)
    regimes.index = pd.to_datetime(regimes.index, errors="coerce")
    data["_regime"] = data[cfg.date_col].map(regimes).fillna("normal")
    fitted: dict[str, dict[str, float]] = {}
    for regime, subset in data.groupby("_regime"):
        if subset[cfg.date_col].nunique() < 20:
            fitted[str(regime)] = dict(cfg.regime_weights.get(str(regime), cfg.regime_weights["normal"]))
            continue
        thirds = np.array_split(sorted(subset[cfg.date_col].dropna().unique()), 3)
        best_candidate: dict[str, float] | None = None
        best_objective = -np.inf
        for candidate in candidates:
            scores = pd.Series(0.0, index=subset.index)
            for sleeve, weight in candidate.items():
                column = cfg.sleeve_score_columns[sleeve]
                scores += weight * _rank_pct(subset, column, cfg.date_col)
            ranked = subset.assign(_score=scores).sort_values(
                [cfg.date_col, "_score"], ascending=[True, False]
            )
            ranked["_rank"] = ranked.groupby(cfg.date_col).cumcount()
            daily = ranked[ranked["_rank"] < top_k].groupby(cfg.date_col)[forward_return_col].mean()
            subperiod_means = [
                float(daily.reindex(pd.DatetimeIndex(chunk)).dropna().mean())
                for chunk in thirds if len(chunk)
            ]
            objective = min(subperiod_means) if subperiod_means else -np.inf
            if objective > best_objective + 1e-12:
                best_objective = objective
                best_candidate = candidate
        fitted[str(regime)] = best_candidate or dict(cfg.regime_weights["normal"])
    return fitted
