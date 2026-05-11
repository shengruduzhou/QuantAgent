from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.factors.preprocessing import neutralize_by_date, zscore_by_date


def standardize_factors(frame: pd.DataFrame, factor_columns: list[str]) -> pd.DataFrame:
    return zscore_by_date(frame, factor_columns)


def neutralize_factors(
    frame: pd.DataFrame,
    factor_columns: list[str],
    exposure_columns: list[str] | None = None,
    industry_column: str | None = "industry",
) -> pd.DataFrame:
    data = frame.copy()
    for column in factor_columns:
        data = neutralize_by_date(
            data,
            column,
            exposure_columns=exposure_columns or (),
            industry_column=industry_column if industry_column is not None and industry_column in data.columns else None,
            output_column=column,
        )
    return data


def weights_by_rolling_icir(ic_frame: pd.DataFrame, window: int = 60) -> pd.Series:
    rolling_mean = ic_frame.rolling(window, min_periods=max(5, window // 5)).mean().iloc[-1]
    rolling_std = ic_frame.rolling(window, min_periods=max(5, window // 5)).std().iloc[-1]
    raw = (rolling_mean / rolling_std.replace(0.0, np.nan)).clip(lower=0.0).fillna(0.0)
    return _normalize_weights(raw)


def weights_by_decay(decay_frame: pd.DataFrame) -> pd.Series:
    raw = decay_frame.mean(axis=0).clip(lower=0.0).fillna(0.0)
    return _normalize_weights(raw)


def weights_by_turnover_adjusted_alpha(alpha: pd.Series, turnover: pd.Series, cost_penalty: float = 0.5) -> pd.Series:
    raw = (alpha - cost_penalty * turnover).clip(lower=0.0).fillna(0.0)
    return _normalize_weights(raw)


def weights_by_capacity(capacity: pd.Series) -> pd.Series:
    raw = np.log1p(capacity.clip(lower=0.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _normalize_weights(raw)


def bayesian_shrinkage_weights(raw_weights: pd.Series, prior_strength: float = 0.25) -> pd.Series:
    if raw_weights.empty:
        return raw_weights
    prior = pd.Series(1.0 / len(raw_weights), index=raw_weights.index)
    normalized = _normalize_weights(raw_weights.clip(lower=0.0))
    return _normalize_weights((1.0 - prior_strength) * normalized + prior_strength * prior)


def factor_crowding_penalty(correlation: pd.DataFrame, threshold: float = 0.7) -> pd.Series:
    if correlation.empty:
        return pd.Series(dtype=float)
    avg_abs = correlation.abs().where(~np.eye(len(correlation), dtype=bool)).mean(axis=1)
    return (1.0 - avg_abs.clip(lower=threshold).fillna(0.0)).clip(lower=0.0)


def composite_factor_score(
    frame: pd.DataFrame,
    factor_columns: list[str],
    weights: pd.Series | None = None,
    crowding_penalty: pd.Series | None = None,
    output_column: str = "composite_factor_score",
) -> pd.DataFrame:
    data = standardize_factors(frame, factor_columns)
    if weights is None:
        weights = pd.Series(1.0 / len(factor_columns), index=factor_columns)
    weights = weights.reindex(factor_columns).fillna(0.0)
    if crowding_penalty is not None:
        weights = weights * crowding_penalty.reindex(factor_columns).fillna(1.0)
    weights = _normalize_weights(weights)
    data[output_column] = data[factor_columns].mul(weights, axis=1).sum(axis=1, skipna=False)
    return data


def combine_weight_models(
    icir_weights: pd.Series,
    decay_weights: pd.Series,
    turnover_weights: pd.Series,
    capacity_weights: pd.Series,
    shrinkage: float = 0.25,
) -> pd.Series:
    all_index = icir_weights.index.union(decay_weights.index).union(turnover_weights.index).union(capacity_weights.index)
    raw = (
        0.40 * icir_weights.reindex(all_index).fillna(0.0)
        + 0.20 * decay_weights.reindex(all_index).fillna(0.0)
        + 0.25 * turnover_weights.reindex(all_index).fillna(0.0)
        + 0.15 * capacity_weights.reindex(all_index).fillna(0.0)
    )
    return bayesian_shrinkage_weights(raw, prior_strength=shrinkage)


def combine_with_model_gate(
    statistical_weights: pd.Series,
    model_gate: pd.Series,
    lifecycle_scores: pd.Series | None = None,
    crowding_penalty: pd.Series | None = None,
    gate_strength: float = 1.0,
) -> pd.Series:
    """Blend statistical factor weights with the lagged model factor gate.

    The model gate is intended to be produced after inference and consumed by
    the next tradable feature build. Callers are responsible for passing a
    point-in-time, lagged gate snapshot.
    """
    all_index = statistical_weights.index.union(model_gate.index)
    stat = _normalize_weights(statistical_weights.reindex(all_index).fillna(0.0).clip(lower=0.0))
    gate = _normalize_weights(model_gate.reindex(all_index).fillna(0.0).clip(lower=0.0))
    strength = float(np.clip(gate_strength, 0.0, 1.0))
    blended = (1.0 - strength) * stat + strength * gate
    if lifecycle_scores is not None:
        lifecycle = lifecycle_scores.reindex(all_index).fillna(1.0).clip(lower=0.0)
        blended = blended * lifecycle
    if crowding_penalty is not None:
        crowding = crowding_penalty.reindex(all_index).fillna(1.0).clip(lower=0.0)
        blended = blended * crowding
    return _normalize_weights(blended)


def _normalize_weights(raw: pd.Series) -> pd.Series:
    clean = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total = clean.sum()
    if total <= 0 and len(clean) > 0:
        return pd.Series(1.0 / len(clean), index=clean.index)
    return clean / total if total > 0 else clean
