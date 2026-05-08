from __future__ import annotations

import numpy as np
import pandas as pd


def add_short_horizon_rule_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Build no-training mean-reversion and momentum-breakout signals."""
    data = frame.copy()
    data["mean_reversion_signal"] = _mean_reversion_score(data)
    data["momentum_breakout_signal"] = _momentum_breakout_score(data)
    data["short_rule_signal"] = (
        0.45 * data["mean_reversion_signal"]
        + 0.45 * data["momentum_breakout_signal"]
        + 0.10 * _liquidity_score(data)
        - 0.20 * _risk_penalty(data)
    ).clip(-1.0, 1.0)
    return data.replace([np.inf, -np.inf], np.nan)


def _mean_reversion_score(data: pd.DataFrame) -> pd.Series:
    below_band = data["close"] < data["bb_lower_20_2"]
    oversold = data["rsi_14d"] < 25
    improving_macd = data["macd_hist_delta"] > 0
    capitulation_volume = data["volume_zscore_20d"] > 1.5
    weak_trend_filter = data["adx_14d"].fillna(0.0) < 30
    raw = (
        below_band.astype(float)
        + oversold.astype(float)
        + improving_macd.astype(float)
        + capitulation_volume.astype(float)
        + weak_trend_filter.astype(float)
    ) / 5.0
    return raw.where(below_band & oversold, 0.0)


def _momentum_breakout_score(data: pd.DataFrame) -> pd.Series:
    breakout = data["close"] > data["donchian_high_20d"]
    macd_positive = data["macd_hist"] > 0
    rsi_constructive = data["rsi_14d"].between(50, 75)
    volatility_expansion = data["rv_ratio_5_20"] > 1.0
    trend_confirmed = data["plus_di_14d"] > data["minus_di_14d"]
    raw = (
        breakout.astype(float)
        + macd_positive.astype(float)
        + rsi_constructive.astype(float)
        + volatility_expansion.astype(float)
        + trend_confirmed.astype(float)
    ) / 5.0
    return raw.where(breakout & macd_positive, 0.0)


def _liquidity_score(data: pd.DataFrame) -> pd.Series:
    amount_z = data.get("amount_zscore_20d", pd.Series(0.0, index=data.index)).fillna(0.0)
    return (1.0 / (1.0 + np.exp(-amount_z))).clip(0.0, 1.0)


def _risk_penalty(data: pd.DataFrame) -> pd.Series:
    high_vol = (data["rv_ratio_5_20"].fillna(1.0) - 1.0).clip(lower=0.0)
    downtrend = (data["close"] < data["ma_20"]).astype(float) * (data["adx_14d"].fillna(0.0) > 30).astype(float)
    return (0.6 * high_vol + 0.4 * downtrend).clip(0.0, 1.0)
