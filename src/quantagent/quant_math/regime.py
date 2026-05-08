from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class MarketRegime(str, Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    HIGH_VOLATILITY = "high_volatility"
    RANGE_BOUND = "range_bound"
    LIQUIDITY_CRISIS = "liquidity_crisis"


@dataclass(frozen=True)
class RegimeThresholds:
    bull_trend_min: float = 0.03
    bear_trend_max: float = -0.05
    high_volatility_min: float = 0.025
    liquidity_drop_max: float = -0.25
    drawdown_crisis_max: float = -0.12


REGIME_MULTIPLIER = {
    MarketRegime.BULL_TREND: 1.2,
    MarketRegime.RANGE_BOUND: 0.8,
    MarketRegime.HIGH_VOLATILITY: 0.4,
    MarketRegime.BEAR_TREND: 0.3,
    MarketRegime.LIQUIDITY_CRISIS: 0.0,
}


def detect_regime(row: pd.Series, thresholds: RegimeThresholds | None = None) -> MarketRegime:
    thresholds = thresholds or RegimeThresholds()
    if (
        row.get("drawdown", 0.0) <= thresholds.drawdown_crisis_max
        or row.get("liquidity_change", 0.0) <= thresholds.liquidity_drop_max
    ):
        return MarketRegime.LIQUIDITY_CRISIS
    if row.get("market_vol", 0.0) >= thresholds.high_volatility_min:
        return MarketRegime.HIGH_VOLATILITY
    if row.get("market_trend", 0.0) >= thresholds.bull_trend_min:
        return MarketRegime.BULL_TREND
    if row.get("market_trend", 0.0) <= thresholds.bear_trend_max:
        return MarketRegime.BEAR_TREND
    return MarketRegime.RANGE_BOUND


def apply_regime_multiplier(alpha: pd.Series, regime: MarketRegime) -> pd.Series:
    return alpha * REGIME_MULTIPLIER[regime]
