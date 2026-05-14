"""Long-short capital allocator that splits AUM across investment horizons.

The user holds positions across three horizons:

  short_event  : 1-5 days, driven by news/momentum/flow, T+1 makes this risky
  medium_theme : 5-60 days, driven by theme strength + sector rotation
  long_fundamental : 60-126 days, driven by fundamentals + valuation + policy

The allocator decides how much capital each sleeve should hold given (a) the
market regime, (b) the average confidence of long-horizon vs short-horizon
signals, and (c) the hedge_need_score. When long-horizon signals are strong
and confident the allocator shifts weight from short to long; when the regime
is risk-off it raises cash and the hedge sleeve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from quantagent.v7.schemas import (
    HedgeDecision,
    MarketRegime,
    MarketRegimeSnapshot,
    MultiHorizonAlpha,
    SleeveType,
    ThematicUniverseMember,
)


@dataclass(frozen=True)
class LongShortAllocatorConfig:
    base_sleeves: dict[SleeveType, float] = field(
        default_factory=lambda: {
            SleeveType.LONG_FUNDAMENTAL: 0.25,
            SleeveType.MEDIUM_THEME: 0.30,
            SleeveType.SHORT_EVENT: 0.10,
            SleeveType.SECTOR_ROTATION: 0.10,
            SleeveType.HEDGE: 0.05,
            SleeveType.CASH_BUFFER: 0.20,
        }
    )
    min_long_share: float = 0.05
    max_long_share: float = 0.55
    min_short_share: float = 0.0
    max_short_share: float = 0.20
    min_cash_share: float = 0.05
    max_cash_share: float = 0.60
    max_total_invested: float = 0.95
    long_horizon_confidence_threshold: float = 0.45
    short_horizon_confidence_threshold: float = 0.55
    hedge_scale: float = 0.30
    risk_off_cash_bonus: float = 0.15


@dataclass(frozen=True)
class LongShortAllocation:
    sleeve_weights: dict[SleeveType, float]
    long_horizon_signal: float
    short_horizon_signal: float
    average_long_confidence: float
    average_short_confidence: float
    rationale: str


def allocate_long_short(
    alphas: dict[str, MultiHorizonAlpha],
    universe_members: Iterable[ThematicUniverseMember],
    market: MarketRegimeSnapshot,
    hedge: HedgeDecision,
    config: LongShortAllocatorConfig | None = None,
) -> LongShortAllocation:
    config = config or LongShortAllocatorConfig()
    if not alphas:
        return LongShortAllocation(
            sleeve_weights=dict(config.base_sleeves),
            long_horizon_signal=0.0,
            short_horizon_signal=0.0,
            average_long_confidence=0.0,
            average_short_confidence=0.0,
            rationale="no_alpha_signals_use_base_sleeves",
        )
    long_signals: list[float] = []
    short_signals: list[float] = []
    long_confidence: list[float] = []
    short_confidence: list[float] = []
    for alpha in alphas.values():
        long_signals.append(0.5 * alpha.alpha_60d + 0.3 * alpha.alpha_120d + 0.2 * alpha.alpha_126d)
        short_signals.append(0.6 * alpha.alpha_5d + 0.4 * alpha.alpha_1d)
        long_confidence.append(alpha.confidence * (1.0 - alpha.risk_penalty))
        short_confidence.append(alpha.conformal_confidence * (1.0 - alpha.risk_penalty))
    long_horizon_signal = float(np.clip(np.mean(long_signals), -1.0, 1.0))
    short_horizon_signal = float(np.clip(np.mean(short_signals), -1.0, 1.0))
    avg_long_conf = float(np.clip(np.mean(long_confidence), 0.0, 1.0))
    avg_short_conf = float(np.clip(np.mean(short_confidence), 0.0, 1.0))

    sleeves = dict(config.base_sleeves)

    long_share = sleeves[SleeveType.LONG_FUNDAMENTAL]
    medium_share = sleeves[SleeveType.MEDIUM_THEME]
    short_share = sleeves[SleeveType.SHORT_EVENT]
    sector_share = sleeves[SleeveType.SECTOR_ROTATION]
    hedge_share = sleeves[SleeveType.HEDGE]
    cash_share = sleeves[SleeveType.CASH_BUFFER]

    long_strength = max(0.0, long_horizon_signal) * avg_long_conf
    short_strength = max(0.0, short_horizon_signal) * avg_short_conf
    if avg_long_conf >= config.long_horizon_confidence_threshold:
        long_share += long_strength * 0.20
        medium_share += long_strength * 0.10
        short_share -= long_strength * 0.10
        cash_share -= long_strength * 0.05
    if avg_short_conf >= config.short_horizon_confidence_threshold:
        short_share += short_strength * 0.10
        cash_share -= short_strength * 0.05
    sector_share += float(np.clip(market.sector_rotation_score_average() if hasattr(market, "sector_rotation_score_average") else _mean_sector_score(market), 0.0, 0.20)) * 0.05

    hedge_weight_input = float(getattr(hedge, "hedge_weight", 0.0)) if hedge is not None else 0.0
    hedge_share = min(0.30, hedge_share + hedge_weight_input * config.hedge_scale)
    if market.market_regime in {MarketRegime.RISK_OFF, MarketRegime.BEAR, MarketRegime.LIQUIDITY_CRUNCH}:
        cash_share += config.risk_off_cash_bonus
        long_share *= 0.70
        medium_share *= 0.85
        short_share *= 0.50

    long_share = float(np.clip(long_share, config.min_long_share, config.max_long_share))
    medium_share = float(np.clip(medium_share, 0.0, 0.45))
    short_share = float(np.clip(short_share, config.min_short_share, config.max_short_share))
    sector_share = float(np.clip(sector_share, 0.0, 0.20))
    hedge_share = float(np.clip(hedge_share, 0.0, 0.30))
    cash_share = float(np.clip(cash_share, config.min_cash_share, config.max_cash_share))

    invested = long_share + medium_share + short_share + sector_share + hedge_share
    if invested + cash_share > 1.0:
        cash_share = max(config.min_cash_share, 1.0 - invested)
    if invested > config.max_total_invested:
        scale = config.max_total_invested / invested
        long_share *= scale
        medium_share *= scale
        short_share *= scale
        sector_share *= scale
        hedge_share *= scale
        invested = long_share + medium_share + short_share + sector_share + hedge_share
        cash_share = 1.0 - invested

    sleeve_weights = {
        SleeveType.LONG_FUNDAMENTAL: long_share,
        SleeveType.MEDIUM_THEME: medium_share,
        SleeveType.SHORT_EVENT: short_share,
        SleeveType.SECTOR_ROTATION: sector_share,
        SleeveType.HEDGE: hedge_share,
        SleeveType.CASH_BUFFER: cash_share,
    }
    rationale = (
        f"regime={market.market_regime.value}; risk_off={market.risk_off_score:.2f}; "
        f"long_signal={long_horizon_signal:+.2f}@{avg_long_conf:.2f}; "
        f"short_signal={short_horizon_signal:+.2f}@{avg_short_conf:.2f}; "
        f"hedge_need={float(getattr(hedge, 'hedge_need_score', 0.0)):.2f}; cash={cash_share:.2f}"
    )
    return LongShortAllocation(
        sleeve_weights=sleeve_weights,
        long_horizon_signal=long_horizon_signal,
        short_horizon_signal=short_horizon_signal,
        average_long_confidence=avg_long_conf,
        average_short_confidence=avg_short_conf,
        rationale=rationale,
    )


def _mean_sector_score(market: MarketRegimeSnapshot) -> float:
    if not market.sector_rotation_score:
        return 0.0
    return float(np.mean(list(market.sector_rotation_score.values())))
