from __future__ import annotations

import numpy as np

from quantagent.v7.schemas import (
    MarketRegime,
    MarketRegimeSnapshot,
    MultiHorizonAlpha,
    PortfolioPlan,
    SleeveType,
    TechnicalTimingPlan,
    ThematicUniverseMember,
    UniverseBucket,
)


def construct_v7_portfolio(
    universe: list[ThematicUniverseMember],
    alphas: dict[str, MultiHorizonAlpha],
    market: MarketRegimeSnapshot,
    timing: dict[str, TechnicalTimingPlan] | None = None,
    max_single_name_weight: float = 0.06,
    max_sector_weight: float = 0.30,
    max_theme_weight: float = 0.35,
    turnover_limit: float = 0.35,
) -> PortfolioPlan:
    timing = timing or {}
    sleeve_weights = _dynamic_sleeves(market)
    candidates = [member for member in universe if member.watchlist_status not in {UniverseBucket.EXCLUSION, UniverseBucket.WATCHLIST}]
    if not candidates:
        return PortfolioPlan(
            sleeve_weights=sleeve_weights,
            target_weights={},
            max_single_name_weight=max_single_name_weight,
            max_sector_weight=max_sector_weight,
            max_theme_weight=max_theme_weight,
            cash_weight=sleeve_weights[SleeveType.CASH_BUFFER],
            hedge_weight=sleeve_weights[SleeveType.HEDGE],
            turnover_limit=turnover_limit,
            position_reason={},
        )

    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for member in candidates:
        alpha = alphas.get(member.symbol)
        if alpha is None:
            continue
        timing_score = timing.get(member.symbol).timing_score if member.symbol in timing else 50.0
        horizon_score = _horizon_blend(member, alpha)
        score = (
            0.40 * horizon_score
            + 0.22 * member.fundamental_score
            + 0.15 * member.exposure_score
            + 0.08 * member.valuation_score
            + 0.08 * timing_score
            + 0.07 * member.liquidity_score
            - 0.35 * member.fraud_risk_score
            - 10.0 * alpha.risk_penalty
        )
        if member.watchlist_status == UniverseBucket.OPTIONAL_SATELLITE:
            score *= 0.65
        scores[member.symbol] = float(max(0.0, score))
        reasons[member.symbol] = (
            f"theme={member.theme}, bucket={member.watchlist_status.value}, "
            f"alpha={horizon_score:.1f}, fundamental={member.fundamental_score:.1f}, "
            f"fraud={member.fraud_risk_score:.1f}"
        )

    risky_capital = max(0.0, 1.0 - sleeve_weights[SleeveType.CASH_BUFFER] - sleeve_weights[SleeveType.HEDGE])
    raw_total = sum(scores.values())
    target_weights: dict[str, float] = {}
    if raw_total > 0:
        for member in candidates:
            score = scores.get(member.symbol, 0.0)
            if score <= 0:
                continue
            bucket_cap = 0.025 if member.watchlist_status == UniverseBucket.OPTIONAL_SATELLITE else max_single_name_weight
            fraud_cap = max(0.0, bucket_cap * (1.0 - max(member.fraud_risk_score - 50.0, 0.0) / 70.0))
            liquidity_cap = bucket_cap * float(np.clip(member.liquidity_score / 60.0, 0.20, 1.0))
            cap = min(max_single_name_weight, bucket_cap, fraud_cap, liquidity_cap)
            target_weights[member.symbol] = min(cap, risky_capital * score / raw_total)
    scale = min(1.0, risky_capital / max(sum(target_weights.values()), 1e-12))
    target_weights = {symbol: float(weight * scale) for symbol, weight in sorted(target_weights.items()) if weight > 0}
    return PortfolioPlan(
        sleeve_weights=sleeve_weights,
        target_weights=target_weights,
        max_single_name_weight=max_single_name_weight,
        max_sector_weight=max_sector_weight,
        max_theme_weight=max_theme_weight,
        cash_weight=sleeve_weights[SleeveType.CASH_BUFFER],
        hedge_weight=sleeve_weights[SleeveType.HEDGE],
        turnover_limit=turnover_limit,
        position_reason={symbol: reasons[symbol] for symbol in target_weights if symbol in reasons},
    )


def _dynamic_sleeves(market: MarketRegimeSnapshot) -> dict[SleeveType, float]:
    base = {
        SleeveType.LONG_FUNDAMENTAL: 0.25,
        SleeveType.MEDIUM_THEME: 0.30,
        SleeveType.SHORT_EVENT: 0.10,
        SleeveType.SECTOR_ROTATION: 0.10,
        SleeveType.HEDGE: 0.05,
        SleeveType.CASH_BUFFER: 0.20,
    }
    if market.market_regime in {MarketRegime.RISK_OFF, MarketRegime.BEAR, MarketRegime.LIQUIDITY_CRUNCH}:
        base[SleeveType.CASH_BUFFER] += 0.20
        base[SleeveType.HEDGE] += 0.08
        base[SleeveType.SHORT_EVENT] *= 0.50
        base[SleeveType.MEDIUM_THEME] *= 0.70
    elif market.market_regime in {MarketRegime.RISK_ON, MarketRegime.BULL, MarketRegime.POLICY_DRIVEN}:
        base[SleeveType.MEDIUM_THEME] += 0.08
        base[SleeveType.LONG_FUNDAMENTAL] += 0.05
        base[SleeveType.CASH_BUFFER] -= 0.08
    base[SleeveType.CASH_BUFFER] += 0.20 * market.drawdown_risk + 0.10 * (1.0 - market.liquidity_score)
    base[SleeveType.HEDGE] += 0.15 * market.hedge_need_score
    total = sum(max(0.0, value) for value in base.values())
    return {sleeve: float(max(0.0, value) / total) for sleeve, value in base.items()}


def _horizon_blend(member: ThematicUniverseMember, alpha: MultiHorizonAlpha) -> float:
    if member.watchlist_status == UniverseBucket.CORE_BENEFICIARY:
        return 100.0 * (0.15 * alpha.alpha_20d + 0.35 * alpha.alpha_60d + 0.35 * alpha.alpha_120d + 0.15 * alpha.alpha_126d)
    if member.watchlist_status == UniverseBucket.STRONG_CORRELATION:
        return 100.0 * (0.20 * alpha.alpha_5d + 0.35 * alpha.alpha_20d + 0.30 * alpha.alpha_60d + 0.15 * alpha.alpha_120d)
    return 100.0 * (0.45 * alpha.alpha_1d + 0.35 * alpha.alpha_5d + 0.20 * alpha.alpha_20d)
