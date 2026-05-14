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
    current_weights: dict[str, float] | None = None,
    max_single_name_weight: float = 0.06,
    max_sector_weight: float = 0.30,
    max_theme_weight: float = 0.35,
    turnover_limit: float = 0.35,
    sleeve_weights_override: dict[SleeveType, float] | None = None,
) -> PortfolioPlan:
    """Build the V7 portfolio plan.

    ``sleeve_weights_override`` lets upstream components (the walk-forward
    sleeve allocator or the long-short allocator) bind the sleeve
    distribution that the portfolio must respect. When omitted, the
    builder falls back to the deterministic ``_dynamic_sleeves`` prior.
    """

    timing = timing or {}
    if sleeve_weights_override:
        sleeve_weights = _normalise_sleeve_override(sleeve_weights_override)
    else:
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
            sector_weights={},
            theme_weights={},
            constraint_notes=("no_eligible_universe_members",),
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
        score = float(max(0.0, score))
        if score >= scores.get(member.symbol, -1.0):
            scores[member.symbol] = score
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
    notes: list[str] = []
    member_by_symbol = {member.symbol: member for member in candidates}
    target_weights, sector_notes = _enforce_group_cap(target_weights, member_by_symbol, "sector", max_sector_weight)
    target_weights, theme_notes = _enforce_group_cap(target_weights, member_by_symbol, "theme", max_theme_weight)
    notes.extend(sector_notes)
    notes.extend(theme_notes)
    if current_weights:
        target_weights, turnover_notes = _enforce_turnover_limit(current_weights, target_weights, turnover_limit)
        target_weights = {symbol: weight for symbol, weight in target_weights.items() if symbol in member_by_symbol}
        notes.extend(turnover_notes)
    scale = min(1.0, risky_capital / max(sum(target_weights.values()), 1e-12))
    target_weights = {symbol: float(weight * scale) for symbol, weight in sorted(target_weights.items()) if weight > 0}
    sector_weights = _group_weights(target_weights, member_by_symbol, "sector")
    theme_weights = _group_weights(target_weights, member_by_symbol, "theme")
    sleeve_target_weights = _sleeve_target_weights(target_weights, member_by_symbol, alphas, market)
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
        sector_weights=sector_weights,
        theme_weights=theme_weights,
        sleeve_target_weights=sleeve_target_weights,
        constraint_notes=tuple(notes),
    )


def _normalise_sleeve_override(override: dict[SleeveType, float]) -> dict[SleeveType, float]:
    """Coerce an external sleeve weight dict into the canonical six-sleeve frame.

    Missing sleeves default to 0.0; the resulting weights are renormalised so
    they sum to 1.0. Cash buffer is guaranteed a minimum of 5% to keep the
    portfolio in a runnable state when an upstream allocator forgets it.
    """

    canonical = [
        SleeveType.LONG_FUNDAMENTAL,
        SleeveType.MEDIUM_THEME,
        SleeveType.SHORT_EVENT,
        SleeveType.SECTOR_ROTATION,
        SleeveType.HEDGE,
        SleeveType.CASH_BUFFER,
    ]
    weights: dict[SleeveType, float] = {sleeve: 0.0 for sleeve in canonical}
    for key, value in override.items():
        if isinstance(key, str):
            try:
                sleeve = SleeveType(key)
            except ValueError:
                continue
        else:
            sleeve = key
        try:
            weights[sleeve] = max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    if weights[SleeveType.CASH_BUFFER] < 0.05:
        weights[SleeveType.CASH_BUFFER] = 0.05
    total = sum(weights.values())
    if total <= 0.0:
        return _dynamic_sleeves_default()
    return {sleeve: float(weights[sleeve] / total) for sleeve in canonical}


def _dynamic_sleeves_default() -> dict[SleeveType, float]:
    return {
        SleeveType.LONG_FUNDAMENTAL: 0.25,
        SleeveType.MEDIUM_THEME: 0.30,
        SleeveType.SHORT_EVENT: 0.10,
        SleeveType.SECTOR_ROTATION: 0.10,
        SleeveType.HEDGE: 0.05,
        SleeveType.CASH_BUFFER: 0.20,
    }


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


def _sleeve_target_weights(
    weights: dict[str, float],
    members: dict[str, ThematicUniverseMember],
    alphas: dict[str, MultiHorizonAlpha],
    market: MarketRegimeSnapshot,
) -> dict[SleeveType, dict[str, float]]:
    sleeves: dict[SleeveType, dict[str, float]] = {}
    for symbol, weight in weights.items():
        member = members.get(symbol)
        if member is None:
            continue
        sleeve = _classify_sleeve(member, alphas.get(symbol), market)
        sleeves.setdefault(sleeve, {})[symbol] = float(weight)
    return sleeves


def _classify_sleeve(
    member: ThematicUniverseMember,
    alpha: MultiHorizonAlpha | None,
    market: MarketRegimeSnapshot,
) -> SleeveType:
    if alpha is not None:
        short_score = max(alpha.alpha_1d, alpha.alpha_5d)
        long_score = max(alpha.alpha_60d, alpha.alpha_120d, alpha.alpha_126d)
        if short_score > long_score * 1.35 and member.membership_ttl_days <= 20:
            return SleeveType.SHORT_EVENT
    if member.watchlist_status == UniverseBucket.CORE_BENEFICIARY and member.membership_ttl_days >= 60:
        return SleeveType.LONG_FUNDAMENTAL
    sector_score = market.sector_rotation_score.get(member.sector or "", 0.0)
    if sector_score >= 0.65 and member.watchlist_status != UniverseBucket.CORE_BENEFICIARY:
        return SleeveType.SECTOR_ROTATION
    if member.watchlist_status == UniverseBucket.OPTIONAL_SATELLITE:
        return SleeveType.SHORT_EVENT
    return SleeveType.MEDIUM_THEME


def _enforce_group_cap(
    weights: dict[str, float],
    members: dict[str, ThematicUniverseMember],
    group_field: str,
    cap: float,
) -> tuple[dict[str, float], list[str]]:
    capped = dict(weights)
    notes: list[str] = []
    groups = _group_weights(capped, members, group_field)
    for group, total in groups.items():
        if group == "unknown" or total <= cap or total <= 0:
            continue
        scale = cap / total
        for symbol, member in members.items():
            if symbol in capped and _member_group(member, group_field) == group:
                capped[symbol] *= scale
        notes.append(f"{group_field}_cap_applied:{group}:{cap:.4f}")
    return capped, notes


def _enforce_turnover_limit(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    turnover_limit: float,
) -> tuple[dict[str, float], list[str]]:
    symbols = set(current_weights) | set(target_weights)
    turnover = 0.5 * sum(abs(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0)) for symbol in symbols)
    if turnover <= turnover_limit or turnover <= 0:
        return dict(target_weights), []
    blend = turnover_limit / turnover
    adjusted = {
        symbol: float(current_weights.get(symbol, 0.0) + (target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0)) * blend)
        for symbol in symbols
    }
    return {symbol: weight for symbol, weight in adjusted.items() if weight > 0}, [f"turnover_limit_applied:{turnover_limit:.4f}"]


def _group_weights(weights: dict[str, float], members: dict[str, ThematicUniverseMember], group_field: str) -> dict[str, float]:
    groups: dict[str, float] = {}
    for symbol, weight in weights.items():
        group = _member_group(members.get(symbol), group_field)
        groups[group] = groups.get(group, 0.0) + float(weight)
    return groups


def _member_group(member: ThematicUniverseMember | None, group_field: str) -> str:
    if member is None:
        return "unknown"
    if group_field == "sector":
        return member.sector or "unknown"
    if group_field == "theme":
        return member.theme or "unknown"
    return "unknown"
