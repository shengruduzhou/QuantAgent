"""Intrinsic-valuation engine.

Combines DCF (FCFF), DDM (Gordon growth), relative multiples, and an
asset-floor (P/B) into a composite fair value with explicit margin of
safety. All inputs are point-in-time financial statement rows and
market state. Confidence is haircut by fraud_risk_score so a stock with
elevated accounting risk gets a smaller margin of safety and a lower
valuation_score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.fundamental.valuation import (
    DCFInputs,
    dcf_equity_value,
    dcf_intrinsic_value_per_share,
    margin_of_safety as _margin_of_safety_pct,
    reverse_dcf_implied_growth,
)


@dataclass(frozen=True)
class IntrinsicValuationConfig:
    default_terminal_growth: float = 0.025
    default_wacc_floor: float = 0.07
    default_wacc_ceiling: float = 0.15
    default_forecast_years: int = 5
    relative_multiples: tuple[str, ...] = ("pe_ttm", "pb", "ps_ttm", "ev_ebitda")
    relative_multiple_target_percentile: float = 0.40
    industry_peer_minimum: int = 3
    margin_of_safety_floor: float = -0.50
    margin_of_safety_ceiling: float = 0.80
    fraud_confidence_haircut_threshold: float = 60.0
    fraud_confidence_haircut_strength: float = 0.60
    asset_floor_weight_when_negative_earnings: float = 0.30


@dataclass(frozen=True)
class IntrinsicValuationReport:
    symbol: str
    as_of_date: str
    current_price: float | None
    market_cap: float | None
    dcf_value_per_share: float | None
    ddm_value_per_share: float | None
    relative_value_per_share: float | None
    asset_value_per_share: float | None
    fair_value_per_share: float | None
    method_weights: dict[str, float]
    margin_of_safety_pct: float
    valuation_score: float
    bubble_risk_score: float
    industry_valuation_percentile: float | None
    history_valuation_percentile: float | None
    reverse_dcf_implied_growth: float | None
    confidence: float
    key_assumptions: dict[str, float] = field(default_factory=dict)
    flags: tuple[str, ...] = ()
    rationale: str = ""


def value_universe(
    fundamentals: pd.DataFrame,
    market_state: pd.DataFrame,
    as_of_date: str,
    config: IntrinsicValuationConfig | None = None,
) -> list[IntrinsicValuationReport]:
    config = config or IntrinsicValuationConfig()
    if fundamentals is None or fundamentals.empty:
        return []
    latest = (
        fundamentals.assign(report_date=pd.to_datetime(fundamentals.get("report_date"), errors="coerce"))
        .sort_values(["symbol", "report_date"])
        .groupby("symbol", sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    market = _market_lookup(market_state)
    industry_percentiles = _industry_percentiles(latest, config.relative_multiples, config)
    history = _history_percentiles(fundamentals, config.relative_multiples)
    reports: list[IntrinsicValuationReport] = []
    for _, row in latest.iterrows():
        symbol = str(row["symbol"])
        market_row = market.get(symbol, {})
        report = _value_symbol(row, market_row, as_of_date, config, industry_percentiles.get(symbol), history.get(symbol))
        reports.append(report)
    return reports


def _value_symbol(
    row: pd.Series,
    market_row: dict,
    as_of_date: str,
    config: IntrinsicValuationConfig,
    industry_percentile: float | None,
    history_percentile: float | None,
) -> IntrinsicValuationReport:
    symbol = str(row["symbol"])
    flags: list[str] = []
    shares = _safe_float(row.get("total_shares") or market_row.get("total_shares") or row.get("shares_outstanding"))
    price = _safe_float(row.get("price") or row.get("close") or market_row.get("close") or market_row.get("price"))
    market_cap = _safe_float(row.get("market_cap") or market_row.get("market_cap"))
    if shares <= 0 and market_cap > 0 and price > 0:
        shares = market_cap / price
    if market_cap <= 0 and shares > 0 and price > 0:
        market_cap = shares * price

    dcf_value = _maybe_dcf(row, shares, config, flags)
    ddm_value = _maybe_ddm(row, shares, config, flags)
    relative_value = _maybe_relative(row, config, industry_percentile, flags)
    asset_value = _maybe_asset_floor(row, shares, flags)
    fair_value, weights = _composite_fair_value(
        dcf=dcf_value,
        ddm=ddm_value,
        relative=relative_value,
        asset=asset_value,
        config=config,
        flags=flags,
        earnings_negative=bool(_safe_float(row.get("net_income")) <= 0),
    )

    margin_of_safety_pct = 0.0
    if fair_value is not None and price > 0:
        margin_of_safety_pct = float(np.clip((fair_value / price) - 1.0, config.margin_of_safety_floor, config.margin_of_safety_ceiling))

    bubble_risk = _bubble_risk_score(price, fair_value, industry_percentile, history_percentile, flags)
    valuation_score = _valuation_score(margin_of_safety_pct, bubble_risk, history_percentile, industry_percentile)
    reverse_growth = _maybe_reverse_dcf(row, market_cap, config, flags)
    confidence = _confidence_after_haircut(row, fair_value, config, flags)

    rationale = (
        f"as_of={as_of_date}; symbol={symbol}; price={price:.2f}; "
        f"fair_value={fair_value if fair_value is not None else 'n/a'}; "
        f"mos={margin_of_safety_pct:.2%}; bubble={bubble_risk:.2f}; "
        f"weights={'+'.join(f'{key}={value:.2f}' for key, value in weights.items())}"
    )
    return IntrinsicValuationReport(
        symbol=symbol,
        as_of_date=as_of_date,
        current_price=price if price > 0 else None,
        market_cap=market_cap if market_cap > 0 else None,
        dcf_value_per_share=dcf_value,
        ddm_value_per_share=ddm_value,
        relative_value_per_share=relative_value,
        asset_value_per_share=asset_value,
        fair_value_per_share=fair_value,
        method_weights=dict(weights),
        margin_of_safety_pct=margin_of_safety_pct,
        valuation_score=valuation_score,
        bubble_risk_score=bubble_risk,
        industry_valuation_percentile=industry_percentile,
        history_valuation_percentile=history_percentile,
        reverse_dcf_implied_growth=reverse_growth,
        confidence=confidence,
        key_assumptions=_key_assumptions(row, config),
        flags=tuple(sorted(set(flags))),
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------


def _maybe_dcf(row: pd.Series, shares: float, config: IntrinsicValuationConfig, flags: list[str]) -> float | None:
    fcff = _safe_float(row.get("fcff") or row.get("free_cash_flow"))
    if fcff == 0:
        ocf = _safe_float(row.get("operating_cash_flow"))
        capex = _safe_float(row.get("capex"))
        fcff = ocf + capex
    growth = _bounded_growth(row)
    wacc = _safe_wacc(row, config)
    terminal_growth = min(config.default_terminal_growth, wacc - 0.005)
    if fcff <= 0 or shares <= 0 or wacc <= terminal_growth:
        flags.append("dcf_inputs_unavailable")
        return None
    net_debt = _safe_float(row.get("net_debt"))
    try:
        return dcf_intrinsic_value_per_share(
            DCFInputs(
                fcff=fcff,
                growth_rate=growth,
                terminal_growth_rate=terminal_growth,
                wacc=wacc,
                years=config.default_forecast_years,
                net_debt=net_debt,
                shares_outstanding=shares,
            )
        )
    except (ValueError, ZeroDivisionError):
        flags.append("dcf_failed")
        return None


def _maybe_ddm(row: pd.Series, shares: float, config: IntrinsicValuationConfig, flags: list[str]) -> float | None:
    dividend = _safe_float(row.get("dividend_per_share") or row.get("dividend"))
    payout = _safe_float(row.get("payout_ratio"))
    if dividend <= 0 and payout > 0:
        eps = _safe_float(row.get("eps")) or (_safe_float(row.get("net_income")) / shares if shares > 0 else 0.0)
        dividend = eps * payout
    growth = min(0.10, _bounded_growth(row))
    wacc = _safe_wacc(row, config)
    if dividend <= 0 or wacc <= growth:
        return None
    return dividend * (1.0 + growth) / (wacc - growth)


def _maybe_relative(
    row: pd.Series,
    config: IntrinsicValuationConfig,
    industry_percentile: float | None,
    flags: list[str],
) -> float | None:
    eps = _safe_float(row.get("eps")) or _safe_float(row.get("net_income_per_share"))
    book = _safe_float(row.get("book_value_per_share"))
    revenue_per_share = _safe_float(row.get("revenue_per_share"))
    if eps <= 0 and book <= 0 and revenue_per_share <= 0:
        return None
    target_pe = _target_multiple(row, "pe_ttm", config.relative_multiple_target_percentile)
    target_pb = _target_multiple(row, "pb", config.relative_multiple_target_percentile)
    target_ps = _target_multiple(row, "ps_ttm", config.relative_multiple_target_percentile)
    components = []
    if target_pe and eps > 0:
        components.append(("pe", target_pe * eps))
    if target_pb and book > 0:
        components.append(("pb", target_pb * book))
    if target_ps and revenue_per_share > 0:
        components.append(("ps", target_ps * revenue_per_share))
    if not components:
        flags.append("relative_inputs_unavailable")
        return None
    return float(np.mean([value for _, value in components]))


def _maybe_asset_floor(row: pd.Series, shares: float, flags: list[str]) -> float | None:
    book_value_per_share = _safe_float(row.get("book_value_per_share"))
    if book_value_per_share > 0:
        return book_value_per_share
    total_equity = _safe_float(row.get("total_equity") or row.get("shareholders_equity"))
    if shares > 0 and total_equity > 0:
        return total_equity / shares
    flags.append("asset_floor_unavailable")
    return None


def _composite_fair_value(
    *,
    dcf: float | None,
    ddm: float | None,
    relative: float | None,
    asset: float | None,
    config: IntrinsicValuationConfig,
    flags: list[str],
    earnings_negative: bool,
) -> tuple[float | None, dict[str, float]]:
    components = {
        "dcf": dcf,
        "ddm": ddm,
        "relative": relative,
        "asset": asset,
    }
    available = {key: value for key, value in components.items() if value is not None and value > 0}
    if not available:
        return None, {}
    weights = {"dcf": 0.40, "ddm": 0.15, "relative": 0.35, "asset": 0.10}
    if earnings_negative:
        weights["asset"] = config.asset_floor_weight_when_negative_earnings
        weights["relative"] = max(0.0, weights["relative"] - 0.15)
    filtered_weights = {key: weights[key] for key in available}
    total_weight = sum(filtered_weights.values()) or 1.0
    normalized = {key: value / total_weight for key, value in filtered_weights.items()}
    fair_value = sum(normalized[key] * available[key] for key in available)
    return fair_value, normalized


def _bubble_risk_score(
    price: float,
    fair_value: float | None,
    industry_percentile: float | None,
    history_percentile: float | None,
    flags: list[str],
) -> float:
    if fair_value is None or fair_value <= 0 or price <= 0:
        return 0.40
    ratio = price / fair_value
    base = float(np.clip((ratio - 1.0) / 0.50, 0.0, 1.0))
    if industry_percentile is not None:
        base = max(base, industry_percentile)
    if history_percentile is not None:
        base = max(base, history_percentile)
    if base >= 0.80:
        flags.append("valuation_bubble")
    return float(np.clip(base, 0.0, 1.0))


def _valuation_score(
    margin_of_safety_pct: float,
    bubble_risk: float,
    history_percentile: float | None,
    industry_percentile: float | None,
) -> float:
    margin_component = float(np.clip(50.0 + 100.0 * margin_of_safety_pct, 0.0, 100.0))
    bubble_penalty = bubble_risk * 30.0
    history_bonus = (1.0 - history_percentile) * 20.0 if history_percentile is not None else 0.0
    industry_bonus = (1.0 - industry_percentile) * 15.0 if industry_percentile is not None else 0.0
    return float(np.clip(margin_component + history_bonus + industry_bonus - bubble_penalty, 0.0, 100.0))


def _maybe_reverse_dcf(row: pd.Series, market_cap: float, config: IntrinsicValuationConfig, flags: list[str]) -> float | None:
    fcff = _safe_float(row.get("fcff") or row.get("free_cash_flow"))
    if fcff == 0:
        fcff = _safe_float(row.get("operating_cash_flow")) + _safe_float(row.get("capex"))
    wacc = _safe_wacc(row, config)
    if fcff <= 0 or market_cap <= 0 or wacc <= config.default_terminal_growth:
        return None
    try:
        return reverse_dcf_implied_growth(
            market_cap=market_cap,
            fcff=fcff,
            wacc=wacc,
            terminal_growth_rate=config.default_terminal_growth,
            years=config.default_forecast_years,
            net_debt=_safe_float(row.get("net_debt")),
        )
    except (ValueError, ZeroDivisionError):
        flags.append("reverse_dcf_failed")
        return None


def _confidence_after_haircut(
    row: pd.Series,
    fair_value: float | None,
    config: IntrinsicValuationConfig,
    flags: list[str],
) -> float:
    base = 0.55 if fair_value is not None else 0.25
    fraud_score = _safe_float(row.get("fraud_risk_score", 30.0))
    haircut = max(0.0, fraud_score - config.fraud_confidence_haircut_threshold) / max(1.0, 100.0 - config.fraud_confidence_haircut_threshold)
    base *= 1.0 - config.fraud_confidence_haircut_strength * haircut
    audit_opinion = str(row.get("audit_opinion", "standard")).lower()
    if audit_opinion in {"qualified", "adverse", "disclaimer"}:
        flags.append("non_standard_audit_opinion")
        base *= 0.50
    if bool(row.get("recent_restatement", False)):
        flags.append("recent_restatement")
        base *= 0.65
    return float(np.clip(base, 0.05, 0.95))


def _bounded_growth(row: pd.Series) -> float:
    rev_growth = _safe_float(row.get("revenue_growth"))
    profit_growth = _safe_float(row.get("profit_growth"))
    blend = 0.5 * rev_growth + 0.5 * profit_growth
    return float(np.clip(blend, -0.10, 0.30))


def _safe_wacc(row: pd.Series, config: IntrinsicValuationConfig) -> float:
    wacc = _safe_float(row.get("wacc"))
    if wacc <= 0:
        wacc = 0.09
    return float(np.clip(wacc, config.default_wacc_floor, config.default_wacc_ceiling))


def _key_assumptions(row: pd.Series, config: IntrinsicValuationConfig) -> dict[str, float]:
    return {
        "growth_rate": _bounded_growth(row),
        "wacc": _safe_wacc(row, config),
        "terminal_growth": config.default_terminal_growth,
        "forecast_years": float(config.default_forecast_years),
    }


# ---------------------------------------------------------------------------
# Cross-sectional helpers
# ---------------------------------------------------------------------------


def _market_lookup(market_state: pd.DataFrame) -> dict[str, dict]:
    if market_state is None or market_state.empty or "symbol" not in market_state.columns:
        return {}
    latest = (
        market_state.assign(trade_date=pd.to_datetime(market_state.get("trade_date"), errors="coerce"))
        .sort_values(["symbol", "trade_date"])
        .groupby("symbol", sort=False)
        .tail(1)
    )
    return {str(row["symbol"]): row.to_dict() for _, row in latest.iterrows()}


def _industry_percentiles(latest: pd.DataFrame, multiples: Iterable[str], config: IntrinsicValuationConfig) -> dict[str, float]:
    if "industry" not in latest.columns:
        return {}
    output: dict[str, float] = {}
    for industry, peers in latest.groupby("industry"):
        if len(peers) < config.industry_peer_minimum:
            continue
        percentiles = {}
        for column in multiples:
            if column in peers.columns:
                series = pd.to_numeric(peers[column], errors="coerce")
                ranked = series.rank(pct=True)
                percentiles[column] = ranked
        for _, peer in peers.iterrows():
            symbol = str(peer["symbol"])
            values = []
            for column, ranked in percentiles.items():
                value = ranked.loc[peer.name]
                if pd.notna(value):
                    values.append(float(value))
            if values:
                output[symbol] = float(np.mean(values))
    return output


def _history_percentiles(fundamentals: pd.DataFrame, multiples: Iterable[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    if fundamentals is None or fundamentals.empty:
        return output
    for symbol, group in fundamentals.groupby("symbol"):
        history = group.sort_values("report_date") if "report_date" in group.columns else group
        ranks = []
        for column in multiples:
            if column not in history.columns or len(history) < 4:
                continue
            series = pd.to_numeric(history[column], errors="coerce").dropna()
            if len(series) < 4:
                continue
            current = float(series.iloc[-1])
            rank = float((series <= current).mean())
            ranks.append(rank)
        if ranks:
            output[str(symbol)] = float(np.mean(ranks))
    return output


def _target_multiple(row: pd.Series, column: str, target_percentile: float) -> float | None:
    industry_median = _safe_float(row.get(f"industry_median_{column}"))
    if industry_median > 0:
        return industry_median * (1.0 - target_percentile + 0.5)
    history_value = _safe_float(row.get(f"history_median_{column}"))
    if history_value > 0:
        return history_value
    current = _safe_float(row.get(column))
    return current * target_percentile if current > 0 else None


def _safe_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(numeric) or np.isinf(numeric):
        return 0.0
    return numeric
