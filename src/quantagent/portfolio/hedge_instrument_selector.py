"""Hedge instrument selector.

The earlier V7 hedge engine only knew how to raise cash or reduce gross
exposure. For A-share retail accounts that is not always enough — the
realistic toolkit is ETFs:

* Broad-index ETF short (沪深300 / 中证500 / 中证1000 / 创业板 / 科创50).
* Sector / theme ETF short (半导体ETF / AI ETF / 新能源ETF / 军工ETF).
* Cash hedge (raise cash, reduce position).
* Beta reduction (drop high-beta names, keep low-beta core).

This module picks a hedge mix that approximates a target *negative*
portfolio beta given the toolkit and per-instrument cost. The intent is
deterministic and dependency-free: an LP / QP solver is overkill for the
4-6 instrument decision space, so we use a constrained closed-form choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np

from quantagent.portfolio.portfolio_beta_model import PortfolioBeta


class HedgeAction(str, Enum):
    NONE = "none"
    REDUCE_GROSS = "reduce_gross"
    CASH_BUFFER = "cash_buffer"
    BROAD_INDEX_HEDGE = "broad_index_hedge"
    SECTOR_HEDGE = "sector_hedge"
    BETA_REDUCTION = "beta_reduction"
    SUSPEND_NEW_OPENS = "suspend_new_opens"


@dataclass(frozen=True)
class HedgeInstrument:
    instrument_id: str
    name: str
    benchmark_symbol: str
    estimated_beta: float
    cost_bps: float = 8.0
    min_notional_weight: float = 0.0
    max_notional_weight: float = 0.20
    short_via_etf_inverse: bool = False
    notes: str = ""


@dataclass(frozen=True)
class HedgeRecommendation:
    actions: tuple[HedgeAction, ...]
    instrument_weights: dict[str, float]
    expected_beta_reduction: float
    expected_cost_bps: float
    rationale: str
    diagnostics: dict[str, float] = field(default_factory=dict)


_DEFAULT_INSTRUMENTS: tuple[HedgeInstrument, ...] = (
    HedgeInstrument("510300", "CSI300 ETF", "000300.SH", estimated_beta=1.00, cost_bps=5.0, short_via_etf_inverse=False, notes="Long ETF used to hedge by short delta from index futures or to express market view; retail uses inverse strategy."),
    HedgeInstrument("159922", "CSI500 ETF", "000905.SH", estimated_beta=1.05, cost_bps=6.0, short_via_etf_inverse=False),
    HedgeInstrument("512760", "Semi ETF", "semiconductor_index", estimated_beta=1.30, cost_bps=8.0),
    HedgeInstrument("515790", "AI ETF", "ai_compute_index", estimated_beta=1.45, cost_bps=9.0),
    HedgeInstrument("515030", "New Energy Vehicle ETF", "ev_supply_chain_index", estimated_beta=1.20, cost_bps=8.0),
)


def select_hedge(
    portfolio_beta: PortfolioBeta,
    target_beta: float = 0.30,
    hedge_need_score: float = 0.0,
    sector_exposure: dict[str, float] | None = None,
    instrument_universe: Iterable[HedgeInstrument] | None = None,
    max_total_hedge_weight: float = 0.25,
    bear_market: bool = False,
) -> HedgeRecommendation:
    """Pick a hedge mix that lowers the portfolio beta toward ``target_beta``."""

    instruments = tuple(instrument_universe) if instrument_universe is not None else _DEFAULT_INSTRUMENTS
    actions: list[HedgeAction] = []
    weights: dict[str, float] = {}
    if hedge_need_score < 0.20 and portfolio_beta.portfolio_beta <= target_beta + 0.20:
        return HedgeRecommendation(
            actions=(HedgeAction.NONE,),
            instrument_weights={},
            expected_beta_reduction=0.0,
            expected_cost_bps=0.0,
            rationale="hedge_need_low_and_beta_within_target",
            diagnostics={"portfolio_beta": portfolio_beta.portfolio_beta},
        )

    excess_beta = max(0.0, portfolio_beta.portfolio_beta - target_beta)
    sector_exposure = sector_exposure or {}
    sector_picks = _pick_sector_hedges(instruments, sector_exposure)
    broad_picks = _pick_broad_hedges(instruments)

    remaining = excess_beta
    total_hedge_weight = 0.0
    for instrument in sector_picks:
        if remaining <= 0.05:
            break
        weight = float(np.clip(remaining / max(1e-6, instrument.estimated_beta), 0.0, instrument.max_notional_weight))
        weight = min(weight, max_total_hedge_weight - total_hedge_weight)
        if weight <= 0.0:
            continue
        weights[instrument.instrument_id] = weight
        total_hedge_weight += weight
        remaining = max(0.0, remaining - weight * instrument.estimated_beta * 0.85)
        actions.append(HedgeAction.SECTOR_HEDGE)
    if remaining > 0.05:
        for instrument in broad_picks:
            weight = float(np.clip(remaining / max(1e-6, instrument.estimated_beta), 0.0, instrument.max_notional_weight))
            weight = min(weight, max_total_hedge_weight - total_hedge_weight)
            if weight <= 0.0:
                continue
            weights[instrument.instrument_id] = weight
            total_hedge_weight += weight
            remaining = max(0.0, remaining - weight * instrument.estimated_beta * 0.95)
            actions.append(HedgeAction.BROAD_INDEX_HEDGE)
            if remaining <= 0.05:
                break

    if hedge_need_score >= 0.45 or bear_market:
        actions.append(HedgeAction.CASH_BUFFER)
    if hedge_need_score >= 0.65:
        actions.append(HedgeAction.SUSPEND_NEW_OPENS)
        actions.append(HedgeAction.BETA_REDUCTION)

    expected_cost = sum(weights.get(instr.instrument_id, 0.0) * instr.cost_bps for instr in instruments)
    expected_beta_reduction = float(max(0.0, excess_beta - remaining))
    rationale = (
        f"target_beta={target_beta:.2f}, current_beta={portfolio_beta.portfolio_beta:.2f}, "
        f"excess_beta={excess_beta:.2f}, residual_beta={remaining:.2f}, hedge_need={hedge_need_score:.2f}"
    )
    return HedgeRecommendation(
        actions=tuple(dict.fromkeys(actions)) if actions else (HedgeAction.CASH_BUFFER,),
        instrument_weights=weights,
        expected_beta_reduction=expected_beta_reduction,
        expected_cost_bps=expected_cost,
        rationale=rationale,
        diagnostics={
            "excess_beta": excess_beta,
            "residual_beta": remaining,
            "hedge_need_score": hedge_need_score,
            "total_hedge_weight": total_hedge_weight,
        },
    )


def _pick_sector_hedges(
    instruments: tuple[HedgeInstrument, ...],
    sector_exposure: dict[str, float],
) -> list[HedgeInstrument]:
    if not sector_exposure:
        return []
    ranked = sorted(sector_exposure.items(), key=lambda item: item[1], reverse=True)
    picks: list[HedgeInstrument] = []
    for sector, weight in ranked:
        if weight <= 0.05:
            continue
        sector_lower = sector.lower()
        for instrument in instruments:
            if sector_lower in instrument.benchmark_symbol.lower() or sector_lower in instrument.name.lower():
                picks.append(instrument)
                break
    return picks


def _pick_broad_hedges(instruments: tuple[HedgeInstrument, ...]) -> list[HedgeInstrument]:
    return [
        instrument
        for instrument in instruments
        if instrument.benchmark_symbol in {"000300.SH", "000905.SH", "000852.SH", "399001.SZ"}
    ]
