"""Retail-account hedge feasibility checker.

Most A-share retail accounts cannot:

* Short individual stocks (no margin).
* Short index futures or options (no qualified-investor permissions).
* Borrow shares for securities lending.

What they *can* do is buy a long ETF (broad index, sector, theme),
keep cash, or sell down high-beta names. The hedge engine therefore
needs to know which hedge actions are actually executable for the
target account before emitting a recommendation. This module is the
seam that converts "what the model wants to do" into "what the broker
will actually accept".

The output is a filtered :class:`HedgeRecommendation` plus a per-action
audit string explaining why each blocked path was rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quantagent.portfolio.hedge_instrument_selector import (
    HedgeAction,
    HedgeInstrument,
    HedgeRecommendation,
)


@dataclass(frozen=True)
class RetailAccountCapabilities:
    can_short_individual_stock: bool = False
    can_short_index_futures: bool = False
    can_trade_options: bool = False
    can_buy_etf_inverse: bool = True
    can_hold_cash: bool = True
    can_reduce_gross_exposure: bool = True
    allow_intraday_hedge: bool = False
    margin_account: bool = False


@dataclass(frozen=True)
class RetailHedgeFeasibilityResult:
    recommendation: HedgeRecommendation
    blocked_actions: tuple[str, ...]
    audit_notes: tuple[str, ...]
    feasibility_score: float = 1.0
    capabilities: RetailAccountCapabilities = field(default_factory=RetailAccountCapabilities)


def filter_recommendation_for_retail(
    recommendation: HedgeRecommendation,
    capabilities: RetailAccountCapabilities | None = None,
    instrument_universe: tuple[HedgeInstrument, ...] = (),
) -> RetailHedgeFeasibilityResult:
    """Drop hedge actions the retail account cannot execute and re-balance the rest."""

    capabilities = capabilities or RetailAccountCapabilities()
    allowed_actions: list[HedgeAction] = []
    blocked: list[str] = []
    notes: list[str] = []
    instruments_by_id = {item.instrument_id: item for item in instrument_universe}
    for action in recommendation.actions:
        ok, reason = _is_action_feasible(action, capabilities)
        if ok:
            allowed_actions.append(action)
        else:
            blocked.append(action.value)
            notes.append(f"{action.value}_blocked:{reason}")
    feasible_weights: dict[str, float] = {}
    dropped_weight = 0.0
    for instrument_id, weight in recommendation.instrument_weights.items():
        instrument = instruments_by_id.get(instrument_id)
        if instrument is None:
            feasible_weights[instrument_id] = weight
            continue
        if instrument.short_via_etf_inverse and not capabilities.can_buy_etf_inverse:
            dropped_weight += weight
            blocked.append(instrument_id)
            notes.append(f"{instrument_id}_blocked:requires_inverse_etf")
            continue
        feasible_weights[instrument_id] = weight
    if dropped_weight > 0.0 and capabilities.can_hold_cash:
        notes.append(f"reroute_{dropped_weight:.4f}_to_cash_buffer")
        if HedgeAction.CASH_BUFFER not in allowed_actions:
            allowed_actions.append(HedgeAction.CASH_BUFFER)
    if not allowed_actions and capabilities.can_hold_cash:
        # Fall back to cash buffer if everything else is blocked
        allowed_actions = [HedgeAction.CASH_BUFFER]
        notes.append("fallback_to_cash_buffer_only")
    feasibility_score = _feasibility_score(recommendation.actions, allowed_actions)
    filtered = HedgeRecommendation(
        actions=tuple(dict.fromkeys(allowed_actions)) if allowed_actions else (HedgeAction.NONE,),
        instrument_weights=feasible_weights,
        expected_beta_reduction=recommendation.expected_beta_reduction * feasibility_score,
        expected_cost_bps=recommendation.expected_cost_bps,
        rationale=f"{recommendation.rationale}; retail_filtered={feasibility_score:.2f}",
        diagnostics=dict(recommendation.diagnostics)
        | {"retail_feasibility_score": feasibility_score, "retail_dropped_weight": dropped_weight},
    )
    return RetailHedgeFeasibilityResult(
        recommendation=filtered,
        blocked_actions=tuple(blocked),
        audit_notes=tuple(notes),
        feasibility_score=feasibility_score,
        capabilities=capabilities,
    )


def _is_action_feasible(action: HedgeAction, caps: RetailAccountCapabilities) -> tuple[bool, str]:
    if action == HedgeAction.NONE:
        return True, "no_op"
    if action == HedgeAction.REDUCE_GROSS:
        return caps.can_reduce_gross_exposure, "reduce_gross_disabled"
    if action == HedgeAction.CASH_BUFFER:
        return caps.can_hold_cash, "cash_buffer_disabled"
    if action == HedgeAction.SUSPEND_NEW_OPENS:
        return True, "suspend_always_allowed"
    if action == HedgeAction.BETA_REDUCTION:
        return caps.can_reduce_gross_exposure, "beta_reduction_requires_gross_reduction"
    if action == HedgeAction.BROAD_INDEX_HEDGE:
        return (
            caps.can_short_index_futures or caps.can_buy_etf_inverse,
            "broad_hedge_requires_futures_or_inverse_etf",
        )
    if action == HedgeAction.SECTOR_HEDGE:
        return caps.can_buy_etf_inverse, "sector_hedge_requires_inverse_etf"
    return False, "unknown_action"


def _feasibility_score(requested: tuple[HedgeAction, ...], allowed: list[HedgeAction]) -> float:
    if not requested:
        return 1.0
    return float(len(allowed)) / float(len(requested))
