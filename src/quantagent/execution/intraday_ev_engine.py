"""Expected-value decision engine for A-share intraday Do-T."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from quantagent.execution.intraday_fill import CostConfig
from quantagent.execution.intraday_ledger import IntradayLedger


VALID_ACTIONS = {"NO_TRADE", "SELL_HIGH", "BUY_BACK", "BUY_LOW", "SELL_AFTER_BUY"}


@dataclass(frozen=True)
class IntradayModelSignals:
    p_sell_high_success: float = 0.0
    expected_sell_high_gain_bps: float = 0.0
    p_fail_new_high: float = 0.0
    expected_chase_loss_bps: float = 0.0
    p_buyback_now: float = 0.0
    expected_buyback_edge_bps: float = 0.0
    wait_extra_edge_bps: float = 0.0
    miss_rebound_risk_bps: float = 0.0
    p_buy_low_success: float = 0.0
    expected_buy_low_gain_bps: float = 0.0
    p_fail_breakdown: float = 0.0
    expected_breakdown_loss_bps: float = 0.0
    p_sell_after_buy_success: float = 0.0
    expected_sell_after_buy_edge_bps: float = 0.0
    p_eod_restore: float = 0.0
    risk_score: float = 0.0
    model_version: str = ""


@dataclass(frozen=True)
class EVDecisionConfig:
    cost: CostConfig = field(default_factory=CostConfig)
    absolute_min_edge_bps: float = 8.0
    base_success_prob: float = 0.58
    min_cash: float = 1_000.0
    round_lot: int = 100
    base_quantity_fraction: float = 0.20
    max_round_trips_per_day: int = 3
    max_sell_fraction: float = 0.50
    max_over_position_fraction: float = 0.30
    no_new_pair_last_minutes: int = 20
    inventory_restore_penalty_bps: float = 10.0
    liquidity_penalty_scale_bps: float = 5.0
    max_success_prob_threshold: float = 0.90
    min_success_prob_threshold: float = 0.50


@dataclass(frozen=True)
class EVDecision:
    action: str
    ev_bps: float = 0.0
    quantity: int = 0
    expected_net_edge_bps: float = 0.0
    calibrated_probability: float = 0.0
    risk_score: float = 0.0
    dynamic_min_edge_bps: float = 0.0
    dynamic_min_success_prob: float = 0.0
    reason: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    candidate_evs: dict[str, float] = field(default_factory=dict)
    legal_check: dict[str, bool] = field(default_factory=dict)
    model_version: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "ev_bps": self.ev_bps,
            "quantity": self.quantity,
            "expected_net_edge_bps": self.expected_net_edge_bps,
            "calibrated_probability": self.calibrated_probability,
            "risk_score": self.risk_score,
            "dynamic_min_edge_bps": self.dynamic_min_edge_bps,
            "dynamic_min_success_prob": self.dynamic_min_success_prob,
            "reason": list(self.reason),
            "risk_flags": list(self.risk_flags),
            "candidate_evs": dict(self.candidate_evs),
            "legal_check": dict(self.legal_check),
            "model_version": self.model_version,
        }


def decide_ev(
    state: Mapping[str, Any] | Any,
    ledger: IntradayLedger,
    signals: IntradayModelSignals,
    config: EVDecisionConfig | None = None,
) -> EVDecision:
    """Choose the highest legal positive-EV Do-T action.

    Default is always ``NO_TRADE``.  New pairs are blocked near the close, under
    one-way trend risk, near price limits, and whenever EV fails dynamic cost
    and probability gates.
    """
    cfg = config or EVDecisionConfig()
    thresholds = dynamic_thresholds(state, cfg)
    min_edge = thresholds["dynamic_min_edge_bps"]
    min_prob = thresholds["dynamic_min_success_prob"]
    risk_flags = list(thresholds["risk_flags"])
    cost_bps = round_trip_total_cost_bps(cfg.cost)
    liquidity_penalty_bps = _liquidity_penalty_bps(state, cfg)
    candidates: list[tuple[str, float, int, float, float, list[str]]] = []
    candidate_evs: dict[str, float] = {}

    if ledger.open_sell_pairs:
        qty = _open_pair_quantity(ledger.open_sell_pairs)
        ev_now = compute_buyback_ev(state, ledger, signals, cfg)
        ev_wait = compute_wait_ev(state, signals, cfg)
        candidate_evs["BUY_BACK"] = ev_now
        if qty > 0 and ev_now > ev_wait and ev_now > cfg.absolute_min_edge_bps and ledger.can_buy(qty, _price(state), 0.0):
            candidates.append((
                "BUY_BACK",
                ev_now,
                _round_lot(qty, cfg.round_lot),
                max(signals.p_buyback_now, 0.5),
                ev_now,
                ["open sell pair: buyback EV exceeds wait EV"],
            ))

    if ledger.open_buy_pairs:
        qty = min(_open_pair_quantity(ledger.open_buy_pairs), ledger.sellable_shares)
        ev = compute_sell_after_buy_ev(state, ledger, signals, cfg)
        candidate_evs["SELL_AFTER_BUY"] = ev
        if qty > 0 and ev > cfg.absolute_min_edge_bps and ledger.can_sell(qty):
            candidates.append((
                "SELL_AFTER_BUY",
                ev,
                _round_lot(qty, cfg.round_lot),
                max(signals.p_sell_after_buy_success, 0.5),
                ev,
                ["open buy pair: sell old shares to restore target"],
            ))

    if not _blocks_new_pairs(state, cfg, risk_flags):
        qty_sell = _new_pair_quantity(ledger.sellable_shares, ledger.target_shares, cfg)
        ev_sell_high = (
            signals.p_sell_high_success * signals.expected_sell_high_gain_bps
            - signals.p_fail_new_high * signals.expected_chase_loss_bps
            - cost_bps
            - liquidity_penalty_bps
            - cfg.inventory_restore_penalty_bps * signals.p_eod_restore
        )
        candidate_evs["SELL_HIGH"] = ev_sell_high
        if (
            qty_sell > 0
            and ledger.can_sell(qty_sell)
            and ledger.today_sold + qty_sell <= int(ledger.carried_shares * cfg.max_sell_fraction)
            and ev_sell_high > min_edge
            and signals.p_sell_high_success > min_prob
            and not _near_limit_up_risk(state)
            and not _one_way_uptrend_risk(state)
        ):
            candidates.append((
                "SELL_HIGH",
                ev_sell_high,
                qty_sell,
                signals.p_sell_high_success,
                signals.expected_sell_high_gain_bps,
                ["reverse-T first leg has positive cost-adjusted EV"],
            ))

        qty_buy = _new_pair_quantity(ledger.sellable_shares, ledger.target_shares, cfg)
        price = _price(state)
        max_position = int(ledger.target_shares * (1.0 + cfg.max_over_position_fraction))
        ev_buy_low = (
            signals.p_buy_low_success * signals.expected_buy_low_gain_bps
            - signals.p_fail_breakdown * signals.expected_breakdown_loss_bps
            - cost_bps
            - liquidity_penalty_bps
            - cfg.inventory_restore_penalty_bps * signals.p_eod_restore
        )
        candidate_evs["BUY_LOW"] = ev_buy_low
        if (
            qty_buy > 0
            and ledger.cash > cfg.min_cash
            and ledger.sellable_shares > 0
            and ledger.current_position + qty_buy <= max_position
            and ledger.can_buy(qty_buy, price, 0.0)
            and ev_buy_low > min_edge
            and signals.p_buy_low_success > min_prob
            and not _near_limit_down_risk(state)
            and not _one_way_downtrend_risk(state)
        ):
            candidates.append((
                "BUY_LOW",
                ev_buy_low,
                qty_buy,
                signals.p_buy_low_success,
                signals.expected_buy_low_gain_bps,
                ["positive-T first leg has positive cost-adjusted EV"],
            ))

    if not candidates:
        return EVDecision(
            "NO_TRADE",
            dynamic_min_edge_bps=min_edge,
            dynamic_min_success_prob=min_prob,
            reason=["no legal candidate cleared EV, probability, cost, and risk gates"],
            risk_flags=risk_flags,
            candidate_evs=candidate_evs,
            legal_check=_legal_check(ledger),
            model_version=signals.model_version,
        )

    action, ev, qty, prob, expected_edge, reasons = max(candidates, key=lambda x: x[1])
    return EVDecision(
        action=action,
        ev_bps=round(float(ev), 6),
        quantity=int(qty),
        expected_net_edge_bps=round(float(expected_edge), 6),
        calibrated_probability=round(float(prob), 6),
        risk_score=round(float(signals.risk_score), 6),
        dynamic_min_edge_bps=round(float(min_edge), 6),
        dynamic_min_success_prob=round(float(min_prob), 6),
        reason=reasons,
        risk_flags=risk_flags,
        candidate_evs={k: round(float(v), 6) for k, v in candidate_evs.items()},
        legal_check=_legal_check(ledger),
        model_version=signals.model_version,
    )


def dynamic_thresholds(state: Mapping[str, Any] | Any, config: EVDecisionConfig) -> dict[str, Any]:
    cost_bps = round_trip_total_cost_bps(config.cost)
    vol_bps = _volatility_bps(state)
    spread_bps = _float_state(state, "estimated_spread_bps", config.cost.spread_bps)
    edge = max(
        2.0 * cost_bps,
        0.35 * vol_bps,
        1.5 * spread_bps,
        config.absolute_min_edge_bps,
    )
    liquidity_penalty = min(0.20, max(0.0, _float_state(state, "volume_capacity_ratio", 0.0) - 1.0) * 0.05)
    trend_penalty = min(0.20, max(0.0, _float_state(state, "one_way_trend_probability", 0.0)) * 0.15)
    near_limit_penalty = 0.10 if _bool_state(state, "near_limit_risk", False) else 0.0
    high_cost_penalty = 0.05 if cost_bps > 25.0 else 0.0
    high_vol_bonus = 0.05 if vol_bps > 80.0 and _float_state(state, "mean_reversion_probability", 0.0) > 0.60 else 0.0
    prob = (
        config.base_success_prob
        + liquidity_penalty
        + trend_penalty
        + near_limit_penalty
        + high_cost_penalty
        - high_vol_bonus
    )
    prob = min(config.max_success_prob_threshold, max(config.min_success_prob_threshold, prob))
    flags: list[str] = []
    if liquidity_penalty > 0:
        flags.append("low_liquidity")
    if trend_penalty > 0.08:
        flags.append("one_way_trend")
    if near_limit_penalty > 0:
        flags.append("near_limit")
    if high_cost_penalty > 0:
        flags.append("high_cost")
    return {
        "dynamic_min_edge_bps": float(edge),
        "dynamic_min_success_prob": float(prob),
        "round_trip_cost_bps": float(cost_bps),
        "risk_flags": flags,
    }


def compute_buyback_ev(
    state: Mapping[str, Any] | Any,
    ledger: IntradayLedger,
    signals: IntradayModelSignals,
    config: EVDecisionConfig,
) -> float:
    avg_sell = _weighted_pair_price(ledger.open_sell_pairs)
    if avg_sell <= 0:
        return -1e9
    current = _price(state)
    realized_edge = (avg_sell - current) / avg_sell * 10_000.0
    return (
        max(signals.p_buyback_now, 0.5) * max(realized_edge, signals.expected_buyback_edge_bps)
        - round_trip_second_leg_cost_bps(config.cost, "buy")
        - signals.miss_rebound_risk_bps * 0.25
    )


def compute_wait_ev(
    state: Mapping[str, Any] | Any,
    signals: IntradayModelSignals,
    config: EVDecisionConfig,
) -> float:
    return signals.wait_extra_edge_bps - signals.miss_rebound_risk_bps * 0.50 - _liquidity_penalty_bps(state, config)


def compute_sell_after_buy_ev(
    state: Mapping[str, Any] | Any,
    ledger: IntradayLedger,
    signals: IntradayModelSignals,
    config: EVDecisionConfig,
) -> float:
    avg_buy = _weighted_pair_price(ledger.open_buy_pairs)
    if avg_buy <= 0:
        return -1e9
    current = _price(state)
    realized_edge = (current - avg_buy) / avg_buy * 10_000.0
    return (
        max(signals.p_sell_after_buy_success, 0.5) * max(realized_edge, signals.expected_sell_after_buy_edge_bps)
        - round_trip_second_leg_cost_bps(config.cost, "sell")
        - signals.risk_score * 5.0
    )


def round_trip_total_cost_bps(cost: CostConfig) -> float:
    explicit = (2.0 * cost.commission_rate + cost.stamp_tax_sell + 2.0 * cost.transfer_fee) * 10_000.0
    execution = 2.0 * (cost.slippage_bps + cost.spread_bps)
    return explicit + execution


def round_trip_second_leg_cost_bps(cost: CostConfig, side: str) -> float:
    explicit = (cost.commission_rate + cost.transfer_fee) * 10_000.0
    if side.lower() == "sell":
        explicit += cost.stamp_tax_sell * 10_000.0
    return explicit + cost.slippage_bps + cost.spread_bps


def _blocks_new_pairs(state: Mapping[str, Any] | Any, config: EVDecisionConfig, risk_flags: list[str]) -> bool:
    if _float_state(state, "minutes_to_close", 999.0) <= config.no_new_pair_last_minutes:
        risk_flags.append("no_new_pair_near_close")
        return True
    if _completed_round_trips_state(state) >= config.max_round_trips_per_day:
        risk_flags.append("max_round_trips_reached")
        return True
    return False


def _near_limit_up_risk(state: Mapping[str, Any] | Any) -> bool:
    return _bool_state(state, "near_limit_up_risk", False) or _float_state(state, "limit_up_distance", 1.0) < 0.003


def _near_limit_down_risk(state: Mapping[str, Any] | Any) -> bool:
    return _bool_state(state, "near_limit_down_risk", False) or _float_state(state, "limit_down_distance", 1.0) < 0.003


def _one_way_uptrend_risk(state: Mapping[str, Any] | Any) -> bool:
    return _float_state(state, "one_way_trend_probability", 0.0) > 0.65 and _float_state(state, "rolling_return_20m", 0.0) > 0


def _one_way_downtrend_risk(state: Mapping[str, Any] | Any) -> bool:
    return _float_state(state, "one_way_trend_probability", 0.0) > 0.65 and _float_state(state, "rolling_return_20m", 0.0) < 0


def _liquidity_penalty_bps(state: Mapping[str, Any] | Any, config: EVDecisionConfig) -> float:
    ratio = _float_state(state, "volume_capacity_ratio", 0.0)
    return max(0.0, ratio - 1.0) * config.liquidity_penalty_scale_bps


def _volatility_bps(state: Mapping[str, Any] | Any) -> float:
    value = _float_state(state, "rolling_volatility_20m", 0.0)
    return value * 10_000.0 if abs(value) < 2.0 else value


def _price(state: Mapping[str, Any] | Any) -> float:
    return _float_state(state, "last", _float_state(state, "close", 0.0))


def _float_state(state: Mapping[str, Any] | Any, key: str, default: float) -> float:
    if isinstance(state, Mapping):
        value = state.get(key, default)
    else:
        value = getattr(state, key, default)
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bool_state(state: Mapping[str, Any] | Any, key: str, default: bool) -> bool:
    if isinstance(state, Mapping):
        value = state.get(key, default)
    else:
        value = getattr(state, key, default)
    return bool(value)


def _completed_round_trips_state(state: Mapping[str, Any] | Any) -> int:
    return int(_float_state(state, "completed_round_trips_today", 0.0))


def _new_pair_quantity(available: int, target_shares: int, config: EVDecisionConfig) -> int:
    base = min(int(available), int(max(target_shares, 0) * config.base_quantity_fraction))
    return _round_lot(base, config.round_lot)


def _round_lot(quantity: int, lot: int) -> int:
    return int(max(0, int(quantity)) // max(1, int(lot)) * max(1, int(lot)))


def _open_pair_quantity(pairs: list[Any]) -> int:
    return sum(int(getattr(p, "quantity", 0)) for p in pairs)


def _weighted_pair_price(pairs: list[Any]) -> float:
    qty = _open_pair_quantity(pairs)
    if qty <= 0:
        return 0.0
    return sum(float(getattr(p, "price", 0.0)) * int(getattr(p, "quantity", 0)) for p in pairs) / qty


def _legal_check(ledger: IntradayLedger) -> dict[str, bool]:
    return {
        "t_plus_1_ok": ledger.sellable_shares == max(0, ledger.carried_shares - ledger.today_sold),
        "sellable_qty_ok": ledger.today_sold <= ledger.carried_shares,
        "target_accounting_ok": isinstance(ledger.position_gap_to_target, int),
    }


decide_expected_value = decide_ev


__all__ = [
    "EVDecision",
    "EVDecisionConfig",
    "IntradayModelSignals",
    "VALID_ACTIONS",
    "compute_buyback_ev",
    "compute_sell_after_buy_ev",
    "compute_wait_ev",
    "decide_ev",
    "decide_expected_value",
    "dynamic_thresholds",
    "round_trip_second_leg_cost_bps",
    "round_trip_total_cost_bps",
]
