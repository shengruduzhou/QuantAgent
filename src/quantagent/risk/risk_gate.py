from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantagent.execution.broker_base import OrderIntent, OrderSide
from quantagent.risk.kill_switch import KillSwitch
from quantagent.risk.risk_limits import V6RiskLimits


@dataclass(frozen=True)
class RiskGateResult:
    passed: bool
    rejected_symbols: dict[str, str] = field(default_factory=dict)
    violations: tuple[str, ...] = ()
    checked_weights: pd.Series | None = None


class RiskGate:
    def __init__(self, limits: V6RiskLimits | None = None, kill_switch: KillSwitch | None = None) -> None:
        self.limits = limits or V6RiskLimits()
        self.kill_switch = kill_switch or KillSwitch()

    def check_target_weights(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series | None = None,
        market_state: pd.DataFrame | None = None,
        sector: pd.Series | None = None,
        data_quality_score: float = 1.0,
        model_drift_score: float = 0.0,
        conformal_width: pd.Series | None = None,
    ) -> RiskGateResult:
        weights = target_weights.fillna(0.0).astype(float).copy()
        rejected: dict[str, str] = {}
        violations: list[str] = []
        if self.kill_switch.triggered:
            violations.append("kill_switch_triggered")
        if data_quality_score < self.limits.min_data_quality_score:
            violations.append("data_quality_below_threshold")
        if model_drift_score > self.limits.max_model_drift_score:
            violations.append("model_drift_above_threshold")
        oversized = weights[weights.abs() > self.limits.max_name_weight]
        for symbol in oversized.index:
            rejected[str(symbol)] = "max_name_weight"
        weights = weights.clip(upper=self.limits.max_name_weight, lower=-self.limits.max_name_weight)
        if sector is not None:
            sector_weights = weights.groupby(sector.reindex(weights.index).fillna("unknown")).sum().abs()
            for sector_name, value in sector_weights.items():
                if value > self.limits.max_sector_weight:
                    violations.append(f"max_sector_weight:{sector_name}")
        if current_weights is not None:
            turnover = float((weights - current_weights.reindex(weights.index).fillna(0.0)).abs().sum())
            if turnover > self.limits.max_turnover:
                violations.append("max_turnover")
        if conformal_width is not None:
            wide = conformal_width.reindex(weights.index).fillna(0.0)
            for symbol in wide[wide > self.limits.conformal_uncertainty_threshold].index:
                rejected[str(symbol)] = "conformal_uncertainty"
                weights.loc[symbol] = 0.0
        if market_state is not None and not market_state.empty:
            state = market_state.set_index("symbol")
            for symbol in weights.index:
                if symbol not in state.index:
                    continue
                row = state.loc[symbol]
                if bool(row.get("is_suspended", row.get("suspended", False))):
                    rejected[str(symbol)] = "suspended"
                    weights.loc[symbol] = 0.0
                if self.limits.no_trade_st and bool(row.get("is_st", False)):
                    rejected[str(symbol)] = "st"
                    weights.loc[symbol] = 0.0
                if self.limits.no_buy_limit_up and bool(row.get("is_limit_up", False)) and weights.loc[symbol] > 0:
                    rejected[str(symbol)] = "limit_up_no_buy"
                    weights.loc[symbol] = 0.0
                if self.limits.no_sell_limit_down and bool(row.get("is_limit_down", False)) and current_weights is not None and weights.loc[symbol] < current_weights.reindex(weights.index).fillna(0.0).loc[symbol]:
                    rejected[str(symbol)] = "limit_down_no_sell"
                    weights.loc[symbol] = current_weights.reindex(weights.index).fillna(0.0).loc[symbol]
        passed = not violations and not rejected
        return RiskGateResult(passed=passed, rejected_symbols=rejected, violations=tuple(violations), checked_weights=weights)

    def check_order_intents(
        self,
        intents: list[OrderIntent],
        market_state: pd.DataFrame | None = None,
        cash_available: float = float("inf"),
    ) -> RiskGateResult:
        rejected: dict[str, str] = {}
        violations: list[str] = []
        if self.kill_switch.triggered:
            violations.append("kill_switch_triggered")
        if len(intents) > self.limits.max_orders_per_day:
            violations.append("max_orders_per_day")
        state = market_state.set_index("symbol") if market_state is not None and not market_state.empty else pd.DataFrame()
        buy_value = 0.0
        for intent in intents:
            value = float(intent.quantity) * float(intent.reference_price)
            if value > self.limits.max_order_value:
                rejected[intent.intent_id] = "max_order_value"
            if intent.quantity % self.limits.min_lot_size != 0:
                rejected[intent.intent_id] = "min_lot_size"
            if intent.side == OrderSide.BUY:
                buy_value += value
            if not state.empty and intent.symbol in state.index:
                row = state.loc[intent.symbol]
                if bool(row.get("is_suspended", row.get("suspended", False))):
                    rejected[intent.intent_id] = "suspended"
                if self.limits.no_buy_limit_up and intent.side == OrderSide.BUY and bool(row.get("is_limit_up", False)):
                    rejected[intent.intent_id] = "limit_up_no_buy"
                if self.limits.no_sell_limit_down and intent.side == OrderSide.SELL and bool(row.get("is_limit_down", False)):
                    rejected[intent.intent_id] = "limit_down_no_sell"
        if buy_value > cash_available:
            violations.append("cash_constraint")
        return RiskGateResult(passed=not violations and not rejected, rejected_symbols=rejected, violations=tuple(violations))
