from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantagent.execution.broker_base import Position


@dataclass(frozen=True)
class BrokerSnapshot:
    positions: tuple[Position, ...]
    account_value: float
    connected: bool = True


@dataclass(frozen=True)
class ReconciliationResult:
    passed: bool
    reason: str
    snapshot: BrokerSnapshot


def reconcile_broker_state(snapshot: BrokerSnapshot, min_account_value: float = 0.0) -> ReconciliationResult:
    if not snapshot.connected:
        return ReconciliationResult(False, "broker_disconnected", snapshot)
    if snapshot.account_value < min_account_value:
        return ReconciliationResult(False, "account_value_below_minimum", snapshot)
    return ReconciliationResult(True, "ok", snapshot)


@dataclass(frozen=True)
class V6ReconciliationReport:
    passed: bool
    expected_position: dict[str, float]
    broker_position: dict[str, float]
    difference: dict[str, float]
    cash_difference: float
    unresolved_orders: int
    rejected_orders: int
    fill_rate: float
    slippage: float
    cost: float
    turnover: float


def reconcile_virtual_state(
    target_weights: pd.Series,
    prices: pd.Series,
    broker_positions: list[Position],
    nav: float,
    cash_expected: float,
    cash_actual: float,
    order_states: list[object] | None = None,
    fills: list[object] | None = None,
    turnover: float = 0.0,
) -> V6ReconciliationReport:
    expected_shares = (target_weights.reindex(prices.index).fillna(0.0) * nav / prices.replace(0.0, pd.NA)).fillna(0.0)
    broker = {p.symbol: float(p.available_shares + p.frozen_shares) for p in broker_positions}
    symbols = sorted(set(expected_shares.index.astype(str)).union(broker))
    expected = {symbol: float(expected_shares.reindex(symbols).fillna(0.0).loc[symbol]) for symbol in symbols}
    actual = {symbol: float(broker.get(symbol, 0.0)) for symbol in symbols}
    diff = {symbol: actual[symbol] - expected[symbol] for symbol in symbols}
    states = order_states or []
    rejected = sum(_status_value(getattr(state, "status", "")) == "rejected" for state in states)
    unresolved = sum(_status_value(getattr(state, "status", "")) in {"pending", "submitted", "partial"} for state in states)
    fill_count = len(fills or [])
    fill_rate = fill_count / max(len(states), 1) if states else 1.0
    cost = float(sum(float(getattr(fill, "commission", 0.0)) + float(getattr(fill, "stamp_duty", 0.0)) + float(getattr(fill, "transfer_fee", 0.0)) for fill in (fills or [])))
    report = V6ReconciliationReport(
        passed=unresolved == 0 and max((abs(v) for v in diff.values()), default=0.0) < 1e6,
        expected_position=expected,
        broker_position=actual,
        difference=diff,
        cash_difference=float(cash_actual - cash_expected),
        unresolved_orders=unresolved,
        rejected_orders=rejected,
        fill_rate=float(fill_rate),
        slippage=0.0,
        cost=cost,
        turnover=float(turnover),
    )
    return report


def _status_value(status: object) -> str:
    return str(getattr(status, "value", status)).lower()
