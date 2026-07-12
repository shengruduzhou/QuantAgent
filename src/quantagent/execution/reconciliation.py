from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
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


def reconcile_broker_state(
    snapshot: BrokerSnapshot,
    min_account_value: float = 0.0,
) -> ReconciliationResult:
    if not snapshot.connected:
        return ReconciliationResult(False, "broker_disconnected", snapshot)
    if not np.isfinite(snapshot.account_value) or snapshot.account_value < min_account_value:
        return ReconciliationResult(False, "account_value_below_minimum", snapshot)
    symbols = [str(position.symbol) for position in snapshot.positions]
    if len(symbols) != len(set(symbols)):
        return ReconciliationResult(False, "duplicate_broker_positions", snapshot)
    if any(position.available_shares < 0 or position.frozen_shares < 0 for position in snapshot.positions):
        return ReconciliationResult(False, "negative_broker_position", snapshot)
    return ReconciliationResult(True, "ok", snapshot)


@dataclass(frozen=True)
class ReconciliationTolerance:
    round_lot: int = 100
    position_lot_tolerance: int = 1
    cash_absolute_tolerance: float = 100.0
    cash_relative_tolerance: float = 0.0005
    min_fill_rate: float = 0.95
    max_abs_slippage_bps: float = 30.0
    allow_unresolved_orders: int = 0


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
    residual_notional: float = 0.0
    discrepancies: tuple[str, ...] = field(default_factory=tuple)


def _round_expected_shares(values: pd.Series, lot: int) -> pd.Series:
    lot = max(1, int(lot))
    return (values.fillna(0.0) / lot).round() * lot


def _fill_slippage_bps(
    fills: list[object],
    reference_prices: pd.Series | None,
) -> float:
    if not fills or reference_prices is None:
        return 0.0
    references = reference_prices.copy()
    references.index = references.index.astype(str)
    weighted_slippage = 0.0
    total_notional = 0.0
    for fill in fills:
        symbol = str(getattr(fill, "symbol", ""))
        reference = float(references.get(symbol, np.nan))
        price = float(getattr(fill, "fill_price", 0.0) or 0.0)
        quantity = int(getattr(fill, "fill_quantity", getattr(fill, "filled_quantity", 0)) or 0)
        if not np.isfinite(reference) or reference <= 0 or price <= 0 or quantity <= 0:
            continue
        side = _status_value(getattr(fill, "side", "buy"))
        signed = (price / reference - 1.0) * (1.0 if side == "buy" else -1.0)
        notional = quantity * reference
        weighted_slippage += signed * notional
        total_notional += notional
    return float(weighted_slippage / total_notional * 10_000.0) if total_notional > 0 else 0.0


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
    *,
    reference_prices: pd.Series | None = None,
    tolerance: ReconciliationTolerance | None = None,
) -> V6ReconciliationReport:
    cfg = tolerance or ReconciliationTolerance()
    if nav <= 0 or not np.isfinite(nav):
        raise ValueError("nav must be positive and finite")

    px = pd.to_numeric(prices, errors="coerce").copy()
    px.index = px.index.astype(str)
    if px.index.duplicated().any():
        raise ValueError("prices contain duplicate symbols after normalization")
    if px.isna().any() or (px <= 0).any():
        raise ValueError("prices must be positive and finite")

    targets = target_weights.copy()
    targets.index = targets.index.astype(str)
    if targets.index.duplicated().any():
        targets = targets.groupby(level=0).sum()
    target = pd.to_numeric(targets.reindex(px.index), errors="coerce").fillna(0.0)
    expected_shares = _round_expected_shares(target * nav / px, cfg.round_lot)
    expected_shares.index = expected_shares.index.astype(str)

    broker: dict[str, float] = {}
    duplicate_symbols: set[str] = set()
    for position in broker_positions:
        symbol = str(position.symbol)
        if symbol in broker:
            duplicate_symbols.add(symbol)
        broker[symbol] = broker.get(symbol, 0.0) + float(
            position.available_shares + position.frozen_shares
        )
    symbols = sorted(set(expected_shares.index).union(broker))
    expected = {
        symbol: float(expected_shares.reindex(symbols).fillna(0.0).loc[symbol])
        for symbol in symbols
    }
    actual = {symbol: float(broker.get(symbol, 0.0)) for symbol in symbols}
    difference = {symbol: actual[symbol] - expected[symbol] for symbol in symbols}

    states = order_states or []
    fill_list = fills or []
    rejected = sum(_status_value(getattr(state, "status", "")) == "rejected" for state in states)
    unresolved = sum(
        _status_value(getattr(state, "status", "")) in {"pending", "submitted", "partial"}
        for state in states
    )
    if states:
        filled = sum(_status_value(getattr(state, "status", "")) == "filled" for state in states)
        partial = sum(
            0.5 for state in states if _status_value(getattr(state, "status", "")) == "partial"
        )
        fill_rate = float((filled + partial) / len(states))
    else:
        fill_rate = 1.0

    cost = float(
        sum(
            float(getattr(fill, "commission", 0.0) or 0.0)
            + float(getattr(fill, "stamp_duty", 0.0) or 0.0)
            + float(getattr(fill, "transfer_fee", 0.0) or 0.0)
            for fill in fill_list
        )
    )
    slippage = _fill_slippage_bps(fill_list, reference_prices)
    cash_difference = float(cash_actual - cash_expected)
    cash_tolerance = max(cfg.cash_absolute_tolerance, abs(nav) * cfg.cash_relative_tolerance)
    lot_tolerance = cfg.round_lot * cfg.position_lot_tolerance
    residual_notional = float(
        sum(abs(difference[symbol]) * float(px.get(symbol, 0.0)) for symbol in symbols)
    )

    discrepancies: list[str] = []
    if duplicate_symbols:
        discrepancies.append("duplicate_positions:" + ",".join(sorted(duplicate_symbols)))
    off_positions = [
        symbol for symbol, delta in difference.items() if abs(delta) > lot_tolerance + 1e-9
    ]
    if off_positions:
        discrepancies.append("position_mismatch:" + ",".join(off_positions))
    if abs(cash_difference) > cash_tolerance:
        discrepancies.append(f"cash_mismatch:{cash_difference:.2f}>tol{cash_tolerance:.2f}")
    if unresolved > cfg.allow_unresolved_orders:
        discrepancies.append(f"unresolved_orders:{unresolved}>{cfg.allow_unresolved_orders}")
    if fill_rate < cfg.min_fill_rate:
        discrepancies.append(f"fill_rate:{fill_rate:.4f}<{cfg.min_fill_rate:.4f}")
    if abs(slippage) > cfg.max_abs_slippage_bps:
        discrepancies.append(f"slippage_bps:{slippage:.4f}>{cfg.max_abs_slippage_bps:.4f}")

    return V6ReconciliationReport(
        passed=not discrepancies,
        expected_position=expected,
        broker_position=actual,
        difference=difference,
        cash_difference=cash_difference,
        unresolved_orders=unresolved,
        rejected_orders=rejected,
        fill_rate=fill_rate,
        slippage=slippage,
        cost=cost,
        turnover=float(turnover),
        residual_notional=residual_notional,
        discrepancies=tuple(discrepancies),
    )


def _status_value(status: object) -> str:
    return str(getattr(status, "value", status)).lower()
