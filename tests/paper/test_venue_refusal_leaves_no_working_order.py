"""A venue refusal is a terminal canonical outcome, never a phantom working order.

The order manager writes CREATED + RISK_APPROVED + SUBMITTED before it calls the
venue. When the venue then declined (no market data for the session, or market
data that is not a measurement) the refusal unwound past the canonical append,
so the record of account kept a SUBMITTED order with its full leaves quantity
that no venue held. Open-order-aware recovery then treated it as live.

Separately, a snapshot whose volume is NaN used to disable the participation
cap, the pre-trade participation check and market impact at once: the whole
order filled uncapped at zero impact.
"""

from __future__ import annotations

import math

import pytest

from quantagent.domain.ledger import CanonicalLedger
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import OrderStatus as CanonicalStatus
from quantagent.execution.broker_base import Order, OrderSide, OrderStatus, OrderType
from quantagent.execution.order_manager import OrderManager, OrderManagerConfig
from quantagent.execution.paper_adapter import (
    MARKET_DATA_INVALID,
    MARKET_DATA_UNAVAILABLE,
    PaperBrokerAdapter,
)
from quantagent.paper import ledger as lg
from quantagent.paper import orders as po
from quantagent.paper.broker import InvalidMarketSnapshot, MarketSnapshot, PaperBroker
from quantagent.paper.portfolio import Portfolio

SYMBOL = "600000.SH"
SESSION = "2026-08-18"


def _snapshot(**overrides) -> MarketSnapshot:
    values = dict(
        symbol=SYMBOL, trade_date=SESSION, last_price=10.0, previous_close=10.0,
        session_volume=10_000_000.0, board="SH_Main",
    )
    values.update(overrides)
    return MarketSnapshot(**values)


def _wired(tmp_path, market_source):
    canonical = CanonicalLedger(str(tmp_path / "canonical.jsonl"))
    broker = PaperBroker(
        Portfolio(portfolio_id="p", cash=1_000_000.0, initial_cash=1_000_000.0),
        lg.EventLedger(tmp_path / "operational.jsonl"),
        run_id="t",
        canonical_ledger=canonical,
    )
    manager = OrderManager(
        broker=PaperBrokerAdapter(broker, market_source),
        config=OrderManagerConfig(strategy_version="t"),
        lineage=Lineage(run_id="t"),
        canonical_ledger=canonical,
        order_book=broker.book,
        idempotency_path=str(tmp_path / "claims.jsonl"),
    )
    return manager, canonical


def _order(key: str = "o1") -> Order:
    return Order(
        client_order_id=key, symbol=SYMBOL, side=OrderSide.BUY, quantity=1_000,
        order_type=OrderType.LIMIT, price=10.0, signal_id=key,
        strategy_version="t", timestamp=f"{SESSION}T14:59:00+08:00",
    )


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -1.0],
)
def test_a_snapshot_with_an_unmeasured_volume_cannot_be_constructed(value) -> None:
    with pytest.raises(InvalidMarketSnapshot, match="market_data_invalid"):
        _snapshot(session_volume=value)


@pytest.mark.parametrize("field", ["last_price", "previous_close"])
@pytest.mark.parametrize("value", [float("nan"), 0.0, -10.0])
def test_a_snapshot_with_an_unusable_price_cannot_be_constructed(field, value) -> None:
    with pytest.raises(InvalidMarketSnapshot):
        _snapshot(**{field: value})


def test_a_zero_volume_bar_is_still_a_measurement() -> None:
    assert _snapshot(session_volume=0.0).invalid_reason() is None


def test_the_venue_rechecks_a_snapshot_mutated_after_construction(tmp_path) -> None:
    """The dataclass is mutable, so construction-time validation is not enough."""
    broker = PaperBroker(
        Portfolio(portfolio_id="p", cash=1_000_000.0, initial_cash=1_000_000.0),
        lg.EventLedger(tmp_path / "operational.jsonl"),
        run_id="t",
        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),
    )
    market = _snapshot()
    market.session_volume = math.nan
    order = broker.submit(
        po.Order(symbol=SYMBOL, side=po.BUY, quantity=1_000, limit_price=10.0), market
    )

    assert order.state == po.REJECTED
    assert order.reject_reason.startswith("market_data_invalid")
    assert broker.fills == []


def test_missing_market_data_inside_the_venue_terminates_the_canonical_order(tmp_path) -> None:
    manager, canonical = _wired(tmp_path, lambda symbol, trade_date: None)

    state = manager.submit_orders([_order()])[0]

    assert state.status is OrderStatus.REJECTED
    assert state.last_message == MARKET_DATA_UNAVAILABLE
    replayed = CanonicalLedger(str(tmp_path / "canonical.jsonl")).replay_book().orders()
    assert [(o.status, o.leaves_quantity) for o in replayed] == [(CanonicalStatus.REJECTED, 0)]
    assert replayed[0].reason == MARKET_DATA_UNAVAILABLE


def test_invalid_market_data_inside_the_venue_terminates_the_canonical_order(tmp_path) -> None:
    def nan_volume(symbol, trade_date):
        return _snapshot(session_volume=float("nan"))

    manager, _ = _wired(tmp_path, nan_volume)

    state = manager.submit_orders([_order()])[0]

    assert state.status is OrderStatus.REJECTED
    assert state.last_message == MARKET_DATA_INVALID
    replayed = CanonicalLedger(str(tmp_path / "canonical.jsonl")).replay_book().orders()
    assert [o.status for o in replayed] == [CanonicalStatus.REJECTED]


def test_a_crash_inside_the_venue_keeps_crash_semantics(tmp_path) -> None:
    """Only a declared refusal is terminated; anything else is left for recovery.

    An arbitrary exception may have been raised after the venue had already
    acted, so writing REJECTED there could contradict what happened.
    """
    def explode(symbol, trade_date):
        raise RuntimeError("process died mid-lookup")

    manager, _ = _wired(tmp_path, explode)

    with pytest.raises(RuntimeError, match="process died"):
        manager.submit_orders([_order()])
    replayed = CanonicalLedger(str(tmp_path / "canonical.jsonl")).replay_book().orders()
    assert [o.status for o in replayed] == [CanonicalStatus.SUBMITTED]
