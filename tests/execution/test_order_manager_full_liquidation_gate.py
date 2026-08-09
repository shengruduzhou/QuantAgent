from __future__ import annotations

from quantagent.domain.lineage import Lineage
from quantagent.execution.broker_base import (
    BrokerBase,
    Order,
    OrderSide,
    OrderState,
    OrderStatus,
    OrderType,
    Position,
)
from quantagent.execution.order_manager import OrderManager


class _Broker(BrokerBase):
    def __init__(self, available: int) -> None:
        self.available = int(available)
        self.submitted: list[Order] = []

    def submit(self, order: Order) -> OrderState:
        self.submitted.append(order)
        return OrderState(order.client_order_id, "broker-1", OrderStatus.SUBMITTED, 0, 0.0)

    def cancel(self, client_order_id: str) -> OrderState:
        return OrderState(client_order_id, "broker-1", OrderStatus.CANCELLED, 0, 0.0)

    def query_order(self, client_order_id: str) -> OrderState:
        return OrderState(client_order_id, "broker-1", OrderStatus.SUBMITTED, 0, 0.0)

    def query_positions(self) -> list[Position]:
        return [Position("600000.SH", available_shares=self.available, frozen_shares=0, avg_cost=9.0)]

    def query_account_value(self) -> float:
        return 1_000_000.0

    def on_trade(self, callback) -> None:
        del callback


def _sell(quantity: int, client_id: str) -> Order:
    return Order(
        client_order_id=client_id,
        symbol="600000.SH",
        side=OrderSide.SELL,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        price=10.0,
        timestamp="2026-08-10T10:00:00+08:00",
    )


def test_explicit_main_board_combined_odd_lot_is_allowed_only_for_full_liquidation() -> None:
    broker = _Broker(available=250)
    manager = OrderManager(broker=broker, lineage=Lineage(run_id="full-liquidation"))

    state = manager.submit_orders([_sell(250, "sell-all-250")])[0]

    assert state.status is OrderStatus.SUBMITTED
    assert [order.quantity for order in broker.submitted] == [250]


def test_explicit_main_board_partial_non_increment_sell_is_rejected() -> None:
    broker = _Broker(available=300)
    manager = OrderManager(broker=broker, lineage=Lineage(run_id="partial-odd-lot"))

    state = manager.submit_orders([_sell(250, "sell-partial-250")])[0]

    assert state.status is OrderStatus.REJECTED
    assert "requires proven full liquidation" in state.last_message
    assert broker.submitted == []
