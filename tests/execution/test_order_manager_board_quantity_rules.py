from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

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
from quantagent.execution.order_manager import OrderManager, OrderManagerConfig
from quantagent.execution.paper_adapter import PaperBrokerAdapter
from quantagent.market_rules import ashare as market_rules
from quantagent.paper import orders as paper_orders
from quantagent.paper.broker import MarketSnapshot


class _Broker:
    def query_positions(self):
        return []


class _CaptureBroker(BrokerBase):
    def __init__(self, positions: list[Position] | None = None) -> None:
        self.positions = positions or []
        self.submitted: list[Order] = []

    def submit(self, order: Order) -> OrderState:
        self.submitted.append(order)
        return OrderState(order.client_order_id, "broker-1", OrderStatus.SUBMITTED, 0, 0.0)

    def cancel(self, client_order_id: str) -> OrderState:
        return OrderState(client_order_id, "broker-1", OrderStatus.CANCELLED, 0, 0.0)

    def query_order(self, client_order_id: str) -> OrderState:
        return OrderState(client_order_id, "broker-1", OrderStatus.SUBMITTED, 0, 0.0)

    def query_positions(self) -> list[Position]:
        return list(self.positions)

    def query_account_value(self) -> float:
        return 1_000_000.0

    def on_trade(self, callback) -> None:
        del callback


class _PaperVenueCapture:
    def __init__(self) -> None:
        self.portfolio = SimpleNamespace(positions={}, equity=lambda prices: 1_000_000.0)
        self.submitted = []
        self.orders = {}

    def attach_canonical(self, paper_order_id: str, canonical_order_id: str) -> None:
        del paper_order_id, canonical_order_id

    def submit(self, order, market):
        del market
        self.submitted.append(order)
        order.state = paper_orders.ACCEPTED
        self.orders[order.order_id] = order
        return order

    def cancel(self, order_id: str):
        order = self.orders[order_id]
        order.state = paper_orders.CANCELLED
        return order


def _manager() -> OrderManager:
    return OrderManager(broker=_Broker(), config=OrderManagerConfig())


def _position(shares: int):
    return SimpleNamespace(available_shares=shares, frozen_shares=0)


def _buy_quantity(symbol: str, shares: int) -> tuple[list, OrderManager]:
    manager = _manager()
    intents = manager.target_weights_to_order_intents(
        target_weights=pd.Series({symbol: shares / 10_000}),
        prices=pd.Series({symbol: 10.0}),
        nav=100_000.0,
    )
    return intents, manager


@pytest.mark.parametrize("shares", [200, 201, 237, 251])
def test_star_buy_keeps_one_share_increment_above_200(shares: int) -> None:
    intents, _ = _buy_quantity("688001.SH", shares)
    assert len(intents) == 1
    assert intents[0].quantity == shares


def test_star_buy_below_200_is_rejected_not_rounded_up() -> None:
    intents, manager = _buy_quantity("688001.SH", 199)
    assert intents == []
    assert manager.last_skipped_orders[0]["reason"] == "skipped_below_min_lot"


@pytest.mark.parametrize("shares", [100, 101, 137])
def test_bse_buy_keeps_one_share_increment_above_100(shares: int) -> None:
    intents, _ = _buy_quantity("830001.BJ", shares)
    assert len(intents) == 1
    assert intents[0].quantity == shares


def test_bse_buy_below_100_is_rejected_not_rounded_up() -> None:
    intents, manager = _buy_quantity("830001.BJ", 99)
    assert intents == []
    assert manager.last_skipped_orders[0]["reason"] == "skipped_below_min_lot"


def test_neutral_rule_rejects_subminimum_normal_sell_but_allows_full_residual() -> None:
    assert market_rules.round_to_lot(199, board=market_rules.STAR, side="SELL") == 0
    assert market_rules.round_to_lot(
        199, board=market_rules.STAR, side="SELL", is_full_liquidation=True
    ) == 199
    assert market_rules.round_to_lot(
        250, board=market_rules.SH_MAIN, side="SELL", is_full_liquidation=True
    ) == 200


def test_star_and_bse_limit_order_maxima_are_encoded() -> None:
    assert market_rules.max_order_quantity(market_rules.STAR, "LIMIT") == 100_000
    assert market_rules.max_order_quantity(market_rules.STAR, "MARKET") == 50_000
    assert market_rules.max_order_quantity(market_rules.BSE, "LIMIT") == 1_000_000


def test_star_target_delta_is_capped_at_limit_order_maximum() -> None:
    manager = _manager()
    intents = manager.target_weights_to_order_intents(
        target_weights=pd.Series({"688001.SH": 0.75}),
        prices=pd.Series({"688001.SH": 10.0}),
        nav=2_000_000.0,
    )
    assert len(intents) == 1
    assert intents[0].quantity == 100_000


def test_bse_target_delta_is_capped_at_limit_order_maximum() -> None:
    manager = _manager()
    intents = manager.target_weights_to_order_intents(
        target_weights=pd.Series({"830001.BJ": 0.75}),
        prices=pd.Series({"830001.BJ": 10.0}),
        nav=20_000_000.0,
    )
    assert len(intents) == 1
    assert intents[0].quantity == 1_000_000


def test_star_partial_sell_below_200_is_not_a_legal_order() -> None:
    manager = _manager()
    intents = manager.target_weights_to_order_intents(
        target_weights=pd.Series({"688001.SH": 0.0237}),
        prices=pd.Series({"688001.SH": 10.0}),
        nav=100_000.0,
        positions={"688001.SH": _position(300)},
    )
    assert intents == []
    assert manager.last_skipped_orders[0]["reason"] == "skipped_not_full_odd_lot_liquidation"


def test_star_partial_sell_201_preserves_one_share_increment() -> None:
    manager = _manager()
    intents = manager.target_weights_to_order_intents(
        target_weights=pd.Series({"688001.SH": 0.0099}),
        prices=pd.Series({"688001.SH": 10.0}),
        nav=100_000.0,
        positions={"688001.SH": _position(300)},
    )
    assert len(intents) == 1
    assert intents[0].side.value == "sell"
    assert intents[0].quantity == 201


def test_bse_partial_sell_below_100_is_not_a_legal_order() -> None:
    manager = _manager()
    intents = manager.target_weights_to_order_intents(
        target_weights=pd.Series({"830001.BJ": 0.0163}),
        prices=pd.Series({"830001.BJ": 10.0}),
        nav=100_000.0,
        positions={"830001.BJ": _position(200)},
    )
    assert intents == []
    assert manager.last_skipped_orders[0]["reason"] == "skipped_not_full_odd_lot_liquidation"


def test_bse_partial_sell_101_preserves_one_share_increment() -> None:
    manager = _manager()
    intents = manager.target_weights_to_order_intents(
        target_weights=pd.Series({"830001.BJ": 0.0099}),
        prices=pd.Series({"830001.BJ": 10.0}),
        nav=100_000.0,
        positions={"830001.BJ": _position(200)},
    )
    assert len(intents) == 1
    assert intents[0].quantity == 101


def test_main_board_partial_sell_still_uses_100_share_increment() -> None:
    manager = _manager()
    intents = manager.target_weights_to_order_intents(
        target_weights=pd.Series({"600000.SH": 0.0113}),
        prices=pd.Series({"600000.SH": 10.0}),
        nav=100_000.0,
        positions={"600000.SH": _position(250)},
    )
    assert len(intents) == 1
    assert intents[0].quantity == 100


def test_main_board_full_target_does_not_submit_invalid_250_share_order() -> None:
    manager = _manager()
    intents = manager.target_weights_to_order_intents(
        target_weights=pd.Series({"600000.SH": 0.0}),
        prices=pd.Series({"600000.SH": 10.0}),
        nav=100_000.0,
        positions={"600000.SH": _position(250)},
    )
    assert len(intents) == 1
    assert intents[0].quantity == 200


def test_sub_minimum_residual_can_be_sold_once_when_target_is_zero() -> None:
    manager = _manager()
    intents = manager.target_weights_to_order_intents(
        target_weights=pd.Series({"600000.SH": 0.0}),
        prices=pd.Series({"600000.SH": 10.0}),
        nav=100_000.0,
        positions={"600000.SH": _position(50)},
    )
    assert len(intents) == 1
    assert intents[0].quantity == 50


def test_explicit_star_order_above_maximum_is_rejected_before_broker_touch() -> None:
    broker = _CaptureBroker()
    manager = OrderManager(broker=broker, lineage=Lineage(run_id="qty-test"))
    state = manager.submit_orders([
        Order(
            client_order_id="star-too-large",
            symbol="688001.SH",
            side=OrderSide.BUY,
            quantity=100_001,
            order_type=OrderType.LIMIT,
            price=10.0,
            timestamp="2026-08-10T10:00:00+08:00",
        )
    ])[0]
    assert state.status is OrderStatus.REJECTED
    assert "exchange maximum 100000" in state.last_message
    assert broker.submitted == []


def test_explicit_subminimum_star_sell_requires_proven_full_residual() -> None:
    broker = _CaptureBroker([
        Position("688001.SH", available_shares=300, frozen_shares=0, avg_cost=9.0)
    ])
    manager = OrderManager(broker=broker, lineage=Lineage(run_id="qty-test"))
    state = manager.submit_orders([
        Order(
            client_order_id="star-partial-199",
            symbol="688001.SH",
            side=OrderSide.SELL,
            quantity=199,
            order_type=OrderType.LIMIT,
            price=10.0,
            timestamp="2026-08-10T10:00:00+08:00",
        )
    ])[0]
    assert state.status is OrderStatus.REJECTED
    assert broker.submitted == []


def test_explicit_subminimum_star_sell_is_allowed_for_proven_full_residual() -> None:
    broker = _CaptureBroker([
        Position("688001.SH", available_shares=199, frozen_shares=0, avg_cost=9.0)
    ])
    manager = OrderManager(broker=broker, lineage=Lineage(run_id="qty-test"))
    state = manager.submit_orders([
        Order(
            client_order_id="star-residual-199",
            symbol="688001.SH",
            side=OrderSide.SELL,
            quantity=199,
            order_type=OrderType.LIMIT,
            price=10.0,
            timestamp="2026-08-10T10:00:00+08:00",
        )
    ])[0]
    assert state.status is OrderStatus.SUBMITTED
    assert broker.submitted[0].quantity == 199


def test_mixed_board_reconcile_reaches_paper_adapter_without_quantity_rewrite() -> None:
    venue = _PaperVenueCapture()
    market = {
        "688001.SH": MarketSnapshot(
            symbol="688001.SH", trade_date="2026-08-10", last_price=10.0,
            previous_close=10.0, session_volume=1_000_000, board=market_rules.STAR,
        ),
        "830001.BJ": MarketSnapshot(
            symbol="830001.BJ", trade_date="2026-08-10", last_price=10.0,
            previous_close=10.0, session_volume=1_000_000, board=market_rules.BSE,
        ),
        "600000.SH": MarketSnapshot(
            symbol="600000.SH", trade_date="2026-08-10", last_price=10.0,
            previous_close=10.0, session_volume=1_000_000, board=market_rules.SH_MAIN,
        ),
    }
    adapter = PaperBrokerAdapter(
        broker=venue,
        market_source=lambda symbol, trade_date: market[symbol],
    )
    manager = OrderManager(
        broker=adapter,
        lineage=Lineage(run_id="paper-qty-e2e"),
    )
    manager.reconcile(
        target_weights=pd.Series({
            "688001.SH": 0.0201,
            "830001.BJ": 0.0101,
            "600000.SH": 0.0200,
        }),
        prices=pd.Series({"688001.SH": 10.0, "830001.BJ": 10.0, "600000.SH": 10.0}),
        nav=100_000.0,
        signal_id="mixed-board",
    )
    quantities = {order.symbol: int(order.quantity) for order in venue.submitted}
    assert quantities == {"688001.SH": 201, "830001.BJ": 101, "600000.SH": 200}
