"""``OrderManagerConfig`` must not carry a participation knob nobody reads.

Round 21 A-09 / Round 22 Q-01 finding, executed in Round 24.

``OrderManagerConfig.max_participation_rate`` was declared with a default of
0.05 and written by two production callers
(``backtest/ashare_execution_simulator_impl.py`` and
``paper/continuous_execution.py``), but ``order_manager.py`` never read it —
the identifier appeared exactly once in the file, on its own declaration line.
A caller that set it believed it had constrained something; nothing happened.

It was removed rather than wired up.  The field's semantics ("how much of one
bar a single order may consume") are a *fill* quantity concept, and the
``OrderManager`` does not match orders.  Routing it to the pre-trade gate would
reproduce an already-argued defect: a pre-trade participation limit set to the
same number as the venue's fill cap rejects every order large enough to leave a
remainder, which makes a partial fill unreachable — see the comment block in
``paper/continuous_execution.py`` that deliberately keeps
``RiskLimits(max_participation=1.0)`` different from
``BrokerConfig(participation_cap=...)``.

The participation limit that *is* enforced on the production order path lives on
``ExecutionConstraintSet.max_single_stock_participation_rate`` and is covered by
``tests/execution/test_pretrade_unmeasured_constraints.py``.

The pinned fixture below was produced on the pre-removal tree with the knob set
to 0.0, 0.05 and 0.99 in turn; all three runs produced byte-identical orders and
rejections, which is the measurement that made the field safe to delete.  The
same values are asserted here after the removal.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pandas as pd

from quantagent.domain.lineage import Lineage
from quantagent.execution.broker_base import OrderState, OrderStatus
from quantagent.execution.order_manager import OrderManager, OrderManagerConfig


class _Broker:
    def __init__(self, positions):
        self._positions = positions
        self.config = SimpleNamespace(dry_run=True, live_trading_enabled=False)

    def submit(self, order):
        return OrderState(order.client_order_id, "b1", OrderStatus.SUBMITTED, 0, 0.0)

    def cancel(self, client_order_id):
        return OrderState(client_order_id, None, OrderStatus.CANCELLED, 0, 0.0)

    def query_order(self, client_order_id):
        return OrderState(client_order_id, None, OrderStatus.PENDING, 0, 0.0)

    def query_positions(self):
        return self._positions

    def query_account_value(self) -> float:
        return 1_000_000.0

    def on_trade(self, callback) -> None:
        return None


def _position(symbol: str, shares: int):
    return SimpleNamespace(symbol=symbol, available_shares=shares, frozen_shares=0)


def _routed_fixture() -> tuple[list[tuple], list[tuple]]:
    """Route one fixed book through ``OrderManager`` and return orders + rejections."""
    broker = _Broker([_position("600519.SH", 500), _position("000002.SZ", 1_000)])
    manager = OrderManager(
        broker=broker,
        lineage=Lineage(run_id="knob-removal"),
        config=OrderManagerConfig(
            lot_size=100,
            min_order_value_yuan=1_000.0,
            strategy_version="knob-removal",
        ),
    )
    target = pd.Series(
        {
            "600519.SH": 0.40,    # held 500, target 250 -> sell 200
            "000002.SZ": 0.0,     # full liquidation
            "000001.SZ": 0.02,    # new buy
            "600000.SH": 1e-9,    # negligible weight, no position -> silently ignored
            "300750.SZ": 0.10,    # NaN price -> rejection
            "002594.SZ": 0.0005,  # 2 shares -> below one lot -> rejection
        }
    )
    prices = pd.Series(
        {
            "600519.SH": 1_600.0,
            "000002.SZ": 9.0,
            "000001.SZ": 12.0,
            "600000.SH": 7.0,
            "300750.SZ": float("nan"),
            "002594.SZ": 250.0,
        }
    )
    intents = manager.target_weights_to_order_intents(
        target_weights=target,
        prices=prices,
        nav=1_000_000.0,
        signal_id="knob-fixture",
        model_version="m1",
        feature_version="f1",
    )
    orders = sorted(
        (i.symbol, i.side.value, i.quantity, round(i.target_weight, 10), i.reference_price)
        for i in intents
    )
    rejections = sorted(
        (r["symbol"], r["side"], r["quantity"], r["reason"], round(r["delta_value"], 6))
        for r in manager.last_skipped_orders
    )
    return orders, rejections


def test_order_manager_config_has_no_participation_knob() -> None:
    """A knob no code reads is worse than no knob: it reads as a live constraint."""
    fields = {f.name for f in dataclasses.fields(OrderManagerConfig)}

    assert "max_participation_rate" not in fields, (
        "OrderManagerConfig must not re-declare a participation limit it never "
        "reads; the enforced one is "
        "ExecutionConstraintSet.max_single_stock_participation_rate"
    )


def test_order_manager_never_declares_or_reads_a_participation_rate() -> None:
    """Guard the source itself, not only the dataclass surface.

    The identifier may still appear in the explanatory comment that records why
    the field is absent; what must not come back is a declaration or a read.
    """
    import inspect

    from quantagent.execution import order_manager

    source = inspect.getsource(order_manager)

    assert "max_participation_rate:" not in source, "field declaration is back"
    assert "config.max_participation_rate" not in source, "a read is back"
    assert "self.max_participation_rate" not in source, "a read is back"


def test_routing_is_byte_identical_to_the_pre_removal_tree() -> None:
    """Same inputs, same orders and same rejections as before the field was removed.

    These exact values were captured on the pre-removal tree, and were identical
    for ``max_participation_rate`` in {0.0, 0.05, 0.99}.
    """
    orders, rejections = _routed_fixture()

    assert orders == [
        ("000001.SZ", "buy", 1_600, 0.02, 12.0),
        ("000002.SZ", "sell", 1_000, 0.0, 9.0),
        ("600519.SH", "sell", 200, 0.4, 1_600.0),
    ]
    assert rejections == [
        ("002594.SZ", "buy", 0, "skipped_below_min_lot", 500.0),
        ("300750.SZ", "buy", 0, "skipped_invalid_price", 0.0),
    ]


def test_the_enforced_participation_limit_is_the_constraint_set_one() -> None:
    """The removal took away a no-op, not an enforcement."""
    from quantagent.execution.constraints import ExecutionConstraintSet

    fields = {f.name for f in dataclasses.fields(ExecutionConstraintSet)}

    assert "max_single_stock_participation_rate" in fields
    assert ExecutionConstraintSet().max_single_stock_participation_rate == 0.10
