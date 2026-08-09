from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quantagent.execution.order_manager import OrderManager, OrderManagerConfig


class _Broker:
    def query_positions(self):
        return []


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


@pytest.mark.parametrize("shares", [200, 201, 237])
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
