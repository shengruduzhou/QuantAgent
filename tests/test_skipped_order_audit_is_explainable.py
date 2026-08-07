"""Every skipped order must name a reason an operator can act on.

A 400-name pilot backtest reported 2,291 skipped orders against 157 executed
ones — it read as though A-share execution friction had blocked 93% of the
book's trading. It had not:

* 1,971 were symbols the book neither held nor wanted. The order loop walks
  every priced symbol, so a zero weight with a zero position took the buy
  branch, rounded to zero shares, and was filed as ``skipped_invalid_lot``.
* 320 carried real intent, but 315 of those implied fewer than one 100-share
  lot — median 0.0025 shares, from target weights around 1e-7.

Neither is a venue rule. Both were reported under a name that suggested one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.execution.order_manager import OrderManager, OrderManagerConfig


class _Broker:
    def query_positions(self):
        return []


@pytest.fixture
def manager():
    return OrderManager(broker=_Broker(), config=OrderManagerConfig())


def _skips(manager, weights, prices, nav=1_000_000.0):
    manager.target_weights_to_order_intents(
        target_weights=pd.Series(weights),
        prices=pd.Series(prices),
        nav=nav,
    )
    return pd.DataFrame(manager.last_skipped_orders)


def test_no_holding_and_no_target_produces_no_audit_row_at_all(manager):
    """The overwhelming majority of the old "skips" were this case."""
    skipped = _skips(
        manager,
        weights={"000001.SZ": 0.0, "000002.SZ": 0.0, "600000.SH": 0.0},
        prices={"000001.SZ": 12.0, "000002.SZ": 20.0, "600000.SH": 8.0},
    )
    assert skipped.empty, "absence of intent is not a skipped order"


def test_dust_weights_below_the_floor_are_not_audited_as_skips(manager):
    """1e-7 weights are what rank weighting leaves behind, not a decision.

    These made up the bulk of the 320 "real intent" skips: median implied size
    0.0025 shares. Treating them as intent produced an audit trail nobody could
    act on.
    """
    skipped = _skips(
        manager,
        weights={"000001.SZ": 1e-7},
        prices={"000001.SZ": 200.0},
    )
    assert skipped.empty


def test_a_sub_lot_intent_is_named_and_carries_its_implied_size(manager):
    # 1e-3 of a 1,000,000 NAV at 200 CNY is 5 shares: a real allocation the
    # 100-share lot rule cannot express.
    skipped = _skips(
        manager,
        weights={"000001.SZ": 1e-3},
        prices={"000001.SZ": 200.0},
    )

    assert len(skipped) == 1
    row = skipped.iloc[0]
    assert row["reason"] == "skipped_below_min_lot"
    assert row["implied_shares"] == pytest.approx(5.0, rel=1e-6)
    assert row["target_weight"] == pytest.approx(1e-3)
    # The old name claimed a lot-validity rule had rejected it.
    assert row["reason"] != "skipped_invalid_lot"


def test_a_tradable_weight_still_produces_a_real_order(manager):
    intents = manager.target_weights_to_order_intents(
        target_weights=pd.Series({"000001.SZ": 0.10}),
        prices=pd.Series({"000001.SZ": 20.0}),
        nav=1_000_000.0,
    )

    assert len(intents) == 1
    assert intents[0].quantity == 5000  # 100,000 CNY / 20, whole lots
    assert not manager.last_skipped_orders


def test_the_audit_separates_absence_of_intent_from_untradable_intent(manager):
    """The mix is what makes the report readable."""
    skipped = _skips(
        manager,
        weights={
            "000001.SZ": 0.0,     # no intent -> no row
            "000002.SZ": 1e-9,    # dust below the floor -> no row
            "600000.SH": 1e-3,    # real but sub-lot -> one row
            "600519.SH": 0.10,    # tradable -> order, no row
        },
        prices={
            "000001.SZ": 12.0, "000002.SZ": 20.0,
            "600000.SH": 200.0, "600519.SH": 50.0,
        },
    )

    assert list(skipped["reason"]) == ["skipped_below_min_lot"]
    assert list(skipped["symbol"]) == ["600000.SH"]


def test_an_unpriced_symbol_is_still_reported_distinctly(manager):
    skipped = _skips(
        manager,
        weights={"000001.SZ": 0.10},
        prices={"000001.SZ": float("nan")},
    )

    assert list(skipped["reason"]) == ["skipped_invalid_price"]
