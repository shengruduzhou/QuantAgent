"""Golden-trade tests for the canonical strict A-share engine (round-29 R1-F03/F04).

Hand-computed expectations are written out in each test: dated statutory stamp
duty, and a rotation whose outcome must not depend on ticker sort order.
"""
from __future__ import annotations

import math
import os
import tempfile

import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8
from quantagent.market_rules import ashare as rules

AUDIT = tempfile.mkdtemp(prefix="strict_golden_audit_")


def _panel(dates, symbols, close=10.0, volume=1e9):
    rows = []
    for d in dates:
        for s in symbols:
            rows.append(dict(symbol=s, trade_date=pd.Timestamp(d), open=close, high=close, low=close,
                             close=close, volume=volume, amount=volume * close, is_suspended=False,
                             is_st=False, is_limit_up=False, is_limit_down=False))
    return pd.DataFrame(rows)


def _tw(rows):
    tw = pd.DataFrame(rows).set_index("trade_date").fillna(0.0)
    tw.index = pd.to_datetime(tw.index)
    tw.attrs["target_index_semantics"] = "signal_date"
    return tw


def _cfg():
    return AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=8.0, audit_log_dir=AUDIT)


def _round_trip(dates):
    panel = _panel(dates, ["600000.SH"])
    tw = _tw([{"trade_date": dates[0], "600000.SH": 0.5}, {"trade_date": dates[1], "600000.SH": 0.0}])
    res = run_strict_backtest_v8(tw, panel, config=_cfg())
    t = res.trades
    return t[t["side"].str.lower() == "buy"].iloc[0], t[t["side"].str.lower() == "sell"].iloc[0], res


def test_golden_buy_leg_hand_computed():
    # signal 2022-03-01 -> buy at 2022-03-02 close 10.00; 0.5*1e6/10 = 50,000 sh (lot ok)
    buy, _, _ = _round_trip(["2022-03-01", "2022-03-02", "2022-03-03", "2022-03-04"])
    assert int(buy["filled_quantity"]) == 50_000
    assert buy["avg_price"] == pytest.approx(10.008)          # 10 * (1 + 8bps)
    value = 50_000 * 10.008                                    # 500,400.00
    assert buy["commission"] == pytest.approx(value * 0.0003)  # 150.12 (engine uses 3 bps)
    assert buy["transfer_fee"] == pytest.approx(value * 0.00001)
    assert buy["stamp_duty"] == 0.0
    part = 50_000 / 1e9
    assert buy["impact_cost"] == pytest.approx(value * 10 * math.sqrt(part) / 1e4)


def test_stamp_duty_pre_2023_08_28_is_statutory_0p10pct():
    # sell executed 2022-03-03 at 9.992 -> notional 499,600; statutory stamp 0.10% = 499.60
    _, sell, _ = _round_trip(["2022-03-01", "2022-03-02", "2022-03-03", "2022-03-04"])
    notional = float(sell["filled_quantity"]) * float(sell["avg_price"])
    assert notional == pytest.approx(499_600.0)
    assert rules.stamp_duty_rate("2022-03-03") == 0.0010      # the repo's own dated rule
    assert sell["stamp_duty"] == pytest.approx(notional * rules.stamp_duty_rate("2022-03-03"))


def test_stamp_duty_post_2023_08_28_is_0p05pct():
    _, sell, _ = _round_trip(["2023-09-01", "2023-09-04", "2023-09-05", "2023-09-06"])
    notional = float(sell["filled_quantity"]) * float(sell["avg_price"])
    assert sell["stamp_duty"] == pytest.approx(notional * 0.0005)


def test_rotation_outcome_must_not_depend_on_symbol_sort_order():
    """Full rotation B->A. When the new name sorts BEFORE the old one, the buy is
    submitted before the sell that funds it and is rejected `insufficient_cash`;
    with the names swapped the same economic rotation fills. The two books must
    end in the same state (either both rotate, or both are cash-constrained)."""
    dates = ["2022-03-01", "2022-03-02", "2022-03-03", "2022-03-04"]

    def run(old, new):
        panel = _panel(dates, [old, new])
        tw = _tw([{"trade_date": dates[0], old: 0.98, new: 0.0},
                  {"trade_date": dates[1], old: 0.0, new: 0.98}])
        res = run_strict_backtest_v8(tw, panel, config=_cfg())
        t = res.trades
        new_filled = t[(t["symbol"] == new) & (t["side"].str.lower() == "buy")]["filled_quantity"].sum()
        rejected = t[t["last_message"].astype(str) == "insufficient_cash"]
        return float(new_filled), len(rejected)

    a_filled, a_rej = run(old="000001.SZ", new="600000.SH")   # sell sorts first -> rotation fills
    b_filled, b_rej = run(old="600000.SH", new="000001.SZ")   # buy sorts first -> rejected
    print("old<new:", a_filled, a_rej, " new<old:", b_filled, b_rej)
    assert (a_filled, a_rej) == (b_filled, b_rej)
