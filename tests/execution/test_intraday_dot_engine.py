"""Tests for the 分时做T 因果引擎 + 决策层（T+1合法性、竞价、失败控制、JSON）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.execution.intraday_dot_engine import compute_intraday_state
from quantagent.execution.intraday_dot_decision import DecisionConfig, Position, decide


def _bars(seq):
    """seq: list of (hms, close[, vol]). high/low default to close±0.5%."""
    rows = []
    for item in seq:
        t, c = item[0], item[1]
        v = item[2] if len(item) > 2 else 1000.0
        rows.append({"trade_time": f"2026-06-01 {t}", "close": c,
                     "high": c * 1.002, "low": c * 0.998, "volume": v, "amount": c * v})
    return pd.DataFrame(rows)


def _flat_then_dip(n_flat=12):
    seq = [(f"09:{31+i}:00", 10.0) for i in range(n_flat)]
    seq.append(("09:45:00", 9.55, 3000))       # sharp dip below low_line, volume surge
    for i in range(8):
        seq.append((f"09:{46+i}:00", 9.6 + 0.02 * i))
    return _bars(seq)


def _flat_then_spike(n_flat=12):
    seq = [(f"09:{31+i}:00", 10.0) for i in range(n_flat)]
    seq.append(("09:45:00", 10.45, 3000))      # spike above high_line
    for i in range(8):
        seq.append((f"09:{46+i}:00", 10.4 - 0.02 * i))
    return _bars(seq)


class TestEngineCausality:
    def test_vwap_and_band_finite(self):
        st = compute_intraday_state(_flat_then_dip(), pre_close=10.0)
        assert st is not None
        assert 0.006 <= st.band <= 0.028
        assert st.low_line < st.vwap < st.high_line

    def test_dip_drives_low_score_up(self):
        st = compute_intraday_state(_flat_then_dip(), pre_close=10.0)
        # right at the dip bar the low score should dominate the high score
        dip_only = compute_intraday_state(_flat_then_dip()[:14], pre_close=10.0)
        assert dip_only.low_score >= dip_only.high_score

    def test_no_bars_returns_none(self):
        assert compute_intraday_state(pd.DataFrame(), pre_close=10.0) is None
        assert compute_intraday_state(_flat_then_dip(), pre_close=0.0) is None


class TestPhaseRouting:
    def test_pre_open_waits(self):
        d = decide(None, Position(), symbol="000001", current_time="2026-06-01 09:10:00",
                   pre_close=10.0, limit_up=11.0, limit_down=9.0)
        assert d["action"] == "WAIT" and d["market_phase"] == "pre_open"

    def test_auction_observe_no_order(self):
        st = compute_intraday_state(_flat_then_dip(), pre_close=10.0)
        d = decide(st, Position(sellable_qty=1000), symbol="000001",
                   current_time="2026-06-01 09:17:00", pre_close=10.0,
                   limit_up=11.0, limit_down=9.0)
        assert d["action"] == "WAIT" and d["side"] == "NONE"


class TestT1Legality:
    def test_sell_high_only_uses_sellable(self):
        st = compute_intraday_state(_flat_then_spike()[:14], pre_close=10.0)
        d = decide(st, Position(total_qty=2000, sellable_qty=1000, today_buy_qty=1000),
                   symbol="000001", current_time="2026-06-01 09:45:00",
                   pre_close=10.0, limit_up=11.0, limit_down=9.0,
                   config=DecisionConfig(conf_execute=0.0))
        if d["action"] == "SELL_HIGH":
            assert d["qty"] <= 1000                      # never sells today_buy_qty
            assert d["legal_check"]["sellable_qty_ok"]
            assert d["qty"] % 100 == 0

    def test_no_sellable_cannot_sell_high(self):
        st = compute_intraday_state(_flat_then_spike()[:14], pre_close=10.0)
        d = decide(st, Position(total_qty=1000, sellable_qty=0, today_buy_qty=1000),
                   symbol="000001", current_time="2026-06-01 09:45:00",
                   pre_close=10.0, limit_up=11.0, limit_down=9.0,
                   config=DecisionConfig(conf_execute=0.0))
        assert d["action"] != "SELL_HIGH"               # no yesterday inventory → cannot 高抛

    def test_buy_respects_cash(self):
        st = compute_intraday_state(_flat_then_dip()[:14], pre_close=10.0)
        d = decide(st, Position(total_qty=1000), symbol="000001",
                   current_time="2026-06-01 09:45:00", pre_close=10.0,
                   limit_up=11.0, limit_down=9.0, cash=500.0,
                   config=DecisionConfig(conf_execute=0.0))
        if d["side"] == "BUY":
            assert d["qty"] * d["limit_price"] <= 500.0 + 1e-6
            assert d["legal_check"]["cash_ok"]


class TestLimitConstraints:
    def test_limit_down_no_sellable_rejects(self):
        seq = [(f"09:{31+i}:00", 10.0) for i in range(8)]
        seq += [(f"09:{40+i}:00", 9.05) for i in range(6)]   # near limit-down 9.0
        st = compute_intraday_state(_bars(seq), pre_close=10.0)
        d = decide(st, Position(total_qty=0, sellable_qty=0), symbol="000001",
                   current_time="2026-06-01 09:46:00", pre_close=10.0,
                   limit_up=11.0, limit_down=9.0)
        assert d["action"] in ("REJECT", "SELL_RISK", "HOLD")
        if d["action"] == "REJECT":
            assert not d["legal_check"]["sellable_qty_ok"]

    def test_close_auction_only_adjusts(self):
        st = compute_intraday_state(_flat_then_dip(), pre_close=10.0)
        d = decide(st, Position(total_qty=1000, sellable_qty=1000), symbol="000001",
                   current_time="2026-06-01 14:58:00", pre_close=10.0,
                   limit_up=11.0, limit_down=9.0)
        assert d["market_phase"] == "close_auction"
        assert d["action"] in ("HOLD", "SELL_RISK", "BUY_BACK")


class TestJSONSchema:
    def test_output_has_required_keys(self):
        st = compute_intraday_state(_flat_then_dip(), pre_close=10.0)
        d = decide(st, Position(total_qty=1000, sellable_qty=1000), symbol="000001",
                   current_time="2026-06-01 10:30:00", pre_close=10.0,
                   limit_up=11.0, limit_down=9.0, cash=100000.0)
        for k in ("symbol", "time", "market_phase", "action", "side", "qty",
                  "price_type", "limit_price", "confidence", "t_pair_id", "is_t_trade",
                  "legal_check", "reason", "risk_flags", "failure_control", "next_watch_levels"):
            assert k in d
        assert d["action"] in ("BUY_LOW", "SELL_HIGH", "BUY_BACK", "SELL_RISK",
                               "HOLD", "WAIT", "REJECT")
        assert isinstance(d["reason"], list) and d["reason"]
