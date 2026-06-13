"""Tests for the selective 做T FSM — gates, honest fills, both leg modes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.execution.selective_dot import (
    DayContext,
    SelectiveDotParams,
    build_day_contexts,
    check_gates,
    simulate_selective_dot_day,
)


def _bars(prices: list[tuple[str, float, float, float, float]], volume: float = 1000.0) -> pd.DataFrame:
    rows = [{"trade_time": f"2026-06-01 {t}", "open": o, "high": h, "low": l, "close": c,
             "volume": volume, "symbol": "000001", "trade_date": "2026-06-01"}
            for t, o, h, l, c in prices]
    return pd.DataFrame(rows)


def _flat_then_dip_then_rally() -> pd.DataFrame:
    rows = []
    for i in range(10):  # flat 10.0 → VWAP settles at 10
        rows.append((f"09:{31+i}:00", 10.0, 10.01, 9.99, 10.0))
    rows.append(("09:41:00", 10.0, 10.0, 9.55, 9.6))    # dip −4%
    for i in range(10):  # rally to 10.4
        px = 9.6 + 0.08 * (i + 1)
        rows.append((f"09:{42+i}:00", px, px + 0.02, px - 0.02, px))
    for i in range(5):
        rows.append((f"14:{50+i}:00", 10.4, 10.42, 10.38, 10.4))
    return _bars(rows)


CTX_OK = DayContext(atr_pct=0.05, mom_5d=0.03, gap_open=0.0, regime="bull")


class TestGates:
    def test_low_vol_gated(self):
        mode, reason = check_gates(DayContext(0.005, 0.03, 0.0, "bull"), SelectiveDotParams())
        assert mode is None and reason == "low_vol"

    def test_bear_regime_gated(self):
        mode, reason = check_gates(DayContext(0.05, 0.03, 0.0, "bear"), SelectiveDotParams())
        assert mode is None and reason == "regime_bear"

    def test_extreme_gap_gated(self):
        mode, reason = check_gates(DayContext(0.05, 0.03, 0.08, "bull"), SelectiveDotParams())
        assert mode is None and reason == "extreme_gap"

    def test_weak_trend_blocks_dip_buy(self):
        p = SelectiveDotParams(mode="dip_buy")
        mode, reason = check_gates(DayContext(0.05, -0.02, 0.0, "bull"), p)
        assert mode is None and reason == "weak_trend"

    def test_auto_routes_by_trend(self):
        p = SelectiveDotParams(mode="auto")
        assert check_gates(DayContext(0.05, 0.03, 0.0, "bull"), p)[0] == "dip_buy"
        assert check_gates(DayContext(0.05, -0.03, 0.0, "bull"), p)[0] == "spike_sell"


class TestDipBuy:
    def test_round_trip_profit(self):
        res = simulate_selective_dot_day(_flat_then_dip_then_rally(), CTX_OK,
                                         SelectiveDotParams(mode="dip_buy", dip_atr_mult=0.3,
                                                            target_atr_mult=0.5, stop_atr_mult=0.5))
        assert res.state == "closed_profit"
        assert res.mode == "dip_buy"
        assert res.ret == pytest.approx(0.025, abs=1e-3)  # target = entry·(1+0.5·5%)

    def test_entry_never_better_than_bar_open(self):
        # bar opens BELOW the trigger → limit buy fills at the open, not the trigger
        res = simulate_selective_dot_day(_flat_then_dip_then_rally(), CTX_OK,
                                         SelectiveDotParams(mode="dip_buy", dip_atr_mult=0.3))
        # dip bar opens at 10.0, trigger ≈ 10·(1−0.015)=9.85 → open(10.0) > trig → fill at trig
        assert res.entry_px == pytest.approx(10.0 * (1 - 0.3 * 0.05), abs=0.02)

    def test_stop_gaps_through_at_open(self):
        rows = []
        for i in range(10):
            rows.append((f"09:{31+i}:00", 10.0, 10.01, 9.99, 10.0))
        rows.append(("09:41:00", 9.85, 9.85, 9.80, 9.82))   # entry trigger bar
        rows.append(("09:42:00", 9.0, 9.05, 8.95, 9.0))     # gap DOWN through the stop
        for i in range(3):
            rows.append((f"14:{50+i}:00", 9.0, 9.02, 8.98, 9.0))
        res = simulate_selective_dot_day(_bars(rows), CTX_OK,
                                         SelectiveDotParams(mode="dip_buy", dip_atr_mult=0.3,
                                                            stop_atr_mult=0.5))
        assert res.state == "closed_stop"
        assert res.exit_px == pytest.approx(9.0, abs=1e-6)  # filled at the gap open, not the stop level
        assert res.ret < -0.05

    def test_no_entry_when_no_dip(self):
        rows = [(f"09:{31+i}:00", 10.0, 10.05, 9.99, 10.02) for i in range(20)]
        res = simulate_selective_dot_day(_bars(rows), CTX_OK,
                                         SelectiveDotParams(mode="dip_buy", dip_atr_mult=0.5))
        assert res.state == "waiting_no_entry"


class TestSpikeSell:
    def test_round_trip_profit_sign(self):
        rows = []
        for i in range(10):
            rows.append((f"09:{31+i}:00", 10.0, 10.01, 9.99, 10.0))
        rows.append(("09:41:00", 10.0, 10.45, 10.0, 10.4))   # spike +4% → sell
        for i in range(10):                                   # fade back down
            px = 10.4 - 0.08 * (i + 1)
            rows.append((f"09:{42+i}:00", px, px + 0.02, px - 0.02, px))
        for i in range(3):
            rows.append((f"14:{50+i}:00", 9.6, 9.62, 9.58, 9.6))
        ctx = DayContext(atr_pct=0.05, mom_5d=-0.03, gap_open=0.0, regime="sideways")
        res = simulate_selective_dot_day(_bars(rows), ctx,
                                         SelectiveDotParams(mode="spike_sell", dip_atr_mult=0.3,
                                                            target_atr_mult=0.5, stop_atr_mult=0.5))
        assert res.state == "closed_profit"
        assert res.ret is not None and res.ret > 0.02  # sell high / buy back low


class TestDayContexts:
    def test_atr_and_mom_are_pit_safe(self):
        dates = pd.date_range("2026-01-05", periods=30, freq="B")
        rows = []
        for i, d in enumerate(dates):
            base = 10.0 + 0.1 * i
            rows.append({"symbol": "000001", "trade_date": d, "open": base,
                         "high": base * 1.02, "low": base * 0.98, "close": base})
        panel = pd.DataFrame(rows)
        ctx = build_day_contexts(panel)
        last = ctx.iloc[-1]
        # mom_5d must use closes up to t−1 only: close(t−1)/close(t−6)−1
        expect = (10.0 + 0.1 * 28) / (10.0 + 0.1 * 23) - 1.0
        assert last["mom_5d"] == pytest.approx(expect, abs=1e-9)
        assert np.isfinite(last["atr_pct"]) and last["atr_pct"] > 0
