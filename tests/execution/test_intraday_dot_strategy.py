"""Tests for the causal 做T FSM (executable, not hindsight)."""

from __future__ import annotations

import pandas as pd

from quantagent.execution.intraday_dot_strategy import (
    DotParams, live_dot_action, simulate_dot_day,
)


def _bars(rows, date="2026-05-08", sym="300308.SZ"):
    # rows: list of (hhmmss, open, high, low, close, volume)
    return pd.DataFrame([
        {"symbol": sym, "trade_date": date, "trade_time": f"{date} {hh}",
         "open": o, "high": h, "low": l, "close": c, "volume": v}
        for hh, o, h, l, c, v in rows
    ])


def test_dip_entry_then_profit():
    # flat ~10.0 early (builds VWAP), dip to 9.97 in morning → entry ~9.98, then rally to target
    rows = [(f"09:{m:02d}:00", 10.0, 10.0, 10.0, 10.0, 100) for m in range(31, 40)]
    rows += [("09:40:00", 10.0, 10.0, 9.95, 9.97, 200)]   # dip below running VWAP → entry
    rows += [("10:30:00", 9.98, 10.3, 9.98, 10.25, 300)]  # high 10.3 >= target 9.98*1.015=10.13
    res = simulate_dot_day(_bars(rows), DotParams(target_pct=0.015, stop_pct=0.012))
    assert res.entry_px is not None and res.entry_px < 10.0
    assert res.state == "closed_profit" and res.exit_reason == "止盈"
    assert res.ret > 0


def test_dip_entry_then_stop():
    rows = [(f"09:{m:02d}:00", 10.0, 10.0, 10.0, 10.0, 100) for m in range(31, 40)]
    rows += [("09:40:00", 10.0, 10.0, 9.95, 9.97, 200)]   # entry ~9.95-9.98
    rows += [("11:00:00", 9.97, 9.97, 9.70, 9.72, 300)]   # low 9.70 <= stop ~9.86 → 止损
    res = simulate_dot_day(_bars(rows), DotParams(target_pct=0.015, stop_pct=0.012))
    assert res.state == "closed_stop" and res.ret < 0


def test_eod_force_close():
    rows = [(f"09:{m:02d}:00", 10.0, 10.0, 10.0, 10.0, 100) for m in range(31, 40)]
    rows += [("09:40:00", 10.0, 10.0, 9.95, 9.97, 200)]   # entry
    rows += [("14:00:00", 9.97, 10.02, 9.96, 10.0, 100)]  # neither target nor stop
    rows += [("14:55:00", 10.0, 10.05, 9.99, 10.01, 100)]  # past eod_close → force
    res = simulate_dot_day(_bars(rows), DotParams())
    assert res.state == "closed_eod" and res.exit_reason == "尾盘强平"


def test_no_entry_when_no_dip():
    # monotonically rising, never pulls back below running VWAP in the morning
    rows = [(f"09:{m:02d}:00", 10.0 + i * 0.02, 10.05 + i * 0.02, 10.0 + i * 0.02, 10.03 + i * 0.02, 100)
            for i, m in enumerate(range(31, 50))]
    res = simulate_dot_day(_bars(rows), DotParams())
    assert res.state == "waiting_no_entry" and res.entry_px is None


def test_morning_window_gate():
    # the dip happens AFTER 10:00 → must NOT enter (低吸只在早盘窗口)
    rows = [(f"09:{m:02d}:00", 10.0, 10.0, 10.0, 10.0, 100) for m in range(31, 40)]
    rows += [("11:00:00", 10.0, 10.0, 9.90, 9.95, 200)]   # dip but past morning_deadline
    res = simulate_dot_day(_bars(rows), DotParams(morning_deadline="10:00:00"))
    assert res.entry_px is None


def test_live_action_emits_executable():
    rows = [(f"09:{m:02d}:00", 10.0, 10.0, 10.0, 10.0, 100) for m in range(31, 40)]
    rows += [("09:40:00", 10.0, 10.0, 9.95, 9.97, 200)]
    act = live_dot_action(_bars(rows), in_position=False)
    assert act["action"] in ("加T买入", "观望")
