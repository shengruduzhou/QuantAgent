from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.execution.intraday_ev_engine import IntradayModelSignals
from quantagent.research.intraday_dot_ev_backtest import (
    EVBacktestConfig,
    build_feature_label_table,
    simulate_symbol_day,
)


def _favorable_signal() -> IntradayModelSignals:
    # strong reverse-T edge; buyback only fires once price has actually fallen
    return IntradayModelSignals(
        p_sell_high_success=0.9,
        expected_sell_high_gain_bps=300.0,
        p_fail_new_high=0.05,
        expected_chase_loss_bps=10.0,
        p_buyback_now=0.5,
        expected_buyback_edge_bps=0.0,
        wait_extra_edge_bps=0.0,
        miss_rebound_risk_bps=5.0,
        p_buy_low_success=0.05,
        expected_buy_low_gain_bps=0.0,
        p_fail_breakdown=0.1,
        expected_breakdown_loss_bps=10.0,
        p_sell_after_buy_success=0.05,
        expected_sell_after_buy_edge_bps=0.0,
        p_eod_restore=0.1,
        risk_score=0.1,
        model_version="test",
    )


def _day(n: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    # price sits high (10.5) then falls to 10.0 -> a clean reverse-T window
    px = np.concatenate([
        np.full(10, 10.50),
        np.linspace(10.50, 10.00, 15),
        np.full(n - 25, 10.00),
    ])
    t0 = pd.Timestamp("2026-05-06 09:31:00")
    times = [t0 + pd.Timedelta(minutes=i) for i in range(n)]
    bars = pd.DataFrame({
        "symbol": "000001.SZ",
        "trade_date": pd.Timestamp("2026-05-06"),
        "trade_time": times,
        "open": px, "high": px * 1.001, "low": px * 0.999, "close": px,
        "volume": 1_000_000.0,
        "limit_up": 11.5, "limit_down": 9.4,
    })
    table = pd.DataFrame({
        "symbol": "000001.SZ",
        "trade_date": pd.Timestamp("2026-05-06"),
        "trade_time": times,
        "close": px,
        "estimated_spread_bps": 6.0,
        "rolling_volatility_20m": 0.01,
        "rolling_return_20m": -0.001,
        "volume_capacity_ratio": 0.0,
        "one_way_trend_probability": 0.1,
        "mean_reversion_probability": 0.7,
        "near_limit_risk": 0.0,
        "limit_up_distance": 0.09,
        "limit_down_distance": 0.06,
    })
    return bars, table


def test_closed_loop_completes_legal_profitable_reverse_t():
    bars, table = _day()
    signals = [_favorable_signal() for _ in range(len(table))]
    rows = simulate_symbol_day(day_bars=bars, day_table=table, signals=signals,
                               cfg=EVBacktestConfig(), n_names=1, regime="sideways")
    assert rows, "engine should open and close at least one reverse-T"
    closed = [r for r in rows if r["completed_round_trip"] == 1]
    assert closed, "a completed round trip is expected"
    # selling high near 10.5 and buying back near 10.0 must net positive after cost
    assert any(r["net_pnl_bps"] > 0 for r in closed)
    assert all(r["action"] in {"BUY_BACK", "SELL_AFTER_BUY"} for r in closed)


def test_no_trade_when_signals_are_flat():
    bars, table = _day()
    flat = IntradayModelSignals()  # all zeros -> no positive EV
    rows = simulate_symbol_day(day_bars=bars, day_table=table, signals=[flat] * len(table),
                               cfg=EVBacktestConfig(), n_names=1, regime="sideways")
    assert all(r.get("eod_restore", 0) == 1 for r in rows) or rows == []


def test_feature_label_table_aligns_rows():
    bars, _ = _day()
    bars = bars.assign(amount=bars["close"] * bars["volume"], pre_close=10.45)
    table = build_feature_label_table(bars, EVBacktestConfig(horizon_minutes=10))
    assert len(table) == len(bars)
    assert "label_sell_high_gross_edge_bps" in table.columns
    assert "price_vs_vwap_z" in table.columns
