from __future__ import annotations

import pandas as pd

from quantagent.portfolio.do_t_overlay import DoTOverlayConfig, simulate_do_t_overlay


def _minute_rows(prices):
    start = pd.Timestamp("2024-01-02 09:30")
    return pd.DataFrame([
        {
            "trade_date": pd.Timestamp("2024-01-02"),
            "symbol": "000001.SZ",
            "datetime": start + pd.Timedelta(minutes=i),
            "close": price,
        }
        for i, price in enumerate(prices)
    ])


def test_do_t_overlay_requires_available_base_position():
    minutes = _minute_rows([10.0, 10.5, 10.8, 10.1, 9.8, 10.0])
    inventory = pd.DataFrame([
        {"trade_date": pd.Timestamp("2024-01-02"), "symbol": "000001.SZ", "available_shares": 0}
    ])

    out = simulate_do_t_overlay(minutes, inventory)

    assert out.empty


def test_do_t_overlay_sells_old_shares_then_buys_back_low():
    minutes = _minute_rows([10.0, 10.8, 10.9, 10.2, 9.8, 9.9, 10.0])
    inventory = pd.DataFrame([
        {"trade_date": pd.Timestamp("2024-01-02"), "symbol": "000001.SZ", "available_shares": 1000}
    ])

    out = simulate_do_t_overlay(
        minutes,
        inventory,
        config=DoTOverlayConfig(trade_fraction=0.5, min_edge_pct=0.05, min_minutes_between_legs=2),
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert row["mode"] == "sell_high_buy_low"
    assert row["quantity"] == 500
    assert bool(row["t1_legal"]) is True
    assert row["sell_price"] > row["buy_price"]
    assert row["net_pnl"] > 0


def test_do_t_overlay_can_buy_low_then_sell_old_shares_high():
    minutes = _minute_rows([10.0, 9.6, 9.5, 10.1, 10.5, 10.7])
    inventory = pd.DataFrame([
        {"trade_date": pd.Timestamp("2024-01-02"), "symbol": "000001.SZ", "available_shares": 1000}
    ])

    out = simulate_do_t_overlay(
        minutes,
        inventory,
        config=DoTOverlayConfig(trade_fraction=0.3, min_edge_pct=0.05, min_minutes_between_legs=2),
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert row["mode"] == "buy_low_sell_old_high"
    assert row["quantity"] == 300
    assert bool(row["t1_legal"]) is True
    assert row["sell_time"] > row["buy_time"]
    assert row["net_pnl"] > 0
