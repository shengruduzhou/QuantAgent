from __future__ import annotations

import pandas as pd
import pytest

from quantagent.backtest.paper_report import (
    PaperReportConfig,
    _pnl_frame,
    _selected_stocks,
    _summary,
    _trade_frame,
)


def test_nav_costs_are_not_deducted_twice() -> None:
    pnl = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "nav": [1_000.0, 990.0],
            "daily_return": [0.0, -0.01],
            "drawdown": [0.0, -0.01],
        }
    )
    trades = pd.DataFrame(
        {
            "status": ["filled"],
            "estimated_fee": [5.0],
            "estimated_slippage": [5.0],
        }
    )

    summary = _summary(pnl, trades, pd.DataFrame(), pd.DataFrame(), 1_000.0)

    assert summary["net_return_after_estimated_costs"] == pytest.approx(-0.01)
    assert summary["return_before_estimated_costs"] == pytest.approx(0.0)
    assert summary["estimated_costs_already_in_nav"] is True


def test_symbol_pnl_does_not_double_charge_fill_price_slippage() -> None:
    trades = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-01-02")],
            "symbol": ["A"],
            "side": ["buy"],
            "status": ["filled"],
            "gross_amount": [100.0],
            "estimated_fee": [1.0],
            "estimated_slippage": [2.0],
        }
    )
    holdings = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-01-05")],
            "symbol": ["A"],
            "market_value": [100.0],
        }
    )

    row = _selected_stocks(trades, holdings).iloc[0]

    assert row["estimated_symbol_pnl"] == pytest.approx(-1.0)


def test_sparse_nav_annualises_by_elapsed_time_not_row_count() -> None:
    pnl = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-01-01", "2025-12-31"]),
            "nav": [1_000.0, 1_100.0],
            "daily_return": [0.0, 0.10],
            "drawdown": [0.0, 0.0],
        }
    )
    trades = pd.DataFrame(columns=["status", "estimated_fee", "estimated_slippage"])

    summary = _summary(pnl, trades, pd.DataFrame(), pd.DataFrame(), 1_000.0)

    assert 0.09 < summary["annualized_return"] < 0.11
    assert summary["annualization_elapsed_calendar_days"] == 365


def test_benchmark_is_rebased_to_first_execution_nav_date() -> None:
    nav = pd.Series(
        [1_000.0, 1_000.0],
        index=pd.to_datetime(["2026-01-06", "2026-01-07"]),
    )
    market = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        "symbol": ["000300.SH"] * 3,
        "close": [100.0, 110.0, 121.0],
    })

    pnl = _pnl_frame(nav, 1_000.0, market, "000300.SH")
    summary = _summary(
        pnl,
        pd.DataFrame(columns=["status", "estimated_fee", "estimated_slippage"]),
        pd.DataFrame(),
        pd.DataFrame(),
        1_000.0,
    )

    assert pnl["benchmark_return"].tolist() == pytest.approx([0.0, 0.1])
    assert summary["benchmark_return"] == pytest.approx(0.1)
    assert summary["excess_return"] == pytest.approx(-0.1)


def test_paper_trade_cost_uses_audited_square_root_impact() -> None:
    orders = pd.DataFrame({
        "filled_quantity": [10_000],
        "avg_price": [10.008],
        "reference_price": [10.0],
        "side": ["buy"],
        "total_cost": [36.65],
        "impact_cost": [31.65],
        "status": ["filled"],
    })

    trades = _trade_frame(orders, PaperReportConfig())

    assert trades.loc[0, "estimated_fee"] == pytest.approx(36.65)
    assert trades.loc[0, "estimated_slippage"] == pytest.approx(80.0)
