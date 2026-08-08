from __future__ import annotations

import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import (
    STRICT_CASH_ACCOUNT_SEMANTICS,
    UnsupportedStockShortError,
    simulate_ashare_target_weights,
    validate_cash_account_target_weights,
)
from quantagent.portfolio.alpha_portfolio import AlphaPortfolioConfig, build_alpha_portfolio


def test_cash_target_guard_accepts_long_and_zero_weights() -> None:
    targets = pd.DataFrame(
        {"600000.SH": [0.5, 0.0], "000001.SZ": [0.5, 1.0]},
        index=pd.to_datetime(["2026-01-05", "2026-01-06"]),
    )
    validate_cash_account_target_weights(targets)


def test_cash_target_guard_rejects_negative_final_stock_weight() -> None:
    targets = pd.DataFrame(
        {"600000.SH": [0.5], "000001.SZ": [-0.5]},
        index=pd.to_datetime(["2026-01-05"]),
    )
    with pytest.raises(UnsupportedStockShortError, match="cannot establish negative stock weights"):
        validate_cash_account_target_weights(targets)


def test_strict_cash_simulator_fails_before_short_leg_can_be_silently_dropped() -> None:
    date = pd.Timestamp("2026-01-05")
    targets = pd.DataFrame(
        {"600000.SH": [0.5], "000001.SZ": [-0.5]},
        index=[date],
    )
    market = pd.DataFrame(
        [
            {
                "trade_date": date,
                "symbol": symbol,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 1_000_000.0,
                "amount": 10_000_000.0,
                "is_suspended": False,
                "is_st": False,
                "is_limit_up": False,
                "is_limit_down": False,
            }
            for symbol in ("600000.SH", "000001.SZ")
        ]
    )
    with pytest.raises(UnsupportedStockShortError):
        simulate_ashare_target_weights(targets, market)


def test_research_long_short_constructor_may_exist_but_is_not_cash_executable() -> None:
    date = pd.Timestamp("2026-01-05")
    predictions = pd.DataFrame(
        {
            "trade_date": [date] * 10,
            "symbol": [f"60000{i}.SH" for i in range(10)],
            "alpha_score": list(range(10)),
        }
    )
    weights = build_alpha_portfolio(
        predictions,
        config=AlphaPortfolioConfig(
            book_fraction=0.2,
            max_name_weight=0.5,
            rebalance_interval=1,
            long_short=True,
            min_names_per_date=1,
        ),
    )
    assert (weights.to_numpy() < 0).any()
    with pytest.raises(UnsupportedStockShortError):
        validate_cash_account_target_weights(weights)


def test_successful_cash_simulation_stamps_execution_capability() -> None:
    date = pd.Timestamp("2026-01-05")
    targets = pd.DataFrame({"600000.SH": [0.0]}, index=[date])
    market = pd.DataFrame(
        [
            {
                "trade_date": date,
                "symbol": "600000.SH",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 1_000_000.0,
                "amount": 10_000_000.0,
                "is_suspended": False,
                "is_st": False,
                "is_limit_up": False,
                "is_limit_down": False,
            }
        ]
    )
    result = simulate_ashare_target_weights(targets, market)
    assert result.config["stock_shorting_capability"] == "cash_long_only"
    assert result.config["execution_semantics_version"] == STRICT_CASH_ACCOUNT_SEMANTICS
