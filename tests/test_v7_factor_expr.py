"""Tests for the V7 factor expression DSL."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.factors.expr import (
    Close,
    Delay,
    Delta,
    FactorRegistry,
    Rank,
    Returns,
    TsMean,
    TsMax,
    TsMin,
    TsRank,
    TsStd,
    TsSum,
    Volume,
    build_factor_frame,
    register_factor,
    default_registry,
)


def _toy_panel(num_days: int = 60, symbols: tuple[str, ...] = ("A", "B", "C")) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2024-01-02", periods=num_days, freq="B")
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for symbol in symbols:
        px = 100.0
        for date in dates:
            px *= 1 + rng.standard_normal() * 0.01
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": date,
                    "open": px,
                    "high": px * 1.01,
                    "low": px * 0.99,
                    "close": px,
                    "volume": 1000.0 + rng.standard_normal() * 50,
                    "amount": (1000.0 + rng.standard_normal() * 50) * px,
                }
            )
    return pd.DataFrame(rows)


def test_delay_shifts_per_symbol():
    df = _toy_panel(num_days=10, symbols=("A", "B"))
    delayed = Delay(Close, 1).evaluate(df)
    # First row per symbol must be NaN after the shift.
    sorted_df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    delayed_sorted = delayed.reindex(sorted_df.index)
    first_per_symbol = sorted_df.drop_duplicates("symbol", keep="first").index
    assert delayed_sorted.loc[first_per_symbol].isna().all()


def test_rank_is_cross_sectional_per_date():
    df = _toy_panel(num_days=5, symbols=("A", "B", "C"))
    ranks = Rank(Close).evaluate(df)
    df_with_rank = df.assign(rank=ranks.values)
    for _, day in df_with_rank.groupby("trade_date"):
        # Ranks within a single date are in (0, 1].
        assert day["rank"].between(0.0, 1.0).all()
        assert day["rank"].nunique() == 3


def test_ts_mean_uses_only_trailing_window_no_lookahead():
    df = _toy_panel(num_days=20, symbols=("A",))
    window = 5
    out = TsMean(Close, window).evaluate(df)
    sorted_df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    out_sorted = out.reindex(sorted_df.index).reset_index(drop=True)
    # The first window-1 values must be NaN because min_periods=window.
    assert out_sorted.iloc[: window - 1].isna().all()
    # Position window-1 onward must equal the trailing mean.
    expected_5 = sorted_df["close"].rolling(window).mean()
    pd.testing.assert_series_equal(out_sorted, expected_5, check_names=False)


def test_returns_consistent_with_close_delta():
    df = _toy_panel(num_days=10, symbols=("A",))
    returns = Returns(Close, 1).evaluate(df)
    sorted_df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    expected = sorted_df["close"].pct_change()
    pd.testing.assert_series_equal(
        returns.reindex(sorted_df.index).reset_index(drop=True),
        expected,
        check_names=False,
    )


def test_factor_registry_round_trips_long_and_wide():
    df = _toy_panel(num_days=30)
    registry = FactorRegistry()
    registry.register("momentum_5", Rank(TsMean(Returns(Close, 1), 5)))
    registry.register("volatility_10", Rank(TsStd(Returns(Close, 1), 10)))
    long_frame = build_factor_frame(df, registry, long_format=True)
    wide_frame = build_factor_frame(df, registry, long_format=False)
    assert set(long_frame["factor_name"].unique()) == {"momentum_5", "volatility_10"}
    assert "factor_momentum_5" in wide_frame.columns
    assert "factor_volatility_10" in wide_frame.columns


def test_default_registry_seeded_with_examples():
    # The default registry is seeded with at least one Alpha101-style example.
    factors = default_registry().factors
    assert "momentum_5" in factors
    assert "reversal_1" in factors


def test_register_factor_writes_into_default_registry():
    new_factor = register_factor("debug_close", Close, description="raw close")
    df = _toy_panel(num_days=10)
    long_frame = build_factor_frame(df, default_registry(), long_format=True)
    assert new_factor.name == "debug_close"
    assert "debug_close" in set(long_frame["factor_name"].unique())


def test_polars_backend_matches_pandas_for_core_operators():
    pytest.importorskip("polars")
    df = _toy_panel(num_days=12, symbols=("A", "B", "C"))
    factors = {
        "rank": Rank(Close),
        "delay": Delay(Close, 1),
        "delta": Delta(Close, 1),
        "returns": Returns(Close, 1),
        "mean": TsMean(Close, 3),
        "std": TsStd(Close, 3),
        "sum": TsSum(Close, 3),
        "tsrank": TsRank(Close, 3),
        "min": TsMin(Close, 3),
        "max": TsMax(Close, 3),
    }
    pandas_result = build_factor_frame(df, factors, backend="pandas")
    polars_result = build_factor_frame(df, factors, backend="polars")
    pd.testing.assert_frame_equal(
        pandas_result.drop(columns=["trade_date"]),
        polars_result.drop(columns=["trade_date"]),
        check_dtype=False,
        atol=1e-10,
        rtol=1e-10,
    )
