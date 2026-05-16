"""Tests pinning the V7 Qlib PIT contract.

These tests use a synthetic fixture rather than calling pyqlib so they
work offline. They exercise the parts of the data layer that own the
``available_at`` convention.
"""

from __future__ import annotations

import pandas as pd

from quantagent.data.providers.qlib_provider import (
    QLIB_MARKET_COLUMNS,
    QLIB_MARKET_OPTIONAL_COLUMNS,
    validate_qlib_market_schema,
)
from quantagent.data.v7_dataset_builder import build_market_features


def _synthetic_panel() -> pd.DataFrame:
    rows = []
    for symbol in ("A", "B"):
        for i, day in enumerate(pd.bdate_range("2025-01-02", periods=10)):
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": day,
                    "open": 10.0 + i,
                    "high": 11.0 + i,
                    "low": 9.0 + i,
                    "close": 10.5 + i,
                    "volume": 1000.0,
                    "amount": 10000.0,
                    "available_at": day,
                }
            )
    return pd.DataFrame(rows)


def test_qlib_schema_lists_required_and_optional_columns():
    frame = _synthetic_panel()
    report = validate_qlib_market_schema(frame, as_of_date="2025-01-31")
    assert report["status"] == "passed"
    assert set(QLIB_MARKET_COLUMNS).issubset(set(frame.columns))
    assert set(report["optional_columns_missing"]) == set(QLIB_MARKET_OPTIONAL_COLUMNS)


def test_close_derived_features_available_next_trading_day():
    """``build_market_features`` shifts ``available_at`` to the next trading row.

    This contract is critical: close-derived technicals must not be
    visible on the same trade_date they were computed from.
    """
    frame = _synthetic_panel()
    features = build_market_features(frame)
    grouped = features.sort_values(["symbol", "trade_date"]).groupby("symbol", sort=False)
    for _, sub in grouped:
        # ``available_at`` for row i should equal trade_date for row i+1 (until the last row).
        rows = sub.reset_index(drop=True)
        for i in range(len(rows) - 1):
            current_available = pd.Timestamp(rows.loc[i, "available_at"])
            next_trade = pd.Timestamp(rows.loc[i + 1, "trade_date"])
            assert current_available == next_trade, (
                "available_at must equal next trading row's trade_date"
            )


def test_validate_qlib_market_schema_flags_future_available_at():
    frame = _synthetic_panel()
    frame.loc[0, "available_at"] = pd.Timestamp("2030-01-01")
    report = validate_qlib_market_schema(frame, as_of_date="2025-12-31")
    assert report["pit_violation_count"] >= 1
    assert report["status"] == "failed"
