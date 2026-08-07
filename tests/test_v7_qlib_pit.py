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


def test_close_derived_features_are_available_at_their_own_close():
    """``build_market_features`` stamps ``available_at == trade_date``.

    The previous version of this test asserted the shift to the *next* trading row,
    on the reasoning that "close-derived technicals must not be visible on the same
    trade_date they were computed from". That reasoning was applied to one side of
    the pair only: `v7_label_builder` scores the very same row on
    ``close(t+h)/close(t) - 1``, a window that opens at ``close(t)``. So the row
    was declared unusable until ``t+1`` while being credited with a return starting
    at ``t`` — a look-ahead in 100% of rows (DEF-026), and a violation of the
    ``available_at <= trade_date`` invariant the gold builder asserts.

    `available_at` is not a comment: `merge_pit_features` uses it as the as-of join
    key. The extra session it granted let a feature published on ``t+1`` join onto
    the row scored on the ``t -> t+1`` return, measured at rank IC +1.0000.

    Latency between knowing a signal and filling an order is real, but it belongs
    to the execution layer, which models it explicitly. Encoding it in the
    availability stamp bought no conservatism and cost a leak.
    """
    frame = _synthetic_panel()
    features = build_market_features(frame)
    available = pd.to_datetime(features["available_at"])
    trade_date = pd.to_datetime(features["trade_date"])
    assert (available == trade_date).all()


def test_validate_qlib_market_schema_flags_future_available_at():
    frame = _synthetic_panel()
    frame.loc[0, "available_at"] = pd.Timestamp("2030-01-01")
    report = validate_qlib_market_schema(frame, as_of_date="2025-12-31")
    assert report["pit_violation_count"] >= 1
    assert report["status"] == "failed"
