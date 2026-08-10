from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.intraday_sessions import (
    ExecutionPriceProvenanceError,
    IntradaySessionError,
    aggregate_ashare_bars,
    assert_raw_execution_prices,
)


def _five_minute_day(*, adjustflag: str = "3") -> pd.DataFrame:
    timestamps = list(
        pd.date_range("2024-03-01 09:35", "2024-03-01 11:30", freq="5min")
    ) + list(
        pd.date_range("2024-03-01 13:05", "2024-03-01 15:00", freq="5min")
    )
    rows = []
    for index, stamp in enumerate(timestamps):
        close = 100.0 + index / 100.0
        rows.append(
            {
                "symbol": "600519.SH",
                "timestamp": stamp,
                "open": close - 0.05,
                "high": close + 0.10,
                "low": close - 0.10,
                "close": close,
                "volume": 100 + index,
                "amount": (100 + index) * close,
                "adjustflag": adjustflag,
            }
        )
    return pd.DataFrame(rows)


def test_ten_minute_bars_are_session_aligned_and_complete() -> None:
    out = aggregate_ashare_bars(_five_minute_day(), minutes=10)

    assert len(out) == 24
    assert out.groupby("session").size().to_dict() == {"afternoon": 12, "morning": 12}
    assert set(out["timestamp"].dt.strftime("%H:%M")).isdisjoint(
        {"11:40", "11:50", "12:00", "12:10", "12:20", "12:30", "12:40", "12:50", "13:00"}
    )
    assert (out["price_adjustment"] == "raw").all()
    assert out["execution_eligible"].all()
    assert (out["timezone"] == "Asia/Shanghai").all()


def test_sixty_minute_bars_never_cross_lunch_break() -> None:
    out = aggregate_ashare_bars(_five_minute_day(), minutes=60)

    assert len(out) == 4
    assert out["timestamp"].dt.strftime("%H:%M").tolist() == [
        "10:30",
        "11:30",
        "14:00",
        "15:00",
    ]
    assert out["session"].tolist() == [
        "morning",
        "morning",
        "afternoon",
        "afternoon",
    ]
    assert out["bar_start"].dt.strftime("%H:%M").tolist() == [
        "09:30",
        "10:30",
        "13:00",
        "14:00",
    ]


def test_partial_window_is_not_emitted_by_default() -> None:
    timestamps = pd.date_range("2024-03-01 09:31", "2024-03-01 10:02", freq="1min")
    frame = pd.DataFrame(
        {
            "symbol": ["600519.SH"] * len(timestamps),
            "timestamp": timestamps,
            "open": 100.0,
            "high": 100.2,
            "low": 99.8,
            "close": 100.1,
            "volume": 100,
            "adjustflag": "3",
        }
    )

    closed_only = aggregate_ashare_bars(frame, minutes=10, emit_partial=False)
    with_partial = aggregate_ashare_bars(frame, minutes=10, emit_partial=True)

    assert closed_only["timestamp"].dt.strftime("%H:%M").tolist() == [
        "09:40",
        "09:50",
        "10:00",
    ]
    assert with_partial["timestamp"].dt.strftime("%H:%M").tolist() == [
        "09:40",
        "09:50",
        "10:00",
        "10:10",
    ]


def test_adjusted_bars_without_raw_amount_fail_closed() -> None:
    frame = _five_minute_day(adjustflag="1").drop(columns=["amount"])

    with pytest.raises(IntradaySessionError, match="provider-supplied raw amount"):
        aggregate_ashare_bars(frame, minutes=10)


def test_mixed_naive_and_aware_timestamps_are_rejected() -> None:
    frame = _five_minute_day().iloc[:2].copy()
    frame["timestamp"] = [
        pd.Timestamp("2024-03-01 09:35"),
        pd.Timestamp("2024-03-01 09:40", tz="Asia/Shanghai"),
    ]

    with pytest.raises(IntradaySessionError, match="mixed naive"):
        aggregate_ashare_bars(frame, minutes=10)


@pytest.mark.parametrize("adjustflag", ["1", "2"])
def test_adjusted_prices_are_never_execution_eligible(adjustflag: str) -> None:
    frame = _five_minute_day(adjustflag=adjustflag)

    with pytest.raises(ExecutionPriceProvenanceError, match="raw/unadjusted"):
        assert_raw_execution_prices(frame)


def test_missing_price_provenance_fails_closed_for_execution() -> None:
    frame = _five_minute_day().drop(columns=["adjustflag"])

    with pytest.raises(ExecutionPriceProvenanceError, match="observed price_adjustment"):
        assert_raw_execution_prices(frame)


def test_raw_prices_pass_execution_guard() -> None:
    assert_raw_execution_prices(_five_minute_day(adjustflag="3"))
