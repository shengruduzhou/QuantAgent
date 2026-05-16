"""Trading-calendar based available_at resolver."""
from __future__ import annotations

import pandas as pd

from quantagent.data.trading_calendar import TradingCalendar, calendar_day_resolver


def test_resolver_snaps_forward_to_next_trading_day():
    calendar = TradingCalendar.from_dates(
        ["2026-05-08", "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15"]
    )
    # 2026-05-09 = Saturday; +1 calendar day -> 2026-05-10 (Sunday) -> snap to 2026-05-11.
    assert calendar.next_trading_day("2026-05-09", lag_days=0) == pd.Timestamp("2026-05-11")
    assert calendar.next_trading_day("2026-05-09", lag_days=1) == pd.Timestamp("2026-05-11")
    # weekday with lag 1 hops one calendar day to the next trading day.
    assert calendar.next_trading_day("2026-05-12", lag_days=1) == pd.Timestamp("2026-05-13")
    # exact trading day with no lag stays put.
    assert calendar.next_trading_day("2026-05-12", lag_days=0) == pd.Timestamp("2026-05-12")
    # beyond known trading days returns NaT to flag downstream consumers.
    assert pd.isna(calendar.next_trading_day("2026-05-20", lag_days=0))


def test_resolver_handles_series_input():
    calendar = TradingCalendar.from_dates(
        ["2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15"]
    )
    series = pd.Series(["2026-05-09", "2026-05-12", "2026-05-15", None], dtype=object)
    resolved = calendar.resolve_available_at(series, lag_days=1)
    assert resolved.iloc[0] == pd.Timestamp("2026-05-11")
    assert resolved.iloc[1] == pd.Timestamp("2026-05-13")
    assert pd.isna(resolved.iloc[2])
    assert pd.isna(resolved.iloc[3])


def test_calendar_day_resolver_falls_back_when_no_calendar():
    series = pd.Series(["2026-05-09", "2026-05-12"], dtype=object)
    resolved = calendar_day_resolver(series, lag_days=1)
    assert resolved.iloc[0] == pd.Timestamp("2026-05-10")
    assert resolved.iloc[1] == pd.Timestamp("2026-05-13")


def test_calendar_from_market_panel():
    market = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-05-11", "2026-05-12", "2026-05-13"]),
        }
    )
    calendar = TradingCalendar.from_market_panel(market)
    assert calendar.contains("2026-05-12")
    assert not calendar.contains("2026-05-09")
