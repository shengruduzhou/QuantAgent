from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_market_data_smoke import (
    validate_calendar_smoke,
    validate_daily_smoke,
    validate_minute_smoke,
)


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["600000.SH", "000001.SZ"],
            "trade_date": pd.to_datetime(["2026-08-06", "2026-08-06"]),
            "open": [10.0, 12.0],
            "high": [10.5, 12.5],
            "low": [9.8, 11.7],
            "close": [10.2, 12.2],
            "volume": [1_000_000.0, 2_000_000.0],
            "amount": [10_200_000.0, 24_400_000.0],
            "tradestatus": ["1", "1"],
            "isST": ["0", "0"],
            "adjustflag": ["3", "3"],
            "available_at": pd.to_datetime(["2026-08-07", "2026-08-07"]),
        }
    )


def _minute() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "trade_date": pd.to_datetime(["2026-08-06", "2026-08-06"]),
            "timestamp": pd.to_datetime(["2026-08-06 09:35", "2026-08-06 09:40"]),
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [10_000.0, 12_000.0],
            "amount": [101_000.0, 122_400.0],
            "adjustflag": ["3", "3"],
            "available_at": pd.to_datetime(["2026-08-06 09:35", "2026-08-06 09:40"]),
        }
    )


def test_daily_smoke_accepts_raw_schema_and_reports_non_research_fields() -> None:
    evidence = validate_daily_smoke(
        _daily(),
        requested_symbols=("600000.SH", "000001.SZ"),
    )
    assert evidence["adjustment"] == "raw"
    assert evidence["rows"] == 2
    assert evidence["suspended_rows"] == 0


def test_daily_smoke_rejects_adjusted_history() -> None:
    frame = _daily()
    frame["adjustflag"] = "1"
    with pytest.raises(RuntimeError, match="expected raw adjustflag=3"):
        validate_daily_smoke(
            frame,
            requested_symbols=("600000.SH", "000001.SZ"),
        )


def test_daily_smoke_rejects_missing_symbol_and_duplicates() -> None:
    with pytest.raises(RuntimeError, match="missing requested symbols"):
        validate_daily_smoke(_daily().iloc[[0]], requested_symbols=("600000.SH", "000001.SZ"))

    duplicate = pd.concat([_daily(), _daily().iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate symbol/trade_date"):
        validate_daily_smoke(duplicate, requested_symbols=("600000.SH", "000001.SZ"))


def test_daily_smoke_rejects_impossible_availability() -> None:
    frame = _daily()
    frame.loc[0, "available_at"] = pd.Timestamp("2026-08-05")
    with pytest.raises(RuntimeError, match="available_at before trade_date"):
        validate_daily_smoke(frame, requested_symbols=("600000.SH", "000001.SZ"))


def test_daily_smoke_rejects_malformed_high_and_low() -> None:
    bad_high = _daily()
    bad_high.loc[0, "high"] = 10.05  # below close=10.2
    with pytest.raises(RuntimeError, match="high invariant"):
        validate_daily_smoke(bad_high, requested_symbols=("600000.SH", "000001.SZ"))

    bad_low = _daily()
    bad_low.loc[0, "low"] = 10.1  # above open=10.0
    with pytest.raises(RuntimeError, match="low invariant"):
        validate_daily_smoke(bad_low, requested_symbols=("600000.SH", "000001.SZ"))


def test_minute_smoke_requires_raw_unique_timestamped_bars() -> None:
    evidence = validate_minute_smoke(_minute(), frequency="5")
    assert evidence["frequency_minutes"] == 5
    assert evidence["rows"] == 2

    adjusted = _minute()
    adjusted["adjustflag"] = "2"
    with pytest.raises(RuntimeError, match="expected raw adjustflag=3"):
        validate_minute_smoke(adjusted, frequency="5")

    duplicate = pd.concat([_minute(), _minute().iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate symbol/timestamp"):
        validate_minute_smoke(duplicate, frequency="5")


def test_minute_smoke_rejects_non_numeric_and_bad_ohlc() -> None:
    non_numeric = _minute()
    non_numeric.loc[0, "amount"] = "bad"
    with pytest.raises(RuntimeError, match="non-numeric OHLCVA"):
        validate_minute_smoke(non_numeric, frequency="5")

    bad_high = _minute()
    bad_high.loc[0, "high"] = 10.05
    with pytest.raises(RuntimeError, match="high invariant"):
        validate_minute_smoke(bad_high, frequency="5")


def test_calendar_smoke_requires_real_trading_sessions() -> None:
    calendar = pd.DataFrame(
        {
            "calendar_date": pd.to_datetime(["2026-08-08", "2026-08-09", "2026-08-10"]),
            "is_trading_day": ["0", "0", "1"],
        }
    )
    evidence = validate_calendar_smoke(calendar)
    assert evidence["trading_sessions"] == 1
    assert evidence["independent_of_symbol_bars"] is True

    no_sessions = calendar.copy()
    no_sessions["is_trading_day"] = "0"
    with pytest.raises(RuntimeError, match="no trading sessions"):
        validate_calendar_smoke(no_sessions)
