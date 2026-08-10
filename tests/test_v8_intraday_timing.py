from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import typer

from quantagent.cli.v8_intraday import (
    _next_market_session_available_at,
    _read_market_sessions,
)


def test_intraday_availability_uses_explicit_exchange_sessions_not_bday() -> None:
    sessions = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-01", "2024-03-04", "2024-03-05"])
    )
    trade_date = pd.Series(pd.to_datetime(["2024-03-01", "2024-03-04"]))

    resolved = _next_market_session_available_at(trade_date, sessions)

    assert resolved.tolist() == [
        pd.Timestamp("2024-03-04"),
        pd.Timestamp("2024-03-05"),
    ]


def test_intraday_availability_fails_when_calendar_does_not_extend() -> None:
    sessions = pd.DatetimeIndex(pd.to_datetime(["2024-03-01", "2024-03-04"]))
    trade_date = pd.Series(pd.to_datetime(["2024-03-04"]))

    with pytest.raises(typer.BadParameter, match="extend the explicit calendar"):
        _next_market_session_available_at(trade_date, sessions)


def test_read_market_sessions_accepts_independent_calendar_csv(tmp_path: Path) -> None:
    path = tmp_path / "calendar.csv"
    pd.DataFrame(
        {
            "calendar_date": ["2024-03-01", "2024-03-04", "2024-03-05"],
        }
    ).to_csv(path, index=False)

    sessions = _read_market_sessions(path)

    assert sessions.strftime("%Y-%m-%d").tolist() == [
        "2024-03-01",
        "2024-03-04",
        "2024-03-05",
    ]


def test_read_market_sessions_filters_non_trading_calendar_rows(tmp_path: Path) -> None:
    path = tmp_path / "calendar.csv"
    pd.DataFrame(
        {
            "calendar_date": [
                "2024-03-01",
                "2024-03-02",
                "2024-03-03",
                "2024-03-04",
            ],
            "is_trading_day": ["1", "0", "0", "1"],
        }
    ).to_csv(path, index=False)

    sessions = _read_market_sessions(path)

    assert sessions.strftime("%Y-%m-%d").tolist() == [
        "2024-03-01",
        "2024-03-04",
    ]


def test_read_market_sessions_rejects_unknown_trading_flags(tmp_path: Path) -> None:
    path = tmp_path / "calendar.csv"
    pd.DataFrame(
        {
            "calendar_date": ["2024-03-01", "2024-03-02"],
            "is_trading_day": ["1", "maybe"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(typer.BadParameter, match="unknown is_trading_day values"):
        _read_market_sessions(path)
