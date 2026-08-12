from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.bootstrap.valuation_bootstrap import (
    _assert_requested_dates_are_sessions,
)
from quantagent.data.trading_calendar import TradingCalendar


def _calendar() -> TradingCalendar:
    return TradingCalendar.from_dates(
        ["2024-01-05", "2024-01-08", "2024-01-09"]
    )


def test_valuation_requested_dates_must_be_actual_sessions() -> None:
    with pytest.raises(ValueError, match="actual A-share trading sessions"):
        _assert_requested_dates_are_sessions(
            pd.DatetimeIndex([pd.Timestamp("2024-01-06")]),
            _calendar(),
        )


def test_valuation_requested_sessions_are_accepted_without_snapping() -> None:
    _assert_requested_dates_are_sessions(
        pd.DatetimeIndex(
            [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")]
        ),
        _calendar(),
    )


def test_valuation_missing_calendar_fails_closed() -> None:
    with pytest.raises(ValueError, match="requires an explicit A-share trading calendar"):
        _assert_requested_dates_are_sessions(
            pd.DatetimeIndex([pd.Timestamp("2024-01-05")]),
            TradingCalendar.from_dates(()),
        )
