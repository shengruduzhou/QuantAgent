from __future__ import annotations

import pandas as pd

from quantagent.data.integrity import DailyOHLCVIntegrityPolicy, validate_daily_ohlcv
from quantagent.data.providers.base import ProviderRequest, ProviderResult


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "trade_date": ["2026-01-05", "2026-01-06"],
            "open": [10.0, 10.5],
            "high": [10.8, 11.0],
            "low": [9.8, 10.2],
            "close": [10.4, 10.8],
            "volume": [1000.0, 1200.0],
            "amount": [10_400.0, 12_960.0],
        }
    )


def _metadata() -> dict[str, object]:
    return {
        "frequency": "1d",
        "timezone": "Asia/Shanghai",
        "volume_unit": "shares",
        "amount_unit": "CNY",
        "adjustment": "raw",
        "pit_semantics": "trade_date_observed_no_future_adjustment",
    }


def test_live_critical_daily_data_is_stale_against_explicit_latest_trade_date() -> None:
    request = ProviderRequest(
        start_date="2026-01-05",
        end_date="2026-01-08",
        symbols=("600000.SH",),
    )
    policy = DailyOHLCVIntegrityPolicy.production(
        expected_trade_dates=("2026-01-05", "2026-01-06"),
        live_critical=True,
        expected_latest_trade_date="2026-01-08",
        max_staleness_calendar_days=0,
    )
    validated = validate_daily_ohlcv(
        ProviderResult(
            _bars(),
            source="real",
            point_in_time=True,
            metadata=_metadata(),
        ),
        request,
        policy,
    )
    assert "stale_daily_data:2d" in validated.report.hard_violations
    assert validated.report.status == "failed"


def test_freshness_reference_outside_request_window_is_not_valid_evidence() -> None:
    request = ProviderRequest(
        start_date="2026-01-05",
        end_date="2026-01-06",
        symbols=("600000.SH",),
    )
    policy = DailyOHLCVIntegrityPolicy.production(
        expected_trade_dates=("2026-01-05", "2026-01-06"),
        live_critical=True,
        expected_latest_trade_date="2026-01-08",
        max_staleness_calendar_days=10,
    )
    validated = validate_daily_ohlcv(
        ProviderResult(
            _bars(),
            source="real",
            point_in_time=True,
            metadata=_metadata(),
        ),
        request,
        policy,
    )
    assert "freshness_reference_outside_request_window" in validated.report.hard_violations
    assert validated.report.status == "failed"
