from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.data.integrity import DailyOHLCVIntegrityPolicy, validate_daily_ohlcv
from quantagent.data.providers.base import ProviderRequest, ProviderResult


def _request(*, fields: tuple[str, ...] = ()) -> ProviderRequest:
    return ProviderRequest(
        start_date="2026-01-05",
        end_date="2026-01-06",
        symbols=("600000.SH",),
        fields=fields,
    )


def _metadata(**overrides):
    base = {
        "frequency": "1d",
        "timezone": "Asia/Shanghai",
        "volume_unit": "shares",
        "amount_unit": "CNY",
        "adjustment": "none",
        "pit_semantics": "trade_date_observed_no_future_adjustment",
    }
    base.update(overrides)
    return base


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


def _production(**kwargs) -> DailyOHLCVIntegrityPolicy:
    return DailyOHLCVIntegrityPolicy.production(
        expected_trade_dates=("2026-01-05", "2026-01-06"),
        **kwargs,
    )


def test_valid_production_daily_bars_pass() -> None:
    validated = validate_daily_ohlcv(
        ProviderResult(_bars(), source="real", point_in_time=True, metadata=_metadata()),
        _request(),
        _production(),
    )
    assert validated.report.status == "pass"
    assert validated.report.hard_violations == ()
    assert len(validated.valid_frame) == 2
    assert validated.quarantine_frame.empty
    assert validated.report.observed_expected_key_count == 2


def test_invalid_prices_volume_and_ohlc_are_quarantined_never_repaired() -> None:
    frame = _bars()
    frame.loc[0, "open"] = np.nan
    frame.loc[1, "high"] = 9.0
    frame.loc[1, "volume"] = -1.0
    original = frame.copy(deep=True)
    validated = validate_daily_ohlcv(
        ProviderResult(frame, source="real", point_in_time=True, metadata=_metadata()),
        _request(),
        _production(),
    )
    assert validated.valid_frame.empty
    assert len(validated.quarantine_frame) == 2
    assert "invalid_daily_rows_present" in validated.report.hard_violations
    pd.testing.assert_frame_equal(frame, original)


def test_duplicate_identity_is_quarantined() -> None:
    frame = pd.concat([_bars().iloc[[0]], _bars().iloc[[0]]], ignore_index=True)
    validated = validate_daily_ohlcv(
        ProviderResult(frame, source="real", point_in_time=True, metadata=_metadata()),
        _request(),
        DailyOHLCVIntegrityPolicy.research(),
    )
    assert validated.valid_frame.empty
    assert validated.report.duplicate_keys == 2
    assert "duplicate_symbol_trade_date_rows_quarantined" in validated.report.warnings


def test_production_requires_real_pit_declared_semantics() -> None:
    result = ProviderResult(
        _bars(),
        source="mock_provider",
        point_in_time=False,
        metadata={"mock": True},
    )
    validated = validate_daily_ohlcv(result, _request(), _production())
    violations = set(validated.report.hard_violations)
    assert "provider_not_point_in_time" in violations
    assert "pit_semantics_missing" in violations
    assert "mock_or_synthetic_source" in violations
    assert "frequency_metadata_missing" in violations
    assert "timezone_metadata_missing" in violations
    assert "volume_unit_missing" in violations
    assert "amount_unit_missing" in violations
    assert "adjustment_metadata_missing" in violations


def test_default_point_in_time_true_is_not_sufficient_production_evidence() -> None:
    metadata = _metadata()
    metadata.pop("pit_semantics")
    validated = validate_daily_ohlcv(
        ProviderResult(_bars(), source="real", metadata=metadata),
        _request(),
        _production(),
    )
    assert "provider_not_point_in_time" not in validated.report.hard_violations
    assert "pit_semantics_missing" in validated.report.hard_violations
    assert validated.report.status == "failed"


def test_adjustment_mismatch_fails_instead_of_mixing_price_semantics() -> None:
    validated = validate_daily_ohlcv(
        ProviderResult(
            _bars(),
            source="real",
            point_in_time=True,
            metadata=_metadata(adjustment="forward"),
        ),
        _request(),
        _production(),
    )
    assert "adjustment_mismatch:forward" in validated.report.hard_violations


def test_research_field_slice_remains_supported() -> None:
    frame = _bars()[["symbol", "trade_date", "close"]]
    validated = validate_daily_ohlcv(
        ProviderResult(frame, source="research", metadata={}),
        _request(fields=("close",)),
        DailyOHLCVIntegrityPolicy.research(),
    )
    assert len(validated.valid_frame) == 2
    assert validated.report.status == "degraded"
    assert not validated.report.hard_violations


def test_requested_field_missing_quarantines_the_response() -> None:
    frame = _bars()[["symbol", "trade_date", "close"]]
    validated = validate_daily_ohlcv(
        ProviderResult(frame, source="research", metadata={}),
        _request(fields=("open",)),
        DailyOHLCVIntegrityPolicy.research(),
    )
    assert "missing_column:open" in validated.report.hard_violations
    assert validated.valid_frame.empty
    assert len(validated.quarantine_frame) == 2
    assert validated.report.valid_rows == 0


def test_live_freshness_requires_explicit_reference_and_rejects_stale() -> None:
    missing_reference = validate_daily_ohlcv(
        ProviderResult(_bars(), source="real", metadata=_metadata()),
        _request(),
        _production(live_critical=True),
    )
    assert "freshness_reference_missing" in missing_reference.report.hard_violations

    stale = validate_daily_ohlcv(
        ProviderResult(_bars(), source="real", metadata=_metadata()),
        _request(),
        _production(
            live_critical=True,
            expected_latest_trade_date="2026-01-08",
            max_staleness_calendar_days=0,
        ),
    )
    assert "stale_daily_data:2d" in stale.report.hard_violations


def test_intraday_timestamp_cannot_masquerade_as_daily_trade_date() -> None:
    frame = _bars().iloc[[0]].copy()
    frame.loc[:, "trade_date"] = "2026-01-05 10:30:00"
    validated = validate_daily_ohlcv(
        ProviderResult(frame, source="real", metadata=_metadata()),
        ProviderRequest("2026-01-05", "2026-01-05", symbols=("600000.SH",)),
        DailyOHLCVIntegrityPolicy.research(),
    )
    assert validated.valid_frame.empty
    assert "non_daily_trade_date_rows_quarantined" in validated.report.warnings
