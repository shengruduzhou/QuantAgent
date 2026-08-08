from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.integrity import DailyOHLCVIntegrityPolicy
from quantagent.data.providers.base import ProviderRequest, ProviderResult
from quantagent.data.router import (
    MultiSourceDataRouter,
    RoutedProvider,
    RouterConfig,
    RouterDataIntegrityError,
)


def _meta(**overrides):
    metadata = {
        "frequency": "1d",
        "timezone": "Asia/Shanghai",
        "volume_unit": "shares",
        "amount_unit": "CNY",
        "adjustment": "none",
    }
    metadata.update(overrides)
    return metadata


def _row(date: str, *, close: float, open_: float | None = None) -> dict[str, object]:
    open_value = close if open_ is None else open_
    return {
        "symbol": "600000.SH",
        "trade_date": date,
        "open": open_value,
        "high": max(open_value, close) + 0.2,
        "low": min(open_value, close) - 0.2,
        "close": close,
        "volume": 1000.0,
        "amount": 1000.0 * close,
    }


class StaticDailyProvider:
    def __init__(self, frame: pd.DataFrame, *, source: str, metadata: dict | None = None, pit: bool = True):
        self.frame = frame
        self.source = source
        self.metadata = _meta() if metadata is None else metadata
        self.pit = pit
        self.calls = 0

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        self.calls += 1
        return ProviderResult(
            self.frame.copy(),
            source=self.source,
            point_in_time=self.pit,
            metadata=dict(self.metadata),
        )


def _router(primary: StaticDailyProvider, fallback: StaticDailyProvider) -> MultiSourceDataRouter:
    router = MultiSourceDataRouter(
        RouterConfig(daily_priority=("primary", "fallback"), merge_partial_results=True)
    )
    router.register(RoutedProvider("primary", primary))
    router.register(RoutedProvider("fallback", fallback))
    return router


def _request() -> ProviderRequest:
    return ProviderRequest(
        start_date="2026-01-05",
        end_date="2026-01-06",
        symbols=("600000.SH",),
    )


def _policy(**kwargs) -> DailyOHLCVIntegrityPolicy:
    return DailyOHLCVIntegrityPolicy.production(
        expected_trade_dates=("2026-01-05", "2026-01-06"),
        **kwargs,
    )


def test_production_fallback_replaces_only_invalid_missing_key() -> None:
    primary = StaticDailyProvider(
        pd.DataFrame(
            [
                _row("2026-01-05", close=10.0),
                _row("2026-01-06", close=-1.0, open_=-1.0),
            ]
        ),
        source="primary-real",
    )
    fallback = StaticDailyProvider(
        pd.DataFrame(
            [
                _row("2026-01-05", close=99.0),
                _row("2026-01-06", close=11.0),
            ]
        ),
        source="fallback-real",
    )
    result = _router(primary, fallback).daily_ohlcv(_request(), integrity_policy=_policy())

    assert len(result.frame) == 2
    assert not result.frame.duplicated(["symbol", "trade_date"]).any()
    by_date = result.frame.assign(trade_date=pd.to_datetime(result.frame["trade_date"]).dt.date.astype(str)).set_index("trade_date")
    assert by_date.loc["2026-01-05", "close"] == 10.0
    assert by_date.loc["2026-01-05", "source_name"] == "primary"
    assert by_date.loc["2026-01-06", "close"] == 11.0
    assert by_date.loc["2026-01-06", "source_name"] == "fallback"
    assert result.primary_source == "primary"
    assert result.integrity["missing_expected_keys"] == []
    assert result.per_source["primary"]["integrity"]["status"] == "failed"
    assert fallback.calls == 1


def test_invalid_primary_cannot_fallback_without_explicit_policy_permission() -> None:
    primary = StaticDailyProvider(
        pd.DataFrame([_row("2026-01-05", close=-1.0, open_=-1.0)]),
        source="primary-real",
    )
    fallback = StaticDailyProvider(
        pd.DataFrame([_row("2026-01-05", close=10.0), _row("2026-01-06", close=11.0)]),
        source="fallback-real",
    )
    with pytest.raises(RouterDataIntegrityError, match="daily integrity failed at primary"):
        _router(primary, fallback).daily_ohlcv(
            _request(),
            integrity_policy=_policy(allow_invalid_primary_fallback=False),
        )
    assert fallback.calls == 0


def test_mock_primary_is_never_served_in_production_and_real_fallback_wins() -> None:
    mock = StaticDailyProvider(
        pd.DataFrame([_row("2026-01-05", close=10.0), _row("2026-01-06", close=11.0)]),
        source="mock_provider",
        metadata=_meta(mock=True, fallback=True),
    )
    real = StaticDailyProvider(
        pd.DataFrame([_row("2026-01-05", close=20.0), _row("2026-01-06", close=21.0)]),
        source="real-provider",
    )
    result = _router(mock, real).daily_ohlcv(_request(), integrity_policy=_policy())
    assert set(result.frame["source_name"]) == {"fallback"}
    assert result.primary_source == "fallback"
    assert "mock_or_synthetic_source" in result.per_source["primary"]["integrity"]["hard_violations"]


def test_source_semantic_failure_is_not_partially_served() -> None:
    wrong_adjustment = StaticDailyProvider(
        pd.DataFrame([_row("2026-01-05", close=10.0), _row("2026-01-06", close=11.0)]),
        source="qfq-provider",
        metadata=_meta(adjustment="forward"),
    )
    real = StaticDailyProvider(
        pd.DataFrame([_row("2026-01-05", close=20.0), _row("2026-01-06", close=21.0)]),
        source="raw-provider",
    )
    result = _router(wrong_adjustment, real).daily_ohlcv(_request(), integrity_policy=_policy())
    assert set(result.frame["source_name"]) == {"fallback"}
    assert "adjustment_mismatch:forward" in result.per_source["primary"]["integrity"]["hard_violations"]


def test_production_without_authoritative_calendar_reports_observed_only_not_fake_gaps() -> None:
    primary = StaticDailyProvider(
        pd.DataFrame([_row("2026-01-05", close=10.0)]),
        source="primary-real",
    )
    fallback = StaticDailyProvider(
        pd.DataFrame([_row("2026-01-06", close=11.0)]),
        source="fallback-real",
    )
    policy = DailyOHLCVIntegrityPolicy.production(expected_trade_dates=())
    result = _router(primary, fallback).daily_ohlcv(_request(), integrity_policy=policy)
    assert len(result.frame) == 2
    assert result.integrity["coverage_basis"] == "observed_only"
    assert "daily_integrity_expected_calendar_not_supplied_observed_only" in result.warnings
    assert primary.calls == 1 and fallback.calls == 1


def test_authoritative_expected_calendar_fails_if_final_key_is_missing() -> None:
    primary = StaticDailyProvider(
        pd.DataFrame([_row("2026-01-05", close=10.0)]),
        source="primary-real",
    )
    fallback = StaticDailyProvider(pd.DataFrame(), source="fallback-real")
    with pytest.raises(RouterDataIntegrityError, match="coverage incomplete"):
        _router(primary, fallback).daily_ohlcv(_request(), integrity_policy=_policy())


def test_research_field_slice_keeps_historical_first-source_semantics() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": ["2026-01-05"],
            "close": [10.0],
        }
    )
    primary = StaticDailyProvider(frame, source="research-primary", metadata={})
    fallback = StaticDailyProvider(frame.assign(close=99.0), source="research-fallback", metadata={})
    router = _router(primary, fallback)
    request = ProviderRequest(
        "2026-01-05",
        "2026-01-05",
        symbols=("600000.SH",),
        fields=("close",),
    )
    result = router.daily_ohlcv(request)
    assert result.frame.iloc[0]["close"] == 10.0
    assert result.primary_source == "primary"
    assert fallback.calls == 0


def test_production_requires_declared_units_frequency_timezone_and_adjustment() -> None:
    primary = StaticDailyProvider(
        pd.DataFrame([_row("2026-01-05", close=10.0), _row("2026-01-06", close=11.0)]),
        source="undeclared-provider",
        metadata={},
    )
    fallback = StaticDailyProvider(pd.DataFrame(), source="empty-fallback")
    with pytest.raises(RouterDataIntegrityError, match="no daily rows satisfied"):
        _router(primary, fallback).daily_ohlcv(_request(), integrity_policy=_policy())
    integrity = validate_result = primary.daily_ohlcv(_request())
    # The provider response itself is untouched; the router does not invent
    # unit/frequency metadata to make it production-eligible.
    assert validate_result.metadata == {}
