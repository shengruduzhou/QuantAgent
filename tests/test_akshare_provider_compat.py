from __future__ import annotations

import pandas as pd
import pytest

import quantagent.data.providers.akshare_provider as compat
from quantagent.data.providers.akshare_calendar import AkShareCalendarEvidence
from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable
from quantagent.data.trading_calendar import TradingCalendar


def _request() -> ProviderRequest:
    return ProviderRequest(
        "2024-01-02",
        "2024-01-03",
        symbols=("600000.SH",),
    )


def test_compat_provider_never_returns_mock_when_network_disabled() -> None:
    with pytest.raises(ProviderUnavailable, match="refuses mock fallback"):
        compat.AkShareProvider(allow_network=False).daily_ohlcv(_request())


def test_compat_provider_fails_closed_without_research_calendar(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "load_akshare_research_calendar",
        lambda **_: AkShareCalendarEvidence(
            TradingCalendar.from_dates(()),
            {"source": "test", "status": "empty", "production_certified": False},
        ),
    )

    with pytest.raises(ProviderUnavailable, match="calendar is unavailable"):
        compat.AkShareProvider(allow_network=True).daily_ohlcv(_request())


def test_compat_provider_delegates_to_governed_live_provider(monkeypatch) -> None:
    calendar = TradingCalendar.from_dates(["2024-01-02", "2024-01-03", "2024-01-04"])
    monkeypatch.setattr(
        compat,
        "load_akshare_research_calendar",
        lambda **_: AkShareCalendarEvidence(
            calendar,
            {
                "source": "test_calendar",
                "status": "passed",
                "production_certified": False,
            },
        ),
    )

    captured: dict[str, object] = {}

    class FakeLiveProvider:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
            captured["request"] = request
            frame = pd.DataFrame(
                [
                    {
                        "symbol": "600000.SH",
                        "trade_date": "2024-01-02",
                        "close": 10.0,
                    }
                ]
            )
            return ProviderResult(
                frame=frame,
                source="akshare_live_provider:multi_source",
                point_in_time=True,
                quality_score=0.78,
                warnings=("upstream_warning",),
                metadata={"mock_fallback": False, "akshare_version": "test"},
            )

    monkeypatch.setattr(compat, "AkShareLiveProvider", FakeLiveProvider)

    result = compat.AkShareProvider(allow_network=True, adjust="").daily_ohlcv(_request())

    assert captured["allow_network"] is True
    assert captured["adjust"] == ""
    assert captured["trading_calendar"] is calendar
    assert captured["calendar_source"] == "test_calendar"
    assert captured["request"] == _request()
    assert result.source == "akshare_live_provider:multi_source"
    assert result.point_in_time is True
    assert result.metadata["mock_fallback"] is False
    assert result.metadata["compatibility_provider"] == "akshare_provider_compat"
    assert result.metadata["calendar"]["source"] == "test_calendar"
    assert "akshare_compat_provider_delegated_to_live_provider" in result.warnings
