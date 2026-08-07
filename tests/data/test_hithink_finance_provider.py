from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.providers.hithink_finance_provider import (
    HithinkFinanceApiError,
    HithinkFinanceProvider,
)


class _Response:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict, dict]] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, *, params=None, headers=None, timeout=None, **kwargs):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError("unexpected request")
        return self.responses.pop(0)

    def close(self) -> None:
        return None


def test_daily_normalises_and_delays_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-key-not-secret")
    provider = HithinkFinanceProvider(allow_network=True)
    provider._session = _Session([
        _Response({
            "code": 0,
            "message": "success",
            "request_id": "req-1",
            "data": {
                "timestamp": 1767225600000,
                "item": [{
                    "date_ms": 1767196800000,
                    "open_price": 10.0,
                    "high_price": 11.0,
                    "low_price": 9.5,
                    "close_price": 10.5,
                    "volume": 1234,
                    "turnover": 54321.0,
                }],
            },
        })
    ])

    result = provider.daily_ohlcv(ProviderRequest("2026-01-01", "2026-01-01", ("600519.SH",)))

    assert list(result.frame["symbol"]) == ["600519.SH"]
    row = result.frame.iloc[0]
    assert row["close"] == pytest.approx(10.5)
    assert row["amount"] == pytest.approx(54321.0)
    assert pd.Timestamp(row["available_at"]) == pd.Timestamp(row["trade_date"]) + pd.Timedelta(days=1)
    assert bool(row["point_in_time_valid"])
    assert result.metadata["adjust"] == "none"
    _, params, headers = provider._session.calls[0]
    assert params["adjust"] == "none"
    assert params["interval"] == "1d"
    assert headers["X-api-key"] == "test-key-not-secret"


def test_adjusted_prices_requests_forward_adjust(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-key-not-secret")
    provider = HithinkFinanceProvider(allow_network=True)
    provider._session = _Session([
        _Response({
            "code": 0,
            "data": {"item": [{
                "date_ms": 1767196800000,
                "open_price": 1.0,
                "high_price": 1.0,
                "low_price": 1.0,
                "close_price": 1.0,
                "volume": 1,
                "turnover": 1.0,
            }]},
        })
    ])
    provider.adjusted_prices(ProviderRequest("2026-01-01", "2026-01-01", ("000001.SZ",)))
    assert provider._session.calls[0][1]["adjust"] == "forward"


def test_business_error_is_not_silently_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-key-not-secret")
    provider = HithinkFinanceProvider(allow_network=True, max_retries=1)
    provider._session = _Session([
        _Response({"code": 2003, "message": "no capability", "request_id": "req-entitlement", "data": None})
    ])
    with pytest.raises(HithinkFinanceApiError) as exc:
        provider.capability("prices_snapshot", thscodes="600519.SH")
    assert exc.value.code == 2003
    assert "test-key-not-secret" not in str(exc.value)


def test_missing_key_fails_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HITHINK_FINANCE_API_KEY", "FUYAO_API_KEY", "FUYAO_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    provider = HithinkFinanceProvider(allow_network=True)
    with pytest.raises(ProviderUnavailable, match="HITHINK_FINANCE_API_KEY"):
        provider.capability("prices_snapshot", thscodes="600519.SH")


def test_tradability_fails_loud_instead_of_fabricating_flags() -> None:
    provider = HithinkFinanceProvider(allow_network=False)
    with pytest.raises(ProviderUnavailable, match="historical tradability"):
        provider.tradability(ProviderRequest("2026-01-01", "2026-01-31", ("600519.SH",)))


def test_arbitrary_endpoint_is_rejected() -> None:
    provider = HithinkFinanceProvider(allow_network=False)
    with pytest.raises(ValueError, match="unsupported HiThink Finance capability"):
        provider.capability("https://example.com/not-allowed")
