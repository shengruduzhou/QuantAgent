from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.providers.fuyao_provider import FuyaoProvider


@dataclass
class _Outcome:
    ok: bool
    payload: object
    retry_class: str = "OK"
    error: str | None = None


class _FakeHttp:
    def __init__(self, payloads: dict[str, object]):
        self.payloads = payloads
        self.calls: list[tuple[str, dict | None, dict | None]] = []

    def get_json(self, url: str, params=None, headers=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        path = url.split("fuyao.aicubes.cn", 1)[-1]
        payload = self.payloads[path]
        return _Outcome(True, payload)


def _ok(items: list[dict], *, timestamp: int = 1_720_000_000_000) -> dict:
    return {"code": 0, "message": "ok", "request_id": "req-test", "data": {"timestamp": timestamp, "item": items}}


def test_network_is_fail_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-key-not-secret")
    provider = FuyaoProvider(allow_network=False)
    with pytest.raises(ProviderUnavailable, match="allow_network=True"):
        provider.daily_ohlcv(ProviderRequest("2024-01-01", "2024-01-10", ("600519.SH",)))


def test_missing_key_is_fail_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HITHINK_FINANCE_API_KEY", raising=False)
    provider = FuyaoProvider(allow_network=True)
    with pytest.raises(ProviderUnavailable, match="HITHINK_FINANCE_API_KEY"):
        provider.daily_ohlcv(ProviderRequest("2024-01-01", "2024-01-10", ("600519.SH",)))


def test_daily_prices_are_normalized_with_provenance(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-key-not-secret")
    date_ms = int(pd.Timestamp("2024-01-02", tz="Asia/Shanghai").timestamp() * 1000)
    fake = _FakeHttp({
        "/api/a-share/prices/historical": _ok([{
            "date_ms": date_ms,
            "open_price": 10.0,
            "high_price": 11.0,
            "low_price": 9.5,
            "close_price": 10.5,
            "volume": 12345,
            "turnover": 456789.0,
        }]),
    })
    provider = FuyaoProvider(allow_network=True, _http=fake)
    result = provider.daily_ohlcv(ProviderRequest("2024-01-01", "2024-01-10", ("600519.SH",)))

    assert result.point_in_time is True
    assert result.frame.loc[0, "symbol"] == "600519.SH"
    assert result.frame.loc[0, "close"] == pytest.approx(10.5)
    assert result.frame.loc[0, "amount"] == pytest.approx(456789.0)
    assert result.frame.loc[0, "available_at"] == pd.Timestamp("2024-01-03")
    assert result.frame.loc[0, "source"] == "hithink_fuyao"
    assert result.frame.loc[0, "quality_status"] == "official_api"
    assert fake.calls[0][2]["X-api-key"] == "test-key-not-secret"
    assert fake.calls[0][1]["adjust"] == "none"


def test_adjusted_prices_request_forward_adjustment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-key-not-secret")
    fake = _FakeHttp({"/api/a-share/prices/historical": _ok([])})
    provider = FuyaoProvider(allow_network=True, _http=fake)
    provider.adjusted_prices(ProviderRequest("2024-01-01", "2024-01-10", ("600519.SH",)))
    assert fake.calls[0][1]["adjust"] == "forward"


def test_backward_adjustment_is_explicitly_supported(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-key-not-secret")
    fake = _FakeHttp({"/api/a-share/prices/historical": _ok([])})
    provider = FuyaoProvider(allow_network=True, _http=fake)
    provider.historical_prices(
        ProviderRequest("2024-01-01", "2024-01-10", ("600519.SH",)),
        adjust="backward",
    )
    assert fake.calls[0][1]["adjust"] == "backward"


def test_financial_available_at_uses_disclosure_date_not_period_end(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-key-not-secret")
    period_ms = int(pd.Timestamp("2024-03-31", tz="Asia/Shanghai").timestamp() * 1000)
    disclosure_ms = int(pd.Timestamp("2024-04-29", tz="Asia/Shanghai").timestamp() * 1000)
    row = {
        "thscode": "600519.SH",
        "period_end_ms": period_ms,
        "report_date_ms": disclosure_ms,
        "revenue": 100.0,
    }
    fake = _FakeHttp({
        "/api/a-share/financials/income-statements": _ok([row]),
        "/api/a-share/financials/balance-sheets": _ok([{**row, "assets_total": 1000.0}]),
        "/api/a-share/financials/cash-flow-statements": _ok([{**row, "act_cash_flow_net": 50.0}]),
    })
    provider = FuyaoProvider(allow_network=True, _http=fake)
    result = provider.fundamentals(ProviderRequest("2024-01-01", "2024-06-30", ("600519.SH",)))

    assert not result.frame.empty
    assert set(result.frame["statement_type"]) == {"income", "balance", "cashflow"}
    assert (result.frame["report_period"] == pd.Timestamp("2024-03-31")).all()
    assert (result.frame["ann_date"] == pd.Timestamp("2024-04-29")).all()
    assert (result.frame["available_at"] == pd.Timestamp("2024-04-29")).all()
    assert result.metadata["pit_key"] == "report_date_ms"


def test_financials_fail_if_upstream_drops_pit_fields(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-key-not-secret")
    bad = {"thscode": "600519.SH", "period_end_ms": 1_700_000_000_000}
    fake = _FakeHttp({
        "/api/a-share/financials/income-statements": _ok([bad]),
        "/api/a-share/financials/balance-sheets": _ok([]),
        "/api/a-share/financials/cash-flow-statements": _ok([]),
    })
    provider = FuyaoProvider(allow_network=True, _http=fake)
    with pytest.raises(ProviderUnavailable, match="report_date_ms"):
        provider.fundamentals(ProviderRequest("2024-01-01", "2024-06-30", ("600519.SH",)))


def test_financial_indicators_follow_official_abilities_array_and_are_not_pit_eligible(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-key-not-secret")
    payload = {
        "code": 0,
        "message": "ok",
        "request_id": "req-indicator",
        "data": {
            "thscode": "600519.SH",
            "report": "2024-4",
            "abilities": [
                {
                    "ability": "growth",
                    "indicators": [
                        {"index_id": "revenue_yoy", "value": 0.15},
                        {"index_id": "profit_yoy", "value": 0.12},
                    ],
                },
                {
                    "ability": "profitability",
                    "indicators": [{"index_id": "roe", "value": 0.31}],
                },
            ],
        },
    }
    fake = _FakeHttp({"/api/a-share/financials/indicators": payload})
    provider = FuyaoProvider(allow_network=True, _http=fake)

    frame = provider.financial_indicators("600519.SH", "2024-4")

    assert frame[["ability", "index_id"]].to_records(index=False).tolist() == [
        ("growth", "revenue_yoy"),
        ("growth", "profit_yoy"),
        ("profitability", "roe"),
    ]
    assert (~frame["pit_eligible"]).all()
    assert (~frame["point_in_time_valid"]).all()
    assert frame["available_at"].notna().all()


def test_tradability_is_not_fabricated():
    provider = FuyaoProvider(allow_network=True)
    with pytest.raises(ProviderUnavailable, match="tradability"):
        provider.tradability(ProviderRequest("2024-01-01", "2024-01-10", ("600519.SH",)))
