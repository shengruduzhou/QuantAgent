from __future__ import annotations

import pandas as pd

import services.quant_api.services.market_data as market_module
from services.quant_api.services.connections import ConnectionManager
from services.quant_api.services.market_data import MarketDataService


class _FakeResult:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame


class _FakeProvider:
    def ticker_search(self, query: str, **_: object) -> pd.DataFrame:
        return pd.DataFrame([{
            "thscode": "600519.SH",
            "ticker": "600519",
            "name": "贵州茅台",
            "exchange": "SH",
            "asset_type": "a-share",
            "currency": "CNY",
        }])

    def snapshot(self, symbols: tuple[str, ...]) -> pd.DataFrame:
        assert symbols == ("600519.SH",)
        return pd.DataFrame([{
            "symbol": "600519.SH",
            "ticker": "600519",
            "last_price": 1500.0,
            "price_change": 15.0,
            "price_change_ratio_pct": 1.01,
            "open_price": 1488.0,
            "high_price": 1510.0,
            "low_price": 1480.0,
            "prev_price": 1485.0,
            "volume": 1_000_000,
            "amount": 1_500_000_000.0,
            "available_at": pd.Timestamp("2026-08-07 15:00:00"),
        }])

    def historical_prices(self, request, *, adjust: str):
        assert request.symbols == ("600519.SH",)
        assert adjust == "forward"
        dates = pd.date_range("2025-08-01", periods=260, freq="B")
        return _FakeResult(pd.DataFrame({
            "symbol": "600519.SH",
            "trade_date": dates,
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": [100.0 + index for index in range(len(dates))],
            "volume": 1_000_000,
            "amount": 100_000_000.0,
        }))

    def valuations_snapshot(self, symbols: tuple[str, ...]) -> pd.DataFrame:
        assert symbols == ("600519.SH",)
        return pd.DataFrame([{
            "thscode": "600519.SH",
            "name": "贵州茅台",
            "pe_ttm": 20.5,
            "pe_mrq": 20.1,
            "pb_mrq": 7.2,
            "ps_ttm": 10.3,
            "pcf_ttm": 18.4,
        }])


def test_market_data_service_builds_frontend_safe_overview(monkeypatch):
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-key-not-secret")
    monkeypatch.setattr(market_module, "FuyaoProvider", lambda allow_network: _FakeProvider())
    service = MarketDataService(ConnectionManager())

    result = service.stock_overview("600519.SH")

    assert result["source"] == "hithink_fuyao"
    assert result["adjustment"] == "forward"
    assert result["name"] == "贵州茅台"
    assert result["snapshot"]["lastPrice"] == 1500.0
    assert result["valuation"]["peTtm"] == 20.5
    assert len(result["bars"]) == 260
    assert result["bars"][-1]["datetime"].startswith("2026")
    assert "test-key-not-secret" not in str(result)
    assert "api_key" not in str(result).lower()


def test_market_search_returns_normalized_a_share_identity(monkeypatch):
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-key-not-secret")
    monkeypatch.setattr(market_module, "FuyaoProvider", lambda allow_network: _FakeProvider())
    service = MarketDataService(ConnectionManager())

    rows = service.search("茅台")

    assert rows == [{
        "symbol": "600519.SH",
        "ticker": "600519",
        "name": "贵州茅台",
        "exchange": "SH",
        "assetType": "a-share",
        "currency": "CNY",
    }]
