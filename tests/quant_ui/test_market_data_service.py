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
        return pd.DataFrame([{
            "symbol": symbol,
            "ticker": symbol.split(".")[0],
            "last_price": 1500.0,
            "price_change": 15.0,
            "price_change_ratio_pct": 1.01,
            "open": 1488.0,
            "high": 1510.0,
            "low": 1480.0,
            "prev": 1485.0,
            "volume": 1_000_000,
            "amount": 1_500_000_000.0,
            "available_at": pd.Timestamp("2026-08-07 15:00:00"),
        } for symbol in symbols])

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

    def index_daily(self, request):
        dates = pd.date_range("2026-01-01", periods=120, freq="B")
        return _FakeResult(pd.DataFrame({
            "symbol": request.symbols[0],
            "trade_date": dates,
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": [100 + index * 0.1 for index in range(len(dates))],
            "volume": 2_000_000,
            "amount": 300_000_000.0,
        }))

    def index_constituents(self, thscode: str) -> pd.DataFrame:
        assert thscode == "881101.TI"
        return pd.DataFrame([
            {"thscode": "600519.SH", "ticker": "600519", "name": "贵州茅台"},
            {"thscode": "000858.SZ", "ticker": "000858", "name": "五粮液"},
        ])

    def get_capability(self, path: str, params=None):
        if path.endswith("ths-index-list"):
            return {"timestamp": 1, "item": [{"thscode": "881101.TI", "name": "食品饮料"}]}
        if "financials" in path:
            base = {
                "thscode": "600519.SH",
                "period": "annual",
                "fiscal_year": 2025,
                "report_date_ms": 1767110400000,
                "period_end_ms": 1767110400000,
                "currency": "CNY",
            }
            if "income" in path:
                base.update({"operating_revenue": 180_000_000_000, "net_profit": 90_000_000_000})
            if "balance" in path:
                base.update({"total_assets": 300_000_000_000, "total_liabilities": 60_000_000_000})
            if "cash-flow" in path:
                base.update({"act_cash_flow_net": 100_000_000_000})
            return {"timestamp": 1, "item": [base]}
        if path.endswith("hot-stock-list"):
            return {"timestamp": 1, "item": [{"thscode": "600519.SH", "name": "贵州茅台", "rank": 1, "heat": "1000", "rank_change": 2}]}
        if path.endswith("skyrocket-list"):
            return {"timestamp": 1, "item": [{"thscode": "000858.SZ", "name": "五粮液", "rank": 1, "heat": "900", "rank_change": 5}]}
        if path.endswith("dragon-tiger-list"):
            return {"timestamp": 1, "trade_date": "2026-08-07", "stock_count": 1, "stock_items": [{"thscode": "600519.SH", "name": "贵州茅台", "net_value": 100_000_000}]}
        if path.endswith("limit-up-ladder"):
            return {"timestamp": 1, "window": {"length": 30}, "item": [{"date": "20260807", "boards": {"two_board": [{"thscode": "600519.SH", "name": "贵州茅台", "board_num": 2}]}}]}
        raise AssertionError(path)


def _service(monkeypatch) -> MarketDataService:
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-key-not-secret")
    monkeypatch.setattr(market_module, "FuyaoProvider", lambda allow_network: _FakeProvider())
    return MarketDataService(ConnectionManager())


def test_market_data_service_builds_frontend_safe_overview(monkeypatch):
    result = _service(monkeypatch).stock_overview("600519.SH")

    assert result["source"] == "hithink_fuyao"
    assert result["adjustment"] == "forward"
    assert result["name"] == "贵州茅台"
    assert result["snapshot"]["lastPrice"] == 1500.0
    assert result["snapshot"]["open"] == 1488.0
    assert result["snapshot"]["high"] == 1510.0
    assert result["snapshot"]["low"] == 1480.0
    assert result["snapshot"]["prevClose"] == 1485.0
    assert result["valuation"]["peTtm"] == 20.5
    assert len(result["bars"]) == 260
    assert result["bars"][-1]["datetime"].startswith("2026")
    assert "test-key-not-secret" not in str(result)
    assert "api_key" not in str(result).lower()


def test_market_search_returns_normalized_a_share_identity(monkeypatch):
    rows = _service(monkeypatch).search("茅台")
    assert rows == [{
        "symbol": "600519.SH",
        "ticker": "600519",
        "name": "贵州茅台",
        "exchange": "SH",
        "assetType": "a-share",
        "currency": "CNY",
    }]


def test_financial_health_preserves_disclosure_fields(monkeypatch):
    result = _service(monkeypatch).financial_health("600519.SH")
    assert result["pitKey"] == "report_date_ms"
    assert result["periodKey"] == "period_end_ms"
    assert result["statements"]["income"][0]["report_date_ms"] == 1767110400000
    assert result["statements"]["cashflow"][0]["act_cash_flow_net"] == 100_000_000_000


def test_index_views_and_constituent_snapshot(monkeypatch):
    service = _service(monkeypatch)
    catalog = service.index_catalog("industry")
    overview = service.index_overview("881101.TI")
    assert catalog["items"][0]["thscode"] == "881101.TI"
    assert overview["constituentCount"] == 2
    assert len(overview["snapshots"]) == 2
    assert len(overview["bars"]) == 120


def test_market_intelligence_is_composed_from_real_capabilities(monkeypatch):
    result = _service(monkeypatch).market_intelligence()
    assert result["issues"] == []
    assert result["panels"]["hotDay"]["item"][0]["rank"] == 1
    assert result["panels"]["dragonAll"]["stock_count"] == 1
    assert result["panels"]["limitLadder"]["window"]["length"] == 30
    assert "test-key-not-secret" not in str(result)
