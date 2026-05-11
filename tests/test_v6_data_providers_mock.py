from quantagent.data.providers.base import ProviderRequest
from quantagent.data.providers.mock_provider import MockProvider


def test_mock_provider_returns_core_v6_data_contracts():
    provider = MockProvider()
    request = ProviderRequest("2026-01-02", "2026-01-16", universe="CSI300")
    market = provider.daily_ohlcv(request)
    news = provider.news(request)
    fundamentals = provider.fundamentals(request)
    calendar = provider.trading_days(request)

    assert {"trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"}.issubset(market.frame.columns)
    assert {"timestamp", "symbol", "title", "summary"}.issubset(news.frame.columns)
    assert {"symbol", "announcement_time", "roe", "debt_to_asset"}.issubset(fundamentals.frame.columns)
    assert calendar.frame["is_trading_day"].all()
    assert market.quality_score >= 0.9

