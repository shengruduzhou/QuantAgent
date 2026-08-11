"""Offline tests for TickflowProvider — uses a fake SDK client."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.providers import tickflow_provider as tp
from quantagent.data.providers.st_pit import HistoricalSTCoverageError


def _base_daily(symbol: str) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    return pd.DataFrame({
        "symbol": [symbol] * 5,
        "name": ["FOO"] * 5,
        "timestamp": list(range(5)),
        "trade_date": dates,
        "trade_time": dates,
        "open": [10.0, 10.5, 11.0, 11.5, 12.0],
        "high": [10.6, 10.9, 11.5, 11.9, 12.5],
        "low": [9.5, 9.9, 10.5, 11.0, 11.5],
        "close": [10.5, 11.55, 11.0, 12.0, 10.80],
        "volume": [1000, 2000, 0, 1500, 1800],
        "amount": [1e6, 2e6, 0.0, 1.5e6, 1.8e6],
    })


_FAKE_FORWARD_FACTOR = [1.0, 1.0, 1.0, 2.0, 2.0]


@dataclass
class _FakeKlines:
    def get(self, symbol: str, *, period: str, count: int, as_dataframe: bool,
            adjust: str | None = None):
        df = _base_daily(symbol)
        if adjust == "forward":
            for col in ("open", "high", "low", "close"):
                df[col] = df[col] * pd.Series(_FAKE_FORWARD_FACTOR)
        return df

    def batch(self, symbols, *, period, count, as_dataframe, show_progress,
              adjust: str | None = None):
        return {
            sym: self.get(
                sym,
                period=period,
                count=count,
                as_dataframe=as_dataframe,
                adjust=adjust,
            )
            for sym in symbols
        }


@dataclass
class _FakeKlinesBatchGated(_FakeKlines):
    def batch(self, symbols, *, period, count, as_dataframe, show_progress,
              adjust: str | None = None):
        raise RuntimeError("无日/周/月K线查询批量查询权限")


@dataclass
class _FakeExchanges:
    def get_instruments(self, exchange: str):
        if exchange == "SH":
            return [
                {"symbol": "600519.SH", "name": "贵州茅台", "type": "stock", "ext": {"listing_date": "2001-08-27"}},
                {"symbol": "601318.SH", "name": "中国平安", "type": "stock", "ext": {"listing_date": "2007-03-01"}},
                {"symbol": "600001.SH", "name": "ST邯钢", "type": "stock", "ext": {"listing_date": "1999-01-01"}},
                {"symbol": "000852.SH", "name": "中证1000", "type": "index", "ext": {}},
            ]
        if exchange == "SZ":
            return [
                {"symbol": "000001.SZ", "name": "平安银行", "type": "stock", "ext": {}},
                {"symbol": "002001.SZ", "name": "*ST 新和", "type": "stock", "ext": {}},
            ]
        if exchange == "BJ":
            return []
        return []


@dataclass
class _FakeUniverses:
    def list(self):
        return [
            {"id": "CN_Equity_SW1_111", "name": "SW1食品饮料", "symbols": []},
            {"id": "CN_Equity_SW1_222", "name": "SW1金融", "symbols": []},
            {"id": "CN_Equity_SW2_333", "name": "SW2白酒", "symbols": []},
            {"id": "HK_Equity", "name": "HK", "symbols": []},
        ]

    def get(self, uid: str):
        if uid == "CN_Equity_SW1_111":
            return {"id": uid, "name": "SW1食品饮料", "symbols": ["600519.SH"]}
        if uid == "CN_Equity_SW1_222":
            return {"id": uid, "name": "SW1金融", "symbols": ["601318.SH", "000001.SZ"]}
        if uid == "CN_Equity_SW2_333":
            return {"id": uid, "name": "SW2白酒", "symbols": ["600519.SH"]}
        return {"symbols": []}


@dataclass
class _FakeTickFlow:
    api_key: str = ""
    klines: _FakeKlines = None
    exchanges: _FakeExchanges = None
    universes: _FakeUniverses = None

    def __post_init__(self):
        self.klines = _FakeKlines()
        self.exchanges = _FakeExchanges()
        self.universes = _FakeUniverses()

    def close(self):
        pass


@pytest.fixture
def fake_provider(monkeypatch):
    fake = _FakeTickFlow(api_key="x")
    monkeypatch.setenv("TICKFLOW_API_KEY", "fake")
    provider = tp.TickflowProvider(allow_network=True)
    provider._client = fake
    return provider


def test_fail_loud_without_network():
    provider = tp.TickflowProvider(allow_network=False)
    with pytest.raises(ProviderUnavailable, match="allow_network=False"):
        provider.daily_ohlcv(
            ProviderRequest("2024-01-02", "2024-06-30", ("600519.SH",))
        )


def test_daily_ohlcv_allows_free_client_without_token(monkeypatch):
    monkeypatch.delenv("TICKFLOW_API_KEY", raising=False)
    provider = tp.TickflowProvider(allow_network=True)
    provider._client = _FakeTickFlow(api_key="")
    result = provider.daily_ohlcv(
        ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH",))
    )
    assert not result.frame.empty
    assert result.source == "tickflow"


def test_full_service_still_fails_without_token(monkeypatch):
    monkeypatch.delenv("TICKFLOW_API_KEY", raising=False)
    provider = tp.TickflowProvider(allow_network=True)
    with pytest.raises(ProviderUnavailable, match="TICKFLOW_API_KEY"):
        provider.stock_basic()


def test_daily_ohlcv_canonical_columns(fake_provider):
    result = fake_provider.daily_ohlcv(
        ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH", "000001.SZ"))
    )
    assert result.source == "tickflow"
    assert result.frame.shape[0] == 10
    assert list(result.frame.columns) == list(tp.CANONICAL_OHLCV_COLUMNS)
    assert (result.frame["available_at"] >= result.frame["trade_date"]).all()


def test_adjusted_prices_uses_forward_adjust(fake_provider):
    result = fake_provider.adjusted_prices(
        ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH",))
    )
    closes = result.frame["close"].tolist()
    assert closes[0] == pytest.approx(10.5)
    assert closes[3] == pytest.approx(24.0)
    assert result.metadata.get("adjust_kind") == "qfq"


def test_daily_ohlcv_is_raw_not_adjusted(fake_provider):
    result = fake_provider.daily_ohlcv(
        ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH",))
    )
    assert result.frame["close"].iloc[3] == pytest.approx(12.0)


def test_daily_ohlcv_batch_gated_falls_back_to_per_symbol(monkeypatch):
    fake = _FakeTickFlow(api_key="x")
    fake.klines = _FakeKlinesBatchGated()
    monkeypatch.setenv("TICKFLOW_API_KEY", "fake")
    provider = tp.TickflowProvider(allow_network=True)
    provider._client = fake
    result = provider.daily_ohlcv(
        ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH", "000001.SZ"))
    )
    assert result.frame.shape[0] == 10
    assert set(result.frame["symbol"]) == {"600519.SH", "000001.SZ"}


def test_historical_tradability_fails_closed_without_dated_st_state(fake_provider):
    with pytest.raises(ProviderUnavailable, match="historical tradability is fail-closed"):
        fake_provider.tradability(
            ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH",))
        )


def test_current_snapshot_tradability_derives_flags_but_is_not_pit(fake_provider):
    result = fake_provider.current_snapshot_tradability(
        ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH",))
    )
    frame = result.frame
    assert bool(frame["is_limit_up"].iloc[1])
    assert bool(frame["is_suspended"].iloc[2])
    assert bool(frame["is_limit_down"].iloc[4])
    assert not frame["is_st"].any()
    assert result.point_in_time is False
    assert result.metadata["st_coverage_status"] == "current_snapshot"
    assert result.metadata["historical_pit_certified"] is False
    assert frame["st_coverage_status"].eq("current_snapshot").all()
    assert not frame["point_in_time_valid"].any()


def test_current_snapshot_tradability_still_uses_board_aware_caps(fake_provider):
    main = fake_provider.current_snapshot_tradability(
        ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH",))
    ).frame
    chinext = fake_provider.current_snapshot_tradability(
        ProviderRequest("2024-01-02", "2024-01-31", ("300001.SZ",))
    ).frame
    assert bool(main["is_limit_up"].iloc[1])
    assert not bool(chinext["is_limit_up"].iloc[1])


def test_current_snapshot_st_is_explicitly_non_historical(fake_provider):
    result = fake_provider.current_snapshot_tradability(
        ProviderRequest("2024-01-02", "2024-01-31", ("600001.SH",))
    )
    assert result.frame["is_st"].all()
    assert result.point_in_time is False
    assert "tickflow_current_snapshot_st_not_historical_pit" in result.warnings


def test_private_historical_tradability_path_cannot_bypass_guard(fake_provider):
    with pytest.raises(HistoricalSTCoverageError, match="current snapshot"):
        fake_provider._call_tickflow_tradability(
            ProviderRequest("2024-01-02", "2024-01-31", ("600001.SH",))
        )


def test_stock_basic_joins_industry(fake_provider):
    basic = fake_provider.stock_basic()
    assert "industry" in basic.columns
    moutai = basic[basic["symbol"] == "600519.SH"].iloc[0]
    assert "食品饮料" in str(moutai["industry"])
    assert "白酒" in str(moutai["industry_sub"])
    assert "000852.SH" not in basic["symbol"].tolist()


def test_industry_map_is_cached(fake_provider):
    calls = {"n": 0}
    original_get = fake_provider._client.universes.get

    def counted_get(uid):
        calls["n"] += 1
        return original_get(uid)

    fake_provider._client.universes.get = counted_get
    fake_provider.stock_basic()
    first = calls["n"]
    fake_provider.stock_basic()
    assert calls["n"] == first


def test_namechange_history_is_empty_frame(fake_provider):
    history = fake_provider.namechange_history()
    assert isinstance(history, pd.DataFrame)
    assert history.empty
    assert set(history.columns) == {"symbol", "name", "start_date", "end_date"}
