"""Offline tests for TickflowProvider — uses a fake SDK client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.providers import tickflow_provider as tp


# ---------------------------------------------------------------------------
# Fake SDK
# ---------------------------------------------------------------------------


def _base_daily(symbol: str) -> pd.DataFrame:
    # Return 5 deterministic days for any symbol
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    return pd.DataFrame({
        "symbol":     [symbol] * 5,
        "name":       ["FOO"] * 5,
        "timestamp":  list(range(5)),
        "trade_date": dates,
        "trade_time": dates,
        "open":  [10.0, 10.5, 11.0, 11.5, 12.0],
        "high":  [10.6, 10.9, 11.5, 11.9, 12.5],
        "low":   [ 9.5,  9.9, 10.5, 11.0, 11.5],
        # close engineered: day 1 is a limit-up off prev=10.5 (=11.55), day 4 is a limit-down off 12.0 (=10.80)
        "close": [10.5, 11.55, 11.0, 12.0, 10.80],
        "volume": [1000, 2000, 0, 1500, 1800],  # day 3 = suspended
        "amount": [1e6, 2e6, 0.0, 1.5e6, 1.8e6],
    })


# Per-row qfq factor the fake applies when adjust="forward"; engineered so a
# 2:1 split lands on day-4 (raw close 12.0 → 24.0) while day-1 stays at 10.5.
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
        return {sym: self.get(sym, period=period, count=count,
                              as_dataframe=as_dataframe, adjust=adjust)
                for sym in symbols}


@dataclass
class _FakeKlinesBatchGated(_FakeKlines):
    """Models the live subscription: per-symbol get works, batch is tier-gated."""

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
                {"symbol": "600001.SH", "name": "ST邯钢",   "type": "stock", "ext": {"listing_date": "1999-01-01"}},
                {"symbol": "000852.SH", "name": "中证1000", "type": "index", "ext": {}},  # filtered out
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
            {"id": "CN_Equity_SW1_222", "name": "SW1金融",     "symbols": []},
            {"id": "CN_Equity_SW2_333", "name": "SW2白酒",     "symbols": []},
            {"id": "HK_Equity", "name": "HK", "symbols": []},  # filtered out
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

    def close(self): pass


@pytest.fixture
def fake_provider(monkeypatch):
    """Yield a TickflowProvider wired to the fake SDK."""
    fake = _FakeTickFlow(api_key="x")
    monkeypatch.setenv("TICKFLOW_API_KEY", "fake")

    p = tp.TickflowProvider(allow_network=True)
    # Bypass the lazy SDK import by injecting the client directly.
    p._client = fake
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fail_loud_without_network():
    p = tp.TickflowProvider(allow_network=False)
    with pytest.raises(ProviderUnavailable, match="allow_network=False"):
        p.daily_ohlcv(ProviderRequest("2024-01-02", "2024-06-30", ("600519.SH",)))


def test_daily_ohlcv_allows_free_client_without_token(monkeypatch):
    monkeypatch.delenv("TICKFLOW_API_KEY", raising=False)
    p = tp.TickflowProvider(allow_network=True)
    p._client = _FakeTickFlow(api_key="")

    result = p.daily_ohlcv(ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH",)))

    assert not result.frame.empty
    assert result.source == "tickflow"


def test_full_service_still_fails_without_token(monkeypatch):
    monkeypatch.delenv("TICKFLOW_API_KEY", raising=False)
    p = tp.TickflowProvider(allow_network=True)
    with pytest.raises(ProviderUnavailable, match="TICKFLOW_API_KEY"):
        p.stock_basic()


def test_daily_ohlcv_canonical_columns(fake_provider):
    r = fake_provider.daily_ohlcv(
        ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH", "000001.SZ")),
    )
    assert r.source == "tickflow"
    assert r.frame.shape[0] == 10  # 5 days × 2 syms
    assert list(r.frame.columns) == list(tp.CANONICAL_OHLCV_COLUMNS)
    # PIT invariant
    assert (r.frame["available_at"] >= r.frame["trade_date"]).all()


def test_adjusted_prices_uses_forward_adjust(fake_provider):
    """adjusted_prices must request SDK-side qfq (adjust='forward'), not ex_factors.

    The fake applies a 2:1 split factor on day-4 when adjust='forward'. The
    dedicated ex_factors endpoint is permission-gated on the live tier, so the
    provider relies on the K-line ``adjust`` argument instead.
    """
    r = fake_provider.adjusted_prices(
        ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH",)),
    )
    closes = r.frame["close"].tolist()
    assert closes[0] == pytest.approx(10.5)   # raw 10.5 × 1.0
    assert closes[3] == pytest.approx(24.0)   # raw 12.0 × 2.0 (split)
    assert r.metadata.get("adjust_kind") == "qfq"


def test_daily_ohlcv_is_raw_not_adjusted(fake_provider):
    """daily_ohlcv must NOT apply the forward adjustment (raw OHLCV path)."""
    r = fake_provider.daily_ohlcv(
        ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH",)),
    )
    # Day-4 raw close stays 12.0 (no split applied) — proves adjust is not passed.
    assert r.frame["close"].iloc[3] == pytest.approx(12.0)


def test_daily_ohlcv_batch_gated_falls_back_to_per_symbol(monkeypatch):
    """When the batch K-line tier is denied, multi-symbol fetch loops get()."""
    fake = _FakeTickFlow(api_key="x")
    fake.klines = _FakeKlinesBatchGated()
    monkeypatch.setenv("TICKFLOW_API_KEY", "fake")
    p = tp.TickflowProvider(allow_network=True)
    p._client = fake

    r = p.daily_ohlcv(
        ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH", "000001.SZ")),
    )
    # Same 10 rows (5 days × 2 syms) as the batch path would have produced.
    assert r.frame.shape[0] == 10
    assert set(r.frame["symbol"]) == {"600519.SH", "000001.SZ"}


def test_tradability_derives_flags(fake_provider):
    r = fake_provider.tradability(
        ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH",)),
    )
    df = r.frame
    # Day 1: limit-up (close 11.55 ≈ 10.5 × 1.10)
    assert bool(df["is_limit_up"].iloc[1])
    # Day 2: volume == 0 → suspended
    assert bool(df["is_suspended"].iloc[2])
    # Day 4: limit-down (10.71 ≈ 11.9 × 0.90)
    assert bool(df["is_limit_down"].iloc[4])
    # 贵州茅台 isn't ST
    assert not df["is_st"].any()


def test_tradability_board_aware_chinext(fake_provider):
    """A ChiNext name at +10% is NOT sealed (its limit is 20%).

    The fake klines engineer day-1 close 11.55 = 10.5 × 1.10. Under the old
    flat-10% rule this was flagged limit-up for every board; the board-aware
    rule must leave a ChiNext (300xxx) name unflagged at +10%, while a
    main-board name at the same +10% IS sealed.
    """
    main = fake_provider.tradability(
        ProviderRequest("2024-01-02", "2024-01-31", ("600519.SH",)),
    ).frame
    chinext = fake_provider.tradability(
        ProviderRequest("2024-01-02", "2024-01-31", ("300001.SZ",)),
    ).frame
    # Main board: +10% off 10.5 is a real seal.
    assert bool(main["is_limit_up"].iloc[1])
    # ChiNext: +10% is NOT a seal (limit is 20%) — the board-aware fix.
    assert not bool(chinext["is_limit_up"].iloc[1])


def test_tradability_detects_current_st(fake_provider):
    r = fake_provider.tradability(
        ProviderRequest("2024-01-02", "2024-01-31", ("600001.SH",)),
    )
    # 600001.SH is "ST邯钢" in fake exchanges → all rows is_st=True
    assert r.frame["is_st"].all()


def test_stock_basic_joins_industry(fake_provider):
    basic = fake_provider.stock_basic()
    assert "industry" in basic.columns
    # 600519.SH → SW1 食品饮料
    moutai = basic[basic["symbol"] == "600519.SH"].iloc[0]
    assert "食品饮料" in str(moutai["industry"])
    assert "白酒" in str(moutai["industry_sub"])
    # Index is filtered out
    assert "000852.SH" not in basic["symbol"].tolist()


def test_industry_map_is_cached(fake_provider, monkeypatch):
    calls = {"n": 0}
    orig_get = fake_provider._client.universes.get

    def counted_get(uid):
        calls["n"] += 1
        return orig_get(uid)

    fake_provider._client.universes.get = counted_get
    fake_provider.stock_basic()
    first = calls["n"]
    fake_provider.stock_basic()
    # Second call should not re-walk universes
    assert calls["n"] == first


def test_namechange_history_is_empty_frame(fake_provider):
    nc = fake_provider.namechange_history()
    assert isinstance(nc, pd.DataFrame)
    assert nc.empty
    assert set(nc.columns) == {"symbol", "name", "start_date", "end_date"}
