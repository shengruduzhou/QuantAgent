from __future__ import annotations

import sys
import types

import pandas as pd

from quantagent.data.bootstrap.akshare_market_bootstrap import _normalise_dtypes
from quantagent.data.providers.akshare_live_provider import (
    AkShareLiveProvider,
    _normalize_akshare_daily,
    akshare_market_schema_report,
)
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.trading_calendar import TradingCalendar


def _calendar() -> TradingCalendar:
    return TradingCalendar.from_dates(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )


def test_eastmoney_daily_volume_lots_are_normalized_to_shares() -> None:
    raw = pd.DataFrame(
        [
            {
                "日期": "2024-01-02",
                "开盘": 10.0,
                "最高": 11.0,
                "最低": 9.5,
                "收盘": 10.5,
                "成交量": 1234,
                "成交额": 12_950_000.0,
            }
        ]
    )

    result = _normalize_akshare_daily(
        raw,
        "600000.SH",
        source="east_money",
        adjust="",
        trading_calendar=_calendar(),
    )

    assert result["volume"].tolist() == [123_400.0]
    assert result["volume_unit"].tolist() == ["shares"]
    assert result["raw_volume_unit"].tolist() == ["lots_100_shares"]
    assert result["amount_unit"].tolist() == ["CNY"]
    assert result["available_at"].tolist() == ["2024-01-03"]
    assert result["point_in_time_valid"].tolist() == [True]


def test_sina_daily_volume_shares_are_not_rescaled() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "open": 10.0,
                "high": 11.0,
                "low": 9.5,
                "close": 10.5,
                "volume": 123_400,
                "amount": 12_950_000.0,
            }
        ]
    )

    result = _normalize_akshare_daily(
        raw,
        "600000.SH",
        source="sina",
        adjust="",
        trading_calendar=_calendar(),
    )

    assert result["volume"].tolist() == [123_400]
    assert result["raw_volume_unit"].tolist() == ["shares"]
    assert result["point_in_time_valid"].tolist() == [True]


def test_adjusted_history_is_not_pit_without_vintaged_adjustment_factors() -> None:
    raw = pd.DataFrame(
        [
            {
                "日期": "2024-01-02",
                "开盘": 10.0,
                "最高": 11.0,
                "最低": 9.5,
                "收盘": 10.5,
                "成交量": 10,
                "成交额": 10_500.0,
            }
        ]
    )

    result = _normalize_akshare_daily(
        raw,
        "600000.SH",
        source="east_money",
        adjust="qfq",
        trading_calendar=_calendar(),
    )

    assert result["price_adjustment"].tolist() == ["qfq"]
    assert result["adjustment_pit_vintage_bound"].tolist() == [False]
    assert result["point_in_time_valid"].tolist() == [False]
    assert akshare_market_schema_report(result)["status"] == "failed"


def test_missing_trading_calendar_never_falls_back_to_weekday_arithmetic() -> None:
    raw = pd.DataFrame(
        [
            {
                "日期": "2024-02-09",
                "开盘": 10.0,
                "最高": 11.0,
                "最低": 9.5,
                "收盘": 10.5,
                "成交量": 10,
                "成交额": 10_500.0,
            }
        ]
    )

    result = _normalize_akshare_daily(
        raw,
        "600000.SH",
        source="east_money",
        adjust="",
        trading_calendar=None,
    )

    assert result["available_at"].isna().all()
    assert result["point_in_time_valid"].tolist() == [False]
    assert akshare_market_schema_report(result)["pit_violation_count"] >= 1


def test_bootstrap_dtype_normalization_never_upgrades_unknown_pit_to_true() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "trade_date": "2024-01-02",
                "available_at": None,
                "point_in_time_valid": None,
                "open": 10.0,
                "high": 11.0,
                "low": 9.5,
                "close": 10.5,
                "volume": 1000,
                "amount": 10_500.0,
            }
        ]
    )

    result = _normalise_dtypes(frame)

    assert result["available_at"].isna().all()
    assert result["point_in_time_valid"].tolist() == [False]


def test_provider_binds_version_units_and_complete_symbol_coverage(monkeypatch) -> None:
    def stock_zh_a_hist(**kwargs):
        return pd.DataFrame(
            [
                {
                    "日期": "2024-01-02",
                    "开盘": 10.0,
                    "最高": 11.0,
                    "最低": 9.5,
                    "收盘": 10.5,
                    "成交量": 1234,
                    "成交额": 12_950_000.0,
                }
            ]
        )

    fake_akshare = types.SimpleNamespace(
        __version__="1.18.84",
        stock_zh_a_hist=stock_zh_a_hist,
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)
    provider = AkShareLiveProvider(
        allow_network=True,
        adjust="",
        source_order=("east_money",),
        trading_calendar=_calendar(),
        calendar_source="test_calendar",
    )

    result = provider.daily_ohlcv(
        ProviderRequest("2024-01-02", "2024-01-02", symbols=("600000.SH",))
    )

    assert result.point_in_time is True
    assert result.metadata["akshare_version"] == "1.18.84"
    assert result.metadata["canonical_volume_unit"] == "shares"
    assert result.metadata["raw_volume_unit_by_source"]["east_money"] == "lots_100_shares"
    assert result.metadata["failed_symbols"] == []
    assert result.frame["volume"].tolist() == [123_400.0]


def test_provider_partial_symbol_failure_is_not_pit_certified(monkeypatch) -> None:
    def stock_zh_a_hist(symbol: str, **kwargs):
        if symbol == "000001":
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "日期": "2024-01-02",
                    "开盘": 10.0,
                    "最高": 11.0,
                    "最低": 9.5,
                    "收盘": 10.5,
                    "成交量": 1234,
                    "成交额": 12_950_000.0,
                }
            ]
        )

    fake_akshare = types.SimpleNamespace(
        __version__="1.18.84",
        stock_zh_a_hist=stock_zh_a_hist,
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)
    provider = AkShareLiveProvider(
        allow_network=True,
        source_order=("east_money",),
        trading_calendar=_calendar(),
    )

    result = provider.daily_ohlcv(
        ProviderRequest(
            "2024-01-02",
            "2024-01-02",
            symbols=("600000.SH", "000001.SZ"),
        )
    )

    assert result.point_in_time is False
    assert result.metadata["failed_symbols"] == ["000001.SZ"]
    assert any("incomplete_symbol_coverage" in warning for warning in result.warnings)
