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


class Provider:
    def __init__(self, frame: pd.DataFrame, *, metadata: dict[str, object]) -> None:
        self.frame = frame
        self.metadata = metadata

    def daily_ohlcv(self, _request: ProviderRequest) -> ProviderResult:
        return ProviderResult(
            self.frame.copy(),
            source="real-provider",
            point_in_time=True,
            metadata=dict(self.metadata),
        )


def _meta(**overrides: object) -> dict[str, object]:
    out: dict[str, object] = {
        "frequency": "1d",
        "timezone": "Asia/Shanghai",
        "volume_unit": "shares",
        "amount_unit": "CNY",
        "adjustment": "raw",
        "pit_semantics": "trade_date_observed_no_future_adjustment",
    }
    out.update(overrides)
    return out


def _bar(symbol: str, date: str, close: float) -> dict[str, object]:
    return {
        "symbol": symbol,
        "trade_date": date,
        "open": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": 1000.0,
        "amount": close * 1000.0,
    }


def _router(primary: Provider, fallback: Provider) -> MultiSourceDataRouter:
    router = MultiSourceDataRouter(
        RouterConfig(daily_priority=("primary", "fallback"), merge_partial_results=True)
    )
    router.register(RoutedProvider("primary", primary))
    router.register(RoutedProvider("fallback", fallback))
    return router


def test_pit_semantics_must_match_across_production_fallback_sources() -> None:
    request = ProviderRequest(
        "2026-01-05",
        "2026-01-06",
        symbols=("600000.SH",),
    )
    primary = Provider(
        pd.DataFrame([_bar("600000.SH", "2026-01-05", 10.0)]),
        metadata=_meta(),
    )
    fallback = Provider(
        pd.DataFrame([_bar("600000.SH", "2026-01-06", 11.0)]),
        metadata=_meta(pit_semantics="different_unreviewed_pit_contract"),
    )
    policy = DailyOHLCVIntegrityPolicy.production(
        expected_trade_dates=("2026-01-05", "2026-01-06")
    )
    with pytest.raises(RouterDataIntegrityError, match="coverage incomplete"):
        _router(primary, fallback).daily_ohlcv(request, integrity_policy=policy)


def test_explicit_per_symbol_expected_keys_do_not_invent_suspended_bar() -> None:
    request = ProviderRequest(
        "2026-01-05",
        "2026-01-06",
        symbols=("600000.SH", "000001.SZ"),
    )
    primary = Provider(
        pd.DataFrame(
            [
                _bar("600000.SH", "2026-01-05", 10.0),
                _bar("600000.SH", "2026-01-06", 10.5),
                _bar("000001.SZ", "2026-01-05", 12.0),
            ]
        ),
        metadata=_meta(),
    )
    fallback = Provider(pd.DataFrame(), metadata=_meta())
    policy = DailyOHLCVIntegrityPolicy.production(
        expected_symbol_trade_dates=(
            ("600000.SH", "2026-01-05"),
            ("600000.SH", "2026-01-06"),
            ("000001.SZ", "2026-01-05"),
        )
    )
    result = _router(primary, fallback).daily_ohlcv(request, integrity_policy=policy)
    assert len(result.frame) == 3
    assert result.integrity["coverage_basis"] == "authoritative_per_symbol_expected_keys"
    assert result.integrity["missing_expected_keys"] == []
    assert not (
        (result.frame["symbol"] == "000001.SZ")
        & (pd.to_datetime(result.frame["trade_date"]).dt.date.astype(str) == "2026-01-06")
    ).any()
