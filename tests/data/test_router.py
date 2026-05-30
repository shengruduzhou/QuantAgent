"""MultiSourceDataRouter tests (Qlib / AkShare / BaoStock / TuShare)."""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.providers.base import (
    ProviderRequest,
    ProviderResult,
    ProviderUnavailable,
)
from quantagent.data.router import (
    MultiSourceDataRouter,
    RouterAllSourcesUnavailable,
    RouterConfig,
    RoutedProvider,
    build_default_router,
)


# ---------------------------------------------------------------------------
# Stub providers
# ---------------------------------------------------------------------------

class _OkProvider:
    def __init__(self, name: str, n_rows: int = 3):
        self.name = name
        self.n_rows = n_rows
        self.calls = 0

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        self.calls += 1
        dates = pd.bdate_range(request.start_date, periods=self.n_rows)
        rows = []
        for sym in request.symbols or ("A.SH",):
            for d in dates:
                rows.append({"symbol": sym, "trade_date": d, "close": 100.0})
        return ProviderResult(pd.DataFrame(rows), source=self.name, quality_score=0.9)


class _PartialProvider:
    """Only serves a subset of symbols."""

    def __init__(self, name: str, served_symbols: set[str]):
        self.name = name
        self.served = served_symbols
        self.calls = 0

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        self.calls += 1
        served = [s for s in request.symbols if s in self.served]
        if not served:
            return ProviderResult(pd.DataFrame(), source=self.name, quality_score=0.0)
        dates = pd.bdate_range(request.start_date, periods=2)
        rows = [
            {"symbol": s, "trade_date": d, "close": 100.0}
            for s in served for d in dates
        ]
        return ProviderResult(pd.DataFrame(rows), source=self.name, quality_score=0.7)


class _UnavailableProvider:
    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        raise ProviderUnavailable("API quota exhausted")


class _ErrorProvider:
    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        raise RuntimeError("network error")


class _MinuteProvider:
    def __init__(self, name: str):
        self.name = name

    def minute_ohlcv(self, request: ProviderRequest, *, frequency: str) -> ProviderResult:
        return ProviderResult(
            pd.DataFrame([{"symbol": "A.SH", "trade_date": "2024-03-01",
                            "timestamp": pd.Timestamp("2024-03-01 10:00"),
                            "close": 100.0}]),
            source=self.name, quality_score=0.85,
            metadata={"frequency": frequency},
        )


# ---------------------------------------------------------------------------
# Daily routing
# ---------------------------------------------------------------------------

def test_first_priority_source_wins():
    router = MultiSourceDataRouter(RouterConfig(daily_priority=("qlib", "akshare")))
    qlib = _OkProvider("qlib")
    ak = _OkProvider("akshare")
    router.register(RoutedProvider(name="qlib", provider=qlib))
    router.register(RoutedProvider(name="akshare", provider=ak))
    res = router.daily_ohlcv(ProviderRequest(
        start_date="2024-03-01", end_date="2024-03-05",
        symbols=("A.SH",),
    ))
    assert res.primary_source == "qlib"
    assert ak.calls == 0  # akshare never consulted (qlib already covered)
    assert (res.frame["source_name"] == "qlib").all()


def test_router_falls_back_when_first_source_unavailable():
    router = MultiSourceDataRouter(RouterConfig(daily_priority=("qlib", "baostock")))
    router.register(RoutedProvider(name="qlib", provider=_UnavailableProvider()))
    router.register(RoutedProvider(name="baostock", provider=_OkProvider("baostock")))
    res = router.daily_ohlcv(ProviderRequest(
        start_date="2024-03-01", end_date="2024-03-05",
        symbols=("A.SH",),
    ))
    assert res.primary_source == "baostock"
    assert res.per_source["qlib"]["status"] == "unavailable"
    assert "qlib" in res.fallback_chain and "baostock" in res.fallback_chain


def test_router_marks_error_status_for_runtime_failures():
    router = MultiSourceDataRouter(RouterConfig(daily_priority=("qlib", "akshare")))
    router.register(RoutedProvider(name="qlib", provider=_ErrorProvider()))
    router.register(RoutedProvider(name="akshare", provider=_OkProvider("akshare")))
    res = router.daily_ohlcv(ProviderRequest(
        start_date="2024-03-01", end_date="2024-03-05",
        symbols=("A.SH",),
    ))
    assert res.per_source["qlib"]["status"] == "error"
    assert res.primary_source == "akshare"


def test_router_merges_partial_coverage_across_sources():
    router = MultiSourceDataRouter(RouterConfig(
        daily_priority=("qlib", "akshare"), merge_partial_results=True,
    ))
    router.register(RoutedProvider(name="qlib", provider=_PartialProvider("qlib", {"A.SH"})))
    router.register(RoutedProvider(name="akshare", provider=_PartialProvider("akshare", {"B.SH"})))
    res = router.daily_ohlcv(ProviderRequest(
        start_date="2024-03-01", end_date="2024-03-05",
        symbols=("A.SH", "B.SH"),
    ))
    syms = set(res.frame["symbol"].astype(str))
    assert syms == {"A.SH", "B.SH"}
    assert res.primary_source == "qlib"


def test_router_raises_when_all_sources_fail_in_production():
    router = MultiSourceDataRouter(RouterConfig(
        daily_priority=("qlib", "akshare"),
        allow_mock_fallback=False, fail_when_all_unavailable=True,
    ))
    router.register(RoutedProvider(name="qlib", provider=_UnavailableProvider()))
    router.register(RoutedProvider(name="akshare", provider=_ErrorProvider()))
    with pytest.raises(RouterAllSourcesUnavailable):
        router.daily_ohlcv(ProviderRequest(
            start_date="2024-03-01", end_date="2024-03-05",
            symbols=("A.SH",),
        ))


def test_router_allows_mock_fallback_only_when_explicitly_enabled():
    router = MultiSourceDataRouter(RouterConfig(
        daily_priority=("qlib", "akshare"),
        allow_mock_fallback=True,
    ))
    router.register(RoutedProvider(name="qlib", provider=_UnavailableProvider()))
    router.register(RoutedProvider(name="akshare", provider=_ErrorProvider()))
    res = router.daily_ohlcv(ProviderRequest(
        start_date="2024-03-01", end_date="2024-03-05",
        symbols=("A.SH",),
    ))
    # Returns empty but does not raise
    assert res.primary_source is None
    assert res.frame.empty
    assert "router_all_sources_empty" in res.warnings


def test_router_skips_unregistered_priorities():
    router = MultiSourceDataRouter(RouterConfig(daily_priority=("qlib", "akshare", "baostock")))
    router.register(RoutedProvider(name="baostock", provider=_OkProvider("baostock")))
    res = router.daily_ohlcv(ProviderRequest(
        start_date="2024-03-01", end_date="2024-03-05",
        symbols=("A.SH",),
    ))
    assert res.primary_source == "baostock"
    # qlib / akshare not registered → not in fallback_chain
    assert res.fallback_chain == ["baostock"]


# ---------------------------------------------------------------------------
# Minute routing
# ---------------------------------------------------------------------------

def test_minute_router_calls_correct_method():
    router = MultiSourceDataRouter(RouterConfig(minute_priority=("baostock",)))
    router.register(RoutedProvider(name="baostock", provider=_MinuteProvider("baostock")))
    res = router.minute_ohlcv(
        ProviderRequest(start_date="2024-03-01", end_date="2024-03-02",
                         symbols=("A.SH",)),
        frequency="15",
    )
    assert res.primary_source == "baostock"
    assert "frequency" in (res.per_source.get("baostock") or {}).get("warnings", []) or res.per_source["baostock"]["status"] == "ok"


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def test_build_default_router_skips_none_providers():
    router = build_default_router(
        qlib_provider=_OkProvider("qlib"),
        akshare_provider=None,
        baostock_provider=_OkProvider("baostock"),
        tushare_provider=None,
    )
    assert set(router.list_sources()) == {"qlib", "baostock"}


def test_build_default_router_returns_empty_with_all_none():
    router = build_default_router()
    assert router.list_sources() == []
