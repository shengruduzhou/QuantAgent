"""BaoStockProvider unit tests with a stubbed baostock module."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.providers.baostock_provider import (
    BaoStockConfig,
    BaoStockProvider,
    from_baostock_symbol,
    to_baostock_symbol,
)


# ---------------------------------------------------------------------------
# Stub baostock module — emulates the real shape of the API just enough.
# ---------------------------------------------------------------------------

class _FakeRS:
    def __init__(self, fields: list[str], rows: list[list[str]]):
        self.fields = fields
        self._rows = rows
        self._cursor = -1
        self.error_code = "0"
        self.error_msg = ""

    def next(self) -> bool:
        self._cursor += 1
        return self._cursor < len(self._rows)

    def get_row_data(self) -> list[str]:
        return self._rows[self._cursor]


class _FakeLoginResult:
    error_code = "0"
    error_msg = ""


class _FakeBaoStock:
    def __init__(self, daily_rows: dict[str, list[list[str]]] | None = None,
                  minute_rows: dict[str, list[list[str]]] | None = None,
                  login_ok: bool = True):
        self.daily_rows = daily_rows or {}
        self.minute_rows = minute_rows or {}
        self.login_ok = login_ok
        self.login_calls = 0
        self.logout_calls = 0
        self.queries: list[tuple] = []

    def login(self):
        self.login_calls += 1
        rs = _FakeLoginResult()
        if not self.login_ok:
            rs.error_code = "10001"
            rs.error_msg = "login_blocked"
        return rs

    def logout(self):
        self.logout_calls += 1

    def query_history_k_data_plus(
        self, code, fields, *,
        start_date, end_date, frequency, adjustflag,
    ):
        self.queries.append((code, fields, start_date, end_date, frequency, adjustflag))
        if frequency == "d":
            rows = self.daily_rows.get(code, [])
            field_list = (
                "date,code,open,high,low,close,preclose,volume,amount,adjustflag,"
                "turn,tradestatus,pctChg,isST"
            ).split(",")
        else:
            rows = self.minute_rows.get(code, [])
            field_list = (
                "date,time,code,open,high,low,close,volume,amount,adjustflag"
            ).split(",")
        return _FakeRS(field_list, rows)


# ---------------------------------------------------------------------------
# Symbol normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inp,out", [
    ("600519.SH", "sh.600519"),
    ("000001.SZ", "sz.000001"),
    ("sh.600519", "sh.600519"),
    ("600519", "sh.600519"),
    ("000001", "sz.000001"),
])
def test_to_baostock_symbol_round_trip(inp, out):
    assert to_baostock_symbol(inp) == out


def test_from_baostock_symbol_canonicalises():
    assert from_baostock_symbol("sh.600519") == "600519.SH"
    assert from_baostock_symbol("sz.000001") == "000001.SZ"


def test_to_baostock_symbol_rejects_unknown_format():
    with pytest.raises(ValueError):
        to_baostock_symbol("XYZ.NYSE")


# ---------------------------------------------------------------------------
# Daily K-line
# ---------------------------------------------------------------------------

def _row(date: str, *, close: float, ist: str = "0", status: str = "1") -> list[str]:
    return [
        date, "sh.600519",
        str(close - 0.5), str(close + 1.0), str(close - 1.0),
        str(close), str(close - 0.5),
        "1000000", "100000000", "1",
        "0.50", status, "1.00", ist,
    ]


def test_daily_ohlcv_returns_v7_canonical_schema():
    rows = [_row("2024-03-01", close=100.0), _row("2024-03-04", close=101.0)]
    fake = _FakeBaoStock(daily_rows={"sh.600519": rows})
    provider = BaoStockProvider(_bs_module=fake)
    res = provider.daily_ohlcv(ProviderRequest(
        start_date="2024-03-01", end_date="2024-03-05",
        symbols=("600519.SH",),
    ))
    assert not res.frame.empty
    for col in ("symbol", "trade_date", "open", "high", "low", "close",
                "volume", "amount", "available_at",
                "source", "source_reliability", "point_in_time_valid"):
        assert col in res.frame.columns
    assert (res.frame["symbol"] == "600519.SH").all()
    # available_at must come from the next bar (PIT)
    assert res.frame["available_at"].iloc[0] > res.frame["trade_date"].iloc[0]


def test_daily_ohlcv_handles_empty_response():
    fake = _FakeBaoStock(daily_rows={"sh.600519": []})
    provider = BaoStockProvider(_bs_module=fake)
    res = provider.daily_ohlcv(ProviderRequest(
        start_date="2024-03-01", end_date="2024-03-05",
        symbols=("600519.SH",),
    ))
    assert res.frame.empty
    assert res.quality_score == 0.0
    assert any("empty" in w for w in res.warnings)


def test_daily_ohlcv_requires_symbols():
    fake = _FakeBaoStock()
    provider = BaoStockProvider(_bs_module=fake)
    with pytest.raises(ProviderUnavailable):
        provider.daily_ohlcv(ProviderRequest(
            start_date="2024-03-01", end_date="2024-03-05",
        ))


def test_login_failure_raises_provider_unavailable():
    fake = _FakeBaoStock(login_ok=False)
    provider = BaoStockProvider(_bs_module=fake)
    with pytest.raises(ProviderUnavailable):
        provider.daily_ohlcv(ProviderRequest(
            start_date="2024-03-01", end_date="2024-03-05",
            symbols=("600519.SH",),
        ))


# ---------------------------------------------------------------------------
# Minute K-line
# ---------------------------------------------------------------------------

def test_minute_ohlcv_rejects_one_minute():
    fake = _FakeBaoStock()
    provider = BaoStockProvider(_bs_module=fake)
    with pytest.raises(ProviderUnavailable):
        provider.minute_ohlcv(
            ProviderRequest(start_date="2024-03-01", end_date="2024-03-05",
                            symbols=("600519.SH",)),
            frequency="1",
        )


def test_minute_ohlcv_5min_normalises_timestamp():
    rows = [[
        "2024-03-01", "20240301093500000", "sh.600519",
        "100.0", "100.5", "99.5", "100.2", "5000", "500000.0", "1",
    ]]
    fake = _FakeBaoStock(minute_rows={"sh.600519": rows})
    provider = BaoStockProvider(_bs_module=fake)
    res = provider.minute_ohlcv(
        ProviderRequest(start_date="2024-03-01", end_date="2024-03-05",
                        symbols=("600519.SH",)),
        frequency="5",
    )
    assert "timestamp" in res.frame.columns
    assert res.frame["timestamp"].iloc[0] == pd.Timestamp("2024-03-01 09:35:00")


# ---------------------------------------------------------------------------
# Tradability
# ---------------------------------------------------------------------------

def test_tradability_extracts_st_and_suspension_flags():
    rows = [
        _row("2024-03-01", close=100.0, ist="1", status="1"),
        _row("2024-03-04", close=99.0, ist="0", status="0"),
    ]
    fake = _FakeBaoStock(daily_rows={"sh.600519": rows})
    provider = BaoStockProvider(_bs_module=fake)
    res = provider.tradability(ProviderRequest(
        start_date="2024-03-01", end_date="2024-03-05",
        symbols=("600519.SH",),
    ))
    assert res.frame["is_st"].iloc[0]
    assert res.frame["is_suspended"].iloc[1]
