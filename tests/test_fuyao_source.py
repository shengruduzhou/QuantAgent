from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantagent.data.ashare import contracts
from quantagent.data.ashare.fuyao import (
    FUYAO_API_KEY_ENV,
    MARKET_DUMP_ENDPOINTS,
    PRICE_HISTORICAL,
    FuyaoClient,
    FuyaoSource,
)
from quantagent.data.ashare.fuyao_dump import FuyaoDumpSource, validate_dump
from quantagent.data.ashare.http import FetchOutcome, RETRY_ENTITLEMENT, RETRY_OK


def _ms(date: str) -> int:
    return int(pd.Timestamp(date, tz="Asia/Shanghai").timestamp() * 1000)


class FakeHttp:
    def __init__(self, payloads: list[dict]):
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def get_json(self, url, params=None, headers=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        payload = self.payloads.pop(0)
        return FetchOutcome(
            ok=True,
            endpoint=url,
            retry_class=RETRY_OK,
            status_code=200,
            payload=payload,
            retrieved_at="2026-08-07T00:00:00+00:00",
        )


def test_missing_key_fails_closed(monkeypatch):
    monkeypatch.delenv(FUYAO_API_KEY_ENV, raising=False)
    client = FuyaoClient(api_key="")

    outcome = client.get(PRICE_HISTORICAL, {"thscode": "600519.SH"})

    assert not outcome.ok
    assert outcome.retry_class == RETRY_ENTITLEMENT
    assert FUYAO_API_KEY_ENV in (outcome.error or "")


def test_business_auth_code_is_not_mistaken_for_http_success():
    fake = FakeHttp([{"code": 2003, "message": "subscription unavailable", "data": None}])
    client = FuyaoClient(api_key="secret", client=fake)

    outcome = client.get(PRICE_HISTORICAL, {"thscode": "600519.SH"})

    assert not outcome.ok
    assert outcome.retry_class == RETRY_ENTITLEMENT
    assert fake.calls[0]["headers"]["X-api-key"] == "secret"


def test_daily_bars_are_canonical_raw_and_deduplicated():
    item = {
        "date_ms": _ms("2026-08-01"),
        "open_price": 10.0,
        "high_price": 12.0,
        "low_price": 9.5,
        "close_price": 11.0,
        "volume": 123400.0,
        "turnover": 1357900.0,
    }
    fake = FakeHttp([{"code": 0, "message": "ok", "data": {"item": [item, item]}}])
    source = FuyaoSource(api_key="secret", client=fake)

    result = source.daily_bars("600519.SH", "2026-08-01", "2026-08-02")

    assert result.ok
    assert result.rows == 1
    assert list(result.frame.columns) == list(contracts.DAILY_BARS.columns)
    row = result.frame.iloc[0]
    assert row["symbol"] == "600519.SH"
    assert row["trade_date"] == pd.Timestamp("2026-08-01")
    assert row["volume"] == 123400.0
    assert row["amount"] == 1357900.0
    assert row["source"] == "fuyao"
    assert row["available_at"] == "2026-08-01 15:00:00"
    assert fake.calls[0]["params"]["adjust"] == "none"


def test_long_history_is_chunked_below_vendor_maximum():
    first = {
        "date_ms": _ms("2000-01-03"), "open_price": 1, "high_price": 1,
        "low_price": 1, "close_price": 1, "volume": 1, "turnover": 1,
    }
    second = {
        "date_ms": _ms("2012-01-03"), "open_price": 2, "high_price": 2,
        "low_price": 2, "close_price": 2, "volume": 2, "turnover": 2,
    }
    third = {
        "date_ms": _ms("2020-01-03"), "open_price": 3, "high_price": 3,
        "low_price": 3, "close_price": 3, "volume": 3, "turnover": 3,
    }
    fake = FakeHttp([
        {"code": 0, "data": {"item": [first]}},
        {"code": 0, "data": {"item": [second]}},
        {"code": 0, "data": {"item": [third]}},
    ])
    source = FuyaoSource(api_key="secret", client=fake)

    result = source.daily_bars("600519.SH", "2000-01-01", "2020-12-31")

    assert result.ok
    assert len(fake.calls) == 3
    starts = [call["params"]["start"] for call in fake.calls]
    ends = [call["params"]["end"] for call in fake.calls]
    assert all(end >= start for start, end in zip(starts, ends))


def test_market_dump_paths_match_official_reference():
    assert MARKET_DUMP_ENDPOINTS == {
        "daily-k": "/api/dump/market-dumps/daily-k/download-url",
        "daily-k-10d": "/api/dump/market-dumps/daily-k-10d/download-url",
        "adjustment-factors": "/api/dump/market-dumps/adjustment-factors/download-url",
    }


def test_local_dump_source_preserves_u0_contract(tmp_path: Path):
    path = tmp_path / "daily-k.parquet"
    pd.DataFrame(
        {
            "thscode": ["600519.SH", "000001.SZ"],
            "currency": ["CNY", "CNY"],
            "interval": ["1d", "1d"],
            "adjusted": ["none", "none"],
            "date_ms": [_ms("2026-08-01"), _ms("2026-08-01")],
            "open_price": [10.0, 20.0],
            "high_price": [11.0, 21.0],
            "low_price": [9.0, 19.0],
            "close_price": [10.5, 20.5],
            "volume": [1000.0, 2000.0],
            "turnover": [10500.0, 41000.0],
        }
    ).to_parquet(path, index=False)

    artifact = validate_dump(path, "daily-k")
    result = FuyaoDumpSource(path).daily_bars("600519.SH", "2026-07-01", "2026-08-31")

    assert artifact.rows == 2
    assert result.ok
    assert result.rows == 1
    assert result.frame.iloc[0]["source"] == "fuyao_dump"
    assert result.frame.iloc[0]["amount"] == 10500.0


def test_local_dump_rejects_adjusted_prices(tmp_path: Path):
    path = tmp_path / "daily-k.parquet"
    pd.DataFrame(
        {
            "thscode": ["600519.SH"],
            "currency": ["CNY"],
            "interval": ["1d"],
            "adjusted": ["forward"],
            "date_ms": [_ms("2026-08-01")],
            "open_price": [10.0], "high_price": [11.0], "low_price": [9.0],
            "close_price": [10.5], "volume": [1000.0], "turnover": [10500.0],
        }
    ).to_parquet(path, index=False)

    result = FuyaoDumpSource(path).daily_bars("600519.SH", "2026-07-01", "2026-08-31")

    assert not result.ok
    assert "non-raw adjusted" in (result.error or "")
