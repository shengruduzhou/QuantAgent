"""Regression lock: volume + amount survive TickFlow → silver → gold.

History (why this lock exists): two P0 bugs previously corrupted this path —
`adjusted_prices` silently returned UNadjusted prices (fixed via
klines.get(adjust="forward")), and qfq close was mixed with raw volume/amount
at different scales (Stage A audit). The 2026-07-12 TickFlow-integration audit
verified the current chain is clean end-to-end: provider canonical schema
(CANONICAL_OHLCV_COLUMNS) retains both fields; the silver market panel carries
them 100% non-null on active rows with shares/CNY units (implied VWAP/close
median ~1.0); the gold training dataset exposes both to factor code. These
tests fail loudly if any refactor drops, zeroes, or unit-shifts either field.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.data.providers.tickflow_provider import (
    CANONICAL_OHLCV_COLUMNS,
    _filter_window,
    _is_permission_error,
    _normalise_daily_frame,
)

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "runtime/data/v7/silver/market_panel/market_panel.parquet"
GOLD = REPO / "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean_fund.parquet"


def _tickflow_daily_fixture() -> pd.DataFrame:
    """Shape mirrors the documented TickFlow daily kline DataFrame: OHLCV core
    plus vendor extras (name/timestamp/trade_time) that normalisation drops."""
    return pd.DataFrame({
        "symbol": ["600519.SH", "600519.SH", "000001.SZ"],
        "name": ["贵州茅台", "贵州茅台", "平安银行"],
        "timestamp": [1751500800000, 1751587200000, 1751500800000],
        "trade_date": ["2026-07-03", "2026-07-06", "2026-07-03"],
        "trade_time": ["2026-07-03 15:00:00"] * 3,
        "open": [1450.0, 1462.5, 10.20],
        "high": [1466.0, 1470.0, 10.35],
        "low": [1441.0, 1455.1, 10.11],
        "close": [1460.2, 1458.8, 10.30],
        "volume": [2_534_100, 1_988_700, 98_231_400],
        "amount": [3_690_215_400.0, 2_905_442_100.0, 1_006_871_220.0],
    })


def test_normalise_preserves_volume_and_amount_exactly():
    raw = _tickflow_daily_fixture()
    out = _normalise_daily_frame(raw, source="tickflow", source_reliability=0.95)
    assert list(out.columns) == list(CANONICAL_OHLCV_COLUMNS)
    assert "volume" in out.columns and "amount" in out.columns
    m = out.merge(raw[["symbol", "trade_date", "volume", "amount"]].assign(
        trade_date=pd.to_datetime(raw["trade_date"])),
        on=["symbol", "trade_date"], suffixes=("", "_raw"))
    assert len(m) == len(raw)
    assert np.allclose(m["volume"], m["volume_raw"])
    assert np.allclose(m["amount"], m["amount_raw"])
    # PIT tag: bar becomes available the NEXT day
    assert (m["available_at"] == m["trade_date"] + pd.Timedelta(days=1)).all()


def test_normalise_missing_volume_stays_nan_not_zero():
    raw = _tickflow_daily_fixture().drop(columns=["amount"])
    out = _normalise_daily_frame(raw, source="tickflow", source_reliability=0.95)
    assert "amount" in out.columns
    assert out["amount"].isna().all(), "missing vendor field must stay NaN, never fabricated"
    assert out["volume"].notna().all()


def test_filter_window_bounds():
    raw = _tickflow_daily_fixture()
    out = _filter_window(raw, start_date="2026-07-04", end_date="2026-07-31")
    assert set(pd.to_datetime(out["trade_date"]).dt.date.astype(str)) == {"2026-07-06"}


def test_permission_error_classifier():
    assert _is_permission_error(RuntimeError("HTTP 403: forbidden"))
    assert _is_permission_error(RuntimeError("无日/周/月K线查询批量查询权限"))
    assert _is_permission_error(RuntimeError("无市场深度查询权限（市场: CN）"))
    assert not _is_permission_error(RuntimeError("connection timed out"))
    assert not _is_permission_error(RuntimeError("HTTP 500 internal error"))


@pytest.mark.skipif(not PANEL.exists(), reason="silver market panel not on disk")
def test_market_panel_volume_amount_present_and_unit_consistent():
    pan = pd.read_parquet(
        PANEL, columns=["volume", "amount", "close", "is_suspended"],
        filters=[("trade_date", ">=", pd.Timestamp("2026-06-01"))])
    act = pan[~pan["is_suspended"].fillna(False).astype(bool)]
    assert len(act) > 10_000
    assert float(act["volume"].isna().mean()) < 0.01
    assert float(act["amount"].isna().mean()) < 0.01
    assert float((act["volume"] <= 0).mean()) < 0.01
    assert float((act["amount"] <= 0).mean()) < 0.01
    # unit lock: amount/volume is a per-share VWAP ≈ close. A lots-vs-shares
    # regression (×100) or a currency/unit drift breaks this immediately.
    implied = (act["amount"] / act["volume"] / act["close"]).replace(
        [np.inf, -np.inf], np.nan).dropna()
    med = float(implied.median())
    assert 0.9 < med < 1.1, f"implied VWAP/close median {med} — volume/amount unit drift"


@pytest.mark.skipif(not GOLD.exists(), reason="gold training dataset not on disk")
def test_training_dataset_exposes_volume_and_amount():
    import pyarrow.parquet as pq
    names = set(pq.ParquetFile(GOLD).schema_arrow.names)
    assert "volume" in names and "amount" in names, \
        "volume/amount dropped from the training feature source"
