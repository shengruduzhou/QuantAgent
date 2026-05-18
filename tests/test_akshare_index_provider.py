"""Unit tests for AkShareIndexProvider normalisers."""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.providers.akshare_index_provider import (
    AkShareIndexProvider,
    INDEX_AVAILABLE_AT_LAG_DAYS,
    _normalize_ohlcv,
)
from quantagent.data.providers.base import ProviderUnavailable


def test_ohlcv_normalises_chinese_index_columns():
    raw = pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "开盘": [3450.0, 3470.0, 3460.0],
        "最高": [3480.0, 3490.0, 3475.0],
        "最低": [3440.0, 3445.0, 3450.0],
        "收盘": [3470.0, 3460.0, 3465.0],
        "成交量": [120_000_000, 110_000_000, 105_000_000],
        "成交额": [3.3e11, 3.0e11, 2.8e11],
    })
    out = _normalize_ohlcv(raw, symbol="000300", label="csi300", kind="index")
    assert set(out.columns) >= {
        "observation_date", "symbol", "label", "kind",
        "open", "high", "low", "close", "volume", "amount",
        "available_at", "source",
    }
    assert (out["symbol"] == "000300").all()
    assert out["close"].tolist() == [3470.0, 3460.0, 3465.0]
    assert out["available_at"].iloc[0] == pd.Timestamp("2024-01-02") + pd.Timedelta(days=INDEX_AVAILABLE_AT_LAG_DAYS)


def test_ohlcv_respects_start_end_filters():
    raw = pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "收盘": [3.0, 3.1, 3.2],
    })
    filtered = _normalize_ohlcv(raw, symbol="CU0", label="copper", kind="commodity",
                                start_date="2024-01-03", end_date="2024-01-04")
    assert filtered["observation_date"].min() >= pd.Timestamp("2024-01-03")
    assert len(filtered) == 2


def test_ohlcv_requires_date_and_close():
    raw = pd.DataFrame({"open": [1.0], "high": [1.1], "low": [0.9]})
    assert _normalize_ohlcv(raw, symbol="X", label="x", kind="index").empty


def test_provider_requires_network(tmp_path):
    provider = AkShareIndexProvider(allow_network=False, root=str(tmp_path))
    with pytest.raises(ProviderUnavailable):
        provider.fetch_all()
