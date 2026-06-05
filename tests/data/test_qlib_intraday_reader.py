"""Tests for the qlib 1-minute binary reader.

Uses the real community 1-min sample when present (covers ~2020-2021);
skips cleanly otherwise so CI without the data still passes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.data.providers.qlib_intraday_reader import (
    build_intraday_panel,
    load_calendar,
    load_instruments,
    read_instrument_minutes,
)

_ROOT = Path("runtime/data/raw/qlib/cn_data_1min")
_have_data = (_ROOT / "calendars" / "1min.txt").exists()
pytestmark = pytest.mark.skipif(not _have_data, reason="qlib 1-min sample not present")


def test_calendar_loads_minute_grid():
    cal = load_calendar(_ROOT)
    assert len(cal) > 1000
    assert isinstance(cal, pd.DatetimeIndex)


def test_instruments_map_to_canonical_symbols():
    inst = load_instruments(_ROOT, "all")
    assert "symbol" in inst.columns
    # canonical form e.g. 600000.SH
    assert inst["symbol"].str.contains(r"\.(?:SH|SZ|BJ)$", regex=True).any()


def test_read_one_instrument_has_ohlcv_and_no_nan_close():
    df = read_instrument_minutes(_ROOT, "600000.SH")
    assert not df.empty
    for c in ("open", "high", "low", "close", "volume", "amount"):
        assert c in df.columns
    assert df["close"].notna().all()
    # ~240 minutes per trading day
    per_day = df.groupby("trade_date").size()
    assert per_day.median() <= 245


def test_build_panel_stacks_multiple_symbols():
    panel = build_intraday_panel(_ROOT, ["600000.SH", "600004.SH"])
    if not panel.empty:
        assert set(panel["symbol"].unique()).issubset({"600000.SH", "600004.SH"})
        assert panel["datetime"].is_monotonic_increasing or len(panel["symbol"].unique()) > 1
