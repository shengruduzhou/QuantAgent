"""Tests for intraday 做T features from 1-minute bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.execution.intraday_features import compute_intraday_features, features_frame


def _bars(closes, opens=None, vols=None, date="2026-05-08", sym="300308.SZ"):
    n = len(closes)
    opens = opens if opens is not None else [c - 0.1 for c in closes]
    vols = vols if vols is not None else [1000] * n
    highs = [max(o, c) + 0.05 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 0.05 for o, c in zip(opens, closes)]
    amts = [c * v for c, v in zip(closes, vols)]
    return pd.DataFrame({
        "symbol": sym, "trade_date": date,
        "trade_time": [f"{date} {9 + i // 60:02d}:{i % 60:02d}:00" for i in range(n)],
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": vols, "amount": amts,
    })


def test_basic_features_and_range_pos():
    # rising day: closes 10 -> 11, ends near the high
    closes = list(np.linspace(10.0, 11.0, 30))
    f = compute_intraday_features(_bars(closes), prev_close=9.5)  # first open 9.9 vs prev 9.5 → +gap
    assert f is not None
    assert f.day_low < f.vwap < f.day_high
    assert 0.8 <= f.intraday_range_pos <= 1.0      # closed near the high
    assert f.net_buy_pressure > 0                  # every minute close>open
    assert f.intraday_return > 0
    assert f.open_auction_gap > 0.0                # opened above prev close
    assert f.buy_below <= f.sell_above


def test_high_position_sell_bias():
    closes = list(np.linspace(10.0, 11.0, 20))
    # force last few minutes as down-minutes (active selling) at a high range pos
    opens = [c + 0.2 for c in closes]              # close < open => sell minutes
    f = compute_intraday_features(_bars(closes, opens=opens), prev_close=10.0)
    assert f.net_buy_pressure < 0
    # high range pos + selling => 偏空做T
    if f.intraday_range_pos >= 0.7:
        assert f.dot_bias == "偏空做T"


def test_spike_minutes_detected():
    vols = [1000] * 20
    vols[5] = 9000  # a volume spike minute
    vols[12] = 8000
    f = compute_intraday_features(_bars(list(np.linspace(10, 10.5, 20)), vols=vols))
    assert f.spike_minutes >= 2


def test_features_frame_columns_match_guard_inputs():
    bars = {"300308.SZ": _bars(list(np.linspace(10, 11, 15)))}
    df = features_frame(bars, prev_close={"300308.SZ": 9.8})
    for col in ("net_buy_pressure", "vwap_deviation", "intraday_range_pos", "spike_minutes"):
        assert col in df.columns
    assert len(df) == 1


def test_empty_bars_return_none():
    assert compute_intraday_features(pd.DataFrame()) is None
    assert compute_intraday_features(None) is None
