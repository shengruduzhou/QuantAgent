"""Tests for intraday (分时) volume-price factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.factors.intraday_volume_price import (
    FACTOR_COLUMNS,
    compute_intraday_factors,
)


def _one_day(symbol: str, day: str, closes: list[float], vols: list[float]) -> pd.DataFrame:
    n = len(closes)
    dt = pd.date_range(f"{day} 09:31", periods=n, freq="1min")
    frame = pd.DataFrame({
        "symbol": symbol, "datetime": dt, "trade_date": pd.Timestamp(day),
        "open": [closes[0]] + closes[:-1], "high": closes, "low": closes,
        "close": closes, "volume": vols,
    })
    frame["amount"] = frame["close"] * frame["volume"]
    return frame


def test_factor_columns_present_and_one_row_per_symbol_day():
    panel = pd.concat([
        _one_day("A.SH", "2021-01-04", [10, 11, 12, 11, 13], [100, 200, 300, 150, 500]),
        _one_day("A.SH", "2021-01-05", [13, 12, 11, 12, 14], [100, 100, 100, 100, 100]),
        _one_day("B.SZ", "2021-01-04", [20, 20, 20, 20, 20], [50, 50, 50, 50, 50]),
    ], ignore_index=True)
    f = compute_intraday_factors(panel)
    assert len(f) == 3
    for c in FACTOR_COLUMNS:
        assert c in f.columns


def test_vwap_deviation_sign_matches_close_vs_vwap():
    # rising prices on rising volume → close finishes well above VWAP
    panel = _one_day("A.SH", "2021-01-04", [10, 11, 12, 13, 14], [100, 200, 300, 400, 500])
    f = compute_intraday_factors(panel).iloc[0]
    assert f["vwap_deviation"] > 0
    # first-30 and last-30 returns positive in a monotonic up day
    assert f["first30_return"] > 0
    assert f["last30_return"] > 0
    # close at the high → range position ≈ 1
    assert f["intraday_range_pos"] > 0.99


def test_net_buy_pressure_positive_when_up_minutes_carry_volume():
    # up minutes (1→2, 3→4) carry the volume; down minutes are quiet → net > 0
    panel = _one_day("A.SH", "2021-01-04", [10, 11, 10.5, 11.5, 12], [10, 400, 10, 400, 400])
    f = compute_intraday_factors(panel).iloc[0]
    assert -1.0 <= f["net_buy_pressure"] <= 1.0
    assert f["net_buy_pressure"] > 0


def test_spike_minutes_counts_volume_outliers():
    # one giant print among quiet minutes → at least one spike minute
    panel = _one_day("A.SH", "2021-01-04",
                     [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                     [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10_000])
    f = compute_intraday_factors(panel).iloc[0]
    assert f["spike_minutes"] >= 1


def test_cicc_intraday_liquidity_and_price_volume_corr_present():
    panel = _one_day(
        "A.SH", "2021-01-04",
        [10, 10.1, 10.3, 10.6, 11.0, 11.5],
        [100, 120, 200, 300, 500, 800],
    )
    f = compute_intraday_factors(panel).iloc[0]
    assert f["liq_amihud_1min"] >= 0
    assert f["corr_prv"] > 0
    assert 0 < f["open30_volume_share"] <= 1
    assert 0 < f["close30_volume_share"] <= 1
    assert 0 < f["close3_volume_share"] <= 1


def test_cicc_intraday_rolling_mean_factors_after_min_periods():
    frames = []
    for i, day in enumerate(pd.bdate_range("2021-01-04", periods=6)):
        frames.append(
            _one_day(
                "A.SH", str(day.date()),
                [10 + i, 10.1 + i, 10.3 + i, 10.6 + i, 11.0 + i, 11.5 + i],
                [100, 120, 200, 300, 500, 800],
            )
        )
    f = compute_intraday_factors(pd.concat(frames, ignore_index=True))
    assert f["liq_amihud_1min_m20"].iloc[:4].isna().all()
    assert pd.notna(f["liq_amihud_1min_m20"].iloc[-1])
    assert pd.notna(f["corr_prv_m20"].iloc[-1])


def test_empty_panel_returns_empty_well_shaped():
    f = compute_intraday_factors(pd.DataFrame())
    assert list(f.columns) == ["symbol", "trade_date", *FACTOR_COLUMNS]
    assert f.empty
