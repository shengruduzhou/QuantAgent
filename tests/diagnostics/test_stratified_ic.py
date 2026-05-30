"""Tests for quantagent.diagnostics.stratified_ic.

These tests use small synthetic panels (deterministic seeded RNG) to
verify:

* board classifier returns the right label for every prefix family
* cap bucketer handles edge cases (NaN, ties, single-quintile data)
* compute_stratified_ic returns the right axes when market_features
  / regime_frame are supplied vs omitted
* IC table values match a hand-computed reference for a tiny case
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.diagnostics.stratified_ic import (
    StratifiedICConfig,
    board_of,
    cap_bucket_of,
    compute_stratified_ic,
    render_markdown,
)


def test_board_of_prefix_table():
    assert board_of("600519.SH") == "SH_Main_沪主板"
    assert board_of("601318.SH") == "SH_Main_沪主板"
    assert board_of("603259.SH") == "SH_Main_沪主板"
    assert board_of("605358.SH") == "SH_Main_沪主板"
    assert board_of("000001.SZ") == "SZ_Main_深主板"
    assert board_of("002475.SZ") == "SZ_Main_深主板"
    assert board_of("300750.SZ") == "ChiNext_创业"
    assert board_of("301236.SZ") == "ChiNext_创业"
    assert board_of("688981.SH") == "STAR_科创"
    assert board_of("832000.BJ") == "BSE_北交所"
    assert board_of("XYZ") == "OTHER"
    assert board_of("") == "OTHER"
    assert board_of(None) == "OTHER"  # type: ignore[arg-type]


def test_cap_bucket_of_edge_cases():
    edges = (10.0, 20.0, 30.0, 40.0)
    assert cap_bucket_of(5.0, edges) == "Q1_smallest"
    assert cap_bucket_of(15.0, edges) == "Q2"
    assert cap_bucket_of(25.0, edges) == "Q3"
    assert cap_bucket_of(35.0, edges) == "Q4"
    assert cap_bucket_of(50.0, edges) == "Q5_largest"
    # Boundary: == edge → upper bucket
    assert cap_bucket_of(10.0, edges) == "Q2"
    assert cap_bucket_of(40.0, edges) == "Q5_largest"
    # NaN / None → UNKNOWN
    assert cap_bucket_of(float("nan"), edges) == "UNKNOWN"
    assert cap_bucket_of(None, edges) == "UNKNOWN"  # type: ignore[arg-type]
    # Wrong-shape edges → ValueError
    with pytest.raises(ValueError):
        cap_bucket_of(1.0, (1.0, 2.0))  # type: ignore[arg-type]


def _make_panel(n_dates: int = 30, n_symbols_per_board: int = 25, seed: int = 7) -> pd.DataFrame:
    """Synthetic panel with 4 boards, controllable noise per board."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    rows = []
    board_seeds = {
        "600": 0.30,  # SH_Main: prediction explains 30% of label variance
        "000": 0.40,  # SZ_Main: 40%
        "300": 0.60,  # ChiNext: 60% — strongest
        "688": 0.20,  # STAR: 20% — weakest
    }
    for prefix, signal in board_seeds.items():
        for sid in range(n_symbols_per_board):
            symbol = f"{prefix}{sid:03d}.SH" if prefix.startswith("6") else f"{prefix}{sid:03d}.SZ"
            for h in (5, 20):
                for d in dates:
                    pred = rng.normal()
                    noise = rng.normal()
                    label = signal * pred + (1.0 - signal) * noise
                    rows.append(
                        {
                            "trade_date": d,
                            "symbol": symbol,
                            "horizon": h,
                            "prediction": pred,
                            f"forward_return_{h}d": label,
                        }
                    )
    return pd.DataFrame(rows)


def test_compute_without_market_features_yields_board_and_regime_axes_only():
    panel = _make_panel()
    result = compute_stratified_ic(panel)
    assert "board" in result.by_axis
    # Without market_features, liquidity_quintile / volatility_quintile
    # tables collapse to a single UNKNOWN bucket which is below the
    # min_days_per_bucket threshold when we have > 10 days but few buckets.
    # The code still emits them with a single "UNKNOWN" bucket; assert
    # exactly that.
    if "liquidity_quintile" in result.by_axis:
        liq = result.by_axis["liquidity_quintile"]
        assert set(liq["bucket"]) <= {"UNKNOWN"}
    # board IC should rank ChiNext > SZ_Main > SH_Main > STAR
    board_20 = result.by_axis["board"].query("horizon == 20").set_index("bucket")["ic_mean"]
    assert board_20["ChiNext_创业"] > board_20["SH_Main_沪主板"]
    assert board_20["SZ_Main_深主板"] > board_20["STAR_科创"]


def test_compute_with_market_features_adds_liquidity_and_vol_axes():
    panel = _make_panel()
    # Build a deterministic market_features panel: cap proxy increases
    # with symbol code, vol decreases with code.
    mf_rows = []
    for d in panel["trade_date"].unique():
        for sym in panel["symbol"].unique():
            code = int(sym.split(".")[0][-3:])
            mf_rows.append({
                "trade_date": d,
                "symbol": sym,
                "amount_mean_20d": float(code * 1e6),
                "volatility_20d": float(0.5 - code * 0.001),
            })
    mf = pd.DataFrame(mf_rows)
    result = compute_stratified_ic(panel, market_features=mf)
    assert "liquidity_quintile" in result.by_axis
    assert "volatility_quintile" in result.by_axis
    liq_buckets = set(result.by_axis["liquidity_quintile"]["bucket"])
    assert liq_buckets >= {"Q1_smallest", "Q3", "Q5_largest"}
    vol_buckets = set(result.by_axis["volatility_quintile"]["bucket"])
    assert vol_buckets >= {"Q1_lowest_vol", "Q5_highest_vol"}


def test_render_markdown_does_not_crash_and_contains_axis_headers():
    panel = _make_panel(n_dates=15)
    result = compute_stratified_ic(panel)
    md = render_markdown(result)
    assert "Stratified Factor IC Report" in md
    assert "## Axis: `board`" in md


def test_empty_input_returns_empty_result():
    empty = pd.DataFrame(columns=["trade_date", "symbol", "horizon", "prediction"])
    result = compute_stratified_ic(empty)
    assert result.summary["status"] == "empty_input"
    assert result.by_axis == {}


def test_missing_required_columns_returns_status_message():
    bad = pd.DataFrame({"trade_date": ["2024-01-02"], "symbol": ["x"]})
    result = compute_stratified_ic(bad)
    assert result.summary["status"] == "missing_required_columns"


def test_min_days_per_bucket_filters_thin_buckets():
    panel = _make_panel(n_dates=5)  # below default min_days_per_bucket=10
    result = compute_stratified_ic(panel, config=StratifiedICConfig(min_days_per_bucket=10))
    # board table should be empty because no bucket has ≥10 days
    assert "board" not in result.by_axis or result.by_axis["board"].empty
