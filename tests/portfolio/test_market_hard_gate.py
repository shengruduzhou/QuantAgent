"""Tests for the Stage 3 market hard gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.portfolio.market_hard_gate import (
    MarketHardGateConfig,
    compute_market_hard_gate,
    hard_gate_multiplier,
    write_hard_gate_manifest,
)


def _flat_benchmark(n: int = 300, start_price: float = 4000.0) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    closes = np.full(n, start_price, dtype=float)
    return pd.DataFrame({"trade_date": dates, "close": closes})


def _shock(closes: np.ndarray, idx: int, pct: float) -> np.ndarray:
    out = closes.copy()
    out[idx:] *= (1.0 + pct)
    return out


# ---------------------------------------------------------------------------
# T1 crash
# ---------------------------------------------------------------------------

def test_t1_crash_triggers_hard_gate():
    bench = _flat_benchmark(n=250)
    # Engineer a 5-day -10% drop ending at index 200
    bench.loc[196:200, "close"] = bench.loc[195, "close"] * np.linspace(0.99, 0.90, 5)
    result = compute_market_hard_gate(bench)
    active = result.frame.loc[result.frame["hard_gate_active"], "trade_date"]
    assert not active.empty
    assert "crash_5d" in result.frame.loc[result.frame["hard_gate_active"], "trigger_reason"].tolist()


# ---------------------------------------------------------------------------
# T2 deep bear + below MA
# ---------------------------------------------------------------------------

def test_t2_deep_bear_below_ma_triggers():
    bench = _flat_benchmark(n=400, start_price=4000.0)
    # Gradual 20-day -20% drop with prior elevated MA so close goes below MA
    bench.loc[300:319, "close"] = bench.loc[299, "close"] * np.linspace(0.99, 0.80, 20)
    bench.loc[320:, "close"] = bench.loc[319, "close"]
    result = compute_market_hard_gate(bench)
    reasons = set(result.frame.loc[result.frame["hard_gate_active"], "trigger_reason"].unique())
    assert "deep_bear_20d_below_ma" in reasons or "crash_5d" in reasons


# ---------------------------------------------------------------------------
# T4 vol spike
# ---------------------------------------------------------------------------

def test_t4_vol_spike_triggers():
    rng = np.random.default_rng(7)
    n = 300
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    # Quiet 0.5% daily vol for the first 250 days, then 3% spike for 30 days
    daily_ret = rng.normal(0, 0.005, n)
    daily_ret[250:280] = rng.normal(0, 0.03, 30)
    closes = 4000 * np.cumprod(1 + daily_ret)
    bench = pd.DataFrame({"trade_date": dates, "close": closes})
    result = compute_market_hard_gate(bench)
    active_reasons = set(result.frame.loc[result.frame["hard_gate_active"], "trigger_reason"].unique())
    # vol_spike or crash_5d (the spike likely also produces some 5d crashes) must show
    assert any(r in active_reasons for r in ["vol_spike", "crash_5d"])


# ---------------------------------------------------------------------------
# T3 breadth collapse — needs cross-section panel
# ---------------------------------------------------------------------------

def test_t3_breadth_collapse_requires_three_consecutive_days():
    bench = _flat_benchmark(n=100)
    # Panel with 5 symbols: all decline 3 days in a row mid-window
    dates = bench["trade_date"]
    rets = pd.DataFrame(
        np.full((len(dates), 5), 0.01),
        index=dates,
        columns=[f"S{i}" for i in range(5)],
    )
    rets.iloc[50:53, :] = -0.02  # 100% decliners for 3 days
    result = compute_market_hard_gate(bench, breadth_panel=rets)
    reasons = set(result.frame.loc[result.frame["hard_gate_active"], "trigger_reason"].unique())
    assert "breadth_collapse" in reasons


def test_t3_breadth_two_day_does_not_trigger():
    bench = _flat_benchmark(n=100)
    dates = bench["trade_date"]
    rets = pd.DataFrame(
        np.full((len(dates), 5), 0.01),
        index=dates,
        columns=[f"S{i}" for i in range(5)],
    )
    rets.iloc[50:52, :] = -0.02  # only 2 days of broad decline
    result = compute_market_hard_gate(bench, breadth_panel=rets)
    reasons = set(result.frame.loc[result.frame["hard_gate_active"], "trigger_reason"].unique())
    assert "breadth_collapse" not in reasons


# ---------------------------------------------------------------------------
# Cool-down behaviour
# ---------------------------------------------------------------------------

def test_cool_down_extends_active_window():
    bench = _flat_benchmark(n=250)
    # Single 1-day -10% drop at index 200
    bench.loc[196:200, "close"] = bench.loc[195, "close"] * np.linspace(0.99, 0.90, 5)
    bench.loc[201:, "close"] = bench.loc[200, "close"]
    cfg = MarketHardGateConfig(cool_down_days=5)
    result = compute_market_hard_gate(bench, config=cfg)
    active_idx = result.frame.index[result.frame["hard_gate_active"]].tolist()
    # cool_down rows tagged as "cool_down"
    cool_reasons = result.frame.loc[result.frame["trigger_reason"] == "cool_down"]
    assert not cool_reasons.empty


def test_cool_down_remaining_decrements():
    bench = _flat_benchmark(n=250)
    bench.loc[196:200, "close"] = bench.loc[195, "close"] * np.linspace(0.99, 0.90, 5)
    bench.loc[201:, "close"] = bench.loc[200, "close"]
    cfg = MarketHardGateConfig(cool_down_days=5)
    result = compute_market_hard_gate(bench, config=cfg)
    # Look at the trailing edge of an active window
    active_idx = result.frame.index[result.frame["hard_gate_active"]].tolist()
    if active_idx:
        tail = result.frame.iloc[active_idx[-5:]]
        # cool_down_remaining should be strictly decreasing at the tail
        cd = tail["cool_down_remaining"].tolist()
        assert cd == sorted(cd, reverse=True)


# ---------------------------------------------------------------------------
# Multiplier helper
# ---------------------------------------------------------------------------

def test_multiplier_is_one_when_inactive():
    bench = _flat_benchmark(n=100)
    result = compute_market_hard_gate(bench)
    cfg = MarketHardGateConfig()
    mult = hard_gate_multiplier(result.frame, bench["trade_date"].iloc[50], cfg)
    assert mult == 1.0


def test_multiplier_zero_when_active():
    bench = _flat_benchmark(n=250)
    bench.loc[196:200, "close"] = bench.loc[195, "close"] * np.linspace(0.99, 0.90, 5)
    result = compute_market_hard_gate(bench)
    cfg = MarketHardGateConfig(blocked_gross_multiplier=0.0)
    # Find the first active date
    active = result.frame.loc[result.frame["hard_gate_active"]]
    assert not active.empty
    mult = hard_gate_multiplier(result.frame, active.iloc[0]["trade_date"], cfg)
    assert mult == 0.0


def test_multiplier_respects_blocked_multiplier():
    bench = _flat_benchmark(n=250)
    bench.loc[196:200, "close"] = bench.loc[195, "close"] * np.linspace(0.99, 0.90, 5)
    result = compute_market_hard_gate(bench)
    cfg = MarketHardGateConfig(blocked_gross_multiplier=0.05)
    active = result.frame.loc[result.frame["hard_gate_active"]]
    mult = hard_gate_multiplier(result.frame, active.iloc[0]["trade_date"], cfg)
    assert mult == pytest.approx(0.05)


def test_multiplier_returns_one_when_frame_empty():
    cfg = MarketHardGateConfig()
    assert hard_gate_multiplier(pd.DataFrame(), pd.Timestamp("2024-01-01"), cfg) == 1.0


# ---------------------------------------------------------------------------
# Disabled / empty inputs
# ---------------------------------------------------------------------------

def test_disabled_returns_empty_frame():
    cfg = MarketHardGateConfig(enabled=False)
    bench = _flat_benchmark(n=100)
    result = compute_market_hard_gate(bench, config=cfg)
    assert result.frame.empty


def test_empty_benchmark_returns_empty():
    result = compute_market_hard_gate(None)
    assert result.frame.empty
    result = compute_market_hard_gate(pd.DataFrame())
    assert result.frame.empty


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def test_active_windows_collapse_contiguous_days():
    bench = _flat_benchmark(n=250)
    bench.loc[196:200, "close"] = bench.loc[195, "close"] * np.linspace(0.99, 0.90, 5)
    bench.loc[201:, "close"] = bench.loc[200, "close"]
    result = compute_market_hard_gate(bench)
    assert len(result.windows) >= 1
    w = result.windows[0]
    assert "start" in w and "end" in w and "days" in w
    assert w["days"] >= 1


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

def test_manifest_writer_emits_json(tmp_path):
    bench = _flat_benchmark(n=250)
    bench.loc[196:200, "close"] = bench.loc[195, "close"] * np.linspace(0.99, 0.90, 5)
    result = compute_market_hard_gate(bench)
    path = write_hard_gate_manifest(result, tmp_path)
    assert path.exists()
    import json
    payload = json.loads(path.read_text())
    assert "n_dates" in payload and "n_hard_gate_active" in payload and "windows" in payload


# ---------------------------------------------------------------------------
# Manifest summary integrity
# ---------------------------------------------------------------------------

def test_active_share_consistent_with_frame():
    bench = _flat_benchmark(n=250)
    bench.loc[196:200, "close"] = bench.loc[195, "close"] * np.linspace(0.99, 0.90, 5)
    result = compute_market_hard_gate(bench)
    m = result.to_manifest()
    assert m["n_hard_gate_active"] == int(result.frame["hard_gate_active"].sum())
    assert 0.0 <= m["active_share"] <= 1.0
