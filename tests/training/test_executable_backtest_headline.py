"""Executable backtest headline metrics (round-29 R2-F08).

The v7 executable backtest used to report drawdown against a
rolling 252-day peak, so any drawdown lasting longer than a year is understated
in `max_drawdown_pct`, `executable_max_drawdown` and the "Max DD target passed"
line; its "Annualised return" is (1+mean)^252-1, not CAGR; and a missing
benchmark is reported as a 0%-return benchmark.

Drives the REAL `_compute_horizon_sleeve_backtest` with a synthetic
single-name path. All overlays (drawdown ladder, vol target, regime, hard gate)
are neutralised through config so NAV == held-asset path and the true numbers
are known in closed form. FAILS on 22b3f6c.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from quantagent.training.v7_experiment import _compute_horizon_sleeve_backtest


def _config():
    return SimpleNamespace(
        executable_sleeves=(("only", ((1, 1.0),), 1.0, 1, 1),),
        executable_base_gross=1.0,
        executable_max_weight_per_name=1.0,
        executable_max_turnover=0.0,
        executable_vol_target_annual=0.0,          # vol multiplier off
        drawdown_soft_limit=9.0, drawdown_hard_limit=9.5, drawdown_kill_limit=9.9,  # ladder off
        regime_gate_enabled=False, hard_gate_enabled=False,
        cost_bps=0.0, risk_free_rate_annual=0.0, initial_capital=1_000_000.0,
        target_max_drawdown=0.25,
    )


def _predictions(daily_returns: np.ndarray) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-05", periods=len(daily_returns))
    return pd.DataFrame({
        "trade_date": dates, "symbol": "600000.SH", "horizon": 1,
        "prediction": 1.0, "forward_return_1d": daily_returns,
    })


def test_multi_year_drawdown_must_be_reported_against_all_time_peak(tmp_path):
    # 30 up days, then 400 days of -0.1%: a 400-day, ~33% drawdown.
    path = np.concatenate([np.full(30, 0.001), np.full(400, -0.001)])
    out = _compute_horizon_sleeve_backtest(_predictions(path), _config(), tmp_path)
    curve = pd.read_csv(tmp_path / "equity_curve.csv")
    nav = curve["nav"].astype(float)
    true_dd = float((nav / nav.cummax() - 1.0).min())
    print(f"reported max_drawdown_pct = {out['max_drawdown_pct']:.2f}%  "
          f"all-time-peak max DD from the same nav = {true_dd * 100:.2f}%  "
          f"target_max_drawdown=25% -> reported 'passed' = {out.get('max_drawdown_target_passed')}")
    assert abs(out["max_drawdown_pct"] / 100.0 - true_dd) < 1e-6


def test_annualised_return_must_be_geometric(tmp_path):
    rng = np.random.default_rng(7)
    path = rng.normal(0.0004, 0.02, 504)  # 2y, 32% ann. vol
    out = _compute_horizon_sleeve_backtest(_predictions(path), _config(), tmp_path)
    curve = pd.read_csv(tmp_path / "equity_curve.csv")
    nav = curve["nav"].astype(float)
    cagr = (nav.iloc[-1] / 1_000_000.0) ** (252 / len(nav)) - 1.0
    print(f"reported annualised_return_pct = {out['annualised_return_pct']:.2f}%  geometric CAGR = {cagr * 100:.2f}%")
    assert abs(out["annualised_return_pct"] / 100.0 - cagr) < 1e-3


def test_missing_benchmark_must_not_be_a_zero_return_benchmark(tmp_path):
    path = np.full(60, 0.001)
    out = _compute_horizon_sleeve_backtest(_predictions(path), _config(), tmp_path)
    bench_keys = {k: v for k, v in out.items() if "bench" in k or "excess" in k}
    print("benchmark/excess fields with NO benchmark configured:", bench_keys)
    assert out.get("benchmark_annualised_pct") is None and out.get("excess_annualised_pct") is None


def test_rolling_peak_stays_available_as_the_ladder_control_input(tmp_path):
    path = np.concatenate([np.full(30, 0.001), np.full(400, -0.001)])
    _compute_horizon_sleeve_backtest(_predictions(path), _config(), tmp_path)
    curve = pd.read_csv(tmp_path / "equity_curve.csv")
    # The reported drawdown never heals without a new high; the control input does.
    assert curve["drawdown"].iloc[-1] < curve["control_drawdown"].iloc[-1]
