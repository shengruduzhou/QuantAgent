"""Tests for score calibration + conformal uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.ensemble.calibration import fit_calibrator


def _synth(n_days=40, n_stocks=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2025-01-01", periods=n_days):
        score = rng.normal(size=n_stocks)
        # higher score → higher expected forward return (real signal) + noise
        fwd = 0.02 * score + rng.normal(scale=0.03, size=n_stocks)
        rows.append(pd.DataFrame({"trade_date": d, "symbol": [f"{i:06d}.SZ" for i in range(n_stocks)],
                                  "alpha_score": score, "forward_return_5d": fwd}))
    return pd.concat(rows, ignore_index=True)


def test_isotonic_pbeat_monotone_in_rank():
    df = _synth()
    cal = fit_calibrator(df)
    out = cal.apply(df)
    # higher calib_rank → higher p_beat (isotonic non-decreasing)
    g = out.groupby(pd.cut(out["calib_rank"], 5), observed=True)["p_beat"].mean()
    assert g.is_monotonic_increasing
    assert out["p_beat"].between(0, 1).all()


def test_conformal_width_positive_and_riskgate_compatible():
    df = _synth()
    cal = fit_calibrator(df, alpha=0.10)
    out = cal.apply(df)
    assert (out["conformal_width"] > 0).all()
    assert out["uncertainty"].between(0, 1).all()
    # per-symbol width Series indexed by symbol (what RiskGate consumes)
    last = out[out["trade_date"] == out["trade_date"].max()].set_index("symbol")["conformal_width"]
    assert last.notna().all() and len(last) > 0


def test_apply_adds_expected_columns():
    df = _synth()
    out = fit_calibrator(df).apply(df)
    for c in ("calib_rank", "p_beat", "conformal_width", "uncertainty"):
        assert c in out.columns


def test_forward_safe_fit_then_apply_disjoint():
    df = _synth(n_days=60)
    cut = df["trade_date"].quantile(0.6)
    cal = fit_calibrator(df[df["trade_date"] <= cut])   # fit on past
    out = cal.apply(df[df["trade_date"] > cut])         # apply on future
    assert len(out) > 0 and out["p_beat"].between(0, 1).all()
