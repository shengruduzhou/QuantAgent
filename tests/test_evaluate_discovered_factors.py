"""Unit tests for the economic-profile helpers added to the factor eval script.

The script lives under ``scripts/`` (not an importable package), so it is loaded
by file path. Only the pure helpers are exercised here — no parquet / subprocess.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_discovered_factors.py"
_spec = importlib.util.spec_from_file_location("evaluate_discovered_factors", _SCRIPT)
ed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ed)  # type: ignore[union-attr]


def _cross_section(n_days: int = 30, n_sym: int = 10, seed: int = 1):
    """Factor that positively predicts the forward return, plus aligned series."""
    rng = np.random.default_rng(seed)
    dates, symbols, factor, fwd = [], [], [], []
    for di in range(n_days):
        f = rng.normal(0.0, 1.0, n_sym)
        # forward return increases with the factor (clean positive relationship)
        y = 0.01 * (f - f.mean()) / max(f.std(), 1e-9) + rng.normal(0.0, 0.002, n_sym)
        for si in range(n_sym):
            dates.append(pd.Timestamp("2024-08-01") + pd.Timedelta(days=di))
            symbols.append(f"S{si:02d}")
            factor.append(f[si])
            fwd.append(y[si])
    return (pd.Series(factor), pd.Series(fwd), pd.Series(dates), pd.Series(symbols))


def test_ic_decay_curve_returns_all_horizons_finite():
    factor, fwd, dates, _ = _cross_section()
    fwd_by_h = {1: fwd, 5: fwd * 0.5}  # both positively related to factor
    mask = pd.Series(True, index=factor.index)
    curve = ed._ic_decay_curve(factor, fwd_by_h, dates, mask)
    assert set(curve) == {1, 5}
    assert all(np.isfinite(v) for v in curve.values())
    # Positive relationship → positive IC at horizon 1.
    assert curve[1] > 0.0


def test_economic_profile_positive_factor_has_positive_long_short():
    factor, fwd, dates, symbols = _cross_section()
    mask = pd.Series(True, index=factor.index)
    prof = ed._economic_profile(factor, fwd, dates, symbols, mask, sign=1.0, cost_bps=15.0, q=5)
    # Top bucket beats the long-short floor; spread positive; after-cost < gross.
    assert np.isfinite(prof["oos_long_short_return"]) and prof["oos_long_short_return"] > 0.0
    assert np.isfinite(prof["oos_top_bucket_return"])
    assert prof["oos_long_short_after_cost"] <= prof["oos_long_short_return"] + 1e-12
    # Turnover is a churn fraction in [0, 1].
    assert 0.0 - 1e-9 <= prof["oos_turnover"] <= 1.0 + 1e-9


def test_economic_profile_sign_flip_inverts_long_short():
    """Orienting with sign=-1 flips which tail is 'top' → spread changes sign."""
    factor, fwd, dates, symbols = _cross_section()
    mask = pd.Series(True, index=factor.index)
    pos = ed._economic_profile(factor, fwd, dates, symbols, mask, sign=1.0, cost_bps=0.0)
    neg = ed._economic_profile(factor, fwd, dates, symbols, mask, sign=-1.0, cost_bps=0.0)
    assert pos["oos_long_short_return"] > 0.0
    assert neg["oos_long_short_return"] < 0.0


def test_economic_profile_empty_returns_nans():
    empty = pd.Series([], dtype=float)
    prof = ed._economic_profile(empty, empty, empty, empty, empty.astype(bool), sign=1.0, cost_bps=15.0)
    assert all(np.isnan(v) for v in prof.values())


def test_rejected_export_writes_csv_and_grouped_md(tmp_path):
    table = pd.DataFrame([
        {"name": "good", "status": "accepted", "reject_detail": "", "expression": "X"},
        {"name": "bad1", "status": "oos_ic_failed", "reject_detail": "|OOS RankIC| 0.001 < 0.015 min", "expression": "A"},
        {"name": "bad2", "status": "oos_ic_failed", "reject_detail": "sign flip", "expression": "B"},
        {"name": "bad3", "status": "no_monotonicity", "reject_detail": "weak quantiles", "expression": "C"},
    ])
    csv_path, md_path = ed._write_rejected_export(table, tmp_path)
    rej = pd.read_csv(csv_path)
    # Only the 3 rejected rows, accepted excluded.
    assert set(rej["name"]) == {"bad1", "bad2", "bad3"}
    assert "reject_detail" in rej.columns
    md = md_path.read_text()
    assert "Total rejected: **3**" in md
    assert "## oos_ic_failed (2)" in md
    assert "## no_monotonicity (1)" in md
    assert "sign flip" in md


def test_rejected_export_empty_table(tmp_path):
    table = pd.DataFrame([{"name": "good", "status": "accepted", "reject_detail": "", "expression": "X"}])
    csv_path, md_path = ed._write_rejected_export(table, tmp_path)
    assert pd.read_csv(csv_path).empty
    assert "No rejected factors" in md_path.read_text()


def test_standard_report_leaderboard_is_accepted_only_sorted(tmp_path):
    table = pd.DataFrame([
        {"name": "a", "status": "accepted", "oos_rank_ic": 0.02, "oos_icir": 0.5,
         "oos_long_short_return": 0.01, "oos_turnover": 0.3, "expression": "A", "description": "da"},
        {"name": "b", "status": "accepted", "oos_rank_ic": -0.08, "oos_icir": 0.9,
         "oos_long_short_return": 0.03, "oos_turnover": 0.4, "expression": "B", "description": "db"},
        {"name": "c", "status": "oos_ic_failed", "oos_rank_ic": 0.001, "reject_detail": "low IC", "expression": "C"},
    ])
    meta = {"candidates": 3, "train_end": "2024-07-31", "oos_end": None,
            "reference_oos_icir_median": 0.4, "icir_gate": 0.2, "min_oos_ic": 0.015,
            "max_reference_correlation": 0.6, "max_correlation": 0.7, "min_monotonicity_corr": 0.6}
    lb_path, report_path = ed._write_standard_report(table, meta, tmp_path)
    lb = pd.read_csv(lb_path)
    # Accepted only, and sorted by |oos_rank_ic| desc → b (0.08) before a (0.02).
    assert lb["name"].tolist() == ["b", "a"]
    report = report_path.read_text()
    assert "accepted: **2**" in report and "rejected: **1**" in report
    assert "Accepted factor library" in report
    assert "oos_ic_failed" in report  # rejection summary present


def test_md_table_handles_empty_and_values():
    assert ed._md_table(pd.DataFrame(), ["a"]) == ["_(none)_"]
    df = pd.DataFrame([{"name": "x", "ic": 0.1234567}])
    out = ed._md_table(df, ["name", "ic", "missing"])
    assert out[0] == "| name | ic |"  # missing column dropped
    assert "0.1235" in out[2]
