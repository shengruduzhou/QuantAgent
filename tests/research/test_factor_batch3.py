"""Stage A static validation for H-025 batch-3 factors (fu_20260713).

No-lookahead, finite-value, required-column/lookback, duplicate-expression
checks on synthetic panels, plus gate-extension tests for the shared
score_factors scorer (novelty gate + medium-turnover track + batch-1
backward compatibility). No market data touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from factor_batch3_pv import FACTORS3, REF3  # noqa: E402
from dual_track_factor_batch import FACTORS as FACTORS1, score_factors  # noqa: E402
from quantagent.factors import expr as E  # noqa: E402
from quantagent.factors.expr import _expr_lookback, _expr_required_columns  # noqa: E402

PANEL_COLS = {"open", "high", "low", "close", "volume", "amount"}


def _synthetic_panel(n_days=250, n_symbols=4, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rows = []
    for s in range(n_symbols):
        close = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        spread = np.abs(rng.normal(0, 0.01, n_days)) * close
        volume = rng.integers(1_000, 1_000_000, n_days).astype(float)
        rows.append(pd.DataFrame({
            "symbol": f"SYM{s}.SZ", "trade_date": dates,
            "open": close * (1 + rng.normal(0, 0.005, n_days)),
            "high": close + spread, "low": close - spread, "close": close,
            "volume": volume, "amount": volume * close,
        }))
    return pd.concat(rows, ignore_index=True).sort_values(
        ["symbol", "trade_date"]).reset_index(drop=True)


def test_required_columns_and_lookback():
    for name, (_, ex) in {**FACTORS3, **{k: (None, v) for k, v in REF3.items()}}.items():
        cols = _expr_required_columns(ex)
        assert cols <= PANEL_COLS, f"{name} needs unavailable columns {cols - PANEL_COLS}"
        lb = _expr_lookback(ex)
        assert lb <= 130, f"{name} lookback {lb} exceeds loader warmup budget"


def test_no_lookahead():
    panel = _synthetic_panel()
    cutoff = panel["trade_date"].sort_values().unique()[150]
    baseline = {n: ex.evaluate(panel) for n, (_, ex) in FACTORS3.items()}
    mutated = panel.copy()
    future = mutated["trade_date"] > cutoff
    rng = np.random.default_rng(99)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        mutated.loc[future, col] = mutated.loc[future, col].to_numpy() * rng.uniform(
            1.5, 3.0, int(future.sum()))
    past = (panel["trade_date"] <= cutoff).to_numpy()
    for name, (_, ex) in FACTORS3.items():
        after = ex.evaluate(mutated)
        np.testing.assert_allclose(
            baseline[name].to_numpy()[past], after.to_numpy()[past],
            equal_nan=True, err_msg=f"{name} reads future data")


def test_finite_on_degenerate_bars():
    panel = _synthetic_panel(seed=7)
    # limit days: high == low == close; dead days: volume = amount = 0
    idx = panel.sample(frac=0.1, random_state=1).index
    for col in ("high", "low", "open"):
        panel.loc[idx, col] = panel.loc[idx, "close"].to_numpy()
    idx2 = panel.sample(frac=0.1, random_state=2).index
    panel.loc[idx2, ["volume", "amount"]] = 0.0
    for name, (_, ex) in FACTORS3.items():
        vals = ex.evaluate(panel).to_numpy()
        assert not np.isinf(vals).any(), f"{name} produces +-inf on degenerate bars"


def _canon(ex) -> str:
    children = E.expr_children(ex)
    inner = ",".join(_canon(c) for c in children)
    extras = ",".join(f"{k}={v}" for k, v in sorted(vars(ex).items())
                      if isinstance(v, (int, float, str)))
    return f"{type(ex).__name__}({inner};{extras})"


def test_no_duplicate_expressions():
    # D6R is the PREREGISTERED re-gate of batch-1 D6 on the medium-turnover
    # track: same formula by design — lock that identity, exclude from dup scan.
    assert _canon(FACTORS3["D6R_vol_compression_regate"][1]) == _canon(
        FACTORS1["D6_vol_compression"][1])
    all_exprs = {n: ex for n, (_, ex) in {**FACTORS1, **FACTORS3}.items()
                 if n != "D6R_vol_compression_regate"}
    canons = {n: _canon(ex) for n, ex in all_exprs.items()}
    seen = {}
    for name, c in canons.items():
        assert c not in seen, f"{name} duplicates {seen.get(c)}"
        seen[c] = name


def _gate_lab(seed=3):
    """Synthetic lab: stable per-symbol scores => ~zero turnover, engineered IC."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=60)
    syms = [f"S{i:03d}" for i in range(80)]
    x = rng.normal(size=len(syms))   # signal axis 1 (also the ref)
    y = rng.normal(size=len(syms))   # signal axis 2 (orthogonal to ref)
    rows = []
    for d in dates:
        noise = rng.normal(0, 0.2, len(syms))
        rows.append(pd.DataFrame({
            "trade_date": d, "symbol": syms,
            "F_copy": x, "F_good": y, "F_med": 0.7 * y + 0.3 * rng.normal(size=len(syms)),
            "ref_axis": x, "amount": rng.uniform(1e6, 1e8, len(syms)),
            "forward_return_10d": (x + y) * 0.005 + noise * 0.001,
            "forward_return_20d": (x + y) * 0.005 + noise * 0.001,
        }))
    return pd.concat(rows, ignore_index=True)


def test_score_factors_novelty_and_medium_gates(tmp_path):
    lab = _gate_lab()
    meta = {"F_copy": "low_turnover", "F_good": "low_turnover", "F_med": "defensive_medium"}
    df = score_factors(lab, meta, ["ref_axis"], tmp_path / "out.csv",
                       turnover_caps={"defensive_medium": 0.35}, max_ref_corr=0.85,
                       cost_decay_classes=frozenset({"defensive_medium"})).set_index("factor")
    assert df.loc["F_copy", "g_novel"] == False  # noqa: E712
    assert df.loc["F_copy", "verdict"] == "reject"          # killed ONLY by novelty
    assert df.loc["F_copy", "g_ic_pos"] and df.loc["F_copy", "g_turn"] and df.loc["F_copy", "g_cost"]
    assert df.loc["F_good", "verdict"] == "accept"
    assert df.loc["F_med", "verdict"] in ("accept", "redundant")  # medium plumbing works


def test_score_factors_batch1_backward_compat(tmp_path):
    lab = _gate_lab()
    meta = {"F_copy": "low_turnover", "F_good": "low_turnover"}
    df = score_factors(lab, meta, ["ref_axis"], tmp_path / "out.csv").set_index("factor")
    assert df["g_novel"].isna().all()                        # gate disabled by default
    assert df.loc["F_copy", "verdict"] == "accept"           # old behavior preserved


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
