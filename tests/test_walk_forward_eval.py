"""Stage 6 item 1: walk-forward OOS evaluation (per-fold + overall IC/ICIR)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quantagent.training.walk_forward_eval import evaluate_walk_forward_oos


def _signal_oos(corr_strength: float, n_dates: int = 20, n_sym: int = 12, folds: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_dates, freq="B")
    chunks = np.array_split(dates, folds)
    preds, labs = [], []
    for fid, chunk in enumerate(chunks):
        for d in chunk:
            for s in range(n_sym):
                fr = float(rng.normal(0.0, 0.02))
                alpha = corr_strength * fr + rng.normal(0.0, 0.002)
                preds.append({"symbol": f"S{s}", "trade_date": d, "fold_id": fid,
                              "alpha_1d": alpha, "alpha_5d": alpha * 0.5})
                labs.append({"symbol": f"S{s}", "trade_date": d,
                             "forward_return_1d": fr, "forward_return_5d": fr * 0.8})
    return pd.DataFrame(preds), pd.DataFrame(labs)


def test_evaluate_oos_recovers_positive_signal(tmp_path):
    oos, labels = _signal_oos(corr_strength=3.0)
    res = evaluate_walk_forward_oos(oos, labels, horizons=(1, 5), output_dir=str(tmp_path))

    o1 = res.overall[res.overall["horizon"] == 1].iloc[0]
    assert o1["rank_ic"] > 0.3            # strong positive signal recovered
    assert o1["hit_rate"] > 0.6
    assert o1["n_days"] == 20
    # per-fold rows for both folds × both horizons
    assert set(res.metrics_by_fold["fold_id"].unique()) == {0, 1}
    assert set(res.metrics_by_fold["horizon"].unique()) == {1, 5}
    # coverage
    assert res.coverage["n_folds"] == 2
    assert res.coverage["label_match_rate"]["forward_return_1d"] == 1.0
    # artifacts written
    assert (tmp_path / "metrics_by_fold.csv").exists()
    assert (tmp_path / "metrics_overall.csv").exists()
    cov = json.loads((tmp_path / "oos_coverage.json").read_text())
    assert cov["n_oos_rows"] == len(oos)


def test_evaluate_oos_noise_is_near_zero():
    oos, labels = _signal_oos(corr_strength=0.0, seed=5)
    res = evaluate_walk_forward_oos(oos, labels, horizons=(1,))
    assert abs(res.overall.iloc[0]["rank_ic"]) < 0.2   # no signal → IC ~ 0


def test_evaluate_oos_flags_missing_labels():
    oos, labels = _signal_oos(corr_strength=2.0)
    labels = labels[labels["symbol"] != "S0"]          # drop one symbol's labels
    res = evaluate_walk_forward_oos(oos, labels, horizons=(1,))
    assert res.coverage["label_match_rate"]["forward_return_1d"] < 1.0   # surfaced, not hidden
