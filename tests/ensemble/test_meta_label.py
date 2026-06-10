"""Tests for meta-labeling (signal execution filter / sizing)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.ensemble.meta_label import build_dot_meta_dataset, fit_meta_labeler, meta_filter


def test_meta_labeler_learns_separable_signal():
    rng = np.random.default_rng(0)
    n = 2000
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    # success driven by f1 - f2 (a learnable combination)
    prob = 1 / (1 + np.exp(-(1.5 * f1 - 1.2 * f2)))
    y = (rng.uniform(size=n) < prob).astype(int)
    df = pd.DataFrame({"f1": f1, "f2": f2, "success": y})
    ml = fit_meta_labeler(df, ["f1", "f2"])
    p = ml.predict_success(df)
    # higher predicted P on actual successes than failures
    assert p[y == 1].mean() > p[y == 0].mean() + 0.1
    assert ((p >= 0) & (p <= 1)).all()


def test_meta_filter_masks_and_sizes():
    p = np.array([0.3, 0.5, 0.75, 1.0])
    size = meta_filter(p, floor=0.5)
    assert size[0] == 0.0            # below floor → skip
    assert size[1] == 0.0            # at floor → 0 size
    assert 0 < size[2] <= 1.0
    assert size[3] == 1.0


def test_build_dot_meta_dataset_success_label():
    fsm = pd.DataFrame({
        "open_auction_gap": [0.01, -0.02], "intraday_range_pos": [0.2, 0.8],
        "net_buy_pressure": [0.1, -0.3], "vwap_deviation": [-0.01, 0.02],
        "ret": [0.012, -0.01], "exit_reason": ["止盈", "止损"],
    })
    d = build_dot_meta_dataset(fsm)
    assert d["success"].tolist() == [1, 0]
