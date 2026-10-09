"""MinTRL and bootstrap Sharpe interval (round-29 R1-F07)."""

import numpy as np
import pandas as pd

from quantagent.quant_math.performance import (
    minimum_track_record_length,
    sharpe_bootstrap_interval,
)


def test_min_trl_normal_case_matches_closed_form():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 5000))
    sr = r.mean() / r.std(ddof=1)
    expected = 1 + (1 - r.skew() * sr + (r.kurt() + 2) / 4 * sr ** 2) * (1.6448536269514722 / sr) ** 2
    assert abs(minimum_track_record_length(r) - expected) < 1e-6


def test_min_trl_is_infinite_without_a_positive_edge():
    r = pd.Series(np.random.default_rng(1).normal(-0.0005, 0.01, 500))
    assert minimum_track_record_length(r) == float("inf")


def test_bootstrap_interval_brackets_the_point_estimate_and_is_deterministic():
    r = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 750))
    point = r.mean() / r.std(ddof=1) * np.sqrt(252)
    low, high = sharpe_bootstrap_interval(r, n_boot=300, seed=4)
    assert low < point < high
    assert (low, high) == sharpe_bootstrap_interval(r, n_boot=300, seed=4)
