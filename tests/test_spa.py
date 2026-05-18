"""Tests for Hansen (2005) Superior Predictive Ability bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.quant_math.performance import spa_test


def test_spa_rejects_null_when_one_strategy_truly_beats_benchmark():
    rng = np.random.default_rng(0)
    n = 500
    benchmark = pd.Series(rng.normal(0.0, 0.01, size=n))
    candidates = pd.DataFrame(
        {
            "noise_a": benchmark + rng.normal(0.0, 0.01, size=n),
            "noise_b": benchmark + rng.normal(0.0, 0.01, size=n),
            "edge": benchmark + 0.0015 + rng.normal(0.0, 0.01, size=n),
        }
    )
    out = spa_test(candidates, benchmark, n_bootstrap=400, rng_seed=1)
    assert out["best_strategy"] == "edge"
    assert out["test_statistic"] > 0.0
    assert out["p_consistent"] <= 0.10, f"SPA failed to reject null: p={out['p_consistent']}"


def test_spa_fails_to_reject_when_no_strategy_beats_benchmark():
    rng = np.random.default_rng(11)
    n = 400
    benchmark = pd.Series(rng.normal(0.001, 0.01, size=n))
    candidates = pd.DataFrame(
        {f"cand_{k}": benchmark + rng.normal(0.0, 0.01, size=n) for k in range(6)}
    )
    out = spa_test(candidates, benchmark, n_bootstrap=400, rng_seed=2)
    # All candidates are just benchmark + independent noise → no real edge.
    # Consistent p-value should be far from 0.
    assert out["p_consistent"] > 0.10, (
        f"SPA falsely rejected null when no edge exists: p={out['p_consistent']}"
    )


def test_spa_handles_degenerate_input():
    bench = pd.Series([0.0] * 3)
    cands = pd.DataFrame({"x": [0.0] * 3})
    out = spa_test(cands, bench, n_bootstrap=10)
    assert np.isnan(out["p_consistent"])
