"""Tests for Hansen (2005) Superior Predictive Ability bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.quant_math.performance import _spa_recenter_means, spa_test


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
    out = spa_test(candidates, benchmark, n_bootstrap=600, rng_seed=1)
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
    out = spa_test(candidates, benchmark, n_bootstrap=600, rng_seed=2)
    assert out["p_consistent"] > 0.10, (
        f"SPA falsely rejected null when no edge exists: p={out['p_consistent']}"
    )


def test_spa_pvalues_follow_hansen_lower_consistent_upper_ordering():
    rng = np.random.default_rng(21)
    n = 420
    benchmark = pd.Series(rng.normal(0.0, 0.01, size=n))
    candidates = pd.DataFrame(
        {
            "edge": benchmark + 0.0006 + rng.normal(0.0, 0.01, size=n),
            "near_null": benchmark + rng.normal(0.0, 0.01, size=n),
            "bad": benchmark - 0.004 + rng.normal(0.0, 0.01, size=n),
        }
    )

    out = spa_test(candidates, benchmark, n_bootstrap=800, rng_seed=3)

    assert 0.0 <= out["p_lower"] <= out["p_consistent"] <= out["p_upper"] <= 1.0


def test_consistent_recenter_uses_raw_mean_standard_error_loglog_bound():
    n = 500
    omega = np.array([0.0010, 0.0010, 0.0010])
    bound = float(np.sqrt(2.0 * np.log(np.log(n))))
    means = np.array(
        [
            0.0010,
            -0.99 * omega[1] * bound,
            -1.01 * omega[2] * bound,
        ]
    )

    lower, consistent, upper, relevant = _spa_recenter_means(means, omega, n)

    assert relevant.tolist() == [True, True, False]
    assert consistent[0] == pytest.approx(means[0])
    assert consistent[1] == pytest.approx(means[1])
    assert consistent[2] == 0.0
    assert lower.tolist() == pytest.approx([means[0], 0.0, 0.0])
    assert upper.tolist() == pytest.approx(means.tolist())


def test_spa_consistent_pvalue_is_insensitive_to_clearly_irrelevant_bad_alternative():
    rng = np.random.default_rng(31)
    n = 500
    benchmark = pd.Series(rng.normal(0.0, 0.01, size=n))
    base = pd.DataFrame(
        {
            "edge": benchmark + 0.0012 + rng.normal(0.0, 0.01, size=n),
            "noise": benchmark + rng.normal(0.0, 0.01, size=n),
        }
    )
    expanded = base.copy()
    # Same benchmark plus a very poor constant differential. It should be
    # classified irrelevant by the consistent SPA null instead of inflating the
    # multiple-comparison distribution like a valid contender.
    expanded["clearly_bad"] = benchmark - 0.05

    base_out = spa_test(base, benchmark, n_bootstrap=1000, block_length=8, rng_seed=7)
    expanded_out = spa_test(
        expanded,
        benchmark,
        n_bootstrap=1000,
        block_length=8,
        rng_seed=7,
    )

    assert abs(expanded_out["p_consistent"] - base_out["p_consistent"]) <= 0.03
    assert expanded_out["best_strategy"] == "edge"


def test_spa_rejects_invalid_bootstrap_parameters():
    benchmark = pd.Series([0.0] * 10)
    candidates = pd.DataFrame({"x": [0.0] * 10})

    with pytest.raises(ValueError, match="n_bootstrap"):
        spa_test(candidates, benchmark, n_bootstrap=0)
    with pytest.raises(ValueError, match="block_length"):
        spa_test(candidates, benchmark, n_bootstrap=10, block_length=0)


def test_spa_handles_degenerate_input():
    bench = pd.Series([0.0] * 3)
    cands = pd.DataFrame({"x": [0.0] * 3})
    out = spa_test(cands, bench, n_bootstrap=10)
    assert np.isnan(out["p_consistent"])
