"""Tests for combinatorially-symmetric PBO (Bailey et al. 2014)."""

from __future__ import annotations

import numpy as np

from quantagent.quant_math.performance import probability_of_backtest_overfitting


def test_pbo_low_when_one_strategy_truly_dominates():
    """If one strategy has uniformly higher mean across all chunks, PBO must be small."""
    rng = np.random.default_rng(0)
    n_slices = 200
    n_strats = 12
    noise = rng.normal(0, 0.5, size=(n_slices, n_strats))
    # Strategy 0 has a real edge worth ~1 sigma per chunk; rest are noise.
    noise[:, 0] += 1.0
    pbo = probability_of_backtest_overfitting(noise, n_partitions=16, rng_seed=1)
    assert 0.0 <= pbo <= 1.0
    assert pbo < 0.2, f"true-edge case yielded PBO={pbo:.3f}, expected < 0.2"


def test_pbo_high_when_strategies_are_pure_noise():
    """Symmetric noise across strategies should give PBO near or above 0.5."""
    rng = np.random.default_rng(11)
    n_slices = 200
    n_strats = 12
    perf = rng.normal(0, 1.0, size=(n_slices, n_strats))
    pbo = probability_of_backtest_overfitting(perf, n_partitions=16, rng_seed=7)
    assert 0.0 <= pbo <= 1.0
    # No real differentiation → PBO should not be small.
    assert pbo > 0.3, f"noise case yielded PBO={pbo:.3f}, expected > 0.3"


def test_pbo_rejects_invalid_input():
    import pytest

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(rng.normal(size=(100,)))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(rng.normal(size=(100, 1)))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(rng.normal(size=(8, 5)), n_partitions=16)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(rng.normal(size=(64, 5)), n_partitions=15)
