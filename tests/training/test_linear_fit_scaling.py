"""Regression tests for the penalised linear baseline.

The linear model is the control every "the tree is better" claim in this
repository is scored against, so a defect here does not merely lose accuracy —
it manufactures evidence for nonlinearity. Both bugs fixed here did exactly
that:

* ridge solved the normal equations on the raw design, where a turnover column
  in yuan sits ~10 orders of magnitude above a rank-scale momentum column. The
  ``pinv`` singular-value cutoff discarded the momentum direction outright.
* elastic net ran proximal gradient descent with a hard-coded ``lr = 0.05`` on
  that same design, which is far above the ``1/L`` bound convergence requires,
  so it overflowed to NaN within a few iterations.

Neither failed loudly. The first returned near-zero coefficients and a rank IC
of roughly zero; the second returned NaN predictions that the metric layer
turned back into 0.0.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quantagent.training.v7_experiment import (
    V7TrainingConfig,
    _fit_linear,
    _predict_linear,
)


@pytest.fixture
def mixed_scale_data() -> tuple[pd.DataFrame, pd.Series]:
    """One yuan-scale column of noise, one rank-scale column carrying all signal."""
    rng = np.random.default_rng(0)
    n = 4000
    features = pd.DataFrame(
        {
            "amount_mean_20d": rng.lognormal(18.0, 1.0, n),
            "momentum_20d": rng.uniform(-1.0, 1.0, n),
        }
    )
    label = 0.01 * features["momentum_20d"] + pd.Series(rng.normal(0, 0.02, n))
    return features, label


@pytest.mark.parametrize("backend", ["ridge", "elastic_net"])
def test_fit_is_finite_on_mixed_scale_features(backend, mixed_scale_data):
    features, label = mixed_scale_data
    config = V7TrainingConfig() if backend == "ridge" else replace(V7TrainingConfig(), alpha=0.1)

    fit = _fit_linear(features, label, config, backend)

    assert np.all(np.isfinite(fit.coef))
    assert np.isfinite(fit.intercept)
    assert np.all(np.isfinite(_predict_linear(features, fit.coef, fit.intercept)))


@pytest.mark.parametrize("backend", ["ridge", "elastic_net"])
def test_fit_recovers_the_small_scale_signal(backend, mixed_scale_data):
    """The predictive column must survive being 10 orders of magnitude smaller."""
    features, label = mixed_scale_data
    config = V7TrainingConfig() if backend == "ridge" else replace(V7TrainingConfig(), alpha=0.1)

    fit = _fit_linear(features, label, config, backend)
    prediction = pd.Series(_predict_linear(features, fit.coef, fit.intercept))
    rank_ic = prediction.corr(label, method="spearman")

    # The attainable rank IC on this signal-to-noise ratio is ~0.27. Before the
    # fix, ridge scored 0.012 here — the yuan-scale column crowded the signal
    # out of the solve entirely.
    assert rank_ic > 0.20, f"{backend} rank IC {rank_ic:.4f} — the signal was lost"
    # The signal column must carry the larger standardised coefficient.
    assert abs(fit.standardised_coef[1]) > abs(fit.standardised_coef[0])


def test_standardised_coefficients_are_comparable_across_features(mixed_scale_data):
    """`single_factor_dominance` reads these, so they must share a unit."""
    features, label = mixed_scale_data

    fit = _fit_linear(features, label, V7TrainingConfig(), "ridge")

    # Original-space coefficients differ by ~10 orders of magnitude purely
    # because the features do; the standardised ones do not.
    original_ratio = abs(fit.coef[1]) / max(abs(fit.coef[0]), 1e-30)
    standardised_ratio = abs(fit.standardised_coef[1]) / max(abs(fit.standardised_coef[0]), 1e-30)
    assert original_ratio > 1e6
    assert standardised_ratio < 1e3


def test_coefficients_are_returned_in_the_original_feature_space(mixed_scale_data):
    """The published artifact contract is ``x @ coef + intercept``, unscaled.

    ``quantagent.training.v7_predictor._predict_classic`` scores a frame that
    way with no scaler alongside, so folding the standardisation back into the
    coefficients is what keeps a saved model loadable.
    """
    features, label = mixed_scale_data
    fit = _fit_linear(features, label, V7TrainingConfig(), "ridge")

    direct = _predict_linear(features, fit.coef, fit.intercept)
    # Reconstruct through the standardised space and check they agree.
    z = (features.to_numpy(dtype=float) - fit.center) / fit.scale
    via_standardised = z @ fit.standardised_coef + (
        fit.intercept + float(np.dot(fit.coef, fit.center))
    )

    np.testing.assert_allclose(direct, via_standardised, rtol=1e-8, atol=1e-12)


def test_elastic_net_converges_and_reports_it(mixed_scale_data):
    features, label = mixed_scale_data

    fit = _fit_linear(features, label, replace(V7TrainingConfig(), alpha=0.1), "elastic_net")

    assert fit.converged
    assert 1 <= fit.iterations < 2000


def test_elastic_net_penalty_is_scale_free(mixed_scale_data):
    """``alpha`` must mean the same thing whatever units the label carries.

    The penalty is expressed as a fraction of glmnet's ``lambda_max`` — the
    smallest l1 penalty that zeroes every coefficient. Without that, the shipped
    default sat far above the gradient of a daily-return target and the backend
    returned a constant for every dataset.
    """
    features, label = mixed_scale_data
    config = replace(V7TrainingConfig(), alpha=0.1)

    small = _fit_linear(features, label, config, "elastic_net")
    # Same problem, label scaled by 1000. A scale-free penalty must pick the
    # same feature and scale its coefficient by the same factor.
    large = _fit_linear(features, label * 1000.0, config, "elastic_net")

    assert np.argmax(np.abs(small.standardised_coef)) == np.argmax(np.abs(large.standardised_coef))
    ratio = large.standardised_coef[1] / small.standardised_coef[1]
    assert ratio == pytest.approx(1000.0, rel=0.05)


def test_degenerate_all_zero_fit_raises_rather_than_predicting_a_constant():
    """An all-zero fit is a model failure, and must not read as a rejection.

    A constant prediction makes rank IC undefined, and the metric layer fills
    undefined with 0.0 — which downstream looks exactly like "we tested this and
    it was worthless". The two events have to stay distinguishable.
    """
    rng = np.random.default_rng(3)
    features = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    label = pd.Series(rng.normal(0, 0.02, 500))
    # An enormous penalty relative to lambda_max zeroes everything.
    config = replace(V7TrainingConfig(), alpha=50.0, l1_ratio=1.0)

    with pytest.raises(ValueError, match="degenerate, not rejected"):
        _fit_linear(features, label, config, "elastic_net")


def test_non_finite_input_does_not_produce_a_silent_nan_model():
    rng = np.random.default_rng(4)
    features = pd.DataFrame(
        {
            "a": rng.normal(size=500),
            "b": np.concatenate([[np.inf, np.nan], rng.normal(size=498)]),
        }
    )
    label = pd.Series(0.01 * features["a"] + rng.normal(0, 0.02, 500))

    fit = _fit_linear(features, label, V7TrainingConfig(), "ridge")

    assert np.all(np.isfinite(fit.coef))
    assert np.all(np.isfinite(_predict_linear(features, fit.coef, fit.intercept)))


def test_constant_column_does_not_blow_up_the_scaler():
    rng = np.random.default_rng(5)
    features = pd.DataFrame({"constant": np.ones(400), "signal": rng.normal(size=400)})
    label = pd.Series(0.01 * features["signal"] + rng.normal(0, 0.02, 400))

    fit = _fit_linear(features, label, V7TrainingConfig(), "ridge")

    assert np.all(np.isfinite(fit.coef))
    assert fit.scale[0] == 1.0
