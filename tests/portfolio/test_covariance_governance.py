from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.portfolio.covariance_governance import (
    CovarianceGovernanceConfig,
    fit_governed_covariance,
)
from quantagent.portfolio.pareto_allocator import PortfolioHardConstraints
from quantagent.portfolio.robust_pareto import allocate_robust_pareto_portfolio


def _returns(days: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(1729)
    dates = pd.bdate_range("2025-01-02", periods=days)
    market = rng.normal(0.0003, 0.012, days)
    data = {}
    for i, name in enumerate(("A", "B", "C", "D")):
        data[name] = 0.55 * market + rng.normal(0.0001 * (4 - i), 0.007 + i * 0.001, days)
    return pd.DataFrame(data, index=dates)


def test_governed_covariance_is_psd_and_auditable():
    returns = _returns()
    cutoff = returns.index[179]
    result = fit_governed_covariance(
        returns,
        train_end=cutoff,
        config=CovarianceGovernanceConfig(method="auto", min_fit_observations=80, min_calibration_observations=25),
    )
    eig = np.linalg.eigvalsh(result.covariance.to_numpy())
    assert eig.min() >= -1e-12
    assert result.report["train_end"] == cutoff.date().isoformat()
    assert result.report["train_observations"] == 180
    assert result.report["selected_method"] in {"sample", "diagonal_shrinkage", "ewma", "ledoit_wolf"}
    assert result.report["productionEligible"] is False
    assert result.report["researchOnly"] is True
    assert result.report["returns_sha256"]


def test_future_returns_cannot_change_train_window_covariance():
    returns = _returns()
    cutoff = returns.index[169]
    cfg = CovarianceGovernanceConfig(method="auto", min_fit_observations=80, min_calibration_observations=25)
    left = fit_governed_covariance(returns, train_end=cutoff, config=cfg)
    mutated = returns.copy()
    mutated.loc[mutated.index > cutoff] = mutated.loc[mutated.index > cutoff] * 100.0 + 3.0
    right = fit_governed_covariance(mutated, train_end=cutoff, config=cfg)
    pd.testing.assert_frame_equal(left.covariance, right.covariance)
    assert left.report["returns_sha256"] == right.report["returns_sha256"]
    assert left.report["candidate_validation_loss"] == right.report["candidate_validation_loss"]


def test_missing_asset_or_short_window_fails_closed():
    returns = _returns(80)
    with pytest.raises(ValueError, match="missing assets"):
        fit_governed_covariance(returns, train_end=returns.index[-1], assets=["A", "ZZZ"])
    with pytest.raises(ValueError, match="complete observations"):
        fit_governed_covariance(
            returns,
            train_end=returns.index[20],
            config=CovarianceGovernanceConfig(method="sample", min_fit_observations=60),
        )


def test_robust_pareto_uses_governed_covariance_and_keeps_hard_constraints():
    returns = _returns()
    cutoff = returns.index[179]
    assets = list(returns.columns)
    result = allocate_robust_pareto_portfolio(
        alpha=pd.Series({"A": 0.030, "B": 0.025, "C": 0.020, "D": 0.015}),
        return_history=returns,
        train_end=cutoff,
        current_weights=pd.Series(0.0, index=assets),
        cost=pd.Series(0.0005, index=assets),
        adv20_cny=pd.Series(1_000_000_000.0, index=assets),
        constraints=PortfolioHardConstraints(
            target_book_cny=1_000_000.0,
            max_name_weight=0.50,
            max_sector_weight=1.0,
            max_style_exposure=1.0,
            max_turnover=2.0,
            max_adv_participation=0.10,
            min_cash_weight=0.0,
            max_gross_weight=1.0,
            min_names=1,
        ),
        covariance_config=CovarianceGovernanceConfig(
            method="auto",
            min_fit_observations=80,
            min_calibration_observations=25,
        ),
    )
    payload = result.to_dict()
    assert payload["covariance"]["train_end"] == cutoff.date().isoformat()
    assert payload["productionEligible"] is False
    selected = result.allocation.selected
    assert selected.feasible
    assert selected.weights.max() <= 0.50 + 1e-9
    assert selected.turnover <= 2.0 + 1e-9
    assert selected.min_capacity_multiple >= 1.0
