"""Robust covariance -> hard-constrained Pareto portfolio bridge.

This is the governed Stage-5 boundary: alpha/rank research supplies expected
returns, a train-window return panel supplies risk, and the existing Pareto
allocator keeps name/sector/style/turnover/ADV/cash/gross constraints hard.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantagent.portfolio.covariance_governance import (
    CovarianceGovernanceConfig,
    CovarianceFitResult,
    fit_governed_covariance,
)
from quantagent.portfolio.pareto_allocator import (
    ParetoAllocationResult,
    ParetoSearchConfig,
    PortfolioHardConstraints,
    allocate_pareto_portfolio,
)


@dataclass(frozen=True)
class RobustParetoAllocationResult:
    allocation: ParetoAllocationResult
    covariance: CovarianceFitResult

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "quantagent.portfolio.robust_pareto.v1",
            "researchOnly": True,
            "productionEligible": False,
            "covariance": self.covariance.report,
            "selected": self.allocation.selected.to_dict(),
            "frontier": [item.to_dict() for item in self.allocation.frontier],
            "rejected": [item.to_dict() for item in self.allocation.rejected],
            "gross_budget": self.allocation.gross_budget,
        }


def allocate_robust_pareto_portfolio(
    *,
    alpha: pd.Series,
    return_history: pd.DataFrame,
    train_end: object,
    current_weights: pd.Series | None = None,
    cost: pd.Series | None = None,
    sector: pd.Series | None = None,
    style_exposures: pd.DataFrame | None = None,
    adv20_cny: pd.Series | None = None,
    regime: str = "normal",
    blend_confidence: float = 0.5,
    breadth_score: float = 0.5,
    liquidity_score: float = 0.5,
    constraints: PortfolioHardConstraints | None = None,
    search: ParetoSearchConfig | None = None,
    covariance_config: CovarianceGovernanceConfig | None = None,
) -> RobustParetoAllocationResult:
    clean_alpha = pd.to_numeric(alpha, errors="coerce").dropna().astype(float)
    if clean_alpha.empty:
        raise ValueError("alpha is empty")
    covariance = fit_governed_covariance(
        return_history,
        train_end=train_end,
        assets=clean_alpha.index.astype(str),
        config=covariance_config,
    )
    allocation = allocate_pareto_portfolio(
        alpha=clean_alpha,
        covariance=covariance.covariance,
        current_weights=current_weights,
        cost=cost,
        sector=sector,
        style_exposures=style_exposures,
        adv20_cny=adv20_cny,
        regime=regime,
        blend_confidence=blend_confidence,
        breadth_score=breadth_score,
        liquidity_score=liquidity_score,
        constraints=constraints,
        search=search,
    )
    return RobustParetoAllocationResult(allocation=allocation, covariance=covariance)


__all__ = ["RobustParetoAllocationResult", "allocate_robust_pareto_portfolio"]
