"""V5 regime-aware portfolio optimizer.

Wraps the V4 ``solve_v4_portfolio`` with two upgrades:
1. Risk aversion is scaled by the detected market regime (HMM or rule-based).
2. If the CVXPY solve fails and the universe is large enough, fall back to
   Hierarchical Risk Parity (HRP) rather than the simple score-based fallback.

The wrapper does not duplicate optimizer logic; it delegates to existing modules.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.quant_math.hrp import hrp_weights
from quantagent.quant_math.optimizer import (
    V4PortfolioConfig,
    V4PortfolioResult,
    solve_v4_portfolio,
)
from quantagent.quant_math.regime import MarketRegime


REGIME_RISK_AVERSION_MULTIPLIER: dict[str, float] = {
    MarketRegime.BULL_TREND.value: 0.85,
    MarketRegime.RANGE_BOUND.value: 1.0,
    MarketRegime.HIGH_VOLATILITY.value: 1.6,
    MarketRegime.BEAR_TREND.value: 1.5,
    MarketRegime.LIQUIDITY_CRISIS.value: 2.5,
}

REGIME_TURNOVER_MULTIPLIER: dict[str, float] = {
    MarketRegime.BULL_TREND.value: 1.0,
    MarketRegime.RANGE_BOUND.value: 1.0,
    MarketRegime.HIGH_VOLATILITY.value: 0.7,
    MarketRegime.BEAR_TREND.value: 0.6,
    MarketRegime.LIQUIDITY_CRISIS.value: 0.3,
}


@dataclass(frozen=True)
class RegimeAwareConfig:
    regime: str = MarketRegime.RANGE_BOUND.value
    base_config: V4PortfolioConfig | None = None
    hrp_fallback_min_assets: int = 8


def solve_v5_portfolio(
    alpha: pd.Series,
    covariance: pd.DataFrame,
    current_weights: pd.Series | None = None,
    cost: pd.Series | None = None,
    sector: pd.Series | None = None,
    beta: pd.Series | None = None,
    tradability: pd.DataFrame | None = None,
    config: RegimeAwareConfig | None = None,
    historical_returns: pd.DataFrame | None = None,
) -> V4PortfolioResult:
    """Regime-aware variant of solve_v4_portfolio with HRP fallback."""
    cfg = config or RegimeAwareConfig()
    base = cfg.base_config or V4PortfolioConfig()
    risk_mult = REGIME_RISK_AVERSION_MULTIPLIER.get(cfg.regime, 1.0)
    turnover_mult = REGIME_TURNOVER_MULTIPLIER.get(cfg.regime, 1.0)
    adjusted = V4PortfolioConfig(
        mode=base.mode,
        max_name_weight=base.max_name_weight,
        max_sector_weight=base.max_sector_weight,
        max_turnover=base.max_turnover * turnover_mult,
        target_beta=base.target_beta,
        beta_limit=base.beta_limit,
        cost_aware=base.cost_aware,
        no_buy_limit_up=base.no_buy_limit_up,
        no_sell_limit_down=base.no_sell_limit_down,
    )
    # Scale covariance to implicitly raise risk aversion. solve_v4_portfolio uses
    # a fixed risk_aversion=8 internally; multiplying the covariance preserves
    # the optimization objective shape and is solver-agnostic.
    scaled_cov = covariance * risk_mult
    result = solve_v4_portfolio(
        alpha=alpha,
        covariance=scaled_cov,
        current_weights=current_weights,
        cost=cost,
        sector=sector,
        beta=beta,
        tradability=tradability,
        config=adjusted,
    )
    if _should_use_hrp_fallback(result, historical_returns, cfg.hrp_fallback_min_assets):
        weights = hrp_weights(historical_returns)
        weights = weights.reindex(alpha.dropna().index).fillna(0.0)
        gross = weights.abs().sum()
        if gross > 0:
            weights = weights / gross
        diagnostics = dict(result.constraint_diagnostics)
        diagnostics["fallback"] = "hrp"
        diagnostics["regime"] = cfg.regime
        return V4PortfolioResult(
            target_weights=weights.sort_index(),
            expected_turnover=float((weights - (current_weights.reindex(weights.index).fillna(0.0) if current_weights is not None else 0.0)).abs().sum()),
            expected_cost=float(((weights - (current_weights.reindex(weights.index).fillna(0.0) if current_weights is not None else 0.0)).abs() * (cost.reindex(weights.index).fillna(0.0) if cost is not None else 0.0)).sum()),
            active_risk_proxy=float(np.sqrt(max(weights.to_numpy() @ covariance.reindex(weights.index, weights.index).fillna(0.0).to_numpy() @ weights.to_numpy(), 0.0))),
            constraint_diagnostics=diagnostics,
            rejected_symbols=result.rejected_symbols,
            status="hrp_fallback",
        )
    diagnostics = dict(result.constraint_diagnostics)
    diagnostics["regime"] = cfg.regime
    diagnostics["risk_multiplier"] = risk_mult
    diagnostics["turnover_multiplier"] = turnover_mult
    return V4PortfolioResult(
        target_weights=result.target_weights,
        expected_turnover=result.expected_turnover,
        expected_cost=result.expected_cost,
        active_risk_proxy=result.active_risk_proxy,
        constraint_diagnostics=diagnostics,
        rejected_symbols=result.rejected_symbols,
        status=result.status,
    )


def _should_use_hrp_fallback(
    result: V4PortfolioResult,
    historical_returns: pd.DataFrame | None,
    min_assets: int,
) -> bool:
    if historical_returns is None or historical_returns.empty:
        return False
    if historical_returns.shape[1] < min_assets:
        return False
    status = str(result.status).lower()
    return status.startswith("fallback") or status in {"infeasible", "infeasible_inaccurate", "unbounded"}
