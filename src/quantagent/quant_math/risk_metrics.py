from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_volatility(weights: pd.Series, covariance: pd.DataFrame) -> float:
    aligned_cov = covariance.loc[weights.index, weights.index]
    value = float(weights.to_numpy() @ aligned_cov.to_numpy() @ weights.to_numpy())
    return float(np.sqrt(max(value, 0.0)))


def historical_var(returns: pd.Series, alpha: float = 0.95) -> float:
    losses = -returns.dropna()
    if losses.empty:
        return np.nan
    return float(losses.quantile(alpha))


def historical_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    losses = -returns.dropna()
    if losses.empty:
        return np.nan
    var = losses.quantile(alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if not tail.empty else float(var)


def parametric_var(weights: pd.Series, covariance: pd.DataFrame, z_score: float = 1.65) -> float:
    return z_score * portfolio_volatility(weights, covariance)


def drawdown(nav: pd.Series) -> pd.Series:
    peak = nav.cummax()
    return nav / peak - 1.0


def drawdown_risk_multiplier(
    current_drawdown: float,
    kill_switch: float = -0.15,
    half_life: float = 0.06,
) -> float:
    """Continuous gauss-style decay: 1.0 at peak -> 0.0 at kill_switch."""
    if current_drawdown <= kill_switch:
        return 0.0
    if current_drawdown >= 0.0:
        return 1.0
    decay = float(np.exp(-(current_drawdown / half_life) ** 2))
    return max(0.0, min(1.0, decay))
