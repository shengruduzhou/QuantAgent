from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def gram_schmidt_orthogonalize(
    new_factor: pd.Series,
    existing_factors: pd.DataFrame,
) -> pd.Series:
    """Project new factor on the orthogonal complement of existing_factors."""
    aligned = pd.concat([new_factor, existing_factors], axis=1).dropna()
    if aligned.empty or aligned.shape[1] < 2:
        return new_factor
    y = aligned.iloc[:, 0].to_numpy()
    x = aligned.iloc[:, 1:].to_numpy()
    if np.linalg.matrix_rank(x) == 0:
        return new_factor
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    out = new_factor.copy()
    out.loc[aligned.index] = residual
    return out


def cross_sectional_factor_returns(
    returns: pd.Series,
    factor_loadings: pd.DataFrame,
) -> pd.Series:
    """OLS factor return f from r = X f + epsilon."""
    aligned = pd.concat([returns, factor_loadings], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float, index=factor_loadings.columns)
    y = aligned.iloc[:, 0].to_numpy()
    x = aligned.iloc[:, 1:].to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return pd.Series(beta, index=factor_loadings.columns)


@dataclass(frozen=True)
class AttributionResult:
    factor_returns: pd.Series
    factor_contributions: pd.Series
    specific_return: float
    r_squared: float


def portfolio_factor_attribution(
    weights: pd.Series,
    realized_returns: pd.Series,
    factor_loadings: pd.DataFrame,
) -> AttributionResult:
    """Decompose portfolio return into factor + specific contributions."""
    aligned = pd.concat([realized_returns, factor_loadings], axis=1).dropna()
    aligned = aligned.loc[aligned.index.intersection(weights.index)]
    if aligned.empty:
        return AttributionResult(
            factor_returns=pd.Series(dtype=float),
            factor_contributions=pd.Series(dtype=float),
            specific_return=float("nan"),
            r_squared=float("nan"),
        )
    y = aligned.iloc[:, 0].to_numpy()
    x = aligned.iloc[:, 1:].to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    factor_returns = pd.Series(beta, index=factor_loadings.columns)
    portfolio_loadings = (
        weights.reindex(aligned.index).fillna(0.0).to_numpy()[:, None]
        * x
    ).sum(axis=0)
    factor_contributions = pd.Series(portfolio_loadings * beta, index=factor_loadings.columns)
    portfolio_return = float(np.dot(weights.reindex(aligned.index).fillna(0.0).to_numpy(), y))
    specific = portfolio_return - factor_contributions.sum()
    return AttributionResult(
        factor_returns=factor_returns,
        factor_contributions=factor_contributions,
        specific_return=float(specific),
        r_squared=float(r2),
    )


def capacity_curve(
    daily_alpha: pd.Series,
    adv: pd.Series,
    impact_coefficient: float = 0.001,
    impact_exponent: float = 0.5,
    aum_grid: tuple[float, ...] = (1e6, 1e7, 1e8, 1e9, 1e10),
) -> pd.DataFrame:
    """Net annual alpha at each AUM after subtracting market-impact cost."""
    rows = []
    for aum in aum_grid:
        participation = (aum * abs(daily_alpha) / adv).clip(lower=0.0)
        impact_bps = 10000.0 * impact_coefficient * participation.pow(impact_exponent)
        impact_return = impact_bps / 10000.0
        net = daily_alpha - impact_return
        rows.append(
            {
                "aum_rmb": aum,
                "gross_annual": float(daily_alpha.mean() * 252.0),
                "impact_annual": float(impact_return.mean() * 252.0),
                "net_annual": float(net.mean() * 252.0),
            }
        )
    return pd.DataFrame(rows)
