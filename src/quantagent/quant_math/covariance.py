from __future__ import annotations

import numpy as np
import pandas as pd


def sample_covariance(returns: pd.DataFrame, annualize: bool = False, periods_per_year: int = 252) -> pd.DataFrame:
    cov = returns.cov()
    return cov * periods_per_year if annualize else cov


def shrinkage_covariance(
    returns: pd.DataFrame,
    shrinkage: float = 0.1,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Shrink sample covariance toward a diagonal target."""
    sample = returns.cov()
    target = pd.DataFrame(np.diag(np.diag(sample)), index=sample.index, columns=sample.columns)
    cov = (1.0 - shrinkage) * sample + shrinkage * target
    return cov * periods_per_year if annualize else cov


def ewma_covariance(
    returns: pd.DataFrame,
    span: int = 60,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(index=returns.columns, columns=returns.columns, dtype=float)
    alpha = 2.0 / (span + 1.0)
    ages = np.arange(len(returns) - 1, -1, -1)
    weights = alpha * np.power(1.0 - alpha, ages)
    weights = weights / weights.sum()
    demeaned = returns - returns.mean()
    cov = demeaned.mul(weights, axis=0).T @ demeaned
    cov = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    return cov * periods_per_year if annualize else cov
