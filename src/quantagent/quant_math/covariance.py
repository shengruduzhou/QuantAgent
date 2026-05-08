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


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Ledoit-Wolf 2004 optimal linear shrinkage to constant-correlation target."""
    clean = returns.dropna(how="any")
    n, p = clean.shape
    if n < 2 or p < 2:
        return pd.DataFrame(np.eye(p), index=returns.columns, columns=returns.columns)
    x = clean.to_numpy() - clean.mean(axis=0).to_numpy()
    sample = (x.T @ x) / n
    var = np.diag(sample)
    std = np.sqrt(var)
    corr = sample / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    avg_corr = (corr.sum() - p) / (p * (p - 1))
    target = avg_corr * np.outer(std, std)
    np.fill_diagonal(target, var)
    y = x ** 2
    phi_mat = (y.T @ y) / n - sample ** 2
    phi = float(phi_mat.sum())
    rho_diag = float(np.diag(phi_mat).sum())
    term = (x ** 3).T @ x / n - var[:, None] * sample
    rho_off = (avg_corr * (np.outer(1.0 / std, std)) * term).sum() - (
        avg_corr * np.diag(np.outer(1.0 / std, std) * term)
    ).sum()
    rho = rho_diag + rho_off
    gamma = float(np.linalg.norm(target - sample, "fro") ** 2)
    if gamma <= 0:
        shrink = 0.0
    else:
        kappa = (phi - rho) / gamma
        shrink = float(max(0.0, min(1.0, kappa / n)))
    cov = shrink * target + (1.0 - shrink) * sample
    out = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    return out * periods_per_year if annualize else out
