from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Lopez de Prado 2016 distance: sqrt(0.5 * (1 - corr))."""
    return np.sqrt(0.5 * (1.0 - corr.clip(-1.0, 1.0)))


def quasi_diagonalization(link: np.ndarray) -> list[int]:
    """Reorder leaves so similar assets sit adjacent in the covariance matrix."""
    link = link.astype(int)
    n = link.shape[0] + 1
    order = [int(link[-1, 0]), int(link[-1, 1])]
    while max(order) >= n:
        new_order: list[int] = []
        for item in order:
            if item < n:
                new_order.append(item)
            else:
                row = link[item - n]
                new_order.extend([int(row[0]), int(row[1])])
        order = new_order
    return order


def _cluster_variance(cov: np.ndarray, items: np.ndarray) -> float:
    sub = cov[np.ix_(items, items)]
    inv_var = 1.0 / np.diag(sub)
    weights = inv_var / inv_var.sum()
    return float(weights @ sub @ weights)


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity weights (Lopez de Prado 2016)."""
    if returns.shape[1] == 0:
        return pd.Series(dtype=float)
    cov = returns.cov().to_numpy(copy=True)
    corr = returns.corr().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1.0)
    distance = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(distance, 0.0)
    link = linkage(squareform(distance, checks=False), method="single")
    sort_idx = quasi_diagonalization(link)
    n = cov.shape[0]
    weights = np.ones(n)
    clusters = [np.array(sort_idx, dtype=int)]
    while clusters:
        next_clusters: list[np.ndarray] = []
        for cluster in clusters:
            if cluster.size <= 1:
                continue
            mid = cluster.size // 2
            left, right = cluster[:mid], cluster[mid:]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alloc = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alloc
            weights[right] *= 1.0 - alloc
            next_clusters.append(left)
            next_clusters.append(right)
        clusters = next_clusters
    weights = weights / weights.sum()
    return pd.Series(weights, index=returns.columns)


def herc_weights(
    returns: pd.DataFrame,
    n_clusters: int | None = None,
    risk_measure: str = "vol",
) -> pd.Series:
    """Hierarchical Equal Risk Contribution (Raffinot 2018) with vol or CVaR."""
    if returns.shape[1] == 0:
        return pd.Series(dtype=float)
    corr = returns.corr().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1.0)
    distance = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(distance, 0.0)
    link = linkage(squareform(distance, checks=False), method="ward")
    n = returns.shape[1]
    k = n_clusters or max(2, int(np.sqrt(n)))
    labels = fcluster(link, t=k, criterion="maxclust")
    risks = _asset_risk(returns, risk_measure)
    weights = np.zeros(n)
    cluster_risks = np.zeros(k)
    for c in range(1, k + 1):
        members = np.where(labels == c)[0]
        if members.size == 0:
            continue
        sub_risk = risks[members]
        inv = 1.0 / sub_risk
        local = inv / inv.sum()
        weights[members] = local
        cluster_risks[c - 1] = float(sub_risk.mean())
    inv_cluster = 1.0 / np.where(cluster_risks > 0, cluster_risks, np.nan)
    cluster_alloc = np.nan_to_num(inv_cluster) / np.nansum(inv_cluster)
    for c in range(1, k + 1):
        members = np.where(labels == c)[0]
        weights[members] *= cluster_alloc[c - 1]
    weights = weights / weights.sum()
    return pd.Series(weights, index=returns.columns)


def _asset_risk(returns: pd.DataFrame, risk_measure: str) -> np.ndarray:
    if risk_measure == "vol":
        return returns.std(ddof=1).fillna(returns.std(ddof=1).mean()).to_numpy()
    if risk_measure == "cvar":
        var = returns.quantile(0.05)
        cvar = returns.where(returns.le(var, axis=1)).mean()
        return cvar.abs().fillna(cvar.abs().mean()).to_numpy()
    raise ValueError(f"Unsupported risk_measure: {risk_measure}")
