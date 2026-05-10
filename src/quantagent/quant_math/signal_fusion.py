from __future__ import annotations

import numpy as np
import pandas as pd


def precision_weighted_alpha(
    predictions: pd.DataFrame,
    symbol_column: str = "symbol",
    alpha_column: str = "alpha",
    variance_column: str = "error_variance",
    ic_column: str | None = "rank_ic",
    min_ic: float = 0.0,
) -> pd.Series:
    """Fuse model alphas by inverse error variance and optional positive IC."""
    required = {symbol_column, alpha_column, variance_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    data = predictions.copy()
    data["precision"] = 1.0 / data[variance_column].clip(lower=1e-12)
    if ic_column and ic_column in data.columns:
        data["edge"] = data[ic_column].clip(lower=min_ic)
    else:
        data["edge"] = 1.0
    data["raw_weight"] = data["precision"] * data["edge"]

    def _fuse(group: pd.DataFrame) -> float:
        raw = group["raw_weight"]
        if raw.sum() <= 0:
            weight = pd.Series(1.0 / len(group), index=group.index)
        else:
            weight = raw / raw.sum()
        return float((weight * group[alpha_column]).sum())

    return data.groupby(symbol_column).apply(_fuse, include_groups=False)


def ensemble_confidence(alpha_predictions: pd.DataFrame, symbol_column: str = "symbol") -> pd.Series:
    """Convert ensemble disagreement into a 0-1 confidence score."""
    dispersion = alpha_predictions.groupby(symbol_column)["alpha"].std().fillna(0.0)
    return 1.0 / (1.0 + dispersion)


def black_litterman_posterior(
    prior_returns: pd.Series,
    covariance: pd.DataFrame,
    pick_matrix: np.ndarray,
    views: np.ndarray,
    view_uncertainty: np.ndarray,
    tau: float = 0.05,
) -> pd.Series:
    """Black-Litterman posterior expected returns.

    prior_returns are the equilibrium returns pi. Views follow P mu = q + eps,
    where eps covariance is Omega.
    """
    symbols = prior_returns.index
    sigma = covariance.reindex(index=symbols, columns=symbols).fillna(0.0).to_numpy()
    pi = prior_returns.to_numpy()
    p = np.asarray(pick_matrix, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    q = np.asarray(views, dtype=float)
    omega = np.asarray(view_uncertainty, dtype=float)
    if omega.ndim == 1:
        omega = np.diag(omega)

    tau_sigma_inv = np.linalg.pinv(tau * sigma)
    omega_inv = np.linalg.pinv(omega)
    posterior_cov_inv = tau_sigma_inv + p.T @ omega_inv @ p
    posterior_mean = np.linalg.pinv(posterior_cov_inv) @ (tau_sigma_inv @ pi + p.T @ omega_inv @ q)
    return pd.Series(posterior_mean, index=symbols)


def blend_alpha_and_views(
    model_alpha: pd.Series,
    conformal_low: pd.Series | None = None,
    conformal_high: pd.Series | None = None,
    factor_gate_confidence: pd.Series | None = None,
    agent_posterior: pd.Series | None = None,
    regime_multiplier: float | pd.Series = 1.0,
    risk_confidence: pd.Series | None = None,
) -> pd.DataFrame:
    symbols = model_alpha.index
    alpha = model_alpha.astype(float).copy()
    if agent_posterior is not None:
        alpha = 0.7 * alpha + 0.3 * agent_posterior.reindex(symbols).fillna(alpha)
    if conformal_low is not None and conformal_high is not None:
        width = (conformal_high.reindex(symbols) - conformal_low.reindex(symbols)).abs()
        interval_confidence = 1.0 / (1.0 + width.fillna(width.median() if width.notna().any() else 0.0))
    else:
        interval_confidence = pd.Series(1.0, index=symbols)
    gate = factor_gate_confidence.reindex(symbols).fillna(1.0) if factor_gate_confidence is not None else pd.Series(1.0, index=symbols)
    risk = risk_confidence.reindex(symbols).fillna(1.0) if risk_confidence is not None else pd.Series(1.0, index=symbols)
    regime = regime_multiplier.reindex(symbols).fillna(1.0) if isinstance(regime_multiplier, pd.Series) else pd.Series(float(regime_multiplier), index=symbols)
    confidence = (interval_confidence * gate * risk).clip(0.0, 1.0)
    blended = alpha * confidence * regime
    return pd.DataFrame(
        {
            "symbol": symbols,
            "model_alpha": model_alpha.to_numpy(dtype=float),
            "blended_alpha": blended.to_numpy(dtype=float),
            "confidence": confidence.to_numpy(dtype=float),
            "regime_multiplier": regime.to_numpy(dtype=float),
        }
    ).sort_values("symbol").reset_index(drop=True)
