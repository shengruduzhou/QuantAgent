from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantagent.quant_math.risk_metrics import drawdown


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns.dropna() - risk_free_rate / periods_per_year
    if excess.empty or excess.std(ddof=1) == 0:
        return np.nan
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1))


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    sr_benchmark: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Bailey & Lopez de Prado 2012 PSR adjusted for skew and kurtosis."""
    clean = returns.dropna()
    n = len(clean)
    if n < 4:
        return np.nan
    std = clean.std(ddof=1)
    if std <= 1e-12:
        mean = float(clean.mean())
        return 1.0 if mean > sr_benchmark / periods_per_year else 0.0
    sr = sharpe_ratio(clean, periods_per_year=periods_per_year) / np.sqrt(periods_per_year)
    with np.errstate(invalid="ignore"):
        skew_raw = stats.skew(clean, bias=False)
        kurt_raw = stats.kurtosis(clean, fisher=True, bias=False)
    skew = 0.0 if not np.isfinite(skew_raw) else float(skew_raw)
    kurt = 0.0 if not np.isfinite(kurt_raw) else float(kurt_raw)
    sr_b = sr_benchmark / np.sqrt(periods_per_year)
    denom = np.sqrt(max(1.0 - skew * sr + kurt / 4.0 * sr ** 2, 1e-12))
    z = (sr - sr_b) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series,
    candidate_sharpes: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Deflated SR: PSR threshold inflated by max-of-N selection bias."""
    clean = returns.dropna()
    n = len(clean)
    if n < 4 or candidate_sharpes.size < 2:
        return np.nan
    var_sr = float(np.var(candidate_sharpes, ddof=1))
    if var_sr <= 0:
        return np.nan
    n_trials = candidate_sharpes.size
    euler_mascheroni = 0.5772156649
    expected_max = np.sqrt(var_sr) * (
        (1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + euler_mascheroni * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    )
    return probabilistic_sharpe_ratio(
        clean,
        sr_benchmark=float(expected_max) * np.sqrt(periods_per_year),
        periods_per_year=periods_per_year,
    )


def newey_west_t_stat(series: pd.Series, max_lag: int | None = None) -> float:
    """Newey-West HAC t-stat for the mean of an autocorrelated series."""
    clean = series.dropna().to_numpy()
    n = len(clean)
    if n < 2:
        return np.nan
    if max_lag is None:
        max_lag = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    mean = clean.mean()
    centered = clean - mean
    gamma0 = float(np.dot(centered, centered) / n)
    nw_var = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        nw_var += 2.0 * weight * gamma
    nw_var = max(nw_var, 1e-12)
    return float(mean / np.sqrt(nw_var / n))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns.dropna() - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if excess.empty or downside.std(ddof=1) == 0:
        return np.nan
    return float(np.sqrt(periods_per_year) * excess.mean() / downside.std(ddof=1))


def max_drawdown(nav: pd.Series) -> float:
    dd = drawdown(nav.dropna())
    return float(dd.min()) if not dd.empty else np.nan


def calmar_ratio(nav: pd.Series, periods_per_year: int = 252) -> float:
    clean = nav.dropna()
    if len(clean) < 2:
        return np.nan
    years = len(clean) / periods_per_year
    cagr = (clean.iloc[-1] / clean.iloc[0]) ** (1.0 / years) - 1.0
    max_dd = abs(max_drawdown(clean))
    return float(cagr / max_dd) if max_dd > 0 else np.nan


def hit_ratio(returns: pd.Series) -> float:
    clean = returns.dropna()
    return float((clean > 0).mean()) if not clean.empty else np.nan


def profit_factor(returns: pd.Series) -> float:
    clean = returns.dropna()
    gains = clean[clean > 0].sum()
    losses = -clean[clean < 0].sum()
    return float(gains / losses) if losses > 0 else np.nan


def turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.fillna(0.0).diff().abs().sum(axis=1)
