from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.quant_math.risk_metrics import drawdown


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns.dropna() - risk_free_rate / periods_per_year
    if excess.empty or excess.std(ddof=1) == 0:
        return np.nan
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1))


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
