from __future__ import annotations

import numpy as np
import pandas as pd


def parkinson(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson 1980: range-based vol using high-low extremes."""
    log_hl = np.log(high / low) ** 2
    return np.sqrt(log_hl.rolling(window).mean() / (4.0 * np.log(2.0)))


def garman_klass(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Garman-Klass 1980 OHLC vol estimator."""
    rs = 0.5 * np.log(high / low) ** 2 - (2.0 * np.log(2.0) - 1.0) * np.log(close / open_) ** 2
    return np.sqrt(rs.rolling(window).mean().clip(lower=0.0))


def rogers_satchell(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rogers-Satchell 1991: drift-independent OHLC vol."""
    term = (
        np.log(high / close) * np.log(high / open_)
        + np.log(low / close) * np.log(low / open_)
    )
    return np.sqrt(term.rolling(window).mean().clip(lower=0.0))


def yang_zhang(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Yang-Zhang 2000: overnight + open-to-close + Rogers-Satchell with min variance."""
    log_co = np.log(open_ / close.shift(1))
    log_oc = np.log(close / open_)
    sigma_o2 = log_co.rolling(window).var(ddof=0)
    sigma_c2 = log_oc.rolling(window).var(ddof=0)
    sigma_rs2 = rogers_satchell(open_, high, low, close, window=window) ** 2
    k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))
    sigma2 = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs2
    return np.sqrt(sigma2.clip(lower=0.0))


def add_realized_vol_features(
    prices: pd.DataFrame,
    symbol_column: str = "symbol",
    window: int = 20,
) -> pd.DataFrame:
    """Append parkinson_xd, gk_xd, rs_xd, yz_xd per symbol."""
    data = prices.copy()
    data = data.sort_values([symbol_column]).reset_index(drop=True)
    grouped = data.groupby(symbol_column, group_keys=False)
    data[f"parkinson_{window}d"] = grouped.apply(
        lambda g: parkinson(g["high"], g["low"], window)
    ).reset_index(level=0, drop=True)
    data[f"gk_{window}d"] = grouped.apply(
        lambda g: garman_klass(g["open"], g["high"], g["low"], g["close"], window)
    ).reset_index(level=0, drop=True)
    data[f"rs_{window}d"] = grouped.apply(
        lambda g: rogers_satchell(g["open"], g["high"], g["low"], g["close"], window)
    ).reset_index(level=0, drop=True)
    data[f"yz_{window}d"] = grouped.apply(
        lambda g: yang_zhang(g["open"], g["high"], g["low"], g["close"], window)
    ).reset_index(level=0, drop=True)
    return data.replace([np.inf, -np.inf], np.nan)
