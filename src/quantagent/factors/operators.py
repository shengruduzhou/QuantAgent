from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats


def delay(frame: pd.DataFrame, column: str, periods: int = 1) -> pd.Series:
    data = _sorted(frame)
    values = data.groupby("symbol", sort=False)[column].shift(periods)
    return _restore(values, data)


def delta(frame: pd.DataFrame, column: str, periods: int = 1) -> pd.Series:
    data = _sorted(frame)
    values = data[column].astype(float) - data.groupby("symbol", sort=False)[column].shift(periods)
    return _restore(values, data)


def rank(frame: pd.DataFrame, column: str, pct: bool = True) -> pd.Series:
    data = _sorted(frame)
    values = data.groupby("trade_date", sort=False)[column].rank(method="average", pct=pct)
    return _restore(values, data)


def ts_rank(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    data = _sorted(frame)
    values = _rolling_apply(
        data,
        column,
        window,
        lambda x: pd.Series(x).rank(method="average").iloc[-1] / len(x),
    )
    return _restore(values, data)


def ts_argmax(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    data = _sorted(frame)
    values = _rolling_apply(data, column, window, lambda x: float(np.argmax(x) + 1))
    return _restore(values, data)


def ts_argmin(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    data = _sorted(frame)
    values = _rolling_apply(data, column, window, lambda x: float(np.argmin(x) + 1))
    return _restore(values, data)


def ts_min(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    data = _sorted(frame)
    values = data.groupby("symbol", sort=False)[column].rolling(window, min_periods=window).min()
    return _restore(values.reset_index(level=0, drop=True), data)


def ts_max(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    data = _sorted(frame)
    values = data.groupby("symbol", sort=False)[column].rolling(window, min_periods=window).max()
    return _restore(values.reset_index(level=0, drop=True), data)


def ts_sum(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    data = _sorted(frame)
    values = data.groupby("symbol", sort=False)[column].rolling(window, min_periods=window).sum()
    return _restore(values.reset_index(level=0, drop=True), data)


def ts_mean(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    data = _sorted(frame)
    values = data.groupby("symbol", sort=False)[column].rolling(window, min_periods=window).mean()
    return _restore(values.reset_index(level=0, drop=True), data)


def ts_std(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    data = _sorted(frame)
    values = data.groupby("symbol", sort=False)[column].rolling(window, min_periods=window).std()
    return _restore(values.reset_index(level=0, drop=True), data)


def ts_corr(frame: pd.DataFrame, left: str, right: str, window: int) -> pd.Series:
    data = _sorted(frame)
    values = pd.Series(np.nan, index=data.index, dtype=float)
    for _, group in data.groupby("symbol", sort=False):
        values.loc[group.index] = group[left].rolling(window, min_periods=window).corr(group[right])
    return _restore(values, data)


def ts_cov(frame: pd.DataFrame, left: str, right: str, window: int) -> pd.Series:
    data = _sorted(frame)
    values = pd.Series(np.nan, index=data.index, dtype=float)
    for _, group in data.groupby("symbol", sort=False):
        values.loc[group.index] = group[left].rolling(window, min_periods=window).cov(group[right])
    return _restore(values, data)


def decay_linear(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    data = _sorted(frame)
    weights = np.arange(1.0, window + 1.0)
    weights = weights / weights.sum()
    values = _rolling_apply(data, column, window, lambda x: float(np.dot(x, weights)))
    return _restore(values, data)


def scale(frame: pd.DataFrame, column: str, k: float = 1.0) -> pd.Series:
    data = _sorted(frame)
    denom = data.groupby("trade_date", sort=False)[column].transform(lambda s: s.abs().sum())
    values = data[column].astype(float) * k / denom.replace(0.0, np.nan)
    return _restore(values, data)


def signed_power(frame: pd.DataFrame, column: str, power: float) -> pd.Series:
    data = _sorted(frame)
    values = np.sign(data[column].astype(float)) * np.abs(data[column].astype(float)) ** power
    return _restore(pd.Series(values, index=data.index), data)


def industry_neutralize(frame: pd.DataFrame, column: str, industry_column: str = "industry") -> pd.Series:
    data = _sorted(frame)
    group_mean = data.groupby(["trade_date", industry_column], sort=False)[column].transform("mean")
    values = data[column].astype(float) - group_mean
    return _restore(values, data)


def winsorize_mad(frame: pd.DataFrame, column: str, n_mad: float = 5.0) -> pd.Series:
    data = _sorted(frame)

    def _clip(s: pd.Series) -> pd.Series:
        median = s.median()
        mad = np.median(np.abs(s - median))
        if not np.isfinite(mad) or mad <= 1e-12:
            return s.astype(float)
        lower = median - n_mad * 1.4826 * mad
        upper = median + n_mad * 1.4826 * mad
        return s.clip(lower, upper)

    values = data.groupby("trade_date", sort=False)[column].transform(_clip)
    return _restore(values, data)


def zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    data = _sorted(frame)

    def _z(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (s - s.mean()) / std

    values = data.groupby("trade_date", sort=False)[column].transform(_z)
    return _restore(values, data)


def group_zscore(frame: pd.DataFrame, column: str, group_column: str) -> pd.Series:
    data = _sorted(frame)

    def _z(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (s - s.mean()) / std

    values = data.groupby(["trade_date", group_column], sort=False)[column].transform(_z)
    return _restore(values, data)


def rank_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    data = _sorted(frame)

    def _rank_to_z(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        out = pd.Series(np.nan, index=s.index, dtype=float)
        n = len(valid)
        if n < 2:
            return out
        p = (valid.rank(method="average") - 0.5) / n
        out.loc[valid.index] = stats.norm.ppf(p.clip(1e-6, 1.0 - 1e-6))
        return out

    values = data.groupby("trade_date", sort=False)[column].transform(_rank_to_z)
    return _restore(values, data)


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "symbol"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    if "_original_index" in data.columns:
        data["_qa_restore_index"] = data["_original_index"]
    else:
        data["_qa_restore_index"] = data.index
    return data.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _restore(values: pd.Series, data: pd.DataFrame) -> pd.Series:
    restored = pd.Series(values.to_numpy(dtype=float), index=data["_qa_restore_index"].to_numpy())
    return restored.sort_index()


def _rolling_apply(
    data: pd.DataFrame,
    column: str,
    window: int,
    func: Callable[[np.ndarray], float],
) -> pd.Series:
    values = pd.Series(np.nan, index=data.index, dtype=float)
    for _, group in data.groupby("symbol", sort=False):
        values.loc[group.index] = group[column].rolling(window, min_periods=window).apply(func, raw=True)
    return values
