from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from quantagent.quant_math.performance import newey_west_t_stat


@dataclass(frozen=True)
class ICSummary:
    mean_ic: float
    mean_rank_ic: float
    icir: float
    rank_icir: float
    t_stat: float
    rank_t_stat: float
    positive_ratio: float


@dataclass(frozen=True)
class ICResult:
    ic_by_date: pd.Series
    rank_ic_by_date: pd.Series
    summary: ICSummary


@dataclass(frozen=True)
class DecayResult:
    horizon_days: tuple[int, ...]
    rank_ic: pd.Series
    ic: pd.Series


@dataclass(frozen=True)
class QuantileBacktestResult:
    group_returns: pd.DataFrame
    long_short: pd.Series
    monotonicity: float
    turnover: pd.Series
    cost_adjusted_long_short: pd.Series


@dataclass(frozen=True)
class CapacityResult:
    average_amount: float
    top_quantile_amount: float
    capacity_rmb: float
    participation_rate: float


@dataclass(frozen=True)
class FactorSummary:
    factor_name: str
    horizon_days: int
    ic: float
    rank_ic: float
    icir: float
    rank_icir: float
    monotonicity: float
    turnover: float
    capacity_rmb: float


def forward_return_labels(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20),
    price_column: str = "close",
) -> pd.DataFrame:
    data = frame.copy().sort_values(["symbol", "trade_date"])
    for horizon in horizons:
        future = data.groupby("symbol", sort=False)[price_column].shift(-horizon)
        data[f"forward_return_{horizon}d"] = future / data[price_column] - 1.0
    return data.sort_index()


def excess_forward_returns(
    frame: pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20),
    date_column: str = "trade_date",
) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(benchmark_returns, pd.DataFrame):
        benchmark = benchmark_returns.set_index(date_column)["return"]
    else:
        benchmark = benchmark_returns
    benchmark.index = pd.to_datetime(benchmark.index)
    dates = pd.to_datetime(data[date_column])
    for horizon in horizons:
        column = f"forward_return_{horizon}d"
        if column not in data.columns:
            continue
        rolling_benchmark = (1.0 + benchmark).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True) - 1.0
        aligned = dates.map(rolling_benchmark.shift(-horizon + 1))
        data[f"excess_{column}"] = data[column] - aligned.to_numpy(dtype=float)
    return data


def information_coefficient(
    frame: pd.DataFrame,
    factor_column: str,
    return_column: str,
    date_column: str = "trade_date",
) -> ICResult:
    ic = _corr_by_date(frame, factor_column, return_column, date_column, method="pearson")
    rank_ic = _corr_by_date(frame, factor_column, return_column, date_column, method="spearman")
    summary = ICSummary(
        mean_ic=_safe_mean(ic),
        mean_rank_ic=_safe_mean(rank_ic),
        icir=_icir(ic),
        rank_icir=_icir(rank_ic),
        t_stat=newey_west_t_stat(ic),
        rank_t_stat=newey_west_t_stat(rank_ic),
        positive_ratio=float((rank_ic.dropna() > 0).mean()) if rank_ic.dropna().shape[0] else np.nan,
    )
    return ICResult(ic_by_date=ic, rank_ic_by_date=rank_ic, summary=summary)


def factor_decay_curve(
    frame: pd.DataFrame,
    factor_column: str,
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20),
    price_column: str = "close",
) -> DecayResult:
    labeled = forward_return_labels(frame, horizons=horizons, price_column=price_column)
    ic_values: dict[int, float] = {}
    rank_values: dict[int, float] = {}
    for horizon in horizons:
        result = information_coefficient(labeled, factor_column, f"forward_return_{horizon}d")
        ic_values[horizon] = result.summary.mean_ic
        rank_values[horizon] = result.summary.mean_rank_ic
    return DecayResult(
        horizon_days=horizons,
        rank_ic=pd.Series(rank_values, dtype=float),
        ic=pd.Series(ic_values, dtype=float),
    )


def quantile_group_backtest(
    frame: pd.DataFrame,
    factor_column: str,
    return_column: str,
    quantiles: int = 5,
    cost_bps: float = 0.0,
    date_column: str = "trade_date",
) -> QuantileBacktestResult:
    data = frame[[date_column, "symbol", factor_column, return_column]].copy()
    data["quantile"] = data.groupby(date_column, sort=False)[factor_column].transform(
        lambda s: _quantile_labels(s, quantiles)
    )
    group_returns = data.pivot_table(
        index=date_column,
        columns="quantile",
        values=return_column,
        aggfunc="mean",
    ).sort_index()
    if 1 in group_returns.columns and quantiles in group_returns.columns:
        long_short = group_returns[quantiles] - group_returns[1]
    else:
        long_short = pd.Series(np.nan, index=group_returns.index, dtype=float)
    turnover_series = quantile_turnover(data, quantile_column="quantile", top_quantile=quantiles)
    cost = turnover_series.reindex(long_short.index).fillna(0.0) * cost_bps / 10000.0
    return QuantileBacktestResult(
        group_returns=group_returns,
        long_short=long_short,
        monotonicity=monotonicity_score(group_returns),
        turnover=turnover_series,
        cost_adjusted_long_short=long_short - cost,
    )


def long_short_spread(group_returns: pd.DataFrame) -> pd.Series:
    columns = sorted(group_returns.columns)
    if not columns:
        return pd.Series(dtype=float)
    return group_returns[columns[-1]] - group_returns[columns[0]]


def quantile_turnover(
    frame: pd.DataFrame,
    date_column: str = "trade_date",
    symbol_column: str = "symbol",
    quantile_column: str = "quantile",
    top_quantile: int | None = None,
) -> pd.Series:
    data = frame.sort_values([date_column, symbol_column])
    if top_quantile is None:
        top_quantile = int(data[quantile_column].max())
    dates = list(data[date_column].drop_duplicates())
    values: dict[pd.Timestamp, float] = {}
    previous: set[str] | None = None
    for date in dates:
        current = set(data.loc[(data[date_column] == date) & (data[quantile_column] == top_quantile), symbol_column])
        if previous is None or not current:
            values[pd.Timestamp(date)] = 0.0
        else:
            values[pd.Timestamp(date)] = 1.0 - len(current & previous) / max(len(current), 1)
        previous = current
    return pd.Series(values, dtype=float)


def capacity_proxy(
    frame: pd.DataFrame,
    factor_column: str,
    amount_column: str = "amount",
    participation_rate: float = 0.05,
    quantile: float = 0.8,
) -> CapacityResult:
    clean = frame[[factor_column, amount_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return CapacityResult(np.nan, np.nan, np.nan, participation_rate)
    cutoff = clean[factor_column].quantile(quantile)
    top = clean[clean[factor_column] >= cutoff]
    top_amount = float(top[amount_column].mean()) if not top.empty else np.nan
    avg_amount = float(clean[amount_column].mean())
    capacity = top_amount * participation_rate if np.isfinite(top_amount) else np.nan
    return CapacityResult(avg_amount, top_amount, float(capacity), participation_rate)


def monotonicity_score(group_returns: pd.DataFrame) -> float:
    if group_returns.empty or group_returns.shape[1] < 2:
        return np.nan
    means = group_returns.mean(axis=0).dropna()
    if len(means) < 2 or means.nunique() <= 1:
        return np.nan
    order = pd.Series(np.arange(1, len(means) + 1), index=means.index, dtype=float)
    return float(order.corr(means, method="spearman"))


def factor_correlation_matrix(
    frame: pd.DataFrame,
    factor_columns: list[str] | None = None,
    method: Literal["pearson", "spearman"] = "pearson",
) -> pd.DataFrame:
    if factor_columns is not None:
        matrix = frame[factor_columns]
    elif {"trade_date", "symbol", "factor_name", "factor_value"}.issubset(frame.columns):
        matrix = frame.pivot_table(
            index=["trade_date", "symbol"],
            columns="factor_name",
            values="factor_value",
            aggfunc="last",
        )
    else:
        numeric = frame.select_dtypes(include=[np.number])
        matrix = numeric
    return matrix.corr(method=method)


def factor_summary_table(
    frame: pd.DataFrame,
    factor_columns: list[str],
    return_column: str,
    amount_column: str = "amount",
    quantiles: int = 5,
    horizon_days: int = 1,
) -> pd.DataFrame:
    rows: list[FactorSummary] = []
    for column in factor_columns:
        ic = information_coefficient(frame, column, return_column)
        groups = quantile_group_backtest(frame, column, return_column, quantiles=quantiles)
        capacity = capacity_proxy(frame, column, amount_column=amount_column)
        rows.append(
            FactorSummary(
                factor_name=column,
                horizon_days=horizon_days,
                ic=ic.summary.mean_ic,
                rank_ic=ic.summary.mean_rank_ic,
                icir=ic.summary.icir,
                rank_icir=ic.summary.rank_icir,
                monotonicity=groups.monotonicity,
                turnover=float(groups.turnover.mean()) if not groups.turnover.empty else np.nan,
                capacity_rmb=capacity.capacity_rmb,
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows])


def _corr_by_date(
    frame: pd.DataFrame,
    factor_column: str,
    return_column: str,
    date_column: str,
    method: str,
) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date, group in frame.groupby(date_column, sort=False):
        clean = group[[factor_column, return_column]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 3 or clean[factor_column].nunique() <= 1 or clean[return_column].nunique() <= 1:
            values[pd.Timestamp(date)] = np.nan
        else:
            values[pd.Timestamp(date)] = float(clean[factor_column].corr(clean[return_column], method=method))
    return pd.Series(values, dtype=float)


def _safe_mean(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.mean()) if not clean.empty else np.nan


def _icir(series: pd.Series) -> float:
    clean = series.dropna()
    std = clean.std(ddof=1)
    if clean.empty or not np.isfinite(std) or std <= 1e-12:
        return np.nan
    return float(clean.mean() / std)


def _quantile_labels(series: pd.Series, quantiles: int) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    if len(valid) < quantiles or valid.nunique() < 2:
        return out
    ranks = valid.rank(method="first", pct=True)
    out.loc[valid.index] = np.ceil(ranks * quantiles).clip(1, quantiles)
    return out

