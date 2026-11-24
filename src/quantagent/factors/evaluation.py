from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from quantagent.quant_math.performance import newey_west_t_stat, nonoverlapping_cohort_t_stat


@dataclass(frozen=True)
class ICSummary:
    mean_ic: float
    mean_rank_ic: float
    icir: float
    rank_icir: float
    t_stat: float
    rank_t_stat: float
    positive_ratio: float
    # Non-overlapping-cohort t-stats (ROADMAP P1-6, GOVERNANCE R2: screen on
    # |t| >= 3).  NaN when the label horizon is unknown.
    nonoverlap_t_stat: float = np.nan
    rank_nonoverlap_t_stat: float = np.nan


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


_HORIZON_PATTERNS = (
    re.compile(r"(?:forward|future|exec)_return_(\d+)d$"),
    re.compile(r"(?:forward|future|exec)_(\d+)d_return$"),
    re.compile(r"_(\d+)d$"),
)


def infer_horizon_days(return_column: str) -> int | None:
    """Recover the label horizon from a return-column name, or None if unnamed.

    Every in-repo label column embeds its horizon (``forward_return_20d``,
    ``future_20d_return``, ``exec_return_10d``).  Returning None rather than
    defaulting to 1 is deliberate: silently assuming a 1-day horizon would
    reinstate the under-corrected t-stat this inference exists to prevent.
    """
    for pattern in _HORIZON_PATTERNS:
        match = pattern.search(return_column)
        if match:
            horizon = int(match.group(1))
            if horizon >= 1:
                return horizon
    return None


def information_coefficient(
    frame: pd.DataFrame,
    factor_column: str,
    return_column: str,
    date_column: str = "trade_date",
    *,
    horizon_days: int | None = None,
) -> ICResult:
    """Cross-sectional IC per date, summarised with horizon-aware HAC t-stats.

    An h-day forward-return label makes the per-date IC series overlap on h-1
    days.  The horizon is taken from ``horizon_days`` when given and otherwise
    inferred from ``return_column``; it sets the HAC truncation-lag floor so the
    t-stat is not inflated by the label's own construction.  When the horizon
    cannot be determined the estimator falls back to its sample-size bandwidth
    and a warning is emitted -- the t-stat is then NOT overlap-corrected.

    ``summary.(rank_)nonoverlap_t_stat`` is the average of the ``h``
    non-overlapping cohort t-stats (``nonoverlapping_cohort_t_stat``); screens
    use it with |t| >= 3 (GOVERNANCE R2).  The HAC t is kept for continuity but
    is oversized for h > 1 (measured: ~12% rejections at nominal 5% for h=20).
    """
    horizon = horizon_days if horizon_days is not None else infer_horizon_days(return_column)
    if horizon is None:
        warnings.warn(
            f"could not infer label horizon from return column {return_column!r}; "
            "the Newey-West t-stat is not corrected for label overlap and will be "
            "inflated for multi-day labels. Pass horizon_days= explicitly.",
            RuntimeWarning,
            stacklevel=2,
        )
    ic = _corr_by_date(frame, factor_column, return_column, date_column, method="pearson")
    rank_ic = _corr_by_date(frame, factor_column, return_column, date_column, method="spearman")
    summary = ICSummary(
        mean_ic=_safe_mean(ic),
        mean_rank_ic=_safe_mean(rank_ic),
        icir=_icir(ic),
        rank_icir=_icir(rank_ic),
        t_stat=newey_west_t_stat(ic, horizon_days=horizon),
        rank_t_stat=newey_west_t_stat(rank_ic, horizon_days=horizon),
        positive_ratio=float((rank_ic.dropna() > 0).mean()) if rank_ic.dropna().shape[0] else np.nan,
        nonoverlap_t_stat=_nonoverlap_t(ic, horizon),
        rank_nonoverlap_t_stat=_nonoverlap_t(rank_ic, horizon),
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


def factor_group_metrics(
    frame: pd.DataFrame,
    factor_columns: list[str],
    return_column: str,
    group_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for column in factor_columns:
        ic = information_coefficient(frame, column, return_column)
        rows.append(
            {
                "factor_name": column,
                "group": (group_map or {}).get(column, "ungrouped"),
                "rank_ic": ic.summary.mean_rank_ic,
                "rank_icir": ic.summary.rank_icir,
                "positive_ratio": ic.summary.positive_ratio,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["group", "mean_rank_ic", "mean_rank_icir", "positive_ratio"])
    data = pd.DataFrame(rows)
    return (
        data.groupby("group", sort=True)
        .agg(
            mean_rank_ic=("rank_ic", "mean"),
            mean_rank_icir=("rank_icir", "mean"),
            positive_ratio=("positive_ratio", "mean"),
        )
        .reset_index()
    )


def cross_sectional_rank_ic(
    frame: pd.DataFrame,
    factor_column: str,
    return_column: str,
    date_column: str = "trade_date",
    *,
    method: Literal["spearman", "pearson"] = "spearman",
    min_names: int = 3,
    inf_as_missing: bool = True,
    drop_undefined: bool = False,
    sort: bool = False,
) -> pd.Series:
    """Per-date cross-sectional correlation of a factor with a return label.

    This is the single implementation of the cross-sectional (rank-)IC; every
    other module delegates here.  Semantics, in order:

    1. ``inf_as_missing``: +/-inf is treated as missing (default).  With
       ``False`` an infinite value takes the extreme rank instead (the historic
       behaviour of several callers; the result differs whenever inf is present).
    2. Rows missing the date, the factor or the label are dropped *before*
       ranking, so ranks are always taken over complete pairs.
    3. A date is undefined (NaN) when it has fewer than ``min_names`` complete
       pairs or either side is constant on that date.
    4. ``method="spearman"``: Pearson correlation of within-date average ranks
       (ties share their mean rank), i.e. the Spearman coefficient.

    The output is indexed by the raw date keys.  With ``drop_undefined=False``
    every date that has at least one row in ``frame`` appears (undefined ones
    as NaN); with ``True`` undefined dates are omitted.  ``sort`` orders dates
    ascending; otherwise they follow first appearance in ``frame``.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"unknown correlation method: {method!r}")
    keys = frame[date_column]
    all_keys = pd.Index(keys.dropna().unique())
    if sort:
        all_keys = all_keys.sort_values()
    data = pd.DataFrame(
        {
            "_date": keys.to_numpy(),
            "_x": pd.to_numeric(frame[factor_column], errors="coerce").to_numpy(dtype=float),
            "_y": pd.to_numeric(frame[return_column], errors="coerce").to_numpy(dtype=float),
        }
    )
    if inf_as_missing:
        data[["_x", "_y"]] = data[["_x", "_y"]].replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    if data.empty:
        values = pd.Series(np.nan, index=all_keys, dtype=float)
        return values.iloc[0:0] if drop_undefined else values
    grouped = data.groupby("_date", sort=sort)
    if method == "spearman":
        x = grouped["_x"].rank(method="average")
        y = grouped["_y"].rank(method="average")
    else:
        x = data["_x"]
        y = data["_y"]
    dx = x - x.groupby(data["_date"], sort=sort).transform("mean")
    dy = y - y.groupby(data["_date"], sort=sort).transform("mean")
    sums = pd.DataFrame({"xy": dx * dy, "xx": dx * dx, "yy": dy * dy, "_date": data["_date"]}).groupby(
        "_date", sort=sort
    ).sum()
    if method == "spearman":
        # Average ranks of a constant cross-section are exactly (n+1)/2, so
        # zero rank dispersion <=> a constant side (cheaper than nunique).
        defined = (grouped.size() >= int(min_names)) & (sums["xx"] > 0) & (sums["yy"] > 0)
    else:
        # Raw values: a constant side can leave ~1e-17 of rounding dispersion.
        stats = grouped.agg(n=("_x", "size"), ux=("_x", "nunique"), uy=("_y", "nunique"))
        defined = (stats["n"] >= int(min_names)) & (stats["ux"] >= 2) & (stats["uy"] >= 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = sums["xy"] / np.sqrt(sums["xx"] * sums["yy"])
    corr = corr.clip(-1.0, 1.0).where(defined.reindex(corr.index, fill_value=False))
    corr.name = None
    corr.index.name = None
    if drop_undefined:
        return corr.dropna().astype(float)
    return corr.reindex(all_keys).astype(float)


def _corr_by_date(
    frame: pd.DataFrame,
    factor_column: str,
    return_column: str,
    date_column: str,
    method: Literal["spearman", "pearson"],
) -> pd.Series:
    series = cross_sectional_rank_ic(
        frame, factor_column, return_column, date_column, method=method, min_names=3
    )
    return pd.Series(
        series.to_numpy(dtype=float),
        index=pd.Index([pd.Timestamp(key) for key in series.index]),
        dtype=float,
    )


def _nonoverlap_t(series: pd.Series, horizon: int | None) -> float:
    # Cohorts need the session order; undefined dates stay as NaN placeholders.
    if horizon is None or series.empty:
        return np.nan
    return nonoverlapping_cohort_t_stat(series.sort_index(), horizon)


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
