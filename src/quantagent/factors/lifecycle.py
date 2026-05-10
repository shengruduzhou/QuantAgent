from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.factors.evaluation import (
    capacity_proxy,
    factor_correlation_matrix,
    factor_decay_curve,
    information_coefficient,
    quantile_group_backtest,
)


@dataclass(frozen=True)
class LifecycleThresholds:
    active_rank_icir: float = 0.10
    degraded_rank_icir: float = 0.0
    positive_ratio: float = 0.50
    monotonicity: float = 0.20
    retirement_rank_icir: float = -0.05
    drift_limit: float = 3.0


@dataclass(frozen=True)
class FactorLifecycleReport:
    factor_name: str
    rolling_ic: float
    rolling_rank_ic: float
    icir: float
    rank_icir: float
    positive_ic_ratio: float
    newey_west_t_stat: float
    decay_1d: float
    monotonicity: float
    turnover: float
    capacity_proxy: float
    crowding_proxy: float
    max_correlation_to_existing: float
    live_drift: float
    recommended_status: str


def build_factor_lifecycle_report(
    frame: pd.DataFrame,
    factor_column: str,
    return_column: str,
    existing_factor_columns: list[str] | None = None,
    amount_column: str = "amount",
    thresholds: LifecycleThresholds | None = None,
) -> FactorLifecycleReport:
    thresholds = thresholds or LifecycleThresholds()
    ic = information_coefficient(frame, factor_column, return_column)
    groups = quantile_group_backtest(frame, factor_column, return_column)
    decay = factor_decay_curve(frame, factor_column, horizons=(1,)) if "close" in frame.columns else None
    capacity = capacity_proxy(frame, factor_column, amount_column=amount_column) if amount_column in frame.columns else None
    max_corr = _max_existing_corr(frame, factor_column, existing_factor_columns)
    live_drift = _live_drift(frame, factor_column)
    crowding = float(max_corr) if np.isfinite(max_corr) else 0.0
    status = recommend_factor_status(
        rank_icir=ic.summary.rank_icir,
        positive_ratio=ic.summary.positive_ratio,
        monotonicity=groups.monotonicity,
        live_drift=live_drift,
        thresholds=thresholds,
    )
    return FactorLifecycleReport(
        factor_name=factor_column,
        rolling_ic=ic.summary.mean_ic,
        rolling_rank_ic=ic.summary.mean_rank_ic,
        icir=ic.summary.icir,
        rank_icir=ic.summary.rank_icir,
        positive_ic_ratio=ic.summary.positive_ratio,
        newey_west_t_stat=ic.summary.rank_t_stat,
        decay_1d=float(decay.rank_ic.loc[1]) if decay is not None and 1 in decay.rank_ic.index else np.nan,
        monotonicity=groups.monotonicity,
        turnover=float(groups.turnover.mean()) if not groups.turnover.empty else np.nan,
        capacity_proxy=float(capacity.capacity_rmb) if capacity is not None else np.nan,
        crowding_proxy=crowding,
        max_correlation_to_existing=float(max_corr),
        live_drift=float(live_drift),
        recommended_status=status,
    )


def recommend_factor_status(
    rank_icir: float,
    positive_ratio: float,
    monotonicity: float,
    live_drift: float = 0.0,
    thresholds: LifecycleThresholds | None = None,
) -> str:
    thresholds = thresholds or LifecycleThresholds()
    rank_icir = _finite_or(rank_icir, -np.inf)
    positive_ratio = _finite_or(positive_ratio, 0.0)
    monotonicity = _finite_or(monotonicity, 0.0)
    live_drift = abs(_finite_or(live_drift, 0.0))
    if live_drift > thresholds.drift_limit:
        return "watch"
    if rank_icir <= thresholds.retirement_rank_icir:
        return "retired"
    if (
        rank_icir >= thresholds.active_rank_icir
        and positive_ratio >= thresholds.positive_ratio
        and monotonicity >= thresholds.monotonicity
    ):
        return "active"
    if rank_icir >= thresholds.degraded_rank_icir:
        return "degraded"
    return "watch"


def lifecycle_reports_to_frame(reports: list[FactorLifecycleReport]) -> pd.DataFrame:
    return pd.DataFrame([report.__dict__ for report in reports])


def _max_existing_corr(frame: pd.DataFrame, factor_column: str, existing_factor_columns: list[str] | None) -> float:
    if not existing_factor_columns:
        return 0.0
    cols = [factor_column, *existing_factor_columns]
    matrix = factor_correlation_matrix(frame[cols], factor_columns=cols).abs()
    if factor_column not in matrix.index:
        return 0.0
    values = matrix.loc[factor_column, [c for c in existing_factor_columns if c in matrix.columns]].dropna()
    return float(values.max()) if not values.empty else 0.0


def _live_drift(frame: pd.DataFrame, factor_column: str, date_column: str = "trade_date") -> float:
    data = frame[[date_column, factor_column]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if data.empty:
        return 0.0
    data[date_column] = pd.to_datetime(data[date_column])
    dates = sorted(data[date_column].drop_duplicates())
    if len(dates) < 4:
        return 0.0
    split = dates[int(len(dates) * 0.7)]
    hist = data.loc[data[date_column] <= split, factor_column]
    live = data.loc[data[date_column] > split, factor_column]
    std = hist.std(ddof=1)
    if not np.isfinite(std) or std <= 1e-12 or live.empty:
        return 0.0
    return float((live.mean() - hist.mean()) / std)


def _finite_or(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(value) else fallback
