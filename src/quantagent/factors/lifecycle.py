from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.factors.evaluation import (
    capacity_proxy,
    factor_correlation_matrix,
    information_coefficient,
    quantile_group_backtest,
)
from quantagent.factors.executable_labels import (
    FACTOR_LABEL_SEMANTICS,
    executable_factor_decay_curve,
)


@dataclass(frozen=True)
class LifecycleThresholds:
    # Historical field name retained for API compatibility. Crossing this gate
    # now means VALIDATED, never economically ACTIVE; ACTIVE is owned by the
    # state machine after shadow/promotion evidence.
    active_rank_icir: float = 0.10
    degraded_rank_icir: float = 0.0
    positive_ratio: float = 0.50
    monotonicity: float = 0.20
    retirement_rank_icir: float = -0.05
    drift_limit: float = 3.0
    min_effective_dates: int = 60
    min_median_symbols_per_date: int = 20
    min_newey_west_rank_t_stat: float = 2.0
    max_existing_correlation_for_active: float = 0.90
    min_capacity_rmb_for_active: float = 0.0


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
    effective_dates: int
    median_symbols_per_date: float
    label_semantics: str
    recommended_status: str


def build_factor_lifecycle_report(
    frame: pd.DataFrame,
    factor_column: str,
    return_column: str,
    existing_factor_columns: list[str] | None = None,
    amount_column: str = "amount",
    thresholds: LifecycleThresholds | None = None,
) -> FactorLifecycleReport:
    """Build a research lifecycle diagnostic using executable decay semantics.

    ``return_column`` is caller-supplied and may be used for research analysis;
    the decay evidence is always recomputed with the governed T-close -> next
    session entry convention when prices are available.  A strong report can
    recommend ``validated`` but can never directly declare a factor ``active``.
    """

    thresholds = thresholds or LifecycleThresholds()
    required = {"trade_date", "symbol", factor_column, return_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"factor lifecycle frame missing columns: {missing}")
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "symbol"])
    if data.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("factor lifecycle requires unique trade_date/symbol rows")

    counts = data.groupby("trade_date")["symbol"].nunique()
    effective_dates = int(len(counts))
    median_symbols = float(counts.median()) if not counts.empty else 0.0
    ic = information_coefficient(data, factor_column, return_column)
    groups = quantile_group_backtest(data, factor_column, return_column)
    decay = (
        executable_factor_decay_curve(data, factor_column, horizons=(1,))
        if "close" in data.columns
        else None
    )
    capacity = (
        capacity_proxy(data, factor_column, amount_column=amount_column)
        if amount_column in data.columns
        else None
    )
    max_corr = _max_existing_corr(data, factor_column, existing_factor_columns)
    live_drift = _live_drift(data, factor_column)
    crowding = float(max_corr) if np.isfinite(max_corr) else np.nan
    capacity_rmb = float(capacity.capacity_rmb) if capacity is not None else np.nan
    status = recommend_factor_status(
        rank_icir=ic.summary.rank_icir,
        positive_ratio=ic.summary.positive_ratio,
        monotonicity=groups.monotonicity,
        live_drift=live_drift,
        max_existing_correlation=max_corr,
        capacity_rmb=capacity_rmb,
        effective_dates=effective_dates,
        median_symbols_per_date=median_symbols,
        newey_west_rank_t_stat=ic.summary.rank_t_stat,
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
        decay_1d=(
            float(decay.rank_ic.loc[1])
            if decay is not None and 1 in decay.rank_ic.index
            else np.nan
        ),
        monotonicity=groups.monotonicity,
        turnover=float(groups.turnover.mean()) if not groups.turnover.empty else np.nan,
        capacity_proxy=capacity_rmb,
        crowding_proxy=crowding,
        max_correlation_to_existing=float(max_corr),
        live_drift=float(live_drift),
        effective_dates=effective_dates,
        median_symbols_per_date=median_symbols,
        label_semantics=FACTOR_LABEL_SEMANTICS,
        recommended_status=status,
    )


def recommend_factor_status(
    rank_icir: float,
    positive_ratio: float,
    monotonicity: float,
    live_drift: float = 0.0,
    max_existing_correlation: float = 0.0,
    capacity_rmb: float = np.nan,
    effective_dates: int | None = None,
    median_symbols_per_date: float | None = None,
    newey_west_rank_t_stat: float = np.nan,
    thresholds: LifecycleThresholds | None = None,
) -> str:
    """Recommend a *research* lifecycle status, never direct economic ACTIVE."""

    thresholds = thresholds or LifecycleThresholds()
    rank_icir = _finite_or(rank_icir, -np.inf)
    positive_ratio = _finite_or(positive_ratio, 0.0)
    monotonicity = _finite_or(monotonicity, 0.0)

    # Coverage is mandatory. Legacy direct callers that do not provide it are
    # deliberately held in WATCH rather than treating missing evidence as safe.
    if effective_dates is None or int(effective_dates) < thresholds.min_effective_dates:
        return "watch"
    if (
        median_symbols_per_date is None
        or not np.isfinite(median_symbols_per_date)
        or float(median_symbols_per_date) < thresholds.min_median_symbols_per_date
    ):
        return "watch"
    if (
        not np.isfinite(newey_west_rank_t_stat)
        or float(newey_west_rank_t_stat) < thresholds.min_newey_west_rank_t_stat
    ):
        return "watch"

    if not np.isfinite(live_drift) or abs(float(live_drift)) > thresholds.drift_limit:
        return "watch"

    if np.isfinite(max_existing_correlation):
        if abs(float(max_existing_correlation)) > thresholds.max_existing_correlation_for_active:
            return "watch"

    if rank_icir <= thresholds.retirement_rank_icir:
        return "retired_candidate"

    capacity_gate = True
    if thresholds.min_capacity_rmb_for_active > 0:
        capacity_gate = bool(
            np.isfinite(capacity_rmb)
            and float(capacity_rmb) >= thresholds.min_capacity_rmb_for_active
        )

    if (
        rank_icir >= thresholds.active_rank_icir
        and positive_ratio >= thresholds.positive_ratio
        and monotonicity >= thresholds.monotonicity
        and capacity_gate
    ):
        return "validated"
    if rank_icir >= thresholds.degraded_rank_icir:
        return "watch"
    return "degraded_candidate"


def lifecycle_reports_to_frame(reports: list[FactorLifecycleReport]) -> pd.DataFrame:
    return pd.DataFrame([report.__dict__ for report in reports])


def _max_existing_corr(
    frame: pd.DataFrame,
    factor_column: str,
    existing_factor_columns: list[str] | None,
) -> float:
    if not existing_factor_columns:
        return 0.0
    existing = [column for column in existing_factor_columns if column in frame.columns]
    if not existing:
        return np.nan
    cols = [factor_column, *existing]
    # Correlation used for redundancy is cross-sectional rank correlation, not
    # raw pooled scale correlation. Rank each date first so market-level scale
    # drift cannot manufacture similarity.
    ranked = frame[["trade_date", *cols]].copy()
    ranked[cols] = ranked.groupby("trade_date")[cols].rank(pct=True)
    matrix = factor_correlation_matrix(ranked[cols], factor_columns=cols, method="spearman").abs()
    if factor_column not in matrix.index:
        return np.nan
    values = matrix.loc[factor_column, existing].dropna()
    return float(values.max()) if not values.empty else np.nan


def _live_drift(frame: pd.DataFrame, factor_column: str, date_column: str = "trade_date") -> float:
    data = frame[[date_column, factor_column]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if data.empty:
        return np.nan
    data[date_column] = pd.to_datetime(data[date_column])
    dates = sorted(data[date_column].drop_duplicates())
    if len(dates) < 10:
        return np.nan
    split_index = max(1, min(len(dates) - 1, int(len(dates) * 0.7)))
    split = dates[split_index - 1]
    hist = data.loc[data[date_column] <= split, factor_column]
    live = data.loc[data[date_column] > split, factor_column]
    std = hist.std(ddof=1)
    if not np.isfinite(std) or std <= 1e-12 or live.empty:
        return np.nan
    return float((live.mean() - hist.mean()) / std)


def _finite_or(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(value) else fallback
