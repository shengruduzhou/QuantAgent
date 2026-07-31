"""Evaluate one factor blend on one panel segment.

Scoring convention
------------------
Factors are combined **after per-date cross-sectional rank normalisation**, not
on their raw scale. Raw Alpha101/Alpha181 columns differ by orders of magnitude,
so a weight vector over raw columns encodes scale as much as conviction and the
weights are not comparable between factors. Normalising each factor to a
per-date uniform rank mapped onto ``[-1, 1]`` makes a weight vector mean exactly
"how much of the blend's conviction this factor carries".

The normalisation uses only rows of the same ``trade_date``, so it introduces no
lookahead.

Return-series resolution
------------------------
The panel supplies *forward* returns over ``horizon_days``. Consecutive daily
rebalances therefore overlap, and compounding them would fabricate returns. We
instead decompose into ``horizon_days`` non-overlapping cohorts
(``dates[offset::horizon]``): each cohort is a genuinely tradable schedule that
rebalances every ``horizon_days``. Scalar metrics are averaged across cohorts;
the returned NAV is the equally weighted average of the cohort NAVs, i.e. the
staggered sleeve implementation.

Consequence, stated so the UI can state it too: drawdown is measured on a
rebalance-frequency series, not a daily one, and is therefore a lower bound on
the daily-marked drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantagent.quant_math.performance import (
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
)

REQUIRED_PANEL_COLUMNS = ("trade_date", "symbol")


@dataclass(frozen=True)
class SegmentEvaluation:
    """Out-of-sample performance of one blend on one segment."""

    periods: int
    observations: int
    annual_return: float
    excess_return: float
    benchmark_annual_return: float
    max_drawdown: float
    sharpe: float
    calmar: float
    average_turnover: float
    cost_drag: float
    win_rate: float
    cohort_count: int
    nav: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    period_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def is_empty(self) -> bool:
        return self.observations == 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "periods": int(self.periods),
            "observations": int(self.observations),
            "annualReturn": _finite(self.annual_return),
            "excessReturn": _finite(self.excess_return),
            "benchmarkAnnualReturn": _finite(self.benchmark_annual_return),
            "maxDrawdown": _finite(self.max_drawdown),
            "sharpe": _finite(self.sharpe),
            "calmar": _finite(self.calmar),
            "averageTurnover": _finite(self.average_turnover),
            "costDrag": _finite(self.cost_drag),
            "winRate": _finite(self.win_rate),
            "cohortCount": int(self.cohort_count),
        }


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _nanmean(values: list[float]) -> float:
    """Mean over finite entries; 0.0 when none are finite.

    ``np.nanmean`` warns and returns NaN on an all-NaN slice, which happens for
    ratio metrics like Calmar whenever a cohort has zero drawdown.
    """
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else 0.0


def _empty_evaluation(periods: int) -> SegmentEvaluation:
    return SegmentEvaluation(
        periods=periods,
        observations=0,
        annual_return=0.0,
        excess_return=0.0,
        benchmark_annual_return=0.0,
        max_drawdown=0.0,
        sharpe=0.0,
        calmar=0.0,
        average_turnover=0.0,
        cost_drag=0.0,
        win_rate=0.0,
        cohort_count=0,
    )


def cross_sectional_scores(
    panel: pd.DataFrame,
    factor_names: list[str],
    weights: np.ndarray,
) -> pd.DataFrame:
    """Blend factors into a per-row score using per-date rank normalisation.

    Returns a frame with ``trade_date``, ``symbol`` and ``score``. Rows whose
    factors are entirely missing on a date are dropped; a factor missing for a
    subset of names on a date contributes its cross-sectional median rank (0.0)
    for those names rather than removing the name from the universe.
    """
    if len(factor_names) != len(weights):
        raise ValueError("factor_names and weights must have equal length")
    missing = [name for name in factor_names if name not in panel.columns]
    if missing:
        raise KeyError(f"factor columns missing from panel: {sorted(missing)}")

    work = panel[["trade_date", "symbol", *factor_names]].copy()
    numeric = work[factor_names].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    work = work.loc[numeric.notna().any(axis=1)].copy()
    if work.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "score"])
    numeric = numeric.loc[work.index]

    grouped = numeric.groupby(work["trade_date"], sort=False)
    # Rank -> (0, 1) -> centred on 0 with unit range endpoints at +/-1.
    ranked = grouped.rank(pct=True, method="average")
    centred = (ranked - 0.5) * 2.0
    centred = centred.fillna(0.0)

    work["score"] = centred.to_numpy(dtype=float) @ np.asarray(weights, dtype=float)
    return work[["trade_date", "symbol", "score"]]


def _cohort_metrics(
    selected: pd.DataFrame,
    cohort_dates: pd.DatetimeIndex,
    *,
    transaction_cost_bps: float,
) -> tuple[pd.Series, pd.Series] | None:
    """Return ``(net_period_returns, turnover)`` for one non-overlapping cohort."""
    cohort = selected[selected["trade_date"].isin(cohort_dates)]
    if cohort.empty:
        return None
    gross = cohort.groupby("trade_date", sort=True).apply(
        lambda group: float((group["forward_return"] * group["weight"]).sum()),
        include_groups=False,
    )
    if gross.empty:
        return None
    weight_panel = cohort.pivot_table(
        index="trade_date",
        columns="symbol",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    turnover = weight_panel.diff().abs().sum(axis=1) / 2.0
    if not turnover.empty:
        turnover.iloc[0] = float(weight_panel.iloc[0].abs().sum())
    turnover = turnover.reindex(gross.index).fillna(0.0)
    net = gross - turnover * (float(transaction_cost_bps) / 10_000.0)
    return net, turnover


def _benchmark_period_returns(
    forward_panel: pd.DataFrame,
    cohort_dates: pd.DatetimeIndex,
    benchmark: pd.Series | None,
) -> pd.Series:
    """Benchmark return over the same cohort dates.

    When an explicit benchmark series is supplied it is reindexed onto the
    cohort dates. Otherwise the equal-weight universe basket is used; that
    fallback includes names the strategy may not be able to trade, so callers
    must label it as ``universe_equal_weight`` rather than as an index.
    """
    if benchmark is not None and not benchmark.empty:
        return benchmark.reindex(cohort_dates).dropna()
    universe = forward_panel[forward_panel["trade_date"].isin(cohort_dates)]
    if universe.empty:
        return pd.Series(dtype=float)
    return universe.groupby("trade_date", sort=True)["forward_return"].mean()


def _annualise(period_returns: pd.Series, periods: int) -> float:
    clean = period_returns.dropna()
    if clean.empty:
        return 0.0
    growth = float(np.prod(1.0 + clean.to_numpy(dtype=float)))
    if growth <= 0.0:
        return -1.0
    years = len(clean) / float(periods)
    if years <= 0:
        return 0.0
    return float(growth ** (1.0 / years) - 1.0)


def evaluate_blend(
    *,
    factor_panel: pd.DataFrame,
    forward_panel: pd.DataFrame,
    factor_names: list[str],
    weights: np.ndarray,
    top_k: int,
    horizon_days: int,
    transaction_cost_bps: float = 8.0,
    benchmark_returns: pd.Series | None = None,
    min_label_coverage: float = 0.80,
    min_cohort_observations: int = 2,
) -> SegmentEvaluation:
    """Score one blend on one segment of the panel.

    ``factor_panel`` and ``forward_panel`` must already be restricted to the
    segment being evaluated; this function performs no splitting and therefore
    cannot leak across a fold boundary on its own.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    periods = max(1, int(round(252 / horizon_days)))

    scores = cross_sectional_scores(factor_panel, factor_names, weights)
    if scores.empty:
        return _empty_evaluation(periods)

    scores = scores.sort_values(
        ["trade_date", "score", "symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    scores["rank"] = scores.groupby("trade_date", sort=False).cumcount()
    selected = scores[scores["rank"] < top_k][["trade_date", "symbol"]].copy()
    if selected.empty:
        return _empty_evaluation(periods)

    labels = forward_panel[["trade_date", "symbol", "forward_return"]].copy()
    labels["forward_return"] = pd.to_numeric(labels["forward_return"], errors="coerce")
    selected = selected.merge(labels, on=["trade_date", "symbol"], how="left")
    available = selected["forward_return"].notna()
    coverage = available.groupby(selected["trade_date"]).transform("mean")
    selected = selected[available & (coverage >= min_label_coverage)].copy()
    if selected.empty:
        return _empty_evaluation(periods)

    counts = selected.groupby("trade_date")["symbol"].transform("size")
    selected["weight"] = 1.0 / counts.astype(float)
    valid_dates = pd.DatetimeIndex(sorted(selected["trade_date"].unique()))

    cohort_navs: list[pd.Series] = []
    cohort_returns: list[pd.Series] = []
    annual_returns: list[float] = []
    benchmark_annuals: list[float] = []
    drawdowns: list[float] = []
    sharpes: list[float] = []
    calmars: list[float] = []
    turnovers: list[float] = []
    cost_drags: list[float] = []
    win_rates: list[float] = []

    for offset in range(min(horizon_days, len(valid_dates))):
        cohort_dates = valid_dates[offset::horizon_days]
        if len(cohort_dates) < min_cohort_observations:
            continue
        computed = _cohort_metrics(
            selected,
            cohort_dates,
            transaction_cost_bps=transaction_cost_bps,
        )
        if computed is None:
            continue
        net, turnover = computed
        if net.empty:
            continue
        nav = (1.0 + net).cumprod()
        benchmark_period = _benchmark_period_returns(
            forward_panel, pd.DatetimeIndex(net.index), benchmark_returns
        )
        cohort_navs.append(nav)
        cohort_returns.append(net)
        annual_returns.append(_annualise(net, periods))
        benchmark_annuals.append(_annualise(benchmark_period, periods))
        drawdowns.append(abs(max_drawdown(nav)) if len(nav) > 1 else 0.0)
        sharpes.append(sharpe_ratio(net, periods_per_year=periods))
        calmars.append(calmar_ratio(nav, periods_per_year=periods))
        turnovers.append(float(turnover.mean()))
        cost_drags.append(float(turnover.mean() * transaction_cost_bps / 10_000.0))
        win_rates.append(float((net > 0).mean()))

    if not cohort_navs:
        return _empty_evaluation(periods)

    combined_nav = (
        pd.concat(cohort_navs, axis=1).sort_index().ffill().mean(axis=1)
    )
    combined_returns = pd.concat(cohort_returns).groupby(level=0).mean().sort_index()
    annual = float(np.mean(annual_returns))
    benchmark_annual = float(np.mean(benchmark_annuals))

    return SegmentEvaluation(
        periods=periods,
        observations=int(sum(len(item) for item in cohort_returns)),
        annual_return=annual,
        excess_return=annual - benchmark_annual,
        benchmark_annual_return=benchmark_annual,
        max_drawdown=float(np.mean(drawdowns)),
        sharpe=_nanmean(sharpes),
        calmar=_nanmean(calmars),
        average_turnover=float(np.mean(turnovers)),
        cost_drag=float(np.mean(cost_drags)),
        win_rate=float(np.mean(win_rates)),
        cohort_count=len(cohort_navs),
        nav=combined_nav,
        period_returns=combined_returns,
    )


__all__ = [
    "REQUIRED_PANEL_COLUMNS",
    "SegmentEvaluation",
    "cross_sectional_scores",
    "evaluate_blend",
]
