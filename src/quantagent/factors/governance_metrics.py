"""Factor validity and promotion governance.

A factor is not admitted because of one attractive full-sample IC.  Core
validity and production promotion are intentionally separate:

* validity: executable predictive evidence, coverage, stability, redundancy,
  decay and capacity;
* promotion: pre-registration/OOS identity, cumulative search evidence,
  multiple-testing gates, strict long-only economics and shadow evidence.

Unknown promotion evidence fails closed.  This module never arms trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.quant_math.performance import newey_west_t_stat


@dataclass(frozen=True)
class FactorGateConfig:
    min_dates: int = 120
    min_symbols_per_date: int = 30
    min_mean_rank_ic: float = 0.01
    # Standard ICIR = mean(IC) / std(IC).  It is deliberately not multiplied
    # by sqrt(n); significance is separately represented by the NW t-stat.
    min_ic_information_ratio: float = 0.20
    min_newey_west_rank_t_stat: float = 2.0
    min_positive_ic_ratio: float = 0.52
    max_losing_period_rate: float = 0.40
    max_recent_predictive_drift_z: float = 2.5
    max_library_abs_correlation: float = 0.85
    min_decay_retention: float = 0.20
    max_decay_reversal: float = -0.02
    target_book_cny: float = 10_000_000.0
    max_adv_participation: float = 0.10
    min_capacity_multiple: float = 1.0
    max_pbo: float = 0.25
    min_dsr_probability: float = 0.95
    max_spa_pvalue: float = 0.05
    min_shadow_days: int = 20


@dataclass(frozen=True)
class FactorPromotionContext:
    """Evidence external to a one-factor IC calculation.

    These facts should be produced by the governed selection/backtest/shadow
    paths.  The factor gate consumes them; it does not fabricate them.
    """

    label_semantics: str
    preregistered: bool
    oos_only: bool
    cumulative_trials: int
    multiple_testing_passed: bool
    pbo: float
    dsr_probability: float
    spa_pvalue: float
    strict_long_only_backtest_passed: bool
    shadow_days: int
    shadow_passed: bool
    pit_data_certified: bool
    evidence_digest: str


@dataclass
class FactorGovernanceReport:
    factor_name: str
    passed: bool
    promotion_ready: bool
    mean_rank_ic: float
    # Historical public field retained; semantics are now explicitly standard
    # RankICIR, not mean/std*sqrt(n).
    ic_information_ratio: float
    newey_west_rank_t_stat: float
    positive_ic_ratio: float
    losing_period_rate: float
    recent_predictive_drift_z: float
    max_library_abs_correlation: float
    most_correlated_factor: str | None
    decay_curve: dict[int, float]
    decay_retention: float
    estimated_capacity_cny: float
    capacity_multiple: float
    coverage_dates: int
    median_symbols_per_date: float
    label_semantics: str | None
    cumulative_trials: int | None
    rejection_reasons: list[str] = field(default_factory=list)
    promotion_blockers: list[str] = field(default_factory=list)

    @property
    def rank_icir(self) -> float:
        return self.ic_information_ratio

    def to_dict(self) -> dict[str, object]:
        return {
            "factor_name": self.factor_name,
            "passed": self.passed,
            "promotion_ready": self.promotion_ready,
            "mean_rank_ic": self.mean_rank_ic,
            "rank_icir": self.rank_icir,
            "ic_information_ratio": self.ic_information_ratio,
            "newey_west_rank_t_stat": self.newey_west_rank_t_stat,
            "positive_ic_ratio": self.positive_ic_ratio,
            "losing_period_rate": self.losing_period_rate,
            "recent_predictive_drift_z": self.recent_predictive_drift_z,
            "max_library_abs_correlation": self.max_library_abs_correlation,
            "most_correlated_factor": self.most_correlated_factor,
            "decay_curve": dict(self.decay_curve),
            "decay_retention": self.decay_retention,
            "estimated_capacity_cny": self.estimated_capacity_cny,
            "capacity_multiple": self.capacity_multiple,
            "coverage_dates": self.coverage_dates,
            "median_symbols_per_date": self.median_symbols_per_date,
            "label_semantics": self.label_semantics,
            "cumulative_trials": self.cumulative_trials,
            "rejection_reasons": list(self.rejection_reasons),
            "promotion_blockers": list(self.promotion_blockers),
        }


def _rank_ic_by_date(frame: pd.DataFrame, factor_col: str, return_col: str) -> pd.Series:
    def _one(group: pd.DataFrame) -> float:
        valid = group[[factor_col, return_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 3 or valid[factor_col].nunique() < 2 or valid[return_col].nunique() < 2:
            return float("nan")
        return float(valid[factor_col].corr(valid[return_col], method="spearman"))

    values = frame.groupby("trade_date", sort=True).apply(_one).dropna()
    values.index = pd.DatetimeIndex(values.index)
    return values.sort_index()


def _ic_ir(ic: pd.Series) -> float:
    """Standard IC information ratio: mean / sample standard deviation."""
    clean = pd.to_numeric(ic, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return float("nan")
    std = float(clean.std(ddof=1))
    if std <= 1e-12:
        return float("nan")
    return float(clean.mean() / std)


def _period_losing_rate(ic: pd.Series, frequency: str = "QE") -> float:
    if ic.empty:
        return 1.0
    period = ic.groupby(ic.index.to_period(frequency)).mean()
    return float((period <= 0).mean()) if not period.empty else 1.0


def _recent_predictive_drift_z(ic: pd.Series, recent_fraction: float = 0.30) -> float:
    """Detect deterioration in predictive IC, not merely factor-value scale."""
    clean = ic.dropna().sort_index()
    if len(clean) < 20:
        return float("nan")
    cut = max(10, min(len(clean) - 5, int(len(clean) * (1.0 - recent_fraction))))
    history = clean.iloc[:cut]
    recent = clean.iloc[cut:]
    std = float(history.std(ddof=1))
    if not np.isfinite(std) or std <= 1e-12 or recent.empty:
        return float("nan")
    return float((recent.mean() - history.mean()) / std)


def _library_correlation(
    frame: pd.DataFrame,
    factor_col: str,
    library_columns: Iterable[str],
) -> tuple[float, str | None]:
    columns = [column for column in library_columns if column in frame.columns and column != factor_col]
    if not columns:
        return 0.0, None
    ranked = frame[["trade_date", factor_col, *columns]].copy()
    ranked[[factor_col, *columns]] = ranked.groupby("trade_date")[[factor_col, *columns]].rank(pct=True)
    corr = ranked[[factor_col, *columns]].corr(method="pearson")[factor_col].drop(factor_col).abs()
    if corr.empty:
        return 0.0, None
    name = str(corr.idxmax())
    return float(corr.loc[name]), name


def _decay_curve(
    frame: pd.DataFrame,
    factor_col: str,
    return_columns: dict[int, str],
) -> dict[int, float]:
    curve: dict[int, float] = {}
    for horizon, column in sorted(return_columns.items()):
        if column not in frame.columns:
            continue
        ic = _rank_ic_by_date(frame, factor_col, column)
        curve[int(horizon)] = float(ic.mean()) if not ic.empty else float("nan")
    return curve


def _decay_retention(curve: dict[int, float], target_horizon: int) -> float:
    finite = {h: value for h, value in curve.items() if np.isfinite(value)}
    if not finite:
        return float("nan")
    nearest = min(finite, key=lambda h: abs(h - target_horizon))
    base = finite[min(finite)]
    if abs(base) <= 1e-12:
        return float("nan")
    return float(finite[nearest] / base)


def _capacity_estimate(
    frame: pd.DataFrame,
    factor_col: str,
    *,
    adv_col: str,
    max_adv_participation: float,
    top_quantile: float = 0.10,
) -> float:
    if adv_col not in frame.columns:
        return 0.0
    work = frame[["trade_date", factor_col, adv_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if work.empty:
        return 0.0
    work["rank_pct"] = work.groupby("trade_date")[factor_col].rank(pct=True)
    selected = work[work["rank_pct"] >= 1.0 - top_quantile]
    if selected.empty:
        return 0.0
    daily = selected.groupby("trade_date")[adv_col].sum() * max_adv_participation
    return float(daily.quantile(0.10)) if not daily.empty else 0.0


def _promotion_blockers(
    context: FactorPromotionContext | None,
    cfg: FactorGateConfig,
) -> list[str]:
    if context is None:
        return ["promotion_context_missing"]
    reasons: list[str] = []
    if context.label_semantics != EXECUTION_TIMING_SEMANTICS:
        reasons.append(
            f"label_semantics={context.label_semantics!r} does not match {EXECUTION_TIMING_SEMANTICS!r}"
        )
    if not context.preregistered:
        reasons.append("pre_registration_missing")
    if not context.oos_only:
        reasons.append("promotion_evidence_not_oos_only")
    if int(context.cumulative_trials) < 1:
        reasons.append("cumulative_trial_count_missing")
    if not context.multiple_testing_passed:
        reasons.append("multiple_testing_gate_not_passed")
    if not np.isfinite(context.pbo) or float(context.pbo) > cfg.max_pbo:
        reasons.append(f"pbo={context.pbo} exceeds {cfg.max_pbo}")
    if not np.isfinite(context.dsr_probability) or float(context.dsr_probability) < cfg.min_dsr_probability:
        reasons.append(f"dsr_probability={context.dsr_probability} below {cfg.min_dsr_probability}")
    if not np.isfinite(context.spa_pvalue) or float(context.spa_pvalue) > cfg.max_spa_pvalue:
        reasons.append(f"spa_pvalue={context.spa_pvalue} exceeds {cfg.max_spa_pvalue}")
    if not context.strict_long_only_backtest_passed:
        reasons.append("strict_long_only_backtest_not_passed")
    if int(context.shadow_days) < cfg.min_shadow_days or not context.shadow_passed:
        reasons.append("shadow_evidence_insufficient")
    if not context.pit_data_certified:
        reasons.append("pit_data_not_certified")
    if not str(context.evidence_digest).strip():
        reasons.append("promotion_evidence_digest_missing")
    return reasons


def evaluate_factor_candidate(
    frame: pd.DataFrame,
    *,
    factor_name: str,
    target_return_col: str,
    target_horizon_days: int,
    decay_return_columns: dict[int, str],
    library_columns: Iterable[str] = (),
    adv_col: str = "adv20_cny",
    label_semantics: str | None = None,
    promotion_context: FactorPromotionContext | None = None,
    config: FactorGateConfig | None = None,
) -> FactorGovernanceReport:
    cfg = config or FactorGateConfig()
    required = {"trade_date", "symbol", factor_name, target_return_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"factor governance frame missing columns: {sorted(missing)}")
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work = work.dropna(subset=["trade_date", "symbol"])
    duplicate_count = int(work.duplicated(["trade_date", "symbol"]).sum())
    if duplicate_count:
        raise ValueError(f"duplicate trade_date/symbol rows: {duplicate_count}")

    counts = work.groupby("trade_date")["symbol"].nunique()
    coverage_dates = int(len(counts))
    median_symbols = float(counts.median()) if not counts.empty else 0.0
    eligible_dates = counts[counts >= cfg.min_symbols_per_date].index
    eligible = work[work["trade_date"].isin(eligible_dates)].copy()
    ic = _rank_ic_by_date(eligible, factor_name, target_return_col)
    mean_ic = float(ic.mean()) if not ic.empty else float("nan")
    icir = _ic_ir(ic)
    rank_t_stat = float(newey_west_t_stat(ic)) if not ic.empty else float("nan")
    positive_ratio = float((ic > 0).mean()) if not ic.empty else float("nan")
    losing_rate = _period_losing_rate(ic)
    predictive_drift = _recent_predictive_drift_z(ic)
    max_corr, corr_name = _library_correlation(eligible, factor_name, library_columns)
    curve = _decay_curve(eligible, factor_name, decay_return_columns)
    retention = _decay_retention(curve, target_horizon_days)
    capacity = _capacity_estimate(
        eligible,
        factor_name,
        adv_col=adv_col,
        max_adv_participation=cfg.max_adv_participation,
    )
    capacity_multiple = capacity / cfg.target_book_cny if cfg.target_book_cny > 0 else float("inf")

    reasons: list[str] = []
    if coverage_dates < cfg.min_dates:
        reasons.append(f"coverage_dates={coverage_dates} below {cfg.min_dates}")
    if median_symbols < cfg.min_symbols_per_date:
        reasons.append(f"median_symbols_per_date={median_symbols:.1f} below {cfg.min_symbols_per_date}")
    if not np.isfinite(mean_ic) or mean_ic < cfg.min_mean_rank_ic:
        reasons.append(f"mean_rank_ic={mean_ic:.6f} below {cfg.min_mean_rank_ic:.6f}")
    if not np.isfinite(icir) or icir < cfg.min_ic_information_ratio:
        reasons.append(f"rank_icir={icir:.4f} below {cfg.min_ic_information_ratio:.4f}")
    if not np.isfinite(rank_t_stat) or rank_t_stat < cfg.min_newey_west_rank_t_stat:
        reasons.append(f"newey_west_rank_t_stat={rank_t_stat:.4f} below {cfg.min_newey_west_rank_t_stat:.4f}")
    if not np.isfinite(positive_ratio) or positive_ratio < cfg.min_positive_ic_ratio:
        reasons.append(f"positive_ic_ratio={positive_ratio:.4f} below {cfg.min_positive_ic_ratio:.4f}")
    if losing_rate > cfg.max_losing_period_rate:
        reasons.append(f"losing_period_rate={losing_rate:.4f} exceeds {cfg.max_losing_period_rate:.4f}")
    # We care specifically about recent deterioration. Positive drift is not a
    # reason to reject; sufficiently negative standardized change is.
    if np.isfinite(predictive_drift) and predictive_drift < -abs(cfg.max_recent_predictive_drift_z):
        reasons.append(
            f"recent_predictive_drift_z={predictive_drift:.4f} below {-abs(cfg.max_recent_predictive_drift_z):.4f}"
        )
    if max_corr > cfg.max_library_abs_correlation:
        reasons.append(
            f"library_correlation={max_corr:.4f} with {corr_name} exceeds {cfg.max_library_abs_correlation:.4f}"
        )
    if not np.isfinite(retention) or retention < cfg.min_decay_retention:
        reasons.append(f"decay_retention={retention:.4f} below {cfg.min_decay_retention:.4f}")
    finite_curve = [value for value in curve.values() if np.isfinite(value)]
    if finite_curve and min(finite_curve) < cfg.max_decay_reversal:
        reasons.append(f"decay_curve reverses to {min(finite_curve):.4f} below {cfg.max_decay_reversal:.4f}")
    if capacity_multiple < cfg.min_capacity_multiple:
        reasons.append(f"capacity_multiple={capacity_multiple:.3f} below {cfg.min_capacity_multiple:.3f}")

    blockers = _promotion_blockers(promotion_context, cfg)
    if label_semantics is not None and label_semantics != EXECUTION_TIMING_SEMANTICS:
        blockers.append(
            f"evaluated_label_semantics={label_semantics!r} does not match governed executable semantics"
        )
    passed = not reasons
    promotion_ready = bool(passed and not blockers)
    return FactorGovernanceReport(
        factor_name=factor_name,
        passed=passed,
        promotion_ready=promotion_ready,
        mean_rank_ic=mean_ic,
        ic_information_ratio=icir,
        newey_west_rank_t_stat=rank_t_stat,
        positive_ic_ratio=positive_ratio,
        losing_period_rate=losing_rate,
        recent_predictive_drift_z=predictive_drift,
        max_library_abs_correlation=max_corr,
        most_correlated_factor=corr_name,
        decay_curve=curve,
        decay_retention=retention,
        estimated_capacity_cny=capacity,
        capacity_multiple=capacity_multiple,
        coverage_dates=coverage_dates,
        median_symbols_per_date=median_symbols,
        label_semantics=label_semantics,
        cumulative_trials=(None if promotion_context is None else int(promotion_context.cumulative_trials)),
        rejection_reasons=reasons,
        promotion_blockers=blockers,
    )


def correlation_clusters(
    factor_frame: pd.DataFrame,
    *,
    factor_columns: Iterable[str],
    threshold: float = 0.85,
) -> list[list[str]]:
    """Greedy absolute cross-sectional rank-correlation clusters."""
    columns = [column for column in factor_columns if column in factor_frame.columns]
    if not columns:
        return []
    ranked = factor_frame[["trade_date", *columns]].copy()
    ranked[columns] = ranked.groupby("trade_date")[columns].rank(pct=True)
    corr = ranked[columns].corr(method="pearson").abs()
    remaining = set(columns)
    clusters: list[list[str]] = []
    while remaining:
        seed = sorted(remaining)[0]
        cluster = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbours = {
                name for name in remaining
                if name != current and float(corr.loc[current, name]) >= threshold
            }
            new = neighbours - cluster
            cluster.update(new)
            frontier.extend(sorted(new))
        remaining -= cluster
        clusters.append(sorted(cluster))
    return clusters


__all__ = [
    "FactorGateConfig",
    "FactorPromotionContext",
    "FactorGovernanceReport",
    "evaluate_factor_candidate",
    "correlation_clusters",
]
