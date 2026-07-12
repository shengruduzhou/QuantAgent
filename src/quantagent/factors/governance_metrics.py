"""Factor governance metrics for admission into model search.

A factor is not admitted because of one attractive full-sample IC.  It must
show point-in-time coverage, cross-sectional IC stability, decay consistent
with its intended horizon, acceptable correlation to the active library and a
capacity estimate compatible with the target book.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorGateConfig:
    min_dates: int = 120
    min_symbols_per_date: int = 30
    min_mean_rank_ic: float = 0.01
    min_ic_information_ratio: float = 0.20
    max_losing_period_rate: float = 0.40
    max_library_abs_correlation: float = 0.85
    min_decay_retention: float = 0.20
    max_decay_reversal: float = -0.02
    target_book_cny: float = 10_000_000.0
    max_adv_participation: float = 0.10
    min_capacity_multiple: float = 1.0


@dataclass
class FactorGovernanceReport:
    factor_name: str
    passed: bool
    mean_rank_ic: float
    ic_information_ratio: float
    losing_period_rate: float
    max_library_abs_correlation: float
    most_correlated_factor: str | None
    decay_curve: dict[int, float]
    decay_retention: float
    estimated_capacity_cny: float
    capacity_multiple: float
    coverage_dates: int
    median_symbols_per_date: float
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "factor_name": self.factor_name,
            "passed": self.passed,
            "mean_rank_ic": self.mean_rank_ic,
            "ic_information_ratio": self.ic_information_ratio,
            "losing_period_rate": self.losing_period_rate,
            "max_library_abs_correlation": self.max_library_abs_correlation,
            "most_correlated_factor": self.most_correlated_factor,
            "decay_curve": dict(self.decay_curve),
            "decay_retention": self.decay_retention,
            "estimated_capacity_cny": self.estimated_capacity_cny,
            "capacity_multiple": self.capacity_multiple,
            "coverage_dates": self.coverage_dates,
            "median_symbols_per_date": self.median_symbols_per_date,
            "rejection_reasons": list(self.rejection_reasons),
        }


def _rank_ic_by_date(frame: pd.DataFrame, factor_col: str, return_col: str) -> pd.Series:
    def _one(group: pd.DataFrame) -> float:
        valid = group[[factor_col, return_col]].dropna()
        if len(valid) < 3:
            return float("nan")
        return float(valid[factor_col].corr(valid[return_col], method="spearman"))

    return frame.groupby("trade_date", sort=True).apply(_one).dropna()


def _ic_ir(ic: pd.Series) -> float:
    if len(ic) < 2:
        return float("nan")
    std = float(ic.std(ddof=1))
    if std <= 1e-12:
        return float("nan")
    return float(ic.mean() / std * np.sqrt(len(ic)))


def _period_losing_rate(ic: pd.Series, frequency: str = "QE") -> float:
    if ic.empty:
        return 1.0
    period = ic.groupby(ic.index.to_period(frequency)).mean()
    return float((period <= 0).mean()) if not period.empty else 1.0


def _library_correlation(
    frame: pd.DataFrame,
    factor_col: str,
    library_columns: Iterable[str],
) -> tuple[float, str | None]:
    columns = [column for column in library_columns if column in frame.columns and column != factor_col]
    if not columns:
        return 0.0, None
    # Rank within date before correlation so scale and outliers do not dominate.
    ranked = frame[["trade_date", factor_col, *columns]].copy()
    ranked[[factor_col, *columns]] = ranked.groupby("trade_date")[[factor_col, *columns]].rank(pct=True)
    corr = ranked[[factor_col, *columns]].corr(method="spearman")[factor_col].drop(factor_col).abs()
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
    work = frame[["trade_date", factor_col, adv_col]].dropna().copy()
    if work.empty:
        return 0.0
    work["rank_pct"] = work.groupby("trade_date")[factor_col].rank(pct=True)
    selected = work[work["rank_pct"] >= 1.0 - top_quantile]
    if selected.empty:
        return 0.0
    # Each date's executable book is the sum of per-name participation limits.
    daily = selected.groupby("trade_date")[adv_col].sum() * max_adv_participation
    return float(daily.quantile(0.10)) if not daily.empty else 0.0


def evaluate_factor_candidate(
    frame: pd.DataFrame,
    *,
    factor_name: str,
    target_return_col: str,
    target_horizon_days: int,
    decay_return_columns: dict[int, str],
    library_columns: Iterable[str] = (),
    adv_col: str = "adv20_cny",
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
    losing_rate = _period_losing_rate(ic)
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
        reasons.append(
            f"median_symbols_per_date={median_symbols:.1f} below {cfg.min_symbols_per_date}"
        )
    if not np.isfinite(mean_ic) or mean_ic < cfg.min_mean_rank_ic:
        reasons.append(f"mean_rank_ic={mean_ic:.6f} below {cfg.min_mean_rank_ic:.6f}")
    if not np.isfinite(icir) or icir < cfg.min_ic_information_ratio:
        reasons.append(f"ic_information_ratio={icir:.4f} below {cfg.min_ic_information_ratio:.4f}")
    if losing_rate > cfg.max_losing_period_rate:
        reasons.append(
            f"losing_period_rate={losing_rate:.4f} exceeds {cfg.max_losing_period_rate:.4f}"
        )
    if max_corr > cfg.max_library_abs_correlation:
        reasons.append(
            f"library_correlation={max_corr:.4f} with {corr_name} exceeds "
            f"{cfg.max_library_abs_correlation:.4f}"
        )
    if not np.isfinite(retention) or retention < cfg.min_decay_retention:
        reasons.append(f"decay_retention={retention:.4f} below {cfg.min_decay_retention:.4f}")
    finite_curve = [value for value in curve.values() if np.isfinite(value)]
    if finite_curve and min(finite_curve) < cfg.max_decay_reversal:
        reasons.append(
            f"decay_curve reverses to {min(finite_curve):.4f} below {cfg.max_decay_reversal:.4f}"
        )
    if capacity_multiple < cfg.min_capacity_multiple:
        reasons.append(
            f"capacity_multiple={capacity_multiple:.3f} below {cfg.min_capacity_multiple:.3f}"
        )

    return FactorGovernanceReport(
        factor_name=factor_name,
        passed=not reasons,
        mean_rank_ic=mean_ic,
        ic_information_ratio=icir,
        losing_period_rate=losing_rate,
        max_library_abs_correlation=max_corr,
        most_correlated_factor=corr_name,
        decay_curve=curve,
        decay_retention=retention,
        estimated_capacity_cny=capacity,
        capacity_multiple=capacity_multiple,
        coverage_dates=coverage_dates,
        median_symbols_per_date=median_symbols,
        rejection_reasons=reasons,
    )


def correlation_clusters(
    factor_frame: pd.DataFrame,
    *,
    factor_columns: Iterable[str],
    threshold: float = 0.85,
) -> list[list[str]]:
    """Greedy absolute-correlation clusters for redundancy control."""
    columns = [column for column in factor_columns if column in factor_frame.columns]
    if not columns:
        return []
    ranked = factor_frame[["trade_date", *columns]].copy()
    ranked[columns] = ranked.groupby("trade_date")[columns].rank(pct=True)
    corr = ranked[columns].corr(method="spearman").abs()
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
