from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quantagent.quant_math.performance import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
    spa_test,
)
from quantagent.research.model_comparison import ComparisonReport, LINEAR_BASELINE


@dataclass(frozen=True)
class NonlinearPromotionConfig:
    """Fail-closed research-to-production gates for nonlinear factor models."""

    max_pbo: float = 0.25
    min_dsr_probability: float = 0.95
    max_spa_pvalue: float = 0.05
    spa_bootstrap: int = 1000

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_pbo <= 1.0:
            raise ValueError("max_pbo must be in [0,1]")
        if not 0.0 <= self.min_dsr_probability <= 1.0:
            raise ValueError("min_dsr_probability must be in [0,1]")
        if not 0.0 <= self.max_spa_pvalue <= 1.0:
            raise ValueError("max_spa_pvalue must be in [0,1]")
        if self.spa_bootstrap < 100:
            raise ValueError("spa_bootstrap must be >= 100")


@dataclass(frozen=True)
class NonlinearPromotionReport:
    raw_verdict: str
    final_verdict: str
    champion: str
    pbo: float
    dsr_probability: float
    spa_pvalue: float
    accepted: bool
    rejection_reasons: tuple[str, ...]
    config: NonlinearPromotionConfig
    cohort_count: int = 0
    periods_per_year: int = 252

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["config"] = asdict(self.config)
        return payload


def _non_overlapping_cohorts(series: pd.Series, horizon: int) -> list[pd.Series]:
    clean = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if clean.empty:
        return []
    horizon = max(1, int(horizon))
    return [clean.iloc[offset::horizon] for offset in range(min(horizon, len(clean))) if len(clean.iloc[offset::horizon]) >= 4]


def _cohort_frame(arms: dict[str, pd.Series], horizon: int, offset: int) -> pd.DataFrame:
    pieces: dict[str, pd.Series] = {}
    for name, series in arms.items():
        clean = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        cohort = clean.iloc[offset::horizon]
        if not cohort.empty:
            pieces[name] = cohort
    return pd.DataFrame(pieces).dropna(how="all")


def _worst_case_pbo(arms: dict[str, pd.Series], horizon: int) -> tuple[float, int]:
    values: list[float] = []
    used = 0
    for offset in range(horizon):
        frame = _cohort_frame(arms, horizon, offset).dropna(how="any")
        if frame.shape[1] < 2 or len(frame) < 8:
            continue
        partitions = min(16, len(frame))
        partitions -= partitions % 2
        if partitions < 4:
            continue
        try:
            value = probability_of_backtest_overfitting(frame, n_partitions=partitions)
        except ValueError:
            continue
        if np.isfinite(value):
            values.append(float(value))
            used += 1
    return (max(values) if values else float("nan"), used)


def _worst_case_dsr(
    arms: dict[str, pd.Series],
    champion: str,
    *,
    horizon: int,
    n_trials: int,
) -> tuple[float, int]:
    values: list[float] = []
    periods = max(1, int(round(252 / horizon)))
    used = 0
    for offset in range(horizon):
        frame = _cohort_frame(arms, horizon, offset).dropna(how="any")
        if champion not in frame or frame.shape[1] < 2 or len(frame) < 4:
            continue
        per_period_sharpes: list[float] = []
        for column in frame.columns:
            annual = sharpe_ratio(frame[column], periods_per_year=periods)
            if np.isfinite(annual):
                per_period_sharpes.append(float(annual) / np.sqrt(periods))
        if len(per_period_sharpes) < 2:
            continue
        value = deflated_sharpe_ratio(
            frame[champion],
            np.asarray(per_period_sharpes, dtype=float),
            periods_per_year=periods,
            n_trials=max(int(n_trials), len(per_period_sharpes)),
        )
        if np.isfinite(value):
            values.append(float(value))
            used += 1
    return (min(values) if values else float("nan"), used)


def _worst_case_spa(
    arms: dict[str, pd.Series],
    *,
    horizon: int,
    n_bootstrap: int,
) -> tuple[float, int]:
    values: list[float] = []
    used = 0
    for offset in range(horizon):
        frame = _cohort_frame(arms, horizon, offset).dropna(how="any")
        if LINEAR_BASELINE not in frame or frame.shape[1] < 2 or len(frame) < 8:
            continue
        challengers = frame.drop(columns=[LINEAR_BASELINE], errors="ignore")
        if challengers.empty:
            continue
        spa = spa_test(
            challengers,
            frame[LINEAR_BASELINE],
            n_bootstrap=n_bootstrap,
            rng_seed=offset,
        )
        value = float(spa.get("p_consistent", float("nan")))
        if np.isfinite(value):
            values.append(value)
            used += 1
    return (max(values) if values else float("nan"), used)


def evaluate_nonlinear_promotion(
    comparison: ComparisonReport,
    *,
    config: NonlinearPromotionConfig | None = None,
) -> NonlinearPromotionReport:
    """Apply conservative PBO/DSR/SPA to a frozen nonlinear champion.

    ``ComparisonReport.daily_returns`` are deliberately labelled as overlapping
    daily-equivalent returns because each observation starts from an H-day
    forward label. Promotion therefore never reuses the raw aggregate PBO/DSR.
    It evaluates every ``offset::H`` non-overlapping cohort independently and
    takes the worst valid evidence across cohorts: maximum PBO, minimum DSR,
    and maximum SPA p-value. Missing cohort evidence fails closed.
    """
    cfg = config or NonlinearPromotionConfig()
    reasons: list[str] = []
    raw = str(comparison.verdict)
    champion = str(comparison.champion or "")
    horizon = max(1, int(comparison.config.horizon_days))
    periods = max(1, int(round(252 / horizon)))

    measured = {
        arm.name: arm.daily_returns.sort_index()
        for arm in comparison.arms
        if arm.status == "measured" and not arm.daily_returns.empty
    }
    if raw != "production_accepted" or not champion or champion == LINEAR_BASELINE:
        reasons.append(
            f"model comparison did not freeze a promotable nonlinear champion: verdict={raw}, champion={champion or 'none'}"
        )
    if LINEAR_BASELINE not in measured or champion not in measured:
        reasons.append("measured linear baseline and frozen nonlinear champion returns are required")

    pbo, pbo_cohorts = _worst_case_pbo(measured, horizon)
    dsr, dsr_cohorts = _worst_case_dsr(
        measured,
        champion,
        horizon=horizon,
        n_trials=max(1, int(comparison.n_trials)),
    )
    spa_pvalue, spa_cohorts = _worst_case_spa(
        measured,
        horizon=horizon,
        n_bootstrap=cfg.spa_bootstrap,
    )
    cohort_count = min(pbo_cohorts, dsr_cohorts, spa_cohorts)

    if not np.isfinite(pbo) or pbo > cfg.max_pbo:
        reasons.append(
            f"worst-cohort pbo={pbo:.4f} exceeds {cfg.max_pbo:.4f}"
            if np.isfinite(pbo)
            else "non-overlapping-cohort PBO is unavailable"
        )
    if not np.isfinite(dsr) or dsr < cfg.min_dsr_probability:
        reasons.append(
            f"worst-cohort dsr={dsr:.4f} below {cfg.min_dsr_probability:.4f}"
            if np.isfinite(dsr)
            else "non-overlapping-cohort DSR is unavailable"
        )
    if not np.isfinite(spa_pvalue) or spa_pvalue > cfg.max_spa_pvalue:
        reasons.append(
            f"worst-cohort spa_pvalue={spa_pvalue:.4f} exceeds {cfg.max_spa_pvalue:.4f}"
            if np.isfinite(spa_pvalue)
            else "non-overlapping-cohort SPA p-value is unavailable"
        )
    if cohort_count < 1:
        reasons.append("no cohort produced complete PBO/DSR/SPA promotion evidence")

    accepted = not reasons
    return NonlinearPromotionReport(
        raw_verdict=raw,
        final_verdict="production_accepted" if accepted else "hypothesis_rejected",
        champion=champion,
        pbo=float(pbo),
        dsr_probability=float(dsr),
        spa_pvalue=float(spa_pvalue),
        accepted=accepted,
        rejection_reasons=tuple(reasons),
        config=cfg,
        cohort_count=cohort_count,
        periods_per_year=periods,
    )


__all__ = ["NonlinearPromotionConfig", "NonlinearPromotionReport", "evaluate_nonlinear_promotion"]
