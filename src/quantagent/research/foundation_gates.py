"""Fail-closed promotion gates for QuantAgent quantitative research.

This module deliberately sits *after* model/factor search.  Search is allowed to
produce interesting candidates; promotion is not.  A candidate can move from
research to a production-eligible state only when the core statistical,
point-in-time, benchmark and execution contracts are all explicitly evidenced.

The thresholds are intentionally policy, not optimiser hyperparameters.  They
must not be relaxed by the search procedure that they supervise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from quantagent.quant_math.performance import deflated_sharpe_ratio, sharpe_ratio, spa_test


@dataclass(frozen=True)
class ResearchGatePolicy:
    """Production research policy; missing evidence fails closed."""

    max_pbo: float = 0.25
    min_dsr_probability: float = 0.95
    max_spa_p_value: float = 0.05
    require_explicit_benchmark: bool = True
    require_pit: bool = True
    require_untouched_holdout: bool = True
    require_t_plus_one_for_close_signals: bool = True


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    observed: object
    required: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchGateReport:
    eligible: bool
    checks: tuple[GateCheck, ...]
    policy: ResearchGatePolicy

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(check.reason for check in self.checks if not check.passed)

    def as_dict(self) -> dict[str, object]:
        return {
            "promotionEligible": self.eligible,
            "policy": asdict(self.policy),
            "checks": [check.as_dict() for check in self.checks],
            "blockers": list(self.blockers),
        }


def evaluate_research_gates(
    *,
    pbo: float | None,
    dsr_probability: float | None,
    spa_p_value: float | None,
    benchmark_symbol: str | None,
    pit_valid: bool | None,
    holdout_untouched: bool | None,
    signal_at_close: bool = False,
    execution_lag_days: int | None = None,
    policy: ResearchGatePolicy | None = None,
) -> ResearchGateReport:
    """Evaluate the non-negotiable research promotion contract.

    ``None`` never means pass.  It means the run has not produced enough
    evidence to be promoted.
    """

    policy = policy or ResearchGatePolicy()
    checks: list[GateCheck] = []

    pbo_ok = pbo is not None and np.isfinite(pbo) and float(pbo) <= policy.max_pbo
    checks.append(GateCheck(
        "pbo", pbo_ok, pbo, f"<= {policy.max_pbo:.2f}",
        "PBO missing/non-finite or above the production ceiling",
    ))

    dsr_ok = dsr_probability is not None and np.isfinite(dsr_probability) and float(dsr_probability) >= policy.min_dsr_probability
    checks.append(GateCheck(
        "dsr_probability", dsr_ok, dsr_probability, f">= {policy.min_dsr_probability:.2f}",
        "Deflated Sharpe probability missing/non-finite or below the production floor",
    ))

    spa_ok = spa_p_value is not None and np.isfinite(spa_p_value) and float(spa_p_value) <= policy.max_spa_p_value
    checks.append(GateCheck(
        "spa_p_value", spa_ok, spa_p_value, f"<= {policy.max_spa_p_value:.2f}",
        "SPA evidence missing/non-finite or not significant after data-mining correction",
    ))

    benchmark_ok = bool((benchmark_symbol or "").strip()) if policy.require_explicit_benchmark else True
    checks.append(GateCheck(
        "explicit_benchmark", benchmark_ok, benchmark_symbol or "", "non-empty benchmark symbol",
        "benchmarkSymbol is required for an excess-return claim",
    ))

    pit_ok = pit_valid is True if policy.require_pit else pit_valid is not False
    checks.append(GateCheck(
        "point_in_time", pit_ok, pit_valid, "PIT validation == true",
        "point-in-time validity is missing or failed",
    ))

    holdout_ok = holdout_untouched is True if policy.require_untouched_holdout else holdout_untouched is not False
    checks.append(GateCheck(
        "untouched_holdout", holdout_ok, holdout_untouched, "final holdout untouched until acceptance",
        "final holdout isolation is missing or failed",
    ))

    if policy.require_t_plus_one_for_close_signals and signal_at_close:
        execution_ok = execution_lag_days is not None and int(execution_lag_days) >= 1
        checks.append(GateCheck(
            "close_signal_execution_lag", execution_ok, execution_lag_days, ">= 1 trading day",
            "close-derived signals must not execute on the same close",
        ))

    return ResearchGateReport(eligible=all(check.passed for check in checks), checks=tuple(checks), policy=policy)


def fusion_statistical_evidence(
    *,
    candidate_navs: Mapping[str, pd.Series],
    preferred_id: str | None,
    benchmark_returns: pd.Series | None,
    periods_per_year: int = 252,
) -> dict[str, float | str | None]:
    """Derive DSR and SPA evidence from already-OOS candidate NAVs.

    No model selection happens here.  The caller supplies the already-selected
    research candidate; this function only measures whether its OOS return
    record survives multiple-testing diagnostics.
    """

    returns: dict[str, pd.Series] = {}
    for name, nav in candidate_navs.items():
        clean = pd.Series(nav, dtype=float).dropna().sort_index()
        if len(clean) >= 2:
            r = clean.pct_change().dropna()
            if not r.empty:
                returns[name] = r

    if not preferred_id or preferred_id not in returns:
        return {"preferred": preferred_id, "dsrProbability": None, "spaPValue": None}

    preferred = returns[preferred_id]
    annual_sharpes = []
    for series in returns.values():
        value = sharpe_ratio(series, periods_per_year=periods_per_year)
        if np.isfinite(value):
            annual_sharpes.append(float(value))
    per_period_sharpes = np.asarray(annual_sharpes, dtype=float) / np.sqrt(periods_per_year)
    dsr = deflated_sharpe_ratio(
        preferred,
        per_period_sharpes,
        periods_per_year=periods_per_year,
        n_trials=max(len(candidate_navs), len(per_period_sharpes)),
    ) if len(per_period_sharpes) >= 2 else float("nan")

    spa_p = float("nan")
    if benchmark_returns is not None and returns:
        common = pd.DataFrame(returns).dropna(how="all")
        benchmark = pd.Series(benchmark_returns, dtype=float).reindex(common.index)
        spa = spa_test(common, benchmark)
        spa_p = float(spa.get("p_consistent", float("nan")))

    return {
        "preferred": preferred_id,
        "dsrProbability": None if not np.isfinite(dsr) else float(dsr),
        "spaPValue": None if not np.isfinite(spa_p) else float(spa_p),
    }


__all__ = [
    "GateCheck",
    "ResearchGatePolicy",
    "ResearchGateReport",
    "evaluate_research_gates",
    "fusion_statistical_evidence",
]
