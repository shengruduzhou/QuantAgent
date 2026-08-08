from __future__ import annotations

import pandas as pd

from quantagent.research.foundation_gates import (
    ResearchGatePolicy,
    evaluate_research_gates,
    fusion_statistical_evidence,
)


def test_default_policy_matches_production_research_thresholds() -> None:
    policy = ResearchGatePolicy()
    assert policy.max_pbo == 0.25
    assert policy.min_dsr_probability == 0.95
    assert policy.max_spa_p_value == 0.05
    assert policy.require_explicit_benchmark is True
    assert policy.require_pit is True
    assert policy.require_untouched_holdout is True


def test_missing_evidence_fails_closed() -> None:
    report = evaluate_research_gates(
        pbo=None,
        dsr_probability=None,
        spa_p_value=None,
        benchmark_symbol="",
        pit_valid=None,
        holdout_untouched=None,
    )
    assert report.eligible is False
    assert len(report.blockers) == 6


def test_all_required_evidence_can_pass() -> None:
    report = evaluate_research_gates(
        pbo=0.20,
        dsr_probability=0.97,
        spa_p_value=0.03,
        benchmark_symbol="000300.SH",
        pit_valid=True,
        holdout_untouched=True,
        signal_at_close=True,
        execution_lag_days=1,
    )
    assert report.eligible is True
    assert not report.blockers


def test_same_close_execution_is_blocked() -> None:
    report = evaluate_research_gates(
        pbo=0.20,
        dsr_probability=0.97,
        spa_p_value=0.03,
        benchmark_symbol="000300.SH",
        pit_valid=True,
        holdout_untouched=True,
        signal_at_close=True,
        execution_lag_days=0,
    )
    assert report.eligible is False
    assert any(check.name == "close_signal_execution_lag" and not check.passed for check in report.checks)


def test_fusion_evidence_reports_missing_benchmark_as_missing_spa() -> None:
    idx = pd.date_range("2025-01-01", periods=10, freq="B")
    navs = {
        "a": pd.Series([1.0 + i * 0.01 for i in range(10)], index=idx),
        "b": pd.Series([1.0 + i * 0.005 for i in range(10)], index=idx),
    }
    result = fusion_statistical_evidence(
        candidate_navs=navs,
        preferred_id="a",
        benchmark_returns=None,
    )
    assert result["preferred"] == "a"
    assert result["spaPValue"] is None
