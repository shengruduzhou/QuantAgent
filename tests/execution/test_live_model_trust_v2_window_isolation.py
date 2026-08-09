from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from quantagent.execution.live_model_evidence import (
    FreshPredictionEvidence,
    StatisticalEvidence,
    StrictReturnEvidence,
)
from quantagent.execution.live_model_trust_v2 import V2VerificationResult
from quantagent.execution import live_model_trust_v2_policy as policy


def _write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_strict_and_statistical_return_files_must_end_before_fresh_start(tmp_path: Path, monkeypatch) -> None:
    fresh_summary = _write(tmp_path / "fresh.json", {"trading_days": 130, "start_date": "2026-02-02", "end_date": "2026-07-31"})
    strict_summary = _write(tmp_path / "strict.json", {"portfolio_total_return": 0.10, "benchmark_total_return": 0.05, "benchmark_excess_positive": True})
    stats_summary = _write(tmp_path / "stats.json", {"selected_candidate": "ft", "pbo": 0.1, "dsr_probability": 0.99, "spa_p_value": 0.01})
    search = _write(tmp_path / "search.json", {"candidate_family": ["linear", "ft"], "selected_candidate": "ft", "trial_count": 10})
    prereg = _write(tmp_path / "prereg.json", {"candidate_family": ["linear", "ft"]})

    base = V2VerificationResult(
        ok=True,
        reasons=(),
        evidence={},
        resolved_paths={
            "fresh_oos_predictions": str(tmp_path / "fresh.csv"),
            "fresh_oos": fresh_summary,
            "strict_backtest_returns": str(tmp_path / "strict.csv"),
            "strict_backtest": strict_summary,
            "statistical_returns": str(tmp_path / "selection.csv"),
            "statistical_gates": stats_summary,
            "search_ledger": search,
            "pre_registration": prereg,
        },
    )
    monkeypatch.setattr(policy, "_verify_digest_bound_v2", lambda *args, **kwargs: base)
    monkeypatch.setattr(
        policy,
        "validate_fresh_predictions",
        lambda path: FreshPredictionEvidence(130, "2026-02-02", "2026-07-31", 260, 2),
    )
    monkeypatch.setattr(
        policy,
        "validate_strict_backtest_returns",
        lambda path: StrictReturnEvidence(120, "2025-08-18", "2026-02-02", 0.10, 0.05, True),
    )
    frozen_report = SimpleNamespace(
        pbo=0.1,
        dsr_probability=0.99,
        spa_pvalue=0.01,
        accepted=True,
        rejection_reasons=(),
        selected_candidate="ft",
        observed_days=160,
        losing_fold_rate=0.0,
    )
    monkeypatch.setattr(
        policy,
        "recompute_statistical_evidence",
        lambda *args, **kwargs: StatisticalEvidence(
            report=frozen_report,
            rows=160,
            start_date="2025-06-23",
            end_date="2026-02-02",
        ),
    )

    result = policy.verify_governed_live_model_trust_v2(
        {"schema_version": 2},
        artifact_roots={"bundle": tmp_path},
    )
    assert result.ok is False
    assert "strict_backtest_returns:end_not_strictly_before_fresh_oos" in result.reasons
    assert "statistical_returns:end_not_strictly_before_fresh_oos" in result.reasons


def test_return_files_ending_previous_session_do_not_trigger_window_violation(tmp_path: Path, monkeypatch) -> None:
    fresh_summary = _write(tmp_path / "fresh.json", {"trading_days": 130, "start_date": "2026-02-02", "end_date": "2026-07-31"})
    strict_summary = _write(tmp_path / "strict.json", {"portfolio_total_return": 0.10, "benchmark_total_return": 0.05, "benchmark_excess_positive": True})
    base = V2VerificationResult(
        ok=True,
        reasons=(),
        evidence={},
        resolved_paths={
            "fresh_oos_predictions": str(tmp_path / "fresh.csv"),
            "fresh_oos": fresh_summary,
            "strict_backtest_returns": str(tmp_path / "strict.csv"),
            "strict_backtest": strict_summary,
        },
    )
    monkeypatch.setattr(policy, "_verify_digest_bound_v2", lambda *args, **kwargs: base)
    monkeypatch.setattr(policy, "validate_fresh_predictions", lambda path: FreshPredictionEvidence(130, "2026-02-02", "2026-07-31", 260, 2))
    monkeypatch.setattr(policy, "validate_strict_backtest_returns", lambda path: StrictReturnEvidence(120, "2025-08-15", "2026-01-30", 0.10, 0.05, True))

    result = policy.verify_governed_live_model_trust_v2(
        {"schema_version": 2},
        artifact_roots={"bundle": tmp_path},
    )
    assert "strict_backtest_returns:end_not_strictly_before_fresh_oos" not in result.reasons
