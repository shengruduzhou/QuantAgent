from __future__ import annotations

import json
from pathlib import Path

from quantagent.execution.live_model_trust import (
    REQUIRED_METRIC_SEMANTICS,
    evaluate_live_model_trust,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _forged_v1_all_pass_payload() -> dict:
    """The historical self-attested shape that must never unlock live again."""
    return {
        "schema_version": 1,
        "status": "production_accepted",
        "model_id": "forged-model-v1",
        "trust_class": "fresh_holdout_validated",
        "evidence": {
            "fresh_oos_days": 130,
            "final_holdout_reads": 1,
            "pbo": 0.20,
            "dsr_probability": 0.97,
            "spa_p_value": 0.02,
            "variant_c_passed": True,
            "benchmark_excess_positive": True,
            "risk_capacity_passed": True,
            "selection_pre_registered": True,
            "contaminated_holdout": False,
            "strict_backtest_metric_semantics": REQUIRED_METRIC_SEMANTICS,
        },
    }


def test_missing_manifest_fails_closed(tmp_path: Path) -> None:
    report = evaluate_live_model_trust(tmp_path / "missing.json")
    assert report.ok is False
    assert "model_trust_manifest_not_found" in report.reasons


def test_forged_all_pass_schema_v1_is_never_production_eligible(tmp_path: Path) -> None:
    path = _write(tmp_path / "forged-v1.json", _forged_v1_all_pass_payload())
    report = evaluate_live_model_trust(path)
    assert report.ok is False
    assert "legacy_schema_v1_not_production_eligible" in report.reasons


def test_legacy_v1_still_reports_gate_diagnostics(tmp_path: Path) -> None:
    candidate = _forged_v1_all_pass_payload()
    candidate["evidence"]["pbo"] = 0.80
    candidate["evidence"]["dsr_probability"] = 0.50
    candidate["evidence"]["contaminated_holdout"] = True
    report = evaluate_live_model_trust(_write(tmp_path / "legacy-bad.json", candidate))
    assert report.ok is False
    assert "legacy_schema_v1_not_production_eligible" in report.reasons
    assert any("pbo_above" in reason for reason in report.reasons)
    assert any("dsr_probability_below" in reason for reason in report.reasons)
    assert any("holdout_contamination" in reason for reason in report.reasons)


def test_missing_metric_semantics_cannot_be_live_accepted_even_in_v1(tmp_path: Path) -> None:
    candidate = _forged_v1_all_pass_payload()
    candidate["evidence"].pop("strict_backtest_metric_semantics")
    report = evaluate_live_model_trust(_write(tmp_path / "missing-semantics.json", candidate))
    assert report.ok is False
    assert any("metric_semantics_mismatch" in reason for reason in report.reasons)


def test_unsupported_future_schema_fails_closed(tmp_path: Path) -> None:
    payload = _forged_v1_all_pass_payload()
    payload["schema_version"] = 99
    report = evaluate_live_model_trust(_write(tmp_path / "future.json", payload))
    assert report.ok is False
    assert "model_trust_schema_version_unsupported:99" in report.reasons


def test_repository_current_model_manifest_is_intentionally_blocked() -> None:
    # The current production blend remains a forensic schema-v1 BLOCKED record.
    # Adding schema-v2 support must not rewrite, bless or auto-migrate it.
    path = Path("configs/live_model_trust.json")
    report = evaluate_live_model_trust(path)
    assert report.ok is False
    assert report.status == "blocked"
    assert report.trust_class == "likely_overfit"
    assert any("pbo" in reason for reason in report.reasons)
    assert any("dsr_probability" in reason for reason in report.reasons)
    assert any("holdout" in reason for reason in report.reasons)
    assert any("metric_semantics" in reason for reason in report.reasons)
