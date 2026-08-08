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


def _accepted_payload() -> dict:
    return {
        "schema_version": 1,
        "status": "production_accepted",
        "model_id": "fresh-model-v1",
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


def test_every_required_acceptance_gate_must_pass(tmp_path: Path) -> None:
    payload = _accepted_payload()
    path = _write(tmp_path / "accepted.json", payload)
    assert evaluate_live_model_trust(path).ok is True

    for key, bad in (
        ("fresh_oos_days", 119),
        ("final_holdout_reads", 2),
        ("pbo", 0.251),
        ("dsr_probability", 0.949),
        ("spa_p_value", 0.051),
        ("variant_c_passed", False),
        ("benchmark_excess_positive", False),
        ("risk_capacity_passed", False),
        ("selection_pre_registered", False),
        ("contaminated_holdout", True),
        ("strict_backtest_metric_semantics", "legacy_post_trade_nav_baseline"),
    ):
        candidate = _accepted_payload()
        candidate["evidence"][key] = bad
        report = evaluate_live_model_trust(_write(tmp_path / f"bad-{key}.json", candidate))
        assert report.ok is False, key


def test_missing_metric_semantics_cannot_be_live_accepted(tmp_path: Path) -> None:
    candidate = _accepted_payload()
    candidate["evidence"].pop("strict_backtest_metric_semantics")
    report = evaluate_live_model_trust(_write(tmp_path / "missing-semantics.json", candidate))
    assert report.ok is False
    assert any("metric_semantics_mismatch" in reason for reason in report.reasons)


def test_repository_current_model_manifest_is_intentionally_blocked() -> None:
    # The current production blend is explicitly classified likely_overfit in
    # configs/production_blend.json. This regression prevents a future code
    # cleanup from silently turning the machine live gate green without fresh
    # evidence being written under the current trusted evaluator semantics.
    path = Path("configs/live_model_trust.json")
    report = evaluate_live_model_trust(path)
    assert report.ok is False
    assert report.status == "blocked"
    assert report.trust_class == "likely_overfit"
    assert any("pbo" in reason for reason in report.reasons)
    assert any("dsr_probability" in reason for reason in report.reasons)
    assert any("holdout" in reason for reason in report.reasons)
    assert any("metric_semantics" in reason for reason in report.reasons)
