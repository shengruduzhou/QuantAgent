from __future__ import annotations

from dataclasses import replace

from quantagent.factors.lifecycle_state import (
    FactorLifecycleLedger,
    FactorLifecycleSnapshot,
    LifecycleEvidence,
    decide_lifecycle_transition,
)


def _evidence(**kwargs) -> LifecycleEvidence:
    defaults = {
        "core_validity_passed": True,
        "promotion_ready": False,
        "shadow_days": 0,
        "severe_semantic_violation": False,
        "evidence_digest": "abc123",
        "reason": "test",
    }
    defaults.update(kwargs)
    return LifecycleEvidence(**defaults)


def test_candidate_cannot_jump_directly_to_active_even_with_promotion_ready() -> None:
    candidate = FactorLifecycleSnapshot("factor_x", "v1")
    updated = decide_lifecycle_transition(
        candidate,
        _evidence(promotion_ready=True, shadow_days=30),
    )
    assert updated.stage == "validated"


def test_validated_requires_shadow_before_active() -> None:
    snapshot = FactorLifecycleSnapshot("factor_x", "v1")
    snapshot = decide_lifecycle_transition(snapshot, _evidence())
    assert snapshot.stage == "validated"
    snapshot = decide_lifecycle_transition(snapshot, _evidence(shadow_days=1))
    assert snapshot.stage == "shadow"
    snapshot = decide_lifecycle_transition(
        snapshot,
        _evidence(shadow_days=30, promotion_ready=True),
    )
    assert snapshot.stage == "active"


def test_degradation_requires_repeated_observations_before_retirement() -> None:
    active = FactorLifecycleSnapshot("factor_x", "v1", stage="active")
    degraded = decide_lifecycle_transition(
        active,
        _evidence(core_validity_passed=False),
    )
    assert degraded.stage == "degraded"
    assert degraded.consecutive_degradations == 1

    still_degraded = decide_lifecycle_transition(
        degraded,
        _evidence(core_validity_passed=False),
    )
    assert still_degraded.stage == "degraded"
    assert still_degraded.consecutive_degradations == 2

    retired = decide_lifecycle_transition(
        still_degraded,
        _evidence(core_validity_passed=False),
    )
    assert retired.stage == "retired"
    assert retired.consecutive_degradations == 3


def test_semantic_violation_quarantines_immediately() -> None:
    active = FactorLifecycleSnapshot("factor_x", "v1", stage="active")
    quarantined = decide_lifecycle_transition(
        active,
        _evidence(severe_semantic_violation=True, reason="pit_violation"),
    )
    assert quarantined.stage == "quarantined"


def test_recovery_from_degraded_without_promotion_proof_returns_to_shadow() -> None:
    degraded = FactorLifecycleSnapshot(
        "factor_x",
        "v1",
        stage="degraded",
        consecutive_degradations=2,
    )
    recovered = decide_lifecycle_transition(degraded, _evidence(core_validity_passed=True))
    assert recovered.stage == "shadow"
    assert recovered.consecutive_degradations == 0


def test_hash_chained_ledger_persists_state_and_detects_tamper(tmp_path) -> None:
    path = tmp_path / "factor_lifecycle.jsonl"
    ledger = FactorLifecycleLedger(path)
    first = ledger.observe("factor_x", "v1", _evidence(evidence_digest="e1"))
    second = ledger.observe(
        "factor_x",
        "v1",
        _evidence(evidence_digest="e2", shadow_days=1),
    )
    assert first.stage == "validated"
    assert second.stage == "shadow"
    assert ledger.verify() is True
    assert ledger.latest("factor_x", "v1").stage == "shadow"

    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"evidence_digest": "e1"', '"evidence_digest": "evil"', 1), encoding="utf-8")
    assert ledger.verify() is False
