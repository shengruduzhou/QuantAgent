from __future__ import annotations

from dataclasses import asdict
import json

from quantagent.factors.lifecycle_state import (
    FactorLifecycleLedger,
    FactorLifecycleSnapshot,
    LifecycleEvidence,
    decide_lifecycle_transition,
    replay_lifecycle,
)


def _evidence(**kwargs) -> LifecycleEvidence:
    defaults = {
        "core_validity_passed": True,
        "promotion_candidate_ready": False,
        "shadow_days": 0,
        "severe_semantic_violation": False,
        "evidence_digest": "abc123",
        "reason": "test",
    }
    defaults.update(kwargs)
    return LifecycleEvidence(**defaults)


def test_candidate_cannot_jump_directly_to_active_even_with_promotion_candidate_evidence() -> None:
    candidate = FactorLifecycleSnapshot("factor_x", "v1")
    updated = decide_lifecycle_transition(
        candidate,
        _evidence(promotion_candidate_ready=True, shadow_days=30),
    )
    assert updated.stage == "validated"


def test_validated_can_enter_shadow_but_preliminary_evidence_cannot_activate() -> None:
    snapshot = FactorLifecycleSnapshot("factor_x", "v1")
    snapshot = decide_lifecycle_transition(snapshot, _evidence())
    assert snapshot.stage == "validated"
    snapshot = decide_lifecycle_transition(snapshot, _evidence(evidence_digest="shadow-1", shadow_days=1))
    assert snapshot.stage == "shadow"
    snapshot = decide_lifecycle_transition(
        snapshot,
        _evidence(evidence_digest="shadow-30", shadow_days=30, promotion_candidate_ready=True),
    )
    assert snapshot.stage == "shadow"


def test_existing_active_factor_can_remain_active_when_validity_is_healthy() -> None:
    active = FactorLifecycleSnapshot("factor_x", "v1", stage="active")
    observed = decide_lifecycle_transition(
        active,
        _evidence(core_validity_passed=True, promotion_candidate_ready=False),
    )
    assert observed.stage == "active"


def test_degradation_requires_repeated_observations_before_retirement() -> None:
    active = FactorLifecycleSnapshot("factor_x", "v1", stage="active")
    degraded = decide_lifecycle_transition(
        active,
        _evidence(core_validity_passed=False, evidence_digest="independent-window-0"),
    )
    assert degraded.stage == "degraded"
    assert degraded.consecutive_degradations == 1

    still_degraded = decide_lifecycle_transition(
        degraded,
        _evidence(core_validity_passed=False, evidence_digest="independent-window-1"),
    )
    assert still_degraded.stage == "degraded"
    assert still_degraded.consecutive_degradations == 2

    retired = decide_lifecycle_transition(
        still_degraded,
        _evidence(core_validity_passed=False, evidence_digest="independent-window-2"),
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


def test_recovery_from_degraded_even_with_preliminary_promotion_evidence_returns_to_shadow() -> None:
    degraded = FactorLifecycleSnapshot(
        "factor_x",
        "v1",
        stage="degraded",
        consecutive_degradations=2,
    )
    recovered = decide_lifecycle_transition(
        degraded,
        _evidence(core_validity_passed=True, promotion_candidate_ready=True),
    )
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


def test_replayed_evidence_does_not_advance_or_retire(tmp_path) -> None:
    active = FactorLifecycleSnapshot("factor_x", "v1", stage="active")
    evidence = _evidence(core_validity_passed=False)
    degraded = decide_lifecycle_transition(active, evidence)
    for _ in range(4):
        degraded = decide_lifecycle_transition(degraded, evidence)
    assert degraded.stage == "degraded"
    assert degraded.consecutive_degradations == 1
    ledger = FactorLifecycleLedger(tmp_path / "ledger.jsonl")
    ledger.observe("factor_x", "v1", _evidence(evidence_digest="first"))
    state = ledger.observe("factor_x", "v1", _evidence(evidence_digest="second", shadow_days=1))
    before = ledger.path.read_bytes()
    assert ledger.observe("factor_x", "v1", _evidence(evidence_digest="first")) == state
    assert ledger.path.read_bytes() == before


def test_nonconsecutive_evidence_replay_does_not_retire_a_pure_snapshot() -> None:
    state = FactorLifecycleSnapshot("factor_x", "v1", stage="active")
    for digest in ("window-a", "window-b", "window-a"):
        state = decide_lifecycle_transition(state, _evidence(core_validity_passed=False, evidence_digest=digest))
    assert (state.stage, state.consecutive_degradations) == ("degraded", 2)
    assert state.seen_evidence_digests == ("window-a", "window-b")
    state = decide_lifecycle_transition(state, _evidence(core_validity_passed=False, evidence_digest="window-c"))
    assert (state.stage, state.consecutive_degradations) == ("retired", 3)


def test_old_failure_cannot_degrade_a_recovered_snapshot_again() -> None:
    state = FactorLifecycleSnapshot("factor_x", "v1", stage="active")
    failure = _evidence(core_validity_passed=False, evidence_digest="failure")
    state = decide_lifecycle_transition(state, failure)
    recovered = decide_lifecycle_transition(state, _evidence(evidence_digest="recovery"))
    assert (recovered.stage, recovered.consecutive_degradations) == ("shadow", 0)
    assert decide_lifecycle_transition(recovered, failure) == recovered


def test_ledger_and_replay_snapshots_preserve_factor_version_evidence_history(tmp_path) -> None:
    ledger = FactorLifecycleLedger(tmp_path / "ledger.jsonl")
    for digest in ("validity", "shadow"):
        ledger.observe("factor_x", "v1", _evidence(evidence_digest=digest, shadow_days=1))
    failure = _evidence(core_validity_passed=False, evidence_digest="failure-a")
    ledger.observe("factor_x", "v1", failure)
    ledger.observe("factor_x", "v1", _evidence(core_validity_passed=False, evidence_digest="failure-b"))
    other = ledger.observe("factor_x", "v2", _evidence(evidence_digest="failure-a"))
    assert other.stage == "validated"
    latest = ledger.latest("factor_x", "v1")
    replayed = replay_lifecycle(ledger.records())[("factor_x", "v1")]
    restored = FactorLifecycleSnapshot(**json.loads(json.dumps(asdict(latest))))
    for snapshot in (latest, replayed, restored):
        assert snapshot == latest
        assert decide_lifecycle_transition(snapshot, failure) == snapshot
        assert (snapshot.stage, snapshot.consecutive_degradations) == ("degraded", 2)
    before = ledger.path.read_bytes()
    assert ledger.observe("factor_x", "v1", failure) == latest
    assert ledger.path.read_bytes() == before
    assert ledger.verify() is True


def test_reused_digest_does_not_suppress_a_severe_semantic_violation() -> None:
    state = FactorLifecycleSnapshot("factor_x", "v1", stage="active")
    state = decide_lifecycle_transition(state, _evidence(evidence_digest="seen"))
    quarantined = decide_lifecycle_transition(
        state, _evidence(evidence_digest="seen", severe_semantic_violation=True)
    )
    assert quarantined.stage == "quarantined"
    assert quarantined.seen_evidence_digests == ("seen",)


def test_legacy_snapshot_still_deduplicates_its_last_known_digest() -> None:
    state = FactorLifecycleSnapshot(
        "factor_x", "v1", stage="degraded", consecutive_degradations=1, last_evidence_digest="known"
    )
    assert decide_lifecycle_transition(state, _evidence(core_validity_passed=False, evidence_digest="known")) == state
