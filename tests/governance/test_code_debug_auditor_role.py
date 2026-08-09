from __future__ import annotations

import pytest

from quantagent.governance.isolated_production_audit import (
    AI_QUANT_AUDIT_ROLE,
    AUDITOR_ROLES,
    CODE_DEBUG_ROLE,
    QUANT_EXPERT_TESTER_ROLE,
    TESTING_ROLE,
    AuditFinding,
    AuditReview,
    IsolatedAuditBoard,
    PostChangeReview,
)


CHANGE = "role9-head"


def _post(role: str, *, repo_wide: bool = False) -> PostChangeReview:
    return PostChangeReview(
        change_id=CHANGE,
        reviewer_role=role,
        verdict="approve",
        evidence_checked=("exact head + diff + regression tests + cross-module call sites",),
        repo_wide=repo_wide,
    )


def test_code_debug_role_is_a_first_class_isolated_auditor() -> None:
    assert CODE_DEBUG_ROLE in AUDITOR_ROLES
    finding = AuditFinding(
        finding_id="CODE-001",
        author_role=CODE_DEBUG_ROLE,
        domain="code_quality",
        severity="P1",
        title="Fail-open branch in trust verifier",
        evidence=("reproduced on exact head",),
        reproduction=("construct boundary fixture",),
        proposed_acceptance_tests=("fixture must fail closed",),
    )
    assert finding.author_role == CODE_DEBUG_ROLE


def test_code_debug_author_cannot_self_review_its_own_finding() -> None:
    board = IsolatedAuditBoard()
    finding = AuditFinding(
        finding_id="CODE-002",
        author_role=CODE_DEBUG_ROLE,
        domain="code_quality",
        severity="P2",
        title="Duplicate unsafe branch",
        evidence=("exact code path",),
        reproduction=("run deterministic reproducer",),
        proposed_acceptance_tests=("regression test",),
    )
    review = AuditReview(
        finding_id=finding.finding_id,
        reviewer_role=CODE_DEBUG_ROLE,
        verdict="approve",
        evidence_checked=("own finding",),
    )
    with pytest.raises(ValueError, match="self-review is prohibited"):
        board.disposition(finding, [review])


def test_code_debug_can_supply_domain_review_but_not_replace_required_testers() -> None:
    board = IsolatedAuditBoard()
    blocked = board.post_change_disposition(
        CHANGE,
        [
            _post(CODE_DEBUG_ROLE),
            _post(AI_QUANT_AUDIT_ROLE, repo_wide=True),
        ],
    )
    assert blocked.ready_for_merge is False
    assert TESTING_ROLE in blocked.missing_required_roles
    assert QUANT_EXPERT_TESTER_ROLE in blocked.missing_required_roles

    accepted = board.require_ready_for_merge(
        CHANGE,
        [
            _post(TESTING_ROLE),
            _post(QUANT_EXPERT_TESTER_ROLE),
            _post(AI_QUANT_AUDIT_ROLE, repo_wide=True),
            _post(CODE_DEBUG_ROLE),
        ],
    )
    assert accepted.ready_for_merge is True
    assert CODE_DEBUG_ROLE in accepted.approvals
