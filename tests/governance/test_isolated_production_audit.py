from __future__ import annotations

import pytest

from quantagent.governance.isolated_production_audit import (
    BACKTEST_ROLE,
    RISK_ROLE,
    STOCK_SELECTION_ROLE,
    TESTING_ROLE,
    AuditFinding,
    AuditReview,
    IsolatedAuditBoard,
)


def _finding(*, author: str = RISK_ROLE, severity: str = "P0") -> AuditFinding:
    return AuditFinding(
        finding_id="RISK-001",
        author_role=author,
        domain="live-risk",
        severity=severity,
        title="production risk state can be bypassed",
        evidence=("src/quantagent/execution/order_manager.py",),
        reproduction=("construct live OMS without a shared kill-switch state",),
        proposed_acceptance_tests=("broker submit calls stay zero when kill-switch is active",),
        proposed_fix="bind the live session to the shared persistent risk state",
    )


def _review(role: str, verdict: str = "approve") -> AuditReview:
    return AuditReview(
        finding_id="RISK-001",
        reviewer_role=role,
        verdict=verdict,
        evidence_checked=("source + regression test",),
    )


def test_author_cannot_review_own_finding() -> None:
    board = IsolatedAuditBoard()
    with pytest.raises(ValueError, match="self-review"):
        board.disposition(_finding(), [_review(RISK_ROLE), _review(TESTING_ROLE)])


def test_p0_requires_testing_role_even_with_two_other_approvals() -> None:
    board = IsolatedAuditBoard()
    result = board.disposition(
        _finding(),
        [_review(BACKTEST_ROLE), _review(STOCK_SELECTION_ROLE)],
    )
    assert result.ready_for_main_fix is False
    assert result.status == "needs_evidence"
    assert any("testing_expert" in reason for reason in result.reasons)


def test_cross_review_allows_main_fix_only_after_independent_test_approval() -> None:
    board = IsolatedAuditBoard()
    result = board.require_ready_for_main_fix(
        _finding(),
        [_review(BACKTEST_ROLE), _review(TESTING_ROLE)],
    )
    assert result.ready_for_main_fix is True
    assert set(result.independent_approvals) == {BACKTEST_ROLE, TESTING_ROLE}


def test_any_rejection_keeps_finding_contested() -> None:
    board = IsolatedAuditBoard()
    result = board.disposition(
        _finding(),
        [_review(BACKTEST_ROLE), _review(TESTING_ROLE), _review(STOCK_SELECTION_ROLE, "reject")],
    )
    assert result.status == "contested"
    assert result.ready_for_main_fix is False


def test_main_role_cannot_be_used_as_an_independent_reviewer() -> None:
    from quantagent.governance.isolated_production_audit import MAIN_ROLE

    with pytest.raises(ValueError, match="isolated auditor roles"):
        _review(MAIN_ROLE)
