from __future__ import annotations

import pytest

from quantagent.governance.isolated_production_audit import (
    AI_QUANT_AUDIT_ROLE,
    BACKTEST_ROLE,
    QUANT_EXPERT_TESTER_ROLE,
    TESTING_ROLE,
    IsolatedAuditBoard,
    PostChangeReview,
)


CHANGE = "deadbeef"


def _review(
    role: str,
    verdict: str = "approve",
    *,
    repo_wide: bool = False,
) -> PostChangeReview:
    return PostChangeReview(
        change_id=CHANGE,
        reviewer_role=role,
        verdict=verdict,
        evidence_checked=("main + exact diff + tests + CI + cross-module impact",),
        repo_wide=repo_wide,
    )


def test_merge_requires_testing_quant_tester_ai_quant_and_domain_review() -> None:
    board = IsolatedAuditBoard()
    accepted = board.require_ready_for_merge(
        CHANGE,
        [
            _review(TESTING_ROLE),
            _review(QUANT_EXPERT_TESTER_ROLE),
            _review(AI_QUANT_AUDIT_ROLE, repo_wide=True),
            _review(BACKTEST_ROLE),
        ],
    )
    assert accepted.ready_for_merge is True
    assert set(accepted.approvals) == {
        TESTING_ROLE,
        QUANT_EXPERT_TESTER_ROLE,
        AI_QUANT_AUDIT_ROLE,
        BACKTEST_ROLE,
    }


def test_ai_quant_review_must_explicitly_be_repo_wide() -> None:
    board = IsolatedAuditBoard()
    result = board.post_change_disposition(
        CHANGE,
        [
            _review(TESTING_ROLE),
            _review(QUANT_EXPERT_TESTER_ROLE),
            _review(AI_QUANT_AUDIT_ROLE, repo_wide=False),
            _review(BACKTEST_ROLE),
        ],
    )
    assert result.ready_for_merge is False
    assert any("repo_wide=True" in reason for reason in result.reasons)


def test_green_test_roles_without_role10_cannot_authorize_merge() -> None:
    board = IsolatedAuditBoard()
    result = board.post_change_disposition(
        CHANGE,
        [_review(TESTING_ROLE), _review(QUANT_EXPERT_TESTER_ROLE), _review(BACKTEST_ROLE)],
    )
    assert result.ready_for_merge is False
    assert AI_QUANT_AUDIT_ROLE in result.missing_required_roles


def test_role10_and_tests_still_need_a_domain_discussion_approval() -> None:
    board = IsolatedAuditBoard()
    result = board.post_change_disposition(
        CHANGE,
        [
            _review(TESTING_ROLE),
            _review(QUANT_EXPERT_TESTER_ROLE),
            _review(AI_QUANT_AUDIT_ROLE, repo_wide=True),
        ],
    )
    assert result.ready_for_merge is False
    assert any("domain reviewer" in reason for reason in result.reasons)


def test_any_post_change_rejection_vetoes_merge() -> None:
    board = IsolatedAuditBoard()
    result = board.post_change_disposition(
        CHANGE,
        [
            _review(TESTING_ROLE),
            _review(QUANT_EXPERT_TESTER_ROLE),
            _review(AI_QUANT_AUDIT_ROLE, repo_wide=True),
            _review(BACKTEST_ROLE, "reject"),
        ],
    )
    assert result.status == "contested"
    assert result.ready_for_merge is False


def test_main_repair_role_cannot_pose_as_post_change_reviewer() -> None:
    from quantagent.governance.isolated_production_audit import MAIN_ROLE

    with pytest.raises(ValueError, match="isolated auditor roles"):
        _review(MAIN_ROLE)
