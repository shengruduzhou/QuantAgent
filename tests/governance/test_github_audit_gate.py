from __future__ import annotations

import json

from quantagent.governance.github_audit_gate import (
    AUDIT_MARKER,
    audit_comment_template,
    evaluate_audit_comments,
)
from quantagent.governance.isolated_production_audit import (
    AI_QUANT_AUDIT_ROLE,
    QUANT_EXPERT_TESTER_ROLE,
    RISK_ROLE,
    TESTING_ROLE,
)


HEAD = "a" * 40
OLD = "b" * 40


def _comment(
    role: str,
    *,
    head: str = HEAD,
    verdict: str = "approve",
    repo_wide: bool = False,
    comment_id: int = 1,
    association: str = "OWNER",
) -> dict[str, object]:
    return {
        "id": comment_id,
        "body": audit_comment_template(
            head_sha=head,
            reviewer_role=role,
            verdict=verdict,
            repo_wide=repo_wide,
            evidence_checked=[f"evidence for {role}"],
        ),
        "author_association": association,
        "user": {"login": "maintainer"},
    }


def _accepted_set() -> list[dict[str, object]]:
    return [
        _comment(TESTING_ROLE, comment_id=1),
        _comment(QUANT_EXPERT_TESTER_ROLE, comment_id=2),
        _comment(AI_QUANT_AUDIT_ROLE, repo_wide=True, comment_id=3),
        _comment(RISK_ROLE, comment_id=4),
    ]


def test_exact_head_required_roles_and_domain_approval_pass() -> None:
    evaluation = evaluate_audit_comments(_accepted_set(), head_sha=HEAD)
    assert evaluation.passed is True
    assert evaluation.disposition.status == "accepted_for_merge"
    assert set(evaluation.disposition.approvals) >= {
        TESTING_ROLE,
        QUANT_EXPERT_TESTER_ROLE,
        AI_QUANT_AUDIT_ROLE,
        RISK_ROLE,
    }


def test_green_roles_for_old_sha_do_not_apply_to_new_head() -> None:
    stale = [dict(row, body=str(row["body"]).replace(HEAD, OLD)) for row in _accepted_set()]
    evaluation = evaluate_audit_comments(stale, head_sha=HEAD)
    assert evaluation.passed is False
    assert evaluation.stale_record_count == 4
    assert evaluation.disposition.missing_required_roles


def test_needs_evidence_vetoes_even_when_other_required_roles_approve() -> None:
    comments = _accepted_set()
    comments[2] = _comment(
        AI_QUANT_AUDIT_ROLE,
        verdict="needs_evidence",
        repo_wide=True,
        comment_id=3,
    )
    evaluation = evaluate_audit_comments(comments, head_sha=HEAD)
    assert evaluation.passed is False
    assert AI_QUANT_AUDIT_ROLE in evaluation.disposition.evidence_gaps


def test_role10_must_be_repo_wide() -> None:
    comments = _accepted_set()
    comments[2] = _comment(AI_QUANT_AUDIT_ROLE, repo_wide=False, comment_id=3)
    evaluation = evaluate_audit_comments(comments, head_sha=HEAD)
    assert evaluation.passed is False
    assert any("repo_wide=True" in reason for reason in evaluation.disposition.reasons)


def test_missing_domain_approval_fails() -> None:
    evaluation = evaluate_audit_comments(_accepted_set()[:3], head_sha=HEAD)
    assert evaluation.passed is False
    assert any("domain reviewer" in reason for reason in evaluation.disposition.reasons)


def test_duplicate_logical_role_fails_closed() -> None:
    comments = [*_accepted_set(), _comment(TESTING_ROLE, comment_id=5)]
    evaluation = evaluate_audit_comments(comments, head_sha=HEAD)
    assert evaluation.passed is False
    assert any(item.startswith("board:duplicate post-change review") for item in evaluation.malformed_comment_ids)


def test_free_text_approve_is_not_machine_audit_evidence() -> None:
    comments = [
        {"id": 99, "body": "testing_expert APPROVE; role10 APPROVE repo_wide=True", "author_association": "OWNER", "user": {"login": "maintainer"}}
    ]
    evaluation = evaluate_audit_comments(comments, head_sha=HEAD)
    assert evaluation.passed is False
    assert len(evaluation.accepted_records) == 0


def test_malformed_trusted_marker_fails_instead_of_being_ignored() -> None:
    comments = _accepted_set()
    comments.append(
        {
            "id": 100,
            "body": f"<!-- {AUDIT_MARKER}\nnot-json\n-->",
            "author_association": "OWNER",
            "user": {"login": "maintainer"},
        }
    )
    evaluation = evaluate_audit_comments(comments, head_sha=HEAD)
    assert evaluation.passed is False
    assert "100" in evaluation.malformed_comment_ids


def test_random_public_commenter_cannot_manufacture_or_veto_role_records() -> None:
    comments = _accepted_set()
    comments.append(_comment(RISK_ROLE, comment_id=101, association="NONE"))
    evaluation = evaluate_audit_comments(comments, head_sha=HEAD)
    assert evaluation.passed is True
    assert "101" in evaluation.unauthorized_comment_ids
    assert len(evaluation.accepted_records) == 4


def test_malformed_public_marker_is_ignored_before_json_parsing() -> None:
    comments = _accepted_set()
    comments.append(
        {
            "id": 103,
            "body": f"<!-- {AUDIT_MARKER}\nthis is maliciously malformed\n-->",
            "author_association": "NONE",
            "user": {"login": "random-user"},
        }
    )
    evaluation = evaluate_audit_comments(comments, head_sha=HEAD)
    assert evaluation.passed is True
    assert "103" in evaluation.unauthorized_comment_ids
    assert "103" not in evaluation.malformed_comment_ids


def test_unknown_fields_are_rejected_to_keep_schema_closed() -> None:
    body = audit_comment_template(
        head_sha=HEAD,
        reviewer_role=TESTING_ROLE,
        verdict="approve",
        evidence_checked=["CI"],
    )
    start = body.index("{")
    end = body.rindex("}") + 1
    payload = json.loads(body[start:end])
    payload["force"] = True
    malformed = (
        f"<!-- {AUDIT_MARKER}\n"
        + json.dumps(payload, sort_keys=True)
        + "\n-->"
    )
    evaluation = evaluate_audit_comments(
        [{"id": 102, "body": malformed, "author_association": "OWNER", "user": {"login": "maintainer"}}],
        head_sha=HEAD,
    )
    assert evaluation.passed is False
    assert "102" in evaluation.malformed_comment_ids
