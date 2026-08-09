"""GitHub-facing exact-head adapter for :mod:`isolated_production_audit`.

The core audit board is repository-owned policy. This adapter parses a strict
machine-readable marker from trusted maintainer PR comments, binds every review
to one exact PR head SHA, and feeds the resulting ``PostChangeReview`` objects
into :class:`IsolatedAuditBoard`.

A human sentence containing words such as "APPROVE" is not evidence. Only the
versioned JSON marker below is accepted::

    <!-- quantagent-post-change-audit:v1
    {"head_sha":"<40hex>","reviewer_role":"testing_expert",
     "verdict":"approve","repo_wide":false,
     "evidence_checked":["CI run #123"],"notes":"..."}
    -->

GitHub identity and logical auditor role are different concepts. A marker from
a non-maintainer is ignored as untrusted input so a random public commenter
cannot manufacture evidence *or* denial-of-service the gate. Once a marker is
posted by a trusted OWNER/MEMBER/COLLABORATOR, malformed structured evidence
fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Iterable, Mapping, Sequence

from quantagent.governance.isolated_production_audit import (
    IsolatedAuditBoard,
    PostChangeDisposition,
    PostChangeReview,
)


AUDIT_MARKER = "quantagent-post-change-audit:v1"
AUDIT_CHECK_NAME = "isolated-multi-role-audit"
ALLOWED_AUTHOR_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_BLOCK_RE = re.compile(
    rf"<!--\s*{re.escape(AUDIT_MARKER)}\s*(\{{.*?\}})\s*-->",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedAuditRecord:
    head_sha: str
    review: PostChangeReview
    comment_id: int | str
    github_login: str
    author_association: str


@dataclass(frozen=True)
class AuditGateEvaluation:
    head_sha: str
    disposition: PostChangeDisposition
    accepted_records: tuple[ParsedAuditRecord, ...]
    stale_record_count: int
    malformed_comment_ids: tuple[str, ...]
    unauthorized_comment_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        # Untrusted audit-looking comments are diagnostics only. Letting them
        # veto would allow any public commenter to DoS every PR. Trusted malformed
        # records still fail closed.
        return self.disposition.ready_for_merge and not self.malformed_comment_ids

    def summary(self) -> str:
        lines = [
            f"head_sha: {self.head_sha}",
            f"disposition: {self.disposition.status}",
            f"accepted_roles: {', '.join(self.disposition.approvals) or '-'}",
            f"missing_required_roles: {', '.join(self.disposition.missing_required_roles) or '-'}",
            f"rejections: {', '.join(self.disposition.rejections) or '-'}",
            f"needs_evidence: {', '.join(self.disposition.evidence_gaps) or '-'}",
            f"stale_records_ignored: {self.stale_record_count}",
        ]
        if self.malformed_comment_ids:
            lines.append("malformed_trusted_audit_comments: " + ",".join(self.malformed_comment_ids))
        if self.unauthorized_comment_ids:
            lines.append("untrusted_audit_comments_ignored: " + ",".join(self.unauthorized_comment_ids))
        lines.extend(f"reason: {reason}" for reason in self.disposition.reasons)
        return "\n".join(lines)


class AuditCommentError(ValueError):
    pass


def _comment_id(comment: Mapping[str, object]) -> str:
    return str(comment.get("id", "unknown"))


def _login(comment: Mapping[str, object]) -> str:
    user = comment.get("user")
    if isinstance(user, Mapping):
        return str(user.get("login", ""))
    return ""


def _association(comment: Mapping[str, object]) -> str:
    return str(comment.get("author_association", "")).upper()


def _extract_payload(body: str) -> dict[str, object] | None:
    if AUDIT_MARKER not in body:
        return None
    matches = _BLOCK_RE.findall(body)
    if len(matches) != 1:
        raise AuditCommentError("audit comment must contain exactly one complete v1 marker block")
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise AuditCommentError(f"invalid audit JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise AuditCommentError("audit JSON must be an object")
    return payload


def parse_audit_comment(
    comment: Mapping[str, object],
    *,
    expected_head_sha: str,
    allowed_author_associations: frozenset[str] = ALLOWED_AUTHOR_ASSOCIATIONS,
) -> ParsedAuditRecord | None:
    """Parse one trusted audit comment.

    A non-audit comment returns ``None``. An audit-looking comment from an
    untrusted GitHub author raises ``PermissionError`` *before JSON parsing* so
    malformed public input cannot veto the gate. Trusted malformed evidence
    raises :class:`AuditCommentError` and fails closed at the evaluation layer.
    """

    if not _SHA_RE.fullmatch(str(expected_head_sha).lower()):
        raise ValueError("expected_head_sha must be an exact 40-character lowercase hex SHA")
    body = str(comment.get("body", "") or "")
    if AUDIT_MARKER not in body:
        return None
    association = _association(comment)
    if association not in allowed_author_associations:
        raise PermissionError(
            f"audit comment {_comment_id(comment)} has untrusted author_association={association!r}"
        )
    payload = _extract_payload(body)
    if payload is None:  # pragma: no cover - marker check above makes this defensive
        return None

    required = {
        "head_sha",
        "reviewer_role",
        "verdict",
        "repo_wide",
        "evidence_checked",
    }
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - (required | {"notes"}))
    if missing:
        raise AuditCommentError(f"audit JSON missing fields: {missing}")
    if unknown:
        raise AuditCommentError(f"audit JSON has unknown fields: {unknown}")

    head_sha = str(payload["head_sha"]).lower()
    if not _SHA_RE.fullmatch(head_sha):
        raise AuditCommentError("head_sha must be an exact 40-character lowercase hex SHA")
    repo_wide = payload["repo_wide"]
    if type(repo_wide) is not bool:  # bool specifically; 1/0 must not coerce.
        raise AuditCommentError("repo_wide must be a JSON boolean")
    evidence = payload["evidence_checked"]
    if not isinstance(evidence, list) or not evidence or not all(
        isinstance(item, str) and item.strip() for item in evidence
    ):
        raise AuditCommentError("evidence_checked must be a non-empty string array")

    review = PostChangeReview(
        change_id=head_sha,
        reviewer_role=str(payload["reviewer_role"]),
        verdict=str(payload["verdict"]),
        evidence_checked=tuple(item.strip() for item in evidence),
        repo_wide=repo_wide,
        notes=str(payload.get("notes", "")),
    )
    return ParsedAuditRecord(
        head_sha=head_sha,
        review=review,
        comment_id=comment.get("id", "unknown"),
        github_login=_login(comment),
        author_association=association,
    )


def evaluate_audit_comments(
    comments: Sequence[Mapping[str, object]],
    *,
    head_sha: str,
    board: IsolatedAuditBoard | None = None,
) -> AuditGateEvaluation:
    head_sha = str(head_sha).lower()
    if not _SHA_RE.fullmatch(head_sha):
        raise ValueError("current PR head must be an exact 40-character lowercase hex SHA")
    board = board or IsolatedAuditBoard()
    accepted: list[ParsedAuditRecord] = []
    stale = 0
    malformed: list[str] = []
    unauthorized: list[str] = []

    for comment in comments:
        body = str(comment.get("body", "") or "")
        if AUDIT_MARKER not in body:
            continue
        try:
            record = parse_audit_comment(comment, expected_head_sha=head_sha)
        except PermissionError:
            unauthorized.append(_comment_id(comment))
            continue
        except (AuditCommentError, ValueError):
            malformed.append(_comment_id(comment))
            continue
        if record is None:
            continue
        if record.head_sha != head_sha:
            stale += 1
            continue
        accepted.append(record)

    try:
        disposition = board.post_change_disposition(
            head_sha,
            [record.review for record in accepted],
        )
    except ValueError as exc:
        malformed.append(f"board:{exc}")
        disposition = board.post_change_disposition(head_sha, [])

    return AuditGateEvaluation(
        head_sha=head_sha,
        disposition=disposition,
        accepted_records=tuple(accepted),
        stale_record_count=stale,
        malformed_comment_ids=tuple(malformed),
        unauthorized_comment_ids=tuple(unauthorized),
    )


def audit_comment_template(
    *,
    head_sha: str,
    reviewer_role: str,
    verdict: str,
    evidence_checked: Iterable[str],
    repo_wide: bool = False,
    notes: str = "",
) -> str:
    payload = {
        "head_sha": head_sha,
        "reviewer_role": reviewer_role,
        "verdict": verdict,
        "repo_wide": bool(repo_wide),
        "evidence_checked": list(evidence_checked),
        "notes": notes,
    }
    return (
        f"<!-- {AUDIT_MARKER}\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        + "\n-->"
    )


__all__ = [
    "AUDIT_MARKER",
    "AUDIT_CHECK_NAME",
    "ALLOWED_AUTHOR_ASSOCIATIONS",
    "ParsedAuditRecord",
    "AuditGateEvaluation",
    "AuditCommentError",
    "parse_audit_comment",
    "evaluate_audit_comments",
    "audit_comment_template",
]
