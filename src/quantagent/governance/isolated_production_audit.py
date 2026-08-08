"""Isolated multi-role production audit protocol.

This module formalises a review pattern in which domain auditors cannot mutate
production code and cannot approve their own findings.  A finding becomes
eligible for the main repair role only after independent cross-review.

The roles intentionally mirror the production-review board used for QuantAgent:
main repair, backtest, risk, factor/strategy, stock selection, testing, system
user, senior quant tester, and design/testing.  The board is not a majority-vote
mechanism: a substantive rejection keeps a finding contested, and P0/P1 fixes
require the testing role to approve the reproduction/acceptance plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable


MAIN_ROLE = "main_repair"
BACKTEST_ROLE = "backtest_expert"
RISK_ROLE = "risk_expert"
FACTOR_STRATEGY_ROLE = "factor_strategy_expert"
STOCK_SELECTION_ROLE = "stock_selection_expert"
TESTING_ROLE = "testing_expert"
QUANT_USER_ROLE = "quant_department_user"
QUANT_EXPERT_TESTER_ROLE = "quant_expert_tester"
DESIGN_TESTING_ROLE = "design_testing_expert"

AUDITOR_ROLES: frozenset[str] = frozenset(
    {
        BACKTEST_ROLE,
        RISK_ROLE,
        FACTOR_STRATEGY_ROLE,
        STOCK_SELECTION_ROLE,
        TESTING_ROLE,
        QUANT_USER_ROLE,
        QUANT_EXPERT_TESTER_ROLE,
        DESIGN_TESTING_ROLE,
    }
)
ALL_PRODUCTION_AUDIT_ROLES: frozenset[str] = frozenset({MAIN_ROLE, *AUDITOR_ROLES})
SEVERITIES = frozenset({"P0", "P1", "P2", "P3"})
REVIEW_VERDICTS = frozenset({"approve", "reject", "needs_evidence"})


@dataclass(frozen=True)
class AuditFinding:
    finding_id: str
    author_role: str
    domain: str
    severity: str
    title: str
    evidence: tuple[str, ...]
    reproduction: tuple[str, ...]
    proposed_acceptance_tests: tuple[str, ...]
    proposed_fix: str = ""

    def __post_init__(self) -> None:
        if self.author_role not in AUDITOR_ROLES:
            raise ValueError("only isolated auditor roles may author findings")
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")
        if not self.finding_id.strip() or not self.title.strip():
            raise ValueError("finding_id and title are required")
        if not self.evidence or not self.reproduction:
            raise ValueError("a finding requires evidence and a reproduction")
        if not self.proposed_acceptance_tests:
            raise ValueError("a finding requires explicit acceptance tests")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AuditReview:
    finding_id: str
    reviewer_role: str
    verdict: str
    evidence_checked: tuple[str, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        if self.reviewer_role not in AUDITOR_ROLES:
            raise ValueError("only isolated auditor roles may review findings")
        if self.verdict not in REVIEW_VERDICTS:
            raise ValueError(f"unknown review verdict {self.verdict!r}")
        if not self.evidence_checked:
            raise ValueError("review must state what evidence was checked")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AuditDisposition:
    finding_id: str
    status: str
    independent_approvals: tuple[str, ...]
    rejections: tuple[str, ...]
    evidence_gaps: tuple[str, ...]
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ready_for_main_fix(self) -> bool:
        return self.status == "accepted_for_main_fix"

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {"ready_for_main_fix": self.ready_for_main_fix}


class IsolatedAuditBoard:
    """Cross-review findings before the main role is allowed to repair them."""

    def __init__(self, *, min_independent_approvals: int = 2) -> None:
        if min_independent_approvals < 2:
            raise ValueError("production audit requires at least two independent approvals")
        self.min_independent_approvals = int(min_independent_approvals)

    def disposition(
        self,
        finding: AuditFinding,
        reviews: Iterable[AuditReview],
    ) -> AuditDisposition:
        relevant = [review for review in reviews if review.finding_id == finding.finding_id]
        if any(review.reviewer_role == finding.author_role for review in relevant):
            raise ValueError("self-review is prohibited: the finding author cannot review its own finding")

        by_role: dict[str, AuditReview] = {}
        for review in relevant:
            if review.reviewer_role in by_role:
                raise ValueError(f"duplicate review from {review.reviewer_role}")
            by_role[review.reviewer_role] = review

        approvals = tuple(sorted(r.reviewer_role for r in relevant if r.verdict == "approve"))
        rejections = tuple(sorted(r.reviewer_role for r in relevant if r.verdict == "reject"))
        gaps = tuple(sorted(r.reviewer_role for r in relevant if r.verdict == "needs_evidence"))
        reasons: list[str] = []

        if rejections:
            status = "contested"
            reasons.append("one or more independent reviewers rejected the finding")
        elif gaps:
            status = "needs_evidence"
            reasons.append("one or more reviewers require additional evidence")
        elif len(approvals) < self.min_independent_approvals:
            status = "needs_evidence"
            reasons.append(
                f"requires {self.min_independent_approvals} independent approvals; got {len(approvals)}"
            )
        elif finding.severity in {"P0", "P1"} and TESTING_ROLE not in approvals:
            status = "needs_evidence"
            reasons.append("P0/P1 findings require testing_expert approval")
        else:
            status = "accepted_for_main_fix"
            reasons.append("cross-review threshold satisfied; main role may implement the scoped fix")

        return AuditDisposition(
            finding_id=finding.finding_id,
            status=status,
            independent_approvals=approvals,
            rejections=rejections,
            evidence_gaps=gaps,
            reasons=tuple(reasons),
        )

    def require_ready_for_main_fix(
        self,
        finding: AuditFinding,
        reviews: Iterable[AuditReview],
    ) -> AuditDisposition:
        disposition = self.disposition(finding, reviews)
        if not disposition.ready_for_main_fix:
            raise RuntimeError(
                f"finding {finding.finding_id} is {disposition.status}; main repair is prohibited"
            )
        return disposition


__all__ = [
    "MAIN_ROLE",
    "BACKTEST_ROLE",
    "RISK_ROLE",
    "FACTOR_STRATEGY_ROLE",
    "STOCK_SELECTION_ROLE",
    "TESTING_ROLE",
    "QUANT_USER_ROLE",
    "QUANT_EXPERT_TESTER_ROLE",
    "DESIGN_TESTING_ROLE",
    "AUDITOR_ROLES",
    "ALL_PRODUCTION_AUDIT_ROLES",
    "AuditFinding",
    "AuditReview",
    "AuditDisposition",
    "IsolatedAuditBoard",
]
