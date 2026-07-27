"""The decision protocol: sequencing, veto resolution, and the final record.

The protocol is deliberately not a vote. Given a set of envelopes it applies
four rules, in this order:

1. **Structural validity.** An envelope that cannot name its evidence is not
   counted as an approval. An APPROVE with no artifact hashes is downgraded to
   ``NEEDS_EVIDENCE`` -- it does not silently pass, and it does not fail the
   decision either; it asks for the missing evidence.
2. **Hard vetoes.** Any ``BLOCK`` from a veto-holding agent ends the decision as
   ``BLOCKED``. Confidence is not consulted.
3. **Mandatory coverage.** Data Quality, Risk, Compliance and Governance must
   each have spoken. A missing mandatory agent is ``NEEDS_EVIDENCE``, never an
   implicit approval, because "nobody checked" and "someone checked and it was
   fine" must not produce the same outcome.
4. **Sequence.** Agents are consulted in the declared order, and the first
   non-advancing verdict stops the run.

Everything the protocol does is written to the append-only audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from quantagent.governance import agents as roles
from quantagent.governance.audit import AuditLog
from quantagent.governance.envelopes import (
    APPROVE,
    BLOCK,
    NEEDS_EVIDENCE,
    REJECT,
    DecisionEnvelope,
)

# --- outcomes ---------------------------------------------------------------
OUTCOME_APPROVED = "APPROVED"
OUTCOME_REJECTED = "REJECTED"
OUTCOME_BLOCKED = "BLOCKED"
OUTCOME_NEEDS_EVIDENCE = "NEEDS_EVIDENCE"

OUTCOMES: tuple[str, ...] = (
    OUTCOME_APPROVED, OUTCOME_REJECTED, OUTCOME_BLOCKED, OUTCOME_NEEDS_EVIDENCE,
)


class LiveTradingAttempt(RuntimeError):
    """Raised when a decision would enable real-account order transmission."""


#: Action fragments that indicate a real-money path. Matched case-insensitively
#: against the proposed action; a hit is refused before any agent is consulted.
LIVE_TRADING_MARKERS: tuple[str, ...] = (
    "live trading", "live_trading", "real account", "real_account",
    "实盘", "submit order to broker", "enable_live", "order_send",
)


@dataclass
class DecisionRecord:
    action: str
    outcome: str
    scope: dict[str, Any] = field(default_factory=dict)
    consulted: list[str] = field(default_factory=list)
    approvals: list[str] = field(default_factory=list)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    evidence_gaps: list[dict[str, Any]] = field(default_factory=list)
    missing_mandatory: list[str] = field(default_factory=list)
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    downgraded_envelopes: list[dict[str, Any]] = field(default_factory=list)
    #: Reported for humans. Never used to decide the outcome.
    mean_confidence: float | None = None
    decided_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    notes: list[str] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return self.outcome == OUTCOME_APPROVED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"approved": self.approved}


def _is_live_trading(action: str) -> bool:
    lowered = action.lower()
    return any(marker.lower() in lowered for marker in LIVE_TRADING_MARKERS)


class DecisionProtocol:
    """Runs the approval sequence and resolves the outcome."""

    def __init__(
        self,
        audit_log: AuditLog | None = None,
        *,
        sequence: Sequence[str] = roles.APPROVAL_SEQUENCE,
        mandatory: Iterable[str] = roles.MANDATORY_AGENTS,
    ) -> None:
        self.audit_log = audit_log
        self.sequence = tuple(sequence)
        self.mandatory = frozenset(mandatory)

    # -- helpers ----------------------------------------------------------
    def required_agents(self, *, involves_intraday: bool) -> tuple[str, ...]:
        """The agents this decision must hear from, given its data scope."""
        if involves_intraday:
            return self.sequence
        return tuple(a for a in self.sequence if a not in roles.INTRADAY_ONLY_AGENTS)

    def _record(self, kind: str, actor: str, subject: str, payload: Mapping[str, Any]) -> None:
        if self.audit_log is not None:
            self.audit_log.append(kind=kind, actor=actor, subject=subject, payload=payload)

    # -- main entry point --------------------------------------------------
    def decide(
        self,
        action: str,
        envelopes: Sequence[DecisionEnvelope],
        *,
        involves_intraday: bool = False,
        scope: Mapping[str, Any] | None = None,
    ) -> DecisionRecord:
        """Resolve a decision from the supplied envelopes."""
        if _is_live_trading(action):
            self._record("LIVE_TRADING_REFUSED", "protocol", action,
                         {"reason": "live trading is disabled for this mission"})
            raise LiveTradingAttempt(
                f"refusing to govern {action!r}: this mission permits paper and "
                "dry-run execution only, so no decision path may authorise "
                "real-account order transmission"
            )

        record = DecisionRecord(action=action, outcome=OUTCOME_NEEDS_EVIDENCE,
                                scope=dict(scope or {}))
        required = self.required_agents(involves_intraday=involves_intraday)
        by_agent: dict[str, DecisionEnvelope] = {}

        for envelope in envelopes:
            self._record("ENVELOPE", envelope.agent, action, envelope.to_dict())
            problems = envelope.validate()
            if problems and envelope.verdict == APPROVE:
                # An unsupported approval is downgraded, not discarded: the
                # decision then reports *which* evidence is missing.
                record.downgraded_envelopes.append({
                    "agent": envelope.agent, "from": APPROVE, "to": NEEDS_EVIDENCE,
                    "problems": problems,
                })
                envelope = DecisionEnvelope(
                    **{**envelope.to_dict(), "verdict": NEEDS_EVIDENCE}
                )
            by_agent[envelope.agent] = envelope
            record.consulted.append(envelope.agent)

        # -- rule 2: hard vetoes, before anything else is weighed
        for agent_name, envelope in by_agent.items():
            if envelope.verdict != BLOCK:
                continue
            try:
                role = roles.role_for(agent_name)
            except KeyError:
                record.notes.append(
                    f"{agent_name} issued BLOCK but is not a declared role; "
                    "treated as advisory REJECT"
                )
                record.rejections.append({"agent": agent_name,
                                          "reasons": envelope.hard_blockers})
                continue
            if role.can_veto:
                record.blockers.append({
                    "agent": agent_name, "authority": role.authority,
                    "veto_domains": list(role.veto_domains),
                    "hard_blockers": list(envelope.hard_blockers),
                })
            else:
                record.notes.append(
                    f"{agent_name} issued BLOCK without veto authority; "
                    "recorded as a rejection"
                )
                record.rejections.append({"agent": agent_name,
                                          "reasons": envelope.hard_blockers})

        for agent_name, envelope in by_agent.items():
            if envelope.verdict == APPROVE:
                record.approvals.append(agent_name)
            elif envelope.verdict == REJECT:
                record.rejections.append({
                    "agent": agent_name,
                    "reasons": envelope.known_limitations or [envelope.method],
                })
            elif envelope.verdict == NEEDS_EVIDENCE:
                record.evidence_gaps.append({
                    "agent": agent_name,
                    "missing": envelope.validate() or ["agent requested more evidence"],
                })

        record.missing_mandatory = sorted(
            name for name in self.mandatory
            if name in required and name not in by_agent
        )

        confidences = [e.confidence for e in by_agent.values()]
        record.mean_confidence = (
            sum(confidences) / len(confidences) if confidences else None
        )

        # Disagreement is recorded whenever approvals coexist with any negative
        # verdict, so a split decision is visible even when the outcome is clear.
        if record.approvals and (record.blockers or record.rejections):
            record.disagreements.append({
                "approving": list(record.approvals),
                "blocking": [b["agent"] for b in record.blockers],
                "rejecting": [r["agent"] for r in record.rejections],
            })

        # -- outcome, in strict precedence order
        if record.blockers:
            record.outcome = OUTCOME_BLOCKED
            record.notes.append(
                "a hard veto ended this decision; approvals and confidence "
                "scores cannot overturn it"
            )
        elif record.missing_mandatory:
            record.outcome = OUTCOME_NEEDS_EVIDENCE
            record.notes.append(
                f"mandatory agents did not report: {record.missing_mandatory}; "
                "an absent check is not a passed check"
            )
        elif record.rejections:
            record.outcome = OUTCOME_REJECTED
        elif record.evidence_gaps:
            record.outcome = OUTCOME_NEEDS_EVIDENCE
        elif all(name in record.approvals for name in required):
            record.outcome = OUTCOME_APPROVED
        else:
            record.outcome = OUTCOME_NEEDS_EVIDENCE
            record.notes.append(
                "not every required agent approved: "
                f"{sorted(set(required) - set(record.approvals))}"
            )

        self._record("DECISION", roles.ORCHESTRATOR.name, action, record.to_dict())
        return record
