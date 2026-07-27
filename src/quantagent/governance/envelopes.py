"""Structured decision envelopes for the multi-agent research protocol.

An envelope is what one agent says about one proposed action. It exists so that
disagreement is resolved on *evidence and authority*, not on rhetoric or an
average confidence score.

Three properties do the real work:

**Evidence is addressed by hash, not described.** ``input_artifact_hashes``
names the exact artifacts a verdict was computed from. An agent that cannot
name its inputs cannot approve anything -- :meth:`DecisionEnvelope.validate`
rejects an APPROVE with no evidence, which is the structural fix for "the model
said it was fine".

**Hard blockers are not votes.** A ``BLOCK`` from an agent holding veto
authority ends the decision. No quantity of high-confidence approvals can
overturn it, because the failure mode being prevented -- five optimistic agents
outvoting the one that checked the data -- is exactly how these systems fail.

**Confidence is reported, never aggregated into authority.** It is metadata for
a human reader. The protocol never multiplies, averages or thresholds it to
decide an outcome.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

# --- verdicts ---------------------------------------------------------------
APPROVE = "APPROVE"
REJECT = "REJECT"
#: A hard stop. Only agents with veto authority may issue it, and it cannot be
#: outvoted.
BLOCK = "BLOCK"
#: "I cannot decide from what I was given." Distinct from REJECT: it asks for
#: more evidence rather than judging the proposal.
NEEDS_EVIDENCE = "NEEDS_EVIDENCE"

VERDICTS: tuple[str, ...] = (APPROVE, REJECT, BLOCK, NEEDS_EVIDENCE)
#: Verdicts that let the approval sequence continue to the next agent.
ADVANCING_VERDICTS: frozenset[str] = frozenset({APPROVE})


class EnvelopeError(ValueError):
    """Raised when an envelope is structurally invalid."""


def artifact_hash(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file on disk, for citing evidence by content."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def payload_hash(payload: Any) -> str:
    """Stable SHA-256 of a JSON-serialisable payload."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class DecisionEnvelope:
    """One agent's evidence-backed position on one proposed action."""

    agent: str
    hypothesis_or_action: str
    verdict: str
    decision_id: str = field(default_factory=lambda: str(uuid4()))
    input_artifact_hashes: dict[str, str] = field(default_factory=dict)
    data_scope: dict[str, Any] = field(default_factory=dict)
    method: str = ""
    quantitative_evidence: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)
    hard_blockers: list[str] = field(default_factory=list)
    confidence: float = 0.0
    output_artifacts: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise EnvelopeError(
                f"unknown verdict {self.verdict!r}; known verdicts: {list(VERDICTS)}"
            )
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise EnvelopeError(f"confidence must be in [0, 1], got {self.confidence}")

    def validate(self) -> list[str]:
        """Structural problems that make this envelope unusable as evidence."""
        problems: list[str] = []
        if not self.agent:
            problems.append("envelope has no agent")
        if not self.hypothesis_or_action:
            problems.append("envelope names no action")
        if self.verdict == APPROVE:
            if not self.input_artifact_hashes:
                problems.append(
                    "APPROVE with no input_artifact_hashes: an approval that "
                    "cannot name the artifacts it read is not evidence"
                )
            if not self.quantitative_evidence:
                problems.append(
                    "APPROVE with no quantitative_evidence: a verdict must rest "
                    "on a measurement, not an impression"
                )
            if not self.method:
                problems.append("APPROVE with no stated method")
        if self.verdict == BLOCK and not self.hard_blockers:
            problems.append("BLOCK with no hard_blockers listed")
        return problems

    @property
    def is_valid(self) -> bool:
        return not self.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("created_at", None)
        payload.pop("decision_id", None)
        return payload_hash(payload)
