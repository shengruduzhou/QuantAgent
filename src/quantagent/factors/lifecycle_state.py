"""Stateful factor lifecycle with append-only transition evidence.

A one-window diagnostic is not a capital-allocation state.  This module keeps
factor promotion deliberately slower than factor discovery:

candidate -> validated -> shadow -> active -> degraded -> retired

PIT/leakage/execution-semantic violations can quarantine immediately.  ACTIVE
cannot be reached without an explicit promotion-ready evidence bundle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Iterable


FACTOR_STAGES = (
    "candidate",
    "validated",
    "shadow",
    "active",
    "degraded",
    "retired",
    "quarantined",
)
TERMINAL_STAGES = frozenset({"retired", "quarantined"})
LIFECYCLE_SCHEMA_VERSION = "factor_lifecycle_state_v1"


@dataclass(frozen=True)
class LifecycleEvidence:
    core_validity_passed: bool
    promotion_ready: bool = False
    shadow_days: int = 0
    severe_semantic_violation: bool = False
    evidence_digest: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if int(self.shadow_days) < 0:
            raise ValueError("shadow_days cannot be negative")
        if not str(self.evidence_digest).strip():
            raise ValueError("lifecycle evidence requires an evidence_digest")


@dataclass(frozen=True)
class FactorLifecycleSnapshot:
    factor_name: str
    factor_version: str
    stage: str = "candidate"
    consecutive_degradations: int = 0
    last_evidence_digest: str = ""

    def __post_init__(self) -> None:
        if self.stage not in FACTOR_STAGES:
            raise ValueError(f"unknown factor lifecycle stage {self.stage!r}")
        if int(self.consecutive_degradations) < 0:
            raise ValueError("consecutive_degradations cannot be negative")


@dataclass(frozen=True)
class FactorLifecycleTransition:
    schema_version: str
    factor_name: str
    factor_version: str
    from_stage: str
    to_stage: str
    consecutive_degradations: int
    evidence_digest: str
    reason: str
    actor: str
    observed_at: str
    previous_record_hash: str
    record_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _transition_hash(payload: dict[str, object]) -> str:
    material = dict(payload)
    material.pop("record_hash", None)
    return sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def decide_lifecycle_transition(
    current: FactorLifecycleSnapshot,
    evidence: LifecycleEvidence,
    *,
    retire_after_consecutive_degradations: int = 3,
) -> FactorLifecycleSnapshot:
    """Pure transition function used by both runtime code and tests."""

    if retire_after_consecutive_degradations < 2:
        raise ValueError("retirement requires at least two consecutive degradation observations")
    if current.stage in TERMINAL_STAGES:
        return current
    if evidence.severe_semantic_violation:
        return FactorLifecycleSnapshot(
            factor_name=current.factor_name,
            factor_version=current.factor_version,
            stage="quarantined",
            consecutive_degradations=current.consecutive_degradations,
            last_evidence_digest=evidence.evidence_digest,
        )

    stage = current.stage
    degraded_count = current.consecutive_degradations
    if stage == "candidate":
        next_stage = "validated" if evidence.core_validity_passed else "candidate"
        degraded_count = 0
    elif stage == "validated":
        next_stage = "shadow" if evidence.core_validity_passed and evidence.shadow_days > 0 else "validated"
        degraded_count = 0
    elif stage == "shadow":
        if not evidence.core_validity_passed:
            next_stage = "degraded"
            degraded_count = 1
        elif evidence.promotion_ready:
            next_stage = "active"
            degraded_count = 0
        else:
            next_stage = "shadow"
            degraded_count = 0
    elif stage == "active":
        if evidence.core_validity_passed:
            next_stage = "active"
            degraded_count = 0
        else:
            next_stage = "degraded"
            degraded_count = 1
    elif stage == "degraded":
        if evidence.core_validity_passed and evidence.promotion_ready:
            next_stage = "active"
            degraded_count = 0
        elif evidence.core_validity_passed:
            # Recovery without a still-valid promotion bundle returns to shadow;
            # it does not silently regain capital.
            next_stage = "shadow"
            degraded_count = 0
        else:
            degraded_count = int(degraded_count) + 1
            next_stage = (
                "retired"
                if degraded_count >= retire_after_consecutive_degradations
                else "degraded"
            )
    else:  # pragma: no cover - protected by snapshot validation
        raise ValueError(f"unsupported lifecycle stage {stage!r}")

    return FactorLifecycleSnapshot(
        factor_name=current.factor_name,
        factor_version=current.factor_version,
        stage=next_stage,
        consecutive_degradations=degraded_count,
        last_evidence_digest=evidence.evidence_digest,
    )


class FactorLifecycleLedger:
    """Append-only hash-chained lifecycle record.

    The ledger does not autonomously run research or change portfolio weights.
    It records a decision produced from already-governed evidence.
    """

    def __init__(self, path: str | Path = "runtime/state/factor_lifecycle.jsonl") -> None:
        self.path = Path(path)

    def records(self) -> list[FactorLifecycleTransition]:
        if not self.path.exists():
            return []
        rows: list[FactorLifecycleTransition] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(FactorLifecycleTransition(**payload))
        return rows

    def verify(self) -> bool:
        previous = ""
        for record in self.records():
            payload = record.to_dict()
            if record.previous_record_hash != previous:
                return False
            if _transition_hash(payload) != record.record_hash:
                return False
            previous = record.record_hash
        return True

    def latest(self, factor_name: str, factor_version: str) -> FactorLifecycleSnapshot:
        relevant = [
            record
            for record in self.records()
            if record.factor_name == factor_name and record.factor_version == factor_version
        ]
        if not relevant:
            return FactorLifecycleSnapshot(factor_name=factor_name, factor_version=factor_version)
        record = relevant[-1]
        return FactorLifecycleSnapshot(
            factor_name=factor_name,
            factor_version=factor_version,
            stage=record.to_stage,
            consecutive_degradations=int(record.consecutive_degradations),
            last_evidence_digest=record.evidence_digest,
        )

    def observe(
        self,
        factor_name: str,
        factor_version: str,
        evidence: LifecycleEvidence,
        *,
        actor: str = "factor_strategy_expert",
        retire_after_consecutive_degradations: int = 3,
        observed_at: str | None = None,
    ) -> FactorLifecycleSnapshot:
        if not self.verify():
            raise RuntimeError("factor lifecycle ledger hash chain is invalid")
        current = self.latest(factor_name, factor_version)
        updated = decide_lifecycle_transition(
            current,
            evidence,
            retire_after_consecutive_degradations=retire_after_consecutive_degradations,
        )
        records = self.records()
        previous_hash = records[-1].record_hash if records else ""
        timestamp = observed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload: dict[str, object] = {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "factor_name": factor_name,
            "factor_version": factor_version,
            "from_stage": current.stage,
            "to_stage": updated.stage,
            "consecutive_degradations": int(updated.consecutive_degradations),
            "evidence_digest": evidence.evidence_digest,
            "reason": evidence.reason,
            "actor": actor,
            "observed_at": timestamp,
            "previous_record_hash": previous_hash,
            "record_hash": "",
        }
        payload["record_hash"] = _transition_hash(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        return updated


def replay_lifecycle(records: Iterable[FactorLifecycleTransition]) -> dict[tuple[str, str], FactorLifecycleSnapshot]:
    snapshots: dict[tuple[str, str], FactorLifecycleSnapshot] = {}
    for record in records:
        key = (record.factor_name, record.factor_version)
        snapshots[key] = FactorLifecycleSnapshot(
            factor_name=record.factor_name,
            factor_version=record.factor_version,
            stage=record.to_stage,
            consecutive_degradations=record.consecutive_degradations,
            last_evidence_digest=record.evidence_digest,
        )
    return snapshots


__all__ = [
    "FACTOR_STAGES",
    "TERMINAL_STAGES",
    "LIFECYCLE_SCHEMA_VERSION",
    "LifecycleEvidence",
    "FactorLifecycleSnapshot",
    "FactorLifecycleTransition",
    "FactorLifecycleLedger",
    "decide_lifecycle_transition",
    "replay_lifecycle",
]
