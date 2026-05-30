"""Canonical EvidenceRecord schema and cross-source adapters.

This package owns the unified schema every v8 evidence builder must
project into when consumers want to reason across sources (policy /
bond / broker / state_team / news / financials). The v8.2 builders
keep emitting their native silver layer; the adapters in
:mod:`.canonical` produce a v1 EvidenceRecord without rewriting them.

Downstream consumers (capital_flow_thesis builder, daily decision
report, contradiction checker) should consume the canonical schema
exclusively to remain source-agnostic.
"""

from quantagent.data.evidence.canonical import (
    CANONICAL_EVIDENCE_COLUMNS,
    CANONICAL_SOURCE_TYPES,
    EvidenceRecord,
    bond_flows_to_evidence,
    broker_reports_to_evidence,
    policy_events_to_evidence,
    state_team_events_to_evidence,
    to_canonical_evidence_frame,
    validate_pit_safety,
)


__all__ = [
    "CANONICAL_EVIDENCE_COLUMNS",
    "CANONICAL_SOURCE_TYPES",
    "EvidenceRecord",
    "bond_flows_to_evidence",
    "broker_reports_to_evidence",
    "policy_events_to_evidence",
    "state_team_events_to_evidence",
    "to_canonical_evidence_frame",
    "validate_pit_safety",
]
