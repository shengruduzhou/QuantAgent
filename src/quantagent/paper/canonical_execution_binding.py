"""Bind paper execution outcomes to immutable canonical-ledger prefixes.

The execution journal proves that *its own* rows were not edited.  That alone is
not enough to prove that an observed/blocked outcome still refers to the same
economic record of account: a canonical ledger could be replaced by another
valid hash chain and the journal would continue to verify independently.

This module binds an execution outcome to the canonical chain at two points:

* immediately before any OMS/broker economic mutation; and
* after session settlement, before the terminal outcome is journaled.

Later replays verify those prefix heads rather than requiring the canonical
ledger's current head to remain unchanged.  Therefore normal future sessions may
append records, while truncation/replacement/rewrite of the prefix that supported
an older outcome fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from quantagent.domain.ledger import CanonicalLedger, GENESIS_HASH


CANONICAL_EXECUTION_BINDING_SCHEMA = "paper_canonical_execution_prefix_v1"
_BOUND_TERMINAL_STATUSES = frozenset({"execution_observed", "execution_blocked"})


class CanonicalExecutionBindingError(RuntimeError):
    """Execution evidence is no longer bound to the canonical economic chain."""


@dataclass(frozen=True, slots=True)
class CanonicalLedgerPoint:
    records: int
    head_hash: str

    def to_dict(self, *, prefix: str) -> dict[str, object]:
        return {
            f"canonical_{prefix}_records": int(self.records),
            f"canonical_{prefix}_head": str(self.head_hash),
        }


def canonical_ledger_point(ledger: CanonicalLedger) -> CanonicalLedgerPoint:
    verification = ledger.verify()
    if not verification.get("valid"):
        raise CanonicalExecutionBindingError(
            f"canonical ledger does not verify: {verification}"
        )
    records = ledger.read()
    head = records[-1].record_hash if records else GENESIS_HASH
    if len(records) != len(ledger):
        raise CanonicalExecutionBindingError(
            "canonical ledger read/count disagreement while capturing execution binding"
        )
    return CanonicalLedgerPoint(records=len(records), head_hash=head)


def canonical_prefix_head(ledger: CanonicalLedger, record_count: int) -> str:
    records = ledger.read()
    count = int(record_count)
    if count < 0 or count > len(records):
        raise CanonicalExecutionBindingError(
            f"canonical prefix length {count} outside current ledger length {len(records)}"
        )
    if count == 0:
        return GENESIS_HASH
    return str(records[count - 1].record_hash)


def binding_details(
    *,
    before: CanonicalLedgerPoint,
    after: CanonicalLedgerPoint,
) -> dict[str, object]:
    if int(after.records) < int(before.records):
        raise CanonicalExecutionBindingError(
            "canonical record count moved backwards during execution"
        )
    return {
        "canonical_binding_schema": CANONICAL_EXECUTION_BINDING_SCHEMA,
        **before.to_dict(prefix="before"),
        **after.to_dict(prefix="after"),
    }


def verify_terminal_canonical_binding(
    *,
    status: str,
    details: Mapping[str, object],
    ledger: CanonicalLedger,
) -> None:
    """Verify the canonical prefix used by one economic terminal outcome.

    Non-economic terminal states (missed/indeterminate) intentionally carry no
    canonical binding because no completed economic execution is being claimed.
    An observed/blocked terminal created before this governance contract is not
    trusted silently: it requires manual reconciliation/migration rather than an
    automatic retry or an unqualified success claim.
    """

    if str(status) not in _BOUND_TERMINAL_STATUSES:
        return
    if details.get("canonical_binding_schema") != CANONICAL_EXECUTION_BINDING_SCHEMA:
        raise CanonicalExecutionBindingError(
            "economic execution terminal lacks supported canonical-prefix binding"
        )
    required = (
        "canonical_before_records",
        "canonical_before_head",
        "canonical_after_records",
        "canonical_after_head",
    )
    missing = [name for name in required if name not in details]
    if missing:
        raise CanonicalExecutionBindingError(
            f"economic execution terminal missing canonical binding fields: {missing}"
        )
    try:
        before_records = int(details["canonical_before_records"])
        after_records = int(details["canonical_after_records"])
    except (TypeError, ValueError) as exc:
        raise CanonicalExecutionBindingError(
            "canonical binding record counts are invalid"
        ) from exc
    if after_records < before_records:
        raise CanonicalExecutionBindingError(
            "canonical execution binding record count moved backwards"
        )
    verification = ledger.verify()
    if not verification.get("valid"):
        raise CanonicalExecutionBindingError(
            f"canonical ledger no longer verifies: {verification}"
        )
    if len(ledger) < after_records:
        raise CanonicalExecutionBindingError(
            "canonical ledger is shorter than the terminal-bound execution prefix"
        )
    observed_before = canonical_prefix_head(ledger, before_records)
    observed_after = canonical_prefix_head(ledger, after_records)
    if observed_before != str(details["canonical_before_head"]):
        raise CanonicalExecutionBindingError(
            "canonical pre-execution prefix no longer matches terminal evidence"
        )
    if observed_after != str(details["canonical_after_head"]):
        raise CanonicalExecutionBindingError(
            "canonical post-execution prefix no longer matches terminal evidence"
        )


__all__ = [
    "CANONICAL_EXECUTION_BINDING_SCHEMA",
    "CanonicalExecutionBindingError",
    "CanonicalLedgerPoint",
    "canonical_ledger_point",
    "canonical_prefix_head",
    "binding_details",
    "verify_terminal_canonical_binding",
]
