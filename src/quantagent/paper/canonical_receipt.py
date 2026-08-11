"""Canonical-ledger prefix binding for terminal paper execution evidence.

The execution journal proves that an outcome record was not edited. It does not
by itself prove that the economic ledger prefix the outcome referred to still
exists unchanged. This module binds every new terminal outcome to the exact
canonical-ledger prefix before and after the attempt.

Later economic events may append to the ledger. Verification therefore checks
prefix heads at the recorded lengths rather than requiring the receipt's
``canonical_after_head`` to remain the ledger's current head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from quantagent.domain.ledger import CanonicalLedger, GENESIS_HASH


CANONICAL_PREFIX_RECEIPT_SCHEMA = "quantagent.paper.canonical_prefix_receipt.v1"


class CanonicalPrefixReceiptError(RuntimeError):
    """Terminal execution evidence cannot be bound to the canonical ledger."""


@dataclass(frozen=True, slots=True)
class CanonicalPrefixVerification:
    bound: bool
    valid: bool
    reason: str | None
    canonical_before_records: int | None = None
    canonical_before_head: str | None = None
    canonical_after_records: int | None = None
    canonical_after_head: str | None = None
    current_records: int | None = None
    current_head: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _prefix_head(ledger: CanonicalLedger, record_count: int) -> str:
    if record_count < 0:
        raise CanonicalPrefixReceiptError("canonical prefix length must be >= 0")
    records = ledger.read()
    if record_count > len(records):
        raise CanonicalPrefixReceiptError(
            f"canonical prefix length {record_count} exceeds current ledger "
            f"length {len(records)}"
        )
    return GENESIS_HASH if record_count == 0 else records[record_count - 1].record_hash


def _verified_ledger(
    ledger_or_path: CanonicalLedger | str | Path,
) -> CanonicalLedger:
    ledger = (
        ledger_or_path
        if isinstance(ledger_or_path, CanonicalLedger)
        else CanonicalLedger(ledger_or_path)
    )
    verification = ledger.verify()
    if not verification.get("valid"):
        raise CanonicalPrefixReceiptError(
            f"canonical ledger hash chain is invalid: {verification}"
        )
    if verification.get("tornTail"):
        raise CanonicalPrefixReceiptError(
            "canonical ledger has a torn tail; terminal evidence cannot be certified"
        )
    if verification.get("writeFailure"):
        raise CanonicalPrefixReceiptError(
            f"canonical ledger write failure is latched: {verification.get('writeFailure')}"
        )
    return ledger


def canonical_snapshot(
    ledger_or_path: CanonicalLedger | str | Path,
) -> tuple[int, str]:
    """Return a verified current canonical record count/head pair."""

    ledger = _verified_ledger(ledger_or_path)
    return len(ledger), ledger.head_hash


def build_canonical_prefix_receipt(
    *,
    ledger: CanonicalLedger,
    canonical_before_records: int,
    canonical_before_head: str,
    target_weights_sha256: str,
    paper_account_identity_sha256: str,
) -> dict[str, object]:
    """Bind one terminal outcome to the ledger prefix it actually produced."""

    verified = _verified_ledger(ledger)
    before_records = int(canonical_before_records)
    before_head = str(canonical_before_head)
    if _prefix_head(verified, before_records) != before_head:
        raise CanonicalPrefixReceiptError(
            "canonical pre-execution prefix changed before terminal receipt creation"
        )
    after_records = len(verified)
    after_head = verified.head_hash
    return {
        "schema_version": CANONICAL_PREFIX_RECEIPT_SCHEMA,
        "canonical_before_records": before_records,
        "canonical_before_head": before_head,
        "canonical_after_records": after_records,
        "canonical_after_head": after_head,
        "target_weights_sha256": str(target_weights_sha256),
        "paper_account_identity_sha256": str(paper_account_identity_sha256),
    }


def verify_canonical_prefix_receipt(
    receipt: Mapping[str, object] | None,
    *,
    ledger_or_path: CanonicalLedger | str | Path,
    expected_target_weights_sha256: str | None = None,
    expected_paper_account_identity_sha256: str | None = None,
) -> CanonicalPrefixVerification:
    """Verify one journal-embedded receipt against the immutable ledger prefix.

    ``receipt is None`` is legacy/unbound evidence rather than corruption so
    completed pre-migration records remain readable. Any receipt that claims to
    be bound must verify completely or fail closed.
    """

    ledger = _verified_ledger(ledger_or_path)
    if receipt is None:
        return CanonicalPrefixVerification(
            bound=False,
            valid=True,
            reason="legacy_terminal_without_canonical_prefix_receipt",
            current_records=len(ledger),
            current_head=ledger.head_hash,
        )
    if not isinstance(receipt, Mapping):
        raise CanonicalPrefixReceiptError(
            "canonical_prefix_receipt must be a JSON object"
        )

    schema = str(receipt.get("schema_version") or "")
    if schema != CANONICAL_PREFIX_RECEIPT_SCHEMA:
        raise CanonicalPrefixReceiptError(
            f"unsupported canonical prefix receipt schema {schema!r}"
        )
    try:
        before_records = int(receipt["canonical_before_records"])
        after_records = int(receipt["canonical_after_records"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CanonicalPrefixReceiptError(
            "canonical prefix receipt has invalid record counts"
        ) from exc
    before_head = str(receipt.get("canonical_before_head") or "")
    after_head = str(receipt.get("canonical_after_head") or "")
    if after_records < before_records:
        raise CanonicalPrefixReceiptError(
            "canonical receipt record count moved backwards"
        )
    if _prefix_head(ledger, before_records) != before_head:
        raise CanonicalPrefixReceiptError(
            "canonical pre-execution prefix no longer matches terminal receipt"
        )
    if _prefix_head(ledger, after_records) != after_head:
        raise CanonicalPrefixReceiptError(
            "canonical post-execution prefix no longer matches terminal receipt"
        )

    target_sha = str(receipt.get("target_weights_sha256") or "")
    identity_sha = str(receipt.get("paper_account_identity_sha256") or "")
    if not target_sha or not identity_sha:
        raise CanonicalPrefixReceiptError(
            "canonical prefix receipt is missing target/account identity binding"
        )
    if (
        expected_target_weights_sha256 is not None
        and target_sha != str(expected_target_weights_sha256)
    ):
        raise CanonicalPrefixReceiptError(
            "canonical prefix receipt target-weight digest mismatch"
        )
    if (
        expected_paper_account_identity_sha256 is not None
        and identity_sha != str(expected_paper_account_identity_sha256)
    ):
        raise CanonicalPrefixReceiptError(
            "canonical prefix receipt paper-account identity mismatch"
        )

    return CanonicalPrefixVerification(
        bound=True,
        valid=True,
        reason=None,
        canonical_before_records=before_records,
        canonical_before_head=before_head,
        canonical_after_records=after_records,
        canonical_after_head=after_head,
        current_records=len(ledger),
        current_head=ledger.head_hash,
    )


__all__ = [
    "CANONICAL_PREFIX_RECEIPT_SCHEMA",
    "CanonicalPrefixReceiptError",
    "CanonicalPrefixVerification",
    "build_canonical_prefix_receipt",
    "canonical_snapshot",
    "verify_canonical_prefix_receipt",
]
