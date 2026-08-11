"""Canonical-ledger prefix binding for terminal paper execution evidence.

The execution journal proves that an outcome record was not edited. It does not
by itself prove that the economic ledger prefix the outcome referred to still
exists unchanged. This module binds every new terminal outcome to the exact
canonical-ledger prefix before and after the attempt.

Later economic events may append to the ledger. Verification therefore checks
historical prefix heads rather than requiring the receipt's post-execution head
to remain the ledger's current head. Read-heavy operator projections build one
verified prefix index and reuse it for O(1) historical-head lookups.
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


@dataclass(frozen=True, slots=True)
class CanonicalPrefixIndex:
    """One verified immutable-in-memory view of all canonical prefix heads."""

    prefix_heads: tuple[str, ...]
    current_head: str

    @property
    def record_count(self) -> int:
        return len(self.prefix_heads) - 1

    def head_at(self, record_count: int) -> str:
        count = int(record_count)
        if count < 0:
            raise CanonicalPrefixReceiptError("canonical prefix length must be >= 0")
        if count > self.record_count:
            raise CanonicalPrefixReceiptError(
                f"canonical prefix length {count} exceeds current ledger length "
                f"{self.record_count}"
            )
        return self.prefix_heads[count]


def _verified_ledger(
    ledger_or_path: CanonicalLedger | str | Path,
) -> CanonicalLedger:
    try:
        ledger = (
            ledger_or_path
            if isinstance(ledger_or_path, CanonicalLedger)
            else CanonicalLedger(ledger_or_path)
        )
        verification = ledger.verify()
    except CanonicalPrefixReceiptError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, OSError, UnicodeError) as exc:
        raise CanonicalPrefixReceiptError(
            f"cannot parse/verify canonical ledger: {exc}"
        ) from exc
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


def build_canonical_prefix_index(
    ledger_or_path: CanonicalLedger | str | Path,
) -> CanonicalPrefixIndex:
    """Read and verify the canonical chain once, then index every prefix head."""

    ledger = _verified_ledger(ledger_or_path)
    records = ledger.read()
    heads = (GENESIS_HASH, *(record.record_hash for record in records))
    current_head = heads[-1]
    if current_head != ledger.head_hash:
        raise CanonicalPrefixReceiptError(
            "canonical ledger head changed while prefix index was being built"
        )
    return CanonicalPrefixIndex(prefix_heads=tuple(heads), current_head=current_head)


def canonical_snapshot(
    ledger_or_path: CanonicalLedger | str | Path,
) -> tuple[int, str]:
    """Return a verified current canonical record count/head pair."""

    index = build_canonical_prefix_index(ledger_or_path)
    return index.record_count, index.current_head


def build_canonical_prefix_receipt(
    *,
    ledger: CanonicalLedger | str | Path,
    canonical_before_records: int,
    canonical_before_head: str,
    target_weights_sha256: str,
    paper_account_identity_sha256: str,
) -> dict[str, object]:
    """Bind one terminal outcome to a freshly verified ledger prefix.

    Callers sealing terminal evidence should pass the canonical *path*, not a
    long-lived ``CanonicalLedger`` instance, so the post-execution snapshot is
    reopened from durable bytes immediately before the journal terminal is
    appended.
    """

    index = build_canonical_prefix_index(ledger)
    before_records = int(canonical_before_records)
    before_head = str(canonical_before_head)
    if index.head_at(before_records) != before_head:
        raise CanonicalPrefixReceiptError(
            "canonical pre-execution prefix changed before terminal receipt creation"
        )
    target_sha = str(target_weights_sha256).strip()
    identity_sha = str(paper_account_identity_sha256).strip()
    if not target_sha or not identity_sha:
        raise CanonicalPrefixReceiptError(
            "canonical prefix receipt requires target/account identity digests"
        )
    return {
        "schema_version": CANONICAL_PREFIX_RECEIPT_SCHEMA,
        "canonical_before_records": before_records,
        "canonical_before_head": before_head,
        "canonical_after_records": index.record_count,
        "canonical_after_head": index.current_head,
        "target_weights_sha256": target_sha,
        "paper_account_identity_sha256": identity_sha,
    }


def verify_canonical_prefix_receipt(
    receipt: Mapping[str, object] | None,
    *,
    ledger_or_path: CanonicalLedger | str | Path | None = None,
    prefix_index: CanonicalPrefixIndex | None = None,
    expected_target_weights_sha256: str | None = None,
    expected_paper_account_identity_sha256: str | None = None,
) -> CanonicalPrefixVerification:
    """Verify one journal-embedded receipt against a canonical prefix snapshot.

    ``receipt is None`` is legacy/unbound evidence rather than corruption so
    completed pre-migration records remain readable. Any receipt that claims to
    be bound must verify completely or fail closed.
    """

    if prefix_index is not None and ledger_or_path is not None:
        raise CanonicalPrefixReceiptError(
            "pass either prefix_index or ledger_or_path, not both"
        )
    if prefix_index is None:
        if ledger_or_path is None:
            raise CanonicalPrefixReceiptError(
                "canonical receipt verification requires a ledger or prefix index"
            )
        prefix_index = build_canonical_prefix_index(ledger_or_path)

    if receipt is None:
        return CanonicalPrefixVerification(
            bound=False,
            valid=True,
            reason="legacy_terminal_without_canonical_prefix_receipt",
            current_records=prefix_index.record_count,
            current_head=prefix_index.current_head,
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
    if prefix_index.head_at(before_records) != before_head:
        raise CanonicalPrefixReceiptError(
            "canonical pre-execution prefix no longer matches terminal receipt"
        )
    if prefix_index.head_at(after_records) != after_head:
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
        current_records=prefix_index.record_count,
        current_head=prefix_index.current_head,
    )


__all__ = [
    "CANONICAL_PREFIX_RECEIPT_SCHEMA",
    "CanonicalPrefixIndex",
    "CanonicalPrefixReceiptError",
    "CanonicalPrefixVerification",
    "build_canonical_prefix_index",
    "build_canonical_prefix_receipt",
    "canonical_snapshot",
    "verify_canonical_prefix_receipt",
]
