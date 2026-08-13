"""Validation for append-only operator reconciliation of legacy terminals.

A legacy binding is deliberately lower assurance than an execution-time receipt.
It never mutates or relabels the terminal: it binds the immutable terminal hash,
target digest and account identity to a later reconciled canonical/account state.
"""

from __future__ import annotations

from quantagent.paper.canonical_receipt import (
    CanonicalPrefixIndex,
    CanonicalPrefixReceiptError,
)
from quantagent.paper.execution_journal import LEGACY_BINDING_STATUS


class LegacyTerminalBindingError(RuntimeError):
    """A legacy terminal binding is absent or cannot be verified."""


def verify_legacy_terminal_binding(
    terminal,
    binding,
    *,
    prefix_index: CanonicalPrefixIndex,
    expected_paper_account_identity_sha256: str,
    expected_target_weights_sha256: str | None = None,
) -> None:
    if binding is None:
        raise LegacyTerminalBindingError(
            "legacy terminal has no append-only operator reconciliation binding"
        )
    if str(getattr(binding, "status", "")) != LEGACY_BINDING_STATUS:
        raise LegacyTerminalBindingError("unexpected legacy terminal binding status")
    details = dict(getattr(binding, "details", {}) or {})
    if str(details.get("terminal_record_sha256") or "") != str(terminal.record_sha256):
        raise LegacyTerminalBindingError(
            "legacy terminal binding does not match the immutable terminal record"
        )
    if str(details.get("paper_account_identity_sha256") or "") != str(
        expected_paper_account_identity_sha256
    ):
        raise LegacyTerminalBindingError(
            "legacy terminal binding paper-account identity mismatch"
        )
    assurance = str(details.get("assurance") or "")
    reconstruction = str(details.get("operational_economic_reconstruction") or "")
    if assurance != "operator_bound_canonical_only_legacy_terminal_v1":
        # Do not accept a parity-sounding assurance without a separate protocol
        # that proves corresponding historical economic events, not merely equal
        # current cash/positions or the presence of arbitrary order records.
        raise LegacyTerminalBindingError("legacy terminal binding assurance is invalid")
    if reconstruction != "not_claimed_canonical_is_record_of_account":
        raise LegacyTerminalBindingError(
            "legacy terminal canonical-only assurance marker is invalid"
        )
    account_state_sha = str(details.get("account_state_sha256") or "")
    if len(account_state_sha) != 64:
        raise LegacyTerminalBindingError(
            "legacy terminal binding lacks a reconciled account-state digest"
        )
    binding_target = str(details.get("target_weights_sha256") or "")
    terminal_target = str(dict(terminal.details or {}).get("target_weights_sha256") or "")
    if len(binding_target) != 64:
        raise LegacyTerminalBindingError("legacy terminal binding lacks target digest")
    if terminal_target and terminal_target != binding_target:
        raise LegacyTerminalBindingError(
            "legacy terminal binding target digest conflicts with terminal evidence"
        )
    if (
        expected_target_weights_sha256 is not None
        and binding_target != str(expected_target_weights_sha256)
    ):
        raise LegacyTerminalBindingError("legacy terminal binding target digest mismatch")
    try:
        canonical_records = int(details["canonical_records"])
        canonical_head = str(details["canonical_head"])
        if prefix_index.head_at(canonical_records) != canonical_head:
            raise LegacyTerminalBindingError(
                "legacy terminal binding canonical prefix no longer matches"
            )
    except LegacyTerminalBindingError:
        raise
    except (CanonicalPrefixReceiptError, KeyError, TypeError, ValueError) as exc:
        raise LegacyTerminalBindingError(
            "legacy terminal binding canonical prefix is invalid"
        ) from exc


__all__ = ["LegacyTerminalBindingError", "verify_legacy_terminal_binding"]
