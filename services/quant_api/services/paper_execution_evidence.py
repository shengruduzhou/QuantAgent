"""Read-only projection of continuous paper execution evidence for operators."""

from __future__ import annotations

from collections import Counter
from typing import Any

from quantagent.paper.account_identity import (
    PaperAccountIdentityError,
    PaperAccountIdentityStore,
)
from quantagent.paper.canonical_receipt import (
    CanonicalPrefixReceiptError,
    build_canonical_prefix_index,
    verify_canonical_prefix_receipt,
)
from quantagent.paper.execution_journal import (
    DAILY_DECISION_STATUS,
    ExecutionJournalCorruption,
    PendingExecutionJournal,
    RECONCILIATION_STATUS,
    TERMINAL_OUTCOMES,
)
from quantagent.paper.legacy_terminal_binding import (
    LegacyTerminalBindingError,
    verify_legacy_terminal_binding,
)
from quantagent.paper.pending_signal import (
    PendingPaperSignalStore,
    PendingSignalCorruption,
)
from quantagent.paper.runtime_paths import paper_runtime_paths

from services.quant_api.config import ApiSettings, project_relative


_CRITICAL_STATUSES = frozenset({"execution_indeterminate"})
_WARNING_STATUSES = frozenset({"execution_blocked", "missed_execution_session"})
_LOWER_ASSURANCE_STATUSES = frozenset(
    {"legacy_terminal_bound", "execution_reconciled"}
)


class PaperExecutionEvidenceService:
    """Project verified paper evidence without upgrading its assurance level."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.paths = paper_runtime_paths(settings.runtime_root)

    def status(self, *, limit: int = 30) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        identity = self._account_identity()
        path = self.paths.execution_journal
        relative_path = project_relative(self.settings, path)
        if not path.exists():
            pending_state = self._pending_artifact_status(None)
            if pending_state["state"] == "invalid":
                return self._invalid(
                    relative_path,
                    str(pending_state["reason"]),
                    account_identity=identity,
                )
            return self._unavailable(
                relative_path,
                account_identity=identity,
                pending_state=pending_state,
            )

        journal = PendingExecutionJournal(path)
        try:
            records = journal.records()
            verified = journal.verify()
        except (ExecutionJournalCorruption, OSError, UnicodeError) as exc:
            return self._invalid(
                relative_path,
                str(exc),
                account_identity=identity,
            )
        if not verified:
            return self._invalid(
                relative_path,
                "execution journal hash-chain verification failed",
                account_identity=identity,
            )
        if identity["state"] == "invalid":
            return self._invalid(
                relative_path,
                f"paper account identity invalid: {identity.get('reason')}",
                account_identity=identity,
            )
        if not records:
            return self._unavailable(
                relative_path,
                reason="journal exists but is empty",
                account_identity=identity,
                pending_state=self._pending_artifact_status(journal),
            )

        by_payload: dict[str, list[Any]] = {}
        for record in records:
            by_payload.setdefault(record.pending_payload_sha256, []).append(record)
        unresolved = [
            payload
            for payload, history in by_payload.items()
            if any(row.status == "execution_started" for row in history)
            and not any(row.status in TERMINAL_OUTCOMES for row in history)
        ]
        terminal_records = [row for row in records if row.status in TERMINAL_OUTCOMES]
        pending_state = self._pending_artifact_status(journal)
        if pending_state["state"] == "invalid":
            return self._invalid(
                relative_path,
                str(pending_state["reason"]),
                account_identity=identity,
            )

        prefix_status = self._canonical_prefix_status(
            terminal_records,
            journal=journal,
            account_identity=identity,
        )
        if prefix_status["state"] == "invalid":
            return self._invalid(
                relative_path,
                str(prefix_status.get("reason") or "canonical terminal prefix invalid"),
                account_identity=identity,
                canonical_prefix=prefix_status,
            )

        latest = records[-1]
        latest_details = dict(latest.details or {})
        latest_status = latest.status
        latest_decision_kind = (
            str(latest_details.get("decision_kind") or "")
            if latest_status == DAILY_DECISION_STATUS
            else None
        )
        counts = Counter(row.status for row in records)
        unreconciled_indeterminate = 0
        for payload, history in by_payload.items():
            terminal = next(
                (row for row in history if row.status in TERMINAL_OUTCOMES),
                None,
            )
            if terminal is None or terminal.status != "execution_indeterminate":
                continue
            reconciliation = next(
                (row for row in history if row.status == RECONCILIATION_STATUS),
                None,
            )
            if reconciliation is None:
                unreconciled_indeterminate += 1

        attention = "ok"
        if (
            unresolved
            or unreconciled_indeterminate
            or identity["state"] != "valid"
            or prefix_status["state"] == "invalid"
            or int(pending_state["missingCommittedCount"]) > 0
        ):
            attention = "critical"
        elif latest_status in _WARNING_STATUSES:
            attention = "warning"
        elif int(pending_state["stagedUncommittedCount"]) > 0:
            attention = "warning"
        elif (
            latest_status == DAILY_DECISION_STATUS
            and latest_decision_kind == "target"
        ):
            attention = "pending"
        elif latest_status in _LOWER_ASSURANCE_STATUSES:
            attention = "warning"
        elif latest_status == "execution_started":
            attention = "pending"

        production_certified = bool(
            latest_details.get("production_pretrade_risk_certified", False)
        )
        calendar_eligible = bool(
            latest_details.get("shadow_acceptance_calendar_eligible", False)
        )
        assurance = str(latest_details.get("calendar_assurance") or "unavailable")
        identity_verified = bool(identity.get("verified", False))
        latest_terminal_bound = bool(prefix_status["latestTerminalBound"])
        has_execution_evidence = any(
            row.status == "execution_started" or row.status in TERMINAL_OUTCOMES
            for row in records
        )

        visible = [self._record(row) for row in reversed(records[-limit:])]
        return {
            "journal": {
                "state": "valid",
                "verified": True,
                "path": relative_path,
                "recordCount": len(records),
                "terminalCount": len(terminal_records),
                "unresolvedCount": len(unresolved),
                "unreconciledIndeterminateCount": unreconciled_indeterminate,
                "committedPendingCount": int(pending_state["committedPendingCount"]),
                "stagedUncommittedCount": int(
                    pending_state["stagedUncommittedCount"]
                ),
                "missingCommittedPendingCount": int(
                    pending_state["missingCommittedCount"]
                ),
                "reason": None,
            },
            "accountIdentity": identity,
            "canonicalPrefix": prefix_status,
            "summary": {
                "attention": attention,
                "latestStatus": latest_status,
                "latestDecisionKind": latest_decision_kind,
                "latestSignalDate": latest.signal_date,
                "latestExecutionDate": latest.execution_date,
                "latestRecordedAt": latest.recorded_at,
                "calendarAssurance": assurance,
                "shadowAcceptanceCalendarEligible": calendar_eligible,
                "productionPretradeRiskCertified": production_certified,
                "accountIdentityVerified": identity_verified,
                "latestTerminalCanonicalPrefixBound": latest_terminal_bound,
                "latestTerminalBindingAssurance": prefix_status[
                    "latestTerminalBindingAssurance"
                ],
                "frozenCanonicalRecordCount": latest_details.get("canonical_records"),
                "frozenCanonicalHead": latest_details.get("canonical_head"),
                "executionEvidenceAvailable": has_execution_evidence,
                "riskScope": latest_details.get("risk_scope"),
                "sessionClosed": latest_details.get("session_closed"),
                "orderCount": latest_details.get("order_count"),
                "fillCount": latest_details.get("fill_count"),
                "navBefore": latest_details.get("nav_before"),
                "navAfter": latest_details.get("nav_after"),
                "statusCounts": dict(sorted(counts.items())),
            },
            "records": visible,
            "operatorTruth": {
                "paperExecutionEvidence": has_execution_evidence,
                "accountIdentityVerified": identity_verified,
                "canonicalExecutionPrefixCertified": latest_terminal_bound,
                "productionLiveCertified": production_certified,
                "authoritativeCalendarCertified": calendar_eligible,
                "message": self._operator_message(
                    latest_status=latest_status,
                    latest_decision_kind=latest_decision_kind,
                    has_execution_evidence=has_execution_evidence,
                    pending_state=pending_state,
                ),
            },
        }

    def _account_identity(self) -> dict[str, Any]:
        path = self.paths.account_identity
        relative_path = project_relative(self.settings, path)
        if not path.exists():
            return {
                "state": "unavailable",
                "verified": False,
                "path": relative_path,
                "accountInstanceId": None,
                "portfolioId": None,
                "initialCashCny": None,
                "payloadSha256": None,
                "reason": "paper account identity not found",
            }
        try:
            identity = PaperAccountIdentityStore(path).read()
        except (PaperAccountIdentityError, OSError, UnicodeError) as exc:
            return {
                "state": "invalid",
                "verified": False,
                "path": relative_path,
                "accountInstanceId": None,
                "portfolioId": None,
                "initialCashCny": None,
                "payloadSha256": None,
                "reason": str(exc),
            }
        if identity is None:
            return {
                "state": "unavailable",
                "verified": False,
                "path": relative_path,
                "accountInstanceId": None,
                "portfolioId": None,
                "initialCashCny": None,
                "payloadSha256": None,
                "reason": "paper account identity file is empty/unavailable",
            }
        return {
            "state": "valid",
            "verified": True,
            "path": relative_path,
            "accountInstanceId": identity.account_instance_id,
            "portfolioId": identity.portfolio_id,
            "initialCashCny": identity.initial_cash_cny,
            "payloadSha256": identity.payload_sha256,
            "reason": None,
        }

    def _pending_artifact_status(
        self,
        journal: PendingExecutionJournal | None,
    ) -> dict[str, Any]:
        root = self.paths.pending_signals
        if not root.exists():
            signals: list[Any] = []
        else:
            store = PendingPaperSignalStore(root)
            signals = []
            try:
                for path in sorted(root.glob("*.json")):
                    signal = store.read(path.stem)
                    if signal is not None:
                        signals.append(signal)
            except (PendingSignalCorruption, OSError, UnicodeError, ValueError) as exc:
                return {
                    "state": "invalid",
                    "committedPendingCount": 0,
                    "stagedUncommittedCount": 0,
                    "missingCommittedCount": 0,
                    "latestSignalDate": None,
                    "reason": f"pending signal evidence is invalid: {exc}",
                }

        committed = 0
        staged = 0
        by_payload = {signal.payload_sha256: signal for signal in signals}
        for signal in signals:
            terminal = journal.terminal(signal.payload_sha256) if journal else None
            if terminal is not None:
                continue
            decision = journal.daily_decision(signal.signal_date) if journal else None
            if decision is None:
                staged += 1
                continue
            details = dict(decision.details or {})
            if (
                decision.pending_payload_sha256 != signal.payload_sha256
                or str(details.get("decision_kind") or "") != "target"
            ):
                return {
                    "state": "invalid",
                    "committedPendingCount": committed,
                    "stagedUncommittedCount": staged,
                    "missingCommittedCount": 0,
                    "latestSignalDate": signal.signal_date,
                    "reason": (
                        "pending signal conflicts with its same-date durable decision"
                    ),
                }
            committed += 1

        missing = 0
        if journal is not None:
            for record in journal.records():
                if (
                    record.status == DAILY_DECISION_STATUS
                    and str(dict(record.details or {}).get("decision_kind") or "")
                    == "target"
                    and record.pending_payload_sha256 not in by_payload
                    and journal.terminal(record.pending_payload_sha256) is None
                ):
                    missing += 1
        latest_signal = max((signal.signal_date for signal in signals), default=None)
        return {
            "state": "valid",
            "committedPendingCount": committed,
            "stagedUncommittedCount": staged,
            "missingCommittedCount": missing,
            "latestSignalDate": latest_signal,
            "reason": None,
        }

    @staticmethod
    def _operator_message(
        *,
        latest_status: str,
        latest_decision_kind: str | None,
        has_execution_evidence: bool,
        pending_state: dict[str, Any],
    ) -> str:
        if int(pending_state["missingCommittedCount"]) > 0:
            return (
                "A frozen target is missing its bound pending artifact; paper "
                "execution is unverifiable and must fail closed."
            )
        if int(pending_state["stagedUncommittedCount"]) > 0:
            return (
                "A pending target is staged but has no daily_decision_frozen commit; "
                "it is not executable and is not paper execution evidence."
            )
        if latest_status == DAILY_DECISION_STATUS:
            if latest_decision_kind == "target":
                return (
                    "The daily target is durably frozen and awaits the exact next "
                    "observed session; no execution outcome is claimed."
                )
            return (
                "The no-target daily decision is durably frozen; no order, fill, "
                "or paper execution outcome is claimed."
            )
        if latest_status == "legacy_terminal_bound":
            return (
                "The latest legacy terminal has a lower-assurance operator binding "
                "to the canonical account; it is not an execution-time receipt."
            )
        if not has_execution_evidence:
            return "No paper execution attempt or terminal outcome is available."
        return (
            "Paper/shadow execution evidence is available. Account identity and "
            "canonical-prefix proof are reported separately; none of these fields "
            "is production/live certification unless explicitly certified."
        )

    def _canonical_prefix_status(
        self,
        terminal_records: list[Any],
        *,
        journal: PendingExecutionJournal,
        account_identity: dict[str, Any],
    ) -> dict[str, Any]:
        bound = 0
        legacy_bound = 0
        legacy_unbound = 0
        latest_bound = False
        latest_assurance = "unavailable"
        latest_terminal = terminal_records[-1] if terminal_records else None
        identity_sha = (
            str(account_identity.get("payloadSha256"))
            if account_identity.get("verified")
            else None
        )
        # Build one verified immutable prefix index for the entire projection.
        # Each terminal lookup is then O(1) rather than reparsing/copying the
        # canonical ledger for every receipt.
        try:
            prefix_index = build_canonical_prefix_index(self.paths.canonical_ledger)
        except (CanonicalPrefixReceiptError, OSError, UnicodeError, ValueError, TypeError) as exc:
            return {
                "state": "invalid",
                "verified": False,
                "boundTerminalCount": 0,
                "legacyBoundTerminalCount": 0,
                "legacyUnboundTerminalCount": 0,
                "latestTerminalBound": False,
                "latestTerminalBindingAssurance": "unavailable",
                "currentRecordCount": None,
                "currentHeadHash": None,
                "reason": f"cannot build canonical prefix index: {exc}",
            }

        if not terminal_records:
            return {
                "state": "unavailable",
                "verified": False,
                "boundTerminalCount": 0,
                "legacyBoundTerminalCount": 0,
                "legacyUnboundTerminalCount": 0,
                "latestTerminalBound": False,
                "latestTerminalBindingAssurance": "unavailable",
                "currentRecordCount": prefix_index.record_count,
                "currentHeadHash": prefix_index.current_head,
                "reason": "no terminal execution record is available",
            }

        for record in terminal_records:
            details = dict(record.details or {})
            receipt = details.get("canonical_prefix_receipt")
            if receipt is None:
                binding = journal.legacy_binding(record.pending_payload_sha256)
                if binding is None:
                    legacy_unbound += 1
                    if record is latest_terminal:
                        latest_bound = False
                        latest_assurance = "legacy_unbound"
                    continue
                try:
                    verify_legacy_terminal_binding(
                        record,
                        binding,
                        prefix_index=prefix_index,
                        expected_paper_account_identity_sha256=str(identity_sha or ""),
                        expected_target_weights_sha256=(
                            str(details.get("target_weights_sha256"))
                            if details.get("target_weights_sha256")
                            else None
                        ),
                    )
                except LegacyTerminalBindingError as exc:
                    return {
                        "state": "invalid",
                        "verified": False,
                        "boundTerminalCount": bound,
                        "legacyBoundTerminalCount": legacy_bound,
                        "legacyUnboundTerminalCount": legacy_unbound,
                        "latestTerminalBound": False,
                        "latestTerminalBindingAssurance": "invalid",
                        "currentRecordCount": prefix_index.record_count,
                        "currentHeadHash": prefix_index.current_head,
                        "reason": (
                            f"terminal sequence {record.sequence} legacy binding "
                            f"verification failed: {exc}"
                        ),
                    }
                bound += 1
                legacy_bound += 1
                if record is latest_terminal:
                    latest_bound = True
                    latest_assurance = "operator_bound_legacy"
                continue
            try:
                verification = verify_canonical_prefix_receipt(
                    receipt,
                    prefix_index=prefix_index,
                    expected_target_weights_sha256=(
                        str(details.get("target_weights_sha256"))
                        if details.get("target_weights_sha256")
                        else None
                    ),
                    expected_paper_account_identity_sha256=(
                        identity_sha
                        or (
                            str(details.get("paper_account_identity_sha256"))
                            if details.get("paper_account_identity_sha256")
                            else None
                        )
                    ),
                )
            except (CanonicalPrefixReceiptError, OSError, UnicodeError, ValueError, TypeError) as exc:
                return {
                    "state": "invalid",
                    "verified": False,
                    "boundTerminalCount": bound,
                    "legacyBoundTerminalCount": legacy_bound,
                    "legacyUnboundTerminalCount": legacy_unbound,
                    "latestTerminalBound": False,
                    "latestTerminalBindingAssurance": "invalid",
                    "currentRecordCount": prefix_index.record_count,
                    "currentHeadHash": prefix_index.current_head,
                    "reason": (
                        f"terminal sequence {record.sequence} canonical prefix "
                        f"verification failed: {exc}"
                    ),
                }
            if not verification.valid or not verification.bound:
                return {
                    "state": "invalid",
                    "verified": False,
                    "boundTerminalCount": bound,
                    "legacyBoundTerminalCount": legacy_bound,
                    "legacyUnboundTerminalCount": legacy_unbound,
                    "latestTerminalBound": False,
                    "latestTerminalBindingAssurance": "invalid",
                    "currentRecordCount": prefix_index.record_count,
                    "currentHeadHash": prefix_index.current_head,
                    "reason": (
                        f"terminal sequence {record.sequence} claimed a receipt "
                        "that was not bound/valid"
                    ),
                }
            bound += 1
            if record is latest_terminal:
                latest_bound = True
                latest_assurance = "execution_time_receipt"

        return {
            "state": "valid",
            "verified": True,
            "boundTerminalCount": bound,
            "legacyBoundTerminalCount": legacy_bound,
            "legacyUnboundTerminalCount": legacy_unbound,
            "latestTerminalBound": latest_bound,
            "latestTerminalBindingAssurance": latest_assurance,
            "currentRecordCount": prefix_index.record_count,
            "currentHeadHash": prefix_index.current_head,
            "reason": None,
        }

    @staticmethod
    def _record(record: Any) -> dict[str, Any]:
        return {
            "sequence": record.sequence,
            "payloadSha256": record.pending_payload_sha256,
            "signalDate": record.signal_date,
            "executionDate": record.execution_date,
            "status": record.status,
            "recordedAt": record.recorded_at,
            "recordSha256": record.record_sha256,
            "details": dict(record.details or {}),
        }

    @staticmethod
    def _base_journal(
        state: str,
        path: str,
        *,
        verified: bool,
        reason: str | None,
    ) -> dict[str, Any]:
        return {
            "state": state,
            "verified": verified,
            "path": path,
            "recordCount": 0,
            "terminalCount": 0,
            "unresolvedCount": 0,
            "unreconciledIndeterminateCount": 0,
            "committedPendingCount": 0,
            "stagedUncommittedCount": 0,
            "missingCommittedPendingCount": 0,
            "reason": reason,
        }

    def _unavailable(
        self,
        path: str,
        *,
        reason: str = "execution journal not found",
        account_identity: dict[str, Any] | None = None,
        pending_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        identity = account_identity or self._account_identity()
        pending = pending_state or {
            "committedPendingCount": 0,
            "stagedUncommittedCount": 0,
            "missingCommittedCount": 0,
            "latestSignalDate": None,
        }
        journal_payload = self._base_journal(
            "unavailable", path, verified=False, reason=reason
        )
        journal_payload.update(
            {
                "committedPendingCount": int(pending["committedPendingCount"]),
                "stagedUncommittedCount": int(pending["stagedUncommittedCount"]),
                "missingCommittedPendingCount": int(
                    pending["missingCommittedCount"]
                ),
            }
        )
        staged = int(pending["stagedUncommittedCount"]) > 0
        return {
            "journal": journal_payload,
            "accountIdentity": identity,
            "canonicalPrefix": {
                "state": "unavailable",
                "verified": False,
                "boundTerminalCount": 0,
                "legacyBoundTerminalCount": 0,
                "legacyUnboundTerminalCount": 0,
                "latestTerminalBound": False,
                "latestTerminalBindingAssurance": "unavailable",
                "currentRecordCount": None,
                "currentHeadHash": None,
                "reason": "no terminal execution record is available",
            },
            "summary": {
                "attention": (
                    "critical"
                    if identity.get("state") == "invalid"
                    else "warning"
                    if staged
                    else "unavailable"
                ),
                "latestStatus": "staged_uncommitted" if staged else None,
                "latestDecisionKind": None,
                "latestSignalDate": pending.get("latestSignalDate"),
                "latestExecutionDate": None,
                "latestRecordedAt": None,
                "calendarAssurance": "unavailable",
                "shadowAcceptanceCalendarEligible": False,
                "productionPretradeRiskCertified": False,
                "accountIdentityVerified": bool(identity.get("verified", False)),
                "latestTerminalCanonicalPrefixBound": False,
                "latestTerminalBindingAssurance": "unavailable",
                "frozenCanonicalRecordCount": None,
                "frozenCanonicalHead": None,
                "executionEvidenceAvailable": False,
                "riskScope": None,
                "sessionClosed": None,
                "orderCount": None,
                "fillCount": None,
                "navBefore": None,
                "navAfter": None,
                "statusCounts": {},
            },
            "records": [],
            "operatorTruth": {
                "paperExecutionEvidence": False,
                "accountIdentityVerified": bool(identity.get("verified", False)),
                "canonicalExecutionPrefixCertified": False,
                "productionLiveCertified": False,
                "authoritativeCalendarCertified": False,
                "message": (
                    "A pending target is staged but uncommitted and cannot execute."
                    if staged
                    else "No verified continuous paper execution evidence is available."
                ),
            },
        }

    def _invalid(
        self,
        path: str,
        reason: str,
        *,
        account_identity: dict[str, Any] | None = None,
        canonical_prefix: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._unavailable(
            path,
            reason=reason,
            account_identity=account_identity,
        )
        payload["journal"]["state"] = "invalid"
        payload["summary"]["attention"] = "critical"
        if canonical_prefix is not None:
            payload["canonicalPrefix"] = canonical_prefix
        payload["operatorTruth"]["message"] = (
            "Paper execution evidence is invalid or unverifiable; fail closed."
        )
        return payload


__all__ = ["PaperExecutionEvidenceService"]
