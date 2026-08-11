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
    verify_canonical_prefix_receipt,
)
from quantagent.paper.execution_journal import (
    ExecutionJournalCorruption,
    PendingExecutionJournal,
    TERMINAL_OUTCOMES,
)
from quantagent.paper.runtime_paths import paper_runtime_paths

from services.quant_api.config import ApiSettings, project_relative


_CRITICAL_STATUSES = frozenset({"execution_indeterminate"})
_WARNING_STATUSES = frozenset({"execution_blocked", "missed_execution_session"})


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
            return self._unavailable(relative_path, account_identity=identity)

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

        prefix_status = self._canonical_prefix_status(
            terminal_records,
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
        counts = Counter(row.status for row in records)

        attention = "ok"
        if (
            unresolved
            or latest_status in _CRITICAL_STATUSES
            or identity["state"] != "valid"
        ):
            attention = "critical"
        elif latest_status in _WARNING_STATUSES:
            attention = "warning"
        elif latest_status == "execution_started":
            attention = "pending"

        # These values are evidence, not derived promotion. Absence is false /
        # unavailable rather than an optimistic default.
        production_certified = bool(
            latest_details.get("production_pretrade_risk_certified", False)
        )
        calendar_eligible = bool(
            latest_details.get("shadow_acceptance_calendar_eligible", False)
        )
        assurance = str(
            latest_details.get("calendar_assurance") or "unavailable"
        )
        identity_verified = bool(identity.get("verified", False))
        latest_terminal_bound = bool(prefix_status["latestTerminalBound"])

        visible = [self._record(row) for row in reversed(records[-limit:])]
        return {
            "journal": {
                "state": "valid",
                "verified": True,
                "path": relative_path,
                "recordCount": len(records),
                "terminalCount": len(terminal_records),
                "unresolvedCount": len(unresolved),
                "reason": None,
            },
            "accountIdentity": identity,
            "canonicalPrefix": prefix_status,
            "summary": {
                "attention": attention,
                "latestStatus": latest_status,
                "latestSignalDate": latest.signal_date,
                "latestExecutionDate": latest.execution_date,
                "latestRecordedAt": latest.recorded_at,
                "calendarAssurance": assurance,
                "shadowAcceptanceCalendarEligible": calendar_eligible,
                "productionPretradeRiskCertified": production_certified,
                "accountIdentityVerified": identity_verified,
                "latestTerminalCanonicalPrefixBound": latest_terminal_bound,
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
                "paperExecutionEvidence": True,
                "accountIdentityVerified": identity_verified,
                "canonicalExecutionPrefixCertified": latest_terminal_bound,
                "productionLiveCertified": production_certified,
                "authoritativeCalendarCertified": calendar_eligible,
                "message": (
                    "Paper/shadow execution evidence is available. Account identity "
                    "and canonical-prefix proof are reported separately; none of "
                    "these fields is production/live certification unless the "
                    "dedicated certification fields explicitly say so."
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

    def _canonical_prefix_status(
        self,
        terminal_records: list[Any],
        *,
        account_identity: dict[str, Any],
    ) -> dict[str, Any]:
        bound = 0
        legacy_unbound = 0
        latest_bound = False
        latest_terminal = terminal_records[-1] if terminal_records else None
        identity_sha = (
            str(account_identity.get("payloadSha256"))
            if account_identity.get("verified")
            else None
        )

        for record in terminal_records:
            details = dict(record.details or {})
            receipt = details.get("canonical_prefix_receipt")
            if receipt is None:
                legacy_unbound += 1
                if record is latest_terminal:
                    latest_bound = False
                continue
            try:
                verification = verify_canonical_prefix_receipt(
                    receipt,
                    ledger_or_path=self.paths.canonical_ledger,
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
            except (CanonicalPrefixReceiptError, OSError, UnicodeError) as exc:
                return {
                    "state": "invalid",
                    "verified": False,
                    "boundTerminalCount": bound,
                    "legacyUnboundTerminalCount": legacy_unbound,
                    "latestTerminalBound": False,
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
                    "legacyUnboundTerminalCount": legacy_unbound,
                    "latestTerminalBound": False,
                    "reason": (
                        f"terminal sequence {record.sequence} claimed a receipt "
                        "that was not bound/valid"
                    ),
                }
            bound += 1
            if record is latest_terminal:
                latest_bound = True

        state = "valid" if terminal_records else "unavailable"
        return {
            "state": state,
            "verified": bool(terminal_records and bound + legacy_unbound == len(terminal_records)),
            "boundTerminalCount": bound,
            "legacyUnboundTerminalCount": legacy_unbound,
            "latestTerminalBound": latest_bound,
            "reason": (
                None
                if terminal_records
                else "no terminal execution record is available"
            ),
        }

    @staticmethod
    def _record(record: Any) -> dict[str, Any]:
        details = dict(record.details or {})
        return {
            "sequence": record.sequence,
            "payloadSha256": record.pending_payload_sha256,
            "signalDate": record.signal_date,
            "executionDate": record.execution_date,
            "status": record.status,
            "recordedAt": record.recorded_at,
            "recordSha256": record.record_sha256,
            "details": details,
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
            "reason": reason,
        }

    def _unavailable(
        self,
        path: str,
        *,
        reason: str = "execution journal not found",
        account_identity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        identity = account_identity or self._account_identity()
        return {
            "journal": self._base_journal(
                "unavailable", path, verified=False, reason=reason
            ),
            "accountIdentity": identity,
            "canonicalPrefix": {
                "state": "unavailable",
                "verified": False,
                "boundTerminalCount": 0,
                "legacyUnboundTerminalCount": 0,
                "latestTerminalBound": False,
                "reason": "no terminal execution record is available",
            },
            "summary": {
                "attention": (
                    "critical" if identity.get("state") == "invalid" else "unavailable"
                ),
                "latestStatus": None,
                "latestSignalDate": None,
                "latestExecutionDate": None,
                "latestRecordedAt": None,
                "calendarAssurance": "unavailable",
                "shadowAcceptanceCalendarEligible": False,
                "productionPretradeRiskCertified": False,
                "accountIdentityVerified": bool(identity.get("verified", False)),
                "latestTerminalCanonicalPrefixBound": False,
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
                "message": "No verified continuous paper execution evidence is available.",
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
