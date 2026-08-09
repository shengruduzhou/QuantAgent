"""Read-only projection of continuous paper execution evidence for operators."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

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
    """Project the hash-chained paper journal without upgrading its assurance."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.paths = paper_runtime_paths(settings.runtime_root)

    def status(self, *, limit: int = 30) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        path = self.paths.execution_journal
        relative_path = project_relative(self.settings, path)
        if not path.exists():
            return self._unavailable(relative_path)

        journal = PendingExecutionJournal(path)
        try:
            records = journal.records()
            verified = journal.verify()
        except (ExecutionJournalCorruption, OSError, UnicodeError) as exc:
            return self._invalid(relative_path, str(exc))
        if not verified:
            return self._invalid(
                relative_path,
                "execution journal hash-chain verification failed",
            )
        if not records:
            return self._unavailable(relative_path, reason="journal exists but is empty")

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
        latest = records[-1]
        latest_terminal = terminal_records[-1] if terminal_records else None
        counts = Counter(row.status for row in records)

        latest_details = dict(latest.details or {})
        latest_terminal_details = (
            dict(latest_terminal.details or {}) if latest_terminal is not None else {}
        )
        effective_details = latest_terminal_details or latest_details
        latest_status = latest.status
        attention = "ok"
        if unresolved or latest_status in _CRITICAL_STATUSES:
            attention = "critical"
        elif latest_status in _WARNING_STATUSES:
            attention = "warning"
        elif latest_status == "execution_started":
            attention = "pending"

        # These values are evidence, not derived promotion.  Absence is false /
        # unavailable rather than an optimistic default.
        production_certified = bool(
            effective_details.get("production_pretrade_risk_certified", False)
        )
        calendar_eligible = bool(
            effective_details.get("shadow_acceptance_calendar_eligible", False)
        )
        assurance = str(
            effective_details.get("calendar_assurance") or "unavailable"
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
                "reason": None,
            },
            "summary": {
                "attention": attention,
                "latestStatus": latest_status,
                "latestSignalDate": latest.signal_date,
                "latestExecutionDate": latest.execution_date,
                "latestRecordedAt": latest.recorded_at,
                "calendarAssurance": assurance,
                "shadowAcceptanceCalendarEligible": calendar_eligible,
                "productionPretradeRiskCertified": production_certified,
                "riskScope": effective_details.get("risk_scope"),
                "sessionClosed": effective_details.get("session_closed"),
                "orderCount": effective_details.get("order_count"),
                "fillCount": effective_details.get("fill_count"),
                "navBefore": effective_details.get("nav_before"),
                "navAfter": effective_details.get("nav_after"),
                "statusCounts": dict(sorted(counts.items())),
            },
            "records": visible,
            "operatorTruth": {
                "paperExecutionEvidence": True,
                "productionLiveCertified": production_certified,
                "authoritativeCalendarCertified": calendar_eligible,
                "message": (
                    "Paper/shadow execution evidence is available. It is not "
                    "production/live certification unless the dedicated evidence "
                    "fields explicitly say so."
                ),
            },
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

    def _unavailable(self, path: str, *, reason: str = "execution journal not found") -> dict[str, Any]:
        return {
            "journal": self._base_journal(
                "unavailable", path, verified=False, reason=reason
            ),
            "summary": {
                "attention": "unavailable",
                "latestStatus": None,
                "latestSignalDate": None,
                "latestExecutionDate": None,
                "latestRecordedAt": None,
                "calendarAssurance": "unavailable",
                "shadowAcceptanceCalendarEligible": False,
                "productionPretradeRiskCertified": False,
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
                "productionLiveCertified": False,
                "authoritativeCalendarCertified": False,
                "message": "No verified continuous paper execution evidence is available.",
            },
        }

    def _invalid(self, path: str, reason: str) -> dict[str, Any]:
        payload = self._unavailable(path, reason=reason)
        payload["journal"]["state"] = "invalid"
        payload["summary"]["attention"] = "critical"
        payload["operatorTruth"]["message"] = (
            "Paper execution evidence is invalid or unverifiable; fail closed."
        )
        return payload


__all__ = ["PaperExecutionEvidenceService"]
