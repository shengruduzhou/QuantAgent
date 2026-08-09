from __future__ import annotations

from pathlib import Path

from quantagent.paper.execution_journal import PendingExecutionJournal
from quantagent.paper.runtime_paths import paper_runtime_paths
from services.quant_api.config import ApiSettings
from services.quant_api.services.paper_execution_evidence import (
    PaperExecutionEvidenceService,
)


def _settings(tmp_path: Path) -> ApiSettings:
    runtime = tmp_path / "runtime"
    return ApiSettings(
        project_root=tmp_path,
        runtime_root=runtime,
        cache_root=runtime / "cache" / "quant_ui",
        jobs_root=runtime / "jobs" / "quant_ui",
    ).ensure()


def test_paper_runtime_paths_are_bound_to_quantagent_home(tmp_path: Path) -> None:
    paths = paper_runtime_paths(tmp_path / "qa-home")
    assert paths.root == tmp_path / "qa-home" / "paper"
    assert paths.pending_signals == paths.root / "pending_signals"
    assert paths.execution_journal == paths.root / "execution_journal.jsonl"
    assert paths.canonical_ledger == paths.root / "canonical_ledger.jsonl"
    assert paths.operational_ledger == paths.root / "operational_ledger.jsonl"
    assert paths.idempotency == paths.root / "idempotency.jsonl"


def test_missing_journal_is_unavailable_not_healthy(tmp_path: Path) -> None:
    result = PaperExecutionEvidenceService(_settings(tmp_path)).status()
    assert result["journal"]["state"] == "unavailable"
    assert result["journal"]["verified"] is False
    assert result["summary"]["attention"] == "unavailable"
    assert result["operatorTruth"]["paperExecutionEvidence"] is False
    assert result["operatorTruth"]["productionLiveCertified"] is False


def test_verified_observation_never_implies_live_certification(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = paper_runtime_paths(settings.runtime_root).execution_journal
    journal = PendingExecutionJournal(path)
    journal.append(
        pending_payload_sha256="payload-a",
        signal_date="2026-08-07",
        execution_date="2026-08-10",
        status="execution_started",
        details={"order_count": 1},
        recorded_at="2026-08-10T06:58:00+00:00",
    )
    journal.append(
        pending_payload_sha256="payload-a",
        signal_date="2026-08-07",
        execution_date="2026-08-10",
        status="execution_observed",
        details={
            "order_count": 1,
            "fill_count": 1,
            "nav_before": 1_000_000.0,
            "nav_after": 1_001_200.0,
            "calendar_assurance": "observed_market_panel_only",
            "shadow_acceptance_calendar_eligible": False,
            "production_pretrade_risk_certified": False,
            "risk_scope": "paper_simulator_admissibility_only",
            "session_closed": True,
        },
        recorded_at="2026-08-10T07:01:00+00:00",
    )

    result = PaperExecutionEvidenceService(settings).status()
    assert result["journal"]["state"] == "valid"
    assert result["journal"]["verified"] is True
    assert result["summary"]["latestStatus"] == "execution_observed"
    assert result["summary"]["fillCount"] == 1
    assert result["summary"]["calendarAssurance"] == "observed_market_panel_only"
    assert result["summary"]["shadowAcceptanceCalendarEligible"] is False
    assert result["summary"]["productionPretradeRiskCertified"] is False
    assert result["operatorTruth"]["paperExecutionEvidence"] is True
    assert result["operatorTruth"]["productionLiveCertified"] is False
    assert result["operatorTruth"]["authoritativeCalendarCertified"] is False


def test_unresolved_execution_start_is_critical(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    journal = PendingExecutionJournal(
        paper_runtime_paths(settings.runtime_root).execution_journal
    )
    journal.append(
        pending_payload_sha256="payload-crash",
        signal_date="2026-08-07",
        execution_date="2026-08-10",
        status="execution_started",
        details={"order_count": 1},
    )

    result = PaperExecutionEvidenceService(settings).status()
    assert result["journal"]["unresolvedCount"] == 1
    assert result["summary"]["attention"] == "critical"
    assert result["operatorTruth"]["productionLiveCertified"] is False


def test_tampered_hash_chain_fails_closed_without_returning_records(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = paper_runtime_paths(settings.runtime_root).execution_journal
    journal = PendingExecutionJournal(path)
    journal.append(
        pending_payload_sha256="payload-a",
        signal_date="2026-08-07",
        execution_date="2026-08-10",
        status="execution_started",
        details={"order_count": 1},
    )
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"order_count": 1', '"order_count": 9'), encoding="utf-8")

    result = PaperExecutionEvidenceService(settings).status()
    assert result["journal"]["state"] == "invalid"
    assert result["summary"]["attention"] == "critical"
    assert result["records"] == []
    assert result["operatorTruth"]["paperExecutionEvidence"] is False
    assert result["operatorTruth"]["productionLiveCertified"] is False
