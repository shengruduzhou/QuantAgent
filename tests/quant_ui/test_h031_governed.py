"""H-031: governed operational commands + read-only governance surface.

Deterministic, fixtured tests over the exact contract the VNext product depends
on — the allowlist, the network gate, path safety, no free-form shell, and the
performance-non-disclosure guard on the governance read surface.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.quant_api.config import ApiSettings
from services.quant_api.services.governance import GovernanceService, PerformanceLeakError
from services.quant_api.services.jobs import COMMANDS, JobManager

H031_COMMANDS = (
    ("governance", "validate-shadow-days"),
    ("governance", "certify-s4-batch-replay"),
    ("data", "build-u0-security-master"),
    ("data", "report-u0-provider-coverage"),
    ("data", "assemble-u0-full-universe"),
    ("data", "audit-u0-full-universe"),
    ("data", "backfill-u0-market-panel"),
)


# --- allowlist ---------------------------------------------------------------
@pytest.mark.parametrize("job_type,command_id", H031_COMMANDS)
def test_governed_command_registered_and_maps_to_backend_script(job_type, command_id, quant_ui_settings) -> None:
    spec = COMMANDS[command_id]
    assert spec["type"] == job_type
    assert spec["entrypoint"].startswith("scripts/")
    assert (Path(__file__).resolve().parents[2] / spec["entrypoint"]).exists()
    # fixed Runtime outputs, no user-supplied output path
    assert spec.get("fixed_outputs")
    for out in spec["fixed_outputs"]:
        assert out.startswith("runtime/")


@pytest.mark.parametrize("job_type,command_id", H031_COMMANDS)
def test_governed_commands_have_no_free_form_shell_field(job_type, command_id) -> None:
    spec = COMMANDS[command_id]
    forbidden = {"shell", "command", "cmd", "exec", "script", "eval", "bash", "sh"}
    assert not (set(spec["allowed"]) & forbidden)
    # allowed set is bounded and explicit (never a wildcard / free string)
    assert isinstance(spec["allowed"], set)


def test_default_u0_commands_validate_parameterless(quant_ui_settings) -> None:
    jm = JobManager(quant_ui_settings)
    for job_type, command_id in H031_COMMANDS:
        params = {"allow_network": True} if command_id == "backfill-u0-market-panel" else {}
        result = jm.validate(job_type, command_id, params)
        assert result["valid"] is True
        assert result["entrypoint"].startswith("scripts/")


# --- network gate ------------------------------------------------------------
def test_backfill_requires_explicit_network_confirmation(quant_ui_settings) -> None:
    jm = JobManager(quant_ui_settings)
    with pytest.raises(ValueError, match="allow_network"):
        jm.validate("data", "backfill-u0-market-panel", {"max_minutes": 30})
    # explicit confirmation passes
    assert jm.validate("data", "backfill-u0-market-panel", {"allow_network": True})["valid"]


def test_only_backfill_declares_a_network_control() -> None:
    for _, command_id in H031_COMMANDS:
        control = COMMANDS[command_id].get("control", set())
        if command_id == "backfill-u0-market-panel":
            assert control == {"allow_network"}
        else:
            assert not control


# --- path / param safety -----------------------------------------------------
def test_rogue_parameter_is_rejected(quant_ui_settings) -> None:
    jm = JobManager(quant_ui_settings)
    with pytest.raises(ValueError, match="unsupported parameters"):
        jm.validate("data", "audit-u0-full-universe", {"output": "/etc/passwd"})
    with pytest.raises(ValueError, match="unsupported parameters"):
        jm.validate("governance", "validate-shadow-days", {"shell": "rm -rf /"})


def test_wrong_job_type_is_rejected(quant_ui_settings) -> None:
    jm = JobManager(quant_ui_settings)
    with pytest.raises(ValueError, match="not allowed"):
        jm.validate("data", "validate-shadow-days", {})
    with pytest.raises(ValueError, match="not allowed"):
        jm.validate("governance", "audit-u0-full-universe", {})


# --- cancellation / resume ---------------------------------------------------
def test_queued_job_cancels_before_process_start(quant_ui_settings) -> None:
    from services.quant_api.services.jobs import JobRecord, _now
    jm = JobManager(quant_ui_settings)
    jm._jobs["job_x"] = JobRecord(id="job_x", type="data", status="queued",
                                  commandId="audit-u0-full-universe", createdAt=_now())
    result = jm.cancel("job_x")
    assert result["status"] == "cancelled"


def test_restart_marks_incomplete_jobs_failed_for_safe_resume(quant_ui_settings) -> None:
    from services.quant_api.services.jobs import JobRecord, _now
    jm = JobManager(quant_ui_settings)
    jm._jobs["job_live"] = JobRecord(id="job_live", type="data", status="running",
                                     commandId="backfill-u0-market-panel", createdAt=_now())
    jm._persist()
    reloaded = JobManager(quant_ui_settings)   # simulates an API restart
    assert reloaded._jobs["job_live"].status == "failed"
    assert "restarted" in (reloaded._jobs["job_live"].error or "")


# --- progress / pagination parsing ------------------------------------------
def test_progress_parses_paginated_counter() -> None:
    from services.quant_api.services.jobs import _progress_from_line
    assert _progress_from_line("[25 / 100] fetching") == pytest.approx(0.25)
    assert _progress_from_line(json.dumps({"rows_written": 3, "total_rows": 6})) == pytest.approx(0.5)
    assert _progress_from_line("no progress here") is None


# --- governance surface: unavailable states ----------------------------------
def test_governance_reports_unavailable_when_manifests_missing(empty_quant_ui_settings) -> None:
    gov = GovernanceService(empty_quant_ui_settings)
    status = gov.status()   # must not raise even with an empty runtime
    assert status["shadow"]["status"] == "unavailable"
    assert status["u0"]["status"] == "unavailable"
    assert status["s4"]["status"] == "unavailable"
    assert len(status["governedCommands"]) == 7


# --- governance surface: honest ready extraction -----------------------------
def _write_governance_fixture(settings: ApiSettings, *, leak: bool = False) -> None:
    root = settings.runtime_root
    fb = root / "paper" / "fresh_blind"
    fb.mkdir(parents=True, exist_ok=True)
    (fb / "shadow_day_registry.json").write_text(json.dumps({
        "valid_shadow_days": 2, "required_days": 7,
        "valid_dates": ["2026-07-21", "2026-07-22"],
        "ledger_chain_valid": True, "ledger_records_total": 11,
        "fidelity_certificate_passes": True, "certificate_sha256": "37193bb82a477abc",
        "unblind_or_nonroutine_accesses": [],
        "days": [{"trade_date": "2026-07-17", "valid_shadow_day": False,
                  "invalid_reason": ("leaked sharpe value" if leak else "data_status=FAILED")}],
    }))
    (fb / "shadow_accumulating_status.json").write_text(json.dumps(
        {"next_expected_valid_date": "2026-07-23"}))
    h030 = root / "reports" / "h030"
    h030.mkdir(parents=True, exist_ok=True)
    (h030 / "s4_readiness_certificate.json").write_text(json.dumps({
        "decision": "S4_BATCH_REPLAY_READY", "exact_reproduction_vs_frozen_trace": True,
        "deterministic_double_run": True, "archived_inputs_complete": True,
        "refit_cutoffs_replayed": 26, "semantics_changed": False, "fresh_access": False,
    }))
    u0 = root / "data" / "u0"
    u0.mkdir(parents=True, exist_ok=True)
    (u0 / "full_universe_readiness_certificate.json").write_text(json.dumps({
        "data_readiness_state": "FULL_UNIVERSE_DATA_NOT_READY_COVERAGE",
        "training_permitted": False,
        "gate_pass": {"integration": True, "provider": True, "coverage": False, "pit": False},
        "gates": {"coverage": {"covered_by_board": {"SH_Main": 1562}, "boards_absent": ["STAR", "BSE"],
                               "blocked_by_data": 1860},
                  "pit": {"st_history": "BLOCKED_BY_DATA"}},
    }))
    (u0 / "provider_coverage_summary.json").write_text(json.dumps({"covered_bar_history": 4028}))
    (u0 / "pit_field_availability.json").write_text(json.dumps(
        {"pit_field_availability": {"st_intervals": "BLOCKED_BY_DATA"}}))


def test_governance_ready_extraction_has_no_performance(quant_ui_settings) -> None:
    _write_governance_fixture(quant_ui_settings)
    status = GovernanceService(quant_ui_settings).status()
    assert status["shadow"]["validDays"] == 2
    assert status["s4"]["decision"] == "S4_BATCH_REPLAY_READY"
    assert status["u0"]["dataReadinessState"] == "FULL_UNIVERSE_DATA_NOT_READY_COVERAGE"
    assert status["u0"]["trainingPermitted"] is False
    assert status["u0"]["boardsAbsent"] == ["STAR", "BSE"]


def test_governance_guard_blocks_a_performance_leak(quant_ui_settings) -> None:
    _write_governance_fixture(quant_ui_settings, leak=True)
    with pytest.raises(PerformanceLeakError):
        GovernanceService(quant_ui_settings).status()


def test_unavailable_status_string_does_not_false_trip_leak_guard(empty_quant_ui_settings) -> None:
    # "unavailable" contains the substring "nav"; the guard must use word bounds.
    GovernanceService(empty_quant_ui_settings).status()  # must not raise
