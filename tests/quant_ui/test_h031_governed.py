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
    ("data", "probe-u0-star-bse"),
    # H-032B additions
    ("data", "benchmark-tickflow-capability"),
    ("data", "audit-bse-identity"),
    ("data", "audit-u0-pit-readiness"),
    ("data", "report-u0-bar-readiness"),
    # H-032C additions
    ("data", "source-u0-pit-metadata"),
    ("data", "audit-tickflow-entitlement"),
    ("data", "report-u0-reconciliation"),
)
NETWORK_COMMANDS = {"backfill-u0-market-panel", "probe-u0-star-bse",
                    "benchmark-tickflow-capability", "audit-bse-identity",
                    "source-u0-pit-metadata", "audit-tickflow-entitlement"}


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
        params = {"allow_network": True} if command_id in NETWORK_COMMANDS else {}
        result = jm.validate(job_type, command_id, params)
        assert result["valid"] is True
        assert result["entrypoint"].startswith("scripts/")


def test_backfill_accepts_priority_boards_and_rejects_unknown(quant_ui_settings) -> None:
    jm = JobManager(quant_ui_settings)
    assert jm.validate("data", "backfill-u0-market-panel",
                       {"allow_network": True, "priority_boards": "STAR,BSE", "max_minutes": 20})["valid"]
    with pytest.raises(ValueError, match="unsupported parameters"):
        jm.validate("data", "backfill-u0-market-panel", {"allow_network": True, "board": "STAR"})


# --- network gate ------------------------------------------------------------
def test_network_commands_require_explicit_confirmation(quant_ui_settings) -> None:
    jm = JobManager(quant_ui_settings)
    for command_id in NETWORK_COMMANDS:
        with pytest.raises(ValueError, match="allow_network"):
            jm.validate("data", command_id, {})
        assert jm.validate("data", command_id, {"allow_network": True})["valid"]


def test_only_network_commands_declare_a_network_control() -> None:
    for _, command_id in H031_COMMANDS:
        control = COMMANDS[command_id].get("control", set())
        if command_id in NETWORK_COMMANDS:
            assert control == {"allow_network"}
        else:
            assert not control


@pytest.mark.parametrize("entrypoint,args", [
    ("scripts/u0_star_bse_probe.py", []),
    ("scripts/u0_full_universe_backfill.py", ["fetch"]),
    ("scripts/tickflow_capability_benchmark.py", []),
    ("scripts/u0_bse_identity.py", []),
])
def test_network_scripts_fail_closed_without_allow_network(entrypoint, args) -> None:
    """The backend scripts themselves refuse (exit 2) before any vendor call."""
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, str(repo / entrypoint), *args],
                          capture_output=True, text=True, cwd=repo, timeout=60)
    assert proc.returncode == 2
    assert "allow-network" in (proc.stdout + proc.stderr) or "refusing" in (proc.stdout + proc.stderr)


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
    # 15 legacy U0/shadow/S4 commands + the 8 A-share data-foundation commands
    command_ids = [c["commandId"] for c in status["governedCommands"]]
    assert len(command_ids) == 23
    for command_id in ("probe-ashare-capabilities", "build-u0-live-security-master",
                       "acquire-u0-daily-bars", "build-u0-pit-intervals",
                       "acquire-u0-intraday-bars", "assemble-u0-raw-panel",
                       "validate-u0-data", "audit-u0-adjustment-forensics"):
        assert command_id in command_ids
    # every acquisition command must declare that it needs explicit network approval
    by_id = {c["commandId"]: c for c in status["governedCommands"]}
    for command_id in ("probe-ashare-capabilities", "acquire-u0-daily-bars",
                       "acquire-u0-intraday-bars", "build-u0-live-security-master"):
        assert by_id[command_id]["requiresNetwork"] is True
    # the empty runtime must also report the foundation surface as unavailable
    assert status["ashareFoundation"]["status"] == "unavailable"


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
    # evidence-driven certificate shape produced by scripts/u0_audit.py
    (u0 / "full_universe_readiness_certificate.json").write_text(json.dumps({
        "data_readiness_state": "FULL_UNIVERSE_DATA_NOT_READY_COVERAGE",
        "training_permitted": False,
        "gate_pass": {"integration": True, "provider": True, "identity": True,
                      "coverage": False, "quality": True, "pit": False},
        "missing_evidence": [],
        "evidence_sources": {"panel_manifest": "runtime/data/u0/panel/panel_manifest.json"},
        "panel_sha256": "abc123",
        "gates": {
            "coverage": {"by_board": {"SH_Main": {"covered": 232, "total": 1848}},
                         "by_status": {"delisted": {"covered": 16, "total": 361}},
                         "boards_with_zero_coverage": ["STAR", "BSE"],
                         "covered_securities": 2020, "master_securities": 5894,
                         "coverage_share": 0.3427, "not_yet_acquired": 3874},
            "identity": {"securities": 5894, "bse_current_920": 330, "bse_legacy_codes": 0,
                         "delisted_in_master": 361, "symbol_normalisation": "PASS"},
            "provider": {"serving_providers_by_family": {"daily_bars": ["tickflow"]},
                         "families_without_provider": [],
                         "fallback_providers_exercised": True,
                         "fallback_provider_symbols_served": 128,
                         "environment_blockers": []},
            "quality": {"verdicts": {"adjustment_is_raw": "PASS"}, "failures": [],
                        "not_run": [], "adjustment_method": "none (raw traded prices)",
                        "volume_unit": "shares", "amount_unit": "CNY", "amount_coverage": 1.0},
            "pit": {"field_availability": {"st_intervals": "BLOCKED_BY_DATA"},
                    "blocked_fields": ["st_intervals"],
                    "suspension_coverage_window": ["20251029", "20260724"]},
        },
    }))
    (u0 / "panel").mkdir(parents=True, exist_ok=True)
    (u0 / "panel" / "panel_manifest.json").write_text(json.dumps({
        "serving_provider_counts": {"tickflow": 2020},
        "quality_checks": {"rows": 2769268, "symbols": 2020, "min_date": "1990-12-19",
                           "max_date": "2026-07-24", "session_gaps_suspended": 523,
                           "session_gaps_unexplained": 44542},
    }))
    (u0 / "capability").mkdir(parents=True, exist_ok=True)
    (u0 / "capability" / "provider_capability_matrix.json").write_text(json.dumps({
        "probes": 71, "supported_probes": 32,
        "providers_with_any_support": ["tickflow", "tencent", "sina"],
        "serving_providers_by_family": {"daily_bars": ["tickflow", "tencent"], "l2_depth": []},
        "families_without_any_provider": ["l2_depth"],
        "blockers": [{"provider": "baostock", "dataset_family": "transport",
                      "status": "BLOCKED_BY_ENVIRONMENT", "detail": "TCP 10030"}],
        "environment": {"egress": "TCP 80/443 only"},
    }))


def test_governance_ready_extraction_has_no_performance(quant_ui_settings) -> None:
    _write_governance_fixture(quant_ui_settings)
    status = GovernanceService(quant_ui_settings).status()
    assert status["shadow"]["validDays"] == 2
    assert status["s4"]["decision"] == "S4_BATCH_REPLAY_READY"
    assert status["u0"]["dataReadinessState"] == "FULL_UNIVERSE_DATA_NOT_READY_COVERAGE"
    assert status["u0"]["trainingPermitted"] is False
    assert status["u0"]["boardsAbsent"] == ["STAR", "BSE"]
    # the surface reports measured coverage, declared units and the fallback fact
    assert status["u0"]["coveredSecurities"] == 2020
    assert status["u0"]["masterSecurities"] == 5894
    assert status["u0"]["quality"]["volumeUnit"] == "shares"
    assert status["u0"]["provider"]["fallbackProvidersExercised"] is True
    assert status["u0"]["blockedPitFields"] == ["st_intervals"]
    # the capability matrix surfaces a dataset family that no provider serves
    foundation = status["ashareFoundation"]
    assert foundation["status"] == "ready"
    assert "l2_depth" in foundation["capability"]["familiesWithoutAnyProvider"]
    assert foundation["capability"]["blockers"][0]["status"] == "BLOCKED_BY_ENVIRONMENT"


def test_governance_guard_blocks_a_performance_leak(quant_ui_settings) -> None:
    _write_governance_fixture(quant_ui_settings, leak=True)
    with pytest.raises(PerformanceLeakError):
        GovernanceService(quant_ui_settings).status()


def test_unavailable_status_string_does_not_false_trip_leak_guard(empty_quant_ui_settings) -> None:
    # "unavailable" contains the substring "nav"; the guard must use word bounds.
    GovernanceService(empty_quant_ui_settings).status()  # must not raise
