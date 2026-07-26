"""U0 readiness gates must be computed from evidence, never hardcoded.

Each test builds a synthetic artifact tree — the same file layout a real
acquisition run produces — and asserts that the certificate reacts to what the
artifacts say. The regression these lock down is a certificate that asserted
``adjustment_method_explicit = True`` and ``volume_amount_units_verified = True``
as literal constants while the panel underneath silently mixed forward-adjusted
and unadjusted prices.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pd = pytest.importorskip("pandas")

from quantagent.data.ashare.readiness import (  # noqa: E402
    BAR_NOT_READY_COVERAGE,
    BAR_NOT_READY_IDENTITY,
    BAR_NOT_READY_QUALITY,
    BAR_READY,
    MANDATORY_PIT_FIELDS,
    NOT_READY_COVERAGE,
    NOT_READY_INTEGRATION,
    NOT_READY_PIT,
    READY,
    build_certificates,
    render_report,
)

ALL_CHECKS = [
    "schema_columns", "schema_dtypes", "timestamp_type", "duplicate_symbol_date",
    "ohlc_relationships", "non_positive_prices", "null_close", "negative_volume",
    "amount_volume_units", "volume_unit_is_shares", "pre_listing_rows",
    "post_delisting_rows", "price_limit_plausibility", "adjustment_is_raw",
    "pit_available_at", "suspension_representation", "freshness",
    "cross_provider_reconciliation", "intraday_to_daily_reconciliation",
    "symbol_normalisation",
]


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def build_tree(root: Path, *, coverage_full: bool = True, failing_checks: tuple[str, ...] = (),
               not_run_checks: tuple[str, ...] = (), pit_complete: bool = True,
               boards: tuple[str, ...] = ("SH_Main", "SZ_Main", "ChiNext", "STAR", "BSE"),
               delisted_in_master: int = 12, drop_evidence: tuple[str, ...] = (),
               pre_listing: tuple[str, ...] = ()) -> Path:
    u0 = root / "runtime/data/u0"
    (u0 / "panel").mkdir(parents=True, exist_ok=True)

    if "panel_manifest" not in drop_evidence:
        _write(u0 / "panel/panel_manifest.json", {
            "adjustment_method": "none (raw traded prices) — verified against an independent provider",
            "volume_unit": "shares", "amount_unit": "CNY",
            "serving_provider_counts": {"tickflow": 40, "tencent": 10},
            "quality_checks": {"rows": 1000, "symbols": 50, "amount_coverage": 1.0},
        })
    if "validation" not in drop_evidence:
        checks = []
        for name in ALL_CHECKS:
            verdict = ("FAIL" if name in failing_checks
                       else "NOT_RUN" if name in not_run_checks else "PASS")
            checks.append({"check": name, "verdict": verdict, "detail": "", "evidence": {}})
        _write(u0 / "validation/validation_report.json", {"checks": checks})
    if "capability" not in drop_evidence:
        _write(u0 / "capability/provider_capability_matrix.json", {
            "serving_providers_by_family": {
                "daily_bars": ["tickflow", "tencent"], "security_master": ["tickflow"],
                "adjust_factors": ["sina"], "corporate_actions": ["sina"],
                "quotes": ["tickflow"], "quotes_l1_depth5": ["tencent"],
                "minute_bars": ["tencent"],
            },
            "blockers": [{"provider": "baostock", "dataset_family": "transport",
                          "status": "BLOCKED_BY_ENVIRONMENT", "detail": "port 10030"}],
        })
    if "master" not in drop_evidence:
        _write(u0 / "security_master_manifest.json", {
            "securities": 50, "by_board": {b: 10 for b in boards},
            "by_status": {"listed": 50 - delisted_in_master, "delisted": delisted_in_master},
            "bse_current_920": 10, "bse_legacy_codes": 0,
            "listing_date_coverage": 50, "delisting_date_coverage": delisted_in_master,
            "sources": {"tickflow_instruments": {"SH": {"rows": 10}}},
        })
    if "coverage" not in drop_evidence:
        rows = []
        for board in boards:
            for index in range(10):
                covered = coverage_full or index < 6
                rows.append({"symbol": f"{board}{index}", "board": board,
                             "status": "delisted" if index == 0 else "listed",
                             "covered": covered, "rows": 100 if covered else 0,
                             "serving_provider": "tickflow" if covered else "none",
                             "blocked_reason": "" if covered else "NOT_YET_ACQUIRED"})
        pd.DataFrame(rows).to_parquet(u0 / "panel/coverage_matrix.parquet", index=False)
        if pre_listing:
            pd.DataFrame([
                {"symbol": s, "disposition": "PRE_LISTING_NO_SESSIONS"} for s in pre_listing
            ] + [
                {"symbol": r["symbol"], "disposition": "LISTED_WITH_HISTORY"}
                for r in rows if r["symbol"] not in pre_listing
            ]).to_parquet(u0 / "master_disposition.parquet", index=False)

    _write(u0 / "pit/trading_calendar_manifest.json",
           {"rows": 8797, "first": "1990-12-19", "last": "2026-12-31"})
    _write(u0 / "pit/adjust_factors_manifest.json", {"rows": 36155, "symbols_with_data": 2803})
    _write(u0 / "pit/corporate_actions_manifest.json", {"rows": 5000, "symbols_with_data": 900})
    _write(u0 / "pit/suspension_manifest.json",
           {"intervals": 460, "symbols_with_halts": 460,
            "snapshot_date_range": ["20251029", "20260724"]})
    _write(u0 / "pit/st_manifest.json", {
        "current_st_names": 333,
        "dated_episodes": 906,
        "securities_with_dated_episodes": 651,
        "exchanges_with_dated_history": ["SSE", "SZSE", "BSE"] if pit_complete else ["SZSE"],
        # an exchange with no dated register must not be treated as never-ST
        "exchanges_without_dated_history": [] if pit_complete else ["SSE", "BSE"],
        "historical_intervals_status": "AVAILABLE",
    })
    return root


def test_complete_evidence_yields_ready_and_permits_training(tmp_path):
    build_tree(tmp_path)
    certificates = build_certificates(tmp_path)
    assert certificates["overall"]["data_readiness_state"] == READY
    assert certificates["overall"]["training_permitted"] is True
    assert certificates["bar"]["decision"] == BAR_READY
    assert certificates["pit"]["training_permitted"] is True


def test_missing_evidence_blocks_at_integration_rather_than_defaulting_to_pass(tmp_path):
    build_tree(tmp_path, drop_evidence=("validation",))
    certificates = build_certificates(tmp_path)
    overall = certificates["overall"]
    assert overall["data_readiness_state"] == NOT_READY_INTEGRATION
    assert overall["training_permitted"] is False
    assert any("validation_report" in path for path in overall["missing_evidence"])


def test_incomplete_universe_coverage_blocks_the_certificate(tmp_path):
    build_tree(tmp_path, coverage_full=False)
    certificates = build_certificates(tmp_path)
    assert certificates["overall"]["data_readiness_state"] == NOT_READY_COVERAGE
    assert certificates["bar"]["decision"] == BAR_NOT_READY_COVERAGE
    coverage = certificates["overall"]["gates"]["coverage"]
    assert coverage["covered_securities"] < coverage["master_securities"]
    assert coverage["not_yet_acquired"] > 0


def test_uncovered_security_blocks_when_no_disposition_evidence_exists(tmp_path):
    """Without the exchange-register artifact the gate stays strict."""
    build_tree(tmp_path, coverage_full=False)
    coverage = build_certificates(tmp_path)["overall"]["gates"]["coverage"]
    assert coverage["pass"] is False
    assert coverage["unexplained_uncovered"], "uncovered names must be named, not tolerated"
    assert "master_disposition.parquet absent" in coverage["disposition_evidence"]


def test_exchange_confirmed_pre_listing_name_is_excluded_from_the_denominator(tmp_path):
    """A security the exchange does not list cannot have bars and never gets placeholders."""
    build_tree(tmp_path, coverage_full=False,
               pre_listing=tuple(f"{b}{i}" for b in ("SH_Main", "SZ_Main", "ChiNext", "STAR", "BSE")
                                 for i in range(6, 10)))
    coverage = build_certificates(tmp_path)["overall"]["gates"]["coverage"]
    assert coverage["unexplained_uncovered"] == []
    assert coverage["pass"] is True
    assert coverage["expected_securities"] < coverage["master_securities"]
    assert coverage["covered_securities"] == coverage["expected_securities"]


def test_a_pre_listing_disposition_cannot_excuse_an_unrelated_gap(tmp_path):
    """Excluding one confirmed non-trading name must not excuse a different gap."""
    build_tree(tmp_path, coverage_full=False, pre_listing=("SH_Main6",))
    coverage = build_certificates(tmp_path)["overall"]["gates"]["coverage"]
    assert "SH_Main6" not in coverage["unexplained_uncovered"]
    assert coverage["unexplained_uncovered"], "other uncovered names must still block"
    assert coverage["pass"] is False


def test_a_failing_validation_check_fails_the_quality_gate(tmp_path):
    build_tree(tmp_path, failing_checks=("adjustment_is_raw",))
    certificates = build_certificates(tmp_path)
    assert certificates["bar"]["decision"] == BAR_NOT_READY_QUALITY
    assert "adjustment_is_raw" in certificates["overall"]["gates"]["quality"]["failures"]
    assert certificates["overall"]["training_permitted"] is False


def test_a_check_that_never_ran_cannot_count_as_a_pass(tmp_path):
    build_tree(tmp_path, not_run_checks=("cross_provider_reconciliation",))
    certificates = build_certificates(tmp_path)
    quality = certificates["overall"]["gates"]["quality"]
    assert quality["pass"] is False
    assert "cross_provider_reconciliation" in quality["not_run"]


def test_blocked_pit_field_blocks_training_but_not_bar_readiness(tmp_path):
    build_tree(tmp_path, pit_complete=False)
    certificates = build_certificates(tmp_path)
    assert certificates["bar"]["decision"] == BAR_READY
    assert certificates["overall"]["data_readiness_state"] == NOT_READY_PIT
    assert certificates["overall"]["training_permitted"] is False
    assert "st_intervals" in certificates["pit"]["blocked_pit_fields"]


def test_partial_st_history_is_blocked_not_rounded_up_to_available(tmp_path):
    """An exchange with no dated ST register must not read as 'never ST'."""
    build_tree(tmp_path, pit_complete=False)
    status = build_certificates(tmp_path)["pit"]["pit_field_availability"]["st_intervals"]
    assert status.startswith("BLOCKED_BY_DATA")
    assert "PARTIAL" in status
    # the partial coverage that DOES exist is still reported, not discarded
    assert "906" in status and "SZSE" in status and "SSE" in status


def test_a_board_missing_from_the_master_fails_identity(tmp_path):
    build_tree(tmp_path, boards=("SH_Main", "SZ_Main", "ChiNext", "STAR"))
    certificates = build_certificates(tmp_path)
    assert certificates["bar"]["decision"] == BAR_NOT_READY_IDENTITY
    assert "BSE" in certificates["overall"]["gates"]["identity"]["boards_absent_from_master"]


def test_a_survivorship_only_master_fails_identity(tmp_path):
    build_tree(tmp_path, delisted_in_master=0)
    certificates = build_certificates(tmp_path)
    assert certificates["overall"]["gate_pass"]["identity"] is False


def test_fallback_provider_usage_is_measured_not_asserted(tmp_path):
    build_tree(tmp_path)
    provider = build_certificates(tmp_path)["overall"]["gates"]["provider"]
    # 10 of the 50 symbols were served by the public fallback in the fixture
    assert provider["fallback_provider_symbols_served"] == 10
    assert provider["fallback_providers_exercised"] is True


def test_environment_blockers_are_carried_into_the_certificate(tmp_path):
    build_tree(tmp_path)
    provider = build_certificates(tmp_path)["overall"]["gates"]["provider"]
    assert any(b["provider"] == "baostock" for b in provider["environment_blockers"])


def test_every_mandatory_pit_field_is_reported_with_a_status(tmp_path):
    build_tree(tmp_path)
    availability = build_certificates(tmp_path)["pit"]["pit_field_availability"]
    for field in MANDATORY_PIT_FIELDS:
        assert field in availability and availability[field]


def test_report_renders_the_decision_and_never_claims_a_hardcoded_gate(tmp_path):
    build_tree(tmp_path, coverage_full=False)
    report = render_report(build_certificates(tmp_path)["overall"])
    assert NOT_READY_COVERAGE in report
    assert "no gate is hardcoded" in report


def test_readiness_module_contains_no_hardcoded_true_gate_constants():
    source = (REPO / "src/quantagent/data/ashare/readiness.py").read_text()
    for banned in ("volume_amount_units_verified\": True", "adjustment_method_explicit\": True",
                   "\"zero_pre_listing_rows\": True"):
        assert banned not in source
