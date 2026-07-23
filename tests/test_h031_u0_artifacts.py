"""H-031 Track U0: schema/provenance/PIT contract over the produced artifacts.

Guarded skip-if-absent so the suite passes in a clean CI checkout, while giving
real coverage where the artifacts have been generated. Never asserts anything
about candidate performance.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
U0 = REPO / "runtime/data/u0"
COVERAGE = U0 / "provider_coverage_matrix.parquet"
MASTER = U0 / "historical_security_master.parquet"
PIT_AVAIL = U0 / "pit_field_availability.json"
READINESS = U0 / "full_universe_readiness_certificate.json"

REQUIRED_COVERAGE_COLUMNS = [
    "symbol", "exchange", "board", "security_type", "listing_date", "delisting_date",
    "current_status", "tickflow_status", "tushare_status", "akshare_status", "qlib_status",
    "exchange_metadata_status", "selected_bar_provider", "selected_metadata_provider",
    "coverage_start", "coverage_end", "bar_count", "expected_trading_days",
    "actual_trading_days", "missing_ratio", "adjustment_method", "volume_unit",
    "amount_unit", "source_timestamp", "source_hash", "blocked_reason",
]

VALID_READINESS_STATES = {
    "FULL_UNIVERSE_DATA_READY", "FULL_UNIVERSE_DATA_NOT_READY_COVERAGE",
    "FULL_UNIVERSE_DATA_NOT_READY_PIT", "FULL_UNIVERSE_DATA_NOT_READY_PROVIDER",
    "FULL_UNIVERSE_DATA_NOT_READY_INTEGRATION",
}


def _pd():
    return pytest.importorskip("pandas")


def test_provider_coverage_matrix_has_exact_required_schema() -> None:
    if not COVERAGE.exists():
        pytest.skip("provider_coverage_matrix not generated")
    pd = _pd()
    m = pd.read_parquet(COVERAGE)
    assert list(m.columns) == REQUIRED_COVERAGE_COLUMNS
    # both parquet and csv are emitted
    assert (U0 / "provider_coverage_matrix.csv").exists()


def test_empty_vendor_response_is_not_recorded_as_no_history() -> None:
    """An EMPTY vendor response must be distinct from NOT_PROBED — never 'no data'."""
    if not COVERAGE.exists():
        pytest.skip("provider_coverage_matrix not generated")
    pd = _pd()
    m = pd.read_parquet(COVERAGE)
    statuses = set(m["tickflow_status"].unique())
    # the classification vocabulary keeps EMPTY and NOT_PROBED separate
    assert "EMPTY" in statuses or "NOT_PROBED" in statuses
    empties = m[m["tickflow_status"] == "EMPTY"]
    for reason in empties["blocked_reason"]:
        assert "not_proof_of_no_history" in str(reason)


def test_uncovered_security_is_blocked_by_data_not_defaulted() -> None:
    if not COVERAGE.exists():
        pytest.skip("provider_coverage_matrix not generated")
    pd = _pd()
    m = pd.read_parquet(COVERAGE)
    uncovered = m[m["selected_bar_provider"] == "NONE"]
    if len(uncovered):
        assert uncovered["blocked_reason"].str.startswith("BLOCKED_BY_DATA").all()


def test_security_master_marks_missing_pit_fields_blocked_not_false() -> None:
    if not PIT_AVAIL.exists():
        pytest.skip("pit_field_availability not generated")
    avail = json.loads(PIT_AVAIL.read_text())["pit_field_availability"]
    # ST / suspension / corporate-action have no PIT source on disk -> BLOCKED_BY_DATA
    for field in ("st_intervals", "suspension_intervals", "corporate_action_identity"):
        assert str(avail[field]).startswith("BLOCKED_BY_DATA")


def test_readiness_decision_uses_the_defined_state_vocabulary() -> None:
    if not READINESS.exists():
        pytest.skip("readiness certificate not generated")
    cert = json.loads(READINESS.read_text())
    assert cert["data_readiness_state"] in VALID_READINESS_STATES
    # training is permitted only in the READY state
    assert cert["training_permitted"] == (cert["data_readiness_state"] == "FULL_UNIVERSE_DATA_READY")


def test_readiness_certificate_carries_no_performance() -> None:
    if not READINESS.exists():
        pytest.skip("readiness certificate not generated")
    import re
    text = READINESS.read_text().lower()
    assert not re.search(r"\b(nav|sharpe|cagr|drawdown|calmar|sortino|pnl)\b", text)
