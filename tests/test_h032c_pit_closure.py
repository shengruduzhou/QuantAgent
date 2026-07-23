"""H-032C: strict PIT metadata sourcing, universe reconciliation, entitlement
re-test contract. Guarded skip-if-absent for CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
U0 = REPO / "runtime/data/u0"
PIT = U0 / "pit"

PROVENANCE_COLS = {"symbol", "effective_start", "available_at", "source",
                   "source_timestamp", "source_hash"}


def _pd():
    return pytest.importorskip("pandas")


def test_pit_metadata_manifest_closes_delisting_with_provenance() -> None:
    m = PIT / "pit_metadata_manifest.json"
    if not m.exists():
        pytest.skip("pit metadata not sourced")
    mj = json.loads(m.read_text())
    # delisting/price-limit/ipo are closed; st/suspension/corp-action remain blocked
    assert "delisting_intervals" in mj["closed_fields"]
    assert "price_limit_regimes" in mj["closed_fields"]
    assert set(mj["blocked_fields"]) <= {"st_intervals", "suspension_intervals",
                                         "corporate_action_identity"}
    assert mj["delisting_dates_sourced"] >= 0


def test_pit_interval_tables_carry_full_provenance() -> None:
    pd = _pd()
    for name in ("price_limit_regimes.parquet", "ipo_special_limit_intervals.parquet",
                 "delisting_intervals.parquet"):
        p = PIT / name
        if not p.exists():
            pytest.skip(f"{name} not generated")
        df = pd.read_parquet(p)
        if len(df):
            assert PROVENANCE_COLS <= set(df.columns), name


def test_strict_pit_reflects_delisting_closure() -> None:
    cert = U0 / "u0_strict_pit_certificate.json"
    if not cert.exists():
        pytest.skip("strict pit cert not generated")
    c = json.loads(cert.read_text())
    # delisting must no longer be a blocked PIT field once sourced
    assert "delisting_intervals" not in c["blocked_pit_fields"]
    # training only on full readiness
    assert c["training_permitted"] == (c["decision"] == "FULL_UNIVERSE_DATA_READY")


def test_reconciliation_prevents_bse_dual_identity() -> None:
    r = U0 / "universe_reconciliation.json"
    if not r.exists():
        pytest.skip("reconciliation not generated")
    rj = json.loads(r.read_text())
    guard = rj["dual_identity_guard"]
    # no old 8xxxxx code may exist as a separate BSE security
    assert guard["dual_identity_collisions"] == 0
    assert guard["old_8xxxxx_codes_in_master"] == []


def test_reconciliation_only_adds_authoritative_listings() -> None:
    r = U0 / "universe_reconciliation.json"
    if not r.exists():
        pytest.skip("reconciliation not generated")
    rj = json.loads(r.read_text())
    # additions come from the authoritative akshare BSE list; none rejected-but-added
    assert rj["rejected_identities"] == []
    for sym in rj.get("supplemental_additions_symbols", []):
        assert sym.endswith(".BJ")  # only BSE recent listings this pass


def test_entitlement_audit_keeps_tickflow_primary_and_no_fabrication() -> None:
    a = REPO / "runtime/reports/h032c/tickflow_entitlement_audit.json"
    if not a.exists():
        pytest.skip("entitlement audit not run yet (may be deferred behind Track-F)")
    aj = json.loads(a.read_text())
    assert "TickFlow" in aj.get("primary_bar_provider", "")
    # if ex_factors is not entitled, corporate actions must be ALTERNATIVE_SOURCE_REQUIRED
    if aj.get("status") != "DEFERRED" and "NOT_ENTITLED" in str(aj.get("ex_factors", "")):
        assert aj["corporate_action_classification"] == "ALTERNATIVE_SOURCE_REQUIRED"
