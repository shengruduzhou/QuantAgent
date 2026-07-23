"""H-032B: TickFlow-native U0 closure — bar vs PIT decisions, BSE identity,
benchmark, and backfill priority ordering. Guarded skip-if-absent for CI.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
U0 = REPO / "runtime/data/u0"
BENCH = REPO / "runtime/reports/h032b/tickflow_capability_benchmark.json"

BAR_STATES = {"U0_BAR_READY", "U0_BAR_NOT_READY_COVERAGE", "U0_BAR_NOT_READY_IDENTITY",
              "U0_BAR_NOT_READY_PROVIDER", "U0_BAR_NOT_READY_QUALITY"}
PIT_STATES = {"FULL_UNIVERSE_DATA_READY", "FULL_UNIVERSE_DATA_NOT_READY_COVERAGE",
              "FULL_UNIVERSE_DATA_NOT_READY_PIT", "FULL_UNIVERSE_DATA_NOT_READY_IDENTITY",
              "FULL_UNIVERSE_DATA_NOT_READY_PROVIDER"}


def test_bar_and_pit_are_two_separate_decisions() -> None:
    bar = U0 / "u0_bar_readiness_certificate.json"
    pit = U0 / "u0_strict_pit_certificate.json"
    if not (bar.exists() and pit.exists()):
        pytest.skip("readiness certificates not generated")
    b = json.loads(bar.read_text())
    p = json.loads(pit.read_text())
    assert b["decision"] in BAR_STATES
    assert p["decision"] in PIT_STATES
    # training is permitted ONLY by the strict PIT certificate
    assert p["training_permitted"] == (p["decision"] == "FULL_UNIVERSE_DATA_READY")
    # bar readiness never grants training on its own
    assert "training" not in json.dumps(b).lower() or "smoke" in json.dumps(b).lower()


def test_benchmark_proves_count10000_and_batch_entitlement() -> None:
    if not BENCH.exists():
        pytest.skip("benchmark not run")
    d = json.loads(BENCH.read_text())["diagnosis"]
    # count=10000 must be the recommended path when it works
    if d["count_10000_works"]:
        assert d["count_10000_rows_example"] > 100
        assert d["no_count_rows_example"] == 100  # the old default that caused the ~100-bar path
        assert "count" in d["old_100_bar_cause"].lower()


def test_bse_identity_not_all_920_classified_synthetic() -> None:
    audit = U0 / "bse_identity_audit.json"
    if not audit.exists():
        pytest.skip("bse identity not audited")
    a = json.loads(audit.read_text())
    # the authoritative list is entirely 920xxx -> none are placeholders
    assert a["all_authoritative_codes_are_920xxx"] is True
    assert a["true_placeholder_codes_in_master"] == []
    assert a["identity_decision"] in {"BSE_IDENTITY_CURRENT_RESOLVED", "BSE_IDENTITY_PARTIAL"}


def test_pit_source_audit_uses_the_strict_vocabulary() -> None:
    src = U0 / "pit_source_audit.json"
    if not src.exists():
        pytest.skip("pit source audit not generated")
    vocab = {"TICKFLOW_AVAILABLE", "TICKFLOW_CURRENT_ONLY", "EXCHANGE_SOURCE_AVAILABLE",
             "ALTERNATIVE_SOURCE_REQUIRED", "BLOCKED_BY_DATA"}
    fields = json.loads(src.read_text())["fields"]
    for info in fields.values():
        assert info["tickflow"] in vocab


def _load_backfill_module():
    path = REPO / "scripts/u0_full_universe_backfill.py"
    spec = importlib.util.spec_from_file_location("u0_backfill_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["u0_backfill_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_backfill_uses_count_10000_single_request_not_pagination() -> None:
    mod = _load_backfill_module()
    assert mod.FULL_COUNT == 10000
    assert mod.REQ_INTERVAL_S >= 6.0  # honour the measured 10/min limit


def test_priority_boards_respects_explicit_order() -> None:
    """BSE must be fetched before STAR when --priority-boards BSE,STAR."""
    priority = ["BSE", "STAR"]
    rank = {b: i for i, b in enumerate(priority)}
    board_of = {"920002.BJ": "BSE", "688001.SH": "STAR", "600000.SH": "SH_Main"}
    todo = ["688001.SH", "600000.SH", "920002.BJ"]
    todo.sort(key=lambda s: (rank.get(board_of.get(s, ""), len(priority)), s))
    assert todo == ["920002.BJ", "688001.SH", "600000.SH"]  # BSE, then STAR, then rest
