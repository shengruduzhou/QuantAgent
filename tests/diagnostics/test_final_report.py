"""Tests for the Stage 6 final report aggregator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantagent.diagnostics.final_report import (
    FinalReportConfig,
    build_final_report,
    collect_data_product_gates,
    derive_stage_verdicts,
    load_integration_audit,
    load_latest_replay,
    write_final_report,
)


def _write_manifest(path: Path, gate_key: str, gate_open: bool, n_rows: int = 100, reason: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "rows": n_rows,
                "extra": {"coverage_report": {"gate": {gate_key: gate_open, "reason": reason}}},
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Data-product gates
# ---------------------------------------------------------------------------

def test_collect_data_product_gates_finds_open_and_closed(tmp_path):
    cfg = FinalReportConfig(
        lake_root=tmp_path / "lake",
        reports_root=tmp_path / "reports",
        models_root=tmp_path / "models",
    )
    # sector_map open
    _write_manifest(
        Path(cfg.lake_root) / "manifests" / "sector_map.json",
        "sector_usable_for_optimization",
        True,
        n_rows=3000,
    )
    # st_flags closed
    _write_manifest(
        Path(cfg.lake_root) / "manifests" / "st_flags.json",
        "st_usable_for_risk_filter",
        False,
        n_rows=500,
        reason="coverage_below_threshold",
    )

    rows = collect_data_product_gates(cfg)
    by_product = {r["product"]: r for r in rows}
    assert by_product["sector_map"]["gate_open"] is True
    assert by_product["sector_map"]["n_rows"] == 3000
    assert by_product["st_flags"]["gate_open"] is False
    assert by_product["st_flags"]["reason"] == "coverage_below_threshold"
    # No manifest for the rest
    assert by_product["policy_events"]["manifest_found"] is False


def test_collect_data_product_gates_covers_all_eight_products(tmp_path):
    cfg = FinalReportConfig(lake_root=tmp_path / "lake")
    rows = collect_data_product_gates(cfg)
    assert len(rows) == 8


# ---------------------------------------------------------------------------
# Replay loading
# ---------------------------------------------------------------------------

def _write_replay(path: Path, per_fold: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"per_fold": per_fold}), encoding="utf-8")


def test_load_latest_replay_picks_first_candidate(tmp_path):
    cfg = FinalReportConfig(reports_root=tmp_path / "reports")
    # Write a v11 replay (first in candidate order)
    _write_replay(
        Path(cfg.reports_root) / "sleeve_replay_v11" / "replay_summary.json",
        [
            {"fold": "f0", "status": "ok", "excess_ann_%": 20.0, "max_DD_%": -5.0, "sharpe": 2.0},
            {"fold": "f1", "status": "ok", "excess_ann_%": 18.0, "max_DD_%": -7.0, "sharpe": 1.8},
            {"fold": "f2", "status": "ok", "excess_ann_%": 22.0, "max_DD_%": -6.0, "sharpe": 2.1},
        ],
    )
    summary = load_latest_replay(cfg)
    assert summary["source_dir"] == "sleeve_replay_v11"
    assert summary["n_folds"] == 3
    assert summary["excess_mean_pct"] == pytest.approx(20.0)
    assert summary["max_dd_worst_pct"] == pytest.approx(-7.0)


def test_load_latest_replay_falls_back_to_v10(tmp_path):
    cfg = FinalReportConfig(reports_root=tmp_path / "reports")
    _write_replay(
        Path(cfg.reports_root) / "sleeve_replay_v10" / "replay_summary.json",
        [
            {"fold": "f0", "status": "ok", "excess_ann_%": 25.0, "max_DD_%": -8.0, "sharpe": 1.9},
        ],
    )
    summary = load_latest_replay(cfg)
    assert summary["source_dir"] == "sleeve_replay_v10"
    assert summary["n_folds"] == 1


def test_load_latest_replay_returns_none_when_nothing_found(tmp_path):
    cfg = FinalReportConfig(reports_root=tmp_path / "reports_empty")
    assert load_latest_replay(cfg) is None


def test_load_latest_replay_ignores_status_error(tmp_path):
    cfg = FinalReportConfig(reports_root=tmp_path / "reports")
    _write_replay(
        Path(cfg.reports_root) / "sleeve_replay_v11" / "replay_summary.json",
        [
            {"fold": "f0", "status": "error", "excess_ann_%": 999.0, "max_DD_%": -50.0, "sharpe": 99.0},
            {"fold": "f1", "status": "ok", "excess_ann_%": 15.0, "max_DD_%": -5.0, "sharpe": 1.5},
        ],
    )
    summary = load_latest_replay(cfg)
    assert summary["n_folds"] == 1  # only the ok one counted
    assert summary["excess_mean_pct"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Integration audit
# ---------------------------------------------------------------------------

def test_load_integration_audit_returns_log(tmp_path):
    cfg = FinalReportConfig(models_root=tmp_path / "models")
    audit_path = Path(cfg.models_root) / "v7_alpha_v11" / "integration_audit" / "v11_attach_log.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({"n_rows": 1000, "features_attached": ["sector_map"], "features_skipped": []}))
    audit = load_integration_audit(cfg)
    assert audit["n_rows"] == 1000
    assert audit["features_attached"] == ["sector_map"]


def test_load_integration_audit_returns_none_when_missing(tmp_path):
    cfg = FinalReportConfig(models_root=tmp_path / "models_empty")
    assert load_integration_audit(cfg) is None


# ---------------------------------------------------------------------------
# Stage verdicts
# ---------------------------------------------------------------------------

def test_stage_verdicts_pass_with_strong_replay():
    replay = {
        "n_folds": 12,
        "excess_mean_pct": 22.0,
        "excess_min_pct": 5.0,
        "excess_max_pct": 50.0,
        "max_dd_worst_pct": -7.5,
        "max_dd_mean_pct": -5.0,
        "per_fold": [],
        "source_dir": "x",
    }
    verdicts = derive_stage_verdicts(replay, data_gates=[])
    by_stage = {v.stage: v for v in verdicts}
    # Stage 1 / 2 / 3 should all pass (excess 22 > {12,14,15}, DD 7.5 ≤ {9,9,8})
    assert by_stage["1"].status == "pass"
    assert by_stage["2"].status == "pass"
    assert by_stage["3"].status == "pass"
    # Stage 5: excess 22 > 16, DD 7.5 ≤ 8 → pass
    assert by_stage["5"].status == "pass"


def test_stage_verdicts_fail_dd_breach():
    replay = {
        "n_folds": 12,
        "excess_mean_pct": 22.0,
        "excess_min_pct": 5.0,
        "excess_max_pct": 50.0,
        "max_dd_worst_pct": -9.5,
        "max_dd_mean_pct": -6.0,
        "per_fold": [],
        "source_dir": "x",
    }
    verdicts = derive_stage_verdicts(replay, data_gates=[])
    by_stage = {v.stage: v for v in verdicts}
    # Stage 3 should fail (DD breach: 9.5 > 8)
    assert by_stage["3"].status == "fail"
    # Stage 2 still passes (DD 9.5 > 9 → fail too)
    assert by_stage["2"].status == "fail"


def test_stage_4_verdict_deferred_when_policy_gate_closed():
    replay = None
    gates = [{"product": "policy_events", "gate_open": False, "stage": "4.1", "gate_key": "x",
              "manifest_found": True, "reason": "no_events", "n_rows": 0}]
    verdicts = derive_stage_verdicts(replay, gates)
    s4 = next(v for v in verdicts if v.stage == "4")
    assert s4.status == "deferred"


def test_stage_verdicts_all_deferred_when_no_replay():
    verdicts = derive_stage_verdicts(None, data_gates=[])
    assert all(v.status == "deferred" for v in verdicts)


# ---------------------------------------------------------------------------
# End-to-end build + write
# ---------------------------------------------------------------------------

def test_build_final_report_minimal(tmp_path):
    cfg = FinalReportConfig(
        lake_root=tmp_path / "lake",
        reports_root=tmp_path / "reports",
        models_root=tmp_path / "models",
    )
    report = build_final_report(cfg)
    # All deferred (no replay, no manifests)
    assert all(v.status == "deferred" for v in report.verdicts)
    assert report.replay is None
    assert report.integration_audit is None


def test_build_final_report_with_replay_and_gates(tmp_path):
    cfg = FinalReportConfig(
        lake_root=tmp_path / "lake",
        reports_root=tmp_path / "reports",
        models_root=tmp_path / "models",
    )
    # Build replay
    _write_replay(
        Path(cfg.reports_root) / "sleeve_replay_v11" / "replay_summary.json",
        [{"fold": f"f{i}", "status": "ok", "excess_ann_%": 25.0, "max_DD_%": -5.0, "sharpe": 2.0}
         for i in range(12)],
    )
    # Open the sector_map and policy_events gates
    _write_manifest(
        Path(cfg.lake_root) / "manifests" / "sector_map.json",
        "sector_usable_for_optimization",
        True,
    )
    _write_manifest(
        Path(cfg.lake_root) / "manifests" / "policy_events.json",
        "policy_events_usable_for_features",
        True,
    )
    report = build_final_report(cfg)
    by_stage = {v.stage: v for v in report.verdicts}
    assert by_stage["3"].status == "pass"
    open_products = [g for g in report.data_gates if g["gate_open"]]
    assert {g["product"] for g in open_products} == {"sector_map", "policy_events"}


def test_write_final_report_emits_md_and_json(tmp_path):
    cfg = FinalReportConfig(
        lake_root=tmp_path / "lake",
        reports_root=tmp_path / "reports",
        models_root=tmp_path / "models",
    )
    report = build_final_report(cfg)
    paths = write_final_report(report, tmp_path / "out")
    md_path = Path(paths["markdown"])
    json_path = Path(paths["json"])
    assert md_path.exists()
    assert json_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "QuantAgent V7 — Final Report" in md
    assert "Stage gate verdicts" in md
    assert "Data layer manifest gates" in md


def test_markdown_includes_per_fold_worst_excerpt(tmp_path):
    cfg = FinalReportConfig(
        lake_root=tmp_path / "lake",
        reports_root=tmp_path / "reports",
        models_root=tmp_path / "models",
    )
    _write_replay(
        Path(cfg.reports_root) / "sleeve_replay_v10" / "replay_summary.json",
        [
            {"fold": "f0", "status": "ok", "excess_ann_%": -10.0, "max_DD_%": -9.0, "sharpe": -0.5},
            {"fold": "f1", "status": "ok", "excess_ann_%": 30.0, "max_DD_%": -3.0, "sharpe": 2.5},
        ],
    )
    report = build_final_report(cfg)
    md = report.to_markdown()
    # Worst fold (f0) must show up in the excerpt section
    assert "f0" in md
    assert "Per-fold excerpt" in md


def test_report_to_dict_is_serialisable(tmp_path):
    cfg = FinalReportConfig(lake_root=tmp_path / "lake")
    report = build_final_report(cfg)
    json.dumps(report.to_dict(), default=str)
