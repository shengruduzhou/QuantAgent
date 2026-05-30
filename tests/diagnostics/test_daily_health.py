"""Tests for the daily data-layer health checker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantagent.diagnostics.daily_health import (
    FAIL,
    OK,
    WARN,
    DailyHealthChecker,
    DailyHealthConfig,
)


def _write_manifest(path: Path, gate_key: str, gate_open: bool, reason: str = "passed") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "extra": {
            "coverage_report": {
                "gate": {gate_key: gate_open, "reason": reason},
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_config(tmp_path: Path) -> DailyHealthConfig:
    return DailyHealthConfig(lake_root=tmp_path / "lake", output_root=tmp_path / "reports")


# ---------------------------------------------------------------------------
# market_features
# ---------------------------------------------------------------------------

def test_market_features_fail_when_nothing_exists(tmp_path):
    cfg = _make_config(tmp_path)
    report = DailyHealthChecker(cfg).check()
    mf = next(p for p in report.products if p.product == "market_features")
    assert mf.status == FAIL
    assert report.aggregate_status == FAIL


def test_market_features_warn_when_only_parquet_exists(tmp_path):
    cfg = _make_config(tmp_path)
    parquet = Path(cfg.lake_root) / "silver" / "market_panel" / "market_features.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.touch()
    report = DailyHealthChecker(cfg).check()
    mf = next(p for p in report.products if p.product == "market_features")
    assert mf.status == WARN
    assert mf.manifest_found is False


def test_market_features_ok_when_manifest_and_parquet_present(tmp_path):
    cfg = _make_config(tmp_path)
    parquet = Path(cfg.lake_root) / "silver" / "market_panel" / "market_features.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.touch()
    manifest = Path(cfg.lake_root) / "manifests" / "market_features.json"
    _write_manifest(manifest, "market_features_usable", True)
    report = DailyHealthChecker(cfg).check()
    mf = next(p for p in report.products if p.product == "market_features")
    assert mf.status == OK
    assert mf.manifest_found is True


# ---------------------------------------------------------------------------
# Gated products
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("product,gate_key,silver_sub", [
    ("sector_map", "sector_usable_for_optimization", "sector_map/sector_map.parquet"),
    ("st_flags", "st_usable_for_risk_filter", "st_flags/st_flags.parquet"),
    ("sector_pool", "sector_pool_usable_for_overlay", "sector_pool/sector_pool.parquet"),
    ("fundamental_ranker", "fundamental_ranker_usable_for_overlay", "fundamental_ranker/fundamental_ranker.parquet"),
])
def test_gated_product_ok_when_gate_open(tmp_path, product, gate_key, silver_sub):
    cfg = _make_config(tmp_path)
    _write_manifest(Path(cfg.lake_root) / "manifests" / f"{product}.json", gate_key, True)
    report = DailyHealthChecker(cfg).check()
    prod = next(p for p in report.products if p.product == product)
    assert prod.status == OK
    assert prod.gate_open is True


@pytest.mark.parametrize("product,gate_key,silver_sub", [
    ("sector_map", "sector_usable_for_optimization", "sector_map/sector_map.parquet"),
    ("st_flags", "st_usable_for_risk_filter", "st_flags/st_flags.parquet"),
    ("sector_pool", "sector_pool_usable_for_overlay", "sector_pool/sector_pool.parquet"),
    ("fundamental_ranker", "fundamental_ranker_usable_for_overlay", "fundamental_ranker/fundamental_ranker.parquet"),
])
def test_gated_product_fail_when_gate_closed(tmp_path, product, gate_key, silver_sub):
    cfg = _make_config(tmp_path)
    _write_manifest(
        Path(cfg.lake_root) / "manifests" / f"{product}.json",
        gate_key,
        False,
        reason="real_sector_share_below_threshold",
    )
    report = DailyHealthChecker(cfg).check()
    prod = next(p for p in report.products if p.product == product)
    assert prod.status == FAIL
    assert prod.gate_open is False
    assert prod.reason == "real_sector_share_below_threshold"


@pytest.mark.parametrize("product,gate_key,silver_sub", [
    ("sector_map", "sector_usable_for_optimization", "sector_map/sector_map.parquet"),
])
def test_gated_product_warn_when_manifest_missing_but_parquet_exists(tmp_path, product, gate_key, silver_sub):
    cfg = _make_config(tmp_path)
    parquet = Path(cfg.lake_root) / "silver" / silver_sub
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.touch()
    report = DailyHealthChecker(cfg).check()
    prod = next(p for p in report.products if p.product == product)
    assert prod.status == WARN
    assert prod.manifest_found is False


@pytest.mark.parametrize("product,gate_key,silver_sub", [
    ("st_flags", "st_usable_for_risk_filter", "st_flags/st_flags.parquet"),
])
def test_gated_product_fail_when_nothing_exists(tmp_path, product, gate_key, silver_sub):
    cfg = _make_config(tmp_path)
    report = DailyHealthChecker(cfg).check()
    prod = next(p for p in report.products if p.product == product)
    assert prod.status == FAIL


# ---------------------------------------------------------------------------
# Aggregate status
# ---------------------------------------------------------------------------

def test_aggregate_ok_when_all_open(tmp_path):
    cfg = _make_config(tmp_path)
    # market_features: parquet + manifest
    mf_parquet = Path(cfg.lake_root) / "silver" / "market_panel" / "market_features.parquet"
    mf_parquet.parent.mkdir(parents=True, exist_ok=True)
    mf_parquet.touch()
    _write_manifest(Path(cfg.lake_root) / "manifests" / "market_features.json", "market_features_usable", True)
    specs = [
        ("sector_map", "sector_usable_for_optimization"),
        ("st_flags", "st_usable_for_risk_filter"),
        ("sector_pool", "sector_pool_usable_for_overlay"),
        ("fundamental_ranker", "fundamental_ranker_usable_for_overlay"),
        ("policy_events", "policy_events_usable_for_features"),
        ("bond_flows", "bond_flows_usable_for_features"),
        ("state_team_inference", "state_team_inference_usable_for_features"),
        ("broker_reports", "broker_reports_usable_for_features"),
    ]
    for product, gate_key in specs:
        _write_manifest(Path(cfg.lake_root) / "manifests" / f"{product}.json", gate_key, True)
    report = DailyHealthChecker(cfg).check()
    assert report.aggregate_status == OK
    assert report.exit_code == 0


def test_aggregate_fail_when_any_fail(tmp_path):
    cfg = _make_config(tmp_path)
    # One product with a closed gate → FAIL
    _write_manifest(
        Path(cfg.lake_root) / "manifests" / "sector_map.json",
        "sector_usable_for_optimization",
        False,
    )
    report = DailyHealthChecker(cfg).check()
    assert report.aggregate_status == FAIL
    assert report.exit_code == 2


def test_aggregate_warn_promoted_to_fail(tmp_path):
    cfg = _make_config(tmp_path)
    # Only a parquet with no manifest → WARN for market_features, FAIL for the others
    mf_parquet = Path(cfg.lake_root) / "silver" / "market_panel" / "market_features.parquet"
    mf_parquet.parent.mkdir(parents=True, exist_ok=True)
    mf_parquet.touch()
    # Everything else missing → FAIL
    report = DailyHealthChecker(cfg).check()
    assert report.aggregate_status == FAIL


# ---------------------------------------------------------------------------
# write=True emits files
# ---------------------------------------------------------------------------

def test_run_writes_json_and_markdown(tmp_path):
    cfg = _make_config(tmp_path)
    DailyHealthChecker(cfg).run(write=True)
    assert (tmp_path / "reports" / "health_report.json").exists()
    assert (tmp_path / "reports" / "health_report.md").exists()
    payload = json.loads((tmp_path / "reports" / "health_report.json").read_text())
    assert "aggregate_status" in payload
    assert "products" in payload
    # market_features + 8 gated products (Stage 2/4/5)
    assert len(payload["products"]) == 9


def test_run_no_write_does_not_create_files(tmp_path):
    cfg = _make_config(tmp_path)
    DailyHealthChecker(cfg).run(write=False)
    assert not (tmp_path / "reports" / "health_report.json").exists()


# ---------------------------------------------------------------------------
# Markdown format
# ---------------------------------------------------------------------------

def test_markdown_contains_product_table(tmp_path):
    cfg = _make_config(tmp_path)
    report = DailyHealthChecker(cfg).check()
    md = report.to_markdown()
    assert "market_features" in md
    assert "sector_map" in md
    assert "st_flags" in md
    assert "sector_pool" in md
    assert "fundamental_ranker" in md


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def test_cli_health_check_exits_with_fail(tmp_path):
    from typer.testing import CliRunner
    from quantagent.cli import app

    result = CliRunner().invoke(
        app,
        [
            "health-check-v7",
            "--lake-root", str(tmp_path / "lake"),
            "--output-root", str(tmp_path / "reports"),
        ],
    )
    # Empty lake → FAIL → exit code 2
    assert result.exit_code == 2
    assert "market_features" in result.output


def test_cli_health_check_ok_exit_with_all_gates_open(tmp_path):
    from typer.testing import CliRunner
    from quantagent.cli import app

    lake = tmp_path / "lake"
    # market_features
    mf_parquet = lake / "silver" / "market_panel" / "market_features.parquet"
    mf_parquet.parent.mkdir(parents=True, exist_ok=True)
    mf_parquet.touch()
    _write_manifest(lake / "manifests" / "market_features.json", "market_features_usable", True)
    specs = [
        ("sector_map", "sector_usable_for_optimization"),
        ("st_flags", "st_usable_for_risk_filter"),
        ("sector_pool", "sector_pool_usable_for_overlay"),
        ("fundamental_ranker", "fundamental_ranker_usable_for_overlay"),
        ("policy_events", "policy_events_usable_for_features"),
        ("bond_flows", "bond_flows_usable_for_features"),
        ("state_team_inference", "state_team_inference_usable_for_features"),
        ("broker_reports", "broker_reports_usable_for_features"),
    ]
    for product, gate_key in specs:
        _write_manifest(lake / "manifests" / f"{product}.json", gate_key, True)

    result = CliRunner().invoke(
        app,
        [
            "health-check-v7",
            "--lake-root", str(lake),
            "--output-root", str(tmp_path / "reports"),
        ],
    )
    assert result.exit_code == 0
    assert "OK" in result.output
