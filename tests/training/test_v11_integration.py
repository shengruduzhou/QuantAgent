"""Tests for the Stage 6 v11 feature integration wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.training.v11_integration import (
    V11IntegrationConfig,
    attach_v11_features,
    write_v11_attach_log,
)


def _base_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2024-01-15", periods=10).repeat(3),
            "symbol": ["600000.SH", "600001.SH", "600002.SH"] * 10,
        }
    )


def _write_manifest(path: Path, gate_key: str, gate_open: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"extra": {"coverage_report": {"gate": {gate_key: gate_open, "reason": "ok"}}}}
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Empty / no-data inputs
# ---------------------------------------------------------------------------

def test_empty_panel_returns_empty_result():
    result = attach_v11_features(pd.DataFrame())
    assert result.panel.empty
    assert result.attach_log == []


def test_no_silver_data_skips_every_attach(tmp_path):
    cfg = V11IntegrationConfig(lake_root=tmp_path)
    result = attach_v11_features(_base_panel(), cfg)
    # Pipeline still walks all 8 products
    assert len(result.attach_log) == 8
    # None of them attached
    assert all(not e.attached for e in result.attach_log)
    # All log "silver_missing"
    assert all(e.reason == "silver_missing" for e in result.attach_log)


def test_attach_log_records_attempted_and_columns_added(tmp_path):
    """When sector_map is fully populated + gated open, it attaches and
    adds the sector_level_1 column."""
    cfg = V11IntegrationConfig(lake_root=tmp_path)
    sector = pd.DataFrame(
        [
            {"symbol": s, "sector_level_1": "Bank", "available_at": pd.Timestamp("2024-01-01")}
            for s in ("600000.SH", "600001.SH", "600002.SH")
        ]
    )
    silver_path = tmp_path / "silver" / "sector_map" / "sector_map.parquet"
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    sector.to_parquet(silver_path, index=False)
    _write_manifest(
        tmp_path / "manifests" / "sector_map.json",
        "sector_usable_for_optimization",
        True,
    )

    result = attach_v11_features(_base_panel(), cfg)
    sector_log = next(e for e in result.attach_log if e.product == "sector_map")
    assert sector_log.attached
    assert "sector_level_1" in sector_log.columns_added
    assert "sector_level_1" in result.panel.columns


def test_closed_gate_skips_attach(tmp_path):
    cfg = V11IntegrationConfig(lake_root=tmp_path)
    sector = pd.DataFrame([{"symbol": "600000.SH", "sector_level_1": "Bank", "available_at": pd.Timestamp("2024-01-01")}])
    silver = tmp_path / "silver" / "sector_map" / "sector_map.parquet"
    silver.parent.mkdir(parents=True, exist_ok=True)
    sector.to_parquet(silver, index=False)
    _write_manifest(
        tmp_path / "manifests" / "sector_map.json",
        "sector_usable_for_optimization",
        False,  # gate closed
    )

    result = attach_v11_features(_base_panel(), cfg)
    sector_log = next(e for e in result.attach_log if e.product == "sector_map")
    assert not sector_log.attached
    assert sector_log.reason == "gate_closed"
    assert "sector_level_1" not in result.panel.columns


# ---------------------------------------------------------------------------
# Pipeline ordering
# ---------------------------------------------------------------------------

def test_pipeline_walks_all_eight_products():
    result = attach_v11_features(_base_panel())
    products = [e.product for e in result.attach_log]
    assert products == [
        "sector_map",
        "st_flags",
        "sector_pool",
        "fundamental_ranker",
        "policy_events",
        "bond_flows",
        "state_team_inference",
        "broker_reports",
    ]


# ---------------------------------------------------------------------------
# Sector-dependent products skip when sector_map is missing
# ---------------------------------------------------------------------------

def test_sector_pool_skips_when_panel_lacks_sector(tmp_path):
    """sector_pool attaches by sector_level_1 — if the panel doesn't have
    that column (e.g. sector_map gate closed), the pool attach must skip
    gracefully rather than crash.
    """
    cfg = V11IntegrationConfig(lake_root=tmp_path)
    pool = pd.DataFrame([{"sector_level_1": "Bank", "pool_tier": "core"}])
    silver = tmp_path / "silver" / "sector_pool" / "sector_pool.parquet"
    silver.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(silver, index=False)
    _write_manifest(
        tmp_path / "manifests" / "sector_pool.json",
        "sector_pool_usable_for_overlay",
        True,
    )

    result = attach_v11_features(_base_panel(), cfg)
    pool_log = next(e for e in result.attach_log if e.product == "sector_pool")
    assert not pool_log.attached
    assert pool_log.reason == "panel_missing_sector"


# ---------------------------------------------------------------------------
# Full happy path with multiple products
# ---------------------------------------------------------------------------

def test_full_pipeline_with_sector_and_bond_flows(tmp_path):
    cfg = V11IntegrationConfig(lake_root=tmp_path)
    # sector_map
    sector = pd.DataFrame(
        [
            {"symbol": s, "sector_level_1": "Bank", "available_at": pd.Timestamp("2024-01-01")}
            for s in ("600000.SH", "600001.SH", "600002.SH")
        ]
    )
    (tmp_path / "silver" / "sector_map").mkdir(parents=True, exist_ok=True)
    sector.to_parquet(tmp_path / "silver" / "sector_map" / "sector_map.parquet", index=False)
    _write_manifest(
        tmp_path / "manifests" / "sector_map.json",
        "sector_usable_for_optimization",
        True,
    )
    # bond_flows
    bf = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2024-01-10", periods=20),
            "available_at": pd.bdate_range("2024-01-11", periods=20),
            "yield_1y": [2.20] * 20,
            "yield_5y": [2.55] * 20,
            "yield_10y": [2.80] * 20,
            "yield_3m": [1.95] * 20,
            "yield_aa": [4.10] * 20,
            "yield_aaa": [3.50] * 20,
            "dr007": [1.85] * 20,
            "bond_fund_flow": [5.0] * 20,
        }
    )
    (tmp_path / "silver" / "bond_flows").mkdir(parents=True, exist_ok=True)
    bf.to_parquet(tmp_path / "silver" / "bond_flows" / "bond_flows.parquet", index=False)
    _write_manifest(
        tmp_path / "manifests" / "bond_flows.json",
        "bond_flows_usable_for_features",
        True,
    )

    result = attach_v11_features(_base_panel(), cfg)
    sector_log = next(e for e in result.attach_log if e.product == "sector_map")
    bond_log = next(e for e in result.attach_log if e.product == "bond_flows")
    assert sector_log.attached
    assert bond_log.attached
    assert "sector_level_1" in result.panel.columns
    assert any(c.startswith("bond_") for c in result.panel.columns)


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------

def test_result_to_dict_is_json_serialisable(tmp_path):
    cfg = V11IntegrationConfig(lake_root=tmp_path)
    result = attach_v11_features(_base_panel(), cfg)
    payload = result.to_dict()
    json.dumps(payload)
    assert payload["n_rows"] > 0
    assert payload["features_skipped"]  # every product skipped in this empty-lake case
    assert len(payload["attach_log"]) == 8


def test_writer_emits_attach_log_json(tmp_path):
    cfg = V11IntegrationConfig(lake_root=tmp_path)
    result = attach_v11_features(_base_panel(), cfg)
    path = write_v11_attach_log(result, tmp_path / "reports")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert "attach_log" in payload
    assert len(payload["attach_log"]) == 8


# ---------------------------------------------------------------------------
# Idempotence: re-running adds no duplicate columns
# ---------------------------------------------------------------------------

def test_no_duplicate_columns_when_panel_already_has_sector_level_1(tmp_path):
    """If sector_level_1 already exists on the panel (e.g. from an
    earlier pre-processing step), the integration must not collide.
    """
    cfg = V11IntegrationConfig(lake_root=tmp_path)
    sector = pd.DataFrame(
        [
            {"symbol": s, "sector_level_1": "Bank", "available_at": pd.Timestamp("2024-01-01")}
            for s in ("600000.SH", "600001.SH", "600002.SH")
        ]
    )
    (tmp_path / "silver" / "sector_map").mkdir(parents=True, exist_ok=True)
    sector.to_parquet(tmp_path / "silver" / "sector_map" / "sector_map.parquet", index=False)
    _write_manifest(
        tmp_path / "manifests" / "sector_map.json",
        "sector_usable_for_optimization",
        True,
    )
    panel = _base_panel().assign(sector_level_1="PreFilled")
    result = attach_v11_features(panel, cfg)
    # pandas merge with overlap creates _x / _y suffixes — we'd fail this
    # test if that happened. The clean version: count occurrences.
    count_sector_cols = sum(1 for c in result.panel.columns if c == "sector_level_1")
    assert count_sector_cols == 1
