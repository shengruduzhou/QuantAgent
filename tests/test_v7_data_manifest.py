"""Unit tests for the V7 DataManifest model and lake layout helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantagent.data.lake import manifest_path, v7_lake_paths
from quantagent.data.manifest import (
    DataManifest,
    build_manifest_for_frame,
    hash_file,
    hash_frame,
)


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "600519.SH", "trade_date": "2026-05-12", "available_at": "2026-05-13", "value": 1.0},
            {"symbol": "600519.SH", "trade_date": "2026-05-13", "available_at": "2026-05-14", "value": 1.2},
            {"symbol": "600519.SH", "trade_date": "2026-05-13", "available_at": "2026-05-14", "value": 1.2},  # duplicate
        ]
    )


def test_lake_paths_create_and_index_manifests(tmp_path):
    lake = v7_lake_paths(tmp_path).ensure()
    for path in (
        lake.raw_qlib,
        lake.raw_akshare,
        lake.silver_market_panel,
        lake.silver_fundamentals,
        lake.gold_training_dataset,
        lake.manifests,
    ):
        assert path.exists()
    assert manifest_path("market_panel", tmp_path) == lake.manifests / "market_panel.json"


def test_hash_file_and_hash_frame_are_deterministic(tmp_path):
    frame = _toy_frame()
    file_path = tmp_path / "frame.csv"
    frame.to_csv(file_path, index=False)
    h1 = hash_file(file_path)
    h2 = hash_file(file_path)
    assert h1 == h2 != ""
    assert hash_file(tmp_path / "missing.csv") == ""
    assert hash_frame(frame) == hash_frame(frame.copy())
    assert hash_frame(pd.DataFrame()) == ""


def test_build_manifest_for_frame_reports_status_and_duplicates(tmp_path):
    output_path = tmp_path / "frame.csv"
    frame = _toy_frame()
    frame.to_csv(output_path, index=False)
    manifest = build_manifest_for_frame(
        dataset_name="toy",
        vendor="unit_test",
        frame=frame,
        output_paths=[output_path],
        required_columns=("symbol", "trade_date", "available_at", "value"),
        pit_violation_count=0,
        warnings=(),
    )
    assert manifest.quality_status == "passed"
    assert manifest.duplicate_row_count == 1
    written = manifest.write(tmp_path / "manifest.json")
    payload = json.loads(Path(written).read_text(encoding="utf-8"))
    assert payload["row_count"] == 3
    assert payload["duplicate_row_count"] == 1
    assert payload["content_hashes"][str(output_path)] == hash_file(output_path)


def test_manifest_status_failed_when_required_columns_missing(tmp_path):
    frame = _toy_frame()[["symbol", "trade_date"]]
    manifest = build_manifest_for_frame(
        dataset_name="toy_missing",
        vendor="unit_test",
        frame=frame,
        output_paths=[tmp_path / "missing.csv"],
        required_columns=("symbol", "trade_date", "available_at"),
    )
    assert manifest.quality_status == "failed"
    assert "available_at" in manifest.missing_columns


def test_manifest_status_warning_when_warnings_present():
    manifest = DataManifest(
        dataset_name="toy_warn",
        vendor="unit_test",
        fetch_time="2026-05-16T00:00:00Z",
        row_count=10,
        warnings=("late_arriving_rows",),
    )
    assert manifest.warnings == ("late_arriving_rows",)
