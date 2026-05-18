"""Tests for the V7 gold-tier training-dataset builder and the CLI smoke path."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from quantagent.cli import app
from quantagent.data.dataset_builder import (
    FORBIDDEN_INFERENCE_COLUMNS,
    V7TrainingDatasetConfig,
    build_v7_training_dataset_artifact,
)
from quantagent.data.v7_label_builder import build_forward_return_labels


def _market_panel(days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    rows: list[dict] = []
    for sidx, symbol in enumerate(("600519.SH", "000858.SZ", "300750.SZ")):
        for didx, date in enumerate(dates):
            close = 10.0 + sidx + didx * (0.05 + sidx * 0.01)
            rows.append(
                {
                    "trade_date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000 + sidx * 100_000,
                    "amount": close * 1_000_000,
                    "available_at": date.strftime("%Y-%m-%d"),
                    "is_suspended": False,
                    "is_st": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                }
            )
    return pd.DataFrame(rows)


def _fundamentals_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "600519.SH", "available_at": "2026-01-30", "revenue": 1000.0, "gross_margin": 0.55},
            {"symbol": "000858.SZ", "available_at": "2026-02-01", "revenue": 800.0, "gross_margin": 0.42},
        ]
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    market = _market_panel()
    labels = build_forward_return_labels(market, horizons=(1, 5)).frame
    fundamentals = _fundamentals_frame()
    market_path = tmp_path / "market.parquet"
    labels_path = tmp_path / "labels.parquet"
    fundamentals_path = tmp_path / "fundamentals.parquet"
    for frame, target in ((market, market_path), (labels, labels_path), (fundamentals, fundamentals_path)):
        try:
            frame.to_parquet(target, index=False)
        except Exception:
            target = target.with_suffix(".csv")
            frame.to_csv(target, index=False)
    return _resolve(market_path), _resolve(labels_path), _resolve(fundamentals_path)


def _resolve(path: Path) -> Path:
    if path.exists():
        return path
    fallback = path.with_suffix(".csv")
    return fallback if fallback.exists() else path


def test_builder_emits_manifest_and_feature_schema(tmp_path):
    market_path, labels_path, fundamentals_path = _write_inputs(tmp_path)
    output = tmp_path / "training_dataset.parquet"
    result = build_v7_training_dataset_artifact(
        V7TrainingDatasetConfig(
            market_panel_path=str(market_path),
            labels_path=str(labels_path),
            output_path=str(output),
            fundamentals_root=str(fundamentals_path),
            horizons=(1, 5),
            min_rows=20,
            min_symbols=2,
            min_dates=10,
        )
    )
    assert result.summary["status"] == "passed"
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["quality_status"] in {"passed", "warning"}
    schema = json.loads(result.feature_schema_path.read_text(encoding="utf-8"))
    for forbidden in FORBIDDEN_INFERENCE_COLUMNS:
        assert forbidden not in schema["feature_columns"], forbidden
    for label_column in schema["label_columns"]:
        assert label_column not in schema["feature_columns"]
    assert "missing_fundamentals" in result.dataset.columns


def test_builder_refuses_synthetic_fallback(tmp_path):
    market_path, labels_path, _ = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="synthetic fallback"):
        build_v7_training_dataset_artifact(
            V7TrainingDatasetConfig(
                market_panel_path=str(market_path),
                labels_path=str(labels_path),
                output_path=str(tmp_path / "training_dataset.parquet"),
                allow_synthetic_fallback=True,
            )
        )


def test_builder_blocks_label_leakage(tmp_path):
    market_path, labels_path, fundamentals_path = _write_inputs(tmp_path)
    output = tmp_path / "training_dataset.parquet"
    result = build_v7_training_dataset_artifact(
        V7TrainingDatasetConfig(
            market_panel_path=str(market_path),
            labels_path=str(labels_path),
            output_path=str(output),
            fundamentals_root=str(fundamentals_path),
            horizons=(1, 5),
            min_rows=20,
            min_symbols=2,
            min_dates=10,
        )
    )
    for label in ("forward_return_1d", "forward_return_5d", "label_end_1d", "label_end_5d"):
        assert label not in result.feature_schema["feature_columns"]
        assert label in result.feature_schema["forbidden_columns"] or label in result.feature_schema["label_columns"]


def test_cli_build_training_dataset_v7_smoke(tmp_path):
    market_path, labels_path, fundamentals_path = _write_inputs(tmp_path)
    output = tmp_path / "gold" / "training_dataset.parquet"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-training-dataset-v7",
            "--market-panel",
            str(market_path),
            "--labels",
            str(labels_path),
            "--fundamentals-root",
            str(fundamentals_path),
            "--output",
            str(output),
            "--horizons",
            "1,5",
            "--min-rows",
            "20",
            "--min-symbols",
            "2",
            "--min-dates",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output.strip().splitlines()[-1])
    assert summary["status"] == "passed"
    assert Path(summary["output_path"]).exists()
    assert Path(summary["manifest_path"]).exists()
    assert Path(summary["feature_schema_path"]).exists()


def test_cli_train_alpha_v7_writes_experiment_manifest(tmp_path):
    market_path, labels_path, fundamentals_path = _write_inputs(tmp_path)
    output = tmp_path / "gold" / "training_dataset.parquet"
    runner = CliRunner()
    build = runner.invoke(
        app,
        [
            "build-training-dataset-v7",
            "--market-panel",
            str(market_path),
            "--labels",
            str(labels_path),
            "--fundamentals-root",
            str(fundamentals_path),
            "--output",
            str(output),
            "--horizons",
            "1,5",
            "--min-rows",
            "20",
            "--min-symbols",
            "2",
            "--min-dates",
            "10",
            # Pin to basic features so this small-fixture test does not
            # exercise alpha181's 60-day rolling factors, which would
            # NaN-out everything on a 30-day synthetic panel.
            "--factor-library",
            "basic",
        ],
    )
    assert build.exit_code == 0, build.output
    dataset_path = Path(json.loads(build.output.strip().splitlines()[-1])["output_path"])
    train = runner.invoke(
        app,
        [
            "train-alpha-v7",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(tmp_path / "alpha"),
            "--min-train-rows",
            "20",
            "--min-train-days",
            "3",
            "--valid-size-days",
            "1",
            "--purge-days",
            "1",
            "--embargo-days",
            "1",
            "--experiment-name",
            "unit_test_run",
            "--registry-root",
            str(tmp_path / "registry"),
        ],
    )
    assert train.exit_code == 0, train.output
    manifest = tmp_path / "alpha" / "experiment_manifest.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "unit_test_run"
    assert payload["horizons"]
    assert (tmp_path / "registry" / "unit_test_run.json").exists()
    assert (tmp_path / "registry" / "latest.json").exists()
