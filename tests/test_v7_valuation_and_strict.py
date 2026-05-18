"""AkShare valuation provider normalisation, valuation bootstrap, strict-mode checks."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pandas as pd
import pytest
from typer.testing import CliRunner

from quantagent.cli import app
from quantagent.data.bootstrap.valuation_bootstrap import ValuationBootstrapConfig, build_valuation_cache
from quantagent.data.dataset_builder import V7TrainingDatasetConfig, build_v7_training_dataset_artifact
from quantagent.data.providers.akshare_valuation_provider import (
    AKSHARE_UNIVERSE_REQUIRED_COLUMNS,
    AKSHARE_VALUATION_REQUIRED_COLUMNS,
    AkShareUniverseProvider,
    AkShareValuationProvider,
    akshare_universe_schema_report,
    akshare_valuation_schema_report,
)
from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.v7_label_builder import build_forward_return_labels


def _market_panel(days: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    rows = []
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


def test_universe_provider_requires_network():
    with pytest.raises(ProviderUnavailable):
        AkShareUniverseProvider(allow_network=False).list_universe()


def test_valuation_provider_normalises_chinese_columns():
    raw = pd.DataFrame(
        [
            {"代码": "600519", "名称": "贵州茅台", "市盈率-动态": 30.0, "市净率": 9.5, "总市值": 2_000_000_000_000, "流通市值": 1_900_000_000_000},
            {"代码": "000858", "名称": "五粮液", "市盈率-动态": 18.0, "市净率": 6.0, "总市值": 800_000_000_000, "流通市值": 600_000_000_000},
        ]
    )
    provider = AkShareValuationProvider(allow_network=False)
    normalised = provider._normalize(raw, "2026-05-15")
    report = akshare_valuation_schema_report(normalised)
    assert report["status"] == "passed"
    assert {"symbol", "trade_date", "available_at", "pe_ttm", "pb", "market_cap"}.issubset(normalised.columns)
    assert normalised["symbol"].tolist() == ["600519.SH", "000858.SZ"]


def test_valuation_provider_falls_back_to_symbol_endpoints(monkeypatch):
    def stock_zh_a_spot_em() -> pd.DataFrame:
        raise ConnectionError("spot disconnected")

    def stock_individual_info_em(symbol: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"item": "股票代码", "value": symbol},
                {"item": "股票简称", "value": "贵州茅台"},
                {"item": "总市值", "value": 2_000_000_000_000},
                {"item": "流通市值", "value": 1_900_000_000_000},
            ]
        )

    def stock_zh_valuation_comparison_em(symbol: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "代码": "600519",
                    "简称": "贵州茅台",
                    "市盈率-TTM": 24.5,
                    "市净率-MRQ": 8.5,
                    "市销率-TTM": 12.0,
                    "PEG": 1.2,
                    "EV/EBITDA-24A": 18.0,
                }
            ]
        )

    fake_akshare = types.SimpleNamespace(
        stock_zh_a_spot_em=stock_zh_a_spot_em,
        stock_individual_info_em=stock_individual_info_em,
        stock_zh_valuation_comparison_em=stock_zh_valuation_comparison_em,
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)

    request = ProviderRequest("2026-01-01", "2026-05-15", symbols=("600519.SH",))
    result = AkShareValuationProvider(allow_network=True, retry_sleep_seconds=0).snapshot("2026-05-15", request)

    assert result.metadata["schema_report"]["status"] == "passed"
    assert result.frame[["symbol", "pe_ttm", "pb", "market_cap"]].iloc[0].to_dict() == {
        "symbol": "600519.SH",
        "pe_ttm": 24.5,
        "pb": 8.5,
        "market_cap": 2_000_000_000_000,
    }
    assert any("akshare_valuation_spot_em_failed" in warning for warning in result.warnings)


def test_universe_schema_report_lists_required_columns():
    assert AKSHARE_UNIVERSE_REQUIRED_COLUMNS == ("symbol", "name", "exchange", "list_date")
    report = akshare_universe_schema_report(pd.DataFrame({"symbol": ["600519.SH"]}))
    assert report["status"] == "failed"
    assert "name" in report["missing_columns"]


def test_valuation_bootstrap_uses_csv_snapshot_and_writes_manifest(tmp_path):
    snapshot = pd.DataFrame(
        [
            {"symbol": "600519.SH", "trade_date": "2026-05-15", "available_at": "2026-05-15", "pe_ttm": 30.0, "pb": 9.5, "market_cap": 2_000_000_000_000},
        ]
    )
    csv_path = tmp_path / "valuation_snapshot.csv"
    snapshot.to_csv(csv_path, index=False)
    result = build_valuation_cache(
        ValuationBootstrapConfig(
            as_of_dates=(),
            lake_root=str(tmp_path / "lake"),
            csv_snapshot=str(csv_path),
        )
    )
    assert result["status"] == "passed"
    assert Path(result["output_path"]).exists()
    assert Path(result["manifest_path"]).exists()


def test_cli_build_valuation_v7(tmp_path):
    snapshot = pd.DataFrame(
        [{"symbol": "600519.SH", "trade_date": "2026-05-15", "available_at": "2026-05-15", "pe_ttm": 30.0, "pb": 9.5, "market_cap": 2_000_000_000_000}]
    )
    csv_path = tmp_path / "snapshot.csv"
    snapshot.to_csv(csv_path, index=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-valuation-v7",
            "--csv-snapshot",
            str(csv_path),
            "--lake-root",
            str(tmp_path / "lake"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_training_dataset_strict_mode_rejects_overlapping_splits(tmp_path):
    market = _market_panel(days=20)
    labels = build_forward_return_labels(market, horizons=(1, 5)).frame
    market_path = tmp_path / "market.parquet"
    labels_path = tmp_path / "labels.parquet"
    for frame, target in ((market, market_path), (labels, labels_path)):
        try:
            frame.to_parquet(target, index=False)
        except Exception:
            target = target.with_suffix(".csv")
            frame.to_csv(target, index=False)
    market_path = market_path if market_path.exists() else market_path.with_suffix(".csv")
    labels_path = labels_path if labels_path.exists() else labels_path.with_suffix(".csv")
    config = V7TrainingDatasetConfig(
        market_panel_path=str(market_path),
        labels_path=str(labels_path),
        output_path=str(tmp_path / "training.parquet"),
        horizons=(1, 5),
        min_rows=20,
        min_symbols=2,
        min_dates=5,
        train_end_date="2026-01-20",
        validation_end_date="2026-01-10",
    )
    with pytest.raises(ValueError, match="validation_end_date must be strictly after train_end_date"):
        build_v7_training_dataset_artifact(config)


def test_training_dataset_strict_mode_rejects_duplicate_rows(tmp_path):
    market = _market_panel(days=20)
    duplicated = pd.concat([market, market.iloc[:1]], ignore_index=True)
    market_path = tmp_path / "market.csv"
    duplicated.to_csv(market_path, index=False)
    labels = build_forward_return_labels(duplicated, horizons=(1,)).frame
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)
    config = V7TrainingDatasetConfig(
        market_panel_path=str(market_path),
        labels_path=str(labels_path),
        output_path=str(tmp_path / "training.csv"),
        horizons=(1,),
        min_rows=10,
        min_symbols=2,
        min_dates=5,
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_v7_training_dataset_artifact(config)


def test_training_dataset_strict_mode_blocks_synthetic_source(tmp_path):
    market = _market_panel(days=20)
    market["source"] = "mock_provider"
    market_path = tmp_path / "market.csv"
    market.to_csv(market_path, index=False)
    labels = build_forward_return_labels(market, horizons=(1,)).frame
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)
    config = V7TrainingDatasetConfig(
        market_panel_path=str(market_path),
        labels_path=str(labels_path),
        output_path=str(tmp_path / "training.csv"),
        horizons=(1,),
        min_rows=10,
        min_symbols=2,
        min_dates=5,
        source_name="realdata",
    )
    with pytest.raises(ValueError):
        build_v7_training_dataset_artifact(config)


def test_cli_lists_new_v7_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for command in (
        "build-market-panel-v7",
        "build-akshare-v7",
        "build-valuation-v7",
        "build-training-dataset-v7",
        "train-alpha-v7",
        "train-deep-alpha-v7",
        "evaluate-alpha-v7",
        "v7-live-readiness-report",
        "run-real-training-v7",
    ):
        assert command in result.output, command


# silence "imported but unused" for the JSON guard used in the strict-mode test loop
_ = json
