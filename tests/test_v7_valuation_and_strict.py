"""AKShare valuation provider normalisation, valuation bootstrap, strict-mode checks."""
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
    AkShareSectorProvider,
    AkShareUniverseProvider,
    AkShareValuationProvider,
    akshare_universe_schema_report,
    akshare_valuation_schema_report,
)
from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.trading_calendar import TradingCalendar
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


def _valuation_calendar() -> TradingCalendar:
    return TradingCalendar.from_dates(
        ["2026-05-14", "2026-05-15", "2026-05-18", "2026-05-19"]
    )


def test_universe_provider_requires_network():
    with pytest.raises(ProviderUnavailable):
        AkShareUniverseProvider(allow_network=False).list_universe()


def test_valuation_internal_backdate_normalizer_is_explicitly_non_pit():
    raw = pd.DataFrame(
        [
            {
                "代码": "600519",
                "名称": "贵州茅台",
                "市盈率-动态": 30.0,
                "市净率": 9.5,
                "总市值": 2_000_000_000_000,
                "流通市值": 1_900_000_000_000,
            }
        ]
    )
    provider = AkShareValuationProvider(allow_network=False)
    normalised = provider._normalize(raw, "2026-05-15")

    assert normalised["symbol"].tolist() == ["600519.SH"]
    assert normalised["point_in_time_valid"].tolist() == [False]
    assert normalised["available_at"].isna().all()
    assert akshare_valuation_schema_report(normalised)["status"] == "failed"


def test_current_spot_snapshot_cannot_be_backdated():
    request = ProviderRequest("2026-01-01", "2026-05-15", symbols=("600519.SH",))
    with pytest.raises(ProviderUnavailable, match="cannot be backdated"):
        AkShareValuationProvider(allow_network=True).snapshot("2026-05-15", request)


def test_historical_baidu_valuation_uses_source_dates_and_canonical_market_cap(monkeypatch):
    def stock_zh_valuation_baidu(symbol: str, indicator: str, period: str) -> pd.DataFrame:
        assert symbol == "600519"
        assert period in {"近一年", "近三年", "近五年", "近十年", "全部"}
        values = {
            "市盈率(TTM)": [20.0, 21.0],
            "市净率": [7.0, 7.2],
            # Baidu UI/source series is displayed in 亿; provider converts to CNY.
            "总市值": [20000.0, 21000.0],
        }[indicator]
        return pd.DataFrame(
            {
                "date": ["2026-05-14", "2026-05-15"],
                "value": values,
            }
        )

    fake_akshare = types.SimpleNamespace(
        __version__="1.18.84",
        stock_zh_valuation_baidu=stock_zh_valuation_baidu,
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)
    provider = AkShareValuationProvider(
        allow_network=True,
        retry_sleep_seconds=0,
        rate_limit_seconds=0,
        trading_calendar=_valuation_calendar(),
    )
    result = provider.historical(
        ProviderRequest("2026-05-14", "2026-05-15", symbols=("600519.SH",))
    )

    assert result.point_in_time is True
    assert result.metadata["function_name"] == "stock_zh_valuation_baidu"
    assert result.metadata["market_cap_source_unit"] == "1e8_CNY"
    assert result.metadata["market_cap_unit"] == "CNY"
    assert result.frame["market_cap"].tolist() == [2_000_000_000_000.0, 2_100_000_000_000.0]
    assert result.frame["market_cap_raw"].tolist() == [20000.0, 21000.0]
    assert pd.to_datetime(result.frame["available_at"]).dt.strftime("%Y-%m-%d").tolist() == [
        "2026-05-15",
        "2026-05-18",
    ]
    assert result.frame["point_in_time_valid"].tolist() == [True, True]


def test_sector_network_snapshot_cannot_be_backdated():
    with pytest.raises(ProviderUnavailable, match="current membership only"):
        AkShareSectorProvider(allow_network=True).industry_classification(
            as_of_date="2026-05-15"
        )


def test_universe_schema_report_lists_required_columns():
    assert AKSHARE_UNIVERSE_REQUIRED_COLUMNS == ("symbol", "name", "exchange", "list_date")
    report = akshare_universe_schema_report(pd.DataFrame({"symbol": ["600519.SH"]}))
    assert report["status"] == "failed"
    assert "name" in report["missing_columns"]


def test_valuation_bootstrap_uses_explicit_pit_csv_snapshot_and_writes_manifest(tmp_path):
    snapshot = pd.DataFrame(
        [
            {
                "symbol": "600519.SH",
                "trade_date": "2026-05-15",
                "available_at": "2026-05-18",
                "pe_ttm": 30.0,
                "pb": 9.5,
                "market_cap": 2_000_000_000_000,
                "point_in_time_valid": True,
            },
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


def test_valuation_bootstrap_blocks_csv_without_explicit_pit_evidence(tmp_path):
    snapshot = pd.DataFrame(
        [
            {
                "symbol": "600519.SH",
                "trade_date": "2026-05-15",
                "pe_ttm": 30.0,
                "pb": 9.5,
                "market_cap": 2_000_000_000_000,
            }
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
    assert result["status"] == "blocked"
    assert result["output_path"] is None
    assert "valuation_local_snapshot_missing_explicit_pit_evidence" in result["blockers"]


def test_cli_build_valuation_v7(tmp_path):
    snapshot = pd.DataFrame(
        [
            {
                "symbol": "600519.SH",
                "trade_date": "2026-05-15",
                "available_at": "2026-05-18",
                "pe_ttm": 30.0,
                "pb": 9.5,
                "market_cap": 2_000_000_000_000,
                "point_in_time_valid": True,
            }
        ]
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


_ = json
