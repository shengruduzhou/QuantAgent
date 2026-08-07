from __future__ import annotations

import pandas as pd
import pytest

from quantagent.qlib.catalog import (
    QLIB_CAPABILITIES,
    QLIB_DOC_COUNT,
    QLIB_MIN_VERSION,
)
from quantagent.qlib.docs_audit import build_coverage_audit
from quantagent.qlib.parquet import build_qlib_static_frame
from quantagent.qlib.runtime import _version_tuple
from quantagent.qlib.workflow import QlibSegments, build_static_parquet_task


EXPECTED_IDS = {
    "introduction",
    "quickstart",
    "installation",
    "initialization",
    "data_retrieval",
    "custom_model_integration",
    "workflow",
    "data_layer",
    "model",
    "strategy_backtest",
    "highfreq_nested_decision",
    "meta_controller",
    "recorder",
    "analysis_report",
    "online",
    "reinforcement_learning",
    "formulaic_alpha",
    "online_offline_server",
    "serialization",
    "task_management",
    "pit_database",
    "code_standard",
    "development_guidance",
    "build_image",
    "api_reference",
    "faq",
    "changelog",
}


def test_full_reference_registry_is_exact_and_versioned():
    assert QLIB_DOC_COUNT == 27
    assert len(QLIB_CAPABILITIES) == 27
    assert {item.capability_id for item in QLIB_CAPABILITIES} == EXPECTED_IDS
    assert QLIB_MIN_VERSION == "0.9.7"
    assert all(item.url.startswith("https://qlib.org.cn/en/latest/") for item in QLIB_CAPABILITIES)


def test_static_audit_exposes_every_reference():
    audit = build_coverage_audit()
    assert audit["status"] == "passed"
    assert audit["expected_reference_count"] == 27
    assert audit["registered_reference_count"] == 27
    assert len(audit["references"]) == 27


def test_version_parser_handles_release_versions():
    assert _version_tuple("0.9.7") == (0, 9, 7)
    assert _version_tuple("0.9.7rc1") == (0, 9, 7)
    assert _version_tuple("1.0") == (1, 0, 0)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ"],
            "trade_date": ["2026-01-05", "2026-01-05"],
            "available_at": ["2026-01-06", "2026-01-06"],
            "quality": [0.8, -0.2],
            "momentum": [0.3, 0.1],
            "forward_return_20d": [0.05, -0.03],
        }
    )


def test_parquet_bridge_uses_available_at_and_separates_labels():
    result = build_qlib_static_frame(
        _sample_frame(),
        feature_columns=["quality", "momentum"],
        label_columns=["forward_return_20d"],
    )
    assert result.index.names == ["datetime", "instrument"]
    assert list(result.columns) == [
        ("feature", "quality"),
        ("feature", "momentum"),
        ("label", "forward_return_20d"),
    ]
    assert set(result.index.get_level_values("instrument")) == {"SH600519", "SZ000001"}
    assert set(result.index.get_level_values("datetime")) == {pd.Timestamp("2026-01-06")}


def test_parquet_bridge_blocks_future_column_as_feature():
    with pytest.raises(ValueError, match="label/future-like"):
        build_qlib_static_frame(
            _sample_frame(),
            feature_columns=["quality", "forward_return_20d"],
        )


def test_parquet_bridge_rejects_duplicate_point_in_time_rows():
    frame = pd.concat([_sample_frame().iloc[[0]], _sample_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        build_qlib_static_frame(frame, feature_columns=["quality"])


def test_segments_enforce_nonoverlap_and_optional_gap():
    segments = QlibSegments(
        train=("2020-01-01", "2022-12-31"),
        valid=("2023-02-01", "2023-12-31"),
        test=("2024-02-01", "2024-12-31"),
    )
    segments.validate(minimum_gap_days=30)
    with pytest.raises(ValueError, match="purge/embargo"):
        segments.validate(minimum_gap_days=32)


def test_workflow_builder_uses_static_loader_and_explicit_benchmark():
    segments = QlibSegments(
        train=("2020-01-01", "2022-12-31"),
        valid=("2023-02-01", "2023-12-31"),
        test=("2024-02-01", "2024-12-31"),
    )
    model = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {"loss": "mse"},
    }
    task = build_static_parquet_task(
        parquet_path="data/qlib/features.parquet",
        model_config=model,
        segments=segments,
        benchmark_symbol="000300.SH",
        minimum_gap_days=30,
    )
    handler = task["dataset"]["kwargs"]["handler"]
    loader = handler["kwargs"]["data_loader"]
    assert handler["class"] == "DataHandlerLP"
    assert loader["class"] == "StaticDataLoader"
    assert loader["kwargs"]["config"].endswith(".parquet")
    port_record = task["record"][-1]
    cfg = port_record["kwargs"]["config"]
    assert cfg["strategy"]["class"] == "TopkDropoutStrategy"
    assert cfg["backtest"]["benchmark"] == "SH000300"
    assert cfg["backtest"]["start_time"] == segments.test[0]


def test_workflow_builder_never_guesses_benchmark():
    segments = QlibSegments(
        train=("2020-01-01", "2022-12-31"),
        valid=("2023-02-01", "2023-12-31"),
        test=("2024-02-01", "2024-12-31"),
    )
    with pytest.raises(ValueError, match="benchmark_symbol"):
        build_static_parquet_task(
            parquet_path="features.parquet",
            model_config={"class": "AnyModel"},
            segments=segments,
            benchmark_symbol="",
        )


def test_processor_fit_window_cannot_cross_train_boundary():
    segments = QlibSegments(
        train=("2020-01-01", "2022-12-31"),
        valid=("2023-02-01", "2023-12-31"),
        test=("2024-02-01", "2024-12-31"),
    )
    with pytest.raises(ValueError, match="exceeds"):
        build_static_parquet_task(
            parquet_path="features.parquet",
            model_config={"class": "AnyModel"},
            segments=segments,
            benchmark_symbol="000300.SH",
            infer_processors=[
                {
                    "class": "ZScoreNorm",
                    "kwargs": {
                        "fit_start_time": "2020-01-01",
                        "fit_end_time": "2023-01-15",
                    },
                }
            ],
        )
