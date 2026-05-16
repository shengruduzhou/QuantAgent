"""Guards that V7 realdata paths never silently use mock/synthetic data."""
from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.dataset_builder import V7TrainingDatasetConfig, build_v7_training_dataset_artifact
from quantagent.data.providers.akshare_financial_provider import AkShareFinancialProvider
from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.providers.tushare_financial_provider import TuShareFinancialProvider
from quantagent.data.v7_quality_gates import (
    V7DataQualityGateConfig,
    evaluate_data_quality_gates,
)


def test_akshare_provider_blocks_network_by_default():
    provider = AkShareFinancialProvider(allow_network=False)
    with pytest.raises(ProviderUnavailable):
        provider.income(ProviderRequest("2024-01-01", "2026-01-01", symbols=("600519.SH",)))


def test_tushare_provider_blocks_network_by_default():
    provider = TuShareFinancialProvider(allow_network=False)
    with pytest.raises(ProviderUnavailable):
        provider.balance_sheet(ProviderRequest("2024-01-01", "2026-01-01", symbols=("600519.SH",)))


def test_dataset_builder_rejects_allow_synthetic_fallback(tmp_path):
    config = V7TrainingDatasetConfig(
        market_panel_path=str(tmp_path / "absent_market.csv"),
        labels_path=str(tmp_path / "absent_labels.csv"),
        output_path=str(tmp_path / "out.csv"),
        allow_synthetic_fallback=True,
    )
    with pytest.raises(ValueError, match="synthetic fallback"):
        build_v7_training_dataset_artifact(config)


def test_data_quality_gate_flags_mock_source_column():
    frame = pd.DataFrame(
        [
            {"symbol": "600519.SH", "trade_date": "2026-05-12", "available_at": "2026-05-13", "source": "mock_provider"},
        ]
    )
    report = evaluate_data_quality_gates(
        frame,
        V7DataQualityGateConfig(min_rows=1, min_symbols=1, min_dates=1, require_real_data=True),
    )
    assert not report.passed
    assert "mock_or_synthetic_data_not_production_ready" in report.failures


def test_data_quality_gate_blocks_pit_violations():
    frame = pd.DataFrame(
        [
            {"symbol": "600519.SH", "trade_date": "2026-05-12", "available_at": "2026-05-13", "as_of_date": "2026-05-10"},
        ]
    )
    report = evaluate_data_quality_gates(
        frame,
        V7DataQualityGateConfig(min_rows=1, min_symbols=1, min_dates=1, require_real_data=False),
    )
    assert not report.passed
    assert "pit_violations_present" in report.failures
