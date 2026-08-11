from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.providers.st_pit import (
    HistoricalSTCoverageError,
    attach_historical_st,
    load_historical_st_evidence,
)
from quantagent.data.providers.tickflow_provider import TickflowProvider


def _raw_main_board() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["600001.SH"] * 3,
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "open": [10.0, 10.0, 10.5],
            "high": [10.0, 10.5, 11.03],
            "low": [10.0, 10.0, 10.5],
            "close": [10.0, 10.5, 11.03],
            "volume": [1_000_000, 1_000_000, 1_000_000],
            "amount": [10_000_000, 10_500_000, 11_030_000],
        }
    )


def _write_daily_evidence(path) -> None:
    pd.DataFrame(
        {
            "symbol": ["600001.SH"] * 3,
            "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "is_st": [False, True, False],
            "available_at": [
                "2024-01-01T18:00:00+08:00",
                "2024-01-02T18:00:00+08:00",
                "2024-01-03T18:00:00+08:00",
            ],
            "point_in_time_valid": [True, True, True],
        }
    ).to_csv(path, index=False)


def test_daily_pit_state_enters_and_exits_st_without_current_snapshot_smear(tmp_path, monkeypatch) -> None:
    evidence_path = tmp_path / "historical_st.csv"
    _write_daily_evidence(evidence_path)
    monkeypatch.setenv("TICKFLOW_API_KEY", "fake")
    provider = TickflowProvider(
        allow_network=True,
        historical_st_path=str(evidence_path),
    )
    monkeypatch.setattr(provider, "_call_tickflow_daily", lambda request: _raw_main_board())

    result = provider.tradability(
        ProviderRequest(
            symbols=("600001.SH",),
            start_date="2024-01-02",
            end_date="2024-01-04",
        )
    )

    assert result.point_in_time is True
    assert result.metadata["st_coverage_status"] == "historical_pit"
    assert result.frame["is_st"].tolist() == [False, True, False]
    # 2024-01-03 is historical ST: 10.00 * 1.05 = 10.50 -> limit-up.
    assert bool(result.frame.loc[1, "is_limit_up"]) is True
    # 2024-01-04 is no longer ST: 10.50 * 1.10 = 11.55; 11.03 is not limit-up.
    assert bool(result.frame.loc[2, "is_limit_up"]) is False
    assert result.frame["point_in_time_valid"].all()
    assert result.frame["is_st_provenance"].eq("dated_pit_st_evidence").all()


def test_missing_daily_pit_row_fails_closed_instead_of_defaulting_non_st(tmp_path, monkeypatch) -> None:
    path = tmp_path / "incomplete.csv"
    pd.DataFrame(
        {
            "symbol": ["600001.SH", "600001.SH"],
            "trade_date": ["2024-01-02", "2024-01-04"],
            "is_st": [False, False],
            "available_at": [
                "2024-01-01T18:00:00+08:00",
                "2024-01-03T18:00:00+08:00",
            ],
        }
    ).to_csv(path, index=False)
    monkeypatch.setenv("TICKFLOW_API_KEY", "fake")
    provider = TickflowProvider(allow_network=True, historical_st_path=str(path))
    monkeypatch.setattr(provider, "_call_tickflow_daily", lambda request: _raw_main_board())

    with pytest.raises(ProviderUnavailable, match="does not fully cover"):
        provider.tradability(
            ProviderRequest(
                symbols=("600001.SH",),
                start_date="2024-01-02",
                end_date="2024-01-04",
            )
        )


def test_future_available_st_status_is_rejected(tmp_path) -> None:
    path = tmp_path / "future.csv"
    pd.DataFrame(
        {
            "symbol": ["600001.SH"],
            "trade_date": ["2024-01-02"],
            "is_st": [True],
            "available_at": ["2024-01-02T10:00:00+08:00"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(HistoricalSTCoverageError, match="09:25"):
        load_historical_st_evidence(path)


def test_interval_evidence_requires_explicit_non_st_periods_and_maps_effective_state(tmp_path) -> None:
    path = tmp_path / "intervals.csv"
    pd.DataFrame(
        {
            "symbol": ["600001.SH", "600001.SH", "600001.SH"],
            "start_date": ["2024-01-01", "2024-01-03", "2024-01-04"],
            "end_date": ["2024-01-02", "2024-01-03", "2024-01-05"],
            "is_st": [False, True, False],
            "available_at": [
                "2023-12-29T18:00:00+08:00",
                "2024-01-02T18:00:00+08:00",
                "2024-01-03T18:00:00+08:00",
            ],
        }
    ).to_csv(path, index=False)
    evidence = load_historical_st_evidence(path)
    attached = attach_historical_st(
        _raw_main_board()[["symbol", "trade_date"]],
        evidence,
    )
    assert attached["is_st"].tolist() == [False, True, False]


def test_overlapping_interval_evidence_is_rejected(tmp_path) -> None:
    path = tmp_path / "overlap.csv"
    pd.DataFrame(
        {
            "symbol": ["600001.SH", "600001.SH"],
            "start_date": ["2024-01-01", "2024-01-03"],
            "end_date": ["2024-01-04", "2024-01-05"],
            "is_st": [False, True],
            "available_at": [
                "2023-12-29T18:00:00+08:00",
                "2024-01-02T18:00:00+08:00",
            ],
        }
    ).to_csv(path, index=False)

    with pytest.raises(HistoricalSTCoverageError, match="overlapping intervals"):
        load_historical_st_evidence(path)
