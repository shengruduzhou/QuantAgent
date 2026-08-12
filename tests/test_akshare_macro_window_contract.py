"""Regression tests for bounded AKShare macro PIT persistence."""

from __future__ import annotations

import pandas as pd

from quantagent.data.providers.akshare_calendar import AkShareCalendarEvidence
from quantagent.data.providers.akshare_macro_provider import (
    AkShareMacroProvider,
    _normalize_repo,
)
from quantagent.data.trading_calendar import TradingCalendar


class _BoundedFakeAkShare:
    __version__ = "test"

    @staticmethod
    def bond_china_yield(**_kwargs):
        return pd.DataFrame()

    @staticmethod
    def rate_interbank(**_kwargs):
        # The source is intentionally unbounded: the row after end_date cannot
        # be allowed to poison PIT validation for the requested 2024-01-05 row.
        return pd.DataFrame(
            {
                "报告日": ["2024-01-05", "2024-01-08"],
                "利率": [1.85, 1.86],
            }
        )

    @staticmethod
    def repo_rate_hist(**_kwargs):
        return pd.DataFrame(
            {
                "date": ["2024-01-05", "2024-01-08"],
                "FR007": [2.10, 2.11],
                "FDR007": [2.05, 2.06],
            }
        )

    @staticmethod
    def macro_china_central_bank_balance(**_kwargs):
        return pd.DataFrame()

    @staticmethod
    def macro_china_shrzgm(**_kwargs):
        return pd.DataFrame()

    @staticmethod
    def macro_china_money_supply(**_kwargs):
        return pd.DataFrame()

    @staticmethod
    def macro_china_cpi_yearly(**_kwargs):
        return pd.DataFrame()

    @staticmethod
    def macro_china_ppi_yearly(**_kwargs):
        return pd.DataFrame()


def _calendar() -> TradingCalendar:
    return TradingCalendar.from_dates(["2024-01-05", "2024-01-08"])


def test_unbounded_shibor_and_repo_are_windowed_before_pit_validation(tmp_path, monkeypatch):
    provider = AkShareMacroProvider(
        allow_network=True,
        root=str(tmp_path),
        trading_calendar=_calendar(),
    )
    monkeypatch.setattr(provider, "_akshare", lambda: _BoundedFakeAkShare())

    results = provider.fetch_all(start_date="2024-01-05", end_date="2024-01-05")

    for table in ("shibor", "repo"):
        result = results[table]
        assert result.point_in_time is True
        assert result.metadata["cached_as_pit"] is True
        assert result.frame["observation_date"].eq(pd.Timestamp("2024-01-05")).all()
        assert result.frame["available_at"].eq(pd.Timestamp("2024-01-08")).all()
        assert provider.cache.path_for(table).exists()


def test_uncached_unresolved_rows_are_not_returned_as_usable_build_rows(tmp_path):
    provider = AkShareMacroProvider(allow_network=False, root=str(tmp_path))
    unresolved = _normalize_repo(
        pd.DataFrame({"date": ["2024-01-05"], "FR007": [2.10]})
    )
    evidence = AkShareCalendarEvidence(
        TradingCalendar.from_dates(()),
        {"source": "test_empty_calendar", "status": "empty", "production_certified": False},
        ("test_calendar_unavailable",),
    )

    result = provider._persist_result(
        "repo",
        unresolved,
        calendar_evidence=evidence,
        akshare_version="test",
    )

    assert result.point_in_time is False
    assert result.frame.empty
    assert result.metadata["row_count"] == 1
    assert result.metadata["cached_as_pit"] is False
    assert result.metadata["path"] is None
    assert not provider.cache.path_for("repo").exists()
