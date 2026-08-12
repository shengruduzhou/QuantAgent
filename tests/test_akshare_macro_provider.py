"""Unit tests for AkShareMacroProvider normalisers (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.providers.akshare_calendar import AkShareCalendarEvidence
from quantagent.data.providers.akshare_macro_provider import (
    AkShareMacroProvider,
    DAILY_AVAILABLE_AT_LAG_DAYS,
    MACRO_AVAILABLE_AT_LAG_DAYS,
    _normalize_aggregate_financing,
    _normalize_central_bank_balance,
    _normalize_cpi_ppi,
    _normalize_money_supply,
    _normalize_repo,
    _normalize_shibor,
    _normalize_yield_curve,
    _parse_yearmonth,
)
from quantagent.data.providers.base import ProviderUnavailable
from quantagent.data.trading_calendar import TradingCalendar


def _calendar() -> TradingCalendar:
    # Friday 2024-01-05 must resolve to Monday 2024-01-08, never Saturday.
    return TradingCalendar.from_dates(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    )


def test_yield_curve_normalises_chinese_columns_on_next_session():
    raw = pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-05"],
        "1Y": [2.10, 2.12],
        "10Y": [2.55, 2.57],
        "30Y": [2.92, 2.94],
    })
    out = _normalize_yield_curve(raw, trading_calendar=_calendar())
    assert set(out.columns) >= {"observation_date", "maturity", "yield_pct", "available_at", "source"}
    assert set(out["maturity"]) >= {"1Y", "10Y", "30Y"}
    one = out[(out["maturity"] == "1Y") & (out["observation_date"] == pd.Timestamp("2024-01-02"))]
    assert one.iloc[0]["yield_pct"] == pytest.approx(2.10)
    assert one.iloc[0]["available_at"] == pd.Timestamp("2024-01-03")
    friday = out[(out["maturity"] == "1Y") & (out["observation_date"] == pd.Timestamp("2024-01-05"))]
    assert friday.iloc[0]["available_at"] == pd.Timestamp("2024-01-08")
    assert DAILY_AVAILABLE_AT_LAG_DAYS == 1


def test_shibor_handles_per_tenor_frame_and_next_session():
    raw = pd.DataFrame({"报告日": ["2024-01-02", "2024-01-05"], "利率": [1.85, 1.88]})
    out = _normalize_shibor(raw, tenor="O/N", trading_calendar=_calendar())
    assert set(out["tenor"]) == {"O/N"}
    assert out["rate_pct"].tolist() == [1.85, 1.88]
    assert out["available_at"].tolist() == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-08")]


def test_repo_extracts_fr007_and_dr007_on_next_session():
    raw = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-05"],
        "FR001": [1.8, 1.9],
        "FR007": [2.10, 2.15],
        "FDR007": [2.05, 2.08],
        "FR014": [2.20, 2.21],
        "FDR001": [1.75, 1.85],
        "FDR014": [2.15, 2.18],
    })
    out = _normalize_repo(raw, trading_calendar=_calendar())
    tenors = set(out["tenor"])
    assert {"FR007", "DR007"}.issubset(tenors)
    fr007_jan2 = out[(out["tenor"] == "FR007") & (out["observation_date"] == pd.Timestamp("2024-01-02"))]
    assert fr007_jan2.iloc[0]["rate_pct"] == pytest.approx(2.10)
    fr007_friday = out[(out["tenor"] == "FR007") & (out["observation_date"] == pd.Timestamp("2024-01-05"))]
    assert fr007_friday.iloc[0]["available_at"] == pd.Timestamp("2024-01-08")


def test_daily_macro_without_calendar_fails_closed_to_nat():
    yield_raw = pd.DataFrame({"日期": ["2024-01-05"], "1Y": [2.10]})
    shibor_raw = pd.DataFrame({"报告日": ["2024-01-05"], "利率": [1.85]})
    repo_raw = pd.DataFrame({"date": ["2024-01-05"], "FR007": [2.10]})

    assert _normalize_yield_curve(yield_raw)["available_at"].isna().all()
    assert _normalize_shibor(shibor_raw)["available_at"].isna().all()
    assert _normalize_repo(repo_raw)["available_at"].isna().all()


def test_unresolved_daily_macro_is_not_cached_as_pit(tmp_path):
    provider = AkShareMacroProvider(allow_network=False, root=str(tmp_path))
    frame = _normalize_repo(pd.DataFrame({"date": ["2024-01-05"], "FR007": [2.10]}))
    evidence = AkShareCalendarEvidence(
        TradingCalendar.from_dates(()),
        {"source": "test_empty_calendar", "status": "empty", "production_certified": False},
        ("test_calendar_unavailable",),
    )

    result = provider._persist_result(
        "repo",
        frame,
        calendar_evidence=evidence,
        akshare_version="test",
    )

    assert result.point_in_time is False
    assert result.metadata["cached_as_pit"] is False
    assert result.metadata["path"] is None
    assert "akshare_macro_available_at_unresolved:not_cached_as_pit" in result.warnings
    assert not provider.cache.path_for("repo").exists()


def test_central_bank_balance_keeps_top_aggregates():
    raw = pd.DataFrame({
        "统计时间": ["2026.4", "2026.3"],
        "总资产": [486327.50, 491398.71],
        "储备货币": [398320.18, 408745.66],
        "国外资产": [229293.34, 228260.72],
        "对其他存款性公司债权": [208060.02, 215879.70],
        "对政府债权": [22519.40, 22615.59],
    })
    out = _normalize_central_bank_balance(raw)
    assert {"total_assets_cny", "reserve_money_cny", "foreign_assets_cny",
            "claims_on_depository_corp_cny", "claims_on_government_cny"}.issubset(set(out.columns))
    assert out.iloc[0]["total_assets_cny"] == pytest.approx(486327.50)
    # 2026.4 → April 2026 month end = 2026-04-30, available_at = +35 days.
    assert out.iloc[0]["available_at"] == pd.Timestamp("2026-04-30") + pd.Timedelta(days=MACRO_AVAILABLE_AT_LAG_DAYS)


def test_money_supply_handles_real_akshare_columns():
    raw = pd.DataFrame({
        "月份": ["2026年04月份", "2026年03月份"],
        "货币和准货币(M2)-数量(亿元)": [3530425.21, 3538636.53],
        "货币和准货币(M2)-同比增长": [8.6, 8.5],
        "货币(M1)-数量(亿元)": [1145833.73, 1193202.99],
        "货币(M1)-同比增长": [5.0, 5.1],
        "流通中的现金(M0)-数量(亿元)": [147477.38, 147082.81],
        "流通中的现金(M0)-同比增长": [12.2, 12.5],
    })
    out = _normalize_money_supply(raw)
    assert out.iloc[0]["m2_cny"] == pytest.approx(3530425.21)
    assert out.iloc[0]["m2_yoy_pct"] == pytest.approx(8.6)
    # 2026年04月 → 2026-04-30, available_at = +35 days.
    assert out.iloc[0]["available_at"] == pd.Timestamp("2026-04-30") + pd.Timedelta(days=MACRO_AVAILABLE_AT_LAG_DAYS)


def test_parse_yearmonth_accepts_multiple_forms():
    assert _parse_yearmonth("2026年04月份") == pd.Timestamp("2026-04-30")
    assert _parse_yearmonth("2026.4") == pd.Timestamp("2026-04-30")
    assert _parse_yearmonth("2026-04") == pd.Timestamp("2026-04-30")
    assert pd.isna(_parse_yearmonth(""))


def test_cpi_ppi_basic_mapping():
    raw = pd.DataFrame({"日期": ["2024-01-15"], "今值": [0.3]})
    cpi = _normalize_cpi_ppi(raw, kind="cpi")
    assert "cpi_yoy_pct" in cpi.columns
    assert cpi.iloc[0]["cpi_yoy_pct"] == pytest.approx(0.3)


def test_aggregate_financing_basic_mapping():
    raw = pd.DataFrame({"月份": ["2024-01-31"], "社会融资规模增量": [5_000_000_000_000.0]})
    out = _normalize_aggregate_financing(raw)
    assert out.iloc[0]["aggregate_financing_cny"] == pytest.approx(5_000_000_000_000.0)
    assert "available_at" in out.columns


def test_provider_requires_explicit_network(tmp_path):
    provider = AkShareMacroProvider(allow_network=False, root=str(tmp_path))
    with pytest.raises(ProviderUnavailable):
        provider.fetch_all()


def test_empty_input_returns_empty():
    assert _normalize_yield_curve(pd.DataFrame()).empty
    assert _normalize_shibor(pd.DataFrame()).empty
    assert _normalize_repo(pd.DataFrame()).empty
    assert _normalize_central_bank_balance(pd.DataFrame()).empty
    assert _normalize_money_supply(pd.DataFrame()).empty
    assert _normalize_cpi_ppi(pd.DataFrame(), kind="cpi").empty
