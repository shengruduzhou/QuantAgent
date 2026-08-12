"""Unit tests for AkShareFlowProvider normalisers."""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.providers.akshare_flow_provider import (
    AkShareFlowProvider,
    FLOW_AVAILABLE_AT_LAG_DAYS,
    _normalize_margin_balance,
    _normalize_northbound,
)
from quantagent.data.providers.base import ProviderUnavailable
from quantagent.data.trading_calendar import TradingCalendar


def _calendar() -> TradingCalendar:
    # 2024-01-05 is Friday; the next decision session is Monday 2024-01-08.
    return TradingCalendar.from_dates(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )


def test_northbound_extracts_three_channels_and_converts_yiyuan_to_cny():
    raw = pd.DataFrame(
        {
            "交易日": ["2024-01-05", "2024-01-05"],
            "类型": ["沪港通", "深港通"],
            "板块": ["沪股通", "深股通"],
            "资金方向": ["北向", "北向"],
            # AKShare stock_hsgt_fund_flow_summary_em source unit is 亿元.
            "成交净买额": [15.0, 7.0],
        }
    )
    out = _normalize_northbound(raw, trading_calendar=_calendar())
    assert set(out["channel"]) == {"north_hgt", "north_sgt", "north_total"}
    by_channel = out.set_index("channel")["net_inflow_cny"]
    assert by_channel["north_hgt"] == pytest.approx(1.5e9)
    assert by_channel["north_sgt"] == pytest.approx(0.7e9)
    assert by_channel["north_total"] == pytest.approx(2.2e9)
    total = out[out["channel"] == "north_total"]
    assert total.iloc[0]["available_at"] == pd.Timestamp("2024-01-08")
    assert FLOW_AVAILABLE_AT_LAG_DAYS == 1


def test_northbound_without_calendar_is_not_silently_weekday_or_calendar_day_pit():
    raw = pd.DataFrame(
        {
            "交易日": ["2024-01-05", "2024-01-05"],
            "板块": ["沪股通", "深股通"],
            "资金方向": ["北向", "北向"],
            "成交净买额": [10.0, 5.0],
        }
    )
    out = _normalize_northbound(raw)
    assert out["available_at"].isna().all()
    assert out.loc[out["channel"] == "north_total", "net_inflow_cny"].iloc[0] == pytest.approx(1.5e9)


def test_northbound_ignores_southbound_rows():
    raw = pd.DataFrame(
        {
            "交易日": ["2024-01-02"] * 3,
            "板块": ["沪股通", "深股通", "港股通(沪)"],
            "资金方向": ["北向", "北向", "南向"],
            "成交净买额": [10.0, 5.0, 20.0],
        }
    )
    out = _normalize_northbound(raw, trading_calendar=_calendar())
    channels = set(out["channel"])
    assert channels == {"north_hgt", "north_sgt", "north_total"}
    total = out[out["channel"] == "north_total"].iloc[0]
    assert total["net_inflow_cny"] == pytest.approx(1.5e9)


def test_margin_balance_keeps_short_and_uses_next_session():
    combined = pd.DataFrame(
        {
            "observation_date": ["2024-01-05", "2024-01-05"],
            "market": ["SH", "SZ"],
            "margin_balance_cny": [9.0e11, 7.5e11],
            "short_balance_cny": [3.0e10, 2.5e10],
        }
    )
    out = _normalize_margin_balance(combined, trading_calendar=_calendar())
    assert len(out) == 2
    assert out["short_balance_cny"].sum() == pytest.approx(5.5e10)
    assert set(out["available_at"]) == {pd.Timestamp("2024-01-08")}


def test_provider_requires_network(tmp_path):
    provider = AkShareFlowProvider(allow_network=False, root=str(tmp_path))
    with pytest.raises(ProviderUnavailable):
        provider.fetch_all()


def test_empty_input_returns_empty():
    assert _normalize_northbound(pd.DataFrame()).empty
    assert _normalize_margin_balance(pd.DataFrame()).empty
