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


def test_northbound_extracts_three_channels_from_real_schema():
    # Mirrors real akshare stock_hsgt_fund_flow_summary_em rows.
    raw = pd.DataFrame({
        "交易日": ["2024-01-02", "2024-01-02"],
        "类型": ["沪港通", "深港通"],
        "板块": ["沪股通", "深股通"],
        "资金方向": ["北向", "北向"],
        "成交净买额": [1.5e9, 0.7e9],
    })
    out = _normalize_northbound(raw)
    assert set(out["channel"]) == {"north_hgt", "north_sgt", "north_total"}
    total = out[out["channel"] == "north_total"]
    assert total.iloc[0]["net_inflow_cny"] == pytest.approx(2.2e9)
    assert total.iloc[0]["available_at"] == pd.Timestamp("2024-01-02") + pd.Timedelta(days=FLOW_AVAILABLE_AT_LAG_DAYS)


def test_northbound_ignores_southbound_rows():
    raw = pd.DataFrame({
        "交易日": ["2024-01-02"] * 3,
        "板块": ["沪股通", "深股通", "港股通(沪)"],
        "资金方向": ["北向", "北向", "南向"],
        "成交净买额": [1.0e9, 0.5e9, 2.0e9],
    })
    out = _normalize_northbound(raw)
    channels = set(out["channel"])
    assert channels == {"north_hgt", "north_sgt", "north_total"}
    total = out[out["channel"] == "north_total"].iloc[0]
    assert total["net_inflow_cny"] == pytest.approx(1.5e9)


def test_margin_balance_keeps_short_when_present():
    combined = pd.DataFrame({
        "observation_date": ["2024-01-02", "2024-01-02"],
        "market": ["SH", "SZ"],
        "margin_balance_cny": [9.0e11, 7.5e11],
        "short_balance_cny": [3.0e10, 2.5e10],
    })
    out = _normalize_margin_balance(combined)
    assert len(out) == 2
    assert out["short_balance_cny"].sum() == pytest.approx(5.5e10)


def test_provider_requires_network(tmp_path):
    provider = AkShareFlowProvider(allow_network=False, root=str(tmp_path))
    with pytest.raises(ProviderUnavailable):
        provider.fetch_all()


def test_empty_input_returns_empty():
    assert _normalize_northbound(pd.DataFrame()).empty
    assert _normalize_margin_balance(pd.DataFrame()).empty
