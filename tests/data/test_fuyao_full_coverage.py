from __future__ import annotations

from datetime import date, datetime, timezone

from quantagent.data.fuyao_catalog import (
    ALL_CAPABILITIES,
    DUMP_CAPABILITIES,
    PLANNED_CAPABILITIES,
    REST_CAPABILITIES,
    coverage_summary,
    validate_catalog,
)
from quantagent.data.fuyao_dump import DUMP_ENDPOINTS
from quantagent.data.fuyao_full_sync import (
    DUMP_STRATEGIES,
    SYNC_STRATEGIES,
    _date_ms,
    build_coverage_audit,
    validate_sync_coverage,
)


def test_official_fuyao_contract_counts_are_locked():
    validate_catalog()
    assert coverage_summary() == {
        "live_rest": 31,
        "live_mcp": 30,
        "market_dumps": 3,
        "coming_soon": 3,
        "documented_total": 37,
    }
    assert len(ALL_CAPABILITIES) == 37


def test_every_live_rest_and_dump_has_an_explicit_sync_strategy():
    validate_sync_coverage()
    assert set(SYNC_STRATEGIES) == {cap.id for cap in REST_CAPABILITIES}
    assert set(DUMP_STRATEGIES) == {cap.id for cap in DUMP_CAPABILITIES}


def test_anomaly_list_is_the_only_documented_rest_only_capability():
    rest_only = [cap.id for cap in REST_CAPABILITIES if cap.mcp_tool is None]
    assert rest_only == ["a_share.anomaly_list"]


def test_coming_soon_capabilities_never_invent_live_endpoints():
    assert {cap.id for cap in PLANNED_CAPABILITIES} == {
        "planned.stock_basics",
        "planned.index_overview",
        "planned.stock_index_membership",
    }
    assert all(cap.rest_path is None for cap in PLANNED_CAPABILITIES)


def test_market_dump_paths_follow_current_public_contract():
    assert DUMP_ENDPOINTS == {
        "daily-k": "/dump/market-dumps/daily-k/download-url",
        "daily-k-10d": "/dump/market-dumps/daily-k-10d/download-url",
        "adjustment-factors": "/dump/market-dumps/adjustment-factors/download-url",
    }


def test_audit_surfaces_vendor_retention_and_enumeration_gaps():
    audit = build_coverage_audit()
    hard_limits = "\n".join(audit["hard_limits"])
    assert "REIT" in hard_limits
    assert "5y" in hard_limits
    assert "one year" in hard_limits
    assert "Financial indicators" in hard_limits
    ids = {entry["id"] for entry in audit["rest"]}
    assert {
        "fund.profile",
        "fund.market_historical",
        "a_share.hot_stock_history",
        "a_share.hot_stock_rank_trend",
        "a_share.anomaly_stock",
        "index.prices_snapshot",
    } <= ids


def test_natural_date_parameters_use_shanghai_midnight():
    value = _date_ms(date(2026, 7, 1))
    utc = datetime.fromtimestamp(value / 1000, timezone.utc)
    assert utc.isoformat() == "2026-06-30T16:00:00+00:00"
