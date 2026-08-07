from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from quantagent.data.fuyao_catalog import (
    ALL_CAPABILITIES,
    DUMP_CAPABILITIES,
    PLANNED_CAPABILITIES,
    REST_CAPABILITIES,
    coverage_summary,
    validate_catalog,
)
from quantagent.data.fuyao_docs_audit import compare_live_contract, parse_llms_full_contract
from quantagent.data.fuyao_dump import DUMP_ENDPOINTS
from quantagent.data.fuyao_exhaustive_sync import ExhaustiveFuyaoSynchronizer
from quantagent.data.fuyao_full_sync import (
    DUMP_STRATEGIES,
    SYNC_STRATEGIES,
    FuyaoUniverse,
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


def test_live_docs_parser_extracts_rest_dump_and_mcp_sets():
    text = """
GET /api/meta/tickers/list
GET /api/a-share/prices/snapshot
GET /dump/market-dumps/daily-k/download-url
## 工具一览
| fuyao-meta-mcp | /mcp/meta | get_meta_tickers_list | 标的列表 | GET /api/meta/tickers/list |
| fuyao-a-share-mcp | /mcp/a-share | get_a_share_prices_snapshot | 快照 | GET /api/a-share/prices/snapshot |
## AI Agent 跨服务调用场景
get_fake_tool_outside_overview
"""
    parsed = parse_llms_full_contract(text)
    assert parsed["rest_paths"] == {
        "/api/meta/tickers/list",
        "/api/a-share/prices/snapshot",
    }
    assert parsed["dump_paths"] == {"/dump/market-dumps/daily-k/download-url"}
    assert parsed["mcp_tools"] == {
        "get_meta_tickers_list",
        "get_a_share_prices_snapshot",
    }


def test_live_contract_comparator_accepts_exact_registry_sets():
    parsed = {
        "rest_paths": {cap.rest_path for cap in REST_CAPABILITIES if cap.rest_path},
        "dump_paths": {cap.rest_path for cap in DUMP_CAPABILITIES if cap.rest_path},
        "mcp_tools": {cap.mcp_tool for cap in REST_CAPABILITIES if cap.mcp_tool},
    }
    result = compare_live_contract(parsed)
    assert result["ok"] is True
    assert all(not value for value in result["diffs"].values())


class _PagedHistoryProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_capability(self, path: str, params=None):
        assert path == "/api/a-share/prices/historical"
        request = dict(params or {})
        self.calls.append(request)
        offset = int(request.get("offset", 0))
        if offset == 0:
            return {
                "timestamp": 1,
                "item": [
                    {"date_ms": 1, "close_price": 10.0},
                    {"date_ms": 2, "close_price": 11.0},
                ],
            }
        if offset == 2:
            return {"timestamp": 1, "item": [{"date_ms": 3, "close_price": 12.0}]}
        return {"timestamp": 1, "item": []}


def test_exhaustive_sync_consumes_every_historical_offset_page(tmp_path: Path):
    provider = _PagedHistoryProvider()
    sync = ExhaustiveFuyaoSynchronizer(tmp_path, provider=provider)  # type: ignore[arg-type]
    universe = FuyaoUniverse(
        a_shares=("600519.SH",),
        indexes=(),
        fund_otc=(),
        fund_etf=(),
        fund_lof=(),
        fund_reits=(),
    )

    sync._sync_adjusted_stock_history(universe, date(2025, 1, 1), date(2026, 1, 1))

    assert [(call["adjust"], call["offset"]) for call in provider.calls] == [
        ("forward", 0),
        ("forward", 2),
        ("forward", 3),
        ("backward", 0),
        ("backward", 2),
        ("backward", 3),
    ]
    assert not any(
        event.status in {"pagination_stalled", "pagination_limit_reached"}
        for event in sync.events
    )
