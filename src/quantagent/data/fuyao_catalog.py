"""Authoritative Fuyao / HiThink capability registry.

This module mirrors the public documentation index at
https://fuyao.aicubes.cn/llms.txt and the generated aggregate contract at
https://fuyao.aicubes.cn/llms-full.txt.

The registry is deliberately explicit.  It gives QuantAgent a machine-auditable
contract so a newly documented endpoint cannot be silently omitted from the
acquisition layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


CapabilityStatus = Literal["available", "coming_soon"]
CapabilityKind = Literal["rest", "dump", "planned"]


@dataclass(frozen=True, slots=True)
class FuyaoCapability:
    id: str
    domain: str
    name: str
    kind: CapabilityKind
    status: CapabilityStatus
    rest_path: str | None
    mcp_tool: str | None
    mcp_service: str | None
    doc_url: str
    scope: str
    retention: str
    pit_note: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _rest(
    id: str,
    domain: str,
    name: str,
    path: str,
    tool: str | None,
    service: str | None,
    doc: str,
    scope: str,
    retention: str,
    pit_note: str,
) -> FuyaoCapability:
    return FuyaoCapability(
        id=id,
        domain=domain,
        name=name,
        kind="rest",
        status="available",
        rest_path=path,
        mcp_tool=tool,
        mcp_service=service,
        doc_url=doc,
        scope=scope,
        retention=retention,
        pit_note=pit_note,
    )


REST_CAPABILITIES: tuple[FuyaoCapability, ...] = (
    _rest("meta.ticker_search", "meta", "标的检索", "/api/meta/tickers/search", "get_meta_tickers_search", "fuyao-meta-mcp", "https://fuyao.aicubes.cn/docs/api-reference/ticker-search/", "cross-universe search", "current metadata", "retrieval-time metadata"),
    _rest("meta.ticker_list", "meta", "标的列表", "/api/meta/tickers/list", "get_meta_tickers_list", "fuyao-meta-mcp", "https://fuyao.aicubes.cn/docs/api-reference/ticker-list/", "paged universe enumeration", "current metadata", "retrieval-time metadata"),
    _rest("a_share.prices_snapshot", "a_share", "A股行情快照", "/api/a-share/prices/snapshot", "get_a_share_prices_snapshot", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/prices/", "batch A-share snapshot", "latest", "snapshot only; not a historical PIT series"),
    _rest("a_share.prices_historical", "a_share", "A股历史K线", "/api/a-share/prices/historical", "get_a_share_prices_historical", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/prices/", "single A-share, none/forward/backward adjustment", "maximum 10-year window; current interval 1d", "daily bar available no earlier than the following trading/model date"),
    _rest("a_share.valuation_snapshot", "a_share", "A股估值快照", "/api/a-share/valuations/snapshot", "get_a_share_valuations_snapshot", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/valuations/", "batch up to documented token limit", "latest only", "not a historical valuation series"),
    _rest("a_share.adjustment_factors", "a_share", "除复权因子事件流", "/api/a-share/corporate-actions/adjustment-factors", "get_a_share_corporate_actions_adjustment_factors", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/corporate-actions/", "single A-share corporate actions", "endpoint window per request", "event use is conservatively gated by ex-date"),
    _rest("a_share.income_statements", "a_share", "利润表", "/api/a-share/financials/income-statements", "get_a_share_financials_income_statements", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/financials/", "single A-share annual or quarterly", "maximum 10-year date window or recent 1-20 reports", "report_date_ms is disclosure/PIT key; period_end_ms is accounting period"),
    _rest("a_share.balance_sheets", "a_share", "资产负债表", "/api/a-share/financials/balance-sheets", "get_a_share_financials_balance_sheets", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/financials/", "single A-share annual or quarterly", "maximum 10-year date window or recent 1-20 reports", "report_date_ms is disclosure/PIT key; period_end_ms is accounting period"),
    _rest("a_share.cash_flow_statements", "a_share", "现金流量表", "/api/a-share/financials/cash-flow-statements", "get_a_share_financials_cash_flow_statements", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/financials/", "single A-share annual or quarterly", "maximum 10-year date window or recent 1-20 reports", "report_date_ms is disclosure/PIT key; period_end_ms is accounting period"),
    _rest("a_share.financial_indicators", "a_share", "财务指标数据", "/api/a-share/financials/indicators", "get_a_share_financials_indicators", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/financial-indicators/", "single A-share + yyyy-quarter", "one report per call", "documented payload has no disclosure timestamp; fail closed for historical training until disclosure is joined"),
    _rest("a_share.trading_calendar", "a_share", "A股交易日历", "/api/a-share/calendar/trading-days", "get_a_share_calendar_trading_days", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/calendar/", "no-argument calendar", "fixed recent one year", "calendar metadata"),
    _rest("a_share.limit_up_pool", "special", "涨停股票池", "/api/a-share/special-data/limit-up-pool", "get_a_share_special_data_limit_up_pool", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/limit-up-data/", "paged per trading date", "date-addressable where upstream retains it", "observation data; no trading decision semantics"),
    _rest("a_share.limit_up_ladder", "special", "连板天梯", "/api/a-share/special-data/limit-up-ladder", "get_a_share_special_data_limit_up_ladder", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/limit-up-data/", "no-argument ladder", "fixed recent 30 trading days", "observation data"),
    _rest("a_share.skyrocket_list", "special", "飙升榜", "/api/a-share/special-data/skyrocket-list", "get_a_share_special_data_skyrocket_list", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/hot-list-data/", "day/hour Top30", "current day/hour", "observation data"),
    _rest("a_share.hot_stock_list", "special", "热股榜单", "/api/a-share/special-data/hot-stock-list", "get_a_share_special_data_hot_stock_list", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/hot-list-data/", "day/hour Top30", "current day/hour", "observation data"),
    _rest("a_share.hot_stock_history", "special", "历史热股排行", "/api/a-share/special-data/hot-stock-list-history", "get_a_share_special_data_hot_stock_list_history", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/hot-list-data/", "one natural date", "one year", "natural-date observation series"),
    _rest("a_share.hot_stock_rank_trend", "special", "个股热榜排名走势", "/api/a-share/special-data/hot-stock-rank-trend", "get_a_share_special_data_hot_stock_rank_trend", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/hot-list-data/", "single A-share date range", "maximum one year", "natural-date observation series"),
    _rest("a_share.anomaly_list", "special", "个股异动原因列表", "/api/a-share/special-data/anomaly-analysis-list", None, None, "https://fuyao.aicubes.cn/docs/api-reference/anomaly-data/", "current-day all/list filter", "current day", "REST-only official capability; observation text"),
    _rest("a_share.anomaly_stock", "special", "指定个股异动原因", "/api/a-share/special-data/anomaly-analysis-stock", "get_a_share_special_data_anomaly_analysis_stock", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/anomaly-data/", "up to 50 A-share tokens", "current day", "observation text"),
    _rest("a_share.dragon_tiger", "special", "龙虎榜", "/api/a-share/special-data/dragon-tiger-list", "get_a_share_special_data_dragon_tiger_list", "fuyao-a-share-mcp", "https://fuyao.aicubes.cn/docs/api-reference/dragon-tiger-data/", "all/org/hot_money per trading date", "one year", "trading-date observation data"),
    _rest("index.catalog", "index", "同花顺指数列表", "/api/a-share-index/catalog/ths-index-list", "get_a_share_index_catalog_ths_index_list", "fuyao-a-share-index-mcp", "https://fuyao.aicubes.cn/docs/api-reference/index/", "industry/cn_concept/region/tszs", "current catalog", "current membership universe metadata"),
    _rest("index.constituents", "index", "同花顺指数成分股", "/api/a-share-index/constituents/ths-stock-list", "get_a_share_index_constituents_ths_stock_list", "fuyao-a-share-index-mcp", "https://fuyao.aicubes.cn/docs/api-reference/index/", "single index", "current constituents only", "must not be treated as historical constituents"),
    _rest("index.prices_snapshot", "index", "指数行情快照", "/api/a-share-index/prices/snapshot", "get_a_share_index_prices_snapshot", "fuyao-a-share-index-mcp", "https://fuyao.aicubes.cn/docs/api-reference/index/", "batch index snapshot", "latest", "snapshot only"),
    _rest("index.prices_historical", "index", "指数历史K线", "/api/a-share-index/prices/historical", "get_a_share_index_prices_historical", "fuyao-a-share-index-mcp", "https://fuyao.aicubes.cn/docs/api-reference/index/", "single index daily history", "maximum 10-year window", "daily observation series; no index adjustment parameter"),
    _rest("fund.profile", "fund", "基金基本资料", "/api/fund/profile/detail", "get_fund_profile_detail", "fuyao-fund-mcp", "https://fuyao.aicubes.cn/docs/api-reference/fund-profile/", "single otc/exchange/reits fund", "current profile", "reference metadata"),
    _rest("fund.holdings", "fund", "基金重仓股", "/api/fund/portfolio/holdings", "get_fund_portfolio_holdings", "fuyao-fund-mcp", "https://fuyao.aicubes.cn/docs/api-reference/fund-holdings/", "single otc/exchange/reits fund", "latest disclosed holdings", "periodic disclosure; not realtime holdings"),
    _rest("fund.nav", "fund", "基金净值", "/api/fund/performance/nav", "get_fund_performance_nav", "fuyao-fund-mcp", "https://fuyao.aicubes.cn/docs/api-reference/fund-performance/", "single otc/exchange/reits fund", "range max fyear (5y)", "NAV observation series"),
    _rest("fund.returns", "fund", "基金区间收益", "/api/fund/performance/returns", "get_fund_performance_returns", "fuyao-fund-mcp", "https://fuyao.aicubes.cn/docs/api-reference/fund-performance/", "single otc/exchange/reits fund", "current return windows incl since inception", "derived current performance snapshot"),
    _rest("fund.holders", "fund", "基金持有人结构", "/api/fund/holders/detail", "get_fund_holders_detail", "fuyao-fund-mcp", "https://fuyao.aicubes.cn/docs/api-reference/fund-holders/", "single otc/exchange/reits fund", "latest merged/separate disclosure", "report_date_ms is disclosure/report date"),
    _rest("fund.market_snapshot", "fund", "场内基金行情快照", "/api/fund/market/snapshot", "get_fund_market_snapshot", "fuyao-fund-mcp", "https://fuyao.aicubes.cn/docs/api-reference/fund-market/", "single ETF only", "latest", "snapshot only"),
    _rest("fund.market_historical", "fund", "场内基金历史日线", "/api/fund/market/historical", "get_fund_market_historical", "fuyao-fund-mcp", "https://fuyao.aicubes.cn/docs/api-reference/fund-market/", "single ETF only, interval 1d", "maximum 5 natural years", "daily ETF observation series"),
)


DUMP_CAPABILITIES: tuple[FuyaoCapability, ...] = (
    FuyaoCapability("dump.daily_k", "market_dump", "全市场近10年日K", "dump", "available", "/dump/market-dumps/daily-k/download-url", None, None, "https://fuyao.aicubes.cn/docs/api-reference/market-dumps/", "full A-share parquet", "approximately recent 10 years", "canonical bulk raw market panel"),
    FuyaoCapability("dump.daily_k_10d", "market_dump", "全市场近10交易日日K", "dump", "available", "/dump/market-dumps/daily-k-10d/download-url", None, None, "https://fuyao.aicubes.cn/docs/api-reference/market-dumps/", "full A-share recent parquet", "recent 10 trading days", "incremental bulk market panel"),
    FuyaoCapability("dump.adjustment_factors", "market_dump", "全市场复权因子", "dump", "available", "/dump/market-dumps/adjustment-factors/download-url", None, None, "https://fuyao.aicubes.cn/docs/api-reference/market-dumps/", "full A-share adjustment-factor parquet", "upstream retained history", "bulk corporate-action adjustment basis"),
)


PLANNED_CAPABILITIES: tuple[FuyaoCapability, ...] = (
    FuyaoCapability("planned.stock_basics", "a_share", "股票基础信息", "planned", "coming_soon", None, None, None, "https://fuyao.aicubes.cn/docs/api-reference/stock-basics/", "planned", "not published", "must remain unavailable until an official live endpoint is documented"),
    FuyaoCapability("planned.index_overview", "index", "指数概况/历史成分/权重", "planned", "coming_soon", None, None, None, "https://fuyao.aicubes.cn/docs/api-reference/index-overview/", "planned", "not published", "historical constituent/weight PIT support is not currently available"),
    FuyaoCapability("planned.stock_index_membership", "index", "个股反查同花顺指数", "planned", "coming_soon", None, None, None, "https://fuyao.aicubes.cn/docs/api-reference/stock-ths-index/", "planned", "not published", "must not infer historical membership from current constituents"),
)


ALL_CAPABILITIES = REST_CAPABILITIES + DUMP_CAPABILITIES + PLANNED_CAPABILITIES


def coverage_summary() -> dict[str, int]:
    return {
        "live_rest": len(REST_CAPABILITIES),
        "live_mcp": sum(cap.mcp_tool is not None for cap in REST_CAPABILITIES),
        "market_dumps": len(DUMP_CAPABILITIES),
        "coming_soon": len(PLANNED_CAPABILITIES),
        "documented_total": len(ALL_CAPABILITIES),
    }


def validate_catalog() -> None:
    summary = coverage_summary()
    expected = {
        "live_rest": 31,
        "live_mcp": 30,
        "market_dumps": 3,
        "coming_soon": 3,
        "documented_total": 37,
    }
    if summary != expected:
        raise RuntimeError(f"Fuyao capability registry count drift: {summary} != {expected}")

    ids = [cap.id for cap in ALL_CAPABILITIES]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Fuyao capability ids must be unique")

    rest_paths = [cap.rest_path for cap in REST_CAPABILITIES]
    if len(rest_paths) != len(set(rest_paths)):
        raise RuntimeError("Fuyao REST paths must be unique")
    if any(path is None or not path.startswith("/api/") for path in rest_paths):
        raise RuntimeError("Every live REST capability must use a documented /api/* path")

    tools = [cap.mcp_tool for cap in REST_CAPABILITIES if cap.mcp_tool]
    if len(tools) != len(set(tools)):
        raise RuntimeError("Fuyao MCP tool names must be unique")

    rest_only = [cap.id for cap in REST_CAPABILITIES if cap.mcp_tool is None]
    if rest_only != ["a_share.anomaly_list"]:
        raise RuntimeError(f"Unexpected REST-only Fuyao capability set: {rest_only}")

    if any(cap.rest_path is not None for cap in PLANNED_CAPABILITIES):
        raise RuntimeError("Coming-soon capabilities must not expose invented REST paths")


validate_catalog()


__all__ = [
    "ALL_CAPABILITIES",
    "DUMP_CAPABILITIES",
    "PLANNED_CAPABILITIES",
    "REST_CAPABILITIES",
    "FuyaoCapability",
    "coverage_summary",
    "validate_catalog",
]
