"""Machine-readable mapping of Fuyao / Financial-API best practices to QuantAgent.

This registry is a product/runtime contract, not sample market data.  It keeps
upstream examples from degenerating into a static checklist in the UI: each
scenario declares the upstream endpoints, QuantAgent destination, analytical
contract and the boundary that must remain visible to the user.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BestPractice:
    id: str
    slug: str
    title: str
    category: str
    quantagent_path: str
    endpoints: tuple[str, ...]
    outputs: tuple[str, ...]
    contract: tuple[str, ...]
    boundaries: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


DATA_GROUPS = (
    {
        "id": "market",
        "title": "行情与历史数据",
        "capabilities": ("最新行情", "历史K线", "复权事件", "成交变化", "指数"),
        "quantagent": ("Market Workbench", "Fuyao full sync", "Market Dump", "Backtester"),
    },
    {
        "id": "financial",
        "title": "财务与公司数据",
        "capabilities": ("三大报表", "财务指标", "公司资料", "机构信息/持股"),
        "quantagent": ("Financial Health", "PIT financial pipeline", "Evidence Center"),
    },
    {
        "id": "fund",
        "title": "基金数据",
        "capabilities": ("公募基金", "ETF", "LOF", "REITs", "净值", "持仓", "收益", "持有人"),
        "quantagent": ("Fuyao full sync", "Data Lab", "Research datasets"),
    },
    {
        "id": "special",
        "title": "盘面特色数据",
        "capabilities": ("涨停池", "连板", "热榜", "飙升榜", "龙虎榜", "异动原因"),
        "quantagent": ("Market Intelligence", "Evidence overlays"),
    },
    {
        "id": "calendar",
        "title": "交易日历与市场基础",
        "capabilities": ("交易日历", "停复牌/可交易性", "指数成分", "市场分类"),
        "quantagent": ("PIT calendar", "tradability gates", "universe construction"),
    },
    {
        "id": "artifact",
        "title": "任务型结果输出",
        "capabilities": ("表格", "图表", "离线HTML", "研究报告", "自选股日报/复盘"),
        "quantagent": ("Evidence Center", "offline HTML export", "Runtime artifacts"),
    },
)


BEST_PRACTICES = (
    BestPractice("01", "stock-overview", "单股行情与趋势速览", "market", "/stock-replay", (
        "/api/meta/tickers/search", "/api/a-share/prices/snapshot", "/api/a-share/prices/historical", "/api/a-share/valuations/snapshot",
    ), ("K线", "成交额", "MA20/60/120", "回撤", "估值", "provenance"), (
        "前复权历史与最新快照分开标记", "交互缩放/十字光标", "保留数据时间与复权口径",
    ), ("无数据不造 mock",)),
    BestPractice("02", "financial-health", "单股财务体检", "financial", "/market-intelligence?view=financial", (
        "/api/a-share/financials/income-statements", "/api/a-share/financials/balance-sheets", "/api/a-share/financials/cash-flow-statements", "/api/a-share/financials/indicators",
    ), ("增长", "盈利", "现金流", "杠杆", "字段完整度"), (
        "财务三表按报告期对齐", "历史使用 report_date_ms 作为可用时点", "指标缺披露时间时不得直接回填历史",
    ), ("period_end_ms 不是历史可用时间",)),
    BestPractice("03", "index-constituents", "同花顺概念板块联动", "market", "/market-intelligence?view=sector", (
        "/api/a-share-index/catalog/ths-index-list", "/api/a-share-index/prices/historical", "/api/a-share-index/constituents/ths-stock-list", "/api/a-share/prices/snapshot",
    ), ("指数走势", "当前成分", "横截面涨跌", "成交额"), (
        "指数与当前成分同屏", "当前成分只用于当前横截面",
    ), ("没有历史成分/权重时不得回填历史归属",)),
    BestPractice("04", "limit-up-market", "涨停池与连板天梯", "special", "/market-intelligence", (
        "/api/a-share/special-data/limit-up-pool", "/api/a-share/special-data/limit-up-ladder",
    ), ("涨停数量", "最高板", "封单", "原因", "30日梯队"), (
        "涨停池与连板天梯分别保留原始语义", "分页/窗口信息随数据展示",
    ), ("盘后观察数据不直接变成交易信号",)),
    BestPractice("05", "watchlist-anomalies", "自选股当日异动监控", "special", "/market-intelligence", (
        "/api/a-share/special-data/anomaly-analysis-list", "/api/a-share/special-data/anomaly-analysis-stock", "/api/a-share/prices/snapshot",
    ), ("自选股行情", "异动标签", "发生时间", "原因"), (
        "列表与指定个股异动能力分开", "保留事件时间与原始解释",
    ), ("异动原因属于观察证据",)),
    BestPractice("06", "marketdb-research", "本地全市场趋势研究", "market", "/runtime?view=data", (
        "/dump/market-dumps/daily-k/download-url", "/dump/market-dumps/daily-k-10d/download-url", "/dump/market-dumps/adjustment-factors/download-url",
    ), ("市场宽度", "趋势", "流动性分层", "DuckDB/Parquet研究"), (
        "按(thscode,date_ms)去重", "原始未复权与复权事件分离", "保存freshness/hash/schema",
    ), ("未应用复权事件的面板必须标 raw",)),
    BestPractice("07", "market-heat-radar", "市场热度与飙升雷达", "special", "/market-intelligence", (
        "/api/a-share/special-data/hot-stock-list", "/api/a-share/special-data/hot-stock-list-history", "/api/a-share/special-data/hot-stock-rank-trend", "/api/a-share/special-data/skyrocket-list",
    ), ("热榜", "飙升", "排名变化", "历史趋势"), (
        "day/hour 口径分开", "排名趋势保留原始方向",
    ), ("关注度不等于预期收益",)),
    BestPractice("08", "dragon-tiger-watch", "龙虎榜机构与游资观察", "special", "/market-intelligence", (
        "/api/a-share/special-data/dragon-tiger-list?board_type=all", "/api/a-share/special-data/dragon-tiger-list?board_type=org", "/api/a-share/special-data/dragon-tiger-list?board_type=hot_money",
    ), ("机构净额", "游资净额", "上榜股票", "席位"), (
        "all/org/hot_money 分栏保留", "range_days 不同窗口不得直接重复累加",
    ), ("龙虎榜为交易日级收盘数据",)),
    BestPractice("09", "industry-strength-rotation", "行业强度作战矩阵", "market", "/market-intelligence?view=sector", (
        "/api/a-share-index/catalog/ths-index-list", "/api/a-share-index/prices/historical", "/api/a-share-index/constituents/ths-stock-list", "/api/a-share/prices/snapshot",
    ), ("横截面强度", "加速度", "当前成分贡献"), (
        "强度与加速度基于一致交易日窗口", "贡献必须注明当前成分近似",
    ), ("当前成分不可冒充历史权重",)),
    BestPractice("10", "cashflow-quality", "现金流质量稽核台", "financial", "/market-intelligence?view=financial", (
        "/api/a-share/financials/income-statements", "/api/a-share/financials/balance-sheets", "/api/a-share/financials/cash-flow-statements",
    ), ("现金转化", "自由现金流", "应计质量", "字段完整度", "披露日"), (
        "同报告期三表对齐", "所有历史比较按披露日可见性", "缺字段显式降级",
    ), ("不得以报告期末提前看到财报",)),
    BestPractice("11", "attention-price-resonance", "热榜—股价关系观察台", "special", "/stock-replay", (
        "/api/a-share/special-data/hot-stock-list-history", "/api/a-share/special-data/hot-stock-rank-trend", "/api/a-share/prices/historical", "/api/a-share-index/prices/historical",
    ), ("排名", "股价", "基准", "Spearman", "分组"), (
        "统一交易日轴", "day/hour 不混用", "排名轴保持名次越小越热的语义", "股票与基准价格可指数化比较",
    ), ("相关性不等于因果或交易建议",)),
    BestPractice("12", "limitup-sentiment-timing", "涨停情绪市场脉冲屏", "special", "/market-intelligence", (
        "/api/a-share/special-data/limit-up-pool", "/api/a-share/special-data/limit-up-ladder",
    ), ("涨停", "梯队", "封单留存", "原因分布"), (
        "交易日时间轴对齐", "保留封单与原因分布原始字段",
    ), ("情绪脉冲是观察量而非仓位指令",)),
    BestPractice("13", "price-volume-breakout", "价格成交量突破回测台", "backtest", "/backtests", (
        "/dump/market-dumps/daily-k/download-url", "/dump/market-dumps/adjustment-factors/download-url", "/api/a-share-index/prices/historical",
    ), ("净值/回撤", "逐笔交易", "假突破", "事件K线", "参数敏感性"), (
        "前55日高点排除当天", "20日退出低点", "量比+MA60", "T收盘信号→T+1开盘成交", "费用/滑点显式",
    ), ("日K只能近似涨跌停成交约束", "除权除息不能制造虚假突破")),
    BestPractice("14", "time-series-momentum", "时间序列动量回测台", "backtest", "/backtests", (
        "/api/a-share-index/catalog/ths-index-list", "/api/a-share-index/prices/historical",
    ), ("净值/回撤", "状态泳道", "现金状态", "风险贡献", "窗口敏感性"), (
        "120日动量+MA120+60日波动率", "周/月信号由日K本地重采样", "等权/逆波动率", "T收盘→T+1开盘", "无Active资产=100%现金",
    ), ("现金状态不是行情缺失", "单次运行资产池必须明确")),
    BestPractice("15", "short-term-reversal", "短期反转回测实验室", "backtest", "/backtests", (
        "/dump/market-dumps/daily-k/download-url", "/dump/market-dumps/adjustment-factors/download-url", "/api/a-share-index/prices/historical",
    ), ("十分组", "Rank IC", "市场状态", "净值/回撤", "形成期×持有期敏感性"), (
        "过去5日相对基准收益", "流动性+MA120+异常跌幅过滤", "底部10%", "T收盘选股→T+1开盘", "固定持有5日+5日冷却",
    ), ("反转因子预期Rank IC为负；长期为正须标反转证据不足",)),
    BestPractice("16", "dragon-tiger-capital-flow", "龙虎榜资金流向拓扑台", "special", "/market-intelligence", (
        "/api/a-share/calendar/trading-days", "/api/a-share/special-data/dragon-tiger-list?board_type=all", "/api/a-share/special-data/dragon-tiger-list?board_type=org", "/api/a-share/special-data/dragon-tiger-list?board_type=hot_money",
    ), ("概念累计净额", "机构/游资/股票穿透", "资金路径", "正负贡献"), (
        "默认range_days=1避免1日/3日榜重复", "多概念股票按概念数等分净额以守恒", "跨交易日聚合",
    ), ("日级数据不得伪装09:31–15:00盘中分时",)),
)


def best_practice_payload() -> dict[str, object]:
    return {
        "source": "HiThink-Tech/Financial-API + Fuyao official best practices",
        "count": len(BEST_PRACTICES),
        "dataGroups": list(DATA_GROUPS),
        "items": [item.as_dict() for item in BEST_PRACTICES],
        "outputContract": {
            "offlineHtml": True,
            "showDataTime": True,
            "showMode": True,
            "showSourceEndpoint": True,
            "showCalculationBasis": True,
            "showNonInvestmentAdvice": True,
            "browserApiKey": False,
            "unavailableDataPolicy": "no synthetic fallback",
        },
    }


__all__ = ["BEST_PRACTICES", "DATA_GROUPS", "BestPractice", "best_practice_payload"]
