from __future__ import annotations

from typing import Any

from services.quant_api.adapters.backtests import BacktestAdapter


class RiskAdapter:
    def __init__(self, backtests: BacktestAdapter) -> None:
        self.backtests = backtests

    def overview(self, backtest_id: str | None = None) -> dict[str, Any]:
        runs = self.backtests.list()
        if not runs:
            return self._empty()
        run = _requested_run(runs, backtest_id)
        equity = self.backtests.equity(run["id"])
        daily_returns = [point["dailyReturn"] for point in equity if point.get("dailyReturn") is not None]
        # No measured daily return means the streak is unknown, not 0 days.
        consecutive = _max_consecutive_losses(daily_returns) if daily_returns else None
        page = self.backtests.risk_events(run["id"], page=1, page_size=1_000)
        events = page["items"]
        counts: dict[str, int] = {}
        for event in events:
            counts[event["type"]] = counts.get(event["type"], 0) + 1
        return {
            "backtestId": run["id"],
            "backtestName": run.get("name"),
            # Persisted events of this backtest (e.g. skipped orders), counted
            # over the first page only when the page was full.
            "eventCountsExact": bool(page.get("totalIsExact", not page.get("hasNext", False))),
            "eventCountsBasis": "persisted_backtest_events",
            "maxDrawdown": run.get("maxDrawdown"),
            "maxSingleStockLoss": self._max_stock_loss(run["id"]),
            "maxDailyLoss": min(daily_returns) if daily_returns else None,
            "consecutiveLossDays": consecutive,
            "concentration": None,
            "sectorConcentration": None,
            "volatilityExposure": run.get("volatility"),
            "liquidityRisk": _event_share(events, ("no_liquidity", "partial")),
            "limitDownRisk": _event_share(events, ("limit_down",)),
            "suspensionRisk": _event_share(events, ("suspend",)),
            "doTFailureRisk": None,
            "eventCounts": counts,
            "rules": self.rules(),
        }

    def events(self, backtest_id: str | None = None, page: int = 1, page_size: int = 100) -> dict[str, Any]:
        runs = self.backtests.list()
        if not runs:
            return {"items": [], "total": 0, "page": page, "pageSize": page_size, "hasNext": False}
        selected = _requested_run(runs, backtest_id)
        return self.backtests.risk_events(selected["id"], page=page, page_size=page_size)

    def stocks(self, backtest_id: str | None = None) -> list[dict[str, Any]]:
        runs = self.backtests.list()
        if not runs:
            return []
        selected = _requested_run(runs, backtest_id)
        directory = self.backtests._resolve(selected["id"])
        from services.quant_api.adapters.utils import read_csv_rows

        rows = read_csv_rows(directory / "profit_by_stock.csv")
        output = []
        for row in rows:
            net_pnl = _float(row.get("net_pnl"))
            output.append({
                "symbol": row.get("symbol"),
                "netPnl": net_pnl,
                "winRate": _float(row.get("win_rate")),
                "tradeCount": _int(row.get("n_trades")),
                "riskScore": max(0.0, -net_pnl) if net_pnl is not None else None,
            })
        return sorted(output, key=lambda item: item["riskScore"] or 0.0, reverse=True)

    @staticmethod
    def rules() -> list[dict[str, Any]]:
        """The limits the paper venue's RiskEngine actually enforces.

        These used to come from ``V6RiskLimits`` / ``KillSwitchLimits`` - config
        classes no order path reads - so the UI showed a 5% name cap, a 15%
        drawdown switch and a 3% daily-loss switch, all "enabled", while the
        venue enforced 10%, 20% and 20,000 CNY (and, before round 29, never
        evaluated drawdown or daily loss at all).
        """
        from quantagent.paper.risk import RiskLimits

        limits = RiskLimits()
        venue = "src/quantagent/paper/risk.py"
        return [
            {
                "id": "max_single_name_weight",
                "name": "Single-name weight cap",
                "description": "买入后单票权重上限（paper venue 逐单检查）。",
                "threshold": limits.max_single_name_weight,
                "unit": "fraction_of_equity",
                "enforcedAt": "pre_trade_order",
                "enabled": True,
                "codeLocation": venue,
            },
            {
                "id": "max_industry_weight",
                "name": "Industry weight cap",
                "description": "买入后行业权重上限；缺少行业映射时拒绝买入（industry_unmeasured）。",
                "threshold": limits.max_industry_weight,
                "unit": "fraction_of_equity",
                "enforcedAt": "pre_trade_order_with_sector_map",
                "enabled": True,
                "codeLocation": venue,
            },
            {
                "id": "max_drawdown",
                "name": "Drawdown kill switch",
                "description": "相对历史最高净值的回撤超过阈值后锁定，只允许减仓，需人工解除。",
                "threshold": limits.max_drawdown,
                "unit": "fraction_from_all_time_peak",
                "enforcedAt": "portfolio_after_fill_and_session_open",
                "enabled": True,
                "codeLocation": venue,
            },
            {
                "id": "max_daily_loss",
                "name": "Daily loss kill switch",
                "description": "单个交易日亏损超过开盘权益的该比例后锁定，只允许减仓，需人工解除。",
                "threshold": limits.max_daily_loss_fraction,
                "unit": "fraction_of_session_opening_equity",
                "enforcedAt": "portfolio_after_fill_and_session_open",
                "enabled": True,
                "codeLocation": venue,
            },
            {
                "id": "max_gross_exposure",
                "name": "Gross exposure cap",
                "description": "总敞口/净值上限。",
                "threshold": limits.max_gross_exposure,
                "unit": "fraction_of_equity",
                "enforcedAt": "portfolio_after_fill_and_session_open",
                "enabled": True,
                "codeLocation": venue,
            },
            {
                "id": "max_daily_turnover",
                "name": "Daily turnover cap",
                "description": "按交易所交易日累计的成交额/净值上限（重启后恢复）。",
                "threshold": limits.max_daily_turnover,
                "unit": "fraction_of_equity_per_session",
                "enforcedAt": "pre_trade_order",
                "enabled": True,
                "codeLocation": venue,
            },
            {
                "id": "max_order_notional",
                "name": "Order notional cap",
                "description": "单笔委托金额上限。",
                "threshold": limits.max_order_notional,
                "unit": "cny",
                "enforcedAt": "pre_trade_order",
                "enabled": True,
                "codeLocation": venue,
            },
            {
                "id": "max_price_deviation",
                "name": "Fat-finger guard",
                "description": "限价偏离参考价上限。",
                "threshold": limits.max_price_deviation,
                "unit": "fraction_of_reference_price",
                "enforcedAt": "pre_trade_order",
                "enabled": True,
                "codeLocation": venue,
            },
            {
                "id": "t_plus_one_and_price_limits",
                "name": "T+1 / limit / suspension / ST rules",
                "description": "T+1 可卖、涨停不买、跌停不卖、停牌不交易、ST 买入受限。",
                "threshold": None,
                "unit": None,
                "enforcedAt": "venue_instrument_rules",
                "enabled": True,
                "codeLocation": "src/quantagent/paper/broker.py",
            },
        ]

    def _max_stock_loss(self, backtest_id: str) -> float | None:
        stocks = self.stocks(backtest_id)
        losses = [item["netPnl"] for item in stocks if item.get("netPnl") is not None]
        return min(losses) if losses else None

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "backtestId": None,
            "maxDrawdown": None,
            "maxSingleStockLoss": None,
            "maxDailyLoss": None,
            "consecutiveLossDays": None,
            "concentration": None,
            "sectorConcentration": None,
            "volatilityExposure": None,
            "liquidityRisk": None,
            "limitDownRisk": None,
            "suspensionRisk": None,
            "doTFailureRisk": None,
            "eventCounts": {},
            "rules": RiskAdapter.rules(),
        }


def _max_consecutive_losses(values: list[float]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value < 0 else 0
        best = max(best, current)
    return best


def _event_share(events: list[dict[str, Any]], tokens: tuple[str, ...]) -> float | None:
    if not events:
        return None
    matched = 0
    for event in events:
        text = f"{event.get('type', '')} {event.get('reason', '')}".lower()
        matched += int(any(token in text for token in tokens))
    return matched / len(events)


def _float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _requested_run(runs: list[dict[str, Any]], backtest_id: str | None) -> dict[str, Any]:
    """The run the caller asked for; KeyError for an unknown id.

    Falling back to ``runs[0]`` for an unknown id returned another backtest's
    risk numbers under status "ready" - the wrong subject, presented as valid.
    """
    if backtest_id is None:
        return runs[0]
    for item in runs:
        if item["id"] == backtest_id:
            return item
    raise KeyError(backtest_id)
