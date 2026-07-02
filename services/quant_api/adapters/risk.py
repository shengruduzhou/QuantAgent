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
        run = next((item for item in runs if item["id"] == backtest_id), runs[0])
        equity = self.backtests.equity(run["id"])
        daily_returns = [point["dailyReturn"] for point in equity if point.get("dailyReturn") is not None]
        consecutive = _max_consecutive_losses(daily_returns)
        events = self.backtests.risk_events(run["id"], page=1, page_size=1_000)["items"]
        counts: dict[str, int] = {}
        for event in events:
            counts[event["type"]] = counts.get(event["type"], 0) + 1
        return {
            "backtestId": run["id"],
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
        selected = next((item for item in runs if item["id"] == backtest_id), runs[0])
        return self.backtests.risk_events(selected["id"], page=page, page_size=page_size)

    def stocks(self, backtest_id: str | None = None) -> list[dict[str, Any]]:
        runs = self.backtests.list()
        if not runs:
            return []
        selected = next((item for item in runs if item["id"] == backtest_id), runs[0])
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
        from quantagent.execution.risk_kill_switch import KillSwitchLimits
        from quantagent.risk.risk_limits import V6RiskLimits

        risk_limits = V6RiskLimits()
        kill_limits = KillSwitchLimits()
        return [
            {
                "id": "max_name_weight",
                "name": "Single-name weight cap",
                "description": "限制单票目标权重。",
                "threshold": risk_limits.max_name_weight,
                "enabled": True,
                "codeLocation": "src/quantagent/risk/risk_gate.py",
            },
            {
                "id": "max_drawdown",
                "name": "Drawdown kill switch",
                "description": "组合回撤超过阈值时触发 kill switch。",
                "threshold": kill_limits.max_drawdown_pct,
                "enabled": True,
                "codeLocation": "src/quantagent/execution/risk_kill_switch.py",
            },
            {
                "id": "t_plus_one",
                "name": "T+1 sellability",
                "description": "卖出数量不得超过昨日已结算可卖库存。",
                "threshold": "available_shares",
                "enabled": True,
                "codeLocation": "src/quantagent/execution/virtual_broker.py",
            },
            {
                "id": "limit_and_suspension",
                "name": "Limit/suspension gate",
                "description": "阻止涨停买入、跌停卖出和停牌交易。",
                "threshold": None,
                "enabled": True,
                "codeLocation": "src/quantagent/risk/risk_gate.py",
            },
            {
                "id": "max_daily_loss",
                "name": "Daily loss kill switch",
                "description": "单日亏损超过阈值时触发 kill switch。",
                "threshold": kill_limits.max_daily_loss_pct,
                "enabled": True,
                "codeLocation": "src/quantagent/execution/risk_kill_switch.py",
            },
            {
                "id": "max_sector_weight",
                "name": "Sector weight cap",
                "description": "限制单一行业组合权重。",
                "threshold": risk_limits.max_sector_weight,
                "enabled": True,
                "codeLocation": "src/quantagent/risk/risk_gate.py",
            },
            {
                "id": "max_turnover",
                "name": "Turnover cap",
                "description": "限制目标权重相对当前权重的换手。",
                "threshold": risk_limits.max_turnover,
                "enabled": True,
                "codeLocation": "src/quantagent/risk/risk_gate.py",
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
