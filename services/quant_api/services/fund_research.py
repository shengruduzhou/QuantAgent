from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from quantagent.data.providers.base import ProviderUnavailable
from services.quant_api.services.market_data import MarketDataService

SHANGHAI = ZoneInfo("Asia/Shanghai")
_ALLOWED_FUND_TYPES = {"otc", "exchange", "reits"}


class FundResearchService:
    """Read-only Fuyao fund research aggregation for OTC, exchange funds and REITs."""

    def __init__(self, market: MarketDataService | None = None) -> None:
        self.market = market or MarketDataService()

    def overview(self, thscode: str, *, fund_type: str = "exchange") -> dict[str, Any]:
        symbol = str(thscode or "").strip().upper()
        kind = str(fund_type or "").strip().lower()
        if not symbol or "." not in symbol:
            raise ValueError("fund thscode must include the market suffix")
        if kind not in _ALLOWED_FUND_TYPES:
            raise ValueError("fund_type must be one of otc/exchange/reits")

        provider = self.market._provider()
        panels: dict[str, Any] = {}
        issues: list[dict[str, str]] = []
        specs: tuple[tuple[str, str, Mapping[str, Any]], ...] = (
            ("profile", "/api/fund/profile/detail", {"fund_type": kind, "thscode": symbol}),
            ("holdings", "/api/fund/portfolio/holdings", {"fund_type": kind, "thscode": symbol}),
            ("nav", "/api/fund/performance/nav", {"fund_type": kind, "thscode": symbol, "range": "fyear", "nav_type": "unit,adj"}),
            ("returns", "/api/fund/performance/returns", {"fund_type": kind, "thscode": symbol}),
            ("holders", "/api/fund/holders/detail", {"fund_type": kind, "thscode": symbol, "merge_scope": "all"}),
        )
        for name, endpoint, params in specs:
            try:
                panels[name] = provider.get_capability(endpoint, params)
            except ProviderUnavailable as exc:
                panels[name] = None
                issues.append({"panel": name, "endpoint": endpoint, "message": str(exc)})

        market_contract = "not_applicable"
        if kind == "exchange":
            market_contract = "etf_only_upstream"
            try:
                panels["marketSnapshot"] = provider.get_capability(
                    "/api/fund/market/snapshot", {"thscode": symbol}
                )
            except ProviderUnavailable as exc:
                panels["marketSnapshot"] = None
                issues.append({"panel": "marketSnapshot", "endpoint": "/api/fund/market/snapshot", "message": str(exc)})
            try:
                now = datetime.now(SHANGHAI)
                start = now - timedelta(days=365 * 5)
                panels["marketHistory"] = provider.get_capability(
                    "/api/fund/market/historical",
                    {"thscode": symbol, "interval": "1d", "start": int(start.timestamp() * 1000), "end": int(now.timestamp() * 1000)},
                )
            except ProviderUnavailable as exc:
                panels["marketHistory"] = None
                issues.append({"panel": "marketHistory", "endpoint": "/api/fund/market/historical", "message": str(exc)})

        return {
            "symbol": symbol,
            "fundType": kind,
            "source": "hithink_fuyao",
            "panels": panels,
            "issues": issues,
            "pit": {
                "holdings": "periodic disclosure; not a realtime position",
                "holders": "report_date_ms controls the disclosed record date",
                "nav": "historical NAV from upstream fund performance contract",
                "market": market_contract,
            },
            "provenance": {
                "profile": "/api/fund/profile/detail",
                "holdings": "/api/fund/portfolio/holdings",
                "nav": "/api/fund/performance/nav",
                "returns": "/api/fund/performance/returns",
                "holders": "/api/fund/holders/detail",
                "marketSnapshot": "/api/fund/market/snapshot" if kind == "exchange" else None,
                "marketHistory": "/api/fund/market/historical" if kind == "exchange" else None,
            },
        }


__all__ = ["FundResearchService"]
