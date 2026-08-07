from __future__ import annotations

from datetime import datetime, timedelta
import json
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.providers.fuyao_provider import FuyaoProvider
from services.quant_api.services.connections import ConnectionManager


SHANGHAI = ZoneInfo("Asia/Shanghai")


class MarketDataService:
    """Read-only market workstation data backed by the official Fuyao contract.

    Secrets stay server-side. The browser receives only normalized market data and
    provenance, never API keys or credential fingerprints.
    """

    def __init__(self, connections: ConnectionManager) -> None:
        self.connections = connections

    def _provider(self) -> FuyaoProvider:
        if not self.connections.has_variable("HITHINK_FINANCE_API_KEY"):
            raise ProviderUnavailable(
                "同花顺 Fuyao 未连接；请在服务端 .env 配置 HITHINK_FINANCE_API_KEY。"
            )
        return FuyaoProvider(allow_network=True)

    def search(self, query: str, *, limit: int = 12) -> list[dict[str, Any]]:
        provider = self._provider()
        frame = provider.ticker_search(
            query.strip(),
            asset_type="a-share",
            limit=max(1, min(50, limit)),
        )
        rows = _records(frame)
        return [
            {
                "symbol": row.get("thscode"),
                "ticker": row.get("ticker"),
                "name": row.get("name"),
                "exchange": row.get("exchange"),
                "assetType": row.get("asset_type"),
                "currency": row.get("currency") or "CNY",
            }
            for row in rows
            if row.get("thscode")
        ]

    def stock_overview(self, symbol: str, *, calendar_days: int = 420) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        provider = self._provider()
        now = datetime.now(SHANGHAI)
        end = now.date()
        start = end - timedelta(days=max(370, min(3650, calendar_days)))
        request = ProviderRequest(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            symbols=(symbol,),
        )

        search_rows = _records(provider.ticker_search(symbol, asset_type="a-share", limit=5))
        identity = next((row for row in search_rows if row.get("thscode") == symbol), None)
        if identity is None and search_rows:
            identity = search_rows[0]

        snapshot_rows = _records(provider.snapshot((symbol,)))
        snapshot = snapshot_rows[0] if snapshot_rows else {}

        history = provider.historical_prices(request, adjust="forward")
        bars = []
        for row in _records(history.frame.tail(320)):
            bars.append(
                {
                    "datetime": row.get("trade_date"),
                    "symbol": symbol,
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                    "amount": row.get("amount"),
                }
            )

        valuation_rows = _records(provider.valuations_snapshot((symbol,)))
        valuation_raw = valuation_rows[0] if valuation_rows else {}
        valuation = {
            "peTtm": valuation_raw.get("pe_ttm"),
            "peMrq": valuation_raw.get("pe_mrq"),
            "pbMrq": valuation_raw.get("pb_mrq"),
            "psTtm": valuation_raw.get("ps_ttm"),
            "pcfTtm": valuation_raw.get("pcf_ttm"),
        }

        normalized_snapshot = {
            "symbol": snapshot.get("symbol") or snapshot.get("thscode") or symbol,
            "ticker": snapshot.get("ticker") or (identity or {}).get("ticker"),
            "lastPrice": snapshot.get("last_price"),
            "priceChange": snapshot.get("price_change"),
            "changePercent": snapshot.get("price_change_ratio_pct"),
            "open": snapshot.get("open") if snapshot.get("open") is not None else snapshot.get("open_price"),
            "high": snapshot.get("high") if snapshot.get("high") is not None else snapshot.get("high_price"),
            "low": snapshot.get("low") if snapshot.get("low") is not None else snapshot.get("low_price"),
            "prevClose": snapshot.get("prev") if snapshot.get("prev") is not None else snapshot.get("prev_price"),
            "volume": snapshot.get("volume"),
            "amount": snapshot.get("amount") if snapshot.get("amount") is not None else snapshot.get("turnover"),
            "asOf": snapshot.get("available_at"),
        }

        return {
            "symbol": symbol,
            "ticker": (identity or {}).get("ticker") or normalized_snapshot["ticker"],
            "name": (identity or {}).get("name") or valuation_raw.get("name"),
            "exchange": (identity or {}).get("exchange"),
            "currency": (identity or {}).get("currency") or "CNY",
            "source": "hithink_fuyao",
            "adjustment": "forward",
            "interval": "1d",
            "asOf": normalized_snapshot.get("asOf") or (bars[-1]["datetime"] if bars else None),
            "snapshot": normalized_snapshot,
            "valuation": valuation,
            "bars": bars,
            "provenance": {
                "snapshotEndpoint": "/api/a-share/prices/snapshot",
                "historyEndpoint": "/api/a-share/prices/historical",
                "valuationEndpoint": "/api/a-share/valuations/snapshot",
                "identityEndpoint": "/api/meta/tickers/search",
                "historyAdjustment": "forward",
                "calendarStart": start.isoformat(),
                "calendarEnd": end.isoformat(),
            },
        }

    def financial_health(self, symbol: str, *, limit: int = 5) -> dict[str, Any]:
        """Return annual statement series for the Financial-API health view.

        The UI deliberately receives the upstream statement field names so it can
        inspect exact accounting semantics. report_date_ms remains attached to
        every row and is the only historical availability timestamp.
        """
        symbol = symbol.strip().upper()
        provider = self._provider()
        count = max(1, min(10, limit))
        statements: dict[str, list[dict[str, Any]]] = {}
        endpoints = {
            "income": "/api/a-share/financials/income-statements",
            "balance": "/api/a-share/financials/balance-sheets",
            "cashflow": "/api/a-share/financials/cash-flow-statements",
        }
        for name, path in endpoints.items():
            data = provider.get_capability(
                path,
                {"thscode": symbol, "period": "annual", "limit": count},
            )
            statements[name] = _mapping_items(data)
        return {
            "symbol": symbol,
            "source": "hithink_fuyao",
            "period": "annual",
            "statements": statements,
            "provenance": endpoints,
            "pitKey": "report_date_ms",
            "periodKey": "period_end_ms",
        }

    def index_catalog(self, tag: str = "industry") -> dict[str, Any]:
        normalized = tag.strip().lower()
        if normalized not in {"industry", "cn_concept", "region", "tszs"}:
            raise ValueError("index tag must be industry/cn_concept/region/tszs")
        provider = self._provider()
        data = provider.get_capability(
            "/api/a-share-index/catalog/ths-index-list",
            {"tag": normalized},
        )
        return {
            "tag": normalized,
            "timestamp": data.get("timestamp"),
            "items": _mapping_items(data),
            "source": "hithink_fuyao",
            "endpoint": "/api/a-share-index/catalog/ths-index-list",
        }

    def index_overview(self, symbol: str, *, calendar_days: int = 180) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        provider = self._provider()
        now = datetime.now(SHANGHAI)
        end = now.date()
        start = end - timedelta(days=max(90, min(730, calendar_days)))
        request = ProviderRequest(start.isoformat(), end.isoformat(), (symbol,))
        history = provider.index_daily(request)
        bars = []
        for row in _records(history.frame.tail(180)):
            bars.append(
                {
                    "datetime": row.get("trade_date"),
                    "symbol": symbol,
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                    "amount": row.get("amount"),
                }
            )
        constituents = _records(provider.index_constituents(symbol))
        constituent_symbols = tuple(
            str(row.get("thscode") or row.get("symbol"))
            for row in constituents[:80]
            if row.get("thscode") or row.get("symbol")
        )
        snapshot_rows = _records(provider.snapshot(constituent_symbols)) if constituent_symbols else []
        names = {
            str(row.get("thscode") or row.get("symbol")): row.get("name")
            for row in constituents
        }
        snapshots = []
        for row in snapshot_rows:
            code = str(row.get("symbol") or row.get("thscode") or "")
            snapshots.append(
                {
                    "symbol": code,
                    "name": names.get(code),
                    "lastPrice": row.get("last_price"),
                    "changePercent": row.get("price_change_ratio_pct"),
                    "amount": row.get("amount") if row.get("amount") is not None else row.get("turnover"),
                }
            )
        return {
            "symbol": symbol,
            "source": "hithink_fuyao",
            "bars": bars,
            "constituentCount": len(constituents),
            "constituents": constituents[:200],
            "snapshots": snapshots,
            "provenance": {
                "history": "/api/a-share-index/prices/historical",
                "constituents": "/api/a-share-index/constituents/ths-stock-list",
                "stockSnapshot": "/api/a-share/prices/snapshot",
            },
        }

    def market_intelligence(self) -> dict[str, Any]:
        """Aggregate current market-observation capabilities used by Fuyao inspirations.

        Each capability is isolated: one unavailable entitlement or empty upstream
        dataset must not erase the rest of the workstation. Failures are returned
        as explicit per-panel issues and are never replaced with synthetic rows.
        """
        provider = self._provider()
        specs: dict[str, tuple[str, Mapping[str, Any] | None]] = {
            "hotDay": ("/api/a-share/special-data/hot-stock-list", {"period": "day"}),
            "hotHour": ("/api/a-share/special-data/hot-stock-list", {"period": "hour"}),
            "skyrocketDay": ("/api/a-share/special-data/skyrocket-list", {"period": "day"}),
            "skyrocketHour": ("/api/a-share/special-data/skyrocket-list", {"period": "hour"}),
            "dragonAll": ("/api/a-share/special-data/dragon-tiger-list", {"board_type": "all"}),
            "dragonOrg": ("/api/a-share/special-data/dragon-tiger-list", {"board_type": "org"}),
            "dragonHotMoney": ("/api/a-share/special-data/dragon-tiger-list", {"board_type": "hot_money"}),
            "limitPool": (
                "/api/a-share/special-data/limit-up-pool",
                {"page": 1, "size": 200, "sort_field": "continue_day_cnt", "sort_dir": "desc"},
            ),
            "limitLadder": ("/api/a-share/special-data/limit-up-ladder", None),
            "anomalyList": ("/api/a-share/special-data/anomaly-analysis-list", None),
        }
        panels: dict[str, Any] = {}
        issues: list[dict[str, str]] = []
        for name, (path, params) in specs.items():
            try:
                panels[name] = provider.get_capability(path, params)
            except ProviderUnavailable as exc:
                panels[name] = None
                issues.append({"panel": name, "endpoint": path, "message": str(exc)})
        return {
            "source": "hithink_fuyao",
            "panels": panels,
            "issues": issues,
            "provenance": {name: path for name, (path, _) in specs.items()},
        }


def _mapping_items(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = data.get("item", [])
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


__all__ = ["MarketDataService"]
