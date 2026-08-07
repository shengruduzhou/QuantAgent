from __future__ import annotations

from datetime import datetime, timedelta
import json
from typing import Any
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
        # QuantAgent loads the repository .env into the backend process. Keep the
        # key inside that process; this service only verifies connection state and
        # never serializes credentials to the browser.
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
            "open": snapshot.get("open_price"),
            "high": snapshot.get("high_price"),
            "low": snapshot.get("low_price"),
            "prevClose": snapshot.get("prev_price"),
            "volume": snapshot.get("volume"),
            "amount": snapshot.get("amount") or snapshot.get("turnover"),
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


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    # pandas/numpy scalars and timestamps are not FastAPI JSON-safe by default.
    # A JSON round-trip keeps nulls as null and emits ISO timestamps consistently.
    return json.loads(frame.to_json(orient="records", date_format="iso"))


__all__ = ["MarketDataService"]