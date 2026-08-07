"""Official Fuyao / HiThink A-share provider.

Contract sources:
- https://fuyao.aicubes.cn/llms-full.txt
- https://github.com/HiThink-Tech/Financial-API/tree/main/docs/api

PIT invariants:
- raw daily bars are the canonical market-panel input; adjusted views are kept
  separate;
- financial statements become usable at ``report_date_ms`` (disclosure date),
  never at ``period_end_ms``;
- every normalised row carries provenance;
- capabilities that cannot satisfy a historical PIT contract fail closed or are
  explicitly marked non-PIT instead of being silently backfilled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from quantagent.data.ashare.http import HttpClient, RETRY_ENTITLEMENT
from quantagent.data.providers.base import (
    FundamentalsProvider,
    IndexDataProvider,
    MarketDataProvider,
    ProviderRequest,
    ProviderResult,
    ProviderUnavailable,
    TradingCalendarProvider,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_BASE_URL = "https://fuyao.aicubes.cn"
DEFAULT_TOKEN_ENV = "HITHINK_FINANCE_API_KEY"


@dataclass
class FuyaoProvider(
    MarketDataProvider,
    FundamentalsProvider,
    TradingCalendarProvider,
    IndexDataProvider,
):
    """REST adapter for the official HiThink/Fuyao public data contract."""

    base_url: str = DEFAULT_BASE_URL
    token_env: str = DEFAULT_TOKEN_ENV
    allow_network: bool = False
    source: str = "hithink_fuyao"
    source_reliability: float = 0.98
    timeout: float = 30.0
    max_attempts: int = 3
    _http: HttpClient | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # A-share market data
    # ------------------------------------------------------------------
    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        return self.historical_prices(request, adjust="none")

    def adjusted_prices(self, request: ProviderRequest) -> ProviderResult:
        return self.historical_prices(request, adjust="forward")

    def historical_prices(
        self,
        request: ProviderRequest,
        *,
        adjust: str = "none",
    ) -> ProviderResult:
        if adjust not in {"none", "forward", "backward"}:
            raise ValueError("adjust must be one of none/forward/backward")
        endpoint = "/api/a-share/prices/historical"
        frames: list[pd.DataFrame] = []
        for symbol in request.symbols:
            data = self._request(
                endpoint,
                params={
                    "thscode": symbol,
                    "interval": "1d",
                    "start": _date_to_ms(request.start_date),
                    "end": _date_to_ms(request.end_date),
                    "adjust": adjust,
                },
            )
            frame = _normalise_price_rows(
                _items(data),
                symbol=symbol,
                source=self.source,
                endpoint=endpoint,
                reliability=self.source_reliability,
            )
            if not frame.empty:
                frame["adjustment"] = adjust
                frames.append(frame)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return ProviderResult(
            frame=out,
            source=self.source,
            point_in_time=True,
            quality_score=self.source_reliability if not out.empty else 0.0,
            warnings=() if not out.empty else ("fuyao_empty_historical",),
            metadata={"adjust": adjust, "endpoint": endpoint},
        )

    def snapshot(self, symbols: tuple[str, ...] = ()) -> pd.DataFrame:
        params: dict[str, Any] = {}
        if symbols:
            params["thscodes"] = ",".join(symbols)
        else:
            # Never accidentally pull the whole market as an auth probe.
            params.update({"limit": 100, "offset": 0})
        data = self._request("/api/a-share/prices/snapshot", params=params)
        frame = pd.DataFrame(_items(data)).rename(
            columns={"thscode": "symbol", "turnover": "amount"}
        )
        return self._with_provenance(
            frame,
            endpoint="/api/a-share/prices/snapshot",
            available_at=_ms_to_timestamp(data.get("timestamp")),
            pit_eligible=True,
        )

    def tradability(self, request: ProviderRequest) -> ProviderResult:
        raise ProviderUnavailable(
            "Fuyao public contract does not expose a complete historical PIT "
            "tradability/ST/suspension panel. Keep QuantAgent's canonical "
            "tradability provider for this capability."
        )

    # ------------------------------------------------------------------
    # Symbol metadata
    # ------------------------------------------------------------------
    def ticker_search(
        self,
        query: str,
        *,
        exchange: str | None = None,
        asset_type: str | None = None,
        limit: int = 10,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {"q": query, "limit": limit}
        if exchange:
            params["exchange"] = exchange
        if asset_type:
            params["asset_type"] = asset_type
        data = self._request("/api/meta/tickers/search", params=params)
        return self._with_provenance(
            pd.DataFrame(_items(data)),
            endpoint="/api/meta/tickers/search",
            available_at=_ms_to_timestamp(data.get("timestamp")),
            pit_eligible=True,
        )

    def ticker_list(
        self,
        *,
        asset_type: str = "a-share",
        limit: int = 1000,
        offset: int = 0,
    ) -> pd.DataFrame:
        data = self._request(
            "/api/meta/tickers/list",
            params={"asset_type": asset_type, "limit": limit, "offset": offset},
        )
        return self._with_provenance(
            pd.DataFrame(_items(data)),
            endpoint="/api/meta/tickers/list",
            available_at=_ms_to_timestamp(data.get("timestamp")),
            pit_eligible=True,
        )

    # ------------------------------------------------------------------
    # Financials / valuation
    # ------------------------------------------------------------------
    def fundamentals(self, request: ProviderRequest) -> ProviderResult:
        """Return long-form PIT-safe income/balance/cash-flow rows.

        The upstream contract returns both ``period_end_ms`` and
        ``report_date_ms``. The former identifies the accounting period; the
        latter controls model availability.
        """
        frames: list[pd.DataFrame] = []
        for symbol in request.symbols:
            for statement, path in (
                ("income", "/api/a-share/financials/income-statements"),
                ("balance", "/api/a-share/financials/balance-sheets"),
                ("cashflow", "/api/a-share/financials/cash-flow-statements"),
            ):
                data = self._request(
                    path,
                    params={
                        "thscode": symbol,
                        "period": "quarterly",
                        "start": _date_to_ms(request.start_date),
                        "end": _date_to_ms(request.end_date),
                    },
                )
                frame = _normalise_financial_rows(
                    _items(data), statement_type=statement, source=self.source, endpoint=path
                )
                if not frame.empty:
                    frames.append(frame)
        out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        return ProviderResult(
            frame=out,
            source=self.source,
            point_in_time=True,
            quality_score=self.source_reliability if not out.empty else 0.0,
            warnings=() if not out.empty else ("fuyao_empty_fundamentals",),
            metadata={
                "pit_key": "report_date_ms",
                "report_period_key": "period_end_ms",
                "period": "quarterly",
            },
        )

    def financial_indicators(self, symbol: str, report: str) -> pd.DataFrame:
        """Flatten the five official indicator blocks for UI/research inspection.

        The indicator endpoint returns ``abilities`` rather than ``item`` and
        does *not* publish ``report_date_ms`` in the documented schema. Therefore
        these rows are marked ``pit_eligible=False`` and use retrieval time as a
        conservative ``available_at``. They must not enter historical training
        until a disclosure timestamp is joined from a PIT statement source.
        """
        endpoint = "/api/a-share/financials/indicators"
        data = self._request(endpoint, params={"thscode": symbol, "report": report})
        rows: list[dict[str, Any]] = []
        abilities = data.get("abilities", [])
        if isinstance(abilities, list):
            for block in abilities:
                if not isinstance(block, Mapping):
                    continue
                ability = str(block.get("ability") or "")
                indicators = block.get("indicators", [])
                if not isinstance(indicators, list):
                    continue
                for indicator in indicators:
                    if not isinstance(indicator, Mapping):
                        continue
                    rows.append(
                        {
                            "symbol": symbol,
                            "report": report,
                            "ability": ability,
                            "index_id": indicator.get("index_id"),
                            "value": indicator.get("value"),
                        }
                    )
        return self._with_provenance(
            pd.DataFrame(rows),
            endpoint=endpoint,
            available_at=pd.Timestamp(datetime.now(timezone.utc)),
            pit_eligible=False,
        )

    def valuations_snapshot(self, symbols: tuple[str, ...]) -> pd.DataFrame:
        endpoint = "/api/a-share/valuations/snapshot"
        data = self._request(endpoint, params={"thscodes": ",".join(symbols)})
        return self._with_provenance(
            pd.DataFrame(_items(data)),
            endpoint=endpoint,
            available_at=_ms_to_timestamp(data.get("timestamp")),
            pit_eligible=True,
        )

    # ------------------------------------------------------------------
    # Corporate actions / calendar / index
    # ------------------------------------------------------------------
    def corporate_actions(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        endpoint = "/api/a-share/corporate-actions/adjustment-factors"
        params: dict[str, Any] = {"thscode": symbol}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        data = self._request(endpoint, params=params)
        frame = pd.DataFrame(_items(data))
        if frame.empty:
            return frame
        frame["symbol"] = symbol
        frame["ex_date"] = _ms_series_to_shanghai_date(frame["ex_date_ms"])
        # The event is applied no earlier than the ex-date. This is conservative
        # for historical adjustment even if the company announced it earlier.
        frame["available_at"] = frame["ex_date"]
        return self._with_provenance(
            frame,
            endpoint=endpoint,
            preserve_available_at=True,
            pit_eligible=True,
        )

    def trading_days(self, request: ProviderRequest) -> ProviderResult:
        endpoint = "/api/a-share/calendar/trading-days"
        data = self._request(endpoint)
        frame = pd.DataFrame(_items(data))
        if not frame.empty and "date_ms" in frame.columns:
            frame["trade_date"] = _ms_series_to_shanghai_date(frame["date_ms"])
            start = pd.Timestamp(request.start_date)
            end = pd.Timestamp(request.end_date)
            frame = frame[frame["trade_date"].between(start, end)].reset_index(drop=True)
        frame = self._with_provenance(
            frame,
            endpoint=endpoint,
            available_at=_ms_to_timestamp(data.get("timestamp")),
            pit_eligible=True,
        )
        return ProviderResult(
            frame=frame,
            source=self.source,
            point_in_time=True,
            quality_score=self.source_reliability if not frame.empty else 0.0,
        )

    def index_daily(self, request: ProviderRequest) -> ProviderResult:
        endpoint = "/api/a-share-index/prices/historical"
        frames: list[pd.DataFrame] = []
        for symbol in request.symbols:
            data = self._request(
                endpoint,
                params={
                    "thscode": symbol,
                    "interval": "1d",
                    "start": _date_to_ms(request.start_date),
                    "end": _date_to_ms(request.end_date),
                },
            )
            frame = _normalise_price_rows(
                _items(data),
                symbol=symbol,
                source=self.source,
                endpoint=endpoint,
                reliability=self.source_reliability,
            )
            if not frame.empty:
                frames.append(frame)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return ProviderResult(
            out,
            self.source,
            True,
            self.source_reliability if not out.empty else 0.0,
        )

    def index_constituents(self, thscode: str) -> pd.DataFrame:
        endpoint = "/api/a-share-index/constituents/ths-stock-list"
        data = self._request(endpoint, params={"thscode": thscode})
        return self._with_provenance(
            pd.DataFrame(_items(data)),
            endpoint=endpoint,
            available_at=_ms_to_timestamp(data.get("timestamp")),
            pit_eligible=True,
        )

    # ------------------------------------------------------------------
    # Generic documented capability boundary
    # ------------------------------------------------------------------
    def get_capability(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a documented Fuyao REST capability under ``/api/``.

        This is used by the market-dump downloader and lets higher layers access
        newly documented read-only capabilities without bypassing auth/retry and
        business-error handling. It intentionally does not accept arbitrary
        hosts or non-API paths.
        """
        if not path.startswith("/api/"):
            raise ValueError("Fuyao capability path must start with /api/")
        return self._request(path, params=params)

    # ------------------------------------------------------------------
    # Transport / provenance
    # ------------------------------------------------------------------
    def _request(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.allow_network:
            raise ProviderUnavailable(
                "FuyaoProvider network access blocked: set allow_network=True explicitly."
            )
        token = os.environ.get(self.token_env, "").strip()
        if not token:
            raise ProviderUnavailable(
                f"FuyaoProvider requires {self.token_env} in the backend environment."
            )
        client = self._http or HttpClient(timeout=self.timeout, max_attempts=self.max_attempts)
        outcome = client.get_json(
            f"{self.base_url.rstrip('/')}{path}",
            params=params,
            headers={"X-api-key": token, "Accept": "application/json"},
        )
        if not outcome.ok:
            hint = (
                "check API key/capability permission"
                if outcome.retry_class == RETRY_ENTITLEMENT
                else "retry upstream request"
            )
            raise ProviderUnavailable(
                f"Fuyao request failed ({outcome.retry_class}): {hint}; "
                f"{outcome.error or ''}".strip()
            )
        envelope = outcome.payload
        if not isinstance(envelope, dict):
            raise ProviderUnavailable("Fuyao response is not an object")
        code = envelope.get("code")
        if code != 0:
            request_id = envelope.get("request_id")
            raise ProviderUnavailable(
                f"Fuyao business error code={code}: "
                f"{envelope.get('message', 'unknown error')}"
                + (f" request_id={request_id}" if request_id else "")
            )
        data = envelope.get("data")
        if not isinstance(data, dict):
            raise ProviderUnavailable("Fuyao response data is not an object")
        return data

    def _with_provenance(
        self,
        frame: pd.DataFrame,
        *,
        endpoint: str,
        available_at: pd.Timestamp | None = None,
        preserve_available_at: bool = False,
        pit_eligible: bool,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        retrieved_at = pd.Timestamp(datetime.now(timezone.utc))
        if not preserve_available_at or "available_at" not in out.columns:
            out["available_at"] = available_at if available_at is not None else retrieved_at
        out["source"] = self.source
        out["source_endpoint"] = endpoint
        out["retrieved_at"] = retrieved_at
        out["quality_status"] = "official_api"
        out["point_in_time_valid"] = bool(pit_eligible)
        out["pit_eligible"] = bool(pit_eligible)
        return out


def _items(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = data.get("item", [])
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _date_to_ms(value: str) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(SHANGHAI)
    else:
        ts = ts.tz_convert(SHANGHAI)
    return int(ts.timestamp() * 1000)


def _ms_to_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(int(value), unit="ms", utc=True)
        return ts.tz_convert(SHANGHAI).tz_localize(None)
    except (TypeError, ValueError, OverflowError):
        return None


def _ms_series_to_shanghai_date(values: pd.Series) -> pd.Series:
    """Decode epoch milliseconds to the upstream Shanghai calendar date.

    Fuyao date fields are epoch milliseconds for a local market timestamp. If
    they are converted as naive UTC and normalized first, every Shanghai
    midnight becomes the prior UTC calendar day. Convert to Asia/Shanghai first,
    then drop timezone/normalize so trade dates and disclosure dates keep their
    documented local-market meaning.
    """
    parsed = pd.to_datetime(values, unit="ms", errors="coerce", utc=True)
    return parsed.dt.tz_convert(SHANGHAI).dt.tz_localize(None).dt.normalize()


def _normalise_price_rows(
    rows: list[dict[str, Any]],
    *,
    symbol: str,
    source: str,
    endpoint: str,
    reliability: float,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).rename(
        columns={
            "open_price": "open",
            "high_price": "high",
            "low_price": "low",
            "close_price": "close",
            "turnover": "amount",
        }
    )
    frame["symbol"] = symbol
    frame["trade_date"] = _ms_series_to_shanghai_date(frame["date_ms"])
    # Conservative daily-bar contract: no same-session use in daily research.
    frame["available_at"] = frame["trade_date"] + pd.Timedelta(days=1)
    frame["source"] = source
    frame["source_endpoint"] = endpoint
    frame["source_type"] = "official_api"
    frame["source_reliability"] = reliability
    frame["retrieved_at"] = pd.Timestamp(datetime.now(timezone.utc))
    frame["quality_status"] = "official_api"
    frame["point_in_time_valid"] = True
    frame["pit_eligible"] = True
    columns = [
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "available_at",
        "source",
        "source_endpoint",
        "source_type",
        "source_reliability",
        "retrieved_at",
        "quality_status",
        "point_in_time_valid",
        "pit_eligible",
    ]
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def _normalise_financial_rows(
    rows: list[dict[str, Any]],
    *,
    statement_type: str,
    source: str,
    endpoint: str,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "report_date_ms" not in frame.columns or "period_end_ms" not in frame.columns:
        raise ProviderUnavailable(
            f"Fuyao {statement_type} response lacks report_date_ms/period_end_ms; "
            "cannot satisfy PIT contract"
        )
    frame["symbol"] = frame.get(
        "thscode", frame.get("ticker", pd.Series(index=frame.index, dtype="object"))
    )
    frame["report_period"] = _ms_series_to_shanghai_date(frame["period_end_ms"])
    frame["ann_date"] = _ms_series_to_shanghai_date(frame["report_date_ms"])
    frame["available_at"] = frame["ann_date"]
    if frame["available_at"].isna().any():
        raise ProviderUnavailable(
            f"Fuyao {statement_type} contains null disclosure dates; PIT merge blocked"
        )
    frame["statement_type"] = statement_type
    frame["source"] = source
    frame["source_endpoint"] = endpoint
    frame["retrieved_at"] = pd.Timestamp(datetime.now(timezone.utc))
    frame["quality_status"] = "official_api"
    frame["point_in_time_valid"] = True
    frame["pit_eligible"] = True
    return frame


__all__ = ["FuyaoProvider", "DEFAULT_BASE_URL", "DEFAULT_TOKEN_ENV"]
