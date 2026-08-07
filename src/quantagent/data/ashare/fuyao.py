"""Fuyao / HiThink Financial API adapter for the governed A-share foundation.

The adapter intentionally sits inside ``quantagent.data.ashare`` so Fuyao data
cannot bypass U0 provenance, point-in-time, unit, and source-boundary rules.
Credentials are read from ``HITHINK_FINANCE_API_KEY`` unless an explicit key is
injected (tests only). The key is never logged or persisted.

Official references:
- https://github.com/HiThink-Tech/Financial-API
- https://fuyao.aicubes.cn/llms-full.txt
- https://fuyao.aicubes.cn/docs/api-reference/overview/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from quantagent.data.ashare import contracts
from quantagent.data.ashare.env import load_repo_env
from quantagent.data.ashare.http import (
    RETRY_EMPTY,
    RETRY_ENTITLEMENT,
    RETRY_OK,
    RETRY_PERMANENT,
    RETRY_TRANSIENT,
    FetchOutcome,
    HttpClient,
)
from quantagent.data.ashare.sources import SourceResult
from quantagent.data.ashare.symbols import identify

FUYAO_BASE_URL = "https://fuyao.aicubes.cn"
FUYAO_API_KEY_ENV = "HITHINK_FINANCE_API_KEY"
FUYAO_HEADER = "X-api-key"

# Official REST / dump endpoints. Keep these explicit: changing a vendor path is
# then a reviewable schema change instead of a string buried inside a caller.
PRICE_SNAPSHOT = "/api/a-share/prices/snapshot"
PRICE_HISTORICAL = "/api/a-share/prices/historical"
VALUATION_SNAPSHOT = "/api/a-share/valuations/snapshot"
CORPORATE_ACTIONS = "/api/a-share/corporate-actions/adjustment-factors"
TRADING_DAYS = "/api/a-share/calendar/trading-days"
TICKER_SEARCH = "/api/meta/tickers/search"
TICKER_LIST = "/api/meta/tickers/list"
FINANCIAL_INCOME = "/api/a-share/financials/income-statements"
FINANCIAL_BALANCE = "/api/a-share/financials/balance-sheets"
FINANCIAL_CASHFLOW = "/api/a-share/financials/cash-flow-statements"
FINANCIAL_INDICATORS = "/api/a-share/financials/indicators"
LIMIT_UP_POOL = "/api/a-share/special-data/limit-up-pool"
LIMIT_UP_LADDER = "/api/a-share/special-data/limit-up-ladder"
HOT_STOCK_LIST = "/api/a-share/special-data/hot-stock-list"
SKYROCKET_LIST = "/api/a-share/special-data/skyrocket-list"
DRAGON_TIGER_LIST = "/api/a-share/special-data/dragon-tiger-list"
ANOMALY_LIST = "/api/a-share/special-data/anomaly-analysis-list"
THS_INDEX_LIST = "/api/a-share-index/catalog/ths-index-list"
THS_INDEX_CONSTITUENTS = "/api/a-share-index/constituents/ths-stock-list"
INDEX_SNAPSHOT = "/api/a-share-index/prices/snapshot"
INDEX_HISTORICAL = "/api/a-share-index/prices/historical"

MARKET_DUMP_ENDPOINTS: Mapping[str, str] = {
    "daily-k": "/api/dump/market-dumps/daily-k/download-url",
    "daily-k-10d": "/api/dump/market-dumps/daily-k-10d/download-url",
    "adjustment-factors": "/api/dump/market-dumps/adjustment-factors/download-url",
}

# Fuyao business codes are returned inside HTTP 200 envelopes. Treating HTTP 200
# as success would silently turn auth or entitlement failures into empty frames.
_AUTH_CODES = {2001, 2002, 2003, 2004}
_EMPTY_CODES = {3002, 4040}


@dataclass(frozen=True)
class FuyaoCapability:
    configured: bool
    endpoint: str
    status: str
    code: int | None = None
    message: str = ""
    rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": "fuyao",
            "configured": self.configured,
            "endpoint": self.endpoint,
            "status": self.status,
            "code": self.code,
            "message": self.message,
            "rows": self.rows,
        }


class FuyaoClient:
    """Small REST client that preserves Fuyao's business-code semantics."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = FUYAO_BASE_URL,
        client: HttpClient | None = None,
    ) -> None:
        load_repo_env()
        self.api_key = api_key if api_key is not None else os.environ.get(FUYAO_API_KEY_ENV, "")
        self.base_url = base_url.rstrip("/")
        self.client = client or HttpClient(timeout=25, max_attempts=3)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> FetchOutcome:
        url = f"{self.base_url}{path}"
        if not self.api_key:
            return FetchOutcome(
                ok=False,
                endpoint=url,
                retry_class=RETRY_ENTITLEMENT,
                error=f"{FUYAO_API_KEY_ENV} is not configured",
            )
        outcome = self.client.get_json(url, params=params, headers={FUYAO_HEADER: self.api_key})
        if not outcome.ok:
            return outcome
        payload = outcome.payload
        if not isinstance(payload, dict):
            outcome.ok = False
            outcome.retry_class = RETRY_PERMANENT
            outcome.error = "Fuyao response is not an ApiResponse object"
            return outcome
        code = payload.get("code")
        if code == 0:
            return outcome
        message = str(payload.get("message") or "")
        outcome.ok = False
        outcome.error = f"Fuyao code={code}: {message}"[:240]
        if code in _AUTH_CODES:
            outcome.retry_class = RETRY_ENTITLEMENT
        elif code in _EMPTY_CODES:
            outcome.retry_class = RETRY_EMPTY
        elif isinstance(code, int) and 5000 <= code < 6000:
            outcome.retry_class = RETRY_TRANSIENT
        else:
            outcome.retry_class = RETRY_PERMANENT
        return outcome

    def data(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return the envelope ``data`` object or raise a concise runtime error."""
        outcome = self.get(path, params=params)
        if not outcome.ok:
            raise RuntimeError(outcome.error or f"Fuyao request failed: {path}")
        data = (outcome.payload or {}).get("data")
        return data if isinstance(data, dict) else {}

    def market_dump_download_url(self, kind: str) -> dict[str, Any]:
        """Sign a short-lived full-market Parquet URL; never persist the URL."""
        if kind not in MARKET_DUMP_ENDPOINTS:
            raise ValueError(f"unknown Fuyao dump kind {kind!r}; expected {sorted(MARKET_DUMP_ENDPOINTS)}")
        return self.data(MARKET_DUMP_ENDPOINTS[kind])


class FuyaoSource:
    """Canonical U0 adapter plus generic access to Fuyao's richer datasets."""

    name = "fuyao"
    # Use windows strictly below the documented 10-year maximum. This avoids
    # leap-day / timezone edge cases at exactly ten years.
    WINDOW_YEARS = 9

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = FUYAO_BASE_URL,
        client: HttpClient | None = None,
    ) -> None:
        self.api = FuyaoClient(api_key, base_url=base_url, client=client)

    @staticmethod
    def _vendor_adjust(adjust: str) -> str:
        mapping = {
            contracts.ADJUST_NONE: "none",
            contracts.ADJUST_QFQ: "forward",
            contracts.ADJUST_HFQ: "backward",
        }
        if adjust not in mapping:
            raise ValueError(f"unsupported adjustment {adjust!r}")
        return mapping[adjust]

    @staticmethod
    def _millis(date: pd.Timestamp) -> int:
        ts = pd.Timestamp(date)
        if ts.tzinfo is None:
            ts = ts.tz_localize(contracts.TIMEZONE_CST)
        else:
            ts = ts.tz_convert(contracts.TIMEZONE_CST)
        return int(ts.timestamp() * 1000)

    @staticmethod
    def _date_from_millis(value: Any) -> pd.Timestamp:
        return (
            pd.to_datetime(int(value), unit="ms", utc=True)
            .tz_convert(contracts.TIMEZONE_CST)
            .tz_localize(None)
            .normalize()
        )

    @staticmethod
    def _empty(outcome: FetchOutcome, columns: Sequence[str], error: str | None = None) -> SourceResult:
        return SourceResult(
            frame=pd.DataFrame(columns=list(columns)),
            source="fuyao",
            endpoint=outcome.endpoint,
            retry_class=outcome.retry_class if not outcome.ok else RETRY_EMPTY,
            retrieved_at=outcome.retrieved_at,
            rows=0,
            error=error or outcome.error,
        )

    @staticmethod
    def _stamp(frame: pd.DataFrame, endpoint: str, retrieved_at: str, available_at: Any) -> pd.DataFrame:
        out = frame.copy()
        out["source"] = "fuyao"
        out["source_endpoint"] = endpoint
        out["retrieved_at"] = retrieved_at
        out["available_at"] = available_at
        out["quality_status"] = contracts.QUALITY_OK
        return out

    def daily_bars(
        self,
        symbol: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        adjust: str = contracts.ADJUST_NONE,
    ) -> SourceResult:
        """Fetch arbitrary history by stitching <=9y vendor windows.

        U0 calls this with ``adjust=none``. Chunking is deterministic and rows
        are deduplicated on ``trade_date`` so a boundary can never double-count.
        """
        ident = identify(symbol)
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        cols = list(contracts.DAILY_BARS.columns)
        if end_ts < start_ts:
            raise ValueError("end must be on or after start")

        rows: list[dict[str, Any]] = []
        requests = 0
        cursor = start_ts
        last: FetchOutcome | None = None
        endpoint = f"{self.api.base_url}{PRICE_HISTORICAL}"
        while cursor <= end_ts:
            window_end = min(end_ts, cursor + pd.DateOffset(years=self.WINDOW_YEARS))
            outcome = self.api.get(
                PRICE_HISTORICAL,
                params={
                    "thscode": ident.symbol,
                    "interval": "1d",
                    "start": self._millis(cursor),
                    "end": self._millis(window_end),
                    "adjust": self._vendor_adjust(adjust),
                    "offset": 0,
                },
            )
            requests += 1
            last = outcome
            if not outcome.ok:
                if rows and outcome.retry_class in {RETRY_EMPTY, RETRY_TRANSIENT}:
                    break
                return self._empty(outcome, cols)
            data = (outcome.payload or {}).get("data") or {}
            for item in data.get("item") or []:
                try:
                    trade_date = self._date_from_millis(item["date_ms"])
                    rows.append(
                        {
                            "symbol": ident.symbol,
                            "trade_date": trade_date,
                            "open": float(item["open_price"]),
                            "high": float(item["high_price"]),
                            "low": float(item["low_price"]),
                            "close": float(item["close_price"]),
                            "volume": float(item.get("volume") or 0.0),
                            "amount": float(item.get("turnover") or 0.0),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            cursor = window_end + pd.Timedelta(days=1)

        if last is None:
            last = FetchOutcome(False, endpoint, RETRY_EMPTY, error="no window requested")
        if not rows:
            return self._empty(last, cols, "no bars in Fuyao response")
        frame = pd.DataFrame(rows)
        frame = (
            frame.drop_duplicates(["symbol", "trade_date"], keep="last")
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        frame = frame[(frame["trade_date"] >= start_ts) & (frame["trade_date"] <= end_ts)]
        available = frame["trade_date"].dt.strftime("%Y-%m-%d") + " 15:00:00"
        frame = self._stamp(frame, endpoint, last.retrieved_at, available)
        return SourceResult(
            frame=frame[cols],
            source=self.name,
            endpoint=endpoint,
            retry_class=RETRY_OK,
            retrieved_at=last.retrieved_at,
            rows=len(frame),
            metadata={
                "adjustment": adjust,
                "requests": requests,
                "volume_unit": contracts.VOLUME_SHARES,
                "amount_unit": contracts.AMOUNT_CNY,
                "vendor": "HiThink/Fuyao",
            },
        )

    def quotes(self, symbols: Sequence[str]) -> SourceResult:
        """Fetch batch Level-1 snapshots; Fuyao currently exposes no depth book."""
        idents = [identify(symbol) for symbol in symbols]
        cols = list(contracts.QUOTES.columns)
        outcome = self.api.get(
            PRICE_SNAPSHOT,
            params={"thscodes": ",".join(ident.symbol for ident in idents)},
        )
        if not outcome.ok:
            return self._empty(outcome, cols)
        data = (outcome.payload or {}).get("data") or {}
        timestamp = data.get("timestamp")
        quote_time = self._date_from_millis(timestamp) if timestamp else pd.Timestamp.now()
        rows = []
        for item in data.get("item") or []:
            try:
                symbol = identify(str(item["thscode"])).symbol
            except (KeyError, ValueError):
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "quote_time": quote_time,
                    "last_price": item.get("last_price"),
                    "prev_close": item.get("prev_price"),
                    "open": item.get("open_price"),
                    "high": item.get("high_price"),
                    "low": item.get("low_price"),
                    "volume": item.get("volume"),
                    "amount": item.get("turnover"),
                    "bid_prices": "[]",
                    "bid_volumes": "[]",
                    "ask_prices": "[]",
                    "ask_volumes": "[]",
                    "depth_levels": 0,
                }
            )
        if not rows:
            return self._empty(outcome, cols, "no snapshot rows in Fuyao response")
        frame = pd.DataFrame(rows)
        frame = self._stamp(frame, outcome.endpoint, outcome.retrieved_at, frame["quote_time"].astype(str))
        return SourceResult(
            frame[cols], self.name, outcome.endpoint, RETRY_OK, outcome.retrieved_at, len(frame),
            metadata={"depth_levels": 0, "depth_class": "L1_no_book"},
        )

    def corporate_actions(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> SourceResult:
        """Map Fuyao's raw dividend/bonus/rights events to U0 corporate actions.

        Fuyao does not expose announcement/record dates in this endpoint, so
        ``available_at`` is conservatively set to the ex-date. This is late but
        PIT-safe: no event is usable by a model before the date we can prove.
        """
        ident = identify(symbol)
        cols = list(contracts.CORPORATE_ACTIONS.columns)
        params: dict[str, Any] = {"thscode": ident.symbol}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        outcome = self.api.get(CORPORATE_ACTIONS, params=params)
        if not outcome.ok:
            return self._empty(outcome, cols)
        data = (outcome.payload or {}).get("data") or {}
        rows: list[dict[str, Any]] = []
        for item in data.get("item") or []:
            try:
                ex_date = self._date_from_millis(item["ex_date_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            cash = float(item.get("dividend_per_share") or 0.0)
            bonus = float(item.get("per_share_bonus") or 0.0)
            rights = float(item.get("allotment_ratio") or 0.0)
            kinds = [name for name, value in (("dividend", cash), ("bonus", bonus), ("rights", rights)) if value]
            rows.append(
                {
                    "symbol": ident.symbol,
                    "ex_date": ex_date,
                    "action_type": "+".join(kinds) or "corporate_action",
                    "cash_dividend_per_share": cash,
                    "stock_dividend_ratio": bonus,
                    "rights_ratio": rights,
                    "announce_date": pd.NaT,
                    "record_date": pd.NaT,
                }
            )
        if not rows:
            return self._empty(outcome, cols, "no corporate-action rows in Fuyao response")
        frame = pd.DataFrame(rows).sort_values("ex_date").reset_index(drop=True)
        available = frame["ex_date"].dt.strftime("%Y-%m-%d")
        frame = self._stamp(frame, outcome.endpoint, outcome.retrieved_at, available)
        return SourceResult(
            frame[cols], self.name, outcome.endpoint, RETRY_OK, outcome.retrieved_at, len(frame),
            metadata={"pit_policy": "ex_date_conservative", "raw_event_stream": True},
        )

    # ---- richer Fuyao surface ---------------------------------------------
    # These return the vendor's documented data object without pretending every
    # proprietary field already belongs in a U0 canonical contract.
    def raw_get(self, path: str, **params: Any) -> dict[str, Any]:
        return self.api.data(path, params=params or None)

    def valuations(self, symbols: Sequence[str] | None = None, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if symbols:
            params["thscodes"] = ",".join(identify(symbol).symbol for symbol in symbols)
        return self.api.data(VALUATION_SNAPSHOT, params)

    def financials(self, symbol: str, statement: str, **params: Any) -> dict[str, Any]:
        endpoint = {
            "income": FINANCIAL_INCOME,
            "balance": FINANCIAL_BALANCE,
            "cashflow": FINANCIAL_CASHFLOW,
            "indicators": FINANCIAL_INDICATORS,
        }.get(statement)
        if endpoint is None:
            raise ValueError("statement must be income/balance/cashflow/indicators")
        return self.api.data(endpoint, {"thscode": identify(symbol).symbol, **params})

    def capability_probe(self, symbol: str = "600519.SH") -> FuyaoCapability:
        if not self.api.configured:
            return FuyaoCapability(False, PRICE_SNAPSHOT, "NO_CREDENTIAL", message=f"{FUYAO_API_KEY_ENV} absent")
        outcome = self.api.get(PRICE_SNAPSHOT, params={"thscodes": identify(symbol).symbol})
        payload = outcome.payload if isinstance(outcome.payload, dict) else {}
        code = payload.get("code") if isinstance(payload, dict) else None
        if not outcome.ok:
            status = "UNAUTHORIZED" if outcome.retry_class == RETRY_ENTITLEMENT else outcome.retry_class
            return FuyaoCapability(True, PRICE_SNAPSHOT, status, code=code, message=outcome.error or "")
        data = payload.get("data") or {}
        rows = len(data.get("item") or []) if isinstance(data, dict) else 0
        return FuyaoCapability(True, PRICE_SNAPSHOT, "SUPPORTED" if rows else "EMPTY_RESPONSE", code=code, rows=rows)


__all__ = [
    "FUYAO_API_KEY_ENV",
    "FUYAO_BASE_URL",
    "FUYAO_HEADER",
    "MARKET_DUMP_ENDPOINTS",
    "FuyaoCapability",
    "FuyaoClient",
    "FuyaoSource",
]
