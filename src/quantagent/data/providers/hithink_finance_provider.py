"""HiThink Finance / Fuyao A-share provider.

This adapter follows the public contracts published by HiThink-Tech's
Financial-API project and https://fuyao.aicubes.cn.  It intentionally keeps two
roles separate:

* HiThink Finance is the authoritative bulk/daily/fundamental/reference source.
  Full-market history should be obtained from Market Dumps instead of issuing
  thousands of per-symbol historical requests.
* TickFlow remains QuantAgent's minute/tick/depth source because those
  capabilities are not part of the current public HiThink Finance API.

Security and PIT invariants
---------------------------
* Credentials are read from environment variables only.  The canonical name is
  ``HITHINK_FINANCE_API_KEY``; ``FUYAO_API_KEY``/``FUYAO_TOKEN`` are accepted as
  local legacy aliases.  Keys are never accepted as function arguments and are
  never included in errors, metadata or logs.
* Network access is opt-in via ``allow_network=True``.
* Fuyao business errors arrive in a HTTP-200 envelope.  Non-zero ``code`` values
  are therefore checked explicitly; authentication/entitlement errors never
  degrade into an empty DataFrame.
* Daily bars are tagged ``available_at = trade_date + 1 day`` so a same-day
  close can never be consumed by a same-day decision in QuantAgent.  This is a
  conservative execution-time convention consistent with the existing
  TickFlow provider.

The low-level ``capability`` method exposes the complete public REST surface via
an allow-list, while the MarketDataProvider methods provide canonical frames for
existing QuantAgent pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any, Mapping

import pandas as pd
import requests

from quantagent.data.providers.base import (
    MarketDataProvider,
    ProviderRequest,
    ProviderResult,
    ProviderUnavailable,
)


BASE_URL = "https://fuyao.aicubes.cn"
PRIMARY_TOKEN_ENV = "HITHINK_FINANCE_API_KEY"
LEGACY_TOKEN_ENVS = ("FUYAO_API_KEY", "FUYAO_TOKEN")
RETRY_CODES = frozenset({4001, 5001, 5002, 5003})
NO_RETRY_CODES = frozenset({1001, 1002, 1003, 1004, 2001, 2003, 3001, 3004})

# Public endpoints documented by llms-full.txt / Financial-API.  A generic
# arbitrary URL escape hatch is deliberately not provided: web jobs may only
# call reviewed finance capabilities on the configured host.
CAPABILITY_PATHS: dict[str, str] = {
    "prices_snapshot": "/api/a-share/prices/snapshot",
    "prices_historical": "/api/a-share/prices/historical",
    "ticker_search": "/api/meta/tickers/search",
    "ticker_list": "/api/meta/tickers/list",
    "corporate_actions": "/api/a-share/corporate-actions",
    "financials_income": "/api/a-share/financials/income-statements",
    "financials_balance": "/api/a-share/financials/balance-sheets",
    "financials_cashflow": "/api/a-share/financials/cash-flow-statements",
    "financial_indicators": "/api/a-share/financials/indicators",
    "calendar": "/api/a-share/calendar/trading-days",
    "valuations_snapshot": "/api/a-share/valuations/snapshot",
    "limit_up_pool": "/api/a-share/special-data/limit-up-pool",
    "limit_up_ladder": "/api/a-share/special-data/limit-up-ladder",
    "anomaly_analysis_list": "/api/a-share/special-data/anomaly-analysis-list",
    "anomaly_analysis_stock": "/api/a-share/special-data/anomaly-analysis-stock",
    "skyrocket_list": "/api/a-share/special-data/skyrocket-list",
    "hot_stock_list": "/api/a-share/special-data/hot-stock-list",
    "hot_stock_list_history": "/api/a-share/special-data/hot-stock-list-history",
    "hot_stock_rank_trend": "/api/a-share/special-data/hot-stock-rank-trend",
    "dragon_tiger_list": "/api/a-share/special-data/dragon-tiger-list",
    "index_catalog": "/api/a-share-index/catalog",
    "index_constituents": "/api/a-share-index/constituents",
    "index_snapshot": "/api/a-share-index/prices/snapshot",
    "index_historical": "/api/a-share-index/prices/historical",
    "fund_profile": "/api/fund/profile",
    "fund_holdings": "/api/fund/holdings",
    "fund_nav": "/api/fund/nav",
    "fund_returns": "/api/fund/returns",
    "fund_holders": "/api/fund/holders",
    "fund_snapshot": "/api/fund/prices/snapshot",
    "fund_historical": "/api/fund/prices/historical",
}

# Current Financial-API repository documents /api/dump/... .  Older website
# pages exposed /dump/...; the fallback keeps the integration tolerant to that
# documented migration without allowing an arbitrary path.
MARKET_DUMP_PATHS: dict[str, tuple[str, ...]] = {
    "daily-k": (
        "/api/dump/market-dumps/daily-k/download-url",
        "/dump/market-dumps/daily-k/download-url",
    ),
    "daily-k-10d": (
        "/api/dump/market-dumps/daily-k-10d/download-url",
        "/dump/market-dumps/daily-k-10d/download-url",
    ),
    "adjustment-factors": (
        "/api/dump/market-dumps/adjustment-factors/download-url",
        "/dump/market-dumps/adjustment-factors/download-url",
    ),
}

CANONICAL_OHLCV_COLUMNS: tuple[str, ...] = (
    "symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
    "available_at", "source", "source_type", "source_reliability", "point_in_time_valid",
)


class HithinkFinanceApiError(ProviderUnavailable):
    """Fuyao business-envelope error with safe diagnostic metadata."""

    def __init__(self, code: int, message: str, request_id: str | None = None) -> None:
        self.code = int(code)
        self.request_id = request_id
        safe_message = str(message or "upstream business error")[:300]
        suffix = f" request_id={request_id}" if request_id else ""
        super().__init__(f"hithink finance code={self.code}: {safe_message}{suffix}")


@dataclass
class HithinkFinanceProvider(MarketDataProvider):
    """QuantAgent adapter for HiThink Finance (Fuyao)."""

    base_url: str = BASE_URL
    token_env: str = PRIMARY_TOKEN_ENV
    source: str = "hithink_finance"
    source_reliability: float = 0.98
    allow_network: bool = False
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_base_seconds: float = 1.0
    _session: requests.Session | None = field(default=None, init=False, repr=False, compare=False)

    def _token(self) -> str:
        names = (self.token_env, *LEGACY_TOKEN_ENVS)
        for name in names:
            value = os.getenv(name)
            if value and value.strip():
                return value.strip()
        raise ProviderUnavailable(
            "HiThink Finance credential missing: set HITHINK_FINANCE_API_KEY in the server .env. "
            "Create/manage the key at https://fuyao.aicubes.cn/admin."
        )

    def _require_network(self) -> None:
        if not self.allow_network:
            raise ProviderUnavailable(
                "HithinkFinanceProvider network access is disabled; construct with allow_network=True"
            )

    def _http(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Accept": "application/json",
                "User-Agent": "QuantAgent/0.4 hithink-finance-adapter",
            })
        return self._session

    def _get_path(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._require_network()
        headers = {"X-api-key": self._token()}
        clean = {key: value for key, value in dict(params or {}).items() if value is not None}
        url = f"{self.base_url.rstrip('/')}{path}"
        last_network_error: Exception | None = None
        attempts = max(1, int(self.max_retries))
        for attempt in range(attempts):
            try:
                response = self._http().get(url, params=clean, headers=headers, timeout=self.timeout_seconds)
                response.raise_for_status()
                payload = response.json()
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_network_error = exc
                if attempt + 1 >= attempts:
                    break
                time.sleep(self.retry_base_seconds * (2**attempt))
                continue
            except requests.RequestException as exc:
                raise ProviderUnavailable(f"hithink finance HTTP failure: {type(exc).__name__}") from exc
            except ValueError as exc:
                raise ProviderUnavailable("hithink finance returned malformed JSON") from exc

            if not isinstance(payload, dict):
                raise ProviderUnavailable("hithink finance returned a non-object response envelope")
            code = int(payload.get("code", -1))
            if code == 0:
                data = payload.get("data")
                return data if isinstance(data, dict) else {"item": data}
            if code in RETRY_CODES and attempt + 1 < attempts:
                time.sleep(self.retry_base_seconds * (2**attempt))
                continue
            raise HithinkFinanceApiError(
                code=code,
                message=str(payload.get("message") or ""),
                request_id=str(payload.get("request_id")) if payload.get("request_id") else None,
            )
        if last_network_error is not None:
            raise ProviderUnavailable(
                f"hithink finance network failure after {attempts} attempts: "
                f"{type(last_network_error).__name__}"
            ) from last_network_error
        raise ProviderUnavailable("hithink finance request failed without a response")

    def capability(self, name: str, **params: Any) -> dict[str, Any]:
        """Call one reviewed public REST capability and return its ``data`` payload."""
        try:
            path = CAPABILITY_PATHS[name]
        except KeyError as exc:
            raise ValueError(f"unsupported HiThink Finance capability: {name}") from exc
        return self._get_path(path, params)

    # ------------------------------------------------------------------
    # Canonical market-data provider contract
    # ------------------------------------------------------------------

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        frame = self._historical_frame(request, adjust="none")
        return ProviderResult(
            frame=frame,
            source=self.source,
            point_in_time=True,
            quality_score=self.source_reliability if not frame.empty else 0.0,
            warnings=() if not frame.empty else ("hithink_finance_empty_daily_ohlcv",),
            metadata={"provider": "hithink finance", "adjust": "none", "interval": "1d"},
        )

    def adjusted_prices(self, request: ProviderRequest) -> ProviderResult:
        frame = self._historical_frame(request, adjust="forward")
        return ProviderResult(
            frame=frame,
            source=self.source,
            point_in_time=True,
            quality_score=self.source_reliability if not frame.empty else 0.0,
            warnings=() if not frame.empty else ("hithink_finance_empty_adjusted",),
            metadata={"provider": "hithink finance", "adjust": "forward", "interval": "1d"},
        )

    def tradability(self, request: ProviderRequest) -> ProviderResult:
        # Fuyao has rich current/special data, but the public contract does not
        # expose a complete historical suspension + ST-status panel.  Returning
        # fabricated False flags would be a PIT/data-quality bug.  Let the
        # router use TickFlow/canonical silver tradability instead.
        raise ProviderUnavailable(
            "HiThink Finance public API does not provide a complete historical tradability panel; "
            "use TickFlow/canonical silver tradability for suspension/ST/limit flags"
        )

    def _historical_frame(self, request: ProviderRequest, *, adjust: str) -> pd.DataFrame:
        symbols = tuple(dict.fromkeys(str(item).strip().upper() for item in request.symbols if str(item).strip()))
        if not symbols:
            return pd.DataFrame(columns=CANONICAL_OHLCV_COLUMNS)
        start_ms = _date_to_shanghai_ms(request.start_date)
        end_ms = _date_to_shanghai_ms(request.end_date)
        if end_ms < start_ms:
            raise ValueError("end_date must be >= start_date")
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            data = self.capability(
                "prices_historical",
                thscode=symbol,
                start=start_ms,
                end=end_ms,
                interval="1d",
                adjust=adjust,
            )
            items = data.get("item") or []
            if not isinstance(items, list) or not items:
                continue
            part = pd.DataFrame(items)
            part["symbol"] = symbol
            frames.append(part)
        if not frames:
            return pd.DataFrame(columns=CANONICAL_OHLCV_COLUMNS)
        return _normalise_daily(pd.concat(frames, ignore_index=True), source=self.source,
                                source_reliability=self.source_reliability)

    # ------------------------------------------------------------------
    # Full public surface helpers
    # ------------------------------------------------------------------

    def ticker_list(self, *, asset_type: str = "a-share", exchange: str = "SH,SZ,BJ") -> pd.DataFrame:
        """Fetch the complete ticker catalog with bounded pagination."""
        rows: list[dict[str, Any]] = []
        offset = 0
        limit = 10_000
        while True:
            data = self.capability("ticker_list", asset_type=asset_type, exchange=exchange,
                                   limit=limit, offset=offset)
            items = data.get("item") or []
            if not isinstance(items, list) or not items:
                break
            rows.extend(item for item in items if isinstance(item, dict))
            if len(items) < limit:
                break
            offset += len(items)
        return pd.DataFrame(rows)

    def snapshot(self, symbols: tuple[str, ...] = ()) -> pd.DataFrame:
        """Fetch selected symbols or, when empty, the paged full A-share snapshot."""
        rows: list[dict[str, Any]] = []
        if symbols:
            data = self.capability("prices_snapshot", thscodes=",".join(symbols))
            return pd.DataFrame(data.get("item") or [])
        offset = 0
        limit = 1000
        while True:
            data = self.capability("prices_snapshot", limit=limit, offset=offset)
            items = data.get("item") or []
            if not isinstance(items, list) or not items:
                break
            rows.extend(item for item in items if isinstance(item, dict))
            if len(items) < limit:
                break
            offset += len(items)
        return pd.DataFrame(rows)

    def financials(self, symbol: str, statement: str, *, period: str = "annual",
                   limit: int = 20) -> pd.DataFrame:
        mapping = {
            "income": "financials_income",
            "balance": "financials_balance",
            "cashflow": "financials_cashflow",
        }
        try:
            capability = mapping[statement]
        except KeyError as exc:
            raise ValueError("statement must be income, balance or cashflow") from exc
        data = self.capability(capability, thscode=symbol, period=period, limit=limit)
        return pd.DataFrame(data.get("item") or [])

    def financial_indicators(self, symbol: str, **params: Any) -> pd.DataFrame:
        data = self.capability("financial_indicators", thscode=symbol, **params)
        return pd.DataFrame(data.get("item") or [])

    def valuations_snapshot(self, symbols: tuple[str, ...] = ()) -> pd.DataFrame:
        params: dict[str, Any] = {}
        if symbols:
            params["thscodes"] = ",".join(symbols)
        data = self.capability("valuations_snapshot", **params)
        return pd.DataFrame(data.get("item") or [])

    def signed_dump_url(self, dump_kind: str) -> str:
        """Return a short-lived Market Dump URL without persisting it."""
        try:
            paths = MARKET_DUMP_PATHS[dump_kind]
        except KeyError as exc:
            raise ValueError(f"unsupported dump kind: {dump_kind}") from exc
        last_error: Exception | None = None
        for path in paths:
            try:
                data = self._get_path(path)
            except ProviderUnavailable as exc:
                # Path migration fallback only. Authentication/entitlement
                # errors must not be hidden by trying another URL spelling.
                if isinstance(exc, HithinkFinanceApiError) and exc.code in {2001, 2003, 2002, 2004}:
                    raise
                last_error = exc
                continue
            url = data.get("presigned_url") or data.get("download_url") or data.get("url")
            if isinstance(url, str) and url.startswith(("https://", "http://")):
                return url
            last_error = ProviderUnavailable("market dump signer returned no presigned_url")
        raise ProviderUnavailable(f"unable to sign market dump {dump_kind}: {last_error}")

    def download_market_dump(self, dump_kind: str, output: str | Path) -> Path:
        """Sign and immediately stream a Market Dump Parquet to ``output``."""
        self._require_network()
        url = self.signed_dump_url(dump_kind)
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with self._http().get(url, stream=True, timeout=max(120.0, self.timeout_seconds)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None


def _date_to_shanghai_ms(value: str) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Shanghai")
    else:
        stamp = stamp.tz_convert("Asia/Shanghai")
    return int(stamp.normalize().timestamp() * 1000)


def _normalise_daily(raw: pd.DataFrame, *, source: str, source_reliability: float) -> pd.DataFrame:
    frame = raw.copy()
    required = {
        "symbol", "date_ms", "open_price", "high_price", "low_price", "close_price", "volume", "turnover",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ProviderUnavailable(f"hithink historical payload missing columns: {sorted(missing)}")
    frame["trade_date"] = (
        pd.to_datetime(frame["date_ms"], unit="ms", utc=True, errors="coerce")
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    frame = frame.rename(columns={
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
        "close_price": "close",
        "turnover": "amount",
    })
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["available_at"] = frame["trade_date"] + pd.Timedelta(days=1)
    frame["source"] = source
    frame["source_type"] = "vendor_api"
    frame["source_reliability"] = float(source_reliability)
    frame["point_in_time_valid"] = True
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame.dropna(subset=["trade_date", "symbol", "close"])
    frame = frame.drop_duplicates(["symbol", "trade_date"], keep="last")
    return frame.loc[:, CANONICAL_OHLCV_COLUMNS].sort_values(["trade_date", "symbol"]).reset_index(drop=True)


# Backward-compatible technical name for callers that think of the service as Fuyao.
FuyaoProvider = HithinkFinanceProvider


__all__ = [
    "BASE_URL",
    "CAPABILITY_PATHS",
    "HithinkFinanceApiError",
    "HithinkFinanceProvider",
    "FuyaoProvider",
    "MARKET_DUMP_PATHS",
]
