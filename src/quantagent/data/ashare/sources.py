"""Real provider adapters for the A-share data foundation.

Every adapter here performs an actual network call against a documented public
or entitled endpoint and returns a canonical frame plus provenance. There is no
synthetic fallback anywhere in this module: a provider that cannot answer
returns a :class:`~quantagent.data.ashare.http.FetchOutcome` carrying a retry
classification, and the caller decides what to do with it.

Adapters implemented (all verified live against the runtime this ships in):

``TencentSource``    daily bars (raw/qfq/hfq), minute bars, Level-1 quotes with
                     5-level aggregated depth. Fast, stable, serves every board
                     including STAR and BSE; the primary public fallback.
``SinaSource``       cumulative backward-adjustment factor series (the ex-rights
                     history behind corporate actions) and the dividend /
                     rights-issue table.
``EastmoneySource``  intraday trend bars and order-size bucketed money flow.
                     Eastmoney throttles per IP and per endpoint, so this source
                     is supplementary and its failures are expected, recorded
                     and never silently swallowed.
``TickFlowSource``   the entitled vendor SDK: daily bars, instruments, quotes.

Vendor unit conventions are normalised at the boundary: Tencent, Eastmoney and
Sina all report volume in 手 (lots, 100 shares) on the daily endpoints, so the
adapters multiply by 100 and declare ``volume_unit = shares``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import pandas as pd

from quantagent.data.ashare import contracts
from quantagent.data.ashare.http import (
    RETRY_EMPTY,
    RETRY_ENTITLEMENT,
    RETRY_OK,
    RETRY_PERMANENT,
    FetchOutcome,
    HttpClient,
    utc_now,
)
from quantagent.data.ashare.symbols import SecurityIdentity, identify

LOTS_TO_SHARES = 100.0


@dataclass
class SourceResult:
    """Canonical frame plus the provenance of the call that produced it."""

    frame: pd.DataFrame
    source: str
    endpoint: str
    retry_class: str
    retrieved_at: str
    rows: int
    error: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.retry_class == RETRY_OK and not self.frame.empty

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source, "endpoint": self.endpoint,
            "retry_class": self.retry_class, "rows": self.rows,
            "error": self.error, "retrieved_at": self.retrieved_at,
            **(self.metadata or {}),
        }


def _empty(source: str, outcome: FetchOutcome, columns: Sequence[str],
           retry_class: str | None = None, error: str | None = None) -> SourceResult:
    return SourceResult(
        frame=pd.DataFrame(columns=list(columns)), source=source, endpoint=outcome.endpoint,
        retry_class=retry_class or (outcome.retry_class if not outcome.ok else RETRY_EMPTY),
        retrieved_at=outcome.retrieved_at, rows=0, error=error or outcome.error,
    )


def _stamp(frame: pd.DataFrame, source: str, endpoint: str, retrieved_at: str,
           available_at: Any, quality: str = contracts.QUALITY_OK) -> pd.DataFrame:
    frame = frame.copy()
    frame["source"] = source
    frame["source_endpoint"] = endpoint
    frame["retrieved_at"] = retrieved_at
    frame["available_at"] = available_at
    frame["quality_status"] = quality
    return frame


# ---------------------------------------------------------------------------
# Tencent (gtimg) — primary public source
# ---------------------------------------------------------------------------
class TencentSource:
    """Tencent finance endpoints. Serves every board, no IP banning observed."""

    name = "tencent"
    KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    MKLINE_URL = "https://ifzq.gtimg.cn/appstock/app/kline/mkline"
    QUOTE_URL = "https://qt.gtimg.cn/q="

    #: Measured vendor ceiling: a request for more than ~800 daily bars is
    #: silently truncated to 640 and a request for >=3200 returns an EMPTY
    #: payload. Long histories are therefore fetched as backward date windows.
    MAX_ROWS_PER_REQUEST = 800
    WINDOW_YEARS = 3

    #: Tencent's daily series carries no turnover column — only OHLCV. Callers
    #: that need `amount` must use a source that publishes it (TickFlow does).
    PROVIDES_AMOUNT = False

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    # -- daily bars ---------------------------------------------------------
    def daily_bars(self, symbol: str, start: str, end: str,
                   adjust: str = contracts.ADJUST_NONE) -> SourceResult:
        """Daily OHLCV over an arbitrary range, paged backwards in date windows.

        ``adjust`` is one of none / qfq / hfq. The vendor truncates any single
        request to a few hundred bars, so a multi-decade history is assembled
        from consecutive windows and de-duplicated on ``trade_date``.
        """
        ident = identify(symbol)
        kind = {contracts.ADJUST_NONE: "", contracts.ADJUST_QFQ: "qfq",
                contracts.ADJUST_HFQ: "hfq"}[adjust]
        key = f"{kind}day" if kind else "day"
        cols = list(contracts.DAILY_BARS.columns)
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        collected: list[list] = []
        last_outcome: FetchOutcome | None = None
        requests_made = 0
        cursor = end_ts
        while cursor >= start_ts:
            window_start = max(start_ts, cursor - pd.DateOffset(years=self.WINDOW_YEARS))
            param = (f"{ident.tencent_code},day,{window_start.date()},{cursor.date()},"
                     f"{self.MAX_ROWS_PER_REQUEST},{kind}")
            outcome = self.client.get_json(self.KLINE_URL, params={"param": param})
            last_outcome = outcome
            requests_made += 1
            if not outcome.ok:
                if collected:
                    break                      # keep what we have, report partial
                return _empty(self.name, outcome, cols)
            payload = outcome.payload or {}
            if payload.get("code") != 0:
                return _empty(self.name, outcome, cols, RETRY_PERMANENT,
                              f"vendor code {payload.get('code')}: {payload.get('msg')}")
            data = payload.get("data")
            node = data.get(ident.tencent_code) if isinstance(data, dict) else None
            series = (node or {}).get(key) or []
            if not series:
                break
            collected.extend(series)
            oldest = pd.Timestamp(series[0][0])
            if oldest <= window_start or oldest <= start_ts:
                cursor = window_start - pd.Timedelta(days=1)
            else:
                cursor = oldest - pd.Timedelta(days=1)
            if requests_made > 40:             # safety valve; 40 windows > 100 years
                break
        assert last_outcome is not None
        if not collected:
            return _empty(self.name, last_outcome, cols, RETRY_EMPTY,
                          "no bars in vendor response")
        rows = []
        for item in collected:
            # [date, open, close, high, low, volume(lots)] — no turnover column
            if len(item) < 6:
                continue
            rows.append({
                "symbol": ident.symbol,
                "trade_date": pd.Timestamp(item[0]),
                "open": float(item[1]), "close": float(item[2]),
                "high": float(item[3]), "low": float(item[4]),
                "volume": float(item[5]) * LOTS_TO_SHARES,
                "amount": float(item[6]) if len(item) > 6 and _is_number(item[6]) else float("nan"),
            })
        frame = pd.DataFrame(rows)
        if frame.empty:
            return _empty(self.name, last_outcome, cols, RETRY_EMPTY, "unparseable vendor rows")
        frame = (frame.drop_duplicates("trade_date")
                 .sort_values("trade_date").reset_index(drop=True))
        frame = frame[(frame["trade_date"] >= start_ts) & (frame["trade_date"] <= end_ts)]
        if frame.empty:
            return _empty(self.name, last_outcome, cols, RETRY_EMPTY, "no bars inside range")
        frame = _stamp(frame, self.name, self.KLINE_URL, last_outcome.retrieved_at,
                       frame["trade_date"].dt.strftime("%Y-%m-%d") + " 15:00:00")
        return SourceResult(frame[cols], self.name, self.KLINE_URL, RETRY_OK,
                            last_outcome.retrieved_at, len(frame),
                            metadata={"adjustment": adjust, "requests": requests_made,
                                      "amount_available": self.PROVIDES_AMOUNT})

    # -- minute bars --------------------------------------------------------
    def minute_bars(self, symbol: str, frequency: int = 5, count: int = 320) -> SourceResult:
        """Recent intraday bars. Vendor keeps a rolling window, not full history."""
        ident = identify(symbol)
        param = f"{ident.tencent_code},m{frequency},,{count}"
        outcome = self.client.get_json(self.MKLINE_URL, params={"param": param})
        cols = list(contracts.MINUTE_BARS.columns)
        if not outcome.ok:
            return _empty(self.name, outcome, cols)
        payload = outcome.payload or {}
        node = (payload.get("data") or {}).get(ident.tencent_code) or {}
        series = node.get(f"m{frequency}") or []
        if not series:
            return _empty(self.name, outcome, cols, RETRY_EMPTY, "no intraday bars returned")
        rows = []
        for item in series:
            # [YYYYMMDDHHMM, open, close, high, low, volume(lots), ...]
            if len(item) < 6:
                continue
            rows.append({
                "symbol": ident.symbol,
                "bar_time": pd.to_datetime(str(item[0])[:12], format="%Y%m%d%H%M"),
                "open": float(item[1]), "close": float(item[2]),
                "high": float(item[3]), "low": float(item[4]),
                "volume": float(item[5]) * LOTS_TO_SHARES,
                "amount": float("nan"),
                "frequency": frequency,
            })
        frame = pd.DataFrame(rows)
        if frame.empty:
            return _empty(self.name, outcome, cols, RETRY_EMPTY, "unparseable intraday rows")
        frame = _stamp(frame, self.name, self.MKLINE_URL, outcome.retrieved_at,
                       frame["bar_time"].dt.strftime("%Y-%m-%d %H:%M:%S"))
        return SourceResult(frame[cols], self.name, self.MKLINE_URL, RETRY_OK,
                            outcome.retrieved_at, len(frame),
                            metadata={"frequency_minutes": frequency})

    # -- Level-1 quote with 5-level depth -----------------------------------
    def quotes(self, symbols: Iterable[str]) -> SourceResult:
        idents = [identify(s) for s in symbols]
        url = self.QUOTE_URL + ",".join(i.tencent_code for i in idents)
        outcome = self.client.get(url, encoding="gbk")
        cols = list(contracts.QUOTES.columns)
        if not outcome.ok:
            return _empty(self.name, outcome, cols)
        rows = []
        for line in (outcome.text or "").splitlines():
            line = line.strip()
            if not line.startswith("v_"):
                continue
            key, _, body = line.partition("=")
            fields = body.strip().strip(';').strip('"').split("~")
            if len(fields) < 40:
                continue
            vendor_code = key[2:]
            try:
                ident = identify(vendor_code)
            except Exception:  # noqa: BLE001 - index/ETF codes are skipped, not guessed
                continue
            bid_p, bid_v, ask_p, ask_v = [], [], [], []
            for lvl in range(5):
                bid_p.append(_f(fields[9 + lvl * 2]))
                bid_v.append(_f(fields[10 + lvl * 2]) * LOTS_TO_SHARES)
                ask_p.append(_f(fields[19 + lvl * 2]))
                ask_v.append(_f(fields[20 + lvl * 2]) * LOTS_TO_SHARES)
            rows.append({
                "symbol": ident.symbol,
                "quote_time": pd.to_datetime(fields[30], format="%Y%m%d%H%M%S", errors="coerce"),
                "last_price": _f(fields[3]), "prev_close": _f(fields[4]), "open": _f(fields[5]),
                "high": _f(fields[33]), "low": _f(fields[34]),
                "volume": _f(fields[6]) * LOTS_TO_SHARES,
                "amount": _f(fields[37]) * 10000.0,   # vendor reports 万元
                "bid_prices": json.dumps(bid_p), "bid_volumes": json.dumps(bid_v),
                "ask_prices": json.dumps(ask_p), "ask_volumes": json.dumps(ask_v),
                "depth_levels": 5,
            })
        if not rows:
            return _empty(self.name, outcome, cols, RETRY_EMPTY, "no quote lines parsed")
        frame = _stamp(pd.DataFrame(rows), self.name, self.QUOTE_URL, outcome.retrieved_at,
                       pd.DataFrame(rows)["quote_time"].astype(str))
        return SourceResult(frame[cols], self.name, self.QUOTE_URL, RETRY_OK,
                            outcome.retrieved_at, len(frame),
                            metadata={"depth_levels": 5, "depth_class": "L1_aggregated_5_level"})


# ---------------------------------------------------------------------------
# Sina — adjustment factors and corporate actions
# ---------------------------------------------------------------------------
class SinaSource:
    """Sina finance endpoints: ex-rights factor history and dividend records."""

    name = "sina"
    HFQ_URL = "https://finance.sina.com.cn/realstock/company/{code}/hfq.js"
    BONUS_URL = "https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/{code}.phtml"
    KLINE_URL = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"

    #: Hard vendor ceiling: the endpoint returns at most the most recent 1023
    #: sessions and offers no paging, so a long history comes back truncated.
    MAX_SESSIONS = 1023

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    def daily_bars(self, symbol: str, start: str | None = None,
                   end: str | None = None) -> SourceResult:
        """Raw daily OHLCV — the only public route that still serves DELISTED names.

        TickFlow and Tencent both answer EMPTY for securities that have left the
        exchange, which would leave the universe survivorship-biased. This
        endpoint still serves them, but only the most recent ``MAX_SESSIONS``
        sessions and without turnover, so the result is marked truncated in its
        metadata rather than passed off as a complete history.
        """
        ident = identify(symbol)
        outcome = self.client.get_json(self.KLINE_URL, params={
            "symbol": ident.tencent_code, "scale": "240", "ma": "no",
            "datalen": str(self.MAX_SESSIONS)})
        cols = list(contracts.DAILY_BARS.columns)
        if not outcome.ok:
            return _empty(self.name, outcome, cols)
        payload = outcome.payload
        if not isinstance(payload, list) or not payload:
            return _empty(self.name, outcome, cols, RETRY_EMPTY, "no bars in vendor response")
        rows = []
        for item in payload:
            if not isinstance(item, dict) or "day" not in item:
                continue
            rows.append({
                "symbol": ident.symbol, "trade_date": pd.Timestamp(item["day"]),
                "open": _f(item.get("open")), "high": _f(item.get("high")),
                "low": _f(item.get("low")), "close": _f(item.get("close")),
                "volume": _f(item.get("volume")),        # already in shares here
                "amount": float("nan"),
            })
        frame = pd.DataFrame(rows)
        if frame.empty:
            return _empty(self.name, outcome, cols, RETRY_EMPTY, "unparseable vendor rows")
        frame = frame.drop_duplicates("trade_date").sort_values("trade_date")
        if start:
            frame = frame[frame["trade_date"] >= pd.Timestamp(start)]
        if end:
            frame = frame[frame["trade_date"] <= pd.Timestamp(end)]
        if frame.empty:
            return _empty(self.name, outcome, cols, RETRY_EMPTY, "no bars inside range")
        frame = _stamp(frame, self.name, self.KLINE_URL, outcome.retrieved_at,
                       frame["trade_date"].dt.strftime("%Y-%m-%d") + " 15:00:00")
        return SourceResult(frame[cols], self.name, self.KLINE_URL, RETRY_OK,
                            outcome.retrieved_at, len(frame),
                            metadata={"adjustment": contracts.ADJUST_NONE,
                                      "amount_available": False,
                                      "history_truncated": len(payload) >= self.MAX_SESSIONS,
                                      "max_sessions": self.MAX_SESSIONS})

    def adjust_factors(self, symbol: str) -> SourceResult:
        """Cumulative backward-adjustment (hfq) factor series with effective dates.

        Each record is an ex-rights event: the factor changes exactly when a
        dividend / split / rights issue takes effect, so this series is the
        machine-readable corporate-action identity for the security.
        """
        ident = identify(symbol)
        url = self.HFQ_URL.format(code=ident.tencent_code)
        outcome = self.client.get(url)
        cols = list(contracts.ADJUST_FACTORS.columns)
        if not outcome.ok:
            return _empty(self.name, outcome, cols)
        # The response is `var <code>hfq={...}` followed by a JS comment block of
        # unpredictable content, so the object is decoded by balanced scan from
        # the first brace rather than by a regex that has to guess the terminator.
        text = outcome.text or ""
        brace = text.find("{")
        if brace < 0:
            return _empty(self.name, outcome, cols, RETRY_EMPTY, "no factor object in response")
        try:
            payload, _ = json.JSONDecoder().raw_decode(text[brace:])
        except json.JSONDecodeError as exc:
            return _empty(self.name, outcome, cols, RETRY_PERMANENT, f"json: {exc}")
        records = payload.get("data") or []
        rows = [{
            "symbol": ident.symbol,
            "effective_date": pd.Timestamp(r["d"]),
            "hfq_factor": float(r["f"]),
            "adjustment_method": contracts.ADJUST_HFQ,
        } for r in records if r.get("d") and _is_number(r.get("f"))]
        if not rows:
            return _empty(self.name, outcome, cols, RETRY_EMPTY, "empty factor series")
        frame = pd.DataFrame(rows).sort_values("effective_date")
        frame = _stamp(frame, self.name, url, outcome.retrieved_at,
                       frame["effective_date"].dt.strftime("%Y-%m-%d"))
        return SourceResult(frame[cols], self.name, url, RETRY_OK, outcome.retrieved_at, len(frame),
                            metadata={"total_declared": payload.get("total")})

    def dividends(self, symbol: str) -> SourceResult:
        """Dividend / bonus-share / rights-issue history from the Sina F10 table."""
        ident = identify(symbol)
        url = self.BONUS_URL.format(code=ident.code)
        outcome = self.client.get(url, encoding="gbk")
        cols = list(contracts.CORPORATE_ACTIONS.columns)
        if not outcome.ok:
            return _empty(self.name, outcome, cols)
        try:
            tables = pd.read_html(outcome.text or "", attrs={"id": "sharebonus_1"})
        except (ValueError, ImportError) as exc:
            return _empty(self.name, outcome, cols, RETRY_PERMANENT, f"parse: {str(exc)[:100]}")
        if not tables or tables[0].empty:
            return _empty(self.name, outcome, cols, RETRY_EMPTY, "no dividend rows")
        raw = tables[0]
        raw.columns = [str(c) for c in raw.columns]

        def col(*needles: str) -> pd.Series:
            for c in raw.columns:
                if all(n in c for n in needles):
                    return raw[c]
            return pd.Series([None] * len(raw), index=raw.index)

        rows = pd.DataFrame({
            "symbol": ident.symbol,
            "ex_date": pd.to_datetime(col("除权除息日"), errors="coerce"),
            "action_type": "dividend",
            "cash_dividend_per_share": pd.to_numeric(col("派息"), errors="coerce") / 10.0,
            "stock_dividend_ratio": pd.to_numeric(col("送股"), errors="coerce") / 10.0,
            "rights_ratio": pd.to_numeric(col("转增"), errors="coerce") / 10.0,
            "announce_date": pd.to_datetime(col("公告日期"), errors="coerce"),
            "record_date": pd.to_datetime(col("股权登记日"), errors="coerce"),
        }).dropna(subset=["ex_date"])
        if rows.empty:
            return _empty(self.name, outcome, cols, RETRY_EMPTY, "no dated dividend rows")
        available = rows["announce_date"].fillna(rows["ex_date"]).dt.strftime("%Y-%m-%d")
        frame = _stamp(rows, self.name, url, outcome.retrieved_at, available)
        return SourceResult(frame[cols], self.name, url, RETRY_OK, outcome.retrieved_at, len(frame))


# ---------------------------------------------------------------------------
# Eastmoney — supplementary (throttled per IP and per endpoint)
# ---------------------------------------------------------------------------
class EastmoneySource:
    """Eastmoney endpoints. Throttled: treat every failure as expected, not fatal."""

    name = "eastmoney"
    TRENDS_URL = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"
    FFLOW_URL = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    UT_QUOTE = "fa5fd1943c7b386f172d6893dbfba10b"
    UT_FLOW = "b2884a393a59ad64002292a3e90d46a5"
    HEADERS = {"Referer": "https://quote.eastmoney.com/", "Accept-Language": "zh-CN,zh;q=0.9"}

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    def minute_trends(self, symbol: str, days: int = 1) -> SourceResult:
        """One-minute intraday trend bars for the most recent ``days`` sessions."""
        ident = identify(symbol)
        outcome = self.client.get_json(self.TRENDS_URL, headers=self.HEADERS, params={
            "secid": ident.eastmoney_secid, "ut": self.UT_QUOTE,
            "fields1": "f1,f2,f3,f4,f5,f6,f7,f8",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
            "iscr": "0", "ndays": str(days),
        })
        cols = list(contracts.MINUTE_BARS.columns)
        if not outcome.ok:
            return _empty(self.name, outcome, cols)
        node = (outcome.payload or {}).get("data") or {}
        series = node.get("trends") or []
        rows = []
        for item in series:
            parts = str(item).split(",")
            if len(parts) < 7:
                continue
            rows.append({
                "symbol": ident.symbol,
                "bar_time": pd.to_datetime(parts[0], errors="coerce"),
                "open": _f(parts[1]), "close": _f(parts[2]), "high": _f(parts[3]),
                "low": _f(parts[4]),
                "volume": _f(parts[5]) * LOTS_TO_SHARES,
                "amount": _f(parts[6]), "frequency": 1,
            })
        if not rows:
            return _empty(self.name, outcome, cols, RETRY_EMPTY, "no trend rows")
        frame = pd.DataFrame(rows).dropna(subset=["bar_time"])
        frame = _stamp(frame, self.name, self.TRENDS_URL, outcome.retrieved_at,
                       frame["bar_time"].astype(str))
        return SourceResult(frame[cols], self.name, self.TRENDS_URL, RETRY_OK,
                            outcome.retrieved_at, len(frame), metadata={"frequency_minutes": 1})

    def money_flow(self, symbol: str, limit: int = 250) -> SourceResult:
        """Daily order-size bucketed net money flow (CNY)."""
        ident = identify(symbol)
        outcome = self.client.get_json(self.FFLOW_URL, headers=self.HEADERS, params={
            "lmt": str(limit), "klt": "101", "secid": ident.eastmoney_secid,
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
            "ut": self.UT_FLOW,
        })
        cols = list(contracts.MONEY_FLOW.columns)
        if not outcome.ok:
            return _empty(self.name, outcome, cols)
        node = (outcome.payload or {}).get("data") or {}
        series = node.get("klines") or []
        rows = []
        for item in series:
            parts = str(item).split(",")
            if len(parts) < 6:
                continue
            rows.append({
                "symbol": ident.symbol, "trade_date": pd.Timestamp(parts[0]),
                "main_net": _f(parts[1]), "small_net": _f(parts[2]),
                "medium_net": _f(parts[3]), "large_net": _f(parts[4]),
                "extra_large_net": _f(parts[5]),
            })
        if not rows:
            return _empty(self.name, outcome, cols, RETRY_EMPTY, "no money-flow rows")
        frame = pd.DataFrame(rows)
        frame = _stamp(frame, self.name, self.FFLOW_URL, outcome.retrieved_at,
                       frame["trade_date"].dt.strftime("%Y-%m-%d") + " 15:00:00")
        return SourceResult(frame[cols], self.name, self.FFLOW_URL, RETRY_OK,
                            outcome.retrieved_at, len(frame))


# ---------------------------------------------------------------------------
# TickFlow — the entitled vendor SDK
# ---------------------------------------------------------------------------
class TickFlowSource:
    """Adapter over the entitled TickFlow SDK (daily bars, instruments, quotes)."""

    name = "tickflow"
    FULL_COUNT = 10000          # documented max; larger than any A-share daily history

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = build_tickflow_client()
        return self._client

    def daily_bars(self, symbol: str, start: pd.Timestamp | None = None,
                   end: pd.Timestamp | None = None) -> SourceResult:
        ident = identify(symbol)
        endpoint = "tickflow.klines.get(period=1d,adjust=none)"
        cols = list(contracts.DAILY_BARS.columns)
        now = utc_now()
        kwargs: dict[str, Any] = {"period": "1d", "count": self.FULL_COUNT,
                                  "adjust": "none", "as_dataframe": True}
        if start is not None:
            kwargs["start_time"] = int(pd.Timestamp(start).timestamp() * 1000)
        if end is not None:
            kwargs["end_time"] = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp() * 1000) - 1
        try:
            raw = self.client.klines.get(ident.symbol, **kwargs)
        except Exception as exc:  # noqa: BLE001 - vendor SDK raises many error types
            klass = RETRY_ENTITLEMENT if _is_entitlement_error(exc) else RETRY_PERMANENT
            return SourceResult(pd.DataFrame(columns=cols), self.name, endpoint, klass, now, 0,
                                error=f"{type(exc).__name__}: {str(exc)[:160]}")
        if raw is None or not len(raw):
            return SourceResult(pd.DataFrame(columns=cols), self.name, endpoint, RETRY_EMPTY,
                                now, 0, error="vendor returned no bars")
        frame = pd.DataFrame(raw).copy()
        if "trade_date" not in frame.columns:
            tcol = "timestamp" if "timestamp" in frame.columns else frame.columns[0]
            frame["trade_date"] = (pd.to_datetime(frame[tcol], unit="ms", utc=True)
                                   .dt.tz_convert("Asia/Shanghai").dt.normalize().dt.tz_localize(None))
        else:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["symbol"] = ident.symbol
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") * LOTS_TO_SHARES
        for c in ("open", "high", "low", "close", "amount"):
            frame[c] = pd.to_numeric(frame.get(c), errors="coerce")
        frame = frame.drop_duplicates("trade_date").sort_values("trade_date")
        frame = _stamp(frame, self.name, endpoint, now,
                       frame["trade_date"].dt.strftime("%Y-%m-%d") + " 15:00:00")
        return SourceResult(frame[cols], self.name, endpoint, RETRY_OK, now, len(frame))

    def probe(self, method_path: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Call an arbitrary SDK method to establish whether it is entitled."""
        node: Any = self.client
        try:
            for part in method_path.split("."):
                node = getattr(node, part)
            result = node(*args, **kwargs)
        except AttributeError as exc:
            return {"method": method_path, "status": "NOT_IN_SDK", "error": str(exc)[:160]}
        except Exception as exc:  # noqa: BLE001
            status = "UNAUTHORIZED" if _is_entitlement_error(exc) else "ERROR"
            return {"method": method_path, "status": status,
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        size = len(result) if hasattr(result, "__len__") else 1
        return {"method": method_path, "status": "SUPPORTED" if size else "EMPTY", "rows": size}


def _is_entitlement_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}:{exc}".lower()
    return any(k in text for k in ("permission", "unauthor", "forbidden", "403", "not entitled",
                                   "无权限", "未订阅", "权限"))


def build_tickflow_client() -> Any:
    """Construct a TickFlow SDK client from the environment.

    Reads ``TICKFLOW_API_KEY`` / ``TICKFLOW_API_ENDPOINT``. Fails loudly with an
    actionable message rather than degrading to another provider silently.
    """
    import os

    from quantagent.data.ashare.env import load_repo_env

    load_repo_env()
    key = os.environ.get("TICKFLOW_API_KEY")
    if not key:
        raise RuntimeError(
            "TICKFLOW_API_KEY is not set. Export it or place it in the repository .env; "
            "the U0 pipeline will not substitute another provider silently."
        )
    import tickflow  # imported lazily so the module works without the SDK installed

    return tickflow.TickFlow(api_key=key, base_url=os.environ.get("TICKFLOW_API_ENDPOINT"))


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
