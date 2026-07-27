"""Public A-share tick / Level-1 adapters, with honest semantic labelling.

These are the sources reachable over plain HTTPS from a machine with no broker
terminal. **None of them is Level-2 and none of them is a true tick feed.** No
public source publishes per-order messages, an exchange sequence number, or
individual trade prints; what they call "分笔" is the trading aggregated between
two 3-second exchange snapshots.

What each one actually is, as measured on 2026-07-27:

``TencentTickDetail``   分笔成交 for one symbol-day, paged. Class
                        ``SNAPSHOT_DERIVED_TRADE_AGGREGATE``: records are 3.0s
                        apart and ``amount != price * volume`` on 48%-82% of
                        rows, so each row buckets several trades. The direction
                        flag is Tencent's own classification, hence
                        ``side_method = QUOTE_RULE_INFERRED``, never an exchange
                        field. **This is the only public tick-like route that
                        still answers.**
``SinaTickDetail``      DECOMMISSIONED. Answers HTTP 200 with the 5-byte body
                        ``服务已下线``. Kept so the adapter can detect the notice
                        and fail permanently rather than report an empty day.
``EastmoneyIntraday``   recent 分笔 with a published buy/sell/neutral flag and
                        short retention. Throttles aggressively; measured
                        HTTP 502 under light load.
``TencentLevel1Quote``  the 5-level aggregated display book plus session
                        cumulatives. Class ``LEVEL1_QUOTE``: five aggregated
                        levels is a display depth, not a Level-2 order book,
                        and is labelled accordingly.

Every adapter returns canonical frames per
:mod:`quantagent.data.microstructure.contracts`, with ``sequence`` left null
because no public source publishes one.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Sequence

import pandas as pd

from quantagent.data.ashare.http import (
    RETRY_EMPTY,
    RETRY_OK,
    RETRY_PERMANENT,
    FetchOutcome,
    HttpClient,
)
from quantagent.data.microstructure import contracts as mc
from quantagent.data.microstructure.store import assign_ingest_sequence


def _now_ns() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)


def _vendor_code(symbol: str) -> str:
    """``600000.SH`` -> ``sh600000`` (the spelling Tencent and Sina both use)."""
    code, _, exchange = symbol.partition(".")
    prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(exchange.upper(), exchange.lower())
    return f"{prefix}{code}"


def _empty(family: str) -> pd.DataFrame:
    return pd.DataFrame(columns=list(mc.contract_for(family).columns))


class TencentTickDetail:
    """腾讯 分笔成交 — 3-second aggregated trade buckets for one symbol-day.

    Measured 2026-07-27 against ``600000.SH`` and ``000001.SZ`` for the
    2026-07-24 session: inter-record spacing is exactly 3.0s on 88%/99% of
    records with all gaps a multiple of 3, and ``amount != price * volume`` on
    48%/82% of records. Both facts say the same thing -- each record is the net
    trading between two consecutive exchange snapshots, with ``price`` being the
    snapshot's last price rather than a single trade's price.

    So this is **not** a per-trade feed and is labelled
    ``SNAPSHOT_DERIVED_TRADE_AGGREGATE``.
    """

    name = "tencent_tick_detail"
    URL = "https://stock.gtimg.cn/data/index.php"
    data_class = mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE
    family = mc.FAMILY_TRADE
    #: Exchange snapshot cadence the aggregates are differenced from.
    AGGREGATION_SECONDS = 3

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    def fetch(
        self, symbol: str, trade_date: str, *, max_pages: int = 40
    ) -> tuple[pd.DataFrame, FetchOutcome]:
        """Fetch one full symbol-day, following the vendor's paging.

        The endpoint serves a fixed-size page per ``p``; a full liquid session
        needs several. Paging stops on the first empty page or when a page
        repeats the previous page's last timestamp, so a vendor that silently
        clamps ``p`` cannot produce duplicated events.
        """
        compact = trade_date.replace("-", "")
        pages: list[pd.DataFrame] = []
        outcome: FetchOutcome | None = None
        seen_last: str | None = None

        for page in range(max_pages):
            params = {
                "appn": "detail", "action": "data",
                "c": _vendor_code(symbol), "p": page, "d": compact,
            }
            outcome = self.client.get(self.URL, params=params, encoding="gbk")
            if not outcome.ok:
                break
            frame = self.parse(outcome.text or "", symbol=symbol, trade_date=trade_date)
            if frame.empty:
                break
            marker = f"{frame['exchange_time'].iloc[-1]}|{len(frame)}"
            if marker == seen_last:
                break
            seen_last = marker
            pages.append(frame)

        if not pages:
            if outcome is not None and outcome.ok:
                outcome.retry_class = RETRY_EMPTY
            return _empty(self.family), outcome or FetchOutcome(
                False, self.URL, RETRY_EMPTY, error="no pages fetched"
            )

        combined = pd.concat(pages, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["exchange_time", "price", "volume_shares", "amount_cny"], keep="first"
        )
        combined = assign_ingest_sequence(combined, order_by=["exchange_time"])
        assert outcome is not None
        return combined, outcome

    @classmethod
    def parse(cls, text: str, *, symbol: str, trade_date: str) -> pd.DataFrame:
        """Parse ``v_detail_data_sh600000=[0,"rec|rec|rec"]``.

        Measured payload shape (2026-07-27, sh600000 2026-07-24)::

            0/09:25:03/9.08/0.01/4655/4226740/B|1/09:30:02/9.08/0.00/...

        Records are ``|``-separated; fields within a record are ``/``-separated:
        ``vendor_index time price price_change volume(手) amount(元) side``.

        ``side`` is ``B``/``S``/``M``. ``M`` is Tencent's neutral bucket -- the
        opening call auction print and any trade it cannot classify -- so it maps
        to a null side rather than being forced into a direction.
        """
        if '="' not in text and "=[" not in text:
            return _empty(cls.family)
        body = text.split("=", 1)[-1].strip()
        # Strip the ``[0,"`` ... ``"]`` envelope when present.
        if body.startswith("["):
            first_quote = body.find('"')
            last_quote = body.rfind('"')
            if first_quote == -1 or last_quote <= first_quote:
                return _empty(cls.family)
            body = body[first_quote + 1:last_quote]
        else:
            body = body.strip('";\n ')

        records: list[dict[str, Any]] = []
        for chunk in body.split("|"):
            parts = chunk.split("/")
            if len(parts) < 7:
                continue
            _index, clock, price, _change, volume_lots, amount, side = parts[:7]
            if ":" not in clock:
                continue
            records.append({
                "clock": clock,
                "price": pd.to_numeric(price, errors="coerce"),
                # Tencent reports 分笔 volume in 手 (lots). Converted at the
                # adapter boundary so nothing downstream sees lots.
                "volume_shares": pd.to_numeric(volume_lots, errors="coerce") * 100.0,
                "amount_cny": pd.to_numeric(amount, errors="coerce"),
                "raw_side": side,
            })
        if not records:
            return _empty(cls.family)

        frame = pd.DataFrame(records)
        exchange_time = pd.to_datetime(
            trade_date + " " + frame["clock"], errors="coerce"
        )
        side = frame["raw_side"].map({"B": "BUY", "S": "SELL"})
        canonical = pd.DataFrame({
            "symbol": symbol,
            "exchange": symbol.rsplit(".", 1)[-1].upper(),
            "trade_date": trade_date,
            "exchange_time": exchange_time,
            "event_time_ns": exchange_time.astype("int64", errors="ignore"),
            "receive_time_ns": _now_ns(),
            # No public source publishes an exchange sequence. Leaving this null
            # is the whole point: ingest_sequence orders storage, nothing claims
            # to reproduce the matching engine's ordering.
            "sequence": pd.Series([pd.NA] * len(frame), dtype="Int64"),
            "source_provider": "tencent",
            "source_channel": "stock.gtimg.cn/data/index.php?appn=detail",
            "data_class": cls.data_class,
            "raw_partition": None,
            "available_at": exchange_time,
            "trade_id": pd.Series([pd.NA] * len(frame), dtype="Int64"),
            "price": frame["price"],
            "volume_shares": frame["volume_shares"],
            "amount_cny": frame["amount_cny"],
            "side": side,
            # Tencent classifies direction itself; it is not an exchange field.
            "side_method": mc.SIDE_QUOTE_RULE,
            "buy_order_id": pd.Series([pd.NA] * len(frame), dtype="Int64"),
            "sell_order_id": pd.Series([pd.NA] * len(frame), dtype="Int64"),
            "trade_kind": None,
        })
        canonical = canonical.dropna(subset=["price", "exchange_time"])
        return assign_ingest_sequence(canonical, order_by=["exchange_time"])


class SinaTickDetail:
    """新浪 历史分笔 — **decommissioned by the vendor**, retained as evidence.

    This used to be the long-retention public tick route. Probed 2026-07-27 for
    ``sh600000`` on 2026-07-24: the endpoint answers ``HTTP 200`` with a 5-byte
    body reading ``服务已下线`` ("service taken offline"). A 200 carrying a
    decommission notice is exactly the failure mode that turns into fake data
    if a parser is lenient, so the adapter detects the notice explicitly and
    reports ``RETRY_PERMANENT`` instead of an empty day.
    """

    name = "sina_tick_detail"
    URL = "https://market.finance.sina.com.cn/downxls.php"
    data_class = mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE
    family = mc.FAMILY_TRADE
    #: Vendor notices that mean "gone", not "no data for this day".
    DECOMMISSION_MARKERS: tuple[str, ...] = ("服务已下线", "服务下线", "已停止服务")

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    def fetch(self, symbol: str, trade_date: str) -> tuple[pd.DataFrame, FetchOutcome]:
        params = {"date": trade_date, "symbol": _vendor_code(symbol)}
        outcome = self.client.get(
            self.URL, params=params, encoding="gbk",
            headers={"Referer": "https://vip.stock.finance.sina.com.cn/"},
        )
        if not outcome.ok:
            return _empty(self.family), outcome
        body = (outcome.text or "").strip()
        if any(marker in body for marker in self.DECOMMISSION_MARKERS):
            outcome.ok = False
            outcome.retry_class = RETRY_PERMANENT
            outcome.error = f"vendor decommission notice: {body[:40]!r}"
            return _empty(self.family), outcome
        frame = self.parse(outcome.text or "", symbol=symbol, trade_date=trade_date)
        if frame.empty:
            outcome.retry_class = RETRY_EMPTY
        return frame, outcome

    @classmethod
    def parse(cls, text: str, *, symbol: str, trade_date: str) -> pd.DataFrame:
        if not text or "\t" not in text:
            return _empty(cls.family)
        try:
            raw = pd.read_csv(io.StringIO(text), sep="\t", engine="python")
        except Exception:  # noqa: BLE001 - malformed body is simply no data
            return _empty(cls.family)
        if raw.empty:
            return _empty(cls.family)

        columns = {str(c).strip(): c for c in raw.columns}
        def pick(*names: str) -> pd.Series | None:
            for name in names:
                if name in columns:
                    return raw[columns[name]]
            return None

        clock = pick("成交时间", "时间")
        price = pick("成交价", "成交价格")
        volume = pick("成交量(手)", "成交量")
        amount = pick("成交额(元)", "成交额")
        direction = pick("性质", "买卖类型")
        if clock is None or price is None:
            return _empty(cls.family)

        exchange_time = pd.to_datetime(
            trade_date + " " + clock.astype(str), errors="coerce"
        )
        volume_shares = (
            pd.to_numeric(volume, errors="coerce") * 100.0
            if volume is not None else pd.Series([pd.NA] * len(raw))
        )
        side = (
            direction.map({"买盘": "BUY", "卖盘": "SELL", "中性盘": None})
            if direction is not None else pd.Series([None] * len(raw))
        )
        canonical = pd.DataFrame({
            "symbol": symbol,
            "exchange": symbol.rsplit(".", 1)[-1].upper(),
            "trade_date": trade_date,
            "exchange_time": exchange_time,
            "event_time_ns": exchange_time.astype("int64", errors="ignore"),
            "receive_time_ns": _now_ns(),
            "sequence": pd.Series([pd.NA] * len(raw), dtype="Int64"),
            "source_provider": "sina",
            "source_channel": "market.finance.sina.com.cn/downxls.php",
            "data_class": cls.data_class,
            "raw_partition": None,
            "available_at": exchange_time,
            "trade_id": pd.Series([pd.NA] * len(raw), dtype="Int64"),
            "price": pd.to_numeric(price, errors="coerce"),
            "volume_shares": volume_shares,
            "amount_cny": pd.to_numeric(amount, errors="coerce") if amount is not None else pd.NA,
            "side": side,
            "side_method": mc.SIDE_QUOTE_RULE,
            "buy_order_id": pd.Series([pd.NA] * len(raw), dtype="Int64"),
            "sell_order_id": pd.Series([pd.NA] * len(raw), dtype="Int64"),
            "trade_kind": None,
        })
        canonical = canonical.dropna(subset=["price", "exchange_time"])
        # Sina serves the day newest-first; canonical order is chronological.
        return assign_ingest_sequence(canonical, order_by=["exchange_time"])


class EastmoneyIntraday:
    """东财 逐笔成交 — recent prints with a published direction flag."""

    name = "eastmoney_intraday"
    URL = "https://push2.eastmoney.com/api/qt/stock/details/get"
    data_class = mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE
    family = mc.FAMILY_TRADE

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    @staticmethod
    def _secid(symbol: str) -> str:
        code, _, exchange = symbol.partition(".")
        market = {"SH": "1", "SZ": "0", "BJ": "0"}.get(exchange.upper(), "0")
        return f"{market}.{code}"

    def fetch(self, symbol: str, *, count: int = 200) -> tuple[pd.DataFrame, FetchOutcome]:
        params = {
            "secid": self._secid(symbol),
            "fields1": "f1,f2,f3,f4",
            "fields2": "f51,f52,f53,f54,f55",
            "pos": f"-{int(count)}",
        }
        outcome = self.client.get_json(self.URL, params=params)
        if not outcome.ok:
            return _empty(self.family), outcome
        frame = self.parse(outcome.payload, symbol=symbol)
        if frame.empty:
            outcome.retry_class = RETRY_EMPTY
        return frame, outcome

    @classmethod
    def parse(cls, payload: Any, *, symbol: str, trade_date: str | None = None) -> pd.DataFrame:
        data = (payload or {}).get("data") or {}
        details = data.get("details") or []
        if not details:
            return _empty(cls.family)
        # Each record: "HH:MM:SS,price,volume(手),unused,direction"
        # direction: 1 sell-initiated, 2 buy-initiated, 4 neutral/auction.
        records: list[dict[str, Any]] = []
        for line in details:
            parts = str(line).split(",")
            if len(parts) < 3:
                continue
            records.append({
                "clock": parts[0],
                "price": pd.to_numeric(parts[1], errors="coerce"),
                "volume_shares": pd.to_numeric(parts[2], errors="coerce") * 100.0,
                "flag": parts[4] if len(parts) > 4 else None,
            })
        if not records:
            return _empty(cls.family)

        frame = pd.DataFrame(records)
        day = trade_date or str(data.get("date") or "")
        if len(day) == 8 and day.isdigit():
            day = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        if not day:
            day = datetime.now().strftime("%Y-%m-%d")
        exchange_time = pd.to_datetime(day + " " + frame["clock"], errors="coerce")
        side = frame["flag"].map({"1": "SELL", "2": "BUY", "4": None})
        canonical = pd.DataFrame({
            "symbol": symbol,
            "exchange": symbol.rsplit(".", 1)[-1].upper(),
            "trade_date": day,
            "exchange_time": exchange_time,
            "event_time_ns": exchange_time.astype("int64", errors="ignore"),
            "receive_time_ns": _now_ns(),
            "sequence": pd.Series([pd.NA] * len(frame), dtype="Int64"),
            "source_provider": "eastmoney",
            "source_channel": "push2.eastmoney.com/api/qt/stock/details/get",
            "data_class": cls.data_class,
            "raw_partition": None,
            "available_at": exchange_time,
            "trade_id": pd.Series([pd.NA] * len(frame), dtype="Int64"),
            "price": frame["price"],
            "volume_shares": frame["volume_shares"],
            "amount_cny": frame["price"] * frame["volume_shares"],
            "side": side,
            "side_method": mc.SIDE_QUOTE_RULE,
            "buy_order_id": pd.Series([pd.NA] * len(frame), dtype="Int64"),
            "sell_order_id": pd.Series([pd.NA] * len(frame), dtype="Int64"),
            "trade_kind": None,
        })
        canonical = canonical.dropna(subset=["price", "exchange_time"])
        return assign_ingest_sequence(canonical, order_by=["exchange_time"])


class TencentLevel1Quote:
    """腾讯 实时行情 — best bid/offer plus a 5-level aggregated display book."""

    name = "tencent_level1_quote"
    URL = "https://qt.gtimg.cn/q="
    data_class = mc.LEVEL1_QUOTE
    family = mc.FAMILY_QUOTE
    #: Five aggregated levels. Explicitly not an order book, and the book
    #: adapter labels its output PRICE_LEVEL_AGGREGATED for the same reason.
    DISPLAY_DEPTH = 5

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    def fetch(self, symbols: Sequence[str]) -> tuple[pd.DataFrame, FetchOutcome]:
        codes = ",".join(_vendor_code(s) for s in symbols)
        outcome = self.client.get(self.URL + codes, encoding="gbk")
        if not outcome.ok:
            return _empty(self.family), outcome
        frame = self.parse(outcome.text or "", symbols=symbols)
        if frame.empty:
            outcome.retry_class = RETRY_EMPTY
        return frame, outcome

    @classmethod
    def parse(cls, text: str, *, symbols: Sequence[str]) -> pd.DataFrame:
        by_code = {_vendor_code(s): s for s in symbols}
        rows: list[dict[str, Any]] = []
        received = _now_ns()
        for line in text.splitlines():
            if "=" not in line or '"' not in line:
                continue
            head, _, payload = line.partition("=")
            code = head.strip().split("_")[-1]
            symbol = by_code.get(code)
            if symbol is None:
                continue
            fields = payload.strip().strip('";').split("~")
            if len(fields) < 45:
                continue
            stamp = fields[30]
            exchange_time = pd.to_datetime(stamp, format="%Y%m%d%H%M%S", errors="coerce")
            rows.append({
                "symbol": symbol,
                "exchange": symbol.rsplit(".", 1)[-1].upper(),
                "trade_date": exchange_time.strftime("%Y-%m-%d")
                if pd.notna(exchange_time) else None,
                "exchange_time": exchange_time,
                "event_time_ns": int(exchange_time.value) if pd.notna(exchange_time) else pd.NA,
                "receive_time_ns": received,
                "sequence": pd.NA,
                "source_provider": "tencent",
                "source_channel": "qt.gtimg.cn",
                "data_class": cls.data_class,
                "raw_partition": None,
                "available_at": exchange_time,
                "bid_price": pd.to_numeric(fields[9], errors="coerce"),
                "bid_volume_shares": pd.to_numeric(fields[10], errors="coerce") * 100.0,
                "ask_price": pd.to_numeric(fields[19], errors="coerce"),
                "ask_volume_shares": pd.to_numeric(fields[20], errors="coerce") * 100.0,
                "last_price": pd.to_numeric(fields[3], errors="coerce"),
                "last_volume_shares": pd.NA,
                "cum_volume_shares": pd.to_numeric(fields[36], errors="coerce") * 100.0,
                # Tencent publishes turnover in 万元 on this endpoint.
                "cum_amount_cny": pd.to_numeric(fields[37], errors="coerce") * 10_000.0,
            })
        if not rows:
            return _empty(cls.family)
        return assign_ingest_sequence(pd.DataFrame(rows), order_by=["symbol"])


PUBLIC_TICK_SOURCES: dict[str, type] = {
    TencentTickDetail.name: TencentTickDetail,
    SinaTickDetail.name: SinaTickDetail,
    EastmoneyIntraday.name: EastmoneyIntraday,
    TencentLevel1Quote.name: TencentLevel1Quote,
}
