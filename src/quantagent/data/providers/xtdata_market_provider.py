"""Read-only QMT / xtquant (XtData) market-data provider and capability probe.

**Read-only by construction.** This module imports ``xtquant.xtdata`` and
nothing else from the QMT stack. It never imports ``xtquant.xttrader``, never
constructs an ``XtQuantTrader``, and never holds an account id. The execution
side lives in :mod:`quantagent.execution.qmt_gateway` and keeps its own
dry-run defaults; the two must not share state, because a market-data bug that
could reach an order path is a different class of incident from one that
cannot.

**Runtime reality.** ``xtquant`` is published as a ``py3-none-any`` wheel, but
its native extensions (``xtpythonclient``, ``datacenter``) ship exclusively as
``*.cp3XX-win_amd64.pyd`` with Windows DLLs alongside. There is no ``.so`` in
the distribution, so on Linux ``from xtquant import xtdata`` raises
``ImportError`` at the ``datacenter`` import. The probe records that as
``CLIENT_UNAVAILABLE`` rather than guessing at entitlements.

**Entitlement is separate from API surface.** ``xtdata`` exposes
``get_l2_quote`` / ``get_l2_order`` / ``get_l2_transaction`` /
``get_l2thousand_queue``. Those functions existing says only that QMT sells
Level-2; whether *this* broker account may read them is a per-account
entitlement that only a live call can settle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from quantagent.data.microstructure import capability as cap
from quantagent.data.microstructure import contracts as mc
from quantagent.data.microstructure.store import assign_ingest_sequence

#: xtdata period strings, mapped to the canonical family and declared class
#: they would produce. Sourced by reading the shipped ``xtdata.py``: each of
#: these is the literal passed to ``client.get_market_data3``.
XTDATA_PERIODS: dict[str, dict[str, str]] = {
    "tick": {
        "api": "get_market_data_ex / get_full_tick",
        "family": mc.FAMILY_QUOTE,
        "data_class": mc.LEVEL1_QUOTE,
        "note": "盘口 tick 快照：最优买卖 + 5/10 档聚合，不是逐笔委托",
    },
    "l2quote": {
        "api": "get_l2_quote",
        "family": mc.FAMILY_BOOK,
        "data_class": mc.LEVEL2_SNAPSHOT,
        "note": "Level-2 实时行情快照（按价位聚合）",
    },
    "l2order": {
        "api": "get_l2_order",
        "family": mc.FAMILY_ORDER,
        "data_class": mc.EXCHANGE_ORDER_EVENT,
        "note": "Level-2 逐笔委托：真正的单笔委托事件流",
    },
    "l2transaction": {
        "api": "get_l2_transaction",
        "family": mc.FAMILY_TRADE,
        "data_class": mc.EXCHANGE_TRADE_EVENT,
        "note": "Level-2 逐笔成交：带成交编号与买卖单号",
    },
    "l2thousand": {
        "api": "get_l2thousand_queue",
        "family": mc.FAMILY_BOOK,
        "data_class": mc.LEVEL2_ORDER_BOOK,
        "note": "千档盘口队列：逐档委托队列，可支撑排队位置估计",
    },
}

#: Non-market-data families xtdata advertises that matter to U0's PIT gaps.
XTDATA_REFERENCE_APIS: dict[str, str] = {
    "st_history": "get_his_st_data / download_his_st_data",
    "corporate_actions": "get_divid_factors",
    "security_master": "get_instrument_detail / get_stock_list_in_sector",
    "index_membership": "get_index_weight",
    "financials": "get_financial_data",
    "suspension_history": "get_market_data_ex (suspendflag field)",
}


class XtDataUnavailable(RuntimeError):
    """Raised when the QMT client cannot be reached on this host."""


@dataclass
class XtDataRuntime:
    probed_at: str
    package_importable: bool
    xtdata_importable: bool
    package_version: str | None = None
    package_path: str | None = None
    import_error: str | None = None
    native_extensions: list[str] = field(default_factory=list)
    platform_supported: bool = False
    client_connected: bool = False
    connect_error: str | None = None
    data_dir: str | None = None
    authorized_markets: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _inspect_native_extensions() -> tuple[list[str], bool]:
    """List the native modules the installed wheel actually ships.

    Returns ``(names, platform_supported)`` where ``platform_supported`` is True
    only when an extension matching this interpreter's platform exists.
    """
    import sysconfig

    try:
        import xtquant  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return [], False

    package_dir = Path(xtquant.__file__).parent
    names = sorted(
        p.name for p in package_dir.iterdir()
        if p.suffix in {".pyd", ".so"} or p.name.endswith(".dll")
    )
    suffixes = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    supported = any(name.endswith(suffixes) for name in names)
    return names, supported


def probe_runtime() -> XtDataRuntime:
    """Establish whether the QMT market-data client is usable on this host."""
    runtime = XtDataRuntime(
        probed_at=_utc_now(), package_importable=False, xtdata_importable=False
    )
    try:
        import xtquant  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        runtime.import_error = f"{type(exc).__name__}: {exc}"
        runtime.detail = "xtquant is not installed; QMT capability is UNMEASURED here"
        return runtime

    runtime.package_importable = True
    runtime.package_path = str(Path(xtquant.__file__).parent)
    runtime.package_version = getattr(xtquant, "__version__", None)
    runtime.native_extensions, runtime.platform_supported = _inspect_native_extensions()

    try:
        from xtquant import xtdata  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        runtime.import_error = f"{type(exc).__name__}: {exc}"
        runtime.detail = (
            "xtquant imports but its market-data module does not: the wheel "
            "ships Windows-only native extensions, so XtData cannot run here"
        )
        return runtime

    runtime.xtdata_importable = True
    try:
        runtime.data_dir = str(xtdata.get_data_dir())
    except Exception as exc:  # noqa: BLE001
        runtime.connect_error = f"get_data_dir: {type(exc).__name__}: {exc}"
    try:
        markets = xtdata.get_authorized_market_list()
        runtime.authorized_markets = [str(m) for m in (markets or [])]
        runtime.client_connected = True
    except Exception as exc:  # noqa: BLE001
        runtime.connect_error = f"get_authorized_market_list: {type(exc).__name__}: {exc}"
        runtime.detail = (
            "xtdata imported but no QMT/MiniQMT client answered; the terminal "
            "must be running and logged in for any data call to resolve"
        )
    return runtime


def probe_capability(
    symbols: Sequence[str] = ("600000.SH", "000001.SZ"),
    *,
    runtime: XtDataRuntime | None = None,
) -> tuple[XtDataRuntime, cap.CapabilityMatrix]:
    """Probe each Level-2 family and return runtime state plus a matrix.

    When the client is unavailable, every cell is ``CLIENT_UNAVAILABLE`` with
    the *claimed* API recorded in evidence -- documenting what an entitled
    Windows host would be able to try, without asserting it works.
    """
    runtime = runtime or probe_runtime()
    matrix = cap.CapabilityMatrix()
    probed_at = runtime.probed_at

    if not runtime.xtdata_importable or not runtime.client_connected:
        status = (
            cap.CLIENT_UNAVAILABLE if not runtime.xtdata_importable else cap.NOT_PROBED
        )
        detail = runtime.detail or runtime.connect_error or "client unavailable"
        for period, meta in XTDATA_PERIODS.items():
            matrix.add(cap.CapabilityCell(
                provider="qmt_xtdata", dataset_family=_family_for_period(period),
                status=status, entitlement=cap.BROKER_ACCOUNT_REQUIRED,
                probed_at=probed_at, endpoint=meta["api"], detail=detail,
                evidence={
                    "xtdata_period": period,
                    "claimed_data_class": meta["data_class"],
                    "vendor_note": meta["note"],
                    "import_error": runtime.import_error,
                    "native_extensions_shipped": runtime.native_extensions[:12],
                    "platform_supported": runtime.platform_supported,
                },
            ))
        for family, api in XTDATA_REFERENCE_APIS.items():
            matrix.add(cap.CapabilityCell(
                provider="qmt_xtdata", dataset_family=family, status=status,
                entitlement=cap.BROKER_ACCOUNT_REQUIRED, probed_at=probed_at,
                endpoint=api, detail=detail,
                evidence={"claimed_api": api, "import_error": runtime.import_error},
            ))
        return runtime, matrix

    from xtquant import xtdata  # type: ignore[import-not-found]

    for period, meta in XTDATA_PERIODS.items():
        rows = 0
        status = cap.NOT_PROBED
        error: str | None = None
        for symbol in symbols:
            try:
                data = xtdata.get_market_data_ex(
                    [], [symbol], period=period, count=10
                )
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                lowered = error.lower()
                status = (
                    cap.UNAUTHORIZED
                    if any(token in lowered for token in ("auth", "权限", "permission"))
                    else cap.NOT_PROBED
                )
                continue
            frame = (data or {}).get(symbol)
            if frame is not None and len(frame):
                rows += int(len(frame))
        if rows:
            status = cap.SERVING
        elif status == cap.NOT_PROBED and error is None:
            status = cap.EMPTY
        matrix.add(cap.CapabilityCell(
            provider="qmt_xtdata", dataset_family=_family_for_period(period),
            status=status, entitlement=cap.BROKER_ACCOUNT_REQUIRED,
            data_class=meta["data_class"] if rows else None,
            probed_at=probed_at, endpoint=meta["api"],
            rows_returned=rows or None, sample_symbol=symbols[0] if symbols else None,
            detail=error or meta["note"],
            evidence={"xtdata_period": period, "authorized_markets": runtime.authorized_markets},
        ))
    return runtime, matrix


def _family_for_period(period: str) -> str:
    return {
        "tick": "level1_quote",
        "l2quote": "level2_snapshot",
        "l2order": "level2_order_events",
        "l2transaction": "level2_transaction_events",
        "l2thousand": "order_queue",
    }[period]


class XtDataMarketProvider:
    """Read-only canonical-event adapter over ``xtquant.xtdata``.

    Constructing the provider does not connect. Every fetch raises
    :class:`XtDataUnavailable` when the client is absent, so a caller on a host
    without QMT gets a loud, specific failure instead of an empty frame that
    looks like "the market was quiet".
    """

    name = "qmt_xtdata"

    def __init__(self, *, runtime: XtDataRuntime | None = None) -> None:
        self._runtime = runtime

    @property
    def runtime(self) -> XtDataRuntime:
        if self._runtime is None:
            self._runtime = probe_runtime()
        return self._runtime

    def _xtdata(self) -> Any:
        runtime = self.runtime
        if not runtime.xtdata_importable:
            raise XtDataUnavailable(
                f"xtquant.xtdata is not importable on this host: {runtime.import_error}. "
                "QMT/MiniQMT is a Windows client; there is no Linux build of its "
                "native extensions."
            )
        if not runtime.client_connected:
            raise XtDataUnavailable(
                f"xtdata imported but no QMT client answered: {runtime.connect_error}"
            )
        from xtquant import xtdata  # type: ignore[import-not-found]

        return xtdata

    # -- canonical fetches -------------------------------------------------
    def fetch_transactions(
        self, symbol: str, trade_date: str, *, count: int = -1
    ) -> pd.DataFrame:
        """逐笔成交 -> canonical trade events."""
        xtdata = self._xtdata()
        raw = xtdata.get_l2_transaction(
            [], symbol, start_time=f"{trade_date}093000", end_time=f"{trade_date}150000",
            count=count,
        )
        return self.normalise_transactions(raw, symbol=symbol, trade_date=trade_date)

    def fetch_orders(
        self, symbol: str, trade_date: str, *, count: int = -1
    ) -> pd.DataFrame:
        """逐笔委托 -> canonical order events."""
        xtdata = self._xtdata()
        raw = xtdata.get_l2_order(
            [], symbol, start_time=f"{trade_date}091500", end_time=f"{trade_date}150000",
            count=count,
        )
        return self.normalise_orders(raw, symbol=symbol, trade_date=trade_date)

    # -- normalisation (pure, unit-testable without the client) ------------
    @staticmethod
    def _exchange_of(symbol: str) -> str:
        return symbol.rsplit(".", 1)[-1].upper() if "." in symbol else "UNKNOWN"

    @classmethod
    def normalise_transactions(
        cls, raw: Any, *, symbol: str, trade_date: str, receive_time_ns: int | None = None
    ) -> pd.DataFrame:
        """Map an xtdata l2transaction payload onto the canonical trade contract.

        QMT publishes ``tradeIndex`` (exchange trade sequence) and the matched
        ``askOrder`` / ``bidOrder`` numbers. Because the aggressor side follows
        from which of the two order numbers is younger, the resulting ``side``
        is recorded as ``ORDER_ID_MATCHED`` -- observed, not tick-rule guessed.
        """
        frame = _as_frame(raw)
        if frame.empty:
            return pd.DataFrame(columns=list(mc.TRADE_EVENT.columns))

        received = receive_time_ns if receive_time_ns is not None else _now_ns()
        time_ms = _column(frame, "time")
        exchange_time = pd.to_datetime(time_ms, unit="ms", errors="coerce")

        price = _column(frame, "price")
        volume = _column(frame, "volume")
        amount = _column(frame, "amount")
        bid_order = _column(frame, "bidOrder")
        ask_order = _column(frame, "askOrder")

        if bid_order.notna().any() and ask_order.notna().any():
            side = pd.Series(
                ["BUY" if b > a else "SELL" for b, a in zip(bid_order, ask_order)],
                index=frame.index,
            )
            side_method = mc.SIDE_ORDER_MATCHED
        else:
            side = pd.Series([None] * len(frame), index=frame.index)
            side_method = mc.SIDE_UNKNOWN

        canonical = pd.DataFrame({
            "symbol": symbol,
            "exchange": cls._exchange_of(symbol),
            "trade_date": trade_date,
            "exchange_time": exchange_time,
            "event_time_ns": (time_ms * 1_000_000).astype("Int64"),
            "receive_time_ns": received,
            "sequence": _column(frame, "tradeIndex").astype("Int64"),
            "source_provider": "qmt_xtdata",
            "source_channel": "l2transaction",
            "data_class": mc.EXCHANGE_TRADE_EVENT,
            "raw_partition": None,
            "available_at": exchange_time,
            "trade_id": _column(frame, "tradeIndex").astype("Int64"),
            "price": price,
            "volume_shares": volume,
            "amount_cny": amount if amount.notna().any() else price * volume,
            "side": side,
            "side_method": side_method,
            "buy_order_id": bid_order.astype("Int64"),
            "sell_order_id": ask_order.astype("Int64"),
            "trade_kind": _raw_column(frame, "tradeKind"),
        })
        return assign_ingest_sequence(canonical, order_by=["event_time_ns", "sequence"])

    @classmethod
    def normalise_orders(
        cls, raw: Any, *, symbol: str, trade_date: str, receive_time_ns: int | None = None
    ) -> pd.DataFrame:
        """Map an xtdata l2order payload onto the canonical order contract."""
        frame = _as_frame(raw)
        if frame.empty:
            return pd.DataFrame(columns=list(mc.ORDER_EVENT.columns))

        received = receive_time_ns if receive_time_ns is not None else _now_ns()
        time_ms = _column(frame, "time")
        exchange_time = pd.to_datetime(time_ms, unit="ms", errors="coerce")
        # SZSE orderKind: 'B'/'S' direction with orderType 1=market 2=limit 3=best.
        side = _raw_column(frame, "orderKind").map(
            {1: "BUY", 2: "SELL", "B": "BUY", "S": "SELL"}
        )

        canonical = pd.DataFrame({
            "symbol": symbol,
            "exchange": cls._exchange_of(symbol),
            "trade_date": trade_date,
            "exchange_time": exchange_time,
            "event_time_ns": (time_ms * 1_000_000).astype("Int64"),
            "receive_time_ns": received,
            "sequence": _column(frame, "orderIndex").astype("Int64"),
            "source_provider": "qmt_xtdata",
            "source_channel": "l2order",
            "data_class": mc.EXCHANGE_ORDER_EVENT,
            "raw_partition": None,
            "available_at": exchange_time,
            "order_id": _column(frame, "orderIndex").astype("Int64"),
            "order_type": _raw_column(frame, "orderType"),
            "side": side,
            "price": _column(frame, "price"),
            "volume_shares": _column(frame, "volume"),
            "event_action": "INSERT",
        })
        return assign_ingest_sequence(canonical, order_by=["event_time_ns", "sequence"])


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Numeric column, or an all-null Series when the vendor omitted it.

    ``DataFrame.get`` returns ``None``/a scalar for a missing key, which then
    breaks every downstream Series method. Vendor payloads legitimately vary in
    which optional fields they carry (SSE l2transaction has no ``askOrder``),
    so absence must produce nulls rather than an exception or a scalar.
    """
    if name not in frame.columns:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[name], errors="coerce")


def _raw_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Untyped column, or an all-null Series when absent."""
    if name not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype="object")
    return frame[name]


def _as_frame(raw: Any) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame()
    if isinstance(raw, pd.DataFrame):
        return raw.reset_index(drop=True)
    if isinstance(raw, Mapping):
        return pd.DataFrame(dict(raw)).reset_index(drop=True)
    try:
        return pd.DataFrame(raw).reset_index(drop=True)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _now_ns() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)


def write_artifacts(
    runtime: XtDataRuntime, matrix: cap.CapabilityMatrix, directory: str | Path
) -> dict[str, str]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    runtime_path = target / "runtime.json"
    runtime_path.write_text(
        json.dumps(runtime.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written["runtime"] = str(runtime_path)
    written.update(matrix.write(target, stem="capability_matrix"))
    return written
