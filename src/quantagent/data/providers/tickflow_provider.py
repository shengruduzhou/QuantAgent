"""TickFlow A-share provider with explicit point-in-time boundaries.

TickFlow is used for daily OHLCV, adjusted prices, current instrument metadata,
financial statements and industry membership. A critical boundary applies to
tradability: TickFlow's instrument-name endpoint is a *current snapshot* and the
provider does not expose dated name/risk-warning intervals. Therefore current
``ST`` / ``*ST`` names must never be broadcast backwards and certified as
historical point-in-time data.

``tradability()`` is consequently fail-closed until a dated ST/risk-warning
source is wired in. Operators that explicitly need a current-monitoring view may
call ``current_snapshot_tradability()``; that result is stamped
``point_in_time=False`` and ``st_coverage_status=current_snapshot`` so it cannot
be promoted as historical backtest/training evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from typing import Any

import pandas as pd

from quantagent.data.providers.base import (
    MarketDataProvider,
    ProviderRequest,
    ProviderResult,
    ProviderUnavailable,
)


_log = logging.getLogger(__name__)


CANONICAL_OHLCV_COLUMNS: tuple[str, ...] = (
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
    "source_type",
    "source_reliability",
    "point_in_time_valid",
)

DAILY_RENAME_FROM_TICKFLOW: dict[str, str] = {}
TRADABILITY_RENAME_FROM_TICKFLOW: dict[str, str] = {}
_DEFAULT_BAR_BUFFER = 12_000
_HISTORICAL_ST_UNAVAILABLE = (
    "Tickflow historical tradability is fail-closed: the available instrument "
    "name list is a current snapshot and cannot certify historical ST/risk-warning "
    "state. Use a dated PIT source (for example a provider exposing row-level "
    "historical isST) for backtest/training, or call "
    "current_snapshot_tradability() explicitly for current monitoring only."
)


@dataclass
class TickflowProvider(MarketDataProvider):
    """Adapter for the TickFlow A-share data service."""

    api_endpoint: str | None = None
    token_env: str = "TICKFLOW_API_KEY"
    source: str = "tickflow"
    source_reliability: float = 0.95
    allow_network: bool = False
    allow_free_daily: bool = True
    _client: Any = field(default=None, init=False, repr=False, compare=False)
    _industry_map: dict[str, tuple[str | None, str | None]] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _all_instruments: list[dict] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # ------------------------------------------------------------------
    # Public provider contract
    # ------------------------------------------------------------------

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        self._require_ready(method="daily_ohlcv")
        raw = self._call_tickflow_daily(request)
        frame = _normalise_daily_frame(
            raw,
            source=self.source,
            source_reliability=self.source_reliability,
        )
        return ProviderResult(
            frame=frame,
            source=self.source,
            point_in_time=True,
            quality_score=self.source_reliability if not frame.empty else 0.0,
            warnings=() if not frame.empty else ("tickflow_empty_daily_ohlcv",),
            metadata={"endpoint": self.api_endpoint or "default"},
        )

    def adjusted_prices(self, request: ProviderRequest) -> ProviderResult:
        self._require_ready(method="adjusted_prices")
        raw = self._call_tickflow_adjusted(request)
        frame = _normalise_daily_frame(
            raw,
            source=self.source,
            source_reliability=self.source_reliability,
        )
        return ProviderResult(
            frame=frame,
            source=self.source,
            point_in_time=True,
            quality_score=self.source_reliability if not frame.empty else 0.0,
            warnings=() if not frame.empty else ("tickflow_empty_adjusted",),
            metadata={"adjust_kind": "qfq"},
        )

    def tradability(self, request: ProviderRequest) -> ProviderResult:
        """Historical tradability contract; fail closed without dated ST state.

        Suspension can be inferred from historical volume, but main-board price
        limits depend on the contemporaneous ST/risk-warning state. Returning a
        partly trustworthy frame would let downstream code silently coerce the
        unknown ST component to ``False``. Refusing the composite capability is
        therefore safer and forces the router/operator onto a real PIT source.
        """

        self._require_ready(method="tradability")
        del request
        raise ProviderUnavailable(_HISTORICAL_ST_UNAVAILABLE)

    def current_snapshot_tradability(self, request: ProviderRequest) -> ProviderResult:
        """Explicit non-PIT tradability view for current monitoring only."""

        self._require_ready(method="tradability")
        raw = self._call_tickflow_current_snapshot_tradability(request)
        frame = _normalise_current_snapshot_tradability_frame(raw, source=self.source)
        warnings = ["tickflow_current_snapshot_st_not_historical_pit"]
        if frame.empty:
            warnings.append("tickflow_empty_current_snapshot_tradability")
        return ProviderResult(
            frame=frame,
            source=self.source,
            point_in_time=False,
            quality_score=self.source_reliability if not frame.empty else 0.0,
            warnings=tuple(warnings),
            metadata={
                "st_coverage_status": "current_snapshot",
                "historical_pit_certified": False,
                "snapshot_semantics": "instrument_name_as_of_fetch",
                "requested_start_date": request.start_date,
                "requested_end_date": request.end_date,
            },
        )

    # ------------------------------------------------------------------
    # Extended endpoints
    # ------------------------------------------------------------------

    def stock_basic(self) -> pd.DataFrame:
        self._require_ready(method="stock_basic")
        return self._call_tickflow_stock_basic()

    def namechange_history(self) -> pd.DataFrame:
        """Return empty history; TickFlow does not expose dated name intervals."""

        self._require_ready(method="namechange_history")
        return self._call_tickflow_namechange()

    def financials_metrics(self, symbol: str) -> pd.DataFrame:
        self._require_ready(method="financials_metrics")
        return self._sdk(require_token=True).financials.metrics(symbol, as_dataframe=True)

    def financials_income(self, symbol: str) -> pd.DataFrame:
        self._require_ready(method="financials_income")
        return self._sdk(require_token=True).financials.income(symbol, as_dataframe=True)

    def financials_balance_sheet(self, symbol: str) -> pd.DataFrame:
        self._require_ready(method="financials_balance_sheet")
        return self._sdk(require_token=True).financials.balance_sheet(symbol, as_dataframe=True)

    def financials_cash_flow(self, symbol: str) -> pd.DataFrame:
        self._require_ready(method="financials_cash_flow")
        return self._sdk(require_token=True).financials.cash_flow(symbol, as_dataframe=True)

    # ------------------------------------------------------------------
    # SDK client/caches
    # ------------------------------------------------------------------

    def _sdk(self, *, require_token: bool = True):
        if self._client is None:
            try:
                from tickflow import TickFlow  # type: ignore
            except ImportError as exc:
                raise ProviderUnavailable(
                    "tickflow SDK not installed in this venv. Run: pip install 'tickflow[all]'"
                ) from exc
            token = os.environ.get(self.token_env)
            if not token:
                if not require_token and self.allow_free_daily and hasattr(TickFlow, "free"):
                    self._client = TickFlow.free()
                    return self._client
                raise ProviderUnavailable(
                    f"TickflowProvider client init blocked: {self.token_env} not set."
                )
            kwargs: dict[str, Any] = {"api_key": token}
            if self.api_endpoint:
                kwargs["base_url"] = self.api_endpoint
            self._client = TickFlow(**kwargs)
        return self._client

    def _ensure_all_instruments(self) -> list[dict]:
        if self._all_instruments is not None:
            return self._all_instruments
        tf = self._sdk(require_token=True)
        rows: list[dict] = []
        for exchange in ("SH", "SZ", "BJ"):
            try:
                exchange_rows = tf.exchanges.get_instruments(exchange)
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "tickflow exchanges.get_instruments(%s) failed: %s",
                    exchange,
                    exc,
                )
                continue
            for instrument in exchange_rows or ():
                if isinstance(instrument, dict) and instrument.get("type") == "stock":
                    rows.append(instrument)
        self._all_instruments = rows
        return rows

    def _ensure_industry_map(self) -> dict[str, tuple[str | None, str | None]]:
        if self._industry_map is not None:
            return self._industry_map
        tf = self._sdk(require_token=True)
        all_universes = tf.universes.list() or []
        mapping: dict[str, list[str | None]] = {}
        for universe in all_universes:
            if not isinstance(universe, dict):
                continue
            uid = str(universe.get("id", ""))
            if uid.startswith("CN_Equity_SW1_"):
                level = 0
            elif uid.startswith("CN_Equity_SW2_"):
                level = 1
            else:
                continue
            try:
                detail = tf.universes.get(uid)
            except Exception as exc:  # noqa: BLE001
                _log.warning("tickflow universes.get(%s) failed: %s", uid, exc)
                continue
            if not isinstance(detail, dict):
                continue
            name = (
                str(detail.get("name", ""))
                .removeprefix("SW1")
                .removeprefix("SW2")
                .strip()
                or None
            )
            for symbol in detail.get("symbols") or ():
                key = str(symbol).strip()
                if not key:
                    continue
                slot = mapping.setdefault(key, [None, None])
                if slot[level] is None:
                    slot[level] = name
        self._industry_map = {key: (value[0], value[1]) for key, value in mapping.items()}
        return self._industry_map

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # TickFlow calls
    # ------------------------------------------------------------------

    def _fetch_daily(
        self,
        request: ProviderRequest,
        *,
        adjust: str | None = None,
        require_token: bool = False,
    ) -> pd.DataFrame:
        symbols = tuple(request.symbols or ())
        if not symbols:
            return pd.DataFrame()
        tf = self._sdk(require_token=require_token)
        kwargs: dict[str, Any] = {
            "period": "1d",
            "count": _DEFAULT_BAR_BUFFER,
            "as_dataframe": True,
        }
        if adjust is not None:
            kwargs["adjust"] = adjust

        def get_one(symbol: str) -> pd.DataFrame | None:
            frame = tf.klines.get(symbol, **kwargs)
            if frame is None or frame.empty:
                return None
            return frame if "symbol" in frame.columns else frame.assign(symbol=symbol)

        frames: list[pd.DataFrame] = []
        if len(symbols) == 1:
            one = get_one(symbols[0])
            if one is not None:
                frames.append(one)
        else:
            try:
                batch = tf.klines.batch(
                    list(symbols), show_progress=False, **kwargs
                ) or {}
                frames = [
                    frame if "symbol" in frame.columns else frame.assign(symbol=symbol)
                    for symbol, frame in batch.items()
                    if frame is not None and not frame.empty
                ]
            except ProviderUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001
                if not _is_permission_error(exc):
                    raise
                _log.info(
                    "tickflow klines.batch gated (%s); falling back to per-symbol get for %d symbols",
                    exc,
                    len(symbols),
                )
                for symbol in symbols:
                    try:
                        one = get_one(symbol)
                    except Exception as nested:  # noqa: BLE001
                        _log.warning("tickflow klines.get(%s) failed: %s", symbol, nested)
                        continue
                    if one is not None:
                        frames.append(one)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        return _filter_window(
            out, start_date=request.start_date, end_date=request.end_date
        )

    def _call_tickflow_daily(self, request: ProviderRequest) -> pd.DataFrame:
        return self._fetch_daily(request, adjust=None, require_token=False)

    def _call_tickflow_adjusted(self, request: ProviderRequest) -> pd.DataFrame:
        return self._fetch_daily(request, adjust="forward", require_token=True)

    def _call_tickflow_tradability(self, request: ProviderRequest) -> pd.DataFrame:
        """Private historical path is guarded too; callers must not bypass public API."""

        del request
        raise ProviderUnavailable(_HISTORICAL_ST_UNAVAILABLE)

    def _call_tickflow_current_snapshot_tradability(
        self, request: ProviderRequest
    ) -> pd.DataFrame:
        """Derive a non-PIT current-snapshot view from K-lines + current names."""

        raw = self._call_tickflow_daily(request)
        if raw.empty:
            return raw
        instruments = self._ensure_all_instruments()
        st_set = {
            str(instrument["symbol"]).strip()
            for instrument in instruments
            if "ST" in str(instrument.get("name", "")).upper()
        }
        from quantagent.quant_math.ashare import board_price_limit_vector

        out_frames: list[pd.DataFrame] = []
        for symbol, group in raw.groupby("symbol", sort=False):
            frame = group.sort_values("trade_date").copy()
            frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame["prev_close"] = frame["close"].shift(1)
            is_st_snapshot = bool(symbol in st_set)
            ratio = board_price_limit_vector(
                frame["symbol"].astype(str),
                pd.Series(is_st_snapshot, index=frame.index, dtype=bool),
                trade_dates=frame["trade_date"],
            )
            cap_up = (frame["prev_close"] * (1.0 + ratio)).round(2)
            cap_down = (frame["prev_close"] * (1.0 - ratio)).round(2)
            close_round = frame["close"].round(2)
            frame["is_suspended"] = (frame["volume"].fillna(0) == 0).astype(bool)
            frame["is_limit_up"] = (
                (close_round - cap_up).abs() < 0.005
            ).fillna(False).astype(bool)
            frame["is_limit_down"] = (
                (close_round - cap_down).abs() < 0.005
            ).fillna(False).astype(bool)
            frame["is_st"] = is_st_snapshot
            out_frames.append(
                frame[
                    [
                        "symbol",
                        "trade_date",
                        "is_suspended",
                        "is_st",
                        "is_limit_up",
                        "is_limit_down",
                    ]
                ]
            )
        return (
            pd.concat(out_frames, ignore_index=True)
            if out_frames
            else pd.DataFrame()
        )

    def _call_tickflow_stock_basic(self) -> pd.DataFrame:
        instruments = self._ensure_all_instruments()
        if not instruments:
            return pd.DataFrame(
                columns=("symbol", "name", "industry", "industry_sub", "list_date")
            )
        industry_map = self._ensure_industry_map()
        rows: list[dict] = []
        for instrument in instruments:
            symbol = str(instrument.get("symbol", "")).strip()
            if not symbol:
                continue
            sw1, sw2 = industry_map.get(symbol, (None, None))
            ext = instrument.get("ext") or {}
            rows.append(
                {
                    "symbol": symbol,
                    "name": str(instrument.get("name", "")),
                    "industry": sw1,
                    "industry_sub": sw2,
                    "list_date": pd.to_datetime(
                        ext.get("listing_date"), errors="coerce"
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _call_tickflow_namechange(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=("symbol", "name", "start_date", "end_date")
        )

    def _require_ready(self, *, method: str) -> None:
        if not self.allow_network:
            raise ProviderUnavailable(
                f"TickflowProvider.{method} blocked: allow_network=False."
            )
        if method == "daily_ohlcv" and self.allow_free_daily:
            return
        token = os.environ.get(self.token_env)
        if not token:
            raise ProviderUnavailable(
                f"TickflowProvider.{method} blocked: env var {self.token_env} not set."
            )


def _is_permission_error(exc: Exception) -> bool:
    try:
        from tickflow import PermissionError as TFPermissionError  # type: ignore

        if isinstance(exc, TFPermissionError):
            return True
    except Exception:  # noqa: BLE001
        pass
    message = str(exc)
    lowered = message.lower()
    return (
        "403" in message
        or "permission" in lowered
        or "forbidden" in lowered
        or "权限" in message
    )


def _filter_window(
    frame: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    if start_date:
        out = out[out["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        out = out[out["trade_date"] <= pd.Timestamp(end_date)]
    return out.reset_index(drop=True)


def _normalise_daily_frame(
    raw: pd.DataFrame,
    *,
    source: str,
    source_reliability: float,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=CANONICAL_OHLCV_COLUMNS)
    frame = raw.rename(columns=DAILY_RENAME_FROM_TICKFLOW).copy()
    frame["trade_date"] = pd.to_datetime(frame.get("trade_date"), errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "symbol" not in frame.columns:
        raise ValueError("Tickflow daily frame missing 'symbol' column")
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    frame["available_at"] = frame["trade_date"] + pd.Timedelta(days=1)
    frame["source"] = source
    frame["source_type"] = "market_data"
    frame["source_reliability"] = float(source_reliability)
    frame["point_in_time_valid"] = True
    return frame[list(CANONICAL_OHLCV_COLUMNS)].sort_values(
        ["symbol", "trade_date"]
    ).reset_index(drop=True)


def _normalise_current_snapshot_tradability_frame(
    raw: pd.DataFrame, *, source: str
) -> pd.DataFrame:
    columns = (
        "symbol",
        "trade_date",
        "is_suspended",
        "is_st",
        "is_limit_up",
        "is_limit_down",
        "available_at",
        "source",
        "st_coverage_status",
        "point_in_time_valid",
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=columns)
    frame = raw.rename(columns=TRADABILITY_RENAME_FROM_TICKFLOW).copy()
    frame["trade_date"] = pd.to_datetime(frame.get("trade_date"), errors="coerce")
    frame = frame.dropna(subset=["trade_date", "symbol"]).reset_index(drop=True)
    for column in ("is_suspended", "is_st", "is_limit_up", "is_limit_down"):
        if column not in frame.columns:
            raise ValueError(
                f"Tickflow current-snapshot tradability missing {column!r}"
            )
        frame[column] = frame[column].astype(bool)
    observed_at = pd.Timestamp.now(tz="UTC")
    frame["available_at"] = observed_at
    frame["source"] = source
    frame["st_coverage_status"] = "current_snapshot"
    frame["point_in_time_valid"] = False
    return frame[list(columns)]


# Backwards-private helper name intentionally remains but cannot normalise an
# unsafe historical frame because `_call_tickflow_tradability` now fail-closes.
def _normalise_tradability_frame(raw: pd.DataFrame, *, source: str) -> pd.DataFrame:
    return _normalise_current_snapshot_tradability_frame(raw, source=source)


__all__ = [
    "CANONICAL_OHLCV_COLUMNS",
    "DAILY_RENAME_FROM_TICKFLOW",
    "TRADABILITY_RENAME_FROM_TICKFLOW",
    "TickflowProvider",
]
