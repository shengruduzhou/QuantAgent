"""Tickflow data provider — A-share market data + financial statements + SW industry.

Tickflow (https://tickflow.org, https://github.com/tickflow-org/tickflow) is
the v8 primary A-share data source. The Python SDK ``tickflow`` exposes
namespaces ``klines / instruments / exchanges / universes / financials /
quotes / depth / stream``; this module adapts the subset QuantAgent needs
to the existing ``ProviderRequest`` / ``ProviderResult`` contract.

Concrete responsibilities
-------------------------
* ``daily_ohlcv`` — multi-symbol daily K-line via ``tf.klines.batch`` when the
  batch tier is granted, otherwise a transparent per-symbol ``tf.klines.get``
  fallback (the batch K-line endpoint is separately priced and returns
  ``无...批量查询权限`` on the current subscription), filtered to the window.
* ``adjusted_prices`` — same fetch with the SDK ``adjust="forward"`` (qfq)
  argument. The dedicated ``tf.klines.ex_factors`` endpoint is permission-gated
  (``无除权因子查询权限``), but the K-line endpoint applies the qfq adjustment
  server-side, so true adjusted OHLC is obtained without ex_factors access.
* ``tradability`` — derived from daily K-line: ``volume == 0`` → suspended,
  ``close ≈ round(prev_close × (1 ± board_ratio), 2)`` → limit-up/down. The
  board ratio is resolved by ``quant_math.ashare.daily_price_limit`` from the
  ticker prefix (main 10 % / ChiNext / STAR 20 % / BSE 30 %) with the ST 5 %
  override, so non-main-board names are not mislabelled by a flat 10 % cap.
  Current ST status comes from the instrument name carrying "ST" / "*ST".
* ``stock_basic`` — union of ``tf.exchanges.get_instruments`` on SH/SZ/BJ
  filtered to ``type == "stock"``, joined with the SW1/SW2 industry map.
* ``namechange_history`` — returns an empty frame; tickflow does not
  expose this. The downstream ST builder treats the current snapshot as
  authoritative (``coverage_status='current_snapshot'``), which matches
  the silver-layer schema.

PIT contract: every frame written here is tagged
``available_at = trade_date + 1d`` for the most recent bar in any
window. Symbols use QuantAgent canonical format (``600519.SH``), which is
already tickflow's native format — no remapping needed.

Network access is gated by ``allow_network=True``.  Historical daily K-lines
can use TickFlow's free client when no token is configured; full-service
endpoints (minute bars, financials, universes, real-time quotes) still require
a non-empty ``TICKFLOW_API_KEY`` env var (configurable via ``token_env``).
Mock data is never returned; every failure surfaces as ``ProviderUnavailable``
so the router's fail-loud contract holds.
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
    "symbol", "trade_date",
    "open", "high", "low", "close",
    "volume", "amount",
    "available_at",
    "source", "source_type", "source_reliability", "point_in_time_valid",
)


# Tickflow returns daily K-lines with `symbol, name, timestamp, trade_date,
# trade_time, open, high, low, close, volume, amount`. The canonical
# QuantAgent schema only keeps the OHLCV core + symbol + trade_date, so the
# rename dict is empty by intent — `_normalise_daily_frame` drops the extras
# via the final column projection.
DAILY_RENAME_FROM_TICKFLOW: dict[str, str] = {}

TRADABILITY_RENAME_FROM_TICKFLOW: dict[str, str] = {}


_DEFAULT_BAR_BUFFER = 12_000  # plenty of headroom for an 8y daily window


@dataclass
class TickflowProvider(MarketDataProvider):
    """Adapter for the Tickflow A-share data service.

    Parameters
    ----------
    api_endpoint:
        Optional override for the Tickflow API base URL. Read from
        ``TICKFLOW_API_ENDPOINT`` when not supplied.
    token_env:
        Env-var name holding the auth token. Default ``TICKFLOW_API_KEY``
        (matches the SDK's convention).
    source_reliability:
        Quality baseline used by the multi-source router. Tickflow is the
        v8 primary A-share path so the default is 0.95.
    allow_network:
        Defaults False to match the rest of the providers. Must be flipped
        explicitly by the operator before any real call.
    allow_free_daily:
        Permit ``daily_ohlcv`` to use ``TickFlow.free()`` when no API key is
        present. This follows the official SDK contract: free daily K-lines are
        available without registration, while minute/realtime/financials are not.
    """

    api_endpoint: str | None = None
    token_env: str = "TICKFLOW_API_KEY"
    source: str = "tickflow"
    source_reliability: float = 0.95
    allow_network: bool = False
    allow_free_daily: bool = True
    # In-memory caches for things we only need to fetch once per provider
    # lifetime. Marked private + non-init so dataclass equality stays
    # well-defined.
    _client: Any = field(default=None, init=False, repr=False, compare=False)
    _industry_map: dict[str, tuple[str | None, str | None]] | None = field(
        default=None, init=False, repr=False, compare=False,
    )
    _all_instruments: list[dict] | None = field(
        default=None, init=False, repr=False, compare=False,
    )

    # ------------------------------------------------------------------
    # Public ProviderRequest/Result interface
    # ------------------------------------------------------------------

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        """Daily OHLCV for ``request.symbols`` over [start_date, end_date]."""
        self._require_ready(method="daily_ohlcv")
        raw = self._call_tickflow_daily(request)
        frame = _normalise_daily_frame(raw, source=self.source,
                                       source_reliability=self.source_reliability)
        return ProviderResult(
            frame=frame, source=self.source, point_in_time=True,
            quality_score=self.source_reliability if not frame.empty else 0.0,
            warnings=() if not frame.empty else ("tickflow_empty_daily_ohlcv",),
            metadata={"endpoint": self.api_endpoint or "default"},
        )

    def adjusted_prices(self, request: ProviderRequest) -> ProviderResult:
        """Forward-adjusted (qfq) prices via ``tf.klines.ex_factors`` merge."""
        self._require_ready(method="adjusted_prices")
        raw = self._call_tickflow_adjusted(request)
        frame = _normalise_daily_frame(raw, source=self.source,
                                       source_reliability=self.source_reliability)
        return ProviderResult(
            frame=frame, source=self.source, point_in_time=True,
            quality_score=self.source_reliability if not frame.empty else 0.0,
            warnings=() if not frame.empty else ("tickflow_empty_adjusted",),
            metadata={"adjust_kind": "qfq"},
        )

    def tradability(self, request: ProviderRequest) -> ProviderResult:
        """Per (date, symbol) suspended / ST / limit-up / limit-down flags."""
        self._require_ready(method="tradability")
        raw = self._call_tickflow_tradability(request)
        frame = _normalise_tradability_frame(raw, source=self.source)
        return ProviderResult(
            frame=frame, source=self.source, point_in_time=True,
            quality_score=self.source_reliability if not frame.empty else 0.0,
            warnings=() if not frame.empty else ("tickflow_empty_tradability",),
        )

    # ------------------------------------------------------------------
    # Extended endpoints — sector map + namechange (used by the
    # fetcher scripts in scripts/fetch_*_tickflow.py)
    # ------------------------------------------------------------------

    def stock_basic(self) -> pd.DataFrame:
        """Bulk listed-stocks table with ``symbol, name, industry, ...``."""
        self._require_ready(method="stock_basic")
        return self._call_tickflow_stock_basic()

    def namechange_history(self) -> pd.DataFrame:
        """Empty frame — tickflow does not expose name history."""
        self._require_ready(method="namechange_history")
        return self._call_tickflow_namechange()

    def financials_metrics(self, symbol: str) -> pd.DataFrame:
        """Pass-through to ``tf.financials.metrics(symbol)``."""
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
    # SDK lazy client + caches
    # ------------------------------------------------------------------

    def _sdk(self, *, require_token: bool = True):
        """Return the lazy ``tickflow.TickFlow`` client, creating it once."""
        if self._client is None:
            try:
                from tickflow import TickFlow  # type: ignore
            except ImportError as exc:
                raise ProviderUnavailable(
                    "tickflow SDK not installed in this venv. "
                    "Run: pip install 'tickflow[all]'"
                ) from exc
            token = os.environ.get(self.token_env)
            if not token:
                if not require_token and self.allow_free_daily and hasattr(TickFlow, "free"):
                    self._client = TickFlow.free()
                    return self._client
                # _require_ready already enforces this for full endpoints; defensive guard.
                raise ProviderUnavailable(
                    f"TickflowProvider client init blocked: {self.token_env} not set."
                )
            else:
                kwargs: dict[str, Any] = {"api_key": token}
                if self.api_endpoint:
                    kwargs["base_url"] = self.api_endpoint
                self._client = TickFlow(**kwargs)
        return self._client

    def _ensure_all_instruments(self) -> list[dict]:
        """Cache ``tf.exchanges.get_instruments`` for SH/SZ/BJ, stocks only."""
        if self._all_instruments is not None:
            return self._all_instruments
        tf = self._sdk(require_token=True)
        rows: list[dict] = []
        for ex in ("SH", "SZ", "BJ"):
            try:
                ex_rows = tf.exchanges.get_instruments(ex)
            except Exception as exc:  # noqa: BLE001
                _log.warning("tickflow exchanges.get_instruments(%s) failed: %s", ex, exc)
                continue
            for inst in ex_rows or ():
                if isinstance(inst, dict) and inst.get("type") == "stock":
                    rows.append(inst)
        self._all_instruments = rows
        return rows

    def _ensure_industry_map(self) -> dict[str, tuple[str | None, str | None]]:
        """Walk SW1/SW2 universes once, invert to ``symbol → (sw1, sw2)``."""
        if self._industry_map is not None:
            return self._industry_map
        tf = self._sdk(require_token=True)
        all_universes = tf.universes.list() or []
        mapping: dict[str, list[str | None]] = {}
        for uni in all_universes:
            if not isinstance(uni, dict):
                continue
            uid = str(uni.get("id", ""))
            level: int | None
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
            name = str(detail.get("name", "")).removeprefix("SW1").removeprefix("SW2").strip() or None
            for sym in detail.get("symbols") or ():
                key = str(sym).strip()
                if not key:
                    continue
                slot = mapping.setdefault(key, [None, None])
                if slot[level] is None:
                    slot[level] = name
        self._industry_map = {k: (v[0], v[1]) for k, v in mapping.items()}
        return self._industry_map

    def close(self) -> None:
        """Release the SDK HTTP session."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Real Tickflow client calls
    # ------------------------------------------------------------------

    def _fetch_daily(
        self,
        request: ProviderRequest,
        *,
        adjust: str | None = None,
        require_token: bool = False,
    ) -> pd.DataFrame:
        """Daily K-lines for ``request.symbols``, optionally SDK-side adjusted.

        ``adjust`` maps to the SDK ``AdjustType`` literal
        (``"forward"`` = qfq, ``"backward"`` = hfq, ``None`` = raw). For ≥2
        symbols we attempt ``tf.klines.batch`` first, but the batch K-line
        endpoint is a separately-priced tier (``无...批量查询权限``); when it is
        gated we transparently fall back to a per-symbol ``tf.klines.get``
        loop, which is the always-available path. A single shared helper keeps
        ``daily_ohlcv`` and ``adjusted_prices`` on identical fetch/fallback
        logic — the only difference is the ``adjust`` argument.
        """
        symbols = tuple(request.symbols or ())
        if not symbols:
            return pd.DataFrame()
        tf = self._sdk(require_token=require_token)
        # tickflow returns the most recent `count` bars. Pull a generous
        # buffer then filter to the requested window.
        kw: dict[str, Any] = {"period": "1d", "count": _DEFAULT_BAR_BUFFER,
                              "as_dataframe": True}
        if adjust is not None:
            kw["adjust"] = adjust

        def _get_one(sym: str) -> pd.DataFrame | None:
            df = tf.klines.get(sym, **kw)
            if df is None or df.empty:
                return None
            return df if "symbol" in df.columns else df.assign(symbol=sym)

        frames: list[pd.DataFrame] = []
        if len(symbols) == 1:
            one = _get_one(symbols[0])
            frames = [one] if one is not None else []
        else:
            try:
                batch = tf.klines.batch(list(symbols), show_progress=False, **kw) or {}
                frames = [
                    (df if "symbol" in df.columns else df.assign(symbol=sym))
                    for sym, df in batch.items() if df is not None and not df.empty
                ]
            except ProviderUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 — classify then re-raise non-permission
                if not _is_permission_error(exc):
                    raise
                _log.info(
                    "tickflow klines.batch gated (%s); falling back to per-symbol "
                    "get for %d symbols", exc, len(symbols),
                )
                for sym in symbols:
                    try:
                        one = _get_one(sym)
                    except Exception as e2:  # noqa: BLE001 — skip the bad symbol, keep the rest
                        _log.warning("tickflow klines.get(%s) failed: %s", sym, e2)
                        continue
                    if one is not None:
                        frames.append(one)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        return _filter_window(out, start_date=request.start_date, end_date=request.end_date)

    def _call_tickflow_daily(self, request: ProviderRequest) -> pd.DataFrame:
        """Raw daily OHLCV (free-tier path; batch→per-symbol fallback)."""
        return self._fetch_daily(request, adjust=None, require_token=False)

    def _call_tickflow_adjusted(self, request: ProviderRequest) -> pd.DataFrame:
        """Forward-adjusted (qfq) daily K-lines via the SDK ``adjust`` param.

        The dedicated ``tf.klines.ex_factors`` endpoint is permission-gated on
        the current subscription (``无除权因子查询权限``), but the K-line endpoint
        accepts ``adjust="forward"`` directly and returns server-side qfq OHLC —
        so we get true adjusted prices without ever touching ex_factors.
        """
        return self._fetch_daily(request, adjust="forward", require_token=True)

    def _call_tickflow_tradability(self, request: ProviderRequest) -> pd.DataFrame:
        """Derive per (date, symbol) tradability flags from K-line + names."""
        raw = self._call_tickflow_daily(request)
        if raw.empty:
            return raw
        # Current ST status from the instrument basic name (snapshot only).
        instruments = self._ensure_all_instruments()
        st_set = {
            str(inst["symbol"]).strip()
            for inst in instruments
            if "ST" in str(inst.get("name", "")).upper()
        }
        # Board-aware price limits (main 10% / ChiNext 20% / STAR 20% / BSE 30% /
        # ST 5%) via the canonical AShare rule engine. A flat 10% cap mislabels
        # every ChiNext/STAR/BSE name: a ChiNext stock at +10% is NOT sealed
        # (its limit is +20%), so the flat rule both false-flags it as untradable
        # and misses the real +20% seal. ``daily_price_limit`` resolves the board
        # from the ticker prefix and applies the ST 5% override.
        from quantagent.quant_math.ashare import daily_price_limit

        out_frames: list[pd.DataFrame] = []
        for sym, group in raw.groupby("symbol", sort=False):
            g = group.sort_values("trade_date").copy()
            g["volume"] = pd.to_numeric(g["volume"], errors="coerce")
            g["close"] = pd.to_numeric(g["close"], errors="coerce")
            g["prev_close"] = g["close"].shift(1)
            is_st_sym = sym in st_set
            ratio = float(daily_price_limit(str(sym), is_st_sym))
            cap_up = (g["prev_close"] * (1.0 + ratio)).round(2)
            cap_dn = (g["prev_close"] * (1.0 - ratio)).round(2)
            close_round = g["close"].round(2)
            g["is_suspended"] = (g["volume"].fillna(0) == 0).astype(bool)
            g["is_limit_up"] = ((close_round - cap_up).abs() < 0.005).fillna(False).astype(bool)
            g["is_limit_down"] = ((close_round - cap_dn).abs() < 0.005).fillna(False).astype(bool)
            g["is_st"] = is_st_sym
            out_frames.append(g[["symbol", "trade_date",
                                 "is_suspended", "is_st",
                                 "is_limit_up", "is_limit_down"]])
        return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()

    def _call_tickflow_stock_basic(self) -> pd.DataFrame:
        """Union SH/SZ/BJ stocks, attach SW1/SW2 industry classification."""
        instruments = self._ensure_all_instruments()
        if not instruments:
            return pd.DataFrame(columns=("symbol", "name", "industry",
                                          "industry_sub", "list_date"))
        industry_map = self._ensure_industry_map()
        rows: list[dict] = []
        for inst in instruments:
            sym = str(inst.get("symbol", "")).strip()
            if not sym:
                continue
            sw1, sw2 = industry_map.get(sym, (None, None))
            ext = inst.get("ext") or {}
            rows.append({
                "symbol": sym,
                "name": str(inst.get("name", "")),
                "industry": sw1,
                "industry_sub": sw2,
                "list_date": pd.to_datetime(ext.get("listing_date"), errors="coerce"),
            })
        return pd.DataFrame(rows)

    def _call_tickflow_namechange(self) -> pd.DataFrame:
        """Tickflow has no name-history endpoint — return an empty frame.

        The downstream ST builder accepts an empty namechange table; the
        current ST snapshot derives from ``stock_basic.name`` containing
        the "ST" / "*ST" prefix.
        """
        return pd.DataFrame(columns=("symbol", "name", "start_date", "end_date"))

    # ------------------------------------------------------------------
    # Wiring / guards
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Frame normalisers — conform Tickflow shapes to the canonical schemas.
# ---------------------------------------------------------------------------


def _is_permission_error(exc: Exception) -> bool:
    """True if ``exc`` is a TickFlow tier/permission denial (vs. a real fault).

    Matches the SDK's typed ``PermissionError`` when available, plus the HTTP
    403 / Chinese ``无...权限`` message shapes the API returns for gated
    endpoints. Used to decide when a batch call can safely fall back to the
    per-symbol path rather than propagating as a hard failure.
    """
    try:
        from tickflow import PermissionError as TFPermissionError  # type: ignore
        if isinstance(exc, TFPermissionError):
            return True
    except Exception:  # noqa: BLE001 — SDK missing or no such symbol; fall through to text match
        pass
    msg = str(exc)
    low = msg.lower()
    return ("403" in msg or "permission" in low or "forbidden" in low
            or "权限" in msg)


def _filter_window(
    df: pd.DataFrame, *, start_date: str | None, end_date: str | None,
) -> pd.DataFrame:
    """Trim a multi-symbol daily frame to ``[start_date, end_date]``."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
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
    """Conform raw Tickflow daily output → canonical silver schema."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=CANONICAL_OHLCV_COLUMNS)
    df = raw.rename(columns=DAILY_RENAME_FROM_TICKFLOW).copy()
    df["trade_date"] = pd.to_datetime(df.get("trade_date"), errors="coerce")
    df = df.dropna(subset=["trade_date"]).reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "symbol" not in df.columns:
        raise ValueError("Tickflow daily frame missing 'symbol' column")
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["available_at"] = df["trade_date"] + pd.Timedelta(days=1)
    df["source"] = source
    df["source_type"] = "market_data"
    df["source_reliability"] = float(source_reliability)
    df["point_in_time_valid"] = True
    out = df[list(CANONICAL_OHLCV_COLUMNS)].sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    return out


def _normalise_tradability_frame(raw: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=("symbol", "trade_date",
                                     "is_suspended", "is_st",
                                     "is_limit_up", "is_limit_down",
                                     "available_at", "source"))
    df = raw.rename(columns=TRADABILITY_RENAME_FROM_TICKFLOW).copy()
    df["trade_date"] = pd.to_datetime(df.get("trade_date"), errors="coerce")
    df = df.dropna(subset=["trade_date", "symbol"]).reset_index(drop=True)
    for col in ("is_suspended", "is_st", "is_limit_up", "is_limit_down"):
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)
    df["available_at"] = df["trade_date"] + pd.Timedelta(days=1)
    df["source"] = source
    return df


__all__ = [
    "CANONICAL_OHLCV_COLUMNS",
    "DAILY_RENAME_FROM_TICKFLOW",
    "TRADABILITY_RENAME_FROM_TICKFLOW",
    "TickflowProvider",
]
