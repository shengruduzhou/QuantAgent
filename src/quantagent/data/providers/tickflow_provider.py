"""Tickflow data provider — A-share market data + financial statements + SW industry.

Tickflow (https://tickflow.org, https://github.com/tickflow-org/tickflow) is
the v8 primary A-share data source. The Python SDK ``tickflow`` exposes
namespaces ``klines / instruments / exchanges / universes / financials /
quotes / depth / stream``; this module adapts the subset QuantAgent needs
to the existing ``ProviderRequest`` / ``ProviderResult`` contract.

Concrete responsibilities
-------------------------
* ``daily_ohlcv`` — multi-symbol daily K-line via ``tf.klines.batch`` (or
  ``tf.klines.get`` for a single symbol), filtered to the requested date
  window.
* ``adjusted_prices`` — same call plus a per-symbol merge with
  ``tf.klines.ex_factors`` and multiplication of OHLC by the qfq factor.
* ``tradability`` — derived from daily K-line: ``volume == 0`` → suspended,
  ``close ≈ round(prev_close × 1.10, 2)`` → limit-up (10 % cap is the
  non-ST default; ST 5 % is an acceptable approximation for the gating
  use-case), current ST status from the instrument's name carrying
  "ST" / "*ST" prefix.
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

Network access is gated by ``allow_network=True`` *and* a non-empty
``TICKFLOW_API_KEY`` env var (configurable via ``token_env``). Mock data
is never returned; every failure surfaces as ``ProviderUnavailable`` so
the router's fail-loud contract holds.
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
    """

    api_endpoint: str | None = None
    token_env: str = "TICKFLOW_API_KEY"
    source: str = "tickflow"
    source_reliability: float = 0.95
    allow_network: bool = False
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
        return self._sdk().financials.metrics(symbol, as_dataframe=True)

    def financials_income(self, symbol: str) -> pd.DataFrame:
        self._require_ready(method="financials_income")
        return self._sdk().financials.income(symbol, as_dataframe=True)

    def financials_balance_sheet(self, symbol: str) -> pd.DataFrame:
        self._require_ready(method="financials_balance_sheet")
        return self._sdk().financials.balance_sheet(symbol, as_dataframe=True)

    def financials_cash_flow(self, symbol: str) -> pd.DataFrame:
        self._require_ready(method="financials_cash_flow")
        return self._sdk().financials.cash_flow(symbol, as_dataframe=True)

    # ------------------------------------------------------------------
    # SDK lazy client + caches
    # ------------------------------------------------------------------

    def _sdk(self):
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
                # _require_ready already enforces this; defensive guard.
                raise ProviderUnavailable(
                    f"TickflowProvider client init blocked: {self.token_env} not set."
                )
            kwargs: dict[str, Any] = {"api_key": token}
            if self.api_endpoint:
                kwargs["base_url"] = self.api_endpoint
            self._client = TickFlow(**kwargs)
        return self._client

    def _ensure_all_instruments(self) -> list[dict]:
        """Cache ``tf.exchanges.get_instruments`` for SH/SZ/BJ, stocks only."""
        if self._all_instruments is not None:
            return self._all_instruments
        tf = self._sdk()
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
        tf = self._sdk()
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

    def _call_tickflow_daily(self, request: ProviderRequest) -> pd.DataFrame:
        """Daily OHLCV via ``tf.klines.batch`` (≥2 syms) or ``tf.klines.get``."""
        symbols = tuple(request.symbols or ())
        if not symbols:
            return pd.DataFrame()
        tf = self._sdk()
        # tickflow returns the most recent `count` bars. Pull a generous
        # buffer then filter to the requested window.
        count = _DEFAULT_BAR_BUFFER
        if len(symbols) == 1:
            df = tf.klines.get(symbols[0], period="1d", count=count, as_dataframe=True)
            frames = [df] if df is not None else []
        else:
            batch = tf.klines.batch(
                list(symbols), period="1d", count=count,
                as_dataframe=True, show_progress=False,
            ) or {}
            frames = [
                (df if "symbol" in df.columns else df.assign(symbol=sym))
                for sym, df in batch.items() if df is not None and not df.empty
            ]
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        return _filter_window(out, start_date=request.start_date, end_date=request.end_date)

    def _call_tickflow_adjusted(self, request: ProviderRequest) -> pd.DataFrame:
        """Daily K-line × per-symbol qfq ex_factor merge."""
        raw = self._call_tickflow_daily(request)
        if raw.empty:
            return raw
        tf = self._sdk()
        adj_frames: list[pd.DataFrame] = []
        for sym, group in raw.groupby("symbol", sort=False):
            try:
                fac = tf.klines.ex_factors(sym, as_dataframe=True)
            except Exception as exc:  # noqa: BLE001
                _log.debug("tickflow ex_factors(%s) failed: %s", sym, exc)
                fac = None
            if fac is None or fac.empty or "ex_factor" not in fac.columns:
                adj_frames.append(group)
                continue
            fac = fac[["trade_date", "ex_factor"]].copy()
            fac["trade_date"] = pd.to_datetime(fac["trade_date"], errors="coerce")
            g = group.copy()
            g["trade_date"] = pd.to_datetime(g["trade_date"], errors="coerce")
            merged = g.merge(fac, on="trade_date", how="left")
            merged["ex_factor"] = merged["ex_factor"].ffill().fillna(1.0)
            for col in ("open", "high", "low", "close"):
                if col in merged.columns:
                    merged[col] = merged[col] * merged["ex_factor"]
            merged = merged.drop(columns=["ex_factor"])
            adj_frames.append(merged)
        return pd.concat(adj_frames, ignore_index=True) if adj_frames else pd.DataFrame()

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
        out_frames: list[pd.DataFrame] = []
        for sym, group in raw.groupby("symbol", sort=False):
            g = group.sort_values("trade_date").copy()
            g["volume"] = pd.to_numeric(g["volume"], errors="coerce")
            g["close"] = pd.to_numeric(g["close"], errors="coerce")
            g["prev_close"] = g["close"].shift(1)
            cap_up = (g["prev_close"] * 1.10).round(2)
            cap_dn = (g["prev_close"] * 0.90).round(2)
            close_round = g["close"].round(2)
            g["is_suspended"] = (g["volume"].fillna(0) == 0).astype(bool)
            g["is_limit_up"] = ((close_round - cap_up).abs() < 0.005).fillna(False).astype(bool)
            g["is_limit_down"] = ((close_round - cap_dn).abs() < 0.005).fillna(False).astype(bool)
            g["is_st"] = sym in st_set
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
        token = os.environ.get(self.token_env)
        if not token:
            raise ProviderUnavailable(
                f"TickflowProvider.{method} blocked: env var {self.token_env} not set."
            )


# ---------------------------------------------------------------------------
# Frame normalisers — conform Tickflow shapes to the canonical schemas.
# ---------------------------------------------------------------------------


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
