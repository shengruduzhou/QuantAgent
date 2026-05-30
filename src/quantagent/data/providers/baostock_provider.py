"""BaoStockProvider — free-tier daily + intraday A-share OHLCV.

BaoStock (https://baostock.com/) is a free Chinese securities data
source. It covers daily and 5/15/30/60-minute K-lines, basic index
constituents, and trading calendar. Compared to TuShare it is free
and has no quota / point system; compared to AkShare it is more
stable for full-history daily K-line pulls. We use it as the third
fallback in the v8 data router (Qlib → AkShare → BaoStock →
TuShare).

Limitations of BaoStock acknowledged in the schema:

* Symbol format ``sh.600519`` (not ``600519.SH``). The provider
  normalises both directions.
* Adjustment flag: 1 = pre-adjust (前复权), 2 = post-adjust (后复权),
  3 = unadjusted (原始). Defaults to pre-adjust to match the v7
  market panel convention.
* The 1-minute endpoint is the only one with a short look-back; for
  multi-year history use the 5-minute endpoint. The provider raises
  :class:`ProviderUnavailable` rather than silently switching freq.

Optional dependency: the actual ``baostock`` package only loads when
a method is called. ``ProviderUnavailable`` is raised when the
package is missing — production paths must surface this rather than
fall back to synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from quantagent.data.providers.base import (
    ProviderRequest,
    ProviderResult,
    ProviderUnavailable,
)


# Daily K-line fields BaoStock supports (full list available in their docs).
_DAILY_FIELDS = (
    "date", "code", "open", "high", "low", "close",
    "preclose", "volume", "amount", "adjustflag",
    "turn", "tradestatus", "pctChg", "isST",
)

# Minute K-line fields
_MINUTE_FIELDS = (
    "date", "time", "code", "open", "high", "low", "close",
    "volume", "amount", "adjustflag",
)

_VALID_FREQS: tuple[str, ...] = ("d", "w", "m", "5", "15", "30", "60")


@dataclass(frozen=True)
class BaoStockConfig:
    adjust_flag: str = "1"        # 1=pre, 2=post, 3=raw
    timeout_seconds: float = 30.0
    chunk_size: int = 200          # symbols per BaoStock login session


# ---------------------------------------------------------------------------
# Symbol normalisation
# ---------------------------------------------------------------------------

def to_baostock_symbol(symbol: str) -> str:
    """``600519.SH`` → ``sh.600519`` / passthrough if already correct."""
    if symbol is None:
        raise ValueError("symbol is required")
    s = str(symbol).strip()
    if s.startswith("sh.") or s.startswith("sz.") or s.startswith("bj."):
        return s
    if "." in s:
        code, exch = s.split(".", 1)
        exch = exch.lower()
        if exch in {"sh", "ss"}:
            return f"sh.{code}"
        if exch == "sz":
            return f"sz.{code}"
        if exch == "bj":
            return f"bj.{code}"
        raise ValueError(f"unknown exchange suffix in symbol {s!r}")
    # bare 6-digit fall back to heuristic — Shanghai (6/9) vs Shenzhen
    if len(s) == 6 and s.isdigit():
        if s.startswith("6") or s.startswith("9"):
            return f"sh.{s}"
        return f"sz.{s}"
    raise ValueError(f"cannot normalise symbol {s!r}")


def from_baostock_symbol(symbol: str) -> str:
    """``sh.600519`` → ``600519.SH`` for the v7 canonical format."""
    s = str(symbol).strip()
    if "." in s and s[:2].lower() in {"sh", "sz", "bj"}:
        exch, code = s.split(".", 1)
        return f"{code}.{exch.upper()}"
    return s


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

@dataclass
class BaoStockProvider:
    """Free-tier daily + minute K-line provider for China A-share.

    Real network calls are isolated behind ``_login_and_query`` which
    can be patched in tests with a mock baostock module.
    """

    config: BaoStockConfig = field(default_factory=BaoStockConfig)
    # Allow tests to inject a stub baostock module.
    _bs_module: object | None = None

    # ──────────────────────────────────────────────────────────────────
    # Daily OHLCV
    # ──────────────────────────────────────────────────────────────────
    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        """Daily K-line for every symbol in ``request.symbols``.

        Returns a long-form frame with the canonical v7 columns
        (``symbol / trade_date / open / high / low / close / volume /
        amount / available_at``) plus BaoStock-specific extras
        (``turn`` = turnover rate, ``isST``, ``tradestatus``).
        """
        bs = self._get_module()
        if not request.symbols:
            raise ProviderUnavailable("baostock request requires explicit symbols")
        frames: list[pd.DataFrame] = []
        warnings: list[str] = []
        try:
            self._login(bs)
            for symbol in request.symbols:
                bao_code = to_baostock_symbol(symbol)
                rs = bs.query_history_k_data_plus(
                    bao_code,
                    ",".join(_DAILY_FIELDS),
                    start_date=request.start_date,
                    end_date=request.end_date,
                    frequency="d",
                    adjustflag=self.config.adjust_flag,
                )
                if getattr(rs, "error_code", "0") != "0":
                    warnings.append(f"baostock_error:{symbol}:{rs.error_msg}")
                    continue
                rows: list[list[str]] = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    continue
                df = pd.DataFrame(rows, columns=_DAILY_FIELDS)
                df["symbol"] = from_baostock_symbol(bao_code)
                frames.append(df)
        finally:
            self._logout(bs)
        if not frames:
            return ProviderResult(
                pd.DataFrame(),
                source="baostock_provider",
                quality_score=0.0,
                warnings=tuple(warnings) or ("baostock_empty_result",),
            )
        full = pd.concat(frames, ignore_index=True)
        normalised = _normalise_daily_frame(full)
        return ProviderResult(
            normalised,
            source="baostock_provider",
            point_in_time=True,
            quality_score=0.85,
            warnings=tuple(warnings),
            metadata={
                "rows": int(len(normalised)),
                "symbols": int(normalised["symbol"].nunique()) if not normalised.empty else 0,
                "adjust_flag": self.config.adjust_flag,
            },
        )

    # ──────────────────────────────────────────────────────────────────
    # Minute OHLCV
    # ──────────────────────────────────────────────────────────────────
    def minute_ohlcv(
        self,
        request: ProviderRequest,
        *,
        frequency: str = "5",
    ) -> ProviderResult:
        """Intraday K-line at the requested frequency.

        ``frequency`` ∈ ``{"5", "15", "30", "60"}``. BaoStock does not
        expose 1-minute data through the historical endpoint, so we
        refuse to silently downgrade — call sites must request a
        supported frequency.
        """
        if frequency not in {"5", "15", "30", "60"}:
            raise ProviderUnavailable(
                f"baostock minute_ohlcv frequency must be 5/15/30/60, got {frequency!r}"
            )
        bs = self._get_module()
        if not request.symbols:
            raise ProviderUnavailable("baostock minute request requires explicit symbols")
        frames: list[pd.DataFrame] = []
        warnings: list[str] = []
        try:
            self._login(bs)
            for symbol in request.symbols:
                bao_code = to_baostock_symbol(symbol)
                rs = bs.query_history_k_data_plus(
                    bao_code,
                    ",".join(_MINUTE_FIELDS),
                    start_date=request.start_date,
                    end_date=request.end_date,
                    frequency=frequency,
                    adjustflag=self.config.adjust_flag,
                )
                if getattr(rs, "error_code", "0") != "0":
                    warnings.append(f"baostock_error:{symbol}:{rs.error_msg}")
                    continue
                rows: list[list[str]] = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    continue
                df = pd.DataFrame(rows, columns=_MINUTE_FIELDS)
                df["symbol"] = from_baostock_symbol(bao_code)
                frames.append(df)
        finally:
            self._logout(bs)
        if not frames:
            return ProviderResult(
                pd.DataFrame(),
                source="baostock_provider",
                quality_score=0.0,
                warnings=tuple(warnings) or ("baostock_empty_minute",),
            )
        full = pd.concat(frames, ignore_index=True)
        out = _normalise_minute_frame(full)
        return ProviderResult(
            out,
            source="baostock_provider",
            point_in_time=True,
            quality_score=0.85,
            warnings=tuple(warnings),
            metadata={"frequency": frequency, "rows": int(len(out))},
        )

    # ──────────────────────────────────────────────────────────────────
    # Index daily
    # ──────────────────────────────────────────────────────────────────
    def index_daily(self, request: ProviderRequest) -> ProviderResult:
        """Daily index K-line. Accepts the same symbol format as A-shares."""
        return self.daily_ohlcv(request)

    # ──────────────────────────────────────────────────────────────────
    # Tradability (ST + suspension)
    # ──────────────────────────────────────────────────────────────────
    def tradability(self, request: ProviderRequest) -> ProviderResult:
        """Use the daily frame's ``isST`` + ``tradestatus`` columns."""
        daily = self.daily_ohlcv(request)
        if daily.frame.empty:
            return daily
        flags = daily.frame[["symbol", "trade_date"]].copy()
        flags["is_st"] = (
            daily.frame.get("isST", "0").astype(str).map({"1": True, "True": True}).fillna(False)
        )
        flags["is_suspended"] = (
            daily.frame.get("tradestatus", "1").astype(str).map({"0": True}).fillna(False)
        )
        return ProviderResult(
            flags, source="baostock_provider",
            point_in_time=True, quality_score=daily.quality_score,
        )

    # ──────────────────────────────────────────────────────────────────
    # Health check
    # ──────────────────────────────────────────────────────────────────
    def health_check(self) -> dict[str, object]:
        try:
            bs = self._get_module()
        except ProviderUnavailable as exc:
            return {"status": "unavailable", "reason": str(exc)}
        try:
            self._login(bs)
            return {"status": "ok"}
        except Exception as exc:  # noqa: BLE001 — must report; never crash health-check
            return {"status": "error", "reason": str(exc)}
        finally:
            try:
                self._logout(bs)
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────
    def _get_module(self):
        if self._bs_module is not None:
            return self._bs_module
        try:
            import baostock as bs  # type: ignore
        except Exception as exc:  # pragma: no cover — optional dep
            raise ProviderUnavailable("baostock is not installed") from exc
        return bs

    def _login(self, bs) -> None:
        res = bs.login()
        code = getattr(res, "error_code", "0")
        if code != "0":
            raise ProviderUnavailable(f"baostock login failed: {res.error_msg}")

    def _logout(self, bs) -> None:
        try:
            bs.logout()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Frame normalisation
# ---------------------------------------------------------------------------

def _normalise_daily_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert BaoStock daily output to v7 canonical schema."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    work = raw.copy()
    work["trade_date"] = pd.to_datetime(work["date"], errors="coerce")
    numeric_cols = ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg")
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    # available_at = next trade_date per symbol; last row falls back to trade_date+1d
    work = work.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    work["available_at"] = work.groupby("symbol")["trade_date"].shift(-1)
    work["available_at"] = work["available_at"].fillna(work["trade_date"] + pd.Timedelta(days=1))
    work["source"] = "baostock"
    work["source_type"] = "market_data"
    work["source_reliability"] = 0.85
    work["point_in_time_valid"] = True
    keep = [
        "symbol", "trade_date", "open", "high", "low", "close",
        "preclose", "volume", "amount", "turn", "pctChg", "isST", "tradestatus",
        "adjustflag", "available_at",
        "source", "source_type", "source_reliability", "point_in_time_valid",
    ]
    keep = [c for c in keep if c in work.columns]
    return work[keep].reset_index(drop=True)


def _normalise_minute_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    work = raw.copy()
    # BaoStock minute frames carry ``date`` (YYYY-MM-DD) and ``time``
    # (YYYYMMDDHHMMSSsss). Combine into a tz-naive timestamp.
    if "time" in work.columns:
        # The "time" field is a 17-char string YYYYMMDDHHMMSSSSS
        ts = pd.to_datetime(work["time"].astype(str).str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
    else:
        ts = pd.to_datetime(work["date"], errors="coerce")
    work["timestamp"] = ts
    work["trade_date"] = pd.to_datetime(work["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["available_at"] = work["timestamp"]
    work["source"] = "baostock"
    work["source_type"] = "market_data_minute"
    work["source_reliability"] = 0.85
    work["point_in_time_valid"] = True
    keep = [
        "symbol", "trade_date", "timestamp", "open", "high", "low",
        "close", "volume", "amount", "adjustflag", "available_at",
        "source", "source_type", "source_reliability", "point_in_time_valid",
    ]
    keep = [c for c in keep if c in work.columns]
    return work[keep].reset_index(drop=True)


__all__ = [
    "BaoStockConfig",
    "BaoStockProvider",
    "from_baostock_symbol",
    "to_baostock_symbol",
]
