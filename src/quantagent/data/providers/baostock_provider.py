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
* Adjustment flag: 1 = post-adjust (后复权), 2 = pre-adjust (前复权),
  3 = unadjusted (原始). The provider defaults to raw prices; adjusted
  research views must be derived explicitly and retain the raw lineage.
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
    adjust_flag: str = "3"        # 1=post, 2=pre, 3=raw
    timeout_seconds: float = 30.0
    chunk_size: int = 200          # symbols per BaoStock login session

    def __post_init__(self) -> None:
        if self.adjust_flag not in {"1", "2", "3"}:
            raise ValueError("BaoStock adjust_flag must be 1=post, 2=pre or 3=raw")


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
        _require_observed_adjustment(full, self.config.adjust_flag)
        is_raw = self.config.adjust_flag == "3"
        if not is_raw:
            warnings.append("baostock_adjusted_series_not_point_in_time")
        normalised = _normalise_daily_frame(full, point_in_time=is_raw)
        return ProviderResult(
            normalised,
            source="baostock_provider",
            point_in_time=is_raw,
            quality_score=0.85,
            warnings=tuple(warnings),
            metadata={
                "rows": int(len(normalised)),
                "symbols": int(normalised["symbol"].nunique()) if not normalised.empty else 0,
                "adjust_flag": self.config.adjust_flag,
                "adjustment": {"1": "post", "2": "pre", "3": "raw"}[self.config.adjust_flag],
                "frequency": "daily",
                "timezone": "Asia/Shanghai",
                "volume_unit": "shares",
                "amount_unit": "CNY",
                "pit_semantics": (
                    "raw_session_bar_available_at_session_close"
                    if is_raw else "adjusted_view_requires_vintaged_factor_lineage"
                ),
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
        _require_observed_adjustment(full, self.config.adjust_flag)
        is_raw = self.config.adjust_flag == "3"
        if not is_raw:
            warnings.append("baostock_adjusted_series_not_point_in_time")
        out = _normalise_minute_frame(full, point_in_time=is_raw)
        return ProviderResult(
            out,
            source="baostock_provider",
            point_in_time=is_raw,
            quality_score=0.85,
            warnings=tuple(warnings),
            metadata={
                "frequency": frequency,
                "rows": int(len(out)),
                "adjust_flag": self.config.adjust_flag,
                "adjustment": {"1": "post", "2": "pre", "3": "raw"}[self.config.adjust_flag],
                "timezone": "Asia/Shanghai",
                "volume_unit": "shares",
                "amount_unit": "CNY",
                "pit_semantics": (
                    "raw_intraday_bar_available_at_bar_timestamp"
                    if is_raw else "adjusted_view_requires_vintaged_factor_lineage"
                ),
            },
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
        is_st = daily.frame.get("isST", pd.Series("0", index=daily.frame.index)).astype(str)
        trade_status = daily.frame.get(
            "tradestatus",
            pd.Series("1", index=daily.frame.index),
        ).astype(str)
        flags["is_st"] = is_st.isin({"1", "True", "true"})
        flags["is_suspended"] = trade_status.eq("0")
        return ProviderResult(
            flags, source="baostock_provider",
            point_in_time=daily.point_in_time,
            quality_score=daily.quality_score,
            warnings=daily.warnings,
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

def _require_observed_adjustment(raw: pd.DataFrame, requested: str) -> None:
    """Reject a vendor response whose row-level adjustment contradicts the request."""
    if "adjustflag" not in raw.columns:
        raise ProviderUnavailable(
            "baostock response omitted adjustflag; price basis cannot be verified"
        )
    observed = {
        str(value).strip()
        for value in raw["adjustflag"].dropna().tolist()
        if str(value).strip()
    }
    if observed != {str(requested)}:
        raise ProviderUnavailable(
            "baostock response adjustment mismatch: "
            f"requested={requested!r}, observed={sorted(observed)!r}"
        )


def _normalise_daily_frame(
    raw: pd.DataFrame,
    *,
    point_in_time: bool,
) -> pd.DataFrame:
    """Convert BaoStock daily output to v7 canonical schema."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    work = raw.copy()
    work["trade_date"] = pd.to_datetime(work["date"], errors="coerce")
    numeric_cols = ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg")
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    # A raw daily bar is knowable at that session's close. Using the next
    # calendar day both invented holiday sessions and violated the training
    # invariant available_at <= trade_date.
    work["available_at"] = work["trade_date"]
    work["is_st"] = work.get("isST", pd.Series("", index=work.index)).astype(str).eq("1")
    work["is_suspended"] = (
        work.get("tradestatus", pd.Series("", index=work.index)).astype(str).eq("0")
    )
    from quantagent.quant_math.ashare import board_price_limit_vector

    limit_ratio = board_price_limit_vector(
        work["symbol"].astype(str),
        work["is_st"],
        trade_dates=work["trade_date"],
    )
    limit_up = (work["preclose"] * (1.0 + limit_ratio)).round(2)
    limit_down = (work["preclose"] * (1.0 - limit_ratio)).round(2)
    close = work["close"].round(2)
    work["is_limit_up"] = ((close - limit_up).abs() < 0.005).fillna(False)
    work["is_limit_down"] = ((close - limit_down).abs() < 0.005).fillna(False)
    work["source"] = "baostock"
    work["source_type"] = "market_data"
    work["source_reliability"] = 0.85
    work["point_in_time_valid"] = bool(point_in_time)
    keep = [
        "symbol", "trade_date", "open", "high", "low", "close",
        "preclose", "volume", "amount", "turn", "pctChg", "isST", "tradestatus",
        "adjustflag", "available_at", "is_st", "is_suspended",
        "is_limit_up", "is_limit_down",
        "source", "source_type", "source_reliability", "point_in_time_valid",
    ]
    keep = [c for c in keep if c in work.columns]
    return work[keep].reset_index(drop=True)


def _normalise_minute_frame(
    raw: pd.DataFrame,
    *,
    point_in_time: bool,
) -> pd.DataFrame:
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
    work["point_in_time_valid"] = bool(point_in_time)
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
