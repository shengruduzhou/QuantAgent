from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantagent.data.ashare.contracts import QUALITY_DERIVED, QUALITY_OK
from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable
from quantagent.data.trading_calendar import TradingCalendar


AKSHARE_MARKET_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "source_endpoint",
    "retrieved_at",
    "available_at",
    "quality_status",
)

# EastMoney remains the preferred documented A-share daily endpoint. The
# Tencent endpoint is the default failover because the current AKShare 1.18.84
# implementation normalises its volume to shares and amount to CNY, and the
# repository live source smoke can reach it when EastMoney/Sina are remote-
# closing GitHub Actions connections. Sina stays explicit opt-in only.
_DEFAULT_SOURCE_ORDER: tuple[str, ...] = ("east_money", "tencent")
_CANONICAL_VOLUME_UNIT = "shares"
_CANONICAL_AMOUNT_UNIT = "CNY"
_RAW_VOLUME_UNIT_BY_SOURCE: dict[str, str] = {
    "east_money": "lots_100_shares",
    "tencent": "shares",
    "sina": "shares",
}
_SOURCE_FUNCTIONS: dict[str, str] = {
    "east_money": "stock_zh_a_hist",
    "tencent": "stock_zh_a_hist_tx",
    "sina": "stock_zh_a_daily",
}
_SOURCE_RELIABILITY: dict[str, float] = {
    "east_money": 0.78,
    "tencent": 0.76,
    "sina": 0.72,
}


@dataclass
class AkShareLiveProvider:
    """Optional AKShare downloader with explicit source/unit/PIT contracts.

    Network access is opt-in. Raw/unadjusted prices are the default because an
    adjusted series computed today is not, by itself, proof of the adjustment
    factors that were knowable at a historical decision time.

    The upstream endpoints do not share one raw unit contract. EastMoney daily
    ``成交量`` is lots (手), while current AKShare Tencent and Sina daily output
    volume in shares. The canonical QuantAgent output is always **shares** and
    amount is **CNY**; the original source unit is retained per row.

    A daily bar is research-PIT valid only when it is raw and its next-session
    availability resolves from an explicit trading calendar. Missing calendar
    coverage fails closed; this adapter never invents sessions with
    ``pandas.BDay``.
    """

    allow_network: bool = False
    adjust: str = ""
    source_order: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_SOURCE_ORDER)
    trading_calendar: TradingCalendar | None = None
    calendar_source: str = ""

    def health_check(self, request: ProviderRequest | None = None) -> dict[str, object]:
        if not self.allow_network:
            return {"status": "disabled", "reason": "allow_network_false"}
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            return {"status": "unavailable", "reason": f"akshare_unavailable:{type(exc).__name__}"}
        if request is None:
            return {
                "status": "passed",
                "adjust": self.adjust,
                "akshare_version": _akshare_version(ak),
                "canonical_volume_unit": _CANONICAL_VOLUME_UNIT,
                "canonical_amount_unit": _CANONICAL_AMOUNT_UNIT,
                "source_order": list(self.source_order),
            }
        try:
            result = self.daily_ohlcv(request)
        except ProviderUnavailable as exc:
            return {"status": "unavailable", "reason": str(exc)}
        return {
            "status": "passed" if result.quality_score > 0 else "failed",
            "point_in_time": result.point_in_time,
            "warnings": list(result.warnings),
            "schema_report": result.metadata.get("schema_report", {}),
            "akshare_version": result.metadata.get("akshare_version"),
            "source_counts": result.metadata.get("source_counts", {}),
        }

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        if not self.allow_network:
            raise ProviderUnavailable(
                "AkShare live download is disabled; set data.allow_network=true explicitly"
            )
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable("akshare is not available; install quantagent[data]") from exc
        if not request.symbols:
            raise ProviderUnavailable("AkShare live daily_ohlcv requires explicit symbols")
        if self.adjust not in {"", "qfq", "hfq"}:
            raise ProviderUnavailable("AkShare adjust must be one of '', 'qfq', or 'hfq'")
        unknown_sources = sorted(set(self.source_order).difference(_SOURCE_FUNCTIONS))
        if unknown_sources:
            raise ProviderUnavailable(
                "unknown AKShare daily source(s): " + ",".join(unknown_sources)
            )

        frames: list[pd.DataFrame] = []
        failed_symbols: list[str] = []
        warnings: list[str] = []
        source_counts: dict[str, int] = {}
        source_by_symbol: dict[str, str] = {}
        for symbol in request.symbols:
            raw, used_source, attempt_warnings = _fetch_daily_with_fallback(
                ak,
                symbol,
                start_date=request.start_date,
                end_date=request.end_date,
                adjust=self.adjust,
                source_order=self.source_order,
            )
            warnings.extend(attempt_warnings)
            if raw is None or raw.empty:
                failed_symbols.append(str(symbol))
                warnings.append(f"akshare_empty_daily_ohlcv:{symbol}")
                continue
            normalised = _normalize_akshare_daily(
                raw,
                symbol,
                source=used_source,
                adjust=self.adjust,
                trading_calendar=self.trading_calendar,
            )
            if normalised.empty:
                failed_symbols.append(str(symbol))
                warnings.append(f"akshare_normalized_empty:{symbol}:{used_source}")
                continue
            outside = _rows_outside_request(normalised, request)
            if outside:
                warnings.append(f"akshare_rows_outside_request:{symbol}:{outside}")
                normalised = _filter_request_dates(normalised, request)
            if normalised.empty:
                failed_symbols.append(str(symbol))
                warnings.append(f"akshare_no_rows_in_request:{symbol}:{used_source}")
                continue
            frames.append(normalised)
            source_counts[used_source] = source_counts.get(used_source, 0) + 1
            source_by_symbol[str(symbol)] = used_source

        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        schema_report = akshare_market_schema_report(frame)
        warnings.extend(
            f"akshare_schema_missing:{column}" for column in schema_report["missing_columns"]
        )
        if failed_symbols:
            warnings.append(
                "akshare_incomplete_symbol_coverage:" + ",".join(sorted(set(failed_symbols)))
            )
        if self.adjust:
            warnings.append(
                "akshare_adjusted_history_not_pit_without_vintaged_adjustment_factors"
            )
        if self.trading_calendar is None or self.trading_calendar.empty:
            warnings.append("akshare_market_calendar_missing:pit_not_certified")
        elif int(
            frame.get("point_in_time_valid", pd.Series(dtype=bool)).fillna(False).sum()
        ) < len(frame):
            warnings.append("akshare_market_calendar_incomplete:some_available_at_unresolved")
        if "tencent" in source_counts:
            warnings.append("akshare_tencent_failover_used")

        point_in_time = bool(
            not frame.empty
            and not failed_symbols
            and self.adjust == ""
            and "point_in_time_valid" in frame.columns
            and frame["point_in_time_valid"].fillna(False).astype(bool).all()
        )
        quality_score = 0.78 if not frame.empty and schema_report["status"] == "passed" else 0.0
        if source_counts and set(source_counts) == {"tencent"}:
            quality_score = min(quality_score, 0.76)
        if failed_symbols or not point_in_time:
            quality_score = min(quality_score, 0.60)

        return ProviderResult(
            frame,
            source="akshare_live_provider:multi_source",
            point_in_time=point_in_time,
            quality_score=quality_score,
            warnings=tuple(dict.fromkeys(warnings)),
            metadata={
                "schema_report": schema_report,
                "function_name": "|".join(_SOURCE_FUNCTIONS[source] for source in self.source_order),
                "source_order": list(self.source_order),
                "source_functions": {
                    source: _SOURCE_FUNCTIONS[source] for source in self.source_order
                },
                "source_counts": source_counts,
                "source_by_symbol": source_by_symbol,
                "failed_symbols": failed_symbols,
                "requested_symbol_count": int(len(request.symbols)),
                "fetched_symbol_count": int(sum(source_counts.values())),
                "adjust": self.adjust,
                "adjustment_pit_vintage_bound": False,
                "canonical_volume_unit": _CANONICAL_VOLUME_UNIT,
                "canonical_amount_unit": _CANONICAL_AMOUNT_UNIT,
                "raw_volume_unit_by_source": dict(_RAW_VOLUME_UNIT_BY_SOURCE),
                "calendar_source": self.calendar_source or None,
                "calendar_bound": bool(
                    self.trading_calendar is not None and not self.trading_calendar.empty
                ),
                "calendar_production_certified": False,
                "akshare_version": _akshare_version(ak),
                "production_integrity_certified": False,
            },
        )


def _akshare_version(ak: object) -> str:
    return str(getattr(ak, "__version__", "unknown") or "unknown")


def _plain_a_code(symbol: str) -> str:
    text = str(symbol).split(".")[0]
    lower = text.lower()
    for prefix in ("sh", "sz", "bj"):
        if lower.startswith(prefix):
            return text[len(prefix):]
    return text


def _prefixed_a_symbol(symbol: str) -> str:
    """Return sh/sz/bj-prefixed code used by Tencent and Sina endpoints."""
    text = str(symbol).strip()
    upper = text.upper()
    if "." in upper:
        code, exchange = upper.split(".", 1)
        return f"{exchange.lower()}{code.zfill(6)}"
    lower = text.lower()
    if lower.startswith(("sh", "sz", "bj")):
        return f"{lower[:2]}{text[2:].zfill(6)}"
    code = upper.zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("4", "8", "92")):
        return f"bj{code}"
    return f"sz{code}"


def _fetch_daily_with_fallback(
    ak,
    symbol: str,
    *,
    start_date: str,
    end_date: str,
    adjust: str,
    source_order: tuple[str, ...],
) -> tuple[pd.DataFrame | None, str, list[str]]:
    """Try each configured source and return the first non-empty response."""
    warnings: list[str] = []
    compact_start = start_date.replace("-", "")
    compact_end = end_date.replace("-", "")
    for source in source_order:
        try:
            if source == "east_money":
                raw = ak.stock_zh_a_hist(
                    symbol=_plain_a_code(symbol),
                    period="daily",
                    start_date=compact_start,
                    end_date=compact_end,
                    adjust=adjust,
                )
            elif source == "tencent":
                api = getattr(ak, "stock_zh_a_hist_tx", None)
                if api is None:
                    raise AttributeError("stock_zh_a_hist_tx unavailable")
                raw = api(
                    symbol=_prefixed_a_symbol(symbol),
                    start_date=compact_start,
                    end_date=compact_end,
                    adjust=adjust,
                    timeout=15,
                )
            elif source == "sina":
                raw = ak.stock_zh_a_daily(
                    symbol=_prefixed_a_symbol(symbol),
                    start_date=compact_start,
                    end_date=compact_end,
                    adjust=adjust,
                )
            else:  # guarded by daily_ohlcv validation; defensive for direct tests
                warnings.append(f"akshare_unknown_source:{source}")
                continue
        except Exception as exc:  # pragma: no cover - network path
            warnings.append(f"akshare_{source}_failed:{symbol}:{type(exc).__name__}:{exc}")
            continue
        if raw is None or raw.empty:
            warnings.append(f"akshare_{source}_empty:{symbol}")
            continue
        return raw, source, warnings
    return None, source_order[-1] if source_order else "unknown", warnings


def _normalize_akshare_daily(
    frame: pd.DataFrame,
    symbol: str,
    *,
    source: str = "east_money",
    adjust: str = "",
    trading_calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Normalise EastMoney/Tencent/Sina output to one economic schema.

    Canonical ``volume`` is shares and ``amount`` is CNY. EastMoney daily raw
    volume is lots and therefore scales by 100. Current AKShare Tencent already
    performs its source-specific lot/share and amount-unit conversions inside
    ``stock_zh_a_hist_tx``; Sina daily volume is shares.
    """
    rename_map = {
        "日期": "trade_date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "date": "trade_date",
    }
    data = frame.rename(columns=rename_map)
    keep = [
        c
        for c in ("trade_date", "open", "high", "low", "close", "volume", "amount")
        if c in data.columns
    ]
    if "trade_date" not in keep:
        return pd.DataFrame()
    data = data[keep].copy()
    data["symbol"] = symbol
    trade_dates = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    data["trade_date"] = trade_dates.dt.strftime("%Y-%m-%d")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    raw_volume_unit = _RAW_VOLUME_UNIT_BY_SOURCE.get(source, "unknown")
    if "volume" in data.columns and source == "east_money":
        data["volume"] = data["volume"] * 100.0
    data["volume_unit"] = _CANONICAL_VOLUME_UNIT
    data["raw_volume_unit"] = raw_volume_unit
    data["amount_unit"] = _CANONICAL_AMOUNT_UNIT

    calendar_ok = trading_calendar is not None and not trading_calendar.empty
    if calendar_ok:
        available = pd.Series(
            [trading_calendar.next_trading_day(value, lag_days=1) for value in trade_dates],
            index=data.index,
            dtype="datetime64[ns]",
        )
    else:
        available = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    data["available_at"] = available.dt.strftime("%Y-%m-%d")

    raw_prices = adjust == ""
    data["price_adjustment"] = adjust or "raw"
    data["adjustment_pit_vintage_bound"] = False
    data["source"] = f"akshare:{source}"
    data["source_endpoint"] = _SOURCE_FUNCTIONS.get(source, "unknown")
    data["retrieved_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    data["quality_status"] = QUALITY_OK if raw_prices else QUALITY_DERIVED
    data["source_type"] = "market_data"
    data["source_reliability"] = _SOURCE_RELIABILITY.get(source, 0.0)
    data["point_in_time_valid"] = bool(raw_prices and calendar_ok) & available.notna()
    return data


def _filter_request_dates(frame: pd.DataFrame, request: ProviderRequest) -> pd.DataFrame:
    if frame.empty or "trade_date" not in frame.columns:
        return frame
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    start = pd.Timestamp(request.start_date).normalize()
    end = pd.Timestamp(request.end_date).normalize()
    return frame.loc[dates.notna() & (dates >= start) & (dates <= end)].reset_index(drop=True)


def _rows_outside_request(frame: pd.DataFrame, request: ProviderRequest) -> int:
    if frame.empty or "trade_date" not in frame.columns:
        return 0
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    start = pd.Timestamp(request.start_date).normalize()
    end = pd.Timestamp(request.end_date).normalize()
    return int((dates.isna() | (dates < start) | (dates > end)).sum())


def akshare_market_schema_report(
    frame: pd.DataFrame, as_of_date: str | None = None
) -> dict[str, object]:
    missing = [column for column in AKSHARE_MARKET_REQUIRED_COLUMNS if column not in frame.columns]
    pit_violations = 0
    if "available_at" in frame.columns:
        available_at = pd.to_datetime(frame["available_at"], errors="coerce")
        trade_date = (
            pd.to_datetime(frame["trade_date"], errors="coerce")
            if "trade_date" in frame.columns
            else None
        )
        pit_violations += int(available_at.isna().sum())
        if trade_date is not None:
            pit_violations += int(
                (available_at.notna() & trade_date.notna() & (available_at <= trade_date)).sum()
            )
        if as_of_date:
            pit_violations += int(
                (available_at.notna() & (available_at > pd.Timestamp(as_of_date))).sum()
            )
    if "point_in_time_valid" in frame.columns:
        pit_violations += int((~frame["point_in_time_valid"].fillna(False).astype(bool)).sum())
    return {
        "status": "passed" if not missing and pit_violations == 0 else "failed",
        "row_count": int(0 if frame is None else len(frame)),
        "required_columns": list(AKSHARE_MARKET_REQUIRED_COLUMNS),
        "missing_columns": missing,
        "pit_violation_count": pit_violations,
        "canonical_volume_unit": _CANONICAL_VOLUME_UNIT,
        "canonical_amount_unit": _CANONICAL_AMOUNT_UNIT,
    }
