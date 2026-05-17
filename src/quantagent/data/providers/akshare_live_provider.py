from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable


AKSHARE_MARKET_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "available_at",
)

_DEFAULT_SOURCE_ORDER: tuple[str, ...] = ("east_money", "sina")


@dataclass
class AkShareLiveProvider:
    """Optional AkShare downloader; network is disabled unless explicitly enabled.

    ``source_order`` controls per-symbol fetch fallback. The default
    tries East Money first (richest column set) and falls back to the
    Sina free endpoint, which is reachable on networks where the East
    Money kline CDN is filtered.
    """

    allow_network: bool = False
    adjust: str = "qfq"
    source_order: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_SOURCE_ORDER)

    def health_check(self, request: ProviderRequest | None = None) -> dict[str, object]:
        if not self.allow_network:
            return {"status": "disabled", "reason": "allow_network_false"}
        try:
            import akshare as ak  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            return {"status": "unavailable", "reason": f"akshare_unavailable:{type(exc).__name__}"}
        if request is None:
            return {"status": "passed", "adjust": self.adjust}
        try:
            result = self.daily_ohlcv(request)
        except ProviderUnavailable as exc:
            return {"status": "unavailable", "reason": str(exc)}
        return {
            "status": "passed" if result.quality_score > 0 else "failed",
            "warnings": list(result.warnings),
            "schema_report": result.metadata.get("schema_report", {}),
        }

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        if not self.allow_network:
            raise ProviderUnavailable("AkShare live download is disabled; set data.allow_network=true explicitly")
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable("akshare is not available; install quantagent[data]") from exc
        if not request.symbols:
            raise ProviderUnavailable("AkShare live daily_ohlcv requires explicit symbols")
        frames = []
        failed_symbols: list[str] = []
        warnings: list[str] = []
        source_counts: dict[str, int] = {}
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
            frames.append(_normalize_akshare_daily(raw, symbol, source=used_source))
            source_counts[used_source] = source_counts.get(used_source, 0) + 1
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        schema_report = akshare_market_schema_report(frame)
        warnings.extend(f"akshare_schema_missing:{column}" for column in schema_report["missing_columns"])
        return ProviderResult(
            frame,
            source="akshare_live_provider:multi_source",
            point_in_time=True,
            quality_score=0.78 if not frame.empty and schema_report["status"] == "passed" else 0.0,
            warnings=tuple(warnings),
            metadata={
                "schema_report": schema_report,
                "function_name": "stock_zh_a_hist|stock_zh_a_daily",
                "source_order": list(self.source_order),
                "source_counts": source_counts,
                "failed_symbols": failed_symbols,
            },
        )


def _plain_a_code(symbol: str) -> str:
    text = str(symbol).split(".")[0]
    lower = text.lower()
    for prefix in ("sh", "sz", "bj"):
        if lower.startswith(prefix):
            return text[len(prefix):]
    return text


def _sina_a_symbol(symbol: str) -> str:
    """Return Sina-format A-share code, e.g. ``sh600519``/``sz000001``."""
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
    if code.startswith(("4", "8")):
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
    """Try each source in ``source_order``; return the first non-empty frame."""
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
            elif source == "sina":
                raw = ak.stock_zh_a_daily(
                    symbol=_sina_a_symbol(symbol),
                    start_date=compact_start,
                    end_date=compact_end,
                    adjust=adjust,
                )
            else:
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


def _normalize_akshare_daily(frame: pd.DataFrame, symbol: str, *, source: str = "east_money") -> pd.DataFrame:
    """Normalise both East Money (Chinese headers) and Sina (English headers) outputs."""
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
    keep = [c for c in ("trade_date", "open", "high", "low", "close", "volume", "amount") if c in data.columns]
    data = data[keep].copy()
    data["symbol"] = symbol
    trade_dates = pd.to_datetime(data["trade_date"], errors="coerce")
    data["trade_date"] = trade_dates.dt.strftime("%Y-%m-%d")
    data["available_at"] = (trade_dates + pd.offsets.BDay(1)).dt.strftime("%Y-%m-%d")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["source"] = f"akshare:{source}"
    data["source_type"] = "market_data"
    data["source_reliability"] = 0.78 if source == "east_money" else 0.72
    data["point_in_time_valid"] = True
    return data


def akshare_market_schema_report(frame: pd.DataFrame, as_of_date: str | None = None) -> dict[str, object]:
    missing = [column for column in AKSHARE_MARKET_REQUIRED_COLUMNS if column not in frame.columns]
    pit_violations = 0
    if as_of_date and "available_at" in frame.columns:
        pit_violations = int((pd.to_datetime(frame["available_at"], errors="coerce") > pd.Timestamp(as_of_date)).sum())
    return {
        "status": "passed" if not missing and pit_violations == 0 else "failed",
        "row_count": int(0 if frame is None else len(frame)),
        "required_columns": list(AKSHARE_MARKET_REQUIRED_COLUMNS),
        "missing_columns": missing,
        "pit_violation_count": pit_violations,
    }
