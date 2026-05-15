from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class AkShareLiveProvider:
    """Optional AkShare downloader; network is disabled unless explicitly enabled."""

    allow_network: bool = False
    adjust: str = "qfq"

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
            raise ProviderUnavailable("akshare is not available") from exc
        if not request.symbols:
            raise ProviderUnavailable("AkShare live daily_ohlcv requires explicit symbols")
        frames = []
        for symbol in request.symbols:
            raw = ak.stock_zh_a_hist(
                symbol=_plain_a_code(symbol),
                period="daily",
                start_date=request.start_date.replace("-", ""),
                end_date=request.end_date.replace("-", ""),
                adjust=self.adjust,
            )
            if raw.empty:
                continue
            frames.append(_normalize_akshare_daily(raw, symbol))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        schema_report = akshare_market_schema_report(frame)
        warnings = [] if not frame.empty else ["akshare_empty_daily_ohlcv"]
        warnings.extend(f"akshare_schema_missing:{column}" for column in schema_report["missing_columns"])
        return ProviderResult(
            frame,
            source="akshare_live_provider",
            point_in_time=True,
            quality_score=0.78 if not frame.empty and schema_report["status"] == "passed" else 0.0,
            warnings=tuple(warnings),
            metadata={"schema_report": schema_report},
        )


def _plain_a_code(symbol: str) -> str:
    return str(symbol).split(".")[0]


def _normalize_akshare_daily(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    columns = {
        "日期": "trade_date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    data = frame.rename(columns=columns)
    keep = [column for column in columns.values() if column in data.columns]
    data = data[keep].copy()
    data["symbol"] = symbol
    data["available_at"] = data["trade_date"]
    data["source"] = "akshare"
    data["source_type"] = "market_data"
    data["source_reliability"] = 0.72
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
