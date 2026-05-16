from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable


QLIB_MARKET_COLUMNS: tuple[str, ...] = (
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

QLIB_MARKET_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "is_suspended",
    "is_st",
    "is_limit_up",
    "is_limit_down",
)


@dataclass
class QlibProvider:
    """Optional qlib adapter for local PIT market data."""

    provider_uri: str | None = None
    region: str = "cn"

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        try:
            import qlib  # type: ignore
            from qlib.data import D  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable("pyqlib is not available") from exc
        if not self.provider_uri:
            raise ProviderUnavailable("qlib provider_uri is required for V7 qlib data")
        qlib.init(provider_uri=self.provider_uri, region=self.region)
        instruments = list(request.symbols) if request.symbols else request.universe
        if not instruments:
            raise ProviderUnavailable("qlib request requires symbols or universe")
        fields = ["$open", "$high", "$low", "$close", "$volume", "$amount"]
        frame = D.features(instruments, fields, start_time=request.start_date, end_time=request.end_date, freq="day")
        if frame.empty:
            return ProviderResult(pd.DataFrame(), source="qlib_provider", quality_score=0.0, warnings=("qlib_empty_daily_ohlcv",))
        data = frame.reset_index().rename(
            columns={
                "datetime": "trade_date",
                "instrument": "symbol",
                "$open": "open",
                "$high": "high",
                "$low": "low",
                "$close": "close",
                "$volume": "volume",
                "$amount": "amount",
            }
        )
        # Close-derived market features become available the next trading row,
        # not the same day. We compute the per-symbol next trade_date and fall
        # back to trade_date + 1 calendar day at the right edge so newly listed
        # tail rows still have an ``available_at``.
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
        data = data.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        data["available_at"] = data.groupby("symbol")["trade_date"].shift(-1)
        data["available_at"] = data["available_at"].fillna(
            data["trade_date"] + pd.Timedelta(days=1)
        )
        data["source"] = "qlib"
        data["source_type"] = "market_data"
        data["source_reliability"] = 0.90
        data["point_in_time_valid"] = True
        schema_report = validate_qlib_market_schema(data)
        warnings = tuple(
            f"qlib_schema_missing:{column}"
            for column in schema_report["missing_columns"]
        )
        return ProviderResult(
            data,
            source="qlib_provider",
            point_in_time=True,
            quality_score=0.90 if schema_report["status"] == "passed" else 0.40,
            warnings=warnings,
            metadata={"schema_report": schema_report},
        )

    def health_check(self, request: ProviderRequest | None = None) -> dict[str, object]:
        if not self.provider_uri:
            return {"status": "unavailable", "reason": "missing_provider_uri"}
        if not Path(self.provider_uri).exists():
            return {"status": "unavailable", "reason": "provider_uri_not_found", "provider_uri": self.provider_uri}
        try:
            import qlib  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            return {"status": "unavailable", "reason": f"pyqlib_unavailable:{type(exc).__name__}"}
        if request is None:
            return {"status": "passed", "provider_uri": self.provider_uri, "region": self.region}
        try:
            result = self.daily_ohlcv(request)
        except ProviderUnavailable as exc:
            return {"status": "unavailable", "reason": str(exc), "provider_uri": self.provider_uri}
        return {
            "status": "passed" if result.quality_score > 0 else "failed",
            "provider_uri": self.provider_uri,
            "region": self.region,
            "schema_report": result.metadata.get("schema_report", {}),
            "warnings": list(result.warnings),
        }


def validate_qlib_market_schema(frame: pd.DataFrame, as_of_date: str | None = None) -> dict[str, object]:
    missing = [column for column in QLIB_MARKET_COLUMNS if column not in frame.columns]
    optional_present = [column for column in QLIB_MARKET_OPTIONAL_COLUMNS if column in frame.columns]
    optional_missing = [column for column in QLIB_MARKET_OPTIONAL_COLUMNS if column not in frame.columns]
    pit_violations = 0
    if as_of_date and "available_at" in frame.columns:
        pit_violations = int((pd.to_datetime(frame["available_at"], errors="coerce") > pd.Timestamp(as_of_date)).sum())
    return {
        "status": "passed" if not missing and pit_violations == 0 else "failed",
        "row_count": int(0 if frame is None else len(frame)),
        "required_columns": list(QLIB_MARKET_COLUMNS),
        "optional_columns": list(QLIB_MARKET_OPTIONAL_COLUMNS),
        "optional_columns_present": optional_present,
        "optional_columns_missing": optional_missing,
        "missing_columns": missing,
        "pit_violation_count": pit_violations,
    }
