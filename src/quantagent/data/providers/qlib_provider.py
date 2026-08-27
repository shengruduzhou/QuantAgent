from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable
from quantagent.data.v7_auto_range import from_qlib_instrument, to_qlib_instrument


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
        instruments = [to_qlib_instrument(symbol) for symbol in request.symbols] if request.symbols else request.universe
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
        if "symbol" in data.columns:
            data["symbol"] = data["symbol"].astype(str).map(from_qlib_instrument)
        # The raw daily bar is known at its own session close.  Execution
        # latency is modelled by the simulator; shifting availability to an
        # invented next calendar day both conflicts with the dataset as-of
        # contract and mishandles weekends/holidays at the trailing edge.
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
        data = data.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        data["available_at"] = data["trade_date"]
        data["source"] = "qlib"
        data["source_type"] = "market_data"
        data["source_reliability"] = 0.90
        data["point_in_time_valid"] = True
        schema_report = validate_qlib_market_schema(data)
        if schema_report["status"] != "passed":
            raise ProviderUnavailable(f"qlib market schema failed: {schema_report}")
        return ProviderResult(
            data,
            source="qlib_provider",
            point_in_time=True,
            quality_score=0.90,
            warnings=(),
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
    columns = set(() if frame is None else frame.columns)
    missing = [column for column in QLIB_MARKET_COLUMNS if column not in columns]
    optional_present = [column for column in QLIB_MARKET_OPTIONAL_COLUMNS if column in columns]
    optional_missing = [column for column in QLIB_MARKET_OPTIONAL_COLUMNS if column not in columns]
    invalid_trade_dates = 0
    invalid_available_at = 0
    available_after_trade_date = 0
    available_after_as_of = 0
    duplicate_keys = 0
    violation_mask = pd.Series(False, index=[] if frame is None else frame.index, dtype=bool)
    if frame is not None and not frame.empty:
        if "trade_date" in columns:
            trade_dates = pd.to_datetime(frame["trade_date"], errors="coerce")
            invalid_trade_dates = int(trade_dates.isna().sum())
            violation_mask |= trade_dates.isna()
        else:
            trade_dates = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        if "available_at" in columns:
            available = pd.to_datetime(frame["available_at"], errors="coerce")
            invalid_available_at = int(available.isna().sum())
            violation_mask |= available.isna()
            valid_pair = trade_dates.notna() & available.notna()
            after_trade = valid_pair & (available > trade_dates)
            available_after_trade_date = int(after_trade.sum())
            violation_mask |= after_trade
            if as_of_date:
                after_as_of = available.notna() & (available > pd.Timestamp(as_of_date))
                available_after_as_of = int(after_as_of.sum())
                violation_mask |= after_as_of
        if {"symbol", "trade_date"}.issubset(columns):
            duplicate_keys = int(frame.duplicated(["symbol", "trade_date"], keep=False).sum())
    pit_violations = int(violation_mask.sum())
    passed = bool(frame is not None and not frame.empty and not missing and pit_violations == 0 and duplicate_keys == 0)
    return {
        "status": "passed" if passed else "failed",
        "row_count": int(0 if frame is None else len(frame)),
        "required_columns": list(QLIB_MARKET_COLUMNS),
        "optional_columns": list(QLIB_MARKET_OPTIONAL_COLUMNS),
        "optional_columns_present": optional_present,
        "optional_columns_missing": optional_missing,
        "missing_columns": missing,
        "pit_violation_count": pit_violations,
        "invalid_trade_date_count": invalid_trade_dates,
        "invalid_available_at_count": invalid_available_at,
        "available_after_trade_date_count": available_after_trade_date,
        "available_after_as_of_count": available_after_as_of,
        "duplicate_key_count": duplicate_keys,
    }
