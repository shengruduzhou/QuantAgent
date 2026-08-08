"""Capability-aware integrity contracts for A-share daily bars.

The validator is intentionally scoped to daily OHLCV. It does not pretend that
fundamentals, news, flows or minute data share the same schema. No invalid bar is
repaired with a synthetic value: research mode quarantines invalid rows with
explicit evidence, while production mode exposes only validated rows and lets
the router seek a lower-priority real provider for missing keys.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult


DAILY_IDENTITY_COLUMNS = ("symbol", "trade_date")
DAILY_OHLC_COLUMNS = ("open", "high", "low", "close")
DAILY_FLOW_COLUMNS = ("volume", "amount")
DAILY_FULL_COLUMNS = DAILY_IDENTITY_COLUMNS + DAILY_OHLC_COLUMNS + DAILY_FLOW_COLUMNS
_DAILY_FREQUENCIES = {"1d", "d", "day", "daily"}


@dataclass(frozen=True)
class DailyOHLCVIntegrityPolicy:
    mode: Literal["research", "production"] = "research"
    require_full_ohlcv: bool = False
    require_declared_units: bool = False
    require_frequency_metadata: bool = False
    require_timezone_metadata: bool = False
    require_point_in_time: bool = False
    require_pit_semantics: bool = False
    reject_mock_or_synthetic: bool = False
    expected_adjustment: str | None = None
    allow_invalid_primary_fallback: bool = True
    expected_trade_dates: tuple[str, ...] = ()
    live_critical: bool = False
    expected_latest_trade_date: str | None = None
    max_staleness_calendar_days: int = 0

    @classmethod
    def research(cls) -> "DailyOHLCVIntegrityPolicy":
        return cls(mode="research")

    @classmethod
    def production(
        cls,
        *,
        expected_trade_dates: tuple[str, ...] = (),
        expected_latest_trade_date: str | None = None,
        live_critical: bool = False,
        max_staleness_calendar_days: int = 0,
        allow_invalid_primary_fallback: bool = True,
    ) -> "DailyOHLCVIntegrityPolicy":
        return cls(
            mode="production",
            require_full_ohlcv=True,
            require_declared_units=True,
            require_frequency_metadata=True,
            require_timezone_metadata=True,
            require_point_in_time=True,
            require_pit_semantics=True,
            reject_mock_or_synthetic=True,
            expected_adjustment="none",
            allow_invalid_primary_fallback=allow_invalid_primary_fallback,
            expected_trade_dates=expected_trade_dates,
            live_critical=live_critical,
            expected_latest_trade_date=expected_latest_trade_date,
            max_staleness_calendar_days=max_staleness_calendar_days,
        )


@dataclass(frozen=True)
class DailyOHLCVIntegrityResult:
    status: Literal["pass", "degraded", "failed"]
    source: str
    input_rows: int
    valid_rows: int
    quarantined_rows: int
    hard_violations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    requested_symbols: tuple[str, ...] = ()
    observed_symbols: tuple[str, ...] = ()
    observed_min_date: str | None = None
    observed_max_date: str | None = None
    duplicate_keys: int = 0
    expected_key_count: int | None = None
    observed_expected_key_count: int | None = None
    missing_expected_keys: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidatedDailyOHLCV:
    valid_frame: pd.DataFrame
    quarantine_frame: pd.DataFrame
    report: DailyOHLCVIntegrityResult


def _normalise_expected_dates(values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return ()
    parsed = pd.to_datetime(pd.Series(list(values)), errors="coerce")
    if parsed.isna().any():
        raise ValueError("expected_trade_dates contains an invalid date")
    return tuple(sorted({ts.date().isoformat() for ts in parsed}))


def expected_daily_keys(
    request: ProviderRequest,
    policy: DailyOHLCVIntegrityPolicy,
) -> set[tuple[str, str]]:
    dates = _normalise_expected_dates(policy.expected_trade_dates)
    if not request.symbols or not dates:
        return set()
    return {(str(symbol), date) for symbol in request.symbols for date in dates}


def daily_bar_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    if frame is None or frame.empty or not set(DAILY_IDENTITY_COLUMNS).issubset(frame.columns):
        return set()
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    keys: set[tuple[str, str]] = set()
    for symbol, ts in zip(frame["symbol"].astype(str), dates, strict=False):
        if pd.isna(ts):
            continue
        keys.add((str(symbol), pd.Timestamp(ts).date().isoformat()))
    return keys


def _metadata_value(result: ProviderResult, *keys: str) -> Any:
    for key in keys:
        if key in result.metadata:
            return result.metadata.get(key)
    return None


def _declared(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"unknown", "unspecified", "n/a", "none"}


def _looks_mock_or_synthetic(result: ProviderResult) -> bool:
    source = str(result.source).lower()
    metadata = result.metadata
    return bool(
        metadata.get("mock") is True
        or metadata.get("synthetic") is True
        or metadata.get("fallback") is True
        or "mock" in source
        or "synthetic" in source
    )


def validate_daily_ohlcv(
    result: ProviderResult,
    request: ProviderRequest,
    policy: DailyOHLCVIntegrityPolicy | None = None,
) -> ValidatedDailyOHLCV:
    policy = policy or DailyOHLCVIntegrityPolicy.research()
    frame = pd.DataFrame() if result.frame is None else result.frame.copy()
    hard: list[str] = []
    warnings: list[str] = []
    invalid = pd.Series(False, index=frame.index, dtype=bool)

    required = set(DAILY_IDENTITY_COLUMNS)
    if policy.require_full_ohlcv:
        required.update(DAILY_OHLC_COLUMNS)
        required.update(DAILY_FLOW_COLUMNS)
    if request.fields:
        required.update(str(field) for field in request.fields)
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        hard.extend(f"missing_column:{column}" for column in missing_columns)
        if not frame.empty:
            invalid.loc[:] = True

    if policy.require_point_in_time and result.point_in_time is not True:
        hard.append("provider_not_point_in_time")
    pit_semantics = _metadata_value(result, "pit_semantics", "point_in_time_semantics")
    if policy.require_pit_semantics:
        if not _declared(pit_semantics):
            hard.append("pit_semantics_missing")
            if not frame.empty:
                invalid.loc[:] = True
    elif not _declared(pit_semantics):
        warnings.append("pit_semantics_missing")

    if policy.reject_mock_or_synthetic and _looks_mock_or_synthetic(result):
        hard.append("mock_or_synthetic_source")

    frequency = _metadata_value(result, "frequency", "interval", "freq")
    if policy.require_frequency_metadata:
        if not _declared(frequency):
            hard.append("frequency_metadata_missing")
        elif str(frequency).strip().lower() not in _DAILY_FREQUENCIES:
            hard.append(f"frequency_not_daily:{frequency}")
    elif not _declared(frequency):
        warnings.append("frequency_metadata_missing")

    timezone = _metadata_value(result, "timezone", "time_zone", "tz")
    if policy.require_timezone_metadata and not _declared(timezone):
        hard.append("timezone_metadata_missing")
    elif not _declared(timezone):
        warnings.append("timezone_metadata_missing")

    volume_unit = _metadata_value(result, "volume_unit")
    amount_unit = _metadata_value(result, "amount_unit")
    if policy.require_declared_units:
        if not _declared(volume_unit):
            hard.append("volume_unit_missing")
        if not _declared(amount_unit):
            hard.append("amount_unit_missing")
    else:
        if not _declared(volume_unit):
            warnings.append("volume_unit_missing")
        if not _declared(amount_unit):
            warnings.append("amount_unit_missing")

    adjustment = _metadata_value(result, "adjust", "adjustment")
    if policy.expected_adjustment is not None:
        if not _declared(adjustment):
            hard.append("adjustment_metadata_missing")
        elif str(adjustment).strip().lower() != str(policy.expected_adjustment).strip().lower():
            hard.append(f"adjustment_mismatch:{adjustment}")

    if frame.empty:
        report = DailyOHLCVIntegrityResult(
            status="failed" if hard else "degraded",
            source=str(result.source),
            input_rows=0,
            valid_rows=0,
            quarantined_rows=0,
            hard_violations=tuple(dict.fromkeys(hard)),
            warnings=tuple(dict.fromkeys(warnings + ["empty_daily_frame"])),
            requested_symbols=tuple(str(s) for s in request.symbols),
            metadata={
                "frequency": frequency,
                "timezone": timezone,
                "volume_unit": volume_unit,
                "amount_unit": amount_unit,
                "adjustment": adjustment,
                "point_in_time": bool(result.point_in_time),
                "pit_semantics": pit_semantics,
                "mock_or_synthetic": _looks_mock_or_synthetic(result),
                "policy_mode": policy.mode,
            },
        )
        return ValidatedDailyOHLCV(frame, frame.copy(), report)

    identity_ready = set(DAILY_IDENTITY_COLUMNS).issubset(frame.columns)
    if identity_ready:
        symbols = frame["symbol"].astype("string")
        bad_symbol = symbols.isna() | symbols.str.strip().eq("")
        invalid |= bad_symbol.fillna(True)

        parsed_dates = pd.to_datetime(frame["trade_date"], errors="coerce")
        bad_date = parsed_dates.isna()
        invalid |= bad_date
        valid_dates = parsed_dates[~bad_date]
        non_midnight = (
            (valid_dates.dt.hour != 0)
            | (valid_dates.dt.minute != 0)
            | (valid_dates.dt.second != 0)
            | (valid_dates.dt.microsecond != 0)
        )
        if bool(non_midnight.any()):
            invalid.loc[non_midnight.index[non_midnight]] = True
            warnings.append("non_daily_trade_date_rows_quarantined")

        key_frame = pd.DataFrame(
            {
                "symbol": symbols.astype(str),
                "trade_date": parsed_dates.dt.date.astype("string"),
            },
            index=frame.index,
        )
        duplicate_mask = key_frame.duplicated(["symbol", "trade_date"], keep=False) & ~bad_date
        duplicate_count = int(duplicate_mask.sum())
        invalid |= duplicate_mask
        if duplicate_count:
            warnings.append("duplicate_symbol_trade_date_rows_quarantined")
    else:
        parsed_dates = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        duplicate_count = 0

    present_ohlc = [column for column in DAILY_OHLC_COLUMNS if column in frame.columns]
    numeric_cache: dict[str, pd.Series] = {}
    for column in present_ohlc:
        values = pd.to_numeric(frame[column], errors="coerce")
        numeric_cache[column] = values
        bad = values.isna() | ~np.isfinite(values) | (values <= 0)
        invalid |= bad
        if bool(bad.any()):
            warnings.append(f"invalid_{column}_rows_quarantined")

    for column in DAILY_FLOW_COLUMNS:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        numeric_cache[column] = values
        bad = values.isna() | ~np.isfinite(values) | (values < 0)
        invalid |= bad
        if bool(bad.any()):
            warnings.append(f"invalid_{column}_rows_quarantined")

    if "high" in numeric_cache and "low" in numeric_cache:
        bad = numeric_cache["high"] < numeric_cache["low"]
        invalid |= bad.fillna(True)
        if bool(bad.fillna(False).any()):
            warnings.append("high_below_low_rows_quarantined")
    if set(DAILY_OHLC_COLUMNS).issubset(numeric_cache):
        low = numeric_cache["low"]
        high = numeric_cache["high"]
        lower_body = pd.concat([numeric_cache["open"], numeric_cache["close"]], axis=1).min(axis=1)
        upper_body = pd.concat([numeric_cache["open"], numeric_cache["close"]], axis=1).max(axis=1)
        bad = (low > lower_body) | (upper_body > high)
        invalid |= bad.fillna(True)
        if bool(bad.fillna(False).any()):
            warnings.append("ohlc_relationship_rows_quarantined")

    if request.symbols and "symbol" in frame.columns:
        allowed_symbols = {str(symbol) for symbol in request.symbols}
        unexpected = ~frame["symbol"].astype(str).isin(allowed_symbols)
        invalid |= unexpected
        if bool(unexpected.any()):
            warnings.append("unexpected_symbol_rows_quarantined")

    quarantine = frame.loc[invalid].copy()
    valid = frame.loc[~invalid].copy()

    expected_keys = expected_daily_keys(request, policy)
    valid_keys = daily_bar_keys(valid)
    missing_expected = sorted(expected_keys - valid_keys)
    if expected_keys and missing_expected:
        warnings.append("expected_trade_date_coverage_incomplete")

    if policy.live_critical:
        if not policy.expected_latest_trade_date:
            hard.append("freshness_reference_missing")
        elif valid.empty or "trade_date" not in valid.columns:
            hard.append("freshness_unverifiable")
        else:
            observed = pd.to_datetime(valid["trade_date"], errors="coerce").dropna()
            expected_latest = pd.Timestamp(policy.expected_latest_trade_date).normalize()
            if observed.empty:
                hard.append("freshness_unverifiable")
            else:
                latest = pd.Timestamp(observed.max()).normalize()
                stale_days = int((expected_latest - latest).days)
                if stale_days < 0:
                    hard.append("trade_date_after_freshness_reference")
                elif stale_days > int(policy.max_staleness_calendar_days):
                    hard.append(f"stale_daily_data:{stale_days}d")

    if expected_keys and missing_expected and policy.mode == "production":
        hard.append("expected_trade_date_coverage_incomplete")

    if policy.mode == "production" and not quarantine.empty:
        hard.append("invalid_daily_rows_present")

    observed_dates = pd.to_datetime(
        valid.get("trade_date", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
    ).dropna()
    observed_symbols = tuple(sorted(set(valid.get("symbol", pd.Series(dtype=str)).astype(str))))
    hard = list(dict.fromkeys(hard))
    warnings = list(dict.fromkeys(warnings))
    if hard:
        status: Literal["pass", "degraded", "failed"] = "failed"
    elif warnings or not quarantine.empty:
        status = "degraded"
    else:
        status = "pass"

    report = DailyOHLCVIntegrityResult(
        status=status,
        source=str(result.source),
        input_rows=int(len(frame)),
        valid_rows=int(len(valid)),
        quarantined_rows=int(len(quarantine)),
        hard_violations=tuple(hard),
        warnings=tuple(warnings),
        requested_symbols=tuple(str(s) for s in request.symbols),
        observed_symbols=observed_symbols,
        observed_min_date=None if observed_dates.empty else pd.Timestamp(observed_dates.min()).date().isoformat(),
        observed_max_date=None if observed_dates.empty else pd.Timestamp(observed_dates.max()).date().isoformat(),
        duplicate_keys=duplicate_count,
        expected_key_count=len(expected_keys) if expected_keys else None,
        observed_expected_key_count=len(expected_keys & valid_keys) if expected_keys else None,
        missing_expected_keys=tuple(f"{symbol}@{date}" for symbol, date in missing_expected),
        metadata={
            "frequency": frequency,
            "timezone": timezone,
            "volume_unit": volume_unit,
            "amount_unit": amount_unit,
            "adjustment": adjustment,
            "point_in_time": bool(result.point_in_time),
            "pit_semantics": pit_semantics,
            "mock_or_synthetic": _looks_mock_or_synthetic(result),
            "policy_mode": policy.mode,
        },
    )
    return ValidatedDailyOHLCV(valid.reset_index(drop=True), quarantine.reset_index(drop=True), report)


__all__ = [
    "DAILY_FULL_COLUMNS",
    "DailyOHLCVIntegrityPolicy",
    "DailyOHLCVIntegrityResult",
    "ValidatedDailyOHLCV",
    "daily_bar_keys",
    "expected_daily_keys",
    "validate_daily_ohlcv",
]
