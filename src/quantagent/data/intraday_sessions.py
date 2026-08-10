"""Exchange-session aware intraday bars and execution-price provenance.

This module is the canonical A-share intraday clock used by research replay and
execution simulation. It intentionally refuses global wall-clock resampling:
Shanghai/Shenzhen cash-equity trading has a lunch break, so a 60-minute bar must
never combine 11:30 with 13:00 data.

Input bars are expected to be *end-labelled* observations. Naive timestamps are
accepted only as explicit exchange-local source timestamps; mixing naive and
timezone-aware values is rejected. Output timestamps are always Asia/Shanghai
aware. Execution-facing consumers must additionally pass
:func:`assert_raw_execution_prices`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.domain.timeline import (
    AFTERNOON_OPEN,
    EXCHANGE_TZ,
    MORNING_CLOSE,
    MORNING_OPEN,
    SESSION_CLOSE,
)

SUPPORTED_BAR_MINUTES = (5, 10, 60)
ADJUSTFLAG_TO_PRICE_ADJUSTMENT = {"1": "qfq", "2": "hfq", "3": "raw"}
VALID_PRICE_ADJUSTMENTS = frozenset({"raw", "qfq", "hfq"})


class IntradaySessionError(ValueError):
    """The intraday feed violates the canonical A-share session contract."""


class ExecutionPriceProvenanceError(ValueError):
    """Execution simulation was given adjusted or unverifiable prices."""


@dataclass(frozen=True, slots=True)
class SessionWindow:
    name: str
    start: time
    end: time


ASHARE_SESSION_WINDOWS: tuple[SessionWindow, ...] = (
    SessionWindow("morning", MORNING_OPEN, MORNING_CLOSE),
    SessionWindow("afternoon", AFTERNOON_OPEN, SESSION_CLOSE),
)


def _timestamp_awareness(values: Iterable[object]) -> set[bool]:
    states: set[bool] = set()
    for value in values:
        if pd.isna(value):
            continue
        stamp = pd.Timestamp(value)
        states.add(stamp.tzinfo is not None and stamp.utcoffset() is not None)
    return states


def to_exchange_timestamps(
    values: pd.Series,
    *,
    naive_timezone: str = "Asia/Shanghai",
) -> pd.Series:
    """Return Asia/Shanghai-aware timestamps and reject mixed timezone semantics.

    Vendor A-share bars commonly encode exchange-local timestamps without a
    timezone. Those are localised using ``naive_timezone``. A column containing a
    mixture of naive and aware values is rejected instead of guessed row-by-row.
    """

    awareness = _timestamp_awareness(values)
    if len(awareness) > 1:
        raise IntradaySessionError(
            "mixed naive/timezone-aware intraday timestamps are forbidden"
        )
    if not awareness or awareness == {False}:
        parsed = pd.to_datetime(values, errors="coerce")
        if parsed.isna().any():
            raise IntradaySessionError("intraday timestamps contain invalid/NaT values")
        try:
            return parsed.dt.tz_localize(naive_timezone).dt.tz_convert(EXCHANGE_TZ)
        except (TypeError, ValueError) as exc:
            raise IntradaySessionError(f"cannot localize intraday timestamps: {exc}") from exc

    converted: list[pd.Timestamp] = []
    for value in values:
        if pd.isna(value):
            raise IntradaySessionError("intraday timestamps contain invalid/NaT values")
        try:
            converted.append(pd.Timestamp(value).tz_convert(EXCHANGE_TZ))
        except (TypeError, ValueError) as exc:
            raise IntradaySessionError(f"cannot convert intraday timestamps: {exc}") from exc
    return pd.Series(converted, index=values.index)


def _canonical_adjustment(frame: pd.DataFrame) -> pd.Series:
    if "price_adjustment" in frame.columns:
        result = frame["price_adjustment"].astype("string").str.strip().str.lower()
    elif "adjustflag" in frame.columns:
        flags = frame["adjustflag"].astype("string").str.strip()
        result = flags.map(ADJUSTFLAG_TO_PRICE_ADJUSTMENT).astype("string")
    else:
        return pd.Series("unknown", index=frame.index, dtype="string")
    return result.fillna("unknown")


def attach_price_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical adjustment and execution eligibility without guessing."""

    out = frame.copy()
    out["price_adjustment"] = _canonical_adjustment(out)
    out["execution_eligible"] = out["price_adjustment"].eq("raw")
    return out


def assert_raw_execution_prices(frame: pd.DataFrame) -> None:
    """Fail closed unless every row is provably raw/unadjusted execution data."""

    if frame is None or frame.empty:
        return
    adjustment = _canonical_adjustment(frame)
    observed = sorted(set(adjustment.astype(str)))
    if observed != ["raw"]:
        raise ExecutionPriceProvenanceError(
            "execution requires canonical raw/unadjusted prices; "
            f"observed price_adjustment={observed}. "
            "Adjusted qfq/hfq series are research features, never fill prices."
        )
    if "execution_eligible" in frame.columns:
        eligible = frame["execution_eligible"].fillna(False).astype(bool)
        if not bool(eligible.all()):
            raise ExecutionPriceProvenanceError(
                "execution_eligible must be true for every execution-price row"
            )


def _session_assignment(stamp: pd.Timestamp) -> SessionWindow | None:
    clock = stamp.timetz().replace(tzinfo=None)
    for window in ASHARE_SESSION_WINDOWS:
        # Input bars are end-labelled, so the opening boundary itself is not a
        # completed bar while the close boundary is.
        if window.start < clock <= window.end:
            return window
    return None


def _ceil_window_end(
    stamp: pd.Timestamp,
    *,
    session_start: pd.Timestamp,
    minutes: int,
) -> pd.Timestamp:
    delta_minutes = (stamp - session_start).total_seconds() / 60.0
    slot = max(1, int(np.ceil(delta_minutes / float(minutes) - 1e-12)))
    return session_start + pd.Timedelta(minutes=slot * minutes)


def aggregate_ashare_bars(
    frame: pd.DataFrame,
    *,
    minutes: int,
    timestamp_column: str = "timestamp",
    emit_partial: bool = False,
    naive_timezone: str = "Asia/Shanghai",
) -> pd.DataFrame:
    """Aggregate end-labelled OHLCV bars on A-share exchange-session boundaries.

    Semantics are right-labelled/right-closed, session anchored and lunch-break
    aware. ``emit_partial=False`` drops a target window whose closing timestamp
    has not yet been observed for that symbol/session; this is suitable for both
    historical replay and live bar generation.
    """

    minutes = int(minutes)
    if minutes not in SUPPORTED_BAR_MINUTES:
        raise IntradaySessionError(
            f"unsupported A-share bar size {minutes}; expected {SUPPORTED_BAR_MINUTES}"
        )
    if frame is None or frame.empty:
        return pd.DataFrame()
    required = {
        "symbol",
        timestamp_column,
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise IntradaySessionError(f"intraday frame missing columns: {missing}")

    work = attach_price_provenance(frame)
    work["__timestamp"] = to_exchange_timestamps(
        work[timestamp_column],
        naive_timezone=naive_timezone,
    )
    work["symbol"] = work["symbol"].astype(str)
    for column in ("open", "high", "low", "close", "volume"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if "amount" in work.columns:
        work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    else:
        if not bool(work["price_adjustment"].eq("raw").all()):
            raise IntradaySessionError(
                "adjusted intraday bars require provider-supplied raw amount; "
                "turnover cannot be reconstructed from adjusted close × volume"
            )
        work["amount"] = work["volume"] * work["close"]
    if work[["open", "high", "low", "close", "volume", "amount"]].isna().any().any():
        raise IntradaySessionError("intraday OHLCVA contains non-numeric/NaN values")

    session_names: list[str | None] = []
    session_starts: list[pd.Timestamp | pd.NaT] = []
    session_ends: list[pd.Timestamp | pd.NaT] = []
    bar_ends: list[pd.Timestamp | pd.NaT] = []
    for stamp in work["__timestamp"]:
        window = _session_assignment(stamp)
        if window is None:
            session_names.append(None)
            session_starts.append(pd.NaT)
            session_ends.append(pd.NaT)
            bar_ends.append(pd.NaT)
            continue
        local_date = stamp.date()
        start = pd.Timestamp.combine(local_date, window.start).tz_localize(EXCHANGE_TZ)
        end = pd.Timestamp.combine(local_date, window.end).tz_localize(EXCHANGE_TZ)
        bar_end = _ceil_window_end(stamp, session_start=start, minutes=minutes)
        if bar_end > end:
            session_names.append(None)
            session_starts.append(pd.NaT)
            session_ends.append(pd.NaT)
            bar_ends.append(pd.NaT)
            continue
        session_names.append(window.name)
        session_starts.append(start)
        session_ends.append(end)
        bar_ends.append(bar_end)

    work["session"] = session_names
    work["__session_start"] = session_starts
    work["__session_end"] = session_ends
    work["__bar_end"] = bar_ends
    work = work.dropna(subset=["session", "__bar_end"]).copy()
    if work.empty:
        return pd.DataFrame()

    max_observed = (
        work.groupby(["symbol", "__session_start"], sort=False)["__timestamp"]
        .max()
        .rename("__max_observed")
        .reset_index()
    )

    keys = ["symbol", "__session_start", "__session_end", "__bar_end", "session"]
    aggregations: dict[str, object] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "amount": "sum",
        "price_adjustment": lambda s: (
            s.iloc[0] if s.nunique(dropna=False) == 1 else "mixed"
        ),
        "execution_eligible": "all",
        "__timestamp": "max",
    }
    out = (
        work.sort_values(["symbol", "__timestamp"])
        .groupby(keys, sort=True, observed=True)
        .agg(aggregations)
        .reset_index()
        .rename(columns={"__bar_end": "timestamp", "__timestamp": "last_source_timestamp"})
    )
    out = out.merge(max_observed, on=["symbol", "__session_start"], how="left")
    if not emit_partial:
        out = out[out["timestamp"] <= out["__max_observed"]].copy()

    if (out["price_adjustment"] == "mixed").any():
        raise IntradaySessionError(
            "cannot aggregate bars with mixed raw/qfq/hfq adjustment inside one window"
        )

    out["bar_start"] = out["timestamp"] - pd.Timedelta(minutes=minutes)
    out["trade_date"] = pd.to_datetime(
        out["timestamp"].dt.tz_localize(None).dt.date
    )
    out["bar_minutes"] = minutes
    out["timezone"] = str(EXCHANGE_TZ)
    out["available_at"] = out["timestamp"]
    out["execution_eligible"] = (
        out["execution_eligible"].astype(bool)
        & out["price_adjustment"].eq("raw")
    )

    columns = [
        "symbol",
        "trade_date",
        "timestamp",
        "bar_start",
        "session",
        "bar_minutes",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "price_adjustment",
        "execution_eligible",
        "available_at",
        "timezone",
        "last_source_timestamp",
    ]
    return out[columns].sort_values(["symbol", "timestamp"]).reset_index(drop=True)


__all__ = [
    "ADJUSTFLAG_TO_PRICE_ADJUSTMENT",
    "ASHARE_SESSION_WINDOWS",
    "ExecutionPriceProvenanceError",
    "IntradaySessionError",
    "SUPPORTED_BAR_MINUTES",
    "aggregate_ashare_bars",
    "assert_raw_execution_prices",
    "attach_price_provenance",
    "to_exchange_timestamps",
]
