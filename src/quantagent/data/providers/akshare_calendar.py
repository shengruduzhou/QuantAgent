"""Shared AKShare research-calendar adapter.

The AKShare/Sina trading-date endpoint is useful as an independent session set
for research-provider availability. It is deliberately **not** promoted to an
exchange-authoritative production calendar: every consumer must retain
``production_certified=False`` until #71 is satisfied by governed exchange or
otherwise authoritative evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from quantagent.data.trading_calendar import TradingCalendar


AKSHARE_RESEARCH_CALENDAR_SOURCE = "akshare:tool_trade_date_hist_sina"


@dataclass(frozen=True, slots=True)
class AkShareCalendarEvidence:
    calendar: TradingCalendar
    metadata: dict[str, object]
    warnings: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return not self.calendar.empty


def load_akshare_research_calendar(
    *,
    allow_network: bool,
    ak_module: object | None = None,
) -> AkShareCalendarEvidence:
    """Fetch and validate AKShare's Sina A-share trading-date set.

    Failure returns an empty calendar rather than manufacturing weekdays. The
    caller can retain raw/descriptive rows, but must not certify them as PIT
    when next-session availability cannot be resolved.
    """
    base_meta: dict[str, object] = {
        "source": AKSHARE_RESEARCH_CALENDAR_SOURCE,
        "production_certified": False,
    }
    if not allow_network:
        return AkShareCalendarEvidence(
            TradingCalendar.from_dates(()),
            {**base_meta, "status": "network_disabled"},
            ("akshare_calendar_network_disabled",),
        )
    ak = ak_module
    if ak is None:
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            return AkShareCalendarEvidence(
                TradingCalendar.from_dates(()),
                {**base_meta, "status": "package_unavailable"},
                (f"akshare_calendar_package_unavailable:{type(exc).__name__}",),
            )
    api = getattr(ak, "tool_trade_date_hist_sina", None)
    version = str(getattr(ak, "__version__", "unknown"))
    if api is None:
        return AkShareCalendarEvidence(
            TradingCalendar.from_dates(()),
            {**base_meta, "status": "api_unavailable", "akshare_version": version},
            ("akshare_calendar_api_unavailable",),
        )
    try:
        raw = api()
    except Exception as exc:  # pragma: no cover - network path
        return AkShareCalendarEvidence(
            TradingCalendar.from_dates(()),
            {**base_meta, "status": "request_failed", "akshare_version": version},
            (f"akshare_calendar_request_failed:{type(exc).__name__}:{exc}",),
        )
    if raw is None or raw.empty or "trade_date" not in raw.columns:
        return AkShareCalendarEvidence(
            TradingCalendar.from_dates(()),
            {**base_meta, "status": "empty_or_invalid", "akshare_version": version},
            ("akshare_calendar_empty_or_invalid",),
        )
    parsed = pd.to_datetime(raw["trade_date"], errors="coerce").dropna().dt.normalize()
    if parsed.empty:
        return AkShareCalendarEvidence(
            TradingCalendar.from_dates(()),
            {**base_meta, "status": "no_valid_dates", "akshare_version": version},
            ("akshare_calendar_no_valid_dates",),
        )
    if parsed.duplicated().any():
        # Duplicate sessions are ambiguous evidence. Fail closed rather than
        # silently de-duplicating a malformed upstream response.
        return AkShareCalendarEvidence(
            TradingCalendar.from_dates(()),
            {**base_meta, "status": "duplicate_sessions", "akshare_version": version},
            ("akshare_calendar_duplicate_sessions",),
        )
    if not parsed.is_monotonic_increasing:
        parsed = parsed.sort_values()
    calendar = TradingCalendar.from_dates(parsed)
    return AkShareCalendarEvidence(
        calendar,
        {
            **base_meta,
            "status": "passed",
            "akshare_version": version,
            "session_count": int(len(calendar.trading_days)),
            "first_session": str(calendar.trading_days[0].date()),
            "last_session": str(calendar.trading_days[-1].date()),
        },
        (),
    )


def next_session_available_at(
    dates: Iterable[object],
    calendar: TradingCalendar | None,
    *,
    lag_sessions: int = 1,
) -> pd.Series:
    """Resolve availability to an actual later session, never a weekday guess."""
    values = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dt.normalize()
    if calendar is None or calendar.empty:
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    resolved = [
        calendar.next_trading_day(value, lag_days=lag_sessions)
        if pd.notna(value)
        else pd.NaT
        for value in values
    ]
    return pd.Series(resolved, index=values.index, dtype="datetime64[ns]")


__all__ = [
    "AKSHARE_RESEARCH_CALENDAR_SOURCE",
    "AkShareCalendarEvidence",
    "load_akshare_research_calendar",
    "next_session_available_at",
]
