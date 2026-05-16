"""Trading-calendar based ``available_at`` resolver.

Most A-share financial filings are published outside trading hours, which
means the announcement is only **actionable** from the next trading session
onward. A naive ``available_at = ann_date + N calendar days`` resolver lets
weekend / holiday announcements appear on non-trading days, breaks as-of
joins keyed on ``trade_date``, and can also under-shoot when a public
holiday follows the filing.

``TradingCalendar`` resolves an ``available_at`` to the next trading day on
or after the chosen lag, with two construction paths:

* ``TradingCalendar.from_market_panel`` derives the calendar from any frame
  that contains ``trade_date`` (e.g. the silver market panel). This keeps
  the calendar in sync with the actual data the pipeline already has.
* ``TradingCalendar.from_dates`` accepts an explicit list of trading days,
  for tests or when an external calendar is provided.

The resolver never invents trading days. If the input is later than the
last known trading day in the calendar it returns ``NaT`` so downstream
PIT checks can flag the row instead of silently leaking a future entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradingCalendar:
    trading_days: tuple[pd.Timestamp, ...]

    @classmethod
    def from_dates(cls, dates: Iterable[str | pd.Timestamp]) -> "TradingCalendar":
        parsed = pd.to_datetime(list(dates), errors="coerce")
        unique = sorted({ts.normalize() for ts in parsed if not pd.isna(ts)})
        return cls(tuple(unique))

    @classmethod
    def from_market_panel(cls, frame: pd.DataFrame, column: str = "trade_date") -> "TradingCalendar":
        if frame is None or frame.empty or column not in frame.columns:
            return cls(())
        return cls.from_dates(frame[column].dropna().unique())

    @property
    def empty(self) -> bool:
        return not self.trading_days

    def next_trading_day(self, date: str | pd.Timestamp, lag_days: int = 0) -> pd.Timestamp:
        """Return the earliest trading day strictly on or after ``date + lag_days``.

        ``lag_days`` is measured in calendar days; the function then snaps
        forward to the next trading session. ``NaT`` propagates.
        """
        if self.empty:
            return pd.to_datetime(date, errors="coerce") + pd.Timedelta(days=int(lag_days))
        target = pd.to_datetime(date, errors="coerce")
        if pd.isna(target):
            return pd.NaT
        target = target.normalize() + timedelta(days=int(lag_days))
        idx = np.searchsorted([ts.value for ts in self.trading_days], target.value, side="left")
        if idx >= len(self.trading_days):
            return pd.NaT
        return self.trading_days[idx]

    def resolve_available_at(
        self,
        ann_dates: pd.Series,
        lag_days: int = 1,
    ) -> pd.Series:
        """Vectorised resolver for a Series of announcement dates."""
        if self.empty:
            parsed = pd.to_datetime(ann_dates, errors="coerce") + pd.Timedelta(days=int(lag_days))
            return parsed
        values = pd.to_datetime(ann_dates, errors="coerce")
        return pd.Series(
            [self.next_trading_day(value, lag_days) if not pd.isna(value) else pd.NaT for value in values],
            index=values.index,
            dtype="datetime64[ns]",
        )

    def contains(self, date: str | pd.Timestamp) -> bool:
        target = pd.to_datetime(date, errors="coerce")
        if pd.isna(target):
            return False
        target = target.normalize()
        idx = np.searchsorted([ts.value for ts in self.trading_days], target.value, side="left")
        if idx >= len(self.trading_days):
            return False
        return self.trading_days[idx] == target


def calendar_day_resolver(ann_dates: pd.Series, lag_days: int = 1) -> pd.Series:
    """Legacy calendar-day resolver kept for the AkShare normaliser fallback."""
    return pd.to_datetime(ann_dates, errors="coerce") + pd.Timedelta(days=int(lag_days))
