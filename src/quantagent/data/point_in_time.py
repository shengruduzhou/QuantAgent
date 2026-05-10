from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class PITConfig:
    event_cutoff: str = "15:00:00"
    date_column: str = "trade_date"
    symbol_column: str = "symbol"


class PITJoiner:
    """Point-in-time joins for fundamentals, events, and universe snapshots."""

    def __init__(self, config: PITConfig | None = None) -> None:
        self.config = config or PITConfig()
        self.cutoff_time = _parse_time(self.config.event_cutoff)

    def panel_timestamp(self, dates: pd.Series) -> pd.Series:
        values = pd.to_datetime(dates)
        return values.dt.normalize() + pd.to_timedelta(
            self.cutoff_time.hour * 3600 + self.cutoff_time.minute * 60 + self.cutoff_time.second,
            unit="s",
        )

    def join_fundamentals(
        self,
        panel: pd.DataFrame,
        fundamentals: pd.DataFrame,
        announcement_column: str = "announcement_time",
        report_period_column: str = "report_period",
        value_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        del report_period_column
        if fundamentals.empty:
            return panel.copy()
        values = list(value_columns) if value_columns is not None else [
            c for c in fundamentals.columns if c not in {self.config.symbol_column, announcement_column}
        ]
        left = self._left_frame(panel)
        right = fundamentals[[self.config.symbol_column, announcement_column, *values]].copy()
        right[announcement_column] = pd.to_datetime(right[announcement_column])
        right = right.sort_values([self.config.symbol_column, announcement_column])
        return _merge_asof_by_symbol(
            left,
            right,
            left_on="_pit_timestamp",
            right_on=announcement_column,
            symbol_column=self.config.symbol_column,
        ).drop(columns=["_pit_timestamp"])

    def join_events(
        self,
        panel: pd.DataFrame,
        events: pd.DataFrame,
        event_time_column: str = "event_time",
        value_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        if events.empty:
            return panel.copy()
        values = list(value_columns) if value_columns is not None else [
            c for c in events.columns if c not in {self.config.symbol_column, event_time_column}
        ]
        left = self._left_frame(panel)
        right = events[[self.config.symbol_column, event_time_column, *values]].copy()
        right[event_time_column] = pd.to_datetime(right[event_time_column])
        right = right.sort_values([self.config.symbol_column, event_time_column])
        return _merge_asof_by_symbol(
            left,
            right,
            left_on="_pit_timestamp",
            right_on=event_time_column,
            symbol_column=self.config.symbol_column,
        ).drop(columns=["_pit_timestamp"])

    def apply_universe_snapshot(
        self,
        panel: pd.DataFrame,
        universe: pd.DataFrame,
        snapshot_time_column: str = "snapshot_time",
        member_column: str = "is_member",
    ) -> pd.DataFrame:
        joined = self.join_events(
            panel,
            universe.rename(columns={snapshot_time_column: "event_time"}),
            event_time_column="event_time",
            value_columns=[member_column],
        )
        return joined[joined[member_column].fillna(False)].reset_index(drop=True)

    def _left_frame(self, panel: pd.DataFrame) -> pd.DataFrame:
        frame = panel.copy()
        frame[self.config.date_column] = pd.to_datetime(frame[self.config.date_column])
        frame["_pit_timestamp"] = self.panel_timestamp(frame[self.config.date_column])
        return frame.sort_values([self.config.symbol_column, "_pit_timestamp"]).reset_index(drop=True)


def _merge_asof_by_symbol(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    symbol_column: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for symbol, left_group in left.groupby(symbol_column, sort=False):
        right_group = right[right[symbol_column] == symbol].sort_values(right_on)
        if right_group.empty:
            missing = left_group.copy()
            for column in right.columns:
                if column not in missing.columns and column != symbol_column:
                    missing[column] = pd.NA
            parts.append(missing)
            continue
        merged = pd.merge_asof(
            left_group.sort_values(left_on),
            right_group.sort_values(right_on).drop(columns=[symbol_column]),
            left_on=left_on,
            right_on=right_on,
            direction="backward",
        )
        merged[symbol_column] = symbol
        parts.append(merged)
    if not parts:
        return left.copy()
    return pd.concat(parts, ignore_index=True).sort_values([left_on, symbol_column]).reset_index(drop=True)


def _parse_time(value: str) -> time:
    parsed = pd.to_datetime(value).time()
    return time(parsed.hour, parsed.minute, parsed.second)
