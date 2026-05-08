from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date


def split_by_date(
    frame: pd.DataFrame,
    split: TimeSplit,
    date_column: str = "trade_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(frame[date_column]).dt.date
    train_mask = (dates >= split.train_start) & (dates <= split.train_end)
    valid_mask = (dates >= split.valid_start) & (dates <= split.valid_end)
    return frame.loc[train_mask].copy(), frame.loc[valid_mask].copy()
