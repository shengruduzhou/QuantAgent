"""Date-aware walk-forward splitters for time-series training.

The V7 training pipeline already supports purged k-fold CV through
``quantagent.quant_math.purged_cv``. Real-data research workflows
typically want more options:

* **Expanding window** — anchored at the first trade date; the validation
  fold rolls forward and the training set grows with each step.
* **Rolling window** — fixed-size training window slid forward together
  with the validation fold (closer to how a deployed model retrains).
* **Purged walk-forward** — same as expanding/rolling but with an embargo
  gap and a label-purge at the boundary to prevent label leakage when the
  forward-return horizon overlaps the validation window.
* **Chronological hold-out** — single-shot train/val/test split, which
  matches how live readiness gates are evaluated.

All splitters operate on ``trade_date``-aware DataFrames and yield
``WalkForwardFold`` objects with positional indices, so callers can do
``frame.iloc[fold.train_idx]`` without re-sorting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    """One walk-forward fold expressed as positional row indices."""

    fold_id: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    train_dates: tuple[pd.Timestamp, pd.Timestamp]
    valid_dates: tuple[pd.Timestamp, pd.Timestamp]
    embargo_days: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "fold_id": int(self.fold_id),
            "train_idx_count": int(self.train_idx.size),
            "valid_idx_count": int(self.valid_idx.size),
            "train_start": str(self.train_dates[0].date()),
            "train_end": str(self.train_dates[1].date()),
            "valid_start": str(self.valid_dates[0].date()),
            "valid_end": str(self.valid_dates[1].date()),
            "embargo_days": int(self.embargo_days),
        }


@dataclass(frozen=True)
class WalkForwardSplitConfig:
    n_splits: int = 4
    valid_size_days: int = 60
    min_train_days: int = 120
    embargo_days: int = 5
    purge_days: int = 0
    mode: str = "expanding"  # "expanding" | "rolling" | "purged" | "chronological"
    rolling_train_days: int = 252


def split_walk_forward(
    frame: pd.DataFrame,
    date_column: str = "trade_date",
    config: WalkForwardSplitConfig | None = None,
) -> list[WalkForwardFold]:
    """Build walk-forward folds against ``frame[date_column]``.

    Each fold's training set ends at least ``embargo_days + purge_days``
    trading days before the validation set starts. ``purge_days`` accounts
    for label horizons that overlap the validation window — pass it as
    ``max(horizons_days)`` to avoid label leakage entirely.
    """
    if frame is None or frame.empty:
        return []
    cfg = config or WalkForwardSplitConfig()
    if cfg.mode not in {"expanding", "rolling", "purged", "chronological"}:
        raise ValueError(f"unknown walk-forward mode: {cfg.mode}")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().all():
        return []
    sorted_dates = pd.Series(dates.dropna().unique()).sort_values().reset_index(drop=True)
    if cfg.mode == "chronological":
        return _chronological(frame, dates, sorted_dates, cfg)

    folds: list[WalkForwardFold] = []
    total_days = len(sorted_dates)
    if total_days < cfg.min_train_days + cfg.valid_size_days:
        return []
    cursor = cfg.min_train_days
    fold_id = 0
    while fold_id < cfg.n_splits and cursor + cfg.valid_size_days <= total_days:
        train_end_pos = cursor - 1 - cfg.embargo_days - cfg.purge_days
        if train_end_pos < 0:
            break
        if cfg.mode == "rolling":
            train_start_pos = max(0, train_end_pos - cfg.rolling_train_days + 1)
        else:
            train_start_pos = 0
        valid_start_pos = cursor
        valid_end_pos = min(cursor + cfg.valid_size_days - 1, total_days - 1)
        train_start = sorted_dates.iloc[train_start_pos]
        train_end = sorted_dates.iloc[train_end_pos]
        valid_start = sorted_dates.iloc[valid_start_pos]
        valid_end = sorted_dates.iloc[valid_end_pos]
        train_mask = (dates >= train_start) & (dates <= train_end)
        valid_mask = (dates >= valid_start) & (dates <= valid_end)
        train_idx = np.flatnonzero(train_mask.to_numpy(dtype=bool))
        valid_idx = np.flatnonzero(valid_mask.to_numpy(dtype=bool))
        if train_idx.size and valid_idx.size:
            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_idx=train_idx,
                    valid_idx=valid_idx,
                    train_dates=(pd.Timestamp(train_start), pd.Timestamp(train_end)),
                    valid_dates=(pd.Timestamp(valid_start), pd.Timestamp(valid_end)),
                    embargo_days=cfg.embargo_days + cfg.purge_days,
                )
            )
        fold_id += 1
        cursor += cfg.valid_size_days
    return folds


def _chronological(
    frame: pd.DataFrame,
    dates: pd.Series,
    sorted_dates: pd.Series,
    cfg: WalkForwardSplitConfig,
) -> list[WalkForwardFold]:
    total = len(sorted_dates)
    if total < cfg.min_train_days + cfg.valid_size_days:
        return []
    train_end_pos = total - cfg.valid_size_days - cfg.embargo_days - 1
    if train_end_pos < cfg.min_train_days - 1:
        train_end_pos = cfg.min_train_days - 1
    valid_start_pos = train_end_pos + cfg.embargo_days + 1
    if valid_start_pos >= total:
        return []
    train_start = sorted_dates.iloc[0]
    train_end = sorted_dates.iloc[train_end_pos]
    valid_start = sorted_dates.iloc[valid_start_pos]
    valid_end = sorted_dates.iloc[-1]
    train_mask = (dates >= train_start) & (dates <= train_end)
    valid_mask = (dates >= valid_start) & (dates <= valid_end)
    train_idx = np.flatnonzero(train_mask.to_numpy(dtype=bool))
    valid_idx = np.flatnonzero(valid_mask.to_numpy(dtype=bool))
    if not train_idx.size or not valid_idx.size:
        return []
    return [
        WalkForwardFold(
            fold_id=0,
            train_idx=train_idx,
            valid_idx=valid_idx,
            train_dates=(pd.Timestamp(train_start), pd.Timestamp(train_end)),
            valid_dates=(pd.Timestamp(valid_start), pd.Timestamp(valid_end)),
            embargo_days=cfg.embargo_days,
        )
    ]


def iter_folds_with_data(
    frame: pd.DataFrame,
    config: WalkForwardSplitConfig | None = None,
    date_column: str = "trade_date",
) -> Iterator[tuple[WalkForwardFold, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(fold, train_frame, valid_frame)`` triples for convenience."""
    for fold in split_walk_forward(frame, date_column=date_column, config=config):
        yield fold, frame.iloc[fold.train_idx].copy(), frame.iloc[fold.valid_idx].copy()


__all__ = [
    "WalkForwardFold",
    "WalkForwardSplitConfig",
    "split_walk_forward",
    "iter_folds_with_data",
]
