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
    #: Trading days of *training* data guaranteed to every fold. This is
    #: exclusive of the embargo/purge gap — the gap is reserved on top of it,
    #: never carved out of it.
    min_train_days: int = 120
    embargo_days: int = 5
    purge_days: int = 0
    mode: str = "expanding"  # "expanding" | "rolling" | "purged" | "chronological"
    rolling_train_days: int = 252
    #: ``"end"`` makes the last fold finish on the last available date, so
    #: ``n_splits`` folds describe the most recent OOS span. ``"start"`` keeps
    #: the legacy behaviour of beginning as early as the data allows, which is
    #: only useful for learning-curve studies.
    anchor: str = "end"
    #: Pin the last validation day instead of letting it fall on the frame's
    #: final date. Multi-horizon training needs this: a 120-day label runs out
    #: 120 days earlier than a 1-day label, so without a shared anchor each
    #: horizon validates a different window and a cross-horizon blend has no
    #: dates in common. Ignored when ``anchor="start"``.
    anchor_end: pd.Timestamp | None = None


@dataclass(frozen=True)
class WalkForwardPlan:
    """What a walk-forward request can actually deliver on a given date span.

    Fold count is pure arithmetic over the number of distinct trading days, so
    it is knowable before any model is fit. Callers that would otherwise
    discover a shortfall only after paying for training use this to fail fast.
    """

    total_days: int
    requested_splits: int
    achievable_splits: int
    valid_size_days: int
    #: Trading days consumed before the first validation day can start.
    lead_days: int

    @property
    def oos_days(self) -> int:
        return int(self.achievable_splits * self.valid_size_days)

    @property
    def requested_oos_days(self) -> int:
        return int(self.requested_splits * self.valid_size_days)

    @property
    def is_satisfiable(self) -> bool:
        return self.achievable_splits >= self.requested_splits

    def days_needed_for(self, splits: int) -> int:
        """Distinct trading days required to deliver ``splits`` folds."""
        return int(self.lead_days + max(0, splits) * self.valid_size_days)

    def as_dict(self) -> dict[str, object]:
        return {
            "total_days": int(self.total_days),
            "requested_splits": int(self.requested_splits),
            "achievable_splits": int(self.achievable_splits),
            "valid_size_days": int(self.valid_size_days),
            "lead_days": int(self.lead_days),
            "oos_days": self.oos_days,
            "requested_oos_days": self.requested_oos_days,
            "is_satisfiable": self.is_satisfiable,
        }


def plan_walk_forward(total_days: int, config: WalkForwardSplitConfig | None = None) -> WalkForwardPlan:
    """Resolve a fold request against a date span without touching the data.

    ``split_walk_forward`` uses this to size its loop, and the web/CLI preflight
    uses it to tell an operator up front how many folds a panel can support.
    Both therefore quote the same number.
    """
    cfg = config or WalkForwardSplitConfig()
    valid_size = max(1, int(cfg.valid_size_days))
    lead = max(0, int(cfg.min_train_days)) + max(0, int(cfg.embargo_days)) + max(0, int(cfg.purge_days))
    usable = max(0, int(total_days) - lead)
    achievable = min(max(0, int(cfg.n_splits)), usable // valid_size)
    return WalkForwardPlan(
        total_days=int(total_days),
        requested_splits=max(0, int(cfg.n_splits)),
        achievable_splits=int(achievable),
        valid_size_days=valid_size,
        lead_days=int(lead),
    )


def split_walk_forward(
    frame: pd.DataFrame,
    date_column: str = "trade_date",
    config: WalkForwardSplitConfig | None = None,
) -> list[WalkForwardFold]:
    """Build walk-forward folds against ``frame[date_column]``.

    Each fold's training set holds at least ``min_train_days`` trading days and
    ends ``embargo_days + purge_days`` trading days before the validation set
    starts. ``purge_days`` accounts for label horizons that overlap the
    validation window — pass it as ``max(horizons_days)`` to avoid label
    leakage entirely.

    With the default ``anchor="end"`` the returned folds are the *last*
    ``n_splits`` windows of the sample, so a caller asking for 5 folds on a
    ten-year panel validates the most recent 5 windows rather than five windows
    from a decade ago.
    """
    if frame is None or frame.empty:
        return []
    cfg = config or WalkForwardSplitConfig()
    if cfg.mode not in {"expanding", "rolling", "purged", "chronological"}:
        raise ValueError(f"unknown walk-forward mode: {cfg.mode}")
    if cfg.anchor not in {"end", "start"}:
        raise ValueError(f"unknown walk-forward anchor: {cfg.anchor}")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().all():
        return []
    sorted_dates = pd.Series(dates.dropna().unique()).sort_values().reset_index(drop=True)
    if cfg.mode == "chronological":
        return _chronological(frame, dates, sorted_dates, cfg)

    folds: list[WalkForwardFold] = []
    total_days = len(sorted_dates)
    # An explicit anchor shortens the span the folds may occupy. Dates past it
    # are still in the frame — they are simply not eligible to be validated on,
    # because a longer-horizon label does not exist that far forward.
    usable_days = total_days
    if cfg.anchor == "end" and cfg.anchor_end is not None:
        position = int(sorted_dates.searchsorted(pd.Timestamp(cfg.anchor_end), side="right"))
        usable_days = max(0, min(total_days, position))
    plan = plan_walk_forward(usable_days, cfg)
    if plan.achievable_splits <= 0:
        return []
    # Anchoring at the end keeps the final validation day on the last usable
    # date; the surplus history in front of it simply becomes training data.
    if cfg.anchor == "end":
        cursor = usable_days - plan.achievable_splits * cfg.valid_size_days
    else:
        cursor = plan.lead_days
    fold_id = 0
    while fold_id < plan.achievable_splits and cursor + cfg.valid_size_days <= usable_days:
        train_end_pos = cursor - 1 - cfg.embargo_days - cfg.purge_days
        if train_end_pos < 0:
            break
        if cfg.mode == "rolling":
            train_start_pos = max(0, train_end_pos - cfg.rolling_train_days + 1)
        else:
            train_start_pos = 0
        valid_start_pos = cursor
        valid_end_pos = min(cursor + cfg.valid_size_days - 1, usable_days - 1)
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
    # The gap has to hold the label horizon as well as the embargo. Sizing it
    # from ``embargo_days`` alone let a 120-day forward return computed inside
    # the training window resolve into the validation window — the training
    # labels then encode the very period being validated.
    gap = max(0, cfg.embargo_days) + max(0, cfg.purge_days)
    if total < cfg.min_train_days + gap + cfg.valid_size_days:
        return []
    train_end_pos = total - cfg.valid_size_days - gap - 1
    if train_end_pos < cfg.min_train_days - 1:
        train_end_pos = cfg.min_train_days - 1
    valid_start_pos = train_end_pos + gap + 1
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
            embargo_days=gap,
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
