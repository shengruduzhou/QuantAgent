"""The fold budget a walk-forward request promises must be the one it delivers.

These lock in the two arithmetic defects that made `run-full-real-training-v7`
unrunnable on its own factory defaults:

1. ``min_train_days`` was consumed by the embargo/purge gap, so the earliest
   folds were handed a training window of a couple of days, failed the
   trainer's row check, and vanished. A caller asking for 5 folds silently got
   4, and the nested-selection protocol then rejected the finished run for
   having 80 OOS days instead of 100.
2. Folds were anchored at the *start* of the sample, so five folds on a decade
   of history validated 2017 and never looked at the following nine years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import replace

from quantagent.training.splitters import (
    WalkForwardSplitConfig,
    plan_walk_forward,
    split_walk_forward,
)


def _frame(num_days: int, num_symbols: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2016-01-04", periods=num_days, freq="B")
    return pd.DataFrame(
        {
            "trade_date": np.repeat(dates.to_numpy(), num_symbols),
            "symbol": np.tile([f"S{i:03d}" for i in range(num_symbols)], num_days),
            "feature": 0.0,
        }
    )


def test_every_fold_gets_the_full_training_window_it_was_promised():
    """The gap is reserved on top of min_train_days, never carved out of it."""
    frame = _frame(2562)
    cfg = WalkForwardSplitConfig(
        n_splits=5,
        valid_size_days=20,
        min_train_days=120,
        embargo_days=5,
        purge_days=120,  # the 120d label horizon in the shipped default
        mode="rolling",
        rolling_train_days=756,
    )
    folds = split_walk_forward(frame, config=cfg)

    assert len(folds) == 5
    trading_days = pd.Index(sorted(frame["trade_date"].unique()))
    for fold in folds:
        train_days = trading_days.slice_indexer(fold.train_dates[0], fold.train_dates[1])
        span = train_days.stop - train_days.start
        assert span >= 120, f"fold {fold.fold_id} trained on {span} days, expected >= 120"
        # The purge gap still has to be there — this is a leakage boundary.
        gap = trading_days.get_loc(fold.valid_dates[0]) - trading_days.get_loc(fold.train_dates[1])
        assert gap >= 125, f"fold {fold.fold_id} left only {gap} days of embargo+purge"


def test_requested_fold_count_survives_to_the_oos_span():
    """5 folds x 20 days must yield 100 distinct OOS trading days, not 80."""
    frame = _frame(2562)
    folds = split_walk_forward(
        frame,
        config=WalkForwardSplitConfig(
            n_splits=5, valid_size_days=20, min_train_days=120,
            embargo_days=5, purge_days=120, mode="rolling",
        ),
    )
    oos_days = set()
    for fold in folds:
        oos_days.update(frame["trade_date"].iloc[fold.valid_idx].unique())
    assert len(oos_days) == 100


def test_folds_are_anchored_to_the_most_recent_data():
    """A 5-fold request on 10 years must validate the last 100 days, not the first."""
    frame = _frame(2562)
    folds = split_walk_forward(
        frame,
        config=WalkForwardSplitConfig(
            n_splits=5, valid_size_days=20, min_train_days=120,
            embargo_days=5, purge_days=120, mode="rolling",
        ),
    )
    last_date = pd.Timestamp(frame["trade_date"].max())
    assert folds[-1].valid_dates[1] == last_date
    # ...and the windows still march forward without overlapping.
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.valid_dates[1] < later.valid_dates[0]


def test_start_anchor_remains_available_for_learning_curve_studies():
    frame = _frame(2562)
    folds = split_walk_forward(
        frame,
        config=WalkForwardSplitConfig(
            n_splits=5, valid_size_days=20, min_train_days=120,
            embargo_days=5, purge_days=120, mode="rolling", anchor="start",
        ),
    )
    assert len(folds) == 5
    assert folds[0].valid_dates[0] < pd.Timestamp("2017-06-01")


def test_plan_reports_the_shortfall_before_any_model_is_fit():
    cfg = WalkForwardSplitConfig(
        n_splits=5, valid_size_days=20, min_train_days=120,
        embargo_days=5, purge_days=120, mode="rolling",
    )
    # 245 lead days + 100 OOS days = 345 needed.
    assert plan_walk_forward(2562, cfg).is_satisfiable
    assert plan_walk_forward(2562, cfg).achievable_splits == 5

    short = plan_walk_forward(300, cfg)
    assert not short.is_satisfiable
    assert short.achievable_splits == 2  # (300 - 245) // 20
    assert short.oos_days == 40
    assert short.days_needed_for(5) == 345

    # A plan that cannot seat even one fold produces no folds at all.
    assert plan_walk_forward(200, cfg).achievable_splits == 0
    assert split_walk_forward(_frame(200), config=cfg) == []


def test_a_shared_anchor_keeps_every_horizon_on_the_same_oos_window():
    """Without it, a 1d and a 120d horizon validate disjoint windows.

    Each horizon's labels run out at a different date. Anchored individually,
    the end-anchored folds land in different places and a cross-horizon blend
    intersects to nothing — the OOS panel goes from dense to a handful of names
    per date.
    """
    frame = _frame(2562, num_symbols=10)
    cfg = WalkForwardSplitConfig(
        n_splits=5, valid_size_days=20, min_train_days=120,
        embargo_days=5, purge_days=120, mode="rolling",
    )
    trading_days = pd.Index(sorted(frame["trade_date"].unique()))
    # 1d labels survive to the last date; 120d labels stop 120 days earlier.
    per_horizon_last = {1: trading_days[-1], 120: trading_days[-121]}
    shared_anchor = min(per_horizon_last.values())

    unanchored = {
        horizon: split_walk_forward(
            frame[frame["trade_date"] <= last], config=cfg
        )
        for horizon, last in per_horizon_last.items()
    }
    windows = [
        {d for fold in folds for d in frame["trade_date"].iloc[fold.valid_idx].unique()}
        for folds in unanchored.values()
    ]
    assert not (windows[0] & windows[1]), "precondition: unanchored horizons diverge"

    anchored = {
        horizon: split_walk_forward(
            frame[frame["trade_date"] <= last],
            config=replace(cfg, anchor_end=shared_anchor),
        )
        for horizon, last in per_horizon_last.items()
    }
    shared_windows = [
        [(f.valid_dates[0], f.valid_dates[1]) for f in folds]
        for folds in anchored.values()
    ]
    assert shared_windows[0] == shared_windows[1]
    assert anchored[1][-1].valid_dates[1] == shared_anchor


def test_chronological_holdout_purges_the_label_horizon_too():
    """An embargo-only gap lets training labels resolve inside the holdout."""
    frame = _frame(600, num_symbols=5)
    folds = split_walk_forward(
        frame,
        config=WalkForwardSplitConfig(
            mode="chronological", valid_size_days=60,
            min_train_days=120, embargo_days=5, purge_days=120,
        ),
    )
    assert len(folds) == 1
    trading_days = pd.Index(sorted(frame["trade_date"].unique()))
    fold = folds[0]
    gap = trading_days.get_loc(fold.valid_dates[0]) - trading_days.get_loc(fold.train_dates[1])
    # A 120-day label started on the last training day must land before the
    # holdout opens, so the gap has to cover purge + embargo.
    assert gap >= 125, f"only {gap} trading days between train end and holdout start"
    assert fold.embargo_days == 125


def test_plan_matches_what_the_splitter_actually_returns():
    """The pre-flight estimate and the real run must never disagree."""
    cfg = WalkForwardSplitConfig(
        n_splits=8, valid_size_days=20, min_train_days=120,
        embargo_days=5, purge_days=120, mode="rolling",
    )
    for num_days in (260, 400, 700, 1200, 2562):
        plan = plan_walk_forward(num_days, cfg)
        folds = split_walk_forward(_frame(num_days, num_symbols=5), config=cfg)
        assert len(folds) == plan.achievable_splits, f"mismatch at {num_days} days"
