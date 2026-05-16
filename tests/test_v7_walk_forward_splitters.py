"""Tests for the V7 walk-forward splitters module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.training.splitters import (
    WalkForwardFold,
    WalkForwardSplitConfig,
    iter_folds_with_data,
    split_walk_forward,
)


def _toy_frame(num_days: int = 400, num_symbols: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=num_days, freq="B")
    rows = []
    for symbol in [f"S{i}" for i in range(num_symbols)]:
        for date in dates:
            rows.append({"symbol": symbol, "trade_date": date, "feature": 0.0})
    return pd.DataFrame(rows)


def test_expanding_splitter_yields_disjoint_train_valid_windows():
    frame = _toy_frame()
    folds = split_walk_forward(
        frame,
        config=WalkForwardSplitConfig(
            n_splits=3,
            valid_size_days=20,
            min_train_days=120,
            embargo_days=5,
            mode="expanding",
        ),
    )
    assert len(folds) == 3
    seen_valid_starts = []
    for fold in folds:
        train_end = fold.train_dates[1]
        valid_start = fold.valid_dates[0]
        assert train_end < valid_start, "train must end before validation in expanding mode"
        seen_valid_starts.append(valid_start)
    # Validation windows roll forward.
    assert seen_valid_starts == sorted(seen_valid_starts)


def test_rolling_splitter_limits_train_window():
    frame = _toy_frame()
    folds = split_walk_forward(
        frame,
        config=WalkForwardSplitConfig(
            n_splits=3,
            valid_size_days=20,
            min_train_days=120,
            rolling_train_days=60,
            embargo_days=5,
            mode="rolling",
        ),
    )
    assert folds
    for fold in folds:
        span_days = (fold.train_dates[1] - fold.train_dates[0]).days
        # Span is in calendar days; with 60 business days the calendar span is roughly <120.
        assert span_days <= 200


def test_purged_splitter_inserts_purge_gap():
    frame = _toy_frame()
    folds = split_walk_forward(
        frame,
        config=WalkForwardSplitConfig(
            n_splits=2,
            valid_size_days=20,
            min_train_days=120,
            embargo_days=5,
            purge_days=10,
            mode="purged",
        ),
    )
    assert folds
    for fold in folds:
        gap = (fold.valid_dates[0] - fold.train_dates[1]).days
        assert gap >= 1, "purged mode must leave a strictly positive gap"


def test_chronological_split_returns_single_fold():
    frame = _toy_frame()
    folds = split_walk_forward(
        frame,
        config=WalkForwardSplitConfig(
            valid_size_days=40,
            min_train_days=200,
            embargo_days=5,
            mode="chronological",
        ),
    )
    assert len(folds) == 1
    fold = folds[0]
    assert fold.train_dates[1] < fold.valid_dates[0]


def test_iter_folds_with_data_returns_disjoint_dataframes():
    frame = _toy_frame()
    fold_data = list(
        iter_folds_with_data(
            frame,
            WalkForwardSplitConfig(
                n_splits=2,
                valid_size_days=20,
                min_train_days=120,
                embargo_days=5,
                mode="expanding",
            ),
        )
    )
    assert fold_data
    for _, train_frame, valid_frame in fold_data:
        train_dates = set(train_frame["trade_date"].unique())
        valid_dates = set(valid_frame["trade_date"].unique())
        assert not (train_dates & valid_dates), "train and validation date sets must be disjoint"


def test_empty_or_short_frame_yields_no_folds():
    assert split_walk_forward(pd.DataFrame()) == []
    short = _toy_frame(num_days=50)
    folds = split_walk_forward(
        short,
        config=WalkForwardSplitConfig(n_splits=3, valid_size_days=30, min_train_days=120),
    )
    assert folds == []
