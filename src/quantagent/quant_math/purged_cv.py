from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedKFoldConfig:
    n_splits: int = 5
    embargo_pct: float = 0.01


def _embargo_size(n: int, embargo_pct: float) -> int:
    return int(np.ceil(n * embargo_pct))


def purged_kfold_split(
    times: pd.Series,
    label_end_times: pd.Series,
    config: PurgedKFoldConfig | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Lopez de Prado AFML section 7.4 purged K-fold with embargo.

    times: index timestamps (sample t0).
    label_end_times: t1 for each sample.
    """
    config = config or PurgedKFoldConfig()
    if not times.index.equals(label_end_times.index):
        raise ValueError("times and label_end_times must share the same index")
    indices = np.arange(len(times))
    fold_sizes = np.full(config.n_splits, len(indices) // config.n_splits, dtype=int)
    fold_sizes[: len(indices) % config.n_splits] += 1
    boundaries = np.cumsum(np.concatenate([[0], fold_sizes]))
    embargo = _embargo_size(len(indices), config.embargo_pct)
    t0 = times.values
    t1 = label_end_times.values
    for k in range(config.n_splits):
        test_start, test_end = boundaries[k], boundaries[k + 1]
        test_idx = indices[test_start:test_end]
        test_t0_min = t0[test_idx].min()
        test_t1_max = t1[test_idx].max()
        train_mask = np.ones(len(indices), dtype=bool)
        train_mask[test_idx] = False
        purge_mask = (t1 >= test_t0_min) & (t0 <= test_t1_max)
        train_mask &= ~purge_mask
        embargo_end = min(test_end + embargo, len(indices))
        train_mask[test_end:embargo_end] = False
        train_idx = indices[train_mask]
        yield train_idx, test_idx


def probability_of_backtest_overfitting(
    in_sample_sharpes: np.ndarray,
    out_sample_sharpes: np.ndarray,
) -> float:
    """Bailey & Lopez de Prado PBO: Pr[OOS rank of best IS strategy < median]."""
    if in_sample_sharpes.shape != out_sample_sharpes.shape:
        raise ValueError("IS and OOS arrays must share shape")
    in_ranks = np.argsort(np.argsort(in_sample_sharpes, axis=0), axis=0)
    best = in_ranks.argmax(axis=0)
    chosen_oos = out_sample_sharpes[best, np.arange(out_sample_sharpes.shape[1])]
    oos_ranks = np.argsort(np.argsort(out_sample_sharpes, axis=0), axis=0)
    chosen_rank = oos_ranks[best, np.arange(out_sample_sharpes.shape[1])]
    n_strategies = out_sample_sharpes.shape[0]
    return float((chosen_rank < n_strategies / 2).mean())
