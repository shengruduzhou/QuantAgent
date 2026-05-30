"""GA weight optimiser tests (spec section 6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.optimization.ga_weight_optimizer import (
    GAConfig,
    WalkForwardConfig,
    optimize_factor_weights_ga,
    purged_walk_forward_splits,
    save_optimisation_artifacts,
)


# ---------------------------------------------------------------------------
# Purged walk-forward splits
# ---------------------------------------------------------------------------

def test_purged_walk_forward_returns_non_overlapping_folds():
    dates = pd.bdate_range("2024-01-01", periods=200).tolist()
    cfg = WalkForwardConfig(n_folds=4, embargo_days=5, min_train_days=60, min_test_days=20)
    splits = purged_walk_forward_splits(dates, cfg)
    assert len(splits) >= 2
    # each train_end must precede test_start by at least embargo days
    for train_start, train_end, test_start, test_end in splits:
        gap = (test_start - train_end).days
        assert gap >= cfg.embargo_days
        assert test_start <= test_end


def test_walk_forward_returns_empty_when_too_few_dates():
    short = pd.bdate_range("2024-01-01", periods=10).tolist()
    cfg = WalkForwardConfig(min_train_days=60, min_test_days=20)
    assert purged_walk_forward_splits(short, cfg) == []


# ---------------------------------------------------------------------------
# Optimiser end-to-end
# ---------------------------------------------------------------------------

def _synth_dataset(n_dates: int = 180, n_symbols: int = 30, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-factor synthetic dataset where factor_a is signal,
    factor_b is noise. The GA should push weight mass onto factor_a.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    syms = [f"60000{i:02d}.SH" for i in range(n_symbols)]
    rows = []
    fwd_rows = []
    for d in dates:
        # daily signal
        a = rng.uniform(-1.0, 1.0, size=n_symbols)
        b = rng.uniform(-1.0, 1.0, size=n_symbols)
        # forward_return correlated with factor_a
        fwd = 0.005 * a + rng.normal(0.0, 0.003, size=n_symbols)
        for sym, ai, bi, fi in zip(syms, a, b, fwd):
            rows.append({"trade_date": d, "symbol": sym, "factor_a": ai, "factor_b": bi})
            fwd_rows.append({"trade_date": d, "symbol": sym, "forward_return": float(fi)})
    return pd.DataFrame(rows), pd.DataFrame(fwd_rows)


def test_ga_assigns_more_weight_to_signal_factor():
    factor_panel, forward_returns = _synth_dataset()
    ga = GAConfig(population_size=12, generations=6, top_k=5, random_seed=2026)
    wf = WalkForwardConfig(n_folds=2, embargo_days=2, min_train_days=60, min_test_days=20)
    result = optimize_factor_weights_ga(
        factor_panel=factor_panel,
        forward_returns=forward_returns,
        factor_names=["factor_a", "factor_b"],
        ga_config=ga, wf_config=wf,
    )
    assert result.best_weights["factor_a"] >= result.best_weights["factor_b"]


def test_ga_produces_fold_results_with_proper_keys():
    factor_panel, forward_returns = _synth_dataset()
    ga = GAConfig(population_size=8, generations=4, top_k=5, random_seed=7)
    wf = WalkForwardConfig(n_folds=2, embargo_days=2, min_train_days=60, min_test_days=20)
    result = optimize_factor_weights_ga(
        factor_panel=factor_panel,
        forward_returns=forward_returns,
        factor_names=["factor_a", "factor_b"],
        ga_config=ga, wf_config=wf,
    )
    assert len(result.fold_results) >= 1
    fold = result.fold_results[0]
    assert set(fold.best_weights) == {"factor_a", "factor_b"}
    assert np.isfinite(fold.best_loss)
    assert "transaction_cost" in fold.components  # Stage-5 term reaches the output


def test_ga_factor_weights_sum_to_one():
    factor_panel, forward_returns = _synth_dataset()
    ga = GAConfig(population_size=8, generations=3, top_k=5, random_seed=11)
    wf = WalkForwardConfig(n_folds=2, embargo_days=2, min_train_days=60, min_test_days=20)
    result = optimize_factor_weights_ga(
        factor_panel=factor_panel,
        forward_returns=forward_returns,
        factor_names=["factor_a", "factor_b"],
        ga_config=ga, wf_config=wf,
    )
    total = sum(result.best_weights.values())
    assert abs(total - 1.0) < 1e-6


def test_ga_raises_on_missing_factor_columns():
    panel, fwd = _synth_dataset()
    with pytest.raises(ValueError, match="missing columns"):
        optimize_factor_weights_ga(
            factor_panel=panel, forward_returns=fwd,
            factor_names=["factor_a", "factor_x"],
            ga_config=GAConfig(population_size=4, generations=1),
            wf_config=WalkForwardConfig(n_folds=2, embargo_days=2,
                                         min_train_days=60, min_test_days=20),
        )


def test_save_artifacts_writes_three_files(tmp_path):
    factor_panel, forward_returns = _synth_dataset()
    ga = GAConfig(population_size=6, generations=2, top_k=5, random_seed=1)
    wf = WalkForwardConfig(n_folds=2, embargo_days=2, min_train_days=60, min_test_days=20)
    result = optimize_factor_weights_ga(
        factor_panel=factor_panel,
        forward_returns=forward_returns,
        factor_names=["factor_a", "factor_b"],
        ga_config=ga, wf_config=wf,
    )
    out = save_optimisation_artifacts(result, output_dir=tmp_path)
    assert out["factor_weights"].exists()
    assert out["walk_forward_backtest"].exists()
    assert out["metrics"].exists()
    payload = json.loads(out["walk_forward_backtest"].read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert all("fold_index" in row for row in payload)


def test_ga_purged_split_no_overlap_with_embargo():
    """Stronger version of the split assertion — explicit gap inspection."""
    dates = pd.bdate_range("2024-01-01", periods=500).tolist()
    cfg = WalkForwardConfig(n_folds=3, embargo_days=10, min_train_days=80, min_test_days=30)
    splits = purged_walk_forward_splits(dates, cfg)
    assert len(splits) >= 2
    for prev, nxt in zip(splits, splits[1:]):
        # next fold's test must start later than previous fold's test
        assert nxt[3] > prev[3]
        # embargo enforced for each fold individually
        assert (nxt[2] - nxt[1]).days >= cfg.embargo_days
