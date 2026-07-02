"""Cross-sectional-feature classical walk-forward trainer: schema-lock + signal recovery."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quantagent.training.splitters import WalkForwardSplitConfig
from quantagent.training.walk_forward_classical import ClassicalWFConfig, run_walk_forward_classical
from quantagent.training.walk_forward_eval import evaluate_walk_forward_oos


def _dataset(n_days: int = 120, n_sym: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rows = []
    for d in dates:
        f0 = rng.normal(0, 1, n_sym)
        f1 = rng.normal(0, 1, n_sym)
        f2 = rng.normal(0, 1, n_sym)
        # forward return increases with the per-day RANK of f0 (cross-sectional signal)
        r0 = pd.Series(f0).rank(pct=True).to_numpy()
        fwd = 0.03 * (r0 - 0.5) + rng.normal(0, 0.01, n_sym)
        for s in range(n_sym):
            rows.append({"trade_date": d, "symbol": f"S{s:03d}",
                         "f0": f0[s], "f1": f1[s], "f2": f2[s],
                         "forward_return_5d": fwd[s]})
    return pd.DataFrame(rows)


def _schema(tmp_path, feats):
    p = tmp_path / "feature_schema.json"
    p.write_text(json.dumps({"feature_version": "cls-test", "schema_hash": "h" * 64,
                             "feature_columns": list(feats), "label_columns": ["forward_return_5d"],
                             "horizons": [5]}), encoding="utf-8")
    return p


def test_classical_wf_recovers_cross_sectional_signal_and_is_schema_locked(tmp_path):
    ds = _dataset()
    schema = _schema(tmp_path, ["f0", "f1", "f2"])
    res = run_walk_forward_classical(
        ds, feature_schema_path=str(schema),
        config=ClassicalWFConfig(horizons=(5,), model="ridge", cross_sectional="rank", seed=1),
        split_config=WalkForwardSplitConfig(mode="purged", n_splits=2, min_train_days=40,
                                            valid_size_days=20, embargo_days=2, purge_days=5),
        output_dir=str(tmp_path / "wf"),
    )
    # schema-locked across folds
    assert set(res.fold_metadata["schema_hash"]) == {"h" * 64}
    assert res.feature_columns == ["f0", "f1", "f2"]
    assert len(res.fold_metadata) >= 2
    # self-describing OOS rows
    oos = res.oos_predictions
    for c in ("symbol", "trade_date", "fold_id", "alpha_5d", "model_version", "schema_hash"):
        assert c in oos.columns
    # the planted cross-sectional signal is recovered (positive OOS rank-IC)
    ev = evaluate_walk_forward_oos(oos, ds[["symbol", "trade_date", "forward_return_5d"]], horizons=(5,))
    assert ev.overall.iloc[0]["rank_ic"] > 0.15
    # manifest persisted
    assert (tmp_path / "wf" / "run_manifest.json").exists()


def test_classical_wf_lightgbm_backend_runs(tmp_path):
    import pytest
    pytest.importorskip("lightgbm")
    ds = _dataset(n_days=90, n_sym=25)
    schema = _schema(tmp_path, ["f0", "f1", "f2"])
    res = run_walk_forward_classical(
        ds, feature_schema_path=str(schema),
        config=ClassicalWFConfig(horizons=(5,), model="lightgbm", cross_sectional="rank",
                                 n_estimators=50, num_leaves=15, min_child_samples=20, seed=1),
        split_config=WalkForwardSplitConfig(mode="purged", n_splits=2, min_train_days=30,
                                            valid_size_days=15, embargo_days=2, purge_days=5),
    )
    assert not res.oos_predictions.empty
    ev = evaluate_walk_forward_oos(res.oos_predictions, ds[["symbol", "trade_date", "forward_return_5d"]], horizons=(5,))
    assert ev.overall.iloc[0]["rank_ic"] > 0.10
