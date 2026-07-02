"""Stage 5 item 1: walk-forward deep trainer is schema-locked across folds.

Every fold trains on the SAME pinned feature schema, so all folds share one
``schema_hash`` and feature column set — the precondition for comparable folds
and a stable walk-forward OOS feature space.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.data.dataset_builder import V7TrainingDatasetConfig, build_v7_training_dataset_artifact
from quantagent.data.v7_label_builder import build_forward_return_labels
from quantagent.training.splitters import WalkForwardSplitConfig
from quantagent.training.v7_deep_trainer import (
    V7DeepAlphaTrainerConfig,
    run_walk_forward_deep_training,
)


def _market_panel(days: int = 120, n_symbols: int = 6, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-06-03", periods=days)
    rows: list[dict] = []
    for sidx in range(n_symbols):
        symbol = f"60{sidx:04d}.SH"
        close = 10.0 + sidx
        for date in dates:
            close = max(1.0, close * (1.0 + rng.normal(0.0, 0.02)))
            rows.append(
                {
                    "trade_date": date, "symbol": symbol,
                    "open": close * 0.99, "high": close * 1.02, "low": close * 0.98, "close": close,
                    "volume": 1_000_000 + sidx * 50_000, "amount": close * (1_000_000 + sidx * 50_000),
                    "available_at": date,
                }
            )
    return pd.DataFrame(rows)


def _build_dataset(tmp_path: Path):
    market = _market_panel()
    labels = build_forward_return_labels(market, horizons=(1, 5)).frame
    market.to_parquet(tmp_path / "m.parquet", index=False)
    labels.to_parquet(tmp_path / "l.parquet", index=False)
    return build_v7_training_dataset_artifact(
        V7TrainingDatasetConfig(
            market_panel_path=str(tmp_path / "m.parquet"),
            labels_path=str(tmp_path / "l.parquet"),
            output_path=str(tmp_path / "ds.parquet"),
            horizons=(1, 5), min_rows=50, min_symbols=2, min_dates=10,
            feature_version="v-wf",
        )
    )


def test_walk_forward_deep_is_schema_locked_across_folds(tmp_path):
    built = _build_dataset(tmp_path)
    schema_path = built.feature_schema_path
    contract = json.loads(Path(schema_path).read_text(encoding="utf-8"))

    wf = run_walk_forward_deep_training(
        built.dataset,
        feature_schema_path=str(schema_path),
        base_config=V7DeepAlphaTrainerConfig(
            horizons=(1, 5), hidden_sizes=(8,), max_epochs=2, use_torch=False, seed=3,
        ),
        split_config=WalkForwardSplitConfig(
            mode="purged", n_splits=2, min_train_days=40, valid_size_days=15,
            embargo_days=2, purge_days=5,
        ),
        output_dir=str(tmp_path / "wf_models"),
    )

    # At least 2 folds, all locked to the dataset's contract hash.
    md = wf.fold_metadata.sort_values("fold_id").reset_index(drop=True)
    assert len(md) >= 2
    assert wf.schema_hash == contract["schema_hash"]
    assert set(md["schema_hash"]) == {contract["schema_hash"]}      # one hash for all folds
    assert (md["feature_count"] == len(contract["feature_columns"])).all()
    assert wf.feature_columns == contract["feature_columns"]
    assert set(md["feature_version"]) == {"v-wf"}

    # Walk-forward shape: chronological, non-overlapping validation windows,
    # and each fold's train window ends before its validation window (embargo).
    for i in range(len(md) - 1):
        assert md.loc[i, "valid_end"] < md.loc[i + 1, "valid_start"]
    assert (md["train_end"] < md["valid_start"]).all()

    # OOS predictions carry fold_id + schema provenance and span multiple folds.
    oos = wf.oos_predictions
    assert not oos.empty
    assert {"symbol", "trade_date", "fold_id", "alpha_1d", "alpha_5d", "schema_hash"}.issubset(oos.columns)
    assert oos["fold_id"].nunique() >= 2
    assert set(oos["schema_hash"]) == {contract["schema_hash"]}

    # Each fold's saved model records the same locked hash.
    for fid in md["fold_id"]:
        model_schema = json.loads(
            (tmp_path / "wf_models" / f"fold_{fid}" / "deep_alpha_feature_schema.json").read_text(encoding="utf-8")
        )
        assert model_schema["schema_hash"] == contract["schema_hash"]


def _wf_kwargs(schema_path: str, seed: int = 3):
    return dict(
        feature_schema_path=schema_path,
        base_config=V7DeepAlphaTrainerConfig(
            horizons=(1, 5), hidden_sizes=(8,), max_epochs=2, use_torch=False, seed=seed,
        ),
        split_config=WalkForwardSplitConfig(
            mode="purged", n_splits=2, min_train_days=40, valid_size_days=15,
            embargo_days=2, purge_days=5,
        ),
    )


def test_walk_forward_persists_reproducibility_manifest(tmp_path):
    built = _build_dataset(tmp_path)
    out = tmp_path / "wf"
    wf = run_walk_forward_deep_training(
        built.dataset, output_dir=str(out), **_wf_kwargs(str(built.feature_schema_path), seed=7)
    )

    # Artifacts written.
    assert (out / "run_manifest.json").exists()
    assert (out / "fold_plan.csv").exists()
    assert (out / "walkforward_predictions.parquet").exists() or (out / "walkforward_predictions.csv").exists()

    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    for key in ("created_at", "model_version", "schema_hash", "feature_version", "feature_columns",
                "seed", "horizons", "split_config", "n_folds", "fold_plan", "fold_checkpoints"):
        assert key in man, key
    assert man["seed"] == 7
    assert man["schema_hash"] == wf.schema_hash
    assert man["n_folds"] == len(wf.fold_metadata)
    assert len(man["fold_checkpoints"]) == man["n_folds"]
    # Every recorded checkpoint actually exists on disk.
    for ckpt in man["fold_checkpoints"].values():
        assert Path(ckpt).exists()

    # Self-describing OOS prediction rows.
    cols = set(wf.oos_predictions.columns)
    for c in ("fold_id", "train_start", "train_end", "valid_start", "valid_end",
              "model_version", "schema_hash", "feature_version"):
        assert c in cols, c
    assert wf.manifest_path == str(out / "run_manifest.json")


def test_walk_forward_deep_is_reproducible(tmp_path):
    built = _build_dataset(tmp_path)
    schema_path = str(built.feature_schema_path)
    a = run_walk_forward_deep_training(built.dataset, **_wf_kwargs(schema_path, seed=5))
    b = run_walk_forward_deep_training(built.dataset, **_wf_kwargs(schema_path, seed=5))

    assert a.schema_hash == b.schema_hash
    assert a.run_manifest["model_version"] == b.run_manifest["model_version"]
    pd.testing.assert_frame_equal(a.fold_metadata, b.fold_metadata)
    pa = a.oos_predictions.sort_values(["fold_id", "symbol", "trade_date"]).reset_index(drop=True)
    pb = b.oos_predictions.sort_values(["fold_id", "symbol", "trade_date"]).reset_index(drop=True)
    assert pa[["symbol", "trade_date", "fold_id"]].equals(pb[["symbol", "trade_date", "fold_id"]])
    np.testing.assert_allclose(
        pa[["alpha_1d", "alpha_5d"]].to_numpy(),
        pb[["alpha_1d", "alpha_5d"]].to_numpy(),
        atol=1e-12,
    )


def test_walk_forward_deep_requires_loadable_schema(tmp_path):
    built = _build_dataset(tmp_path)
    import pytest

    with pytest.raises((FileNotFoundError, ValueError)):
        run_walk_forward_deep_training(
            built.dataset,
            feature_schema_path=str(tmp_path / "does_not_exist.json"),
            base_config=V7DeepAlphaTrainerConfig(horizons=(1, 5), use_torch=False),
        )
