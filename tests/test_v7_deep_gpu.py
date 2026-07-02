"""Stage 5 item 3: GPU backend verification for the deep alpha trainer.

Fail-loud tests (require_gpu must not silently fall back to CPU) run anywhere.
The real-CUDA tests are skipped automatically when no GPU is visible, so the
suite still passes on CPU-only boxes.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantagent.training.v7_deep_trainer import (
    V7DeepAlphaTrainer,
    V7DeepAlphaTrainerConfig,
    run_walk_forward_deep_training,
)


def _dataset(rows: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-02", periods=rows // 6, freq="B")
    out: list[dict] = []
    for date in dates:
        for sidx in range(6):
            feats = rng.standard_normal(8)
            out.append(
                {
                    "trade_date": date, "symbol": f"S{sidx:03d}",
                    **{f"feat_{i}": float(feats[i]) for i in range(8)},
                    "forward_return_1d": float(0.1 * feats[0] + 0.02 * rng.standard_normal()),
                    "forward_return_5d": float(0.2 * feats[0] + 0.05 * rng.standard_normal()),
                }
            )
    return pd.DataFrame(out)


# --------------------------------------------------------- fail-loud (no GPU needed)

def test_require_gpu_fails_loud_when_cpu_requested():
    trainer = V7DeepAlphaTrainer(
        V7DeepAlphaTrainerConfig(device="cpu", require_gpu=True, use_torch=True)
    )
    with pytest.raises(RuntimeError, match="GPU training was required"):
        trainer._resolve_device()


def test_require_gpu_fails_loud_when_torch_disabled():
    trainer = V7DeepAlphaTrainer(
        V7DeepAlphaTrainerConfig(require_gpu=True, use_torch=False)
    )
    with pytest.raises(RuntimeError, match="require_gpu=True"):
        trainer._select_backend()


def test_no_require_gpu_falls_back_to_numpy_on_cpu():
    # Default discipline: without require_gpu the CPU path is allowed.
    trainer = V7DeepAlphaTrainer(V7DeepAlphaTrainerConfig(use_torch=False))
    assert trainer._select_backend() == "numpy"


# --------------------------------------------------------- real CUDA (skipped on CPU box)

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable; GPU verification tests skipped.", allow_module_level=True)


def test_deep_trainer_trains_on_gpu_and_logs_memory(tmp_path):
    dataset = _dataset()
    config = V7DeepAlphaTrainerConfig(
        horizons=(1, 5), hidden_sizes=(16, 8), max_epochs=3, batch_size=128,
        feature_columns=tuple(f"feat_{i}" for i in range(8)),
        device="cuda", require_gpu=True, use_torch=True,
        output_dir=str(tmp_path / "deep"), seed=11,
    )
    trainer = V7DeepAlphaTrainer(config)
    state = trainer.fit(dataset)
    assert state.backend == "torch"          # really used torch, not numpy fallback
    assert state.gpu_peak_mb > 0.0           # CUDA memory was actually allocated + logged
    preds = trainer.predict(dataset)
    assert {"alpha_1d", "alpha_5d"}.issubset(preds.columns)
    # gpu peak is persisted through the checkpoint.
    out = trainer.save(tmp_path / "deep")
    state_json = json.loads((out.parent / "deep_alpha_state.json").read_text())
    assert state_json["gpu_peak_mb"] > 0.0


def test_walk_forward_gpu_logs_peak_memory_per_fold(tmp_path):
    from quantagent.data.dataset_builder import V7TrainingDatasetConfig, build_v7_training_dataset_artifact
    from quantagent.data.v7_label_builder import build_forward_return_labels
    from quantagent.training.splitters import WalkForwardSplitConfig

    # Build a real schema-bearing dataset to pin the walk-forward run.
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-06-03", periods=120)
    rows = []
    for sidx in range(6):
        close = 10.0 + sidx
        for date in dates:
            close = max(1.0, close * (1.0 + rng.normal(0.0, 0.02)))
            rows.append({"trade_date": date, "symbol": f"60{sidx:04d}.SH",
                         "open": close * 0.99, "high": close * 1.02, "low": close * 0.98, "close": close,
                         "volume": 1_000_000 + sidx * 50_000, "amount": close * 1_000_000, "available_at": date})
    market = pd.DataFrame(rows)
    labels = build_forward_return_labels(market, horizons=(1, 5)).frame
    market.to_parquet(tmp_path / "m.parquet", index=False)
    labels.to_parquet(tmp_path / "l.parquet", index=False)
    built = build_v7_training_dataset_artifact(
        V7TrainingDatasetConfig(
            market_panel_path=str(tmp_path / "m.parquet"), labels_path=str(tmp_path / "l.parquet"),
            output_path=str(tmp_path / "ds.parquet"), horizons=(1, 5),
            min_rows=50, min_symbols=2, min_dates=10, feature_version="v-gpu",
        )
    )
    wf = run_walk_forward_deep_training(
        built.dataset, feature_schema_path=str(built.feature_schema_path),
        base_config=V7DeepAlphaTrainerConfig(
            horizons=(1, 5), hidden_sizes=(16,), max_epochs=2,
            device="cuda", require_gpu=True, use_torch=True, seed=3,
        ),
        split_config=WalkForwardSplitConfig(
            mode="purged", n_splits=2, min_train_days=40, valid_size_days=15, embargo_days=2, purge_days=5,
        ),
        output_dir=str(tmp_path / "wf"),
    )
    assert set(wf.fold_metadata["backend"]) == {"torch"}
    assert "gpu_peak_mb" in wf.fold_metadata.columns
    assert (wf.fold_metadata["gpu_peak_mb"] > 0.0).all()
    man = json.loads((tmp_path / "wf" / "run_manifest.json").read_text())
    assert man["require_gpu"] is True
    assert man["gpu_peak_mb_max"] > 0.0
