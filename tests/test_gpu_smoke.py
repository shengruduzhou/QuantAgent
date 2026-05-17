"""GPU smoke tests — verify FT-Transformer trains on CUDA end-to-end.

These tests run on the lab box and exercise the real torch CUDA path; they
are skipped automatically when no CUDA device is visible so the suite still
passes on CPU-only machines.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable; GPU smoke tests skipped.", allow_module_level=True)

from quantagent.training.ft_transformer_trainer import (
    FTTransformerTrainer,
    FTTransformerTrainerConfig,
    _resolve_device,
)


def _build_synthetic_dataset(n_dates: int = 30, n_symbols: int = 40, n_features: int = 16) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-02", periods=n_dates).strftime("%Y-%m-%d")
    rows = []
    for d in dates:
        for s in range(n_symbols):
            features = rng.standard_normal(n_features)
            row = {f"f_{i:02d}": float(features[i]) for i in range(n_features)}
            row["trade_date"] = d
            row["symbol"] = f"SH{600000 + s:06d}"
            base = float(features[0] * 0.01 + rng.standard_normal() * 0.005)
            row["forward_return_1d"] = base
            row["forward_return_5d"] = base * 1.2
            row["forward_return_20d"] = base * 1.5
            rows.append(row)
    return pd.DataFrame(rows)


def test_ft_transformer_trains_on_gpu(tmp_path: Path) -> None:
    dataset = _build_synthetic_dataset()
    cfg = FTTransformerTrainerConfig(
        horizons=(1, 5, 20),
        d_token=32,
        n_blocks=2,
        n_heads=4,
        batch_size=128,
        max_epochs=2,
        early_stopping_patience=1,
        device="cuda",
        require_gpu=True,
        use_amp=True,
        output_dir=str(tmp_path),
    )
    trainer = FTTransformerTrainer(cfg)
    artifacts = trainer.fit_and_save(dataset)
    assert artifacts.device == "cuda"
    assert artifacts.cuda_available is True
    assert artifacts.gpu_name is not None and len(artifacts.gpu_name) > 0
    assert artifacts.checkpoint_path.exists()
    assert artifacts.metrics_path.exists()


def test_resolve_device_require_gpu_raises_when_cuda_missing(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        _resolve_device("auto", require_gpu=True)
    with pytest.raises(RuntimeError):
        _resolve_device("cuda", require_gpu=True)
    with pytest.raises(RuntimeError):
        _resolve_device("cpu", require_gpu=True)


def test_resolve_device_cuda_when_available() -> None:
    assert _resolve_device("auto") == "cuda"
    assert _resolve_device("cuda") == "cuda"
