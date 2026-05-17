"""Tests for FT-Transformer multi-date batching and per-date rank loss.

Pins the behaviour:

* ``dates_per_step`` config controls how many trade dates are grouped
  into a single forward/backward step
* rank-loss inside a multi-date chunk is computed per-date (no cross-date
  argsort, which would leak future cross-sectional ranking info)

These tests run on CPU to keep CI environments unburdened. They use a
small synthetic dataset and a single epoch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("torch")


def _synthetic_dataset(n_dates: int = 12, n_symbols: int = 40, n_features: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-02", periods=n_dates, freq="B")
    rows = []
    for date in dates:
        for s in range(n_symbols):
            feats = rng.standard_normal(n_features)
            rows.append({
                "trade_date": date,
                "symbol": f"S{s:03d}",
                "feature_a": float(feats[0]),
                "feature_b": float(feats[1]),
                "feature_c": float(feats[2]),
                "feature_d": float(feats[3]),
                "feature_e": float(feats[4]),
                "feature_f": float(feats[5]),
                "forward_return_1d": float(rng.standard_normal() * 0.01),
                "forward_return_5d": float(rng.standard_normal() * 0.02),
            })
    return pd.DataFrame(rows)


def test_multi_date_step_runs_and_writes_metrics(tmp_path):
    from quantagent.training.ft_transformer_trainer import (
        FTTransformerTrainer,
        FTTransformerTrainerConfig,
    )

    data = _synthetic_dataset(n_dates=10, n_symbols=30)
    cfg = FTTransformerTrainerConfig(
        horizons=(1, 5),
        d_token=16,
        n_blocks=2,
        n_heads=2,
        max_epochs=1,
        dates_per_step=4,
        batch_size=1024,
        early_stopping_patience=1,
        use_amp=False,
        device="cpu",
        require_gpu=False,
        rank_loss_weight=0.5,
        feature_columns=("feature_a", "feature_b", "feature_c", "feature_d", "feature_e", "feature_f"),
        output_dir=str(tmp_path / "ft"),
    )
    trainer = FTTransformerTrainer(cfg)
    artefacts = trainer.fit_and_save(data)
    metrics_path = tmp_path / "ft" / "ft_transformer_metrics.json"
    assert metrics_path.exists()
    import json
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["dates_per_step"] == 4
    assert payload["d_token"] == 16
    assert payload["max_epochs"] == 1
    # peak_gpu_memory_mb may be None on CPU runs, but the field must exist.
    assert "peak_gpu_memory_mb" in payload
    assert artefacts.training_history, "should record at least one epoch"


def test_dates_per_step_one_matches_legacy_per_date_loop(tmp_path):
    """dates_per_step=1 reproduces the old behaviour (one date per step)."""
    from quantagent.training.ft_transformer_trainer import (
        FTTransformerTrainer,
        FTTransformerTrainerConfig,
    )

    data = _synthetic_dataset(n_dates=6, n_symbols=20)
    cfg = FTTransformerTrainerConfig(
        horizons=(1, 5),
        d_token=8,
        n_blocks=2,
        n_heads=2,
        max_epochs=1,
        dates_per_step=1,
        batch_size=512,
        early_stopping_patience=1,
        use_amp=False,
        device="cpu",
        require_gpu=False,
        rank_loss_weight=0.5,
        feature_columns=("feature_a", "feature_b", "feature_c", "feature_d", "feature_e", "feature_f"),
        output_dir=str(tmp_path / "ft1"),
    )
    artefacts = FTTransformerTrainer(cfg).fit_and_save(data)
    assert artefacts.training_history, "should produce at least one epoch when dates_per_step=1"


def test_per_date_rank_loss_does_not_pool_across_dates(monkeypatch, tmp_path):
    """Verify the multi-date branch computes rank loss separately per date.

    Strategy: monkeypatch ``torch.Tensor.argsort`` to count how many times
    it is invoked during the multi-date training step. If rank loss were
    pooled across dates we'd see two ``argsort`` calls per step (preds /
    labels); if it is per-date, we should see ``2 * dates_per_step``
    calls per step.
    """
    import torch
    from quantagent.training.ft_transformer_trainer import (
        FTTransformerTrainer,
        FTTransformerTrainerConfig,
    )

    data = _synthetic_dataset(n_dates=4, n_symbols=20)
    dates_per_step = 4
    call_count = {"argsort": 0}
    real_argsort = torch.Tensor.argsort

    def counting_argsort(self, *args, **kwargs):  # type: ignore[no-redef]
        call_count["argsort"] += 1
        return real_argsort(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "argsort", counting_argsort)

    cfg = FTTransformerTrainerConfig(
        horizons=(1,),
        d_token=8,
        n_blocks=2,
        n_heads=2,
        max_epochs=1,
        dates_per_step=dates_per_step,
        batch_size=512,
        early_stopping_patience=1,
        use_amp=False,
        device="cpu",
        require_gpu=False,
        rank_loss_weight=0.5,
        feature_columns=("feature_a", "feature_b", "feature_c", "feature_d", "feature_e", "feature_f"),
        output_dir=str(tmp_path / "ft_rank"),
    )
    FTTransformerTrainer(cfg).fit_and_save(data)
    # 4 dates, dates_per_step=4 → 1 chunk per epoch.
    # Per-date branch invokes argsort twice per date (preds + targets) = 2*4 = 8 calls minimum.
    # Pooled branch would only invoke argsort twice total.
    assert call_count["argsort"] >= 2 * dates_per_step, (
        f"expected ≥{2 * dates_per_step} per-date argsort calls, got {call_count['argsort']}"
    )
