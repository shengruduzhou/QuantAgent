"""V7 deep alpha trainer: fit / predict / save / load round-trip."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.training.v7_deep_trainer import V7DeepAlphaTrainer, V7DeepAlphaTrainerConfig


def _synthetic_dataset(rows: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-02", periods=rows // 5, freq="B")
    rows_out: list[dict] = []
    for date in dates:
        for sidx in range(5):
            features = rng.standard_normal(8)
            label_1d = float(0.1 * features[0] + 0.05 * features[1] + 0.02 * rng.standard_normal())
            label_5d = float(0.2 * features[0] + 0.10 * features[2] + 0.05 * rng.standard_normal())
            rows_out.append(
                {
                    "trade_date": date,
                    "symbol": f"S{sidx:03d}",
                    **{f"feat_{idx}": value for idx, value in enumerate(features)},
                    "forward_return_1d": label_1d,
                    "forward_return_5d": label_5d,
                }
            )
    return pd.DataFrame(rows_out)


def test_trainer_fits_and_round_trips(tmp_path):
    dataset = _synthetic_dataset()
    config = V7DeepAlphaTrainerConfig(
        horizons=(1, 5),
        hidden_sizes=(16, 8),
        max_epochs=3,
        batch_size=64,
        early_stopping_patience=2,
        feature_columns=tuple(f"feat_{i}" for i in range(8)),
        output_dir=str(tmp_path / "deep"),
        use_torch=False,
        seed=11,
    )
    trainer = V7DeepAlphaTrainer(config)
    state = trainer.fit(dataset)
    assert state.backend in {"numpy", "torch"}
    assert len(state.horizons) == 2
    predictions = trainer.predict(dataset)
    assert {"alpha_1d", "alpha_5d"}.issubset(predictions.columns)
    assert len(predictions) == len(dataset)
    state_path = trainer.save(tmp_path / "deep")
    reloaded = V7DeepAlphaTrainer(config)
    reloaded.load(state_path)
    reloaded_predictions = reloaded.predict(dataset)
    np.testing.assert_allclose(
        predictions[["alpha_1d", "alpha_5d"]].to_numpy(),
        reloaded_predictions[["alpha_1d", "alpha_5d"]].to_numpy(),
        atol=1e-9,
    )


def test_trainer_requires_features():
    trainer = V7DeepAlphaTrainer(V7DeepAlphaTrainerConfig(feature_columns=("nope",), use_torch=False))
    dataset = pd.DataFrame({"trade_date": [pd.Timestamp("2026-01-02")], "symbol": ["S000"], "forward_return_1d": [0.01]})
    try:
        trainer.fit(dataset)
    except ValueError as exc:
        message = str(exc)
        assert "feature" in message.lower()
    else:  # pragma: no cover - sanity
        raise AssertionError("trainer must reject missing feature columns")


def test_trainer_predict_requires_state():
    trainer = V7DeepAlphaTrainer(V7DeepAlphaTrainerConfig(use_torch=False))
    try:
        trainer.predict(pd.DataFrame({"feat_0": [0.0]}))
    except RuntimeError as exc:
        assert "no fitted state" in str(exc)
    else:  # pragma: no cover - sanity
        raise AssertionError("predict must require a fitted state")
