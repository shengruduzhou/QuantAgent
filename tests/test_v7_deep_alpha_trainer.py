"""V7 deep alpha trainer: fit / predict / save / load round-trip."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

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


def _write_schema(tmp_path, feature_columns, *, feature_version="vtest", schema_hash="abc123"):
    path = tmp_path / "feature_schema.json"
    path.write_text(
        json.dumps(
            {
                "feature_version": feature_version,
                "schema_hash": schema_hash,
                "feature_columns": list(feature_columns),
                "label_columns": ["forward_return_1d", "forward_return_5d"],
                "horizons": [1, 5],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_trainer_uses_pinned_schema_columns_in_order(tmp_path):
    """A pinned schema overrides auto-derivation; column order is preserved."""
    dataset = _synthetic_dataset()
    # Deliberately a non-sorted SUBSET so we can prove order is preserved and
    # the trainer did not just auto-derive feat_0..feat_7.
    pinned = ["feat_3", "feat_1", "feat_5", "feat_0"]
    schema_path = _write_schema(tmp_path, pinned, feature_version="v9", schema_hash="deadbeef")
    config = V7DeepAlphaTrainerConfig(
        horizons=(1, 5), hidden_sizes=(8,), max_epochs=2, batch_size=64,
        feature_schema_path=str(schema_path), output_dir=str(tmp_path / "deep"),
        use_torch=False, seed=3,
    )
    state = V7DeepAlphaTrainer(config).fit(dataset)
    assert state.feature_columns == pinned          # exact contract, order preserved
    assert state.feature_version == "v9"
    assert state.schema_hash == "deadbeef"
    assert state.schema_path == str(schema_path)


def test_trainer_fails_fast_on_missing_schema_columns(tmp_path):
    """Required schema columns absent from the dataset must raise, not vanish."""
    dataset = _synthetic_dataset()
    schema_path = _write_schema(tmp_path, ["feat_0", "feat_1", "feat_does_not_exist"])
    config = V7DeepAlphaTrainerConfig(
        horizons=(1, 5), feature_schema_path=str(schema_path), use_torch=False,
    )
    with pytest.raises(ValueError) as exc:
        V7DeepAlphaTrainer(config).fit(dataset)
    msg = str(exc.value)
    assert "feat_does_not_exist" in msg          # the missing column is named, not silently dropped
    assert "schema" in msg.lower()


def test_trainer_no_schema_falls_back_to_auto_derive():
    """Without a schema (and no explicit list) auto-derivation still works."""
    dataset = _synthetic_dataset()
    config = V7DeepAlphaTrainerConfig(horizons=(1, 5), hidden_sizes=(8,), max_epochs=2, use_torch=False, seed=5)
    state = V7DeepAlphaTrainer(config).fit(dataset)
    assert set(f"feat_{i}" for i in range(8)).issubset(set(state.feature_columns))
    assert state.feature_version == "" and state.schema_hash == ""   # no schema provenance


def test_trainer_run_metadata_records_schema(tmp_path):
    """Saved checkpoint + manifest + schema JSON carry version/hash/path/count."""
    dataset = _synthetic_dataset()
    pinned = ["feat_2", "feat_4", "feat_6"]
    schema_path = _write_schema(tmp_path, pinned, feature_version="v9", schema_hash="cafef00d")
    config = V7DeepAlphaTrainerConfig(
        horizons=(1, 5), hidden_sizes=(8,), max_epochs=2, feature_schema_path=str(schema_path),
        output_dir=str(tmp_path / "deep"), use_torch=False, seed=7,
    )
    trainer = V7DeepAlphaTrainer(config)
    trainer.fit(dataset)
    out = trainer.save(tmp_path / "deep")
    out_dir = out.parent

    state_json = json.loads((out_dir / "deep_alpha_state.json").read_text())
    assert state_json["feature_version"] == "v9"
    assert state_json["schema_hash"] == "cafef00d"

    schema_json = json.loads((out_dir / "deep_alpha_feature_schema.json").read_text())
    assert schema_json["feature_version"] == "v9"
    assert schema_json["schema_hash"] == "cafef00d"
    assert schema_json["feature_count"] == len(pinned)
    assert schema_json["source_schema_path"] == str(schema_path)

    manifest = json.loads((out_dir / "deep_alpha_experiment_manifest.json").read_text())
    assert manifest["schema_hash"] == "cafef00d"
    assert manifest["feature_columns_count"] == len(pinned)

    config_json = json.loads((out_dir / "deep_alpha_config.json").read_text())
    assert config_json["feature_schema_path"] == str(schema_path)

    # Provenance survives a load round-trip.
    reloaded = V7DeepAlphaTrainer(config)
    reloaded.load(out)
    assert reloaded.state.feature_version == "v9"
    assert reloaded.state.schema_hash == "cafef00d"


def test_trainer_predict_requires_state():
    trainer = V7DeepAlphaTrainer(V7DeepAlphaTrainerConfig(use_torch=False))
    try:
        trainer.predict(pd.DataFrame({"feat_0": [0.0]}))
    except RuntimeError as exc:
        assert "no fitted state" in str(exc)
    else:  # pragma: no cover - sanity
        raise AssertionError("predict must require a fitted state")
