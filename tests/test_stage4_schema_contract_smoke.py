"""Stage 4 end-to-end smoke test: dataset feature-schema contract ↔ trainer.

Proves the dataset → trainer contract holds end-to-end:

1. Build a gold training dataset → it writes ``feature_schema.json`` with a
   ``schema_hash`` + ``feature_version`` (the contract).
2. Rebuild pinned to that schema → the rebuild reproduces the same hash
   (``--expected-feature-schema`` reproducibility).
3. Train one tiny fold with the trainer pinned to the same schema → the
   trainer records the identical ``schema_hash`` and uses exactly the schema's
   feature columns (count + order), and that provenance survives save().
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.data.dataset_builder import V7TrainingDatasetConfig, build_v7_training_dataset_artifact
from quantagent.data.v7_label_builder import build_forward_return_labels
from quantagent.training.v7_deep_trainer import V7DeepAlphaTrainer, V7DeepAlphaTrainerConfig


def _market_panel(days: int = 80, n_symbols: int = 6, seed: int = 11) -> pd.DataFrame:
    """Small multi-symbol OHLCV panel with enough history for 20d features."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=days)
    rows: list[dict] = []
    for sidx in range(n_symbols):
        symbol = f"60{sidx:04d}.SH"   # main-board prefix
        close = 10.0 + sidx
        for date in dates:
            close = max(1.0, close * (1.0 + rng.normal(0.0, 0.02)))
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000 + sidx * 50_000,
                    "amount": close * (1_000_000 + sidx * 50_000),
                    "available_at": date,
                }
            )
    return pd.DataFrame(rows)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    market = _market_panel()
    labels = build_forward_return_labels(market, horizons=(1, 5)).frame
    market_path = tmp_path / "market.parquet"
    labels_path = tmp_path / "labels.parquet"
    market.to_parquet(market_path, index=False)
    labels.to_parquet(labels_path, index=False)
    return market_path, labels_path


def _build(tmp_path: Path, out_name: str, *, pinned: str | None = None):
    market_path, labels_path = _write_inputs(tmp_path) if not (tmp_path / "market.parquet").exists() else (
        tmp_path / "market.parquet", tmp_path / "labels.parquet"
    )
    return build_v7_training_dataset_artifact(
        V7TrainingDatasetConfig(
            market_panel_path=str(market_path),
            labels_path=str(labels_path),
            output_path=str(tmp_path / out_name),
            horizons=(1, 5),
            min_rows=50,
            min_symbols=2,
            min_dates=10,
            feature_version="v-smoke",
            expected_feature_schema_path=pinned,
        )
    )


def test_stage4_dataset_schema_contract_flows_into_trainer(tmp_path):
    # 1) Build the contract.
    first = _build(tmp_path, "ds1.parquet")
    schema_path = first.feature_schema_path
    contract = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    contract_hash = contract["schema_hash"]
    contract_cols = contract["feature_columns"]
    assert contract["feature_version"] == "v-smoke"
    assert contract_hash and contract_cols

    # 2) Rebuild pinned to the contract → identical hash (reproducible).
    second = _build(tmp_path, "ds2.parquet", pinned=str(schema_path))
    assert second.feature_schema["schema_hash"] == contract_hash
    assert second.feature_schema["feature_columns"] == contract_cols

    # 3) Train one tiny fold, trainer pinned to the SAME schema.
    trainer = V7DeepAlphaTrainer(
        V7DeepAlphaTrainerConfig(
            horizons=(1, 5),
            hidden_sizes=(8,),
            max_epochs=2,
            feature_schema_path=str(schema_path),
            output_dir=str(tmp_path / "model"),
            use_torch=False,
            seed=3,
        )
    )
    state = trainer.fit(second.dataset)

    # End-to-end contract: trainer agrees with the dataset by hash + columns.
    assert state.schema_hash == contract_hash
    assert state.feature_version == "v-smoke"
    assert state.feature_columns == contract_cols          # count + order match
    assert len(state.feature_columns) == len(contract_cols)

    # Provenance is persisted in the model artifacts.
    out = trainer.save(tmp_path / "model")
    model_schema = json.loads((out.parent / "deep_alpha_feature_schema.json").read_text())
    assert model_schema["schema_hash"] == contract_hash
    assert model_schema["feature_count"] == len(contract_cols)
    assert model_schema["source_schema_path"] == str(schema_path)
