"""Round 31 D-02: the v7 dataset builder must not turn panel columns that are
unknown at the decision close into model features.

Feature selection used to be a blacklist over whatever the input panel carried,
so the certified gold ``dataset.parquet`` (which stores ``entry_close_t1`` and
``forward_return_*`` next to its features) or a silver panel whose ``is_st`` is
today's ST list broadcast to every date leaked straight into training.
"""
from __future__ import annotations

from quantagent.data import label_contract as _lc  # labels must carry their convention stamp
import json

import numpy as np
import pandas as pd
import pytest

from quantagent.data.dataset_builder.v7_training_dataset import (
    V7TrainingDatasetConfig,
    build_v7_training_dataset_artifact,
)
from quantagent.data.v7_label_builder import build_forward_return_labels


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-04", periods=160)
    rows = []
    for k in range(12):
        sym = f"{600000 + k:06d}.SH"
        close = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
        for i, d in enumerate(dates):
            rows.append({
                "symbol": sym, "trade_date": d, "open": close[i] * 0.999,
                "high": close[i] * 1.01, "low": close[i] * 0.99, "close": close[i],
                "volume": 1e6 + i, "amount": 1e7 + i * 10, "available_at": d,
            })
    return pd.DataFrame(rows).sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _build(tmp_path, panel: pd.DataFrame, **overrides):
    market_path = tmp_path / "market.parquet"
    panel.to_parquet(market_path, index=False)
    base = panel[["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount", "available_at"]]
    labels = build_forward_return_labels(base, (1, 5)).frame
    labels_path = tmp_path / "labels.parquet"
    _lc.write_labels_parquet(labels, labels_path, convention=_lc.V7_SAME_CLOSE)
    config = dict(
        market_panel_path=str(market_path), labels_path=str(labels_path),
        output_path=str(tmp_path / "ds.parquet"), horizons=(1, 5),
        min_rows=10, min_symbols=2, min_dates=10, strict_mode=False,
        enable_macro=False, enable_flow=False, enable_index=False,
    )
    config.update(overrides)
    return build_v7_training_dataset_artifact(V7TrainingDatasetConfig(**config))


def test_future_derived_panel_columns_are_not_features(tmp_path):
    panel = _panel()
    grouped = panel.groupby("symbol")["close"]
    panel["entry_close_t1"] = grouped.shift(-1)
    panel["forward_return_20d"] = grouped.shift(-21) / panel["entry_close_t1"] - 1
    panel["fwd_ret_3d"] = grouped.shift(-3) / panel["close"] - 1
    panel["next_open"] = panel.groupby("symbol")["open"].shift(-1)

    result = _build(tmp_path, panel)
    features = set(result.feature_schema["feature_columns"])

    leaked = features & {"entry_close_t1", "forward_return_20d", "fwd_ret_3d", "next_open"}
    assert not leaked, f"post-decision columns selected as features: {sorted(leaked)}"
    assert {"entry_close_t1", "forward_return_20d", "fwd_ret_3d", "next_open"} <= set(
        result.feature_schema["refused_feature_columns"]
    )
    # A legitimate engineered feature still survives.
    assert "momentum_5d" in features


def test_tradability_flag_is_never_auto_selected(tmp_path):
    panel = _panel()
    panel["is_st"] = panel["symbol"].isin(["600000.SH", "600001.SH"])  # today's list
    result = _build(tmp_path, panel)
    assert "is_st" not in result.feature_schema["feature_columns"]


def test_pinned_schema_demanding_a_post_decision_column_fails_closed(tmp_path):
    panel = _panel()
    panel["entry_close_t1"] = panel.groupby("symbol")["close"].shift(-1)
    pinned = tmp_path / "pinned.feature_schema.json"
    pinned.write_text(json.dumps({"feature_columns": ["momentum_5d", "entry_close_t1"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown at the decision close"):
        _build(tmp_path, panel, expected_feature_schema_path=str(pinned))
