from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class V6Dataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]


def build_v6_dataset(feature_frame: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 20)) -> V6Dataset:
    data = feature_frame.copy().sort_values(["symbol", "trade_date"])
    for horizon in horizons:
        column = f"label_{horizon}d"
        if column not in data.columns:
            data[column] = data.groupby("symbol")["close"].shift(-horizon) / data["close"] - 1.0
    numeric = data.select_dtypes(include=[np.number]).columns
    excluded = {"open", "high", "low", "close", "volume", "amount"} | {f"label_{h}d" for h in horizons}
    feature_columns = tuple(str(c) for c in numeric if str(c) not in excluded)
    label_columns = tuple(f"label_{h}d" for h in horizons)
    return V6Dataset(data.dropna(subset=["trade_date", "symbol"]).reset_index(drop=True), feature_columns, label_columns)

