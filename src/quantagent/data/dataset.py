from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - optional training dependency
    raise ImportError("Datasets require the training extra: pip install -e .[training]") from exc


@dataclass(frozen=True)
class WindowSpec:
    lookback_days: int
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    date_column: str = "trade_date"
    symbol_column: str = "symbol"


class EquityWindowDataset(Dataset):
    """Rolling per-symbol windows for daily alpha models."""

    def __init__(self, frame: pd.DataFrame, spec: WindowSpec) -> None:
        self.spec = spec
        self.frame = frame.copy()
        self.frame[spec.date_column] = pd.to_datetime(self.frame[spec.date_column])
        self.frame = self.frame.sort_values([spec.symbol_column, spec.date_column]).reset_index(drop=True)
        self.samples = self._build_sample_index()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        symbol, end_pos = self.samples[index]
        group = self.frame[self.frame[self.spec.symbol_column] == symbol]
        start_pos = end_pos - self.spec.lookback_days + 1
        window = group.iloc[start_pos : end_pos + 1]
        row = group.iloc[end_pos]

        features = window.loc[:, self.spec.feature_columns].to_numpy(dtype=np.float32)
        labels = row.loc[list(self.spec.label_columns)].to_numpy(dtype=np.float32)
        return {
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
            "symbol": str(symbol),
            "trade_date": str(row[self.spec.date_column].date()),
        }

    def _build_sample_index(self) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        required = list(self.spec.feature_columns + self.spec.label_columns)
        for symbol, group in self.frame.groupby(self.spec.symbol_column, sort=False):
            valid = ~group[required].isna().any(axis=1)
            group = group.reset_index(drop=True)
            for end_pos in range(self.spec.lookback_days - 1, len(group)):
                window_valid = valid.iloc[end_pos - self.spec.lookback_days + 1 : end_pos + 1].all()
                label_valid = valid.iloc[end_pos]
                if bool(window_valid and label_valid):
                    samples.append((str(symbol), end_pos))
        return samples
