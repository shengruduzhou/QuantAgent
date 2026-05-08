from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file type: {path}")


def write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
        return
    if path.suffix.lower() in {".csv", ".txt"}:
        frame.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported data file type: {path}")
