from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            try:
                import polars as pl
            except ImportError:
                raise
            return pl.read_parquet(str(path)).to_pandas()
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file type: {path}")


def write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(path, index=False)
        except Exception:
            try:
                import polars as pl
            except ImportError:
                csv_path = path.with_suffix(".csv")
                frame.to_csv(csv_path, index=False)
                return
            pl.from_pandas(frame).write_parquet(str(path))
        return
    if path.suffix.lower() in {".csv", ".txt"}:
        frame.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported data file type: {path}")
