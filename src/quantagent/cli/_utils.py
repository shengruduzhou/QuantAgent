"""Shared Typer app and helpers used by every V7 CLI submodule."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import pandas as pd
import typer


app = typer.Typer(help="QuantAgent V7 research, fundamentals, and execution CLI.")


def json_dump(value: object) -> str:
    """Serialise mixed objects (dataclasses, paths, pandas) to JSON."""

    def default(obj: object) -> object:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        return str(obj)

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=default)


def read_frame(path: Path) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix == ".parquet":
        try:
            return pd.read_parquet(file_path)
        except Exception:
            csv = file_path.with_suffix(".csv")
            if csv.exists():
                return pd.read_csv(csv)
            raise
    return pd.read_csv(file_path)


def write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            frame.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv")
    frame.to_csv(path, index=False)
    return path


def parse_csv_tuple(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())
