"""Shared Typer app and helpers used by every V7 CLI submodule."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import pandas as pd
import typer

from quantagent.config.paths import quant_paths


app = typer.Typer(help="QuantAgent V7 research, fundamentals, and execution CLI.")


def default_v7_lake_root() -> Path:
    """Default V7 lake root for the CLI (unified layout)."""
    return quant_paths().data_root / "v7"


def default_artifact_root() -> Path:
    """Default models / artifact root for the CLI (unified layout)."""
    return quant_paths().models / "v7_alpha"


def default_reports_root() -> Path:
    """Default report output root for the CLI (unified layout)."""
    return quant_paths().reports / "v7"


def default_predictions_root() -> Path:
    """Default predictions root for the CLI (unified layout)."""
    return quant_paths().predictions


def default_target_weights_root() -> Path:
    """Default target-weights root for the CLI (unified layout)."""
    return quant_paths().target_weights


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
            try:
                import polars as pl
            except ImportError:
                raise
            return pl.read_parquet(str(file_path)).to_pandas()
    return pd.read_csv(file_path)


def write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            frame.to_parquet(path, index=False)
            return path
        except Exception:
            try:
                import polars as pl
            except ImportError:
                path = path.with_suffix(".csv")
                frame.to_csv(path, index=False)
                return path
            try:
                pl.from_pandas(frame).write_parquet(str(path))
                return path
            except Exception:
                path = path.with_suffix(".csv")
    frame.to_csv(path, index=False)
    return path


def parse_csv_tuple(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_symbols_file(path: Path | str | None) -> tuple[str, ...]:
    """Read one-symbol-per-line universe files.

    Blank lines and lines starting with ``#`` are ignored. Inline comments
    are also stripped so users can annotate large universe files.
    """
    if path is None:
        return ()
    symbols: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            symbols.append(line)
    return tuple(symbols)


def merge_symbols(symbols: str | None = None, symbols_file: Path | str | None = None) -> tuple[str, ...]:
    """Merge comma-separated symbols and a symbols file with stable de-dupe."""
    merged: list[str] = []
    seen: set[str] = set()
    for symbol in (*parse_csv_tuple(symbols), *parse_symbols_file(symbols_file)):
        normalized = str(symbol).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return tuple(merged)
