from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import polars as pl


Parser = Callable[[Path, int], dict[str, Any]]


def parse_json(path: Path, limit: int = 100) -> dict[str, Any]:
    if path.stat().st_size > 20 * 1024 * 1024:
        return {"status": "partial", "data": None, "message": "large JSON preview disabled"}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return {"status": "ready", "data": data[:limit], "total": len(data)}
    return {"status": "ready", "data": data}


def parse_csv(path: Path, limit: int = 100) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, row in enumerate(reader):
            if index >= limit:
                break
            rows.append(dict(row))
        return {"status": "ready" if rows else "empty", "columns": reader.fieldnames or [], "data": rows}


def parse_parquet(path: Path, limit: int = 100) -> dict[str, Any]:
    schema = pl.read_parquet_schema(path)
    columns = list(schema)
    selected = columns[: min(len(columns), 50)]
    frame = pl.scan_parquet(path).select(selected).head(limit).collect()
    return {
        "status": "ready" if frame.height else "empty",
        "columns": [{"name": name, "dtype": str(schema[name])} for name in columns],
        "data": frame.to_dicts(),
        "truncatedColumns": len(columns) > len(selected),
    }


def parse_log(path: Path, limit: int = 200) -> dict[str, Any]:
    lines = _tail_lines(path, max(1, limit))
    return {"status": "ready" if lines else "empty", "data": [line.rstrip("\n") for line in lines[-limit:]]}


def parse_metadata_only(path: Path, limit: int = 0) -> dict[str, Any]:
    return {
        "status": "partial",
        "data": None,
        "message": "binary artifact is indexed as metadata only",
    }


PARSER_REGISTRY: dict[str, tuple[str, Parser]] = {
    ".json": ("json", parse_json),
    ".jsonl": ("log", parse_log),
    ".csv": ("csv", parse_csv),
    ".parquet": ("parquet", parse_parquet),
    ".log": ("log", parse_log),
    ".txt": ("log", parse_log),
    ".md": ("log", parse_log),
    ".pt": ("metadata", parse_metadata_only),
    ".pth": ("metadata", parse_metadata_only),
    ".pkl": ("metadata", parse_metadata_only),
    ".pickle": ("metadata", parse_metadata_only),
    ".joblib": ("metadata", parse_metadata_only),
    ".zip": ("metadata", parse_metadata_only),
    ".bin": ("metadata", parse_metadata_only),
}


def parser_for(path: Path) -> tuple[str, Parser]:
    return PARSER_REGISTRY.get(path.suffix.lower(), ("metadata", parse_metadata_only))


def _tail_lines(path: Path, limit: int, chunk_size: int = 64 * 1024) -> list[str]:
    if path.stat().st_size == 0:
        return []
    chunks: list[bytes] = []
    newlines = 0
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        while position > 0 and newlines <= limit:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newlines += chunk.count(b"\n")
    text = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
    return text.splitlines()[-limit:]
