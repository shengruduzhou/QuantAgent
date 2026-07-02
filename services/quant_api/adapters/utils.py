from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from services.quant_api.config import ApiSettings, project_relative, safe_project_path


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def read_csv_rows(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        frame = pl.read_csv(path, infer_schema_length=1_000, ignore_errors=True)
    except (OSError, pl.exceptions.PolarsError):
        return []
    if limit is not None:
        frame = frame.head(limit)
    return clean_records(frame.to_dicts())


def read_csv_columns(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            return [str(name) for name in (csv.DictReader(handle).fieldnames or [])]
    except OSError:
        return []


def read_parquet_rows(
    path: Path,
    *,
    columns: list[str] | None = None,
    filters: list[tuple[str, str, Any]] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        lazy = pl.scan_parquet(path)
        schema = lazy.collect_schema()
        if columns:
            lazy = lazy.select([name for name in columns if name in schema])
        for column, operator, value in filters or []:
            if column not in schema:
                continue
            expression = pl.col(column)
            if operator == "eq":
                lazy = lazy.filter(expression == value)
            elif operator == "gte":
                lazy = lazy.filter(expression >= value)
            elif operator == "lte":
                lazy = lazy.filter(expression <= value)
        if limit is not None:
            lazy = lazy.head(limit)
        return clean_records(lazy.collect().to_dicts())
    except (OSError, pl.exceptions.PolarsError):
        return []


def clean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: clean_value(value) for key, value in row.items()} for row in records]


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): clean_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]
    return value


def iter_json_array(path: Path, *, start: int = 0, limit: int = 100) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    index = 0
    yielded = 0
    started = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            chunk = handle.read(64 * 1024)
            if not chunk:
                break
            buffer += chunk
            while True:
                stripped = buffer.lstrip()
                if not stripped:
                    buffer = ""
                    break
                if not started:
                    if not stripped.startswith("["):
                        return
                    stripped = stripped[1:].lstrip()
                    started = True
                if stripped.startswith("]"):
                    return
                if stripped.startswith(","):
                    stripped = stripped[1:].lstrip()
                try:
                    item, consumed = decoder.raw_decode(stripped)
                except json.JSONDecodeError:
                    buffer = stripped
                    break
                buffer = stripped[consumed:]
                if index >= start and yielded < limit and isinstance(item, dict):
                    yield clean_value(item)
                    yielded += 1
                index += 1
                if yielded >= limit:
                    return


def require_relative_path(settings: ApiSettings, path: Path) -> str:
    return project_relative(settings, safe_project_path(settings, path))


def page_slice(items: list[Any], page: int, page_size: int) -> dict[str, Any]:
    page = max(1, int(page))
    page_size = max(1, int(page_size))
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "total": len(items),
        "page": page,
        "pageSize": page_size,
        "hasNext": end < len(items),
    }
