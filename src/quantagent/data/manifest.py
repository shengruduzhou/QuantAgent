"""DataManifest model for V7 real-data lake artifacts.

A DataManifest is a JSON-serialisable record of every dataset materialised
into the V7 lake. It captures provenance (vendor, fetch_time, source paths),
content fingerprints (row counts, content hashes), schema diagnostics
(missing/duplicate/PIT-violation counts), and quality status. Every
production data writer must emit a manifest, so downstream consumers can
prove the artifact is real, PIT-safe and reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCHEMA_VERSION = "v7.manifest.1"


@dataclass
class DataManifest:
    dataset_name: str
    vendor: str
    fetch_time: str
    start_date: str | None = None
    end_date: str | None = None
    symbols: tuple[str, ...] = ()
    universe: str | None = None
    raw_paths: tuple[str, ...] = ()
    output_paths: tuple[str, ...] = ()
    row_count: int = 0
    column_count: int = 0
    schema_version: str = SCHEMA_VERSION
    content_hashes: dict[str, str] = field(default_factory=dict)
    missing_columns: tuple[str, ...] = ()
    duplicate_row_count: int = 0
    duplicate_rate: float = 0.0
    pit_violation_count: int = 0
    warnings: tuple[str, ...] = ()
    quality_status: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["symbols"] = list(self.symbols)
        data["raw_paths"] = list(self.raw_paths)
        data["output_paths"] = list(self.output_paths)
        data["missing_columns"] = list(self.missing_columns)
        data["warnings"] = list(self.warnings)
        return data

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return target


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_file(path: str | Path, chunk_size: int = 1 << 16) -> str:
    """Return a sha256 hash of the file at ``path``.

    The hash is computed in chunks so very large parquet files do not need
    to be loaded into memory. Missing files return an empty string so the
    manifest still records the path without crashing the writer.
    """
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    digest = sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_frame(frame: pd.DataFrame) -> str:
    """Stable sha256 over the canonicalised DataFrame contents."""
    if frame is None or frame.empty:
        return ""
    canonical = frame.sort_index(axis=1).to_csv(index=False).encode("utf-8")
    return sha256(canonical).hexdigest()


def build_manifest_for_frame(
    *,
    dataset_name: str,
    vendor: str,
    frame: pd.DataFrame,
    output_paths: Iterable[str | Path],
    raw_paths: Iterable[str | Path] = (),
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: Iterable[str] = (),
    universe: str | None = None,
    required_columns: Iterable[str] = (),
    pit_violation_count: int = 0,
    warnings: Iterable[str] = (),
    extra: dict[str, Any] | None = None,
) -> DataManifest:
    """Compose a manifest from a finalised DataFrame and the paths it was written to."""
    column_set = set(frame.columns) if frame is not None else set()
    missing = tuple(column for column in required_columns if column not in column_set)
    duplicate_row_count = 0
    if frame is not None and not frame.empty:
        dedupe_keys = [c for c in ("symbol", "trade_date", "report_period", "ann_date") if c in frame.columns]
        if dedupe_keys:
            duplicate_row_count = int(frame.duplicated(subset=dedupe_keys).sum())
    row_count = int(0 if frame is None else len(frame))
    column_count = int(len(column_set))
    duplicate_rate = float(duplicate_row_count / row_count) if row_count else 0.0
    content_hashes: dict[str, str] = {}
    output_path_strings: list[str] = []
    for path in output_paths:
        output_path_strings.append(str(path))
        content_hashes[str(path)] = hash_file(path)
    warnings_tuple = tuple(warnings)
    status = "passed"
    if missing or pit_violation_count or row_count == 0:
        status = "failed"
    if warnings_tuple and status == "passed":
        status = "warning"
    return DataManifest(
        dataset_name=dataset_name,
        vendor=vendor,
        fetch_time=utc_now_iso(),
        start_date=start_date,
        end_date=end_date,
        symbols=tuple(symbols),
        universe=universe,
        raw_paths=tuple(str(p) for p in raw_paths),
        output_paths=tuple(output_path_strings),
        row_count=row_count,
        column_count=column_count,
        content_hashes=content_hashes,
        missing_columns=missing,
        duplicate_row_count=duplicate_row_count,
        duplicate_rate=duplicate_rate,
        pit_violation_count=int(pit_violation_count),
        warnings=warnings_tuple,
        quality_status=status,
        extra=dict(extra or {}),
    )
