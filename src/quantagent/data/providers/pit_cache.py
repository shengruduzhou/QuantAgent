"""Generic Point-In-Time time-series cache.

The existing ``FinancialStatementCache`` is specialised for accounting
statements with (symbol, report_period, ann_date) keys. Macro indicators,
yield curves, money-flow snapshots and index time series share the same
Parquet-with-PIT pattern but use different dedup keys (e.g.
(table_name, observation_date) or (trade_date, symbol, maturity)).

This module exposes a small reusable class that handles the upsert/dedup/
PIT-filter cycle in one place so each new provider can focus on the
akshare-specific schema mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from quantagent.data.providers.base import ProviderResult


@dataclass(frozen=True)
class PITTableSpec:
    """Description of one cached PIT table."""

    name: str
    filename: str
    dedup_keys: tuple[str, ...]
    date_column: str = "available_at"


@dataclass(frozen=True)
class PITCacheConfig:
    root: str
    format: str = "parquet"
    tables: tuple[PITTableSpec, ...] = field(default_factory=tuple)


class PITTimeSeriesCache:
    """Persistent local cache for PIT-aware non-statement time series."""

    def __init__(self, config: PITCacheConfig) -> None:
        if not config.tables:
            raise ValueError("PITCacheConfig.tables must not be empty")
        self.config = config
        self.root = Path(config.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._specs: dict[str, PITTableSpec] = {spec.name: spec for spec in config.tables}

    def list_tables(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def upsert(self, table: str, frame: pd.DataFrame) -> Path:
        spec = self._spec(table)
        if frame is None or frame.empty:
            return self._path(spec)
        existing = self._read(spec)
        merged = (
            pd.concat([existing, frame], ignore_index=True)
            if not existing.empty
            else frame.copy()
        )
        present_keys = [k for k in spec.dedup_keys if k in merged.columns]
        if present_keys:
            merged = merged.drop_duplicates(subset=present_keys, keep="last")
        path = self._path(spec)
        self._write(merged, path)
        return path

    def load_pit_frame(
        self,
        table: str,
        as_of_date: str,
        extra_filters: dict[str, Iterable[str]] | None = None,
    ) -> ProviderResult:
        spec = self._spec(table)
        frame = self._read(spec)
        if frame.empty:
            return ProviderResult(
                pd.DataFrame(),
                source=f"pit_cache_missing:{table}",
                point_in_time=True,
                quality_score=0.0,
                warnings=(f"missing_pit_cache_{table}",),
            )
        filtered = apply_point_in_time_filter(frame, as_of_date, date_column=spec.date_column)
        if extra_filters:
            for column, allowed in extra_filters.items():
                if column not in filtered.columns:
                    continue
                allowed_set = {str(item) for item in allowed}
                filtered = filtered[filtered[column].astype(str).isin(allowed_set)]
        return ProviderResult(
            filtered.reset_index(drop=True),
            source=f"pit_cache:{table}",
            point_in_time=True,
            quality_score=0.85 if not filtered.empty else 0.0,
            warnings=() if not filtered.empty else (f"empty_after_pit_filter_{table}",),
            metadata={
                "as_of_date": as_of_date,
                "path": str(self._path(spec)),
                "date_column": spec.date_column,
            },
        )

    def path_for(self, table: str) -> Path:
        return self._path(self._spec(table))

    def _spec(self, table: str) -> PITTableSpec:
        if table not in self._specs:
            raise KeyError(f"unknown PIT cache table: {table}")
        return self._specs[table]

    def _path(self, spec: PITTableSpec) -> Path:
        suffix = ".parquet" if self.config.format == "parquet" else ".csv"
        filename = spec.filename
        if not filename.endswith(suffix):
            filename = f"{Path(filename).stem}{suffix}"
        return self.root / filename

    def _read(self, spec: PITTableSpec) -> pd.DataFrame:
        path = self._path(spec)
        if not path.exists():
            csv_path = path.with_suffix(".csv") if path.suffix == ".parquet" else None
            if csv_path is not None and csv_path.exists():
                return pd.read_csv(csv_path)
            return pd.DataFrame()
        if path.suffix == ".parquet":
            try:
                return pd.read_parquet(path)
            except Exception:
                csv_path = path.with_suffix(".csv")
                return pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        return pd.read_csv(path)

    def _write(self, frame: pd.DataFrame, path: Path) -> None:
        if path.suffix == ".parquet":
            try:
                frame.to_parquet(path, index=False)
                return
            except Exception:
                path = path.with_suffix(".csv")
        frame.to_csv(path, index=False)


def apply_point_in_time_filter(
    frame: pd.DataFrame,
    as_of_date: str,
    date_column: str = "available_at",
) -> pd.DataFrame:
    """Hard PIT filter: drop every row whose data was not visible at as_of_date.

    Strict — a missing ``available_at`` cell is treated as ``not yet visible``.
    Providers writing into the cache must set ``available_at`` deterministically.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if date_column not in data.columns:
        return pd.DataFrame()
    parsed = pd.to_datetime(data[date_column], errors="coerce")
    cutoff = pd.Timestamp(as_of_date)
    return data[parsed.notna() & (parsed <= cutoff)].copy()


__all__ = [
    "PITTableSpec",
    "PITCacheConfig",
    "PITTimeSeriesCache",
    "apply_point_in_time_filter",
]
