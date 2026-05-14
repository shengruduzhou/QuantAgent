"""Local Parquet/CSV cache for PIT-aware financial statements.

The cache lives under ``data/v7/fundamentals/`` and is structured so that
each statement type is its own file with the columns:

    symbol | report_period | ann_date | available_at | <statement fields...> | source | source_reliability

The cache never silently fills missing rows. It only stores what the
caller provided. When a downstream pipeline queries with an as-of date,
``load_pit_frame`` performs a strict ``available_at <= as_of`` filter so
the same Parquet can be replayed at any historical date without leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from quantagent.data.providers.base import ProviderResult


_STATEMENT_FILES = {
    "income": "income.parquet",
    "balance_sheet": "balance_sheet.parquet",
    "cashflow": "cashflow.parquet",
    "financial_indicator": "financial_indicator.parquet",
    "disclosure_dates": "disclosure_dates.parquet",
}


@dataclass(frozen=True)
class FinancialCacheConfig:
    root: str = "data/v7/fundamentals"
    format: str = "parquet"


class FinancialStatementCache:
    """Persistent local cache for PIT-aware financial statements."""

    def __init__(self, config: FinancialCacheConfig | None = None) -> None:
        self.config = config or FinancialCacheConfig()
        self.root = Path(self.config.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def upsert(self, statement: str, frame: pd.DataFrame) -> Path:
        if statement not in _STATEMENT_FILES:
            raise ValueError(f"Unknown statement type: {statement}")
        if frame is None or frame.empty:
            return self._path(statement)
        existing = self._read(statement)
        merged = pd.concat([existing, frame], ignore_index=True) if not existing.empty else frame.copy()
        key_columns = [column for column in ("symbol", "report_period", "ann_date") if column in merged.columns]
        if key_columns:
            merged = merged.drop_duplicates(subset=key_columns, keep="last")
        path = self._path(statement)
        self._write(merged, path)
        return path

    def load_pit_frame(
        self,
        statement: str,
        as_of_date: str,
        symbols: Iterable[str] | None = None,
    ) -> ProviderResult:
        if statement not in _STATEMENT_FILES:
            raise ValueError(f"Unknown statement type: {statement}")
        frame = self._read(statement)
        if frame.empty:
            return ProviderResult(
                pd.DataFrame(),
                source=f"financial_cache_missing:{statement}",
                point_in_time=True,
                quality_score=0.0,
                warnings=(f"missing_financial_cache_{statement}",),
            )
        filtered = apply_point_in_time_filter(frame, as_of_date)
        if symbols is not None and "symbol" in filtered.columns:
            symbol_set = {str(item) for item in symbols}
            filtered = filtered[filtered["symbol"].astype(str).isin(symbol_set)]
        return ProviderResult(
            filtered.reset_index(drop=True),
            source=f"financial_cache:{statement}",
            point_in_time=True,
            quality_score=0.90 if not filtered.empty else 0.0,
            warnings=() if not filtered.empty else (f"empty_after_pit_filter_{statement}",),
            metadata={"as_of_date": as_of_date, "path": str(self._path(statement))},
        )

    def load_all_pit(self, as_of_date: str, symbols: Iterable[str] | None = None) -> dict[str, ProviderResult]:
        return {
            statement: self.load_pit_frame(statement, as_of_date, symbols)
            for statement in _STATEMENT_FILES
        }

    def _path(self, statement: str) -> Path:
        suffix = ".parquet" if self.config.format == "parquet" else ".csv"
        return self.root / _STATEMENT_FILES[statement].replace(".parquet", suffix)

    def _read(self, statement: str) -> pd.DataFrame:
        path = self._path(statement)
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

    The filter is intentionally strict — a missing ``available_at`` cell
    is treated as ``not yet available``, since we cannot prove visibility.
    Callers should set ``available_at`` deterministically when the
    statement is upserted into the cache.
    """

    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if date_column not in data.columns:
        return pd.DataFrame()
    parsed = pd.to_datetime(data[date_column], errors="coerce")
    cutoff = pd.Timestamp(as_of_date)
    return data[parsed.notna() & (parsed <= cutoff)].copy()
