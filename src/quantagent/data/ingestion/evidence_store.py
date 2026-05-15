"""Unified EvidenceRecord store for the V7 evidence layer.

The :class:`EvidenceStore` writes the daily evidence frame to a partitioned
Parquet (or CSV fallback) tree keyed by ``available_at`` so the downstream
PIT walk-through can read only evidence that was *visible* at a given
``as_of_date``. Each call appends to the partition; duplicate ``raw_hash``
rows are dropped on write so the store is idempotent across reruns.

The store deliberately does no online I/O. Ingestors are responsible for
calling external APIs; once a frame is normalised through
:func:`normalise_evidence_frame`, the store is the single seam where every
:class:`EvidenceRecord` lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantagent.data.ingestion.daily_evidence_job import EVIDENCE_COLUMNS


@dataclass(frozen=True)
class EvidenceStoreConfig:
    root: str = "data/v7/evidence/store"
    partition_column: str = "available_at"
    file_format: str = "parquet"


class EvidenceStore:
    """Append-only PIT evidence store."""

    def __init__(self, config: EvidenceStoreConfig | None = None) -> None:
        self.config = config or EvidenceStoreConfig()

    def write(self, frame: pd.DataFrame) -> list[Path]:
        if frame is None or frame.empty:
            return []
        written: list[Path] = []
        root = Path(self.config.root)
        root.mkdir(parents=True, exist_ok=True)
        column = self.config.partition_column
        if column not in frame.columns:
            raise ValueError(f"evidence frame missing partition column '{column}'")
        data = frame.copy()
        data[column] = pd.to_datetime(data[column], errors="coerce").dt.strftime("%Y-%m-%d")
        for partition, partition_frame in data.dropna(subset=[column]).groupby(column, sort=True):
            path = root / f"{column}={partition}" / f"evidence.{self._extension()}"
            path.parent.mkdir(parents=True, exist_ok=True)
            merged = self._merge_with_existing(path, partition_frame)
            self._write_frame(path, merged)
            written.append(path)
        return written

    def read_visible(self, as_of_date: str) -> pd.DataFrame:
        root = Path(self.config.root)
        if not root.exists():
            return pd.DataFrame(columns=list(EVIDENCE_COLUMNS))
        cutoff = pd.Timestamp(as_of_date)
        frames: list[pd.DataFrame] = []
        column = self.config.partition_column
        for child in sorted(root.glob(f"{column}=*")):
            partition_value = child.name.split("=", 1)[-1]
            try:
                partition_ts = pd.Timestamp(partition_value)
            except (TypeError, ValueError):
                continue
            if partition_ts > cutoff:
                continue
            for path in sorted(child.glob(f"evidence.{self._extension()}")):
                frames.append(self._read_frame(path))
        if not frames:
            return pd.DataFrame(columns=list(EVIDENCE_COLUMNS))
        return pd.concat(frames, ignore_index=True, sort=False)

    def quality_report(self, as_of_date: str | None = None) -> dict[str, object]:
        frame = self.read_visible(as_of_date) if as_of_date else self._read_all()
        return build_evidence_quality_report(frame, as_of_date=as_of_date, required_columns=EVIDENCE_COLUMNS)

    def _merge_with_existing(self, path: Path, partition_frame: pd.DataFrame) -> pd.DataFrame:
        if not path.exists():
            return partition_frame.drop_duplicates(subset=["raw_hash"])
        existing = self._read_frame(path)
        merged = pd.concat([existing, partition_frame], ignore_index=True, sort=False)
        if "raw_hash" in merged.columns:
            merged = merged.drop_duplicates(subset=["raw_hash"], keep="last")
        return merged

    def _write_frame(self, path: Path, frame: pd.DataFrame) -> None:
        if self.config.file_format == "parquet":
            try:
                frame.to_parquet(path, index=False)
                return
            except (ImportError, ValueError):  # pragma: no cover - parquet engine missing
                pass
        frame.to_csv(path.with_suffix(".csv"), index=False)

    def _read_frame(self, path: Path) -> pd.DataFrame:
        if self.config.file_format == "parquet" and path.exists():
            try:
                return pd.read_parquet(path)
            except (ImportError, ValueError):
                pass
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame(columns=list(EVIDENCE_COLUMNS))

    def _extension(self) -> str:
        return "parquet" if self.config.file_format == "parquet" else "csv"

    def _read_all(self) -> pd.DataFrame:
        root = Path(self.config.root)
        if not root.exists():
            return pd.DataFrame(columns=list(EVIDENCE_COLUMNS))
        frames: list[pd.DataFrame] = []
        for path in sorted(root.glob(f"{self.config.partition_column}=*/evidence.{self._extension()}")):
            frames.append(self._read_frame(path))
        if not frames:
            return pd.DataFrame(columns=list(EVIDENCE_COLUMNS))
        return pd.concat(frames, ignore_index=True, sort=False)


def build_evidence_quality_report(
    frame: pd.DataFrame,
    *,
    as_of_date: str | None = None,
    required_columns: tuple[str, ...] = EVIDENCE_COLUMNS,
) -> dict[str, object]:
    if frame is None:
        frame = pd.DataFrame()
    missing = [column for column in required_columns if column not in frame.columns]
    duplicate_rate = 0.0
    if not frame.empty and "raw_hash" in frame.columns:
        duplicate_rate = float(frame["raw_hash"].duplicated().mean())
    pit_violation_count = 0
    if as_of_date and "available_at" in frame.columns:
        pit_violation_count = int((pd.to_datetime(frame["available_at"], errors="coerce") > pd.Timestamp(as_of_date)).sum())
    reliability = 0.0
    if "source_reliability" in frame.columns and not frame.empty:
        reliability = float(pd.to_numeric(frame["source_reliability"], errors="coerce").fillna(0.0).mean())
    return {
        "row_count": int(len(frame)),
        "missing_columns": missing,
        "source_reliability_mean": reliability,
        "duplicate_rate": duplicate_rate,
        "pit_violation_count": pit_violation_count,
    }
