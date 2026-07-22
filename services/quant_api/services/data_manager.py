from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
import os
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from typing import Any, Iterator

import pandas as pd

from services.quant_api.config import ApiSettings, project_relative, safe_project_path


@dataclass(frozen=True)
class DataProviderSpec:
    id: str
    label: str
    module: str | None
    command_id: str | None
    asset_classes: tuple[str, ...]
    intervals: tuple[str, ...]
    operations: tuple[str, ...]
    requires: tuple[str, ...] = ()
    optional_requirements: tuple[str, ...] = ()
    note: str = ""


PROVIDERS: tuple[DataProviderSpec, ...] = (
    DataProviderSpec(
        id="tickflow",
        label="TickFlow A股主数据源",
        module="tickflow",
        command_id="fetch-tickflow-daily",
        asset_classes=("A股", "分钟线", "Level-2"),
        intervals=("1d", "1m", "tick", "depth"),
        operations=("download", "update", "record"),
        optional_requirements=("TICKFLOW_API_KEY",),
        note="沿用仓库现有 TickflowProvider。日线可走 free client；分钟线和盘口录制需要 TICKFLOW_API_KEY。",
    ),
    DataProviderSpec(
        id="akshare_market",
        label="AkShare 备用行情",
        module="akshare",
        command_id="build-akshare-market-panel-v7",
        asset_classes=("A股",),
        intervals=("1d",),
        operations=("download", "update"),
        note="备用日线 provider；不会覆盖 TickFlow 产物或静默替换来源。",
    ),
    DataProviderSpec(
        id="qlib_local",
        label="Qlib 本地历史库",
        module="qlib",
        command_id="build-market-panel-v7",
        asset_classes=("A股",),
        intervals=("1d",),
        operations=("query", "export"),
        note="读取已安装的本地 Qlib provider；网页任务不会隐式下载官方数据包。",
    ),
    DataProviderSpec(
        id="tushare_fundamentals",
        label="TuShare PIT 财务",
        module="tushare",
        command_id="build-fundamentals-v7",
        asset_classes=("A股财务",),
        intervals=("report",),
        operations=("download", "update"),
        requires=("TUSHARE_TOKEN",),
        note="Token 仅从后端环境变量读取，永不回传浏览器。",
    ),
    DataProviderSpec(
        id="runtime_catalog",
        label="QuantAgent Runtime",
        module=None,
        command_id=None,
        asset_classes=("PIT artifact",),
        intervals=("manifest",),
        operations=("preview", "coverage", "import", "export", "cleanup"),
        note="直接在服务器 Runtime 上检查和转换，不要求把大文件上传到浏览器。",
    ),
)


class DataManagerService:
    """Provider registry and bounded, server-side dataset inspection surface."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings

    def overview(self) -> dict[str, Any]:
        quarantine = self._runtime_subdir("import_quarantine")
        exports = self._runtime_subdir("exports")
        imported = self._runtime_subdir("data/imported")
        return {
            "providers": [self._provider_payload(spec) for spec in PROVIDERS],
            "constraints": [
                "network access requires an explicit operator confirmation",
                "downloads require explicit symbols or a project-scoped symbols file",
                "large files stay on the server; the browser submits paths, never file bodies",
                "imports start in Runtime/import_quarantine and require schema/duplicate inspection",
                "all Web job outputs must remain inside Runtime",
                "provider results must pass their existing PIT and schema gates",
                "successful jobs invalidate the canonical RuntimeIndexer cache",
                "deletion remains available only through backend-approved Runtime Cleanup candidates",
            ],
            "jobEndpoint": "/api/jobs/data",
            "coverageEndpoint": "/api/data/coverage",
            "quarantineEndpoint": "/api/data/quarantine",
            "supportsCancellation": True,
            "runtimeRoot": "runtime",
            "serverPaths": {
                "quarantine": project_relative(self.settings, quarantine),
                "imports": project_relative(self.settings, imported),
                "exports": project_relative(self.settings, exports),
            },
        }

    def quarantine_files(self) -> list[dict[str, Any]]:
        root = self._runtime_subdir("import_quarantine")
        items: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
                continue
            stat = path.stat()
            items.append({
                "path": project_relative(self.settings, path),
                "name": path.name,
                "format": path.suffix.lower().lstrip("."),
                "sizeBytes": stat.st_size,
                "modifiedAt": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
            })
        return items

    def inspect_dataset(
        self,
        value: str,
        *,
        date_column: str,
        symbol_column: str,
        deep: bool = False,
    ) -> dict[str, Any]:
        path = self._runtime_file(value)
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".parquet"}:
            raise ValueError("coverage inspection supports only CSV or Parquet")

        columns = self._columns(path)
        missing = [name for name in (date_column, symbol_column) if name not in columns]
        if missing:
            raise ValueError(f"dataset is missing required coverage columns: {missing}")

        observed_dates: set[pd.Timestamp] = set()
        observed_symbols: set[str] = set()
        duplicate_count = 0
        scanned_rows = 0
        database_path: str | None = None
        connection: sqlite3.Connection | None = None
        if deep:
            temp = NamedTemporaryFile(prefix="qa-coverage-", suffix=".sqlite", dir=self.settings.cache_root, delete=False)
            database_path = temp.name
            temp.close()
            connection = sqlite3.connect(database_path)
            connection.execute("CREATE TABLE seen(symbol TEXT NOT NULL, stamp TEXT NOT NULL, PRIMARY KEY(symbol, stamp)) WITHOUT ROWID")

        try:
            for frame in self._iter_frames(path, (symbol_column, date_column)):
                if frame.empty:
                    continue
                symbols = frame[symbol_column].astype("string").fillna("").str.strip()
                dates = pd.to_datetime(frame[date_column], errors="coerce", utc=True).dt.tz_convert(None)
                valid = symbols.ne("") & dates.notna()
                symbols = symbols[valid]
                dates = dates[valid]
                scanned_rows += int(valid.sum())
                observed_symbols.update(str(item) for item in symbols.unique())
                observed_dates.update(pd.Timestamp(item).normalize() for item in dates.unique())
                pairs = list(zip(symbols.astype(str), dates.dt.strftime("%Y-%m-%dT%H:%M:%S")))
                if connection is None:
                    duplicate_count += len(pairs) - len(set(pairs))
                else:
                    before = connection.total_changes
                    connection.executemany("INSERT OR IGNORE INTO seen(symbol, stamp) VALUES (?, ?)", pairs)
                    duplicate_count += len(pairs) - (connection.total_changes - before)
            if connection is not None:
                connection.commit()
        finally:
            if connection is not None:
                connection.close()
            if database_path:
                Path(database_path).unlink(missing_ok=True)

        ordered_dates = sorted(observed_dates)
        missing_business_days: list[str] = []
        if ordered_dates:
            expected = set(pd.bdate_range(ordered_dates[0], ordered_dates[-1]).normalize())
            missing_business_days = [item.date().isoformat() for item in sorted(expected - observed_dates)]
        stat = path.stat()
        return {
            "path": project_relative(self.settings, path),
            "format": suffix.lstrip("."),
            "sizeBytes": stat.st_size,
            "columns": columns,
            "rows": self._row_count(path),
            "scannedKeyRows": scanned_rows,
            "symbolCount": len(observed_symbols),
            "dateCount": len(observed_dates),
            "dateStart": ordered_dates[0].date().isoformat() if ordered_dates else None,
            "dateEnd": ordered_dates[-1].date().isoformat() if ordered_dates else None,
            "duplicateKeys": duplicate_count,
            "duplicateMode": "exact" if deep else "within_batch",
            "missingBusinessDayCandidates": missing_business_days[:240],
            "missingBusinessDayCount": len(missing_business_days),
            "dateColumn": date_column,
            "symbolColumn": symbol_column,
            "warnings": [
                "business-day gaps are candidates; exchange holidays require the canonical trading calendar"
            ] if missing_business_days else [],
        }

    def _provider_payload(self, spec: DataProviderSpec) -> dict[str, Any]:
        installed = True if spec.module is None else _module_available(spec.module)
        missing_requirements = [name for name in spec.requires if not os.getenv(name)]
        missing_optional = [name for name in spec.optional_requirements if not os.getenv(name)]
        configured = installed and not missing_requirements
        status = "ready" if configured and not missing_optional else "partial" if configured else "needs_configuration" if installed else "unavailable"
        payload = asdict(spec)
        payload.update(
            {
                "commandId": payload.pop("command_id"),
                "assetClasses": list(payload.pop("asset_classes")),
                "installed": installed,
                "configured": configured,
                "status": status,
                "missingRequirements": missing_requirements,
                "missingOptionalRequirements": missing_optional,
            }
        )
        payload["intervals"] = list(payload["intervals"])
        payload["operations"] = list(payload["operations"])
        payload["requires"] = list(payload["requires"])
        payload["optionalRequirements"] = list(payload.pop("optional_requirements"))
        return payload

    def _runtime_subdir(self, relative: str) -> Path:
        path = (self.settings.runtime_root / relative).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _runtime_file(self, value: str) -> Path:
        path = safe_project_path(self.settings, value)
        runtime = self.settings.runtime_root.resolve()
        if runtime not in path.parents:
            raise ValueError("dataset path must be inside Runtime")
        if not path.is_file():
            raise ValueError("dataset path does not exist or is not a file")
        return path

    @staticmethod
    def _columns(path: Path) -> list[str]:
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            return list(pq.ParquetFile(path).schema_arrow.names)
        return list(pd.read_csv(path, nrows=0).columns)

    @staticmethod
    def _row_count(path: Path) -> int:
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            return int(pq.ParquetFile(path).metadata.num_rows)
        return sum(len(chunk) for chunk in pd.read_csv(path, usecols=[0], chunksize=100_000))

    @staticmethod
    def _iter_frames(path: Path, columns: tuple[str, str]) -> Iterator[pd.DataFrame]:
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=100_000, columns=list(columns)):
                yield batch.to_pandas()
            return
        yield from pd.read_csv(path, usecols=list(columns), chunksize=100_000)


def _module_available(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
