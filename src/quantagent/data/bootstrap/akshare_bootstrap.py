"""AkShare financial bootstrap for the V7 PIT cache.

Pulls income / balance / cashflow statements through ``AkShareFinancialProvider``,
upserts them into the canonical PIT cache under the unified V7 lake, and emits a
``DataManifest`` recording vendor, rows, schema
violations, warnings and content hashes. Network is opt-in via
``allow_network=True``; otherwise the provider raises ``ProviderUnavailable``
with an actionable hint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.data.lake import v7_lake_paths
from quantagent.data.manifest import build_manifest_for_frame
from quantagent.data.providers.akshare_financial_provider import (
    AKSHARE_FINANCIAL_REQUIRED_COLUMNS,
    AkShareFinancialProvider,
)
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.providers.financial_cache import FinancialCacheConfig, FinancialStatementCache


@dataclass(frozen=True)
class AkShareBootstrapConfig:
    start_date: str
    end_date: str
    symbols: tuple[str, ...]
    fundamentals_root: str | None = None
    allow_network: bool = False
    available_lag_days: int = 1
    retry_count: int = 2
    retry_sleep_seconds: float = 0.5
    rate_limit_seconds: float = 0.2
    use_lake_layout: bool = True
    lake_root: str | None = None


def build_akshare_financial_cache(config: AkShareBootstrapConfig) -> dict[str, object]:
    if not config.symbols:
        raise ValueError("AkShare bootstrap requires at least one symbol")
    request = ProviderRequest(config.start_date, config.end_date, symbols=config.symbols)
    provider = AkShareFinancialProvider(
        allow_network=config.allow_network,
        available_lag_days=config.available_lag_days,
        retry_count=config.retry_count,
        retry_sleep_seconds=config.retry_sleep_seconds,
        rate_limit_seconds=config.rate_limit_seconds,
    )
    statements = provider.all_statements(request)

    resolved_lake_root = config.lake_root or str(quant_paths().data_root / "v7")
    resolved_fundamentals_root = config.fundamentals_root or str(
        quant_paths().data_root / "v7" / "raw" / "akshare" / "fundamentals"
    )
    lake = v7_lake_paths(resolved_lake_root).ensure() if config.use_lake_layout else None
    silver_root = lake.silver_fundamentals if lake else Path(resolved_fundamentals_root)
    silver_root.mkdir(parents=True, exist_ok=True)
    silver_cache = FinancialStatementCache(FinancialCacheConfig(root=str(silver_root)))
    raw_cache = FinancialStatementCache(FinancialCacheConfig(root=resolved_fundamentals_root))

    summary: dict[str, dict[str, object]] = {}
    manifest_paths: list[Path] = []
    combined_warnings: list[str] = []
    aggregate_rows = 0
    for statement, result in statements.items():
        silver_path = silver_cache.upsert(statement, result.frame)
        if lake is not None:
            raw_cache.upsert(statement, result.frame)
        rows = int(0 if result.frame is None else len(result.frame))
        aggregate_rows += rows
        combined_warnings.extend(result.warnings)
        manifest = build_manifest_for_frame(
            dataset_name=f"fundamentals_{statement}",
            vendor="akshare",
            frame=result.frame if result.frame is not None else pd.DataFrame(),
            output_paths=[silver_path],
            start_date=config.start_date,
            end_date=config.end_date,
            symbols=config.symbols,
            required_columns=AKSHARE_FINANCIAL_REQUIRED_COLUMNS,
            pit_violation_count=int(result.metadata.get("schema_report", {}).get("pit_violation_count", 0) if result.metadata else 0),
            warnings=tuple(result.warnings),
            extra={"statement": statement, "source": result.source},
        )
        manifest_path = (lake.manifests / f"fundamentals_{statement}.json") if lake else (silver_root / f"{statement}_manifest.json")
        manifest.write(manifest_path)
        manifest_paths.append(manifest_path)
        summary[statement] = {
            "rows": rows,
            "source": result.source,
            "path": str(_existing_written_path(silver_path)),
            "manifest_path": str(manifest_path),
            "warnings": list(result.warnings),
            "schema_report": result.metadata.get("schema_report", {}) if result.metadata else {},
        }
    return {
        "status": "passed" if aggregate_rows > 0 else "empty",
        "config": asdict(config),
        "fundamentals_root": str(silver_root),
        "raw_fundamentals_root": str(Path(resolved_fundamentals_root)),
        "statements": summary,
        "manifest_paths": [str(p) for p in manifest_paths],
        "total_rows": aggregate_rows,
        "warnings": combined_warnings,
    }


def _existing_written_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = path.with_suffix(".csv")
    return fallback if fallback.exists() else path
