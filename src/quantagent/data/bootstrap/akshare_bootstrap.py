"""AkShare financial bootstrap for the V7 PIT cache.

Raw descriptive rows may be retained for forensic/research inspection, but only
provider results with genuine announcement-date + trading-session availability
evidence may enter the silver PIT fundamentals cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.data.bootstrap.akshare_market_bootstrap import _load_akshare_research_calendar
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
    research_calendar, calendar_meta, calendar_warnings = _load_akshare_research_calendar(
        allow_network=config.allow_network
    )
    provider = AkShareFinancialProvider(
        allow_network=config.allow_network,
        available_lag_days=config.available_lag_days,
        retry_count=config.retry_count,
        retry_sleep_seconds=config.retry_sleep_seconds,
        rate_limit_seconds=config.rate_limit_seconds,
        trading_calendar=research_calendar,
    )
    statements = provider.all_statements(request)

    resolved_lake_root = config.lake_root or str(quant_paths().data_root / "v7")
    resolved_fundamentals_root = config.fundamentals_root or str(
        quant_paths().data_root / "v7" / "raw" / "akshare" / "fundamentals"
    )
    lake = v7_lake_paths(resolved_lake_root).ensure() if config.use_lake_layout else None
    silver_root = lake.silver_fundamentals if lake else Path(resolved_fundamentals_root) / "silver_pit"
    silver_root.mkdir(parents=True, exist_ok=True)
    silver_cache = FinancialStatementCache(FinancialCacheConfig(root=str(silver_root)))
    raw_cache = FinancialStatementCache(FinancialCacheConfig(root=resolved_fundamentals_root))

    summary: dict[str, dict[str, object]] = {}
    manifest_paths: list[Path] = []
    combined_warnings: list[str] = list(calendar_warnings)
    aggregate_rows = 0
    silver_rows = 0
    pit_silver_statements: list[str] = []
    non_pit_raw_only: list[str] = []

    for statement, result in statements.items():
        rows = int(0 if result.frame is None else len(result.frame))
        aggregate_rows += rows
        combined_warnings.extend(result.warnings)
        raw_path = raw_cache.upsert(statement, result.frame)
        schema_report = result.metadata.get("schema_report", {}) if result.metadata else {}
        failed_symbols = list(result.metadata.get("failed_symbols", []) if result.metadata else [])
        silver_eligible = bool(
            rows > 0
            and result.point_in_time
            and schema_report.get("status") == "passed"
            and not failed_symbols
        )

        silver_path: Path | None = None
        if silver_eligible:
            silver_path = silver_cache.upsert(statement, result.frame)
            silver_rows += rows
            pit_silver_statements.append(statement)
        else:
            non_pit_raw_only.append(statement)
            combined_warnings.append(f"akshare_{statement}_silver_blocked_non_pit")

        output_paths = [raw_path]
        if silver_path is not None:
            output_paths.append(silver_path)
        manifest = build_manifest_for_frame(
            dataset_name=f"fundamentals_{statement}",
            vendor="akshare",
            frame=result.frame if result.frame is not None else pd.DataFrame(),
            output_paths=output_paths,
            start_date=config.start_date,
            end_date=config.end_date,
            symbols=config.symbols,
            required_columns=AKSHARE_FINANCIAL_REQUIRED_COLUMNS,
            pit_violation_count=int(schema_report.get("pit_violation_count", 0)),
            warnings=tuple(result.warnings),
            extra={
                "statement": statement,
                "source": result.source,
                "function_name": result.metadata.get("function_name") if result.metadata else None,
                "akshare_version": result.metadata.get("akshare_version") if result.metadata else None,
                "params": result.metadata.get("params", {}) if result.metadata else {},
                "schema_hash": result.metadata.get("schema_hash") if result.metadata else None,
                "fetched_at": result.metadata.get("fetched_at") if result.metadata else None,
                "failed_symbols": failed_symbols,
                "provider_point_in_time": bool(result.point_in_time),
                "silver_eligible": silver_eligible,
                "calendar": calendar_meta,
                "production_integrity_certified": False,
                "license_warning": "AkShare endpoints and upstream website fields may change; verify vendor terms before production use.",
            },
        )
        manifest_path = (
            lake.manifests / f"fundamentals_{statement}.json"
            if lake
            else silver_root / f"{statement}_manifest.json"
        )
        manifest.write(manifest_path)
        manifest_paths.append(manifest_path)
        summary[statement] = {
            "rows": rows,
            "source": result.source,
            "raw_path": str(_existing_written_path(raw_path)),
            "silver_path": None if silver_path is None else str(_existing_written_path(silver_path)),
            "silver_eligible": silver_eligible,
            "point_in_time": bool(result.point_in_time),
            "manifest_path": str(manifest_path),
            "warnings": list(result.warnings),
            "failed_symbols": failed_symbols,
            "function_name": result.metadata.get("function_name") if result.metadata else None,
            "schema_hash": result.metadata.get("schema_hash") if result.metadata else None,
            "schema_report": schema_report,
        }

    core_statements = {"income", "balance_sheet", "cashflow"}
    core_pit_ready = core_statements.issubset(set(pit_silver_statements))
    return {
        "status": "passed" if core_pit_ready else ("blocked" if aggregate_rows > 0 else "empty"),
        "config": asdict(config),
        "fundamentals_root": str(silver_root),
        "raw_fundamentals_root": str(Path(resolved_fundamentals_root)),
        "statements": summary,
        "manifest_paths": [str(p) for p in manifest_paths],
        "total_rows": aggregate_rows,
        "silver_rows": silver_rows,
        "pit_silver_statements": pit_silver_statements,
        "non_pit_raw_only_statements": non_pit_raw_only,
        "core_pit_ready": core_pit_ready,
        "calendar": calendar_meta,
        "warnings": list(dict.fromkeys(combined_warnings)),
        "production_integrity_certified": False,
    }


def _existing_written_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = path.with_suffix(".csv")
    return fallback if fallback.exists() else path
