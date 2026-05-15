"""AkShare financial bootstrap for V7 PIT cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from quantagent.data.providers.akshare_financial_provider import AkShareFinancialProvider
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.providers.financial_cache import FinancialCacheConfig, FinancialStatementCache


@dataclass(frozen=True)
class AkShareBootstrapConfig:
    start_date: str
    end_date: str
    symbols: tuple[str, ...]
    fundamentals_root: str = "data/v7/fundamentals"
    allow_network: bool = False
    available_lag_days: int = 1
    retry_count: int = 2
    retry_sleep_seconds: float = 0.5
    rate_limit_seconds: float = 0.2


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
    cache = FinancialStatementCache(FinancialCacheConfig(root=config.fundamentals_root))
    summary: dict[str, dict[str, object]] = {}
    for statement, result in statements.items():
        path = cache.upsert(statement, result.frame)
        summary[statement] = {
            "rows": int(0 if result.frame is None else len(result.frame)),
            "source": result.source,
            "path": str(_existing_written_path(path)),
            "warnings": list(result.warnings),
            "schema_report": result.metadata.get("schema_report", {}),
        }
    return {
        "status": "passed" if any(item["rows"] for item in summary.values()) else "empty",
        "config": asdict(config),
        "fundamentals_root": str(Path(config.fundamentals_root)),
        "statements": summary,
    }


def _existing_written_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = path.with_suffix(".csv")
    return fallback if fallback.exists() else path
