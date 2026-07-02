from __future__ import annotations

from dataclasses import dataclass

from services.quant_api.adapters.backtests import BacktestAdapter
from services.quant_api.adapters.do_t import DoTAdapter
from services.quant_api.adapters.factors import FactorAdapter
from services.quant_api.adapters.models import ModelAdapter
from services.quant_api.adapters.risk import RiskAdapter
from services.quant_api.adapters.selection import SelectionAdapter
from services.quant_api.config import ApiSettings, default_settings
from services.quant_api.runtime_indexer import RuntimeIndexer
from services.quant_api.services.jobs import JobManager
from services.quant_api.services.runtime_cleanup import RuntimeCleanupService


@dataclass
class ServiceContainer:
    settings: ApiSettings
    indexer: RuntimeIndexer
    backtests: BacktestAdapter
    factors: FactorAdapter
    models: ModelAdapter
    selections: SelectionAdapter
    do_t: DoTAdapter
    risk: RiskAdapter
    jobs: JobManager
    cleanup: RuntimeCleanupService

    @classmethod
    def create(cls, settings: ApiSettings | None = None) -> "ServiceContainer":
        resolved = (settings or default_settings()).ensure()
        indexer = RuntimeIndexer(resolved)
        backtests = BacktestAdapter(resolved, indexer)
        return cls(
            settings=resolved,
            indexer=indexer,
            backtests=backtests,
            factors=FactorAdapter(resolved),
            models=ModelAdapter(resolved),
            selections=SelectionAdapter(resolved),
            do_t=DoTAdapter(resolved),
            risk=RiskAdapter(backtests),
            jobs=JobManager(resolved),
            cleanup=RuntimeCleanupService(resolved),
        )
