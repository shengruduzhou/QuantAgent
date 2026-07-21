from __future__ import annotations

from dataclasses import dataclass

from services.quant_api.adapters.backtests import BacktestAdapter
from services.quant_api.adapters.do_t import DoTAdapter
from services.quant_api.adapters.factors import FactorAdapter
from services.quant_api.adapters.models import ModelAdapter
from services.quant_api.adapters.risk import RiskAdapter
from services.quant_api.adapters.selection import SelectionAdapter
from services.quant_api.config import ApiSettings, default_settings
from services.quant_api.events import EventBroker
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
    events: EventBroker
    jobs: JobManager
    cleanup: RuntimeCleanupService

    @classmethod
    def create(cls, settings: ApiSettings | None = None) -> "ServiceContainer":
        resolved = (settings or default_settings()).ensure()
        indexer = RuntimeIndexer(resolved)
        backtests = BacktestAdapter(resolved, indexer)
        events = EventBroker()
        return cls(
            settings=resolved,
            indexer=indexer,
            backtests=backtests,
            factors=FactorAdapter(resolved),
            models=ModelAdapter(resolved),
            selections=SelectionAdapter(resolved),
            do_t=DoTAdapter(resolved),
            risk=RiskAdapter(backtests),
            events=events,
            jobs=JobManager(resolved, events),
            cleanup=RuntimeCleanupService(resolved),
        )

    def start(self) -> None:
        self.events.start()
        self.events.publish(
            topic="system",
            event_type="service.started",
            payload={"service": "quant_api"},
            source="quant_api.lifecycle",
        )

    def stop(self) -> None:
        self.events.publish(
            topic="system",
            event_type="service.stopping",
            payload={"service": "quant_api"},
            source="quant_api.lifecycle",
        )
        self.events.close()
