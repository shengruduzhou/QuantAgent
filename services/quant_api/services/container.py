from __future__ import annotations

from dataclasses import dataclass
from threading import Thread

from services.quant_api.adapters.backtests import BacktestAdapter
from services.quant_api.adapters.do_t import DoTAdapter
from services.quant_api.adapters.factors import FactorAdapter
from services.quant_api.adapters.fusion import FusionAdapter
from services.quant_api.adapters.models import ModelAdapter
from services.quant_api.adapters.risk import RiskAdapter
from services.quant_api.adapters.selection import SelectionAdapter
from services.quant_api.config import ApiSettings, default_settings
from services.quant_api.events import EventBroker
from services.quant_api.runtime_indexer import RuntimeIndexer
from services.quant_api.services.connections import ConnectionManager
from services.quant_api.services.council import CouncilService
from services.quant_api.services.data_manager import DataManagerService
from services.quant_api.services.governance import GovernanceService
from services.quant_api.services.jobs import JobManager
from services.quant_api.services.paper_orders import PaperOrderService
from services.quant_api.services.runtime_cleanup import RuntimeCleanupService
from services.quant_api.services.strategies import StrategyService
from services.quant_api.services.vnpy_parity import VnpyParityService


def _rebuild_index(indexer: RuntimeIndexer) -> None:
    """Warm the artifact index off the request path; a cold cache is not fatal."""
    try:
        indexer.scan()
    except Exception:
        pass


@dataclass
class ServiceContainer:
    settings: ApiSettings
    indexer: RuntimeIndexer
    backtests: BacktestAdapter
    factors: FactorAdapter
    fusion: FusionAdapter
    models: ModelAdapter
    selections: SelectionAdapter
    do_t: DoTAdapter
    risk: RiskAdapter
    events: EventBroker
    connections: ConnectionManager
    data_manager: DataManagerService
    strategies: StrategyService
    jobs: JobManager
    cleanup: RuntimeCleanupService
    vnpy_parity: VnpyParityService
    governance: GovernanceService
    council: CouncilService
    paper_orders: PaperOrderService

    @classmethod
    def create(cls, settings: ApiSettings | None = None) -> "ServiceContainer":
        resolved = (settings or default_settings()).ensure()
        indexer = RuntimeIndexer(resolved)
        backtests = BacktestAdapter(resolved, indexer)
        events = EventBroker()
        connections = ConnectionManager()
        factors = FactorAdapter(resolved)
        fusion = FusionAdapter(resolved)

        def invalidate_runtime() -> None:
            indexer.invalidate()
            factors.invalidate()
            fusion.invalidate()
            # Rebuilding the index takes tens of seconds on a large runtime.
            # Leaving that to the next request means every finished job is
            # followed by one operator waiting on a frozen page, so pay the cost
            # here instead — on a background thread, right after the job that
            # caused it.
            Thread(
                target=_rebuild_index,
                args=(indexer,),
                name="runtime-index-refresh",
                daemon=True,
            ).start()

        return cls(
            settings=resolved,
            indexer=indexer,
            backtests=backtests,
            factors=factors,
            fusion=fusion,
            models=ModelAdapter(resolved),
            selections=SelectionAdapter(resolved),
            do_t=DoTAdapter(resolved),
            risk=RiskAdapter(backtests),
            events=events,
            connections=connections,
            data_manager=DataManagerService(resolved, connections),
            strategies=StrategyService(resolved),
            jobs=JobManager(
                resolved,
                events,
                invalidate_runtime,
                connection_environment=connections.environment_for,
            ),
            cleanup=RuntimeCleanupService(resolved),
            vnpy_parity=VnpyParityService(),
            governance=GovernanceService(resolved),
            council=CouncilService(resolved, fusion),
            # The single-writer lock is held for the API process's lifetime. Two
            # API processes against one paper account would each size orders from
            # their own in-memory portfolio, so the second is refused at startup
            # rather than allowed to over-commit the same cash.
            paper_orders=PaperOrderService(resolved.runtime_root / "paper_orders"),
        )

    def start(self) -> None:
        self.events.start()
        self.events.publish(
            topic="system",
            event_type="service.started",
            payload={"service": "quant_api"},
            source="quant_api.lifecycle",
        )
        # Indexing a multi-terabyte-capable runtime tree takes tens of seconds.
        # Doing it lazily meant the first operator to open any page paid for it
        # while every other request queued behind the same lock. Warm it here so
        # the cost lands before anyone is waiting on it.
        Thread(
            target=_rebuild_index,
            args=(self.indexer,),
            name="runtime-index-warmup",
            daemon=True,
        ).start()

    def stop(self) -> None:
        self.paper_orders.close()
        self.events.publish(
            topic="system",
            event_type="service.stopping",
            payload={"service": "quant_api"},
            source="quant_api.lifecycle",
        )
        self.events.close()
