"""Execution simulation and broker-state reconciliation.

No object exported here submits live orders.  QMT remains behind broker and risk
gates.
"""

from quantagent.execution.auction_impact import (
    AuctionFillEstimate,
    AuctionSnapshot,
    MarketImpactConfig,
    estimate_continuous_market_impact_bps,
    estimate_opening_auction_fill,
)
from quantagent.execution.reconciliation import (
    BrokerSnapshot,
    ReconciliationResult,
    ReconciliationTolerance,
    V6ReconciliationReport,
    reconcile_broker_state,
    reconcile_virtual_state,
)

__all__ = [
    "AuctionFillEstimate",
    "AuctionSnapshot",
    "BrokerSnapshot",
    "MarketImpactConfig",
    "ReconciliationResult",
    "ReconciliationTolerance",
    "V6ReconciliationReport",
    "estimate_continuous_market_impact_bps",
    "estimate_opening_auction_fill",
    "reconcile_broker_state",
    "reconcile_virtual_state",
]
