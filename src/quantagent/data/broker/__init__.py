"""Stage 5.1 — broker research reports data layer.

Normalises analyst rating + target-price actions from major Chinese
brokers into a silver-layer dataset with PIT-safe ``available_at`` and
broker-credibility tier weighting.
"""

from quantagent.data.broker.builder import (
    BROKER_REPORT_REQUIRED_COLUMNS,
    BROKER_TIER_TABLE,
    BrokerReportBuilder,
    BrokerReportConfig,
    BrokerReportResult,
    apply_broker_report_features,
    broker_reports_for_features,
    build_broker_reports,
)
from quantagent.data.evidence.canonical import broker_reports_to_evidence

__all__ = [
    "BROKER_REPORT_REQUIRED_COLUMNS",
    "BROKER_TIER_TABLE",
    "BrokerReportBuilder",
    "BrokerReportConfig",
    "BrokerReportResult",
    "apply_broker_report_features",
    "broker_reports_for_features",
    "broker_reports_to_evidence",
    "build_broker_reports",
]
