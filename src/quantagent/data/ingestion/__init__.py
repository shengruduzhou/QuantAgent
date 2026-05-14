"""Daily evidence ingestion layer for QuantAgent V7.

This subpackage owns the seam between the outside world (政策、公告、新闻、
财报、订单合同、监管处罚) and the V7 EvidenceRecord schema. Every ingestor
in this package emits a uniform DataFrame with the columns documented in
``EVIDENCE_COLUMNS`` so the daily_evidence_job can concatenate them into a
single PIT-safe evidence table.
"""

from quantagent.data.ingestion.daily_evidence_job import (
    DailyEvidenceJob,
    DailyEvidenceJobConfig,
    DailyEvidenceJobResult,
    EVIDENCE_COLUMNS,
    EvidenceIngestor,
    attach_source_profile,
    enforce_pit,
    normalise_evidence_frame,
)
from quantagent.data.ingestion.disclosure_ingestor import DisclosureIngestor
from quantagent.data.ingestion.evidence_store import EvidenceStore, EvidenceStoreConfig
from quantagent.data.ingestion.financial_ingestor import FinancialIngestor
from quantagent.data.ingestion.news_ingestor import NewsIngestor
from quantagent.data.ingestion.order_contract_ingestor import OrderContractIngestor
from quantagent.data.ingestion.policy_ingestor import PolicyIngestor
from quantagent.data.ingestion.regulatory_penalty_ingestor import RegulatoryPenaltyIngestor
from quantagent.data.ingestion.source_registry import (
    SourceCredibilityRegistry,
    SourceProfile,
    SourceTier,
    merge_user_profiles,
)


__all__ = [
    "DailyEvidenceJob",
    "DailyEvidenceJobConfig",
    "DailyEvidenceJobResult",
    "DisclosureIngestor",
    "EVIDENCE_COLUMNS",
    "EvidenceIngestor",
    "EvidenceStore",
    "EvidenceStoreConfig",
    "FinancialIngestor",
    "NewsIngestor",
    "OrderContractIngestor",
    "PolicyIngestor",
    "RegulatoryPenaltyIngestor",
    "SourceCredibilityRegistry",
    "SourceProfile",
    "SourceTier",
    "attach_source_profile",
    "enforce_pit",
    "merge_user_profiles",
    "normalise_evidence_frame",
]
