"""Financial statement ingestor.

Wraps the PIT-aware TuShare / AkShare financial providers and the local
Parquet cache so the daily evidence job can include "fundamentals" as a
first-class evidence stream alongside policy and news.

The frame returned by :meth:`fetch` is intentionally lightweight: one row
per (symbol, report_period) with the latest visible report, with
``title`` set to ``"Financial report"`` and ``body`` populated by the
key numeric fields. Downstream agents (financial_statement_agent,
fraud_risk_agent, intrinsic_valuation) operate on the canonical PIT
fundamentals frame inside :class:`V7DataHub`; this ingestor only registers
the existence of the report into the evidence stream so attribution and
audit logs can trace where a fundamental signal came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantagent.data.ingestion.daily_evidence_job import (
    DailyEvidenceJobConfig,
    EvidenceIngestor,
    attach_source_profile,
)
from quantagent.data.ingestion.source_registry import SourceCredibilityRegistry, SourceTier
from quantagent.data.providers.financial_cache import FinancialCacheConfig, FinancialStatementCache


@dataclass
class FinancialIngestor(EvidenceIngestor):
    name: str = "financial"
    source_type: str = "financial"
    cache_root: str = "data/v7/fundamentals"

    def fetch(
        self,
        config: DailyEvidenceJobConfig,
        registry: SourceCredibilityRegistry,
    ) -> pd.DataFrame:
        cache = FinancialStatementCache(FinancialCacheConfig(root=self.cache_root))
        statements = cache.load_all_pit(config.as_of_date)
        rows: list[dict[str, object]] = []
        for statement_name, result in statements.items():
            frame = result.frame
            if frame is None or frame.empty:
                continue
            for _, row in frame.iterrows():
                symbol = str(row.get("symbol", ""))
                if not symbol:
                    continue
                body = "; ".join(
                    f"{column}={row[column]}"
                    for column in (
                        "revenue",
                        "net_income",
                        "operating_cash_flow",
                        "gross_margin",
                        "debt_to_asset",
                        "roe",
                    )
                    if column in row.index and not pd.isna(row[column])
                )
                rows.append(
                    {
                        "source_name": "tushare" if statement_name != "akshare" else "akshare",
                        "url": "",
                        "title": f"{statement_name.replace('_', ' ').title()} {row.get('report_period', '')}",
                        "body": body,
                        "published_at": row.get("ann_date"),
                        "available_at": row.get("available_at"),
                        "symbol": symbol,
                        "event_type": "earnings_growth"
                        if statement_name == "income"
                        else "financial_statement",
                        "confidence": 0.85,
                        "theme_candidates": "",
                        "chain_node_candidates": "",
                    }
                )
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        frame = attach_source_profile(frame, registry)
        frame["source_type"] = "financial"
        return frame
