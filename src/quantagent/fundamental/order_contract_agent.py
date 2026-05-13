from __future__ import annotations

from hashlib import sha1

import pandas as pd

from quantagent.v7.schemas import EvidenceRecord, EventType, SourceType


def order_contract_evidence(announcements: pd.DataFrame, as_of_date: str) -> list[EvidenceRecord]:
    """Convert order and capacity announcements into structured V7 evidence."""
    if announcements.empty:
        return []
    records: list[EvidenceRecord] = []
    for index, row in announcements.iterrows():
        amount_score = _amount_score(row.get("contract_amount"), row.get("revenue"))
        official = bool(row.get("is_exchange_disclosure", True))
        reliability = 0.85 if official else 0.55
        confidence = min(0.95, 0.45 + 0.30 * official + 0.20 * amount_score)
        raw = row.to_dict()
        evidence_id = str(row.get("announcement_id", f"order-{index:04d}"))
        records.append(
            EvidenceRecord(
                evidence_id=evidence_id,
                source=str(row.get("source", "company_announcement")),
                source_type=SourceType.COMPANY_ANNOUNCEMENT if official else SourceType.NEWS,
                source_authority_level=reliability,
                timestamp=as_of_date,
                published_at=str(row.get("published_at", as_of_date)),
                symbol=str(row["symbol"]),
                theme=str(row.get("theme")) if row.get("theme") is not None else None,
                chain_node=str(row.get("chain_node")) if row.get("chain_node") is not None else None,
                event_type=EventType.ORDER_CONFIRMED,
                direction=1.0,
                magnitude=amount_score,
                confidence=confidence,
                evidence_quality=reliability,
                source_reliability=reliability,
                cross_validation_count=int(row.get("cross_validation_count", 1 if official else 0)),
                decay_half_life=float(row.get("decay_half_life", 30.0)),
                horizon_days=int(row.get("horizon_days", 60)),
                rationale=str(row.get("title", "order or capacity disclosure"))[:240],
                raw_reference={"hash": sha1(str(raw).encode("utf-8")).hexdigest(), "announcement_id": evidence_id},
                point_in_time_valid=str(row.get("published_at", as_of_date)) <= as_of_date,
                risk_flags=tuple(str(row.get("risk_flag")).split(",")) if row.get("risk_flag") else (),
            ).with_hash()
        )
    return records


def _amount_score(contract_amount: object, revenue: object) -> float:
    if contract_amount is None or revenue is None or pd.isna(contract_amount) or pd.isna(revenue) or float(revenue) <= 0:
        return 0.45
    return float(min(1.0, max(0.05, float(contract_amount) / float(revenue))))
