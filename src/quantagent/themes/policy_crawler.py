from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Any


@dataclass(frozen=True)
class PolicyDocument:
    document_id: str
    title: str
    body: str
    source: str
    source_level: str
    published_at: str
    effective_start_date: str | None = None
    effective_end_date: str | None = None
    raw_reference: dict[str, Any] = field(default_factory=dict)


def local_policy_documents(records: Iterable[Mapping[str, Any]]) -> list[PolicyDocument]:
    """Normalize already-ingested policy rows without network access."""
    documents: list[PolicyDocument] = []
    for index, row in enumerate(records):
        documents.append(
            PolicyDocument(
                document_id=str(row.get("document_id") or row.get("id") or f"policy-{index:04d}"),
                title=str(row.get("title", "")),
                body=str(row.get("body", row.get("summary", ""))),
                source=str(row.get("source", "")),
                source_level=str(row.get("source_level", "media_interpretation")),
                published_at=str(row.get("published_at", row.get("timestamp", ""))),
                effective_start_date=_optional_str(row.get("effective_start_date")),
                effective_end_date=_optional_str(row.get("effective_end_date")),
                raw_reference=dict(row.get("raw_reference", {})),
            )
        )
    return documents


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
