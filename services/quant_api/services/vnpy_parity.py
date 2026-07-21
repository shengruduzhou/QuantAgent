from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from services.quant_api.schemas.parity import (
    VnpyParityCapability,
    VnpyParityRegistry,
    VnpyParityStatus,
    VnpyParitySummary,
    VnpyParityView,
)


PARITY_STATUSES: list[VnpyParityStatus] = [
    "not_audited",
    "missing",
    "planned",
    "in_progress",
    "partial",
    "implemented",
    "verified",
    "blocked",
    "not_applicable",
]

_COMPLETION_WEIGHT: dict[str, float] = {
    "not_audited": 0.0,
    "missing": 0.0,
    "planned": 0.15,
    "in_progress": 0.35,
    "partial": 0.5,
    "implemented": 0.75,
    "verified": 1.0,
    "blocked": 0.0,
}


class VnpyParityService:
    """Validated, read-only projection of the canonical vn.py parity registry."""

    def __init__(self, registry_path: Path | None = None) -> None:
        self.registry_path = registry_path or (
            Path(__file__).resolve().parents[1]
            / "resources"
            / "vnpy_capability_parity.v1.json"
        )
        self._registry: VnpyParityRegistry | None = None
        self._mtime_ns: int | None = None

    def load(self, *, force: bool = False) -> VnpyParityRegistry:
        stat = self.registry_path.stat()
        if (
            not force
            and self._registry is not None
            and self._mtime_ns == stat.st_mtime_ns
        ):
            return self._registry

        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        registry = VnpyParityRegistry.model_validate(payload)
        ids = [capability.id for capability in registry.capabilities]
        duplicate_ids = sorted(
            capability_id
            for capability_id, count in Counter(ids).items()
            if count > 1
        )
        if duplicate_ids:
            raise ValueError(
                f"duplicate vn.py parity capability ids: {', '.join(duplicate_ids)}"
            )

        self._registry = registry
        self._mtime_ns = stat.st_mtime_ns
        return registry

    def view(
        self,
        *,
        category: str | None = None,
        status: VnpyParityStatus | None = None,
        query: str | None = None,
        refresh: bool = False,
    ) -> VnpyParityView:
        registry = self.load(force=refresh)
        normalized_query = (query or "").strip().casefold()
        capabilities = [
            capability
            for capability in registry.capabilities
            if self._matches(
                capability,
                category=category,
                status=status,
                query=normalized_query,
            )
        ]
        categories = sorted({item.category for item in registry.capabilities})
        return VnpyParityView(
            schemaVersion=registry.schema_version,
            registryVersion=registry.registry_version,
            title=registry.title,
            generatedAt=registry.generated_at,
            sourceBaseline=registry.source_baseline,
            completeness=registry.completeness,
            verificationPolicy=registry.verification_policy,
            knownCoverageGaps=registry.known_coverage_gaps,
            categories=categories,
            statuses=list(PARITY_STATUSES),
            summary=self._summary(capabilities),
            capabilities=capabilities,
        )

    @staticmethod
    def _matches(
        capability: VnpyParityCapability,
        *,
        category: str | None,
        status: VnpyParityStatus | None,
        query: str,
    ) -> bool:
        if category and capability.category != category:
            return False
        if status and capability.status != status:
            return False
        if not query:
            return True

        haystack = " ".join(
            [
                capability.id,
                capability.name,
                capability.category,
                capability.description,
                capability.gap,
                capability.adoption,
                capability.next_action,
                capability.source.repo,
                capability.source.module,
                *capability.quantagent.modules,
                *capability.quantagent.api,
                *capability.quantagent.events,
                *capability.quantagent.frontend,
            ]
        ).casefold()
        return query in haystack

    @staticmethod
    def _summary(capabilities: list[VnpyParityCapability]) -> VnpyParitySummary:
        by_status = Counter(item.status for item in capabilities)
        by_category = Counter(item.category for item in capabilities)
        applicable = [
            item for item in capabilities if item.status != "not_applicable"
        ]
        weighted = sum(_COMPLETION_WEIGHT.get(item.status, 0.0) for item in applicable)
        completion_ratio = weighted / len(applicable) if applicable else 0.0
        return VnpyParitySummary(
            total=len(capabilities),
            byStatus=dict(sorted(by_status.items())),
            byCategory=dict(sorted(by_category.items())),
            verified=by_status.get("verified", 0),
            actionable=sum(
                count
                for item_status, count in by_status.items()
                if item_status not in {"verified", "not_applicable"}
            ),
            completionRatio=round(completion_ratio, 4),
        )
