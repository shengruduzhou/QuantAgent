"""Feature-product contracts for training and production inference.

The legacy v11 integration intentionally skipped missing products.  That is
acceptable for exploratory research but unsafe for a declared production model:
a run must not silently train or score with a materially different feature
surface.  This module wraps the existing attachment pipeline and validates its
attach log against an explicit contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

import pandas as pd

from quantagent.training.v11_integration import (
    AttachLogEntry,
    V11IntegrationConfig,
    V11IntegrationResult,
    attach_v11_features,
)


class Requirement(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    FORBIDDEN = "forbidden"


@dataclass(frozen=True)
class FeatureProductSpec:
    name: str
    requirement: Requirement = Requirement.OPTIONAL
    min_added_columns: int = 0
    accepted_noop_reasons: tuple[str, ...] = ("ok_panel_already_has_column",)


@dataclass(frozen=True)
class FeatureContract:
    name: str
    products: tuple[FeatureProductSpec, ...]
    reject_unlisted_products: bool = False

    def by_name(self) -> dict[str, FeatureProductSpec]:
        return {spec.name: spec for spec in self.products}


@dataclass
class FeatureContractReport:
    contract_name: str
    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attached_products: list[str] = field(default_factory=list)
    skipped_optional_products: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_name": self.contract_name,
            "passed": self.passed,
            "violations": list(self.violations),
            "warnings": list(self.warnings),
            "attached_products": list(self.attached_products),
            "skipped_optional_products": list(self.skipped_optional_products),
        }


class FeatureContractError(RuntimeError):
    def __init__(self, report: FeatureContractReport):
        self.report = report
        super().__init__("feature contract failed: " + "; ".join(report.violations))


RESEARCH_CONTRACT = FeatureContract(
    name="research_flexible_v1",
    products=tuple(
        FeatureProductSpec(name=name, requirement=Requirement.OPTIONAL)
        for name in (
            "sector_map",
            "st_flags",
            "sector_pool",
            "fundamental_ranker",
            "policy_events",
            "bond_flows",
            "state_team_inference",
            "broker_reports",
        )
    ),
)

PRODUCTION_CONTRACT = FeatureContract(
    name="production_fail_closed_v1",
    products=(
        FeatureProductSpec("sector_map", Requirement.REQUIRED),
        FeatureProductSpec("st_flags", Requirement.REQUIRED),
        FeatureProductSpec("sector_pool", Requirement.OPTIONAL),
        FeatureProductSpec("fundamental_ranker", Requirement.OPTIONAL),
        FeatureProductSpec("policy_events", Requirement.OPTIONAL),
        FeatureProductSpec("bond_flows", Requirement.OPTIONAL),
        FeatureProductSpec("state_team_inference", Requirement.OPTIONAL),
        FeatureProductSpec("broker_reports", Requirement.OPTIONAL),
    ),
)


def validate_attach_log(
    attach_log: Iterable[AttachLogEntry],
    contract: FeatureContract,
) -> FeatureContractReport:
    specs = contract.by_name()
    entries = {entry.product: entry for entry in attach_log}
    report = FeatureContractReport(contract_name=contract.name, passed=True)

    for product, spec in specs.items():
        entry = entries.get(product)
        if entry is None:
            if spec.requirement == Requirement.REQUIRED:
                report.violations.append(f"required product not attempted: {product}")
            continue
        if spec.requirement == Requirement.FORBIDDEN:
            if entry.attached:
                report.violations.append(f"forbidden product attached: {product}")
            continue
        if entry.attached:
            report.attached_products.append(product)
            if (
                len(entry.columns_added) < spec.min_added_columns
                and entry.reason not in spec.accepted_noop_reasons
            ):
                report.violations.append(
                    f"product {product} added {len(entry.columns_added)} columns; "
                    f"minimum is {spec.min_added_columns}"
                )
        elif spec.requirement == Requirement.REQUIRED:
            report.violations.append(
                f"required product {product} was not attached: {entry.reason}"
            )
        else:
            report.skipped_optional_products.append(product)
            report.warnings.append(f"optional product {product} skipped: {entry.reason}")

    if contract.reject_unlisted_products:
        for product, entry in entries.items():
            if product not in specs and entry.attached:
                report.violations.append(f"unlisted product attached: {product}")

    report.passed = not report.violations
    return report


def attach_v11_features_with_contract(
    panel: pd.DataFrame,
    *,
    integration_config: V11IntegrationConfig | None = None,
    contract: FeatureContract = PRODUCTION_CONTRACT,
    fail_closed: bool = True,
) -> tuple[V11IntegrationResult, FeatureContractReport]:
    result = attach_v11_features(panel, config=integration_config)
    report = validate_attach_log(result.attach_log, contract)
    if fail_closed and not report.passed:
        raise FeatureContractError(report)
    return result, report


def contract_from_mapping(
    name: str,
    mapping: Mapping[str, str],
    *,
    reject_unlisted_products: bool = False,
) -> FeatureContract:
    specs: list[FeatureProductSpec] = []
    for product, requirement in mapping.items():
        try:
            resolved = Requirement(str(requirement).lower())
        except ValueError as exc:
            raise ValueError(
                f"invalid requirement {requirement!r} for product {product!r}"
            ) from exc
        specs.append(FeatureProductSpec(name=str(product), requirement=resolved))
    return FeatureContract(
        name=name,
        products=tuple(specs),
        reject_unlisted_products=reject_unlisted_products,
    )
