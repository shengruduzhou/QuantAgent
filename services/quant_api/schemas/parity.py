from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


VnpyParityStatus = Literal[
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


class ParityModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class VnpySourceReference(ParityModel):
    repo: str
    module: str
    version: str
    commit: str | None = None


class QuantAgentCapabilityMapping(ParityModel):
    modules: list[str] = Field(default_factory=list)
    api: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    frontend: list[str] = Field(default_factory=list)


class VnpyParityCapability(ParityModel):
    id: str
    category: str
    name: str
    status: VnpyParityStatus
    source: VnpySourceReference
    description: str
    quantagent: QuantAgentCapabilityMapping
    gap: str
    adoption: str
    tests: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_action: str = Field(alias="nextAction")


class VnpySourceBaseline(ParityModel):
    repo: str
    release: str
    commit: str
    release_date: str = Field(alias="releaseDate")
    notes: list[str] = Field(default_factory=list)


class VnpyVerificationPolicy(ParityModel):
    verified_requires: list[str] = Field(default_factory=list, alias="verifiedRequires")


class VnpyParityRegistry(ParityModel):
    schema_version: Literal["quantagent.vnpy-parity.v1"] = Field(alias="schemaVersion")
    registry_version: str = Field(alias="registryVersion")
    title: str
    generated_at: str = Field(alias="generatedAt")
    source_baseline: VnpySourceBaseline = Field(alias="sourceBaseline")
    completeness: Literal["partial", "complete"]
    verification_policy: VnpyVerificationPolicy = Field(alias="verificationPolicy")
    known_coverage_gaps: list[str] = Field(default_factory=list, alias="knownCoverageGaps")
    capabilities: list[VnpyParityCapability]


class VnpyParitySummary(ParityModel):
    total: int
    by_status: dict[str, int] = Field(alias="byStatus")
    by_category: dict[str, int] = Field(alias="byCategory")
    verified: int
    actionable: int
    completion_ratio: float = Field(alias="completionRatio")


class VnpyParityView(ParityModel):
    schema_version: str = Field(alias="schemaVersion")
    registry_version: str = Field(alias="registryVersion")
    title: str
    generated_at: str = Field(alias="generatedAt")
    source_baseline: VnpySourceBaseline = Field(alias="sourceBaseline")
    completeness: str
    verification_policy: VnpyVerificationPolicy = Field(alias="verificationPolicy")
    known_coverage_gaps: list[str] = Field(alias="knownCoverageGaps")
    categories: list[str]
    statuses: list[str]
    summary: VnpyParitySummary
    capabilities: list[VnpyParityCapability]
