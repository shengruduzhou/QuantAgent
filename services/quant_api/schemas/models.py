from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


DataStatus = Literal["ready", "partial", "empty", "error", "stale", "unavailable"]
ArtifactTrustClass = Literal[
    "production_ready", "paper_only", "research_only", "contaminated", "unclassified",
]
ArtifactValidationStatus = Literal["verified", "declared", "unverified", "invalid"]
ArtifactFreshnessStatus = Literal["current", "stale", "unknown"]
T = TypeVar("T")


class ApiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class DataIssue(ApiModel):
    code: str
    message: str
    path: str | None = None
    recoverable: bool = True


class Provenance(ApiModel):
    source_path: str | None = Field(None, alias="sourcePath")
    source_type: str | None = Field(None, alias="sourceType")
    parser: str | None = None
    derived_fields: list[str] = Field(default_factory=list, alias="derivedFields")
    generated_at: str | None = Field(None, alias="generatedAt")
    indexed_at: str | None = Field(None, alias="indexedAt")


class ApiResponse(ApiModel, Generic[T]):
    status: DataStatus
    data: T
    issues: list[DataIssue] = Field(default_factory=list)
    provenance: Provenance | None = None


class Page(ApiModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int = Field(alias="pageSize")
    has_next: bool = Field(alias="hasNext")


class RuntimeArtifact(ApiModel):
    id: str
    kind: str
    name: str
    path: str
    extension: str
    size_bytes: int = Field(alias="sizeBytes")
    modified_at: str = Field(alias="modifiedAt")
    status: DataStatus = "ready"
    parser: str | None = None
    run_id: str | None = Field(None, alias="runId")
    horizon: str | None = None
    rows: int | None = None
    date_start: str | None = Field(None, alias="dateStart")
    date_end: str | None = Field(None, alias="dateEnd")
    tags: list[str] = Field(default_factory=list)
    schema_version: str | None = Field(None, alias="schemaVersion")
    trust_class: ArtifactTrustClass = Field("unclassified", alias="trustClass")
    validation_status: ArtifactValidationStatus = Field("unverified", alias="validationStatus")
    freshness_status: ArtifactFreshnessStatus = Field("unknown", alias="freshnessStatus")
    stale_reason: str | None = Field(None, alias="staleReason")
    source_time: str | None = Field(None, alias="sourceTime")
    manifest_path: str | None = Field(None, alias="manifestPath")
    content_hash: str | None = Field(None, alias="contentHash")
    capabilities: list[str] = Field(default_factory=list)
    issues: list[DataIssue] = Field(default_factory=list)


class KlineBar(ApiModel):
    datetime: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    amount: float | None = None
    available_at: str | None = Field(None, alias="availableAt")
    is_st: bool | None = Field(None, alias="isSt")
    is_suspended: bool | None = Field(None, alias="isSuspended")
    is_limit_up: bool | None = Field(None, alias="isLimitUp")
    is_limit_down: bool | None = Field(None, alias="isLimitDown")
    source: str | None = None


class Trade(ApiModel):
    id: str
    datetime: str
    symbol: str
    name: str | None = None
    action: str
    price: float
    quantity: float
    amount: float | None = None
    fee: float | None = None
    commission: float | None = None
    slippage: float | None = None
    tax: float | None = None
    transfer_fee: float | None = Field(None, alias="transferFee")
    impact_cost: float | None = Field(None, alias="impactCost")
    position_after: float | None = Field(None, alias="positionAfter")
    position_weight_after: float | None = Field(None, alias="positionWeightAfter")
    cash_after: float | None = Field(None, alias="cashAfter")
    signal_source: str | None = Field(None, alias="signalSource")
    signal_id: str | None = Field(None, alias="signalId")
    model_version: str | None = Field(None, alias="modelVersion")
    model_score: float | None = Field(None, alias="modelScore")
    factor_contributions: dict[str, float] | None = Field(None, alias="factorContributions")
    risk_reason: str | None = Field(None, alias="riskReason")
    pnl: float | None = None
    cumulative_pnl: float | None = Field(None, alias="cumulativePnl")
    success: bool | None = None
    failure_reason: str | None = Field(None, alias="failureReason")
    status: str | None = None
    t_pair_id: str | None = Field(None, alias="tPairId")
    provenance: dict[str, Any] = Field(default_factory=dict)


class Signal(ApiModel):
    id: str
    datetime: str
    symbol: str
    type: str
    price: float | None = None
    strength: float | None = None
    confidence: float | None = None
    source: str | None = None
    factors: dict[str, float] | None = None
    reason: str | None = None
    risk_flags: list[str] = Field(default_factory=list, alias="riskFlags")
    action_raw: str | None = Field(None, alias="actionRaw")
    t_pair_id: str | None = Field(None, alias="tPairId")


class BacktestSummary(ApiModel):
    id: str
    name: str | None = None
    strategy_version: str | None = Field(None, alias="strategyVersion")
    model_version: str | None = Field(None, alias="modelVersion")
    factor_version: str | None = Field(None, alias="factorVersion")
    horizon: str | None = None
    start_date: str | None = Field(None, alias="startDate")
    end_date: str | None = Field(None, alias="endDate")
    universe_size: int | None = Field(None, alias="universeSize")
    initial_cash: float | None = Field(None, alias="initialCash")
    total_return: float | None = Field(None, alias="totalReturn")
    annual_return: float | None = Field(None, alias="annualReturn")
    max_drawdown: float | None = Field(None, alias="maxDrawdown")
    sharpe: float | None = None
    calmar: float | None = None
    volatility: float | None = None
    win_rate: float | None = Field(None, alias="winRate")
    profit_factor: float | None = Field(None, alias="profitFactor")
    turnover: float | None = None
    trade_count: int | None = Field(None, alias="tradeCount")
    fill_count: int | None = Field(None, alias="fillCount")
    t_trade_count: int | None = Field(None, alias="tTradeCount")
    t_contribution: float | None = Field(None, alias="tContribution")
    total_cost: float | None = Field(None, alias="totalCost")
    status: DataStatus = "ready"
    path: str
    tags: list[str] = Field(default_factory=list)
    trust_class: ArtifactTrustClass = Field("unclassified", alias="trustClass")
    validation_status: ArtifactValidationStatus = Field("unverified", alias="validationStatus")
    manifest_path: str | None = Field(None, alias="manifestPath")
    capabilities: dict[str, bool | str | None] = Field(default_factory=dict)


class Factor(ApiModel):
    name: str
    display_name: str | None = Field(None, alias="displayName")
    category: str | None = None
    description: str | None = None
    code_location: str | None = Field(None, alias="codeLocation")
    formula: str | None = None
    direction: str = "UNKNOWN"
    horizon_days: int | None = Field(None, alias="horizonDays")
    parameters: dict[str, Any] = Field(default_factory=dict)
    data_source: list[str] = Field(default_factory=list, alias="dataSource")
    required_columns: list[str] = Field(default_factory=list, alias="requiredColumns")
    frequency: str | None = None
    lookback: int | None = None
    pit_safe: bool | None = Field(None, alias="pitSafe")
    missing_value_policy: str | None = Field(None, alias="missingValuePolicy")
    standardization: str | None = None
    neutralization: str | None = None
    used_in_training: bool | None = Field(None, alias="usedInTraining")
    used_in_selection: bool | None = Field(None, alias="usedInSelection")
    used_in_timing: bool | None = Field(None, alias="usedInTiming")
    used_in_risk: bool | None = Field(None, alias="usedInRisk")
    lifecycle: str | None = None
    source_kind: str = Field(alias="sourceKind")


class ModelSummary(ApiModel):
    id: str
    model_type: str | None = Field(None, alias="modelType")
    version: str | None = None
    feature_version: str | None = Field(None, alias="featureVersion")
    created_at: str | None = Field(None, alias="createdAt")
    train_start: str | None = Field(None, alias="trainStart")
    train_end: str | None = Field(None, alias="trainEnd")
    test_end: str | None = Field(None, alias="testEnd")
    horizons: list[int] = Field(default_factory=list)
    feature_count: int | None = Field(None, alias="featureCount")
    sample_count: int | None = Field(None, alias="sampleCount")
    device: str | None = None
    gpu_name: str | None = Field(None, alias="gpuName")
    production_ready: bool | None = Field(None, alias="productionReady")
    status: DataStatus = "ready"
    path: str
    issues: list[DataIssue] = Field(default_factory=list)
    model_family: str | None = Field(None, alias="modelFamily")
    source_kind: str | None = Field(None, alias="sourceKind")
    verdict: str | None = None
    capabilities: dict[str, bool] = Field(default_factory=dict)


class JobRequest(ApiModel):
    command_id: str = Field(alias="commandId")
    parameters: dict[str, str | int | float | bool | list[str] | None] = Field(default_factory=dict)


class CleanupRequest(ApiModel):
    candidate_ids: list[str] = Field(default_factory=list, alias="candidateIds")
    confirmation: str


class Job(ApiModel):
    id: str
    type: str
    status: str
    command_id: str = Field(alias="commandId")
    created_at: str = Field(alias="createdAt")
    started_at: str | None = Field(None, alias="startedAt")
    finished_at: str | None = Field(None, alias="finishedAt")
    progress: float | None = None
    message: str | None = None
    output_paths: list[str] = Field(default_factory=list, alias="outputPaths")
    error: str | None = None


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
