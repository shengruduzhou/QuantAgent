from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrategyModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ObjectiveWeights(StrategyModel):
    excess_return: float = Field(0.45, ge=0, le=1, alias="excessReturn")
    annual_return: float = Field(0.30, ge=0, le=1, alias="annualReturn")
    drawdown_control: float = Field(0.25, ge=0, le=1, alias="drawdownControl")

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "ObjectiveWeights":
        total = self.excess_return + self.annual_return + self.drawdown_control
        if abs(total - 1.0) > 0.001:
            raise ValueError("objective weights must sum to 1")
        return self


class RiskLimits(StrategyModel):
    max_drawdown: float = Field(0.15, gt=0, le=0.80, alias="maxDrawdown")
    max_turnover: float = Field(0.50, gt=0, le=2.0, alias="maxTurnover")
    min_sharpe: float = Field(1.0, ge=-5, le=10, alias="minSharpe")


class StrategyDraft(StrategyModel):
    id: str | None = Field(None, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{2,63}$")
    name: str = Field(min_length=3, max_length=120)
    hypothesis: str = Field(min_length=10, max_length=2_000)
    invalidation_criteria: str = Field(min_length=10, max_length=2_000, alias="invalidationCriteria")
    market_panel_path: str = Field(alias="marketPanelPath")
    labels_path: str = Field(alias="labelsPath")
    sector_map_path: str | None = Field(None, alias="sectorMapPath")
    training_dataset_path: str | None = Field(None, alias="trainingDatasetPath")
    synthesized_factors_path: str | None = Field(None, alias="synthesizedFactorsPath")
    output_dir: str = Field(alias="outputDir")
    factor_library: Literal["basic", "alpha101", "alpha181", "cicc_ashare80"] = Field("alpha181", alias="factorLibrary")
    model: Literal["ridge", "ft_transformer"] = "ridge"
    horizons: str = Field("1,5,20,60,120", pattern=r"^\d+(,\d+)*$")
    primary_horizon: int = Field(5, ge=1, le=252, alias="primaryHorizon")
    split_mode: Literal["rolling", "expanding"] = Field("rolling", alias="splitMode")
    n_splits: int = Field(4, ge=2, le=20, alias="nSplits")
    require_gpu: bool = Field(False, alias="requireGpu")
    top_k: int = Field(30, ge=5, le=500, alias="topK")
    max_weight_per_name: float = Field(0.08, gt=0, le=0.25, alias="maxWeightPerName")
    max_sector_weight: float = Field(0.30, gt=0, le=1, alias="maxSectorWeight")
    max_turnover: float = Field(0.50, gt=0, le=2, alias="maxTurnover")
    objective: Literal["max_expected_alpha", "mean_variance", "min_variance"] = "max_expected_alpha"
    weighting: Literal["equal", "rank", "softmax"] = "rank"
    initial_cash: float = Field(1_000_000, ge=100_000, le=10_000_000_000, alias="initialCash")
    benchmark_symbol: str | None = Field(None, max_length=32, alias="benchmarkSymbol")
    objective_weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights, alias="objectiveWeights")
    risk_limits: RiskLimits = Field(default_factory=RiskLimits, alias="riskLimits")
    human_approved: bool = Field(False, alias="humanApproved")

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "StrategyDraft":
        horizons = {int(value) for value in self.horizons.split(",")}
        if self.primary_horizon not in horizons:
            raise ValueError("primaryHorizon must be included in horizons")
        if self.max_turnover > self.risk_limits.max_turnover:
            raise ValueError("portfolio maxTurnover cannot exceed the risk limit")
        return self


class ConnectionRequest(StrategyModel):
    credentials: dict[str, str]
