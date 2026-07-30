from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    fundamentals_root: str | None = Field(None, alias="fundamentalsRoot")
    valuation_path: str | None = Field(None, alias="valuationPath")
    disclosures_path: str | None = Field(None, alias="disclosuresPath")
    sector_map_path: str | None = Field(None, alias="sectorMapPath")
    training_dataset_path: str | None = Field(None, alias="trainingDatasetPath")
    synthesized_factors_path: str | None = Field(None, alias="synthesizedFactorsPath")
    output_dir: str = Field(alias="outputDir")
    factor_library: Literal["all_reviewed", "basic", "alpha101", "alpha181", "cicc_ashare80"] = Field("all_reviewed", alias="factorLibrary")
    model: Literal["ridge", "ft_transformer"] = "ridge"
    horizons: str = Field("1,5,20,60,120", pattern=r"^\d+(,\d+)*$")
    primary_horizon: int = Field(5, ge=1, le=252, alias="primaryHorizon")
    split_mode: Literal["rolling", "expanding"] = Field("rolling", alias="splitMode")
    n_splits: int = Field(4, ge=2, le=20, alias="nSplits")
    require_gpu: bool = Field(False, alias="requireGpu")
    top_k: int = Field(30, ge=5, le=500, alias="topK")
    top_k_candidates: list[int] = Field(
        default_factory=lambda: [10, 20, 30, 50, 80],
        alias="topKCandidates",
    )
    stock_selection_modes: list[Literal["none", "fundamental"]] = Field(
        default_factory=lambda: ["none", "fundamental"],
        alias="stockSelectionModes",
    )
    fundamental_selection_mode: Literal["auto", "fixed", "off"] = Field(
        "auto", alias="fundamentalSelectionMode"
    )
    fundamental_selection_threshold: float = Field(
        0.50, ge=0.0, le=1.0, alias="fundamentalSelectionThreshold"
    )
    fundamental_blend_weight: float = Field(
        0.40, ge=0.0, le=1.0, alias="fundamentalBlendWeight"
    )
    fundamental_threshold_candidates: list[float] = Field(
        default_factory=lambda: [0.25, 0.40, 0.55],
        alias="fundamentalThresholdCandidates",
    )
    fundamental_blend_candidates: list[float] = Field(
        default_factory=lambda: [0.20, 0.40, 0.60],
        alias="fundamentalBlendCandidates",
    )
    selection_max_candidates: int = Field(
        64, ge=2, le=128, alias="selectionMaxCandidates"
    )
    selection_min_oos_days: int = Field(
        80, ge=20, le=1_260, alias="selectionMinOosDays"
    )
    selection_min_holdout_days: int = Field(
        20, ge=10, le=504, alias="selectionMinHoldoutDays"
    )
    max_pbo: float = Field(0.25, ge=0.0, le=1.0, alias="maxPbo")
    min_dsr_probability: float = Field(
        0.95, ge=0.0, le=1.0, alias="minDsrProbability"
    )
    max_spa_pvalue: float = Field(
        0.05, ge=0.0, le=1.0, alias="maxSpaPValue"
    )
    factor_screening_mode: Literal["off", "evaluate_only", "pretrain"] = Field(
        "pretrain", alias="factorScreeningMode"
    )
    do_t_mode: Literal["off", "intraday", "daily_swing", "both"] = Field(
        "daily_swing", alias="doTMode"
    )
    minute_panel_path: str | None = Field(None, alias="minutePanelPath")
    max_weight_per_name: float = Field(0.08, gt=0, le=0.25, alias="maxWeightPerName")
    max_sector_weight: float = Field(0.30, gt=0, le=1, alias="maxSectorWeight")
    max_turnover: float = Field(0.50, gt=0, le=2, alias="maxTurnover")
    objective: Literal["max_expected_alpha", "mean_variance", "min_variance"] = "max_expected_alpha"
    weighting: Literal["equal", "rank", "softmax"] = "rank"
    initial_cash: float = Field(1_000_000, ge=100_000, le=10_000_000_000, alias="initialCash")
    benchmark_symbol: str | None = Field("000300.SH", max_length=32, alias="benchmarkSymbol")
    objective_weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights, alias="objectiveWeights")
    risk_limits: RiskLimits = Field(default_factory=RiskLimits, alias="riskLimits")
    human_approved: bool = Field(False, alias="humanApproved")

    @field_validator("top_k_candidates", mode="before")
    @classmethod
    def parse_top_k_candidates(cls, value: object) -> object:
        if isinstance(value, str):
            return [int(item) for item in value.split(",") if item]
        return value

    @field_validator(
        "fundamental_threshold_candidates",
        "fundamental_blend_candidates",
        mode="before",
    )
    @classmethod
    def parse_float_candidates(cls, value: object) -> object:
        if isinstance(value, str):
            return [float(item) for item in value.split(",") if item]
        return value

    @field_validator("stock_selection_modes", mode="before")
    @classmethod
    def parse_stock_selection_modes(cls, value: object) -> object:
        if isinstance(value, str):
            return [item for item in value.split(",") if item]
        return value

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "StrategyDraft":
        horizons = {int(value) for value in self.horizons.split(",")}
        if self.primary_horizon not in horizons:
            raise ValueError("primaryHorizon must be included in horizons")
        if self.model == "ft_transformer" and not self.require_gpu:
            raise ValueError("FT-Transformer strategy training requires GPU")
        if self.max_turnover > self.risk_limits.max_turnover:
            raise ValueError("portfolio maxTurnover cannot exceed the risk limit")
        if any(value < 5 or value > 500 for value in self.top_k_candidates):
            raise ValueError("topKCandidates must remain between 5 and 500")
        if not self.top_k_candidates:
            raise ValueError("topKCandidates must include at least one candidate")
        if len(set(self.top_k_candidates)) > 12:
            raise ValueError("topKCandidates is bounded to 12 unique values")
        if not self.stock_selection_modes:
            raise ValueError("stockSelectionModes must include at least one mode")
        if len(set(self.stock_selection_modes)) != len(self.stock_selection_modes):
            raise ValueError("stockSelectionModes cannot contain duplicates")
        if self.fundamental_selection_mode == "off" and "fundamental" in self.stock_selection_modes:
            raise ValueError(
                "stockSelectionModes cannot include fundamental when fundamentalSelectionMode is off"
            )
        if self.fundamental_selection_mode == "auto":
            if not self.fundamental_threshold_candidates:
                raise ValueError("fundamentalThresholdCandidates cannot be empty in auto mode")
            if not self.fundamental_blend_candidates:
                raise ValueError("fundamentalBlendCandidates cannot be empty in auto mode")
        if any(
            value <= 0.0 or value >= 1.0
            for value in self.fundamental_threshold_candidates
        ):
            raise ValueError(
                "fundamentalThresholdCandidates must remain strictly between 0 and 1"
            )
        if any(
            value <= 0.0 or value > 1.0
            for value in self.fundamental_blend_candidates
        ):
            raise ValueError(
                "fundamentalBlendCandidates must remain in (0, 1]"
            )
        if len(set(self.fundamental_threshold_candidates)) > 8:
            raise ValueError("fundamentalThresholdCandidates is bounded to 8 unique values")
        if len(set(self.fundamental_blend_candidates)) > 8:
            raise ValueError("fundamentalBlendCandidates is bounded to 8 unique values")
        fundamental_variants = 0
        if "fundamental" in self.stock_selection_modes:
            fundamental_variants = (
                len(set(self.fundamental_threshold_candidates))
                * len(set(self.fundamental_blend_candidates))
                if self.fundamental_selection_mode == "auto"
                else 1
            )
        baseline_variants = 1 if "none" in self.stock_selection_modes else 0
        total_candidates = (
            len(set(self.top_k_candidates))
            * (fundamental_variants + baseline_variants)
        )
        if total_candidates > self.selection_max_candidates:
            raise ValueError(
                "automatic portfolio search exceeds selectionMaxCandidates: "
                f"{total_candidates} > {self.selection_max_candidates}"
            )
        if self.selection_min_holdout_days >= self.selection_min_oos_days:
            raise ValueError(
                "selectionMinHoldoutDays must be smaller than selectionMinOosDays"
            )
        if self.do_t_mode in {"intraday", "both"} and not self.minute_panel_path:
            raise ValueError("minutePanelPath is required for intraday Do-T modes")
        return self


class ConnectionRequest(StrategyModel):
    credentials: dict[str, str]
