from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SleeveType(str, Enum):
    LONG_FUNDAMENTAL = "long_fundamental"
    SHORT_EVENT = "short_event"
    SECTOR_ROTATION = "sector_rotation"
    HEDGE = "hedge"
    CASH_BUFFER = "cash_buffer"


@dataclass(frozen=True)
class SleeveConfig:
    sleeve_type: SleeveType
    min_weight: float
    max_weight: float
    default_weight: float


@dataclass(frozen=True)
class SleeveState:
    sleeve_type: SleeveType
    current_weight: float
    realized_icir: float = 0.0
    turnover: float = 0.0
    capacity_rmb: float | None = None


@dataclass(frozen=True)
class SleeveTarget:
    sleeve_type: SleeveType
    target_weight: float
    confidence: float
    reason: str


@dataclass(frozen=True)
class SleeveAllocationResult:
    targets: tuple[SleeveTarget, ...]
    total_nav: float
    cash_weight: float
    diagnostics: dict[str, float]

    def as_dict(self) -> dict[str, float]:
        return {target.sleeve_type.value: target.target_weight for target in self.targets}


DEFAULT_SLEEVE_CONFIGS: tuple[SleeveConfig, ...] = (
    SleeveConfig(SleeveType.LONG_FUNDAMENTAL, 0.10, 0.65, 0.40),
    SleeveConfig(SleeveType.SHORT_EVENT, 0.00, 0.20, 0.08),
    SleeveConfig(SleeveType.SECTOR_ROTATION, 0.00, 0.25, 0.12),
    SleeveConfig(SleeveType.HEDGE, 0.00, 0.25, 0.05),
    SleeveConfig(SleeveType.CASH_BUFFER, 0.05, 0.50, 0.35),
)

