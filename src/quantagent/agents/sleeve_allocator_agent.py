from __future__ import annotations

from dataclasses import dataclass

from quantagent.portfolio.allocator import SleeveAllocator
from quantagent.portfolio.sleeve import SleeveAllocationResult


@dataclass(frozen=True)
class SleeveAllocatorAgentOutput:
    allocation: SleeveAllocationResult
    confidence: float
    reasoning: tuple[str, ...]


class SleeveAllocatorAgent:
    def __init__(self, allocator: SleeveAllocator | None = None) -> None:
        self.allocator = allocator or SleeveAllocator()

    def run(
        self,
        total_nav: float,
        market_regime: str,
        drawdown_state: float,
        volatility_state: float,
        short_alpha_quality: float,
        long_alpha_quality: float,
        sector_rotation_strength: float,
        hedge_need: float,
        sector_breadth: float = 0.5,
        flow_confirmation: float = 0.0,
    ) -> SleeveAllocatorAgentOutput:
        allocation = self.allocator.allocate(
            total_nav=total_nav,
            market_regime=market_regime,
            drawdown_state=drawdown_state,
            volatility_state=volatility_state,
            short_alpha_quality=short_alpha_quality,
            long_alpha_quality=long_alpha_quality,
            sector_rotation_strength=sector_rotation_strength,
            hedge_need=hedge_need,
            sector_breadth=sector_breadth,
            flow_confirmation=flow_confirmation,
        )
        confidence = min(target.confidence for target in allocation.targets) if allocation.targets else 0.0
        reasoning = tuple(target.reason for target in allocation.targets)
        return SleeveAllocatorAgentOutput(allocation=allocation, confidence=confidence, reasoning=reasoning)

