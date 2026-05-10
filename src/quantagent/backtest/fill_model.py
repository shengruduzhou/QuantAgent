from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FillModelConfig:
    participation_rate: float = 0.05
    volume_cap_ratio: float = 0.10
    slippage_bps: float = 2.0
    impact_bps: float = 1.0
    queue_fill_ratio: float = 1.0


@dataclass(frozen=True)
class FillModelResult:
    filled_quantity: int
    fill_price: float
    fill_ratio: float
    slippage_cost: float
    reject_reason: str | None = None


class AShareFillModel:
    """Deterministic next-bar fill approximation for V4 backtests."""

    def __init__(self, config: FillModelConfig | None = None) -> None:
        self.config = config or FillModelConfig()

    def fill(self, side: str, quantity: int, price: float, volume: float) -> FillModelResult:
        if quantity <= 0:
            return FillModelResult(0, price, 0.0, 0.0, "invalid_lot_quantity")
        if price <= 0:
            return FillModelResult(0, price, 0.0, 0.0, "missing_price")
        if volume <= 0:
            return FillModelResult(0, price, 0.0, 0.0, "zero_volume")
        cap = max(0, int(volume * min(self.config.participation_rate, self.config.volume_cap_ratio) * self.config.queue_fill_ratio))
        filled = min(quantity, cap if cap > 0 else quantity)
        direction = 1.0 if side == "buy" else -1.0
        impact = self.config.impact_bps * (filled / max(volume, 1.0))
        fill_price = price * (1.0 + direction * (self.config.slippage_bps + impact) / 10000.0)
        slippage_cost = abs(fill_price - price) * filled
        return FillModelResult(filled, float(fill_price), filled / max(quantity, 1), float(slippage_cost), None)
