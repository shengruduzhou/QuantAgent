from __future__ import annotations

from dataclasses import dataclass

from quantagent.execution.broker_base import Order, OrderSide


@dataclass(frozen=True)
class FillDecision:
    quantity: int
    price: float
    message: str = "filled"


@dataclass(frozen=True)
class FillSimulator:
    participation_rate: float = 1.0
    slippage_bps: float = 5.0

    def simulate(self, order: Order, available_volume: float | None = None) -> FillDecision:
        quantity = int(order.quantity)
        if available_volume is not None:
            quantity = min(quantity, int(max(0, available_volume) * self.participation_rate))
        price = float(order.price or 0.0)
        slip = self.slippage_bps / 10_000.0
        if order.side == OrderSide.BUY:
            price *= 1.0 + slip
        else:
            price *= 1.0 - slip
        if quantity <= 0:
            return FillDecision(0, price, "no_liquidity")
        return FillDecision(quantity, price, "filled" if quantity == order.quantity else "partial")

