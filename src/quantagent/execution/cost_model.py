from __future__ import annotations

from dataclasses import dataclass

from quantagent.execution.broker_base import OrderSide


@dataclass(frozen=True)
class AShareCostModel:
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001

    def calculate(self, side: OrderSide, quantity: int, price: float) -> dict[str, float]:
        value = float(quantity) * float(price)
        commission = max(self.min_commission if value > 0 else 0.0, value * self.commission_rate)
        stamp = value * self.stamp_tax_rate if side == OrderSide.SELL else 0.0
        transfer = value * self.transfer_fee_rate
        return {"commission": commission, "stamp_duty": stamp, "transfer_fee": transfer, "total": commission + stamp + transfer}

