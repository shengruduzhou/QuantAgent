from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from quantagent.execution.broker_base import OrderSide


@dataclass(frozen=True)
class AShareCostModel:
    """Full A-share execution cost model.

    Components covered:

    * **commission**: bps of order value, floored at ``min_commission``
      (5 yuan is the typical retail floor).
    * **stamp tax**: applied to SELL only — current statutory rate
      is 0.05% (lowered from 0.10% in August 2023). Default 0.0005.
    * **transfer fee**: Shanghai-side micro fee, 0.001%.
    * **impact cost**: square-root market impact model. The
      executed price drifts away from arrival mid by
      ``impact_alpha * sqrt(participation_rate)`` (in bps), where
      participation is the order's share of the day's traded volume.
      Returned as a value (yuan) so callers can aggregate with the
      explicit fees. Set ``impact_alpha`` to 0 to disable impact
      modelling entirely.
    """

    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    # Square-root impact coefficient in bps. 10 bps × sqrt(participation)
    # matches a typical mid-cap A-share line; small/illiquid names need
    # higher alpha (callers can override).
    impact_alpha_bps: float = 10.0

    def calculate(
        self,
        side: OrderSide,
        quantity: int,
        price: float,
        *,
        participation_rate: float = 0.0,
    ) -> dict[str, float]:
        value = float(quantity) * float(price)
        commission = max(self.min_commission if value > 0 else 0.0, value * self.commission_rate)
        stamp = value * self.stamp_tax_rate if side == OrderSide.SELL else 0.0
        transfer = value * self.transfer_fee_rate
        impact = self._impact_cost(value, participation_rate)
        return {
            "commission": commission,
            "stamp_duty": stamp,
            "transfer_fee": transfer,
            "impact_cost": impact,
            "total": commission + stamp + transfer + impact,
        }

    def _impact_cost(self, order_value: float, participation_rate: float) -> float:
        if order_value <= 0 or self.impact_alpha_bps <= 0:
            return 0.0
        p = max(0.0, float(participation_rate))
        if p <= 0:
            return 0.0
        bps = self.impact_alpha_bps * sqrt(p)
        return order_value * bps / 10_000.0

