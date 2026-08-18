from __future__ import annotations

from dataclasses import dataclass

from quantagent.quant_math.transaction_cost import CostModelConfig, square_root_impact_bps


@dataclass(frozen=True)
class FillModelConfig:
    """Quantity-side fill constraints.

    Deliberately carries **no** price-friction fields. Slippage and market
    impact both come from `CostModelConfig`, which is the config a backtest
    serialises and a report quotes. Two copies of "how much does trading cost"
    is how the engine ended up applying 2.0 bps of slippage while every
    configuration on disk declared 5.0, and applying a linear impact term while
    the trust certificate published a square-root one.
    """

    participation_rate: float = 0.05
    volume_cap_ratio: float = 0.10
    queue_fill_ratio: float = 1.0


@dataclass(frozen=True)
class FillModelResult:
    filled_quantity: int
    fill_price: float
    fill_ratio: float
    slippage_cost: float
    reject_reason: str | None = None
    #: Realised share of the session's volume. Published so a caller can audit
    #: the impact charge instead of taking the fill price on trust.
    participation_rate: float = 0.0
    #: Price friction actually applied, split so the two halves stay legible.
    slippage_bps: float = 0.0
    impact_bps: float = 0.0


class AShareFillModel:
    """Deterministic next-bar fill approximation for V4 backtests.

    Price friction has two parts and one source:

    * **slippage** — `CostModelConfig.slippage_bps`, a flat spread crossing.
    * **impact** — `square_root_impact_bps(filled / volume)`, the same
      square-root law `estimate_trade_cost_bps` and `AShareCostModel` use.

    Both move the *price*; neither is charged again as a fee. `EventDrivenBacktester`
    computes commission, transfer and stamp duty on the moved price and does not
    re-apply `cost.slippage_bps` — an earlier version did, which billed the same
    friction twice.
    """

    def __init__(
        self,
        config: FillModelConfig | None = None,
        cost: CostModelConfig | None = None,
    ) -> None:
        self.config = config or FillModelConfig()
        self.cost = cost or CostModelConfig()

    def fill(self, side: str, quantity: int, price: float, volume: float) -> FillModelResult:
        if quantity <= 0:
            return FillModelResult(0, price, 0.0, 0.0, "invalid_lot_quantity")
        if price <= 0:
            return FillModelResult(0, price, 0.0, 0.0, "missing_price")
        if volume <= 0:
            return FillModelResult(0, price, 0.0, 0.0, "zero_volume")
        cap = max(
            0,
            int(
                volume
                * min(self.config.participation_rate, self.config.volume_cap_ratio)
                * self.config.queue_fill_ratio
            ),
        )
        filled = min(quantity, cap if cap > 0 else quantity)
        direction = 1.0 if side == "buy" else -1.0
        participation = filled / volume
        impact_bps = square_root_impact_bps(participation, self.cost)
        slippage_bps = float(self.cost.slippage_bps)
        fill_price = price * (1.0 + direction * (slippage_bps + impact_bps) / 10000.0)
        slippage_cost = abs(fill_price - price) * filled
        return FillModelResult(
            filled,
            float(fill_price),
            filled / max(quantity, 1),
            float(slippage_cost),
            None,
            participation_rate=float(participation),
            slippage_bps=slippage_bps,
            impact_bps=float(impact_bps),
        )
