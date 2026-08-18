from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModelConfig:
    """A-share default fees (bps unless noted).

    This is the **single source** for both the friction a report declares and
    the friction a simulator applies. `slippage_bps` used to be declared here,
    serialised into every backtest configuration, and quoted by reports, while
    `AShareFillModel` moved the fill price by its own private 2.0 -- so any
    report saying "5 bps slippage" was describing a run that charged 2. The
    fill model now reads this field; there is no second one to diverge from.

    `impact_coefficient` / `impact_exponent` describe the square-root impact
    law `bps = 10_000 * coefficient * participation ** exponent`. At the shipped
    defaults that is `10.0 * sqrt(participation)`, numerically identical to
    `execution.cost_model.AShareCostModel(impact_alpha_bps=10.0)`. The two are
    pinned to each other by
    `tests/backtest/test_fill_cost_single_source.py`.
    """

    commission_bps: float = 2.5
    commission_min_rmb: float = 5.0
    sell_stamp_duty_bps: float = 5.0
    transfer_fee_bps: float = 0.1
    slippage_bps: float = 5.0
    impact_coefficient: float = 0.001
    impact_exponent: float = 0.5

    @property
    def impact_alpha_bps(self) -> float:
        """The square-root law's leading coefficient, in bps.

        Named to match `AShareCostModel.impact_alpha_bps` so the two models can
        be compared field-for-field rather than by re-deriving the algebra at
        each call site.
        """
        return 10_000.0 * float(self.impact_coefficient)


def square_root_impact_bps(
    participation_rate: float,
    config: CostModelConfig | None = None,
) -> float:
    """Scalar square-root market impact, in bps of the traded notional.

    `bps = 10_000 * impact_coefficient * participation ** impact_exponent`.

    This is the same law `estimate_trade_cost_bps` applies vectorised, and the
    same law `execution.cost_model.AShareCostModel._impact_cost` applies to
    venue fills. It exists as a scalar so the fast backtester can charge the
    identical function instead of the linear approximation it used to carry --
    `impact_bps * filled / volume`, which under a 5% participation cap could
    never exceed 0.05 bps and so removed capacity effects from the engine
    entirely.

    A non-positive participation rate returns 0.0: no trade, no impact. That is
    an arithmetic identity of the law, not a fallback for missing data -- a
    caller that does not know the participation rate must not call this.
    """
    config = config or CostModelConfig()
    p = float(participation_rate)
    if p <= 0.0 or config.impact_coefficient <= 0.0:
        return 0.0
    return 10_000.0 * float(config.impact_coefficient) * (p ** float(config.impact_exponent))


def estimate_trade_cost_bps(
    order_value: pd.Series,
    adv: pd.Series,
    side: pd.Series | None = None,
    delta_weight: pd.Series | None = None,
    config: CostModelConfig | None = None,
) -> pd.Series:
    """Per-trade cost in bps. Uses delta_weight sign when side is missing."""
    config = config or CostModelConfig()
    aligned_adv = adv.reindex(order_value.index).replace(0, np.nan)
    participation = (order_value.abs() / aligned_adv).clip(lower=0.0).fillna(0.0)
    impact_bps = 10000.0 * config.impact_coefficient * participation.pow(config.impact_exponent)

    sell_mask = _sell_mask(order_value.index, side, delta_weight)
    stamp_bps = sell_mask.astype(float) * config.sell_stamp_duty_bps
    transfer_bps = pd.Series(config.transfer_fee_bps, index=order_value.index)
    return config.commission_bps + config.slippage_bps + impact_bps + stamp_bps + transfer_bps


def expected_cost_return(
    turnover_weight: pd.Series,
    cost_bps: pd.Series,
) -> pd.Series:
    return turnover_weight.abs() * cost_bps.reindex(turnover_weight.index).fillna(0.0) / 10000.0


def commission_with_floor(
    order_value: pd.Series,
    config: CostModelConfig | None = None,
) -> pd.Series:
    """Absolute commission including the regulatory minimum."""
    config = config or CostModelConfig()
    raw = order_value.abs() * config.commission_bps / 10000.0
    return raw.clip(lower=config.commission_min_rmb)


def _sell_mask(
    index: pd.Index,
    side: pd.Series | None,
    delta_weight: pd.Series | None,
) -> pd.Series:
    if side is not None:
        return side.reindex(index).astype(str).str.lower().eq("sell")
    if delta_weight is not None:
        return delta_weight.reindex(index).fillna(0.0).lt(0.0)
    return pd.Series(False, index=index)
