from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModelConfig:
    commission_bps: float = 2.5
    sell_tax_bps: float = 5.0
    slippage_bps: float = 5.0
    impact_coefficient: float = 0.001
    impact_exponent: float = 0.5


def estimate_trade_cost_bps(
    order_value: pd.Series,
    adv: pd.Series,
    side: pd.Series | None = None,
    config: CostModelConfig | None = None,
) -> pd.Series:
    config = config or CostModelConfig()
    aligned_adv = adv.reindex(order_value.index).replace(0, np.nan)
    participation = (order_value.abs() / aligned_adv).clip(lower=0.0).fillna(0.0)
    impact_bps = 10000.0 * config.impact_coefficient * participation.pow(config.impact_exponent)
    tax_bps = pd.Series(0.0, index=order_value.index)
    if side is not None:
        tax_bps = side.reindex(order_value.index).str.lower().eq("sell").astype(float) * config.sell_tax_bps
    return config.commission_bps + config.slippage_bps + impact_bps + tax_bps


def expected_cost_return(
    turnover_weight: pd.Series,
    cost_bps: pd.Series,
) -> pd.Series:
    return turnover_weight.abs() * cost_bps.reindex(turnover_weight.index).fillna(0.0) / 10000.0
