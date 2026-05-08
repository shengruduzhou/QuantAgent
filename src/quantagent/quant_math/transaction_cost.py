from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModelConfig:
    """A-share default fees (bps unless noted)."""

    commission_bps: float = 2.5
    commission_min_rmb: float = 5.0
    sell_stamp_duty_bps: float = 5.0
    transfer_fee_bps: float = 0.1
    slippage_bps: float = 5.0
    impact_coefficient: float = 0.001
    impact_exponent: float = 0.5


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
