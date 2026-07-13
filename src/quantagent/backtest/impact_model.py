"""Square-root market-impact model (H-028 Track B, preregistered).

impact_return = eta * vol20 * sqrt(filled_value / adv20)

- eta is FROZEN before any candidate result is observed: 1.0 (base) / 2.0
  (stressed) — Almgren/Grinold square-root-law literature magnitude. Never
  tune eta against candidate returns.
- Impact is charged ONLY to filled notional; unfilled quantity carries none.
- This module is analysis-layer and opt-in: it does NOT change the trusted
  evaluator's defaults (default-changing requires explicit user approval —
  INC-E1 precedent).
- Missing inputs fail loud per component: NaN adv/vol produce NaN impact and
  a populated ``reason`` — callers must handle explicitly, nothing is
  silently zeroed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ETA_BASE = 1.0
ETA_STRESSED = 2.0


@dataclass(frozen=True)
class SqrtImpactParams:
    eta: float = ETA_BASE
    participation_cap: float = 0.10   # repository-approved ADV participation limit
    min_adv_cny: float = 1.0          # guards division by ~0 for dead names


def sqrt_impact_return(filled_value: float, adv20: float, vol20: float,
                       params: SqrtImpactParams = SqrtImpactParams()) -> tuple[float, str]:
    """One-way impact in return units for a single filled order.

    Returns (impact, reason). impact is NaN when inputs are unusable; reason
    is '' on success.
    """
    if filled_value is None or not np.isfinite(filled_value) or filled_value < 0:
        return float("nan"), "invalid_filled_value"
    if filled_value == 0.0:
        return 0.0, ""  # zero quantity: no impact, by definition
    if adv20 is None or not np.isfinite(adv20):
        return float("nan"), "missing_adv"
    if vol20 is None or not np.isfinite(vol20) or vol20 < 0:
        return float("nan"), "missing_volatility"
    if adv20 < params.min_adv_cny:
        return float("nan"), "adv_below_floor"
    if filled_value > params.participation_cap * adv20 * 1.0000001:
        # the execution simulator must have capped fills already; a violation
        # here means impact is being asked about an impossible fill
        return float("nan"), "fill_exceeds_participation_cap"
    return float(params.eta * vol20 * np.sqrt(filled_value / adv20)), ""


def apply_impact(fills: pd.DataFrame, params: SqrtImpactParams = SqrtImpactParams(),
                 value_col: str = "filled_value", adv_col: str = "adv20",
                 vol_col: str = "vol20") -> pd.DataFrame:
    """Vectorized per-order impact. Adds ``impact_return``, ``impact_cost_cny``
    and ``impact_reason`` columns; never mutates the input frame."""
    out = fills.copy()
    v = out[value_col].to_numpy(dtype="float64")
    adv = out[adv_col].to_numpy(dtype="float64")
    vol = out[vol_col].to_numpy(dtype="float64")
    impact = np.full(len(out), np.nan)
    reason = np.full(len(out), "", dtype=object)

    bad_v = ~np.isfinite(v) | (v < 0)
    zero = ~bad_v & (v == 0)
    bad_adv = ~bad_v & ~zero & ~np.isfinite(adv)
    bad_vol = ~bad_v & ~zero & ~bad_adv & (~np.isfinite(vol) | (vol < 0))
    low_adv = ~bad_v & ~zero & ~bad_adv & ~bad_vol & (adv < params.min_adv_cny)
    over = (~bad_v & ~zero & ~bad_adv & ~bad_vol & ~low_adv
            & (v > params.participation_cap * adv * 1.0000001))
    ok = ~bad_v & ~zero & ~bad_adv & ~bad_vol & ~low_adv & ~over

    reason[bad_v] = "invalid_filled_value"
    reason[bad_adv] = "missing_adv"
    reason[bad_vol] = "missing_volatility"
    reason[low_adv] = "adv_below_floor"
    reason[over] = "fill_exceeds_participation_cap"
    impact[zero] = 0.0
    impact[ok] = params.eta * vol[ok] * np.sqrt(v[ok] / adv[ok])

    out["impact_return"] = impact
    out["impact_cost_cny"] = impact * v
    out["impact_reason"] = reason
    return out
