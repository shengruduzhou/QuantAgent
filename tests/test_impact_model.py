"""H-028 Track B: unit tests for the preregistered sqrt market-impact model.

Covers the mandated edge cases: zero quantity, low ADV, missing volatility,
participation/limit blocks (oversized fills), partial fills (impact on filled
part only) and magnitude sanity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.backtest.impact_model import (
    ETA_BASE, ETA_STRESSED, SqrtImpactParams, apply_impact, sqrt_impact_return,
)


def test_zero_quantity_no_impact():
    imp, reason = sqrt_impact_return(0.0, adv20=1e7, vol20=0.02)
    assert imp == 0.0 and reason == ""


def test_magnitude_and_eta_scaling():
    # 1% of ADV at 2% daily vol: impact = eta * 0.02 * sqrt(0.01) = 20bps * eta/10
    imp, _ = sqrt_impact_return(1e5, adv20=1e7, vol20=0.02)
    assert imp == pytest.approx(ETA_BASE * 0.02 * 0.1)
    imp2, _ = sqrt_impact_return(1e5, adv20=1e7, vol20=0.02,
                                 params=SqrtImpactParams(eta=ETA_STRESSED))
    assert imp2 == pytest.approx(2 * imp)


def test_missing_volatility_fails_loud():
    imp, reason = sqrt_impact_return(1e5, adv20=1e7, vol20=float("nan"))
    assert np.isnan(imp) and reason == "missing_volatility"


def test_missing_adv_fails_loud():
    imp, reason = sqrt_impact_return(1e5, adv20=float("nan"), vol20=0.02)
    assert np.isnan(imp) and reason == "missing_adv"


def test_low_adv_floor():
    imp, reason = sqrt_impact_return(10.0, adv20=0.5, vol20=0.02)
    assert np.isnan(imp) and reason == "adv_below_floor"


def test_oversized_fill_rejected_not_priced():
    # a fill above the participation cap is impossible; impact must refuse
    imp, reason = sqrt_impact_return(2e6, adv20=1e7, vol20=0.02)  # 20% > 10% cap
    assert np.isnan(imp) and reason == "fill_exceeds_participation_cap"


def test_partial_fill_charged_on_filled_only():
    # order wanted 2e6 but simulator filled the 10% cap = 1e6; impact priced
    # on 1e6 and equals the direct call on the filled value
    filled = 0.10 * 1e7
    imp, reason = sqrt_impact_return(filled, adv20=1e7, vol20=0.02)
    assert reason == "" and imp == pytest.approx(ETA_BASE * 0.02 * np.sqrt(0.10))
    # unfilled remainder carries no impact anywhere: pricing is on fills only


def test_vectorized_matches_scalar_and_flags():
    df = pd.DataFrame({
        "filled_value": [0.0, 1e5, 2e6, 1e5, 1e5, -5.0],
        "adv20": [1e7, 1e7, 1e7, np.nan, 1e7, 1e7],
        "vol20": [0.02, 0.02, 0.02, 0.02, np.nan, 0.02],
    })
    out = apply_impact(df)
    assert out["impact_return"].iloc[0] == 0.0
    s, _ = sqrt_impact_return(1e5, 1e7, 0.02)
    assert out["impact_return"].iloc[1] == pytest.approx(s)
    assert out["impact_reason"].iloc[2] == "fill_exceeds_participation_cap"
    assert out["impact_reason"].iloc[3] == "missing_adv"
    assert out["impact_reason"].iloc[4] == "missing_volatility"
    assert out["impact_reason"].iloc[5] == "invalid_filled_value"
    assert (out["impact_cost_cny"].iloc[1]
            == pytest.approx(out["impact_return"].iloc[1] * 1e5))
    # input frame not mutated
    assert "impact_return" not in df.columns


def test_limit_block_zero_fill_path():
    # limit-up block => simulator fills 0 => zero impact through the same API
    imp, reason = sqrt_impact_return(0.0, adv20=5e6, vol20=0.05)
    assert imp == 0.0 and reason == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
