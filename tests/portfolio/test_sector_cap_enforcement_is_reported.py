"""The target optimiser must say when its sector cap was not applied.

Without a sector map `build_v7_target_weights` skips the sector cap entirely,
yet its diagnostics published `constraint_surface.sector_cap = 0.30` — so a
100% single-sector book read exactly like a capped one (the DEF-040 pattern,
for the sector cap). The paper CLI's `--sector-map` is optional, so this was a
reachable production state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

DATE = pd.Timestamp("2025-06-03")
BANKS = [f"6010{i:02d}.SH" for i in range(10)]   # the 10 top-ranked names: one industry
OTHERS = [f"0000{i:02d}.SZ" for i in range(30)]


def _inputs():
    symbols = BANKS + OTHERS
    predictions = pd.DataFrame({
        "trade_date": DATE, "symbol": symbols,
        "prediction": np.r_[np.linspace(0.05, 0.04, 10), np.linspace(0.01, -0.02, 30)],
        "confidence": 0.9,
    })
    market = pd.DataFrame({
        "trade_date": DATE, "symbol": symbols, "amount": 5e9, "close": 10.0,
        "is_suspended": False, "is_st": False, "is_limit_up": False, "is_limit_down": False,
    })
    sector = pd.DataFrame({"symbol": symbols, "industry": ["bank"] * 10 + ["other"] * 30})
    return predictions, market, sector


def _config() -> V7TargetWeightsConfig:
    return V7TargetWeightsConfig(selection_mode="top_k", top_k=10, top_k_ratio=None,
                                 max_weight_per_name=0.10, max_sector_weight=0.30,
                                 max_turnover=0.0, weighting="equal")


def _bank_share(result) -> float:
    frame = result.target_weights.set_index("trade_date")
    row = pd.to_numeric(frame.iloc[-1], errors="coerce").fillna(0.0)
    return float(row.reindex(BANKS).fillna(0.0).sum())


def test_without_a_sector_map_the_cap_is_published_as_unenforced() -> None:
    predictions, market, _ = _inputs()
    result = build_v7_target_weights(predictions, market, sector_map=None, config=_config())
    diagnostics = result.diagnostics

    assert _bank_share(result) > 0.30  # the cap really did not bind
    assert diagnostics["sector_cap_enforced"] is False
    assert diagnostics["sector_cap_unenforced"] == {
        "reason": "sector_map_absent", "requested": 0.30,
    }
    assert diagnostics["constraint_surface"]["sector_cap"] is None
    assert diagnostics["config"]["max_sector_weight"] is None
    assert diagnostics["config_requested"]["max_sector_weight"] == 0.30


def test_with_a_sector_map_the_cap_binds_and_is_published_as_enforced() -> None:
    predictions, market, sector = _inputs()
    result = build_v7_target_weights(predictions, market, sector_map=sector, config=_config())
    diagnostics = result.diagnostics

    assert _bank_share(result) <= 0.30 + 1e-9
    assert diagnostics["sector_cap_enforced"] is True
    assert diagnostics["sector_cap_unenforced"] is None
    assert diagnostics["constraint_surface"]["sector_cap"] == 0.30
