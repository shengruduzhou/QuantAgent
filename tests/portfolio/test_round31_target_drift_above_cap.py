"""Round-31 R31-RISK-06: target construction must not crash on ordinary price drift.

A name bought at the 10% single-name cap that rallies 20% is held at ~11.8%.
With the paper loop's binding turnover cap the projection keeps it above the
cap, and the long-only validator then refuses to emit any target: the paper
daily loop produces nothing for that day. Expected (hand): the drifted name is
trimmed toward the cap (a sell is always feasible unless the name is frozen).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

DATES = pd.bdate_range("2025-06-02", periods=1)
NAMES = [f"0000{i:02d}.SZ" for i in range(40)]


def _market():
    return pd.DataFrame({"trade_date": DATES[0], "symbol": NAMES, "amount": 5e9, "close": 10.0,
                         "is_suspended": False, "is_st": False, "is_limit_up": False,
                         "is_limit_down": False})


def _preds():
    return pd.DataFrame({"trade_date": DATES[0], "symbol": NAMES,
                         "prediction": np.linspace(0.05, -0.05, 40), "confidence": 0.9})


def _cfg(**kw):
    base = dict(selection_mode="top_k", top_k=10, top_k_ratio=None, max_weight_per_name=0.10,
                max_sector_weight=1.0, max_turnover=0.40, weighting="equal")
    base.update(kw)
    return V7TargetWeightsConfig(**base)


def test_drifted_winner_above_single_name_cap_with_binding_turnover_cap():
    # Held: 9 low-ranked names at ~9.8% + one winner drifted to 11.8% (all tradeable).
    # The winner is still top-ranked (desired 10%), so the turnover projection keeps
    # it at 0.118 + (0.10-0.118)*scale > 0.10.
    held = pd.Series({n: 0.0980 for n in NAMES[31:40]})
    held[NAMES[0]] = 0.118
    res = build_v7_target_weights(_preds(), _market(), sector_map=None, config=_cfg(), initial_weights=held)
    row = res.target_weights.set_index("trade_date").iloc[-1].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    assert float(row.max()) <= 0.10 + 1e-9


def test_drifted_sector_above_cap_with_binding_turnover_cap():
    sector = pd.DataFrame({"symbol": NAMES, "industry": ["B"] * 4 + ["A"] * 16 + ["B"] * 20})
    # Held: four sector-B names that rallied to 8% each = 32% > 30% cap.
    held = pd.Series({n: 0.08 for n in NAMES[20:24]})  # top-ranked B names
    res = build_v7_target_weights(_preds(), _market(), sector_map=sector,
                                  config=_cfg(max_sector_weight=0.30, max_turnover=0.10),
                                  initial_weights=held)
    assert not res.target_weights.empty
