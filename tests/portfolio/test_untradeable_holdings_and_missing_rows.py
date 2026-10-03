"""Target weights must not trade what cannot trade (round-29 R4-F03 / R4-F04 / R9-F05).

* A name with no market row on the date has unmeasured tradability; it used to
  read as tradable and could take the full name cap.
* A held name blocked today keeps its previous weight: a limit-up buy block
  must not force-sell the holding, and a limit-down / suspended holding cannot
  be sold, so a lower target is an order no venue can fill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

DATE = pd.Timestamp("2025-06-03")
NAMES = [f"0000{i:02d}.SZ" for i in range(20)]


def _config() -> V7TargetWeightsConfig:
    return V7TargetWeightsConfig(selection_mode="top_k", top_k=5, top_k_ratio=None,
                                 max_weight_per_name=0.20, max_sector_weight=1.0,
                                 max_turnover=0.0, weighting="equal")


def _market(**flags):
    frame = pd.DataFrame({
        "trade_date": DATE, "symbol": NAMES, "amount": 5e9, "close": 10.0,
        "is_suspended": False, "is_st": False, "is_limit_up": False, "is_limit_down": False,
    })
    for column, symbols in flags.items():
        frame.loc[frame["symbol"].isin(symbols), column] = True
    return frame


def _predictions():
    return pd.DataFrame({"trade_date": DATE, "symbol": NAMES,
                         "prediction": np.linspace(0.05, -0.05, len(NAMES)), "confidence": 0.9})


def _row(result) -> pd.Series:
    frame = result.target_weights.set_index("trade_date")
    return pd.to_numeric(frame.iloc[-1], errors="coerce").fillna(0.0)


def test_a_name_without_a_market_row_is_rejected_not_weighted() -> None:
    market = _market()
    top = NAMES[0]
    result = build_v7_target_weights(_predictions(), market[market["symbol"] != top],
                                     sector_map=None, config=_config())
    assert float(_row(result).get(top, 0.0)) == 0.0
    reasons = {(r["symbol"], r["reason"]) for r in result.diagnostics["rejected"]}
    assert (top, "no_market_row") in reasons


def test_held_limit_up_and_limit_down_names_keep_their_weight() -> None:
    held = pd.Series({NAMES[0]: 0.15, NAMES[19]: 0.10})  # top-ranked and bottom-ranked holdings
    market = _market(is_limit_up=[NAMES[0]], is_limit_down=[NAMES[19]])
    result = build_v7_target_weights(_predictions(), market, sector_map=None, config=_config(),
                                     initial_weights=held)
    row = _row(result)
    assert row.get(NAMES[0], 0.0) == 0.15   # not force-sold by a buy block
    assert row.get(NAMES[19], 0.0) == 0.10  # no unexecutable sell target
    assert row.abs().sum() <= 1.0 + 1e-9
