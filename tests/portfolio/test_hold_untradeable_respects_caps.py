"""Pinned untradeable holdings respect sector/turnover caps and are not held forever (round-29 R10-F02)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

DATES = pd.bdate_range("2025-06-02", periods=4)
NAMES = [f"0000{i:02d}.SZ" for i in range(40)]


def _market(dates=DATES, names=NAMES, **flags):
    f = pd.concat([pd.DataFrame({"trade_date": d, "symbol": names, "amount": 5e9, "close": 10.0,
                                 "is_suspended": False, "is_st": False, "is_limit_up": False,
                                 "is_limit_down": False}) for d in dates], ignore_index=True)
    for col, syms in flags.items():
        f.loc[f["symbol"].isin(syms), col] = True
    return f


def _preds(dates=DATES):
    return pd.concat([pd.DataFrame({"trade_date": d, "symbol": NAMES, "prediction": np.linspace(0.05, -0.05, 40),
                                    "confidence": 0.9}) for d in dates], ignore_index=True)


def test_sector_cap_survives_holding_a_limit_down_name():
    sector = pd.DataFrame({"symbol": NAMES, "industry": [chr(65 + i % 4) for i in range(40)]})
    held_name = NAMES[30]  # sector C, ranked low, limit-down today -> frozen at 0.20
    held = pd.Series({held_name: 0.20})
    cfg = V7TargetWeightsConfig(selection_mode="top_k", top_k=10, top_k_ratio=None, max_weight_per_name=0.20,
                                max_sector_weight=0.30, max_turnover=0.0, weighting="equal")
    res = build_v7_target_weights(_preds(DATES[:1]), _market(DATES[:1], is_limit_down=[held_name]),
                                  sector_map=sector, config=cfg, initial_weights=held)
    row = res.target_weights.set_index("trade_date").iloc[-1].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row = row[[c for c in row.index if c in NAMES]]
    exp = row.groupby(sector.set_index("symbol")["industry"].reindex(row.index)).sum()

    assert float(exp.max()) <= 0.30 + 1e-9


def test_a_delisted_holding_is_not_held_forever():
    held = pd.Series({"600999.SH": 0.10})          # no market row on any date (delisted)
    cfg = V7TargetWeightsConfig(selection_mode="top_k", top_k=5, top_k_ratio=None, max_weight_per_name=0.20,
                                max_sector_weight=1.0, max_turnover=0.0, weighting="equal")
    preds = pd.concat([_preds(), pd.DataFrame({"trade_date": DATES, "symbol": "600999.SH", "prediction": 0.1, "confidence": 0.9})])
    res = build_v7_target_weights(preds, _market(), sector_map=None, config=cfg, initial_weights=held)
    tw = res.target_weights.set_index("trade_date")
    column = tw["600999.SH"] if "600999.SH" in tw.columns else pd.Series(0.0, index=tw.index)
    w = pd.to_numeric(column, errors="coerce").fillna(0.0)
    assert res.diagnostics["held_untradeable_released"]

    assert float(w.iloc[-1]) == 0.0 or len(w) < len(DATES)


def test_turnover_cap_survives_holding_a_limit_down_name():
    # Held: 10 bottom-ranked names at 0.10. Today the top-ranked differ; one held name is limit-down.
    held = pd.Series({n: 0.10 for n in NAMES[30:40]})
    frozen = NAMES[39]
    cfg = V7TargetWeightsConfig(selection_mode="top_k", top_k=10, top_k_ratio=None, max_weight_per_name=0.20,
                                max_sector_weight=1.0, max_turnover=0.20, weighting="equal")
    res = build_v7_target_weights(_preds(DATES[:1]), _market(DATES[:1], is_limit_down=[frozen]),
                                  sector_map=None, config=cfg, initial_weights=held)
    row = res.target_weights.set_index("trade_date").iloc[-1].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row = row[[c for c in row.index if c in NAMES]]
    u = row.index.union(held.index)
    turnover = 0.5 * float((row.reindex(u).fillna(0) - held.reindex(u).fillna(0)).abs().sum())

    assert turnover <= 0.20 + 1e-9
