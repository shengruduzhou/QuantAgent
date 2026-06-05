from __future__ import annotations

import pandas as pd

from quantagent.factors.cicc_selection import compute_cicc_selection_scores


def test_cicc_selection_scores_include_sector_score():
    d = pd.Timestamp("2024-01-02")
    factors = pd.DataFrame({
        "trade_date": [d, d, d, d],
        "symbol": ["A", "B", "C", "D"],
        "cicc_mmt_ret_5d": [0.10, 0.05, -0.02, -0.01],
        "cicc_liq_amihud_20d": [0.1, 0.2, 0.8, 0.9],
        "cicc_crowd_volume_conc_20d": [0.1, 0.3, 0.8, 0.9],
    })
    sector_map = pd.DataFrame({
        "symbol": ["A", "B", "C", "D"],
        "sector_level_1": ["S1", "S1", "S2", "S2"],
    })

    out = compute_cicc_selection_scores(factors, sector_map=sector_map)

    assert "cicc_stock_selection_score" in out.columns
    assert "cicc_sector_selection_score" in out.columns
    assert out.set_index("symbol").loc["A", "cicc_stock_selection_score"] > out.set_index("symbol").loc["D", "cicc_stock_selection_score"]
