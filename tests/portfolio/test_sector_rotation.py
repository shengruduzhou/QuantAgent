"""Tests for 板块轮动 / 高低切 / 做T overlay."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.portfolio.sector_rotation import attach_rotation_and_dot, compute_sector_heat


def _panel():
    dates = pd.bdate_range("2026-01-01", periods=80)
    rows = []
    # 高位 sector A (rising to high), 低位 sector B (falling then flat)
    for i, d in enumerate(dates):
        for sym, base in [("A1", 10 + i * 0.2), ("A2", 12 + i * 0.18),
                          ("A3", 11 + i * 0.19), ("A4", 13 + i * 0.17), ("A5", 9 + i * 0.21),
                          ("B1", 30 - i * 0.1), ("B2", 28 - i * 0.09),
                          ("B3", 26 - i * 0.08), ("B4", 25 - i * 0.07), ("B5", 27 - i * 0.06)]:
            rows.append({"symbol": sym, "trade_date": d, "close": base})
    return pd.DataFrame(rows)


def _sector_map():
    return pd.DataFrame([{"symbol": s, "sector_level_1": ("电子" if s.startswith("A") else "传媒")}
                         for s in ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5"]])


def test_sector_heat_separates_high_and_low():
    heat = compute_sector_heat(_panel(), _sector_map(), pd.Timestamp("2026-04-20"))
    tags = heat.set_index("sector_level_1")["regime_tag"].to_dict()
    # rising sector should be 高位, falling 低位 (with 2 sectors -> one each side of median)
    assert tags["电子"] == "高位"
    assert tags["传媒"] == "低位"
    assert {"rotation_score", "rotation_tag"}.issubset(heat.columns)


def test_attach_rotation_and_dot_flags_hot_sector_for_dot():
    heat = compute_sector_heat(_panel(), _sector_map(), pd.Timestamp("2026-04-20"))
    pool = pd.DataFrame([
        {"symbol": "A1", "sector_level_1": "电子", "do_t_suitability_score": 0.7},
        {"symbol": "B1", "sector_level_1": "传媒", "do_t_suitability_score": 0.2},
    ])
    out = attach_rotation_and_dot(pool, heat)
    assert {"sector_rotation_score", "sector_regime_tag", "do_t_action"}.issubset(out.columns)
    # 高位 sector holding with tradable do_t -> 做T管理
    a1 = out[out.symbol == "A1"].iloc[0]
    assert a1["sector_regime_tag"] == "高位"
    assert "做T" in a1["do_t_action"]
