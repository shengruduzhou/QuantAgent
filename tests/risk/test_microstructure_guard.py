"""Tests for the DEFENSIVE microstructure risk guard (detect-to-avoid)."""
from __future__ import annotations

import pandas as pd

from quantagent.risk.microstructure_guard import market_risk_off_level, microstructure_guard


def _frame():
    rows = []
    # one clear 砸盘/断魂刀 name: closes at low, strong net sell, volume climax
    rows.append({"symbol": "DUMP", "net_buy_pressure": -0.9, "vwap_deviation": -0.05,
                 "intraday_range_pos": 0.02, "volume_concentration": 0.95, "close30_volume_share": 0.5})
    # healthy names
    for i in range(20):
        rows.append({"symbol": f"OK{i}", "net_buy_pressure": 0.2 + i * 0.01, "vwap_deviation": 0.01,
                     "intraday_range_pos": 0.7, "volume_concentration": 0.3, "close30_volume_share": 0.15})
    return pd.DataFrame(rows)


def test_guard_flags_sweep_dump_for_avoid():
    out = microstructure_guard(_frame())
    dump = out[out.symbol == "DUMP"].iloc[0]
    assert dump["sweep_dump_risk"] >= 0.8
    assert dump["guard_action"] == "avoid"
    # healthy names should mostly be ok
    assert (out[out.symbol.str.startswith("OK")]["guard_action"] == "ok").mean() >= 0.8


def test_market_risk_off_detects_broad_selling():
    selling = pd.DataFrame([{"net_buy_pressure": -0.5} for _ in range(80)]
                           + [{"net_buy_pressure": 0.1} for _ in range(20)])
    r = market_risk_off_level(selling, north_total=-50.0)
    assert r["level"] == "risk_off"
    assert r["recommended_gross_cap"] <= 0.40
    calm = pd.DataFrame([{"net_buy_pressure": 0.2} for _ in range(100)])
    assert market_risk_off_level(calm, north_total=10.0)["level"] == "normal"
