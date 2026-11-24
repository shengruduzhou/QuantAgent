"""R31-RISK-07: on a bonus ex-date the canonical simulator rejected a whole
full-liquidation sell (1,300 requested: 1,000 settled + 300 frozen bonus
shares) instead of selling the 1,000 that were available."""
from __future__ import annotations

import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    simulate_ashare_target_weights,
)

A = "600000.SH"
DATES = ["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07"]
EX_CLOSE = (10.0 - 0.5) / 1.3


def _panel() -> pd.DataFrame:
    rows = []
    for d in DATES:
        ex = d == "2024-06-06"
        rows.append({"trade_date": d, "symbol": A, "close": 10.0 if d < "2024-06-06" else EX_CLOSE,
                     "volume": 1e9, "ca_cash_per_share": 0.5 if ex else 0.0,
                     "ca_share_ratio": 0.3 if ex else 0.0})
    panel = pd.DataFrame(rows)
    for col in ("is_suspended", "is_st", "is_limit_up", "is_limit_down"):
        panel[col] = False
    panel["amount"] = panel["close"] * panel["volume"]
    return panel


def test_full_liquidation_on_bonus_ex_date_sells_the_available_shares(tmp_path):
    targets = {"2024-06-03": 0.01, "2024-06-05": 0.0, "2024-06-06": 0.0}
    tw = pd.DataFrame({A: list(targets.values())}, index=pd.to_datetime(list(targets)))
    tw.attrs["target_index_semantics"] = "signal_date"
    cfg = AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=8.0,
                                          audit_log_dir=str(tmp_path / "audit"))
    result = simulate_ashare_target_weights(tw, _panel(), cfg)

    credit = result.corporate_action_audit
    credit = credit[credit["basis"] != "dividend_tax"]
    assert int(credit["bonus_shares"].iloc[0]) == 300
    orders = result.order_audit
    ex_day = orders[orders["trade_date"] == pd.Timestamp("2024-06-06")]
    assert int(ex_day["filled_quantity"].iloc[0]) == 1000, ex_day.to_dict("records")
    # The 300 bonus shares are sold on the next session.
    next_day = orders[orders["trade_date"] == pd.Timestamp("2024-06-07")]
    assert int(next_day["filled_quantity"].sum()) == 300, next_day.to_dict("records")
