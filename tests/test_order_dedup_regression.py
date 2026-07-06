"""INC-E1 regression contract (EVALUATOR_ORDER_DEDUP_BUG.md).

A backtest must be able to re-trade the same (symbol, side) on later days.
Today it cannot: OrderManager's idempotency dedupe uses a deterministic
per-(symbol, side) client_order_id and a never-cleared history, so the
day-3 rebuy below is silently dropped.

Marked xfail(strict=True): when the approved fix lands this test will
XPASS and pytest will FAIL until the marker is removed — a deliberate
forcing function to acknowledge the evaluator semantic change.
"""
import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    simulate_ashare_target_weights,
)


@pytest.mark.xfail(
    strict=True,
    reason="INC-E1: cross-day (symbol, side) order dedupe drops the rebuy — "
    "fix pending user approval (EVALUATOR_ORDER_DEDUP_BUG.md §6)",
)
def test_buy_cut_rebuy_all_three_orders_fill(tmp_path):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    sym = "000001.SZ"
    panel = pd.DataFrame(
        {
            "trade_date": list(dates),
            "symbol": [sym] * 4,
            "open": [10.0] * 4,
            "high": [10.5] * 4,
            "low": [9.5] * 4,
            "close": [10.0] * 4,
            "volume": [1e8] * 4,
            "amount": [1e9] * 4,
            "is_suspended": [False] * 4,
            "is_st": [False] * 4,
            "is_limit_up": [False] * 4,
            "is_limit_down": [False] * 4,
        }
    )
    tw = pd.DataFrame({sym: [0.50, 0.25, 0.50, 0.50]}, index=dates)
    cfg = AShareExecutionSimulationConfig(
        initial_cash=1_000_000.0, audit_log_dir=str(tmp_path)
    )
    sim = simulate_ashare_target_weights(tw, panel, cfg)
    filled = sim.order_audit[
        sim.order_audit["filled_quantity"].astype(float).abs() > 0
    ]
    # buy on d1, sell on d2, and the d3 rebuy MUST also execute
    assert len(filled) == 3, f"expected buy/sell/rebuy, got:\n{filled}"
    sides = list(filled["side"]) if "side" in filled.columns else []
    assert sides.count("buy") == 2 and sides.count("sell") == 1
