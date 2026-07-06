"""INC-E1 regression contract (EVALUATOR_ORDER_DEDUP_BUG.md).

A backtest must be able to re-trade the same (symbol, side) on later days.
Before the fix it could not: OrderManager's idempotency dedupe used a
deterministic per-(symbol, side) client_order_id and a never-cleared history,
so the day-3 rebuy below was silently dropped.

The fix (fix_cross_day_order_dedup, PROMOTED to the trusted-evaluator default
2026-07-06 with user approval) resets per-symbol counters each simulated day
and stamps orders with a per-day signal_id. This test now asserts the corrected
behaviour under the default config; a companion test pins the legacy (buggy)
path still reproduces the drop when the flag is explicitly disabled, so
pre-INC-E1 artifacts remain forensically reproducible.
"""
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    simulate_ashare_target_weights,
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


def test_legacy_pre_inc_e1_path_still_reproduces_the_drop(tmp_path):
    """Forensic guard: with the fix explicitly disabled the pre-INC-E1 simulator
    must still silently drop the day-3 rebuy (2 fills), so stamped pre-INC-E1
    artifacts can be regenerated and compared against the corrected numbers."""
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
        initial_cash=1_000_000.0,
        audit_log_dir=str(tmp_path),
        fix_cross_day_order_dedup=False,
    )
    sim = simulate_ashare_target_weights(tw, panel, cfg)
    filled = sim.order_audit[
        sim.order_audit["filled_quantity"].astype(float).abs() > 0
    ]
    # legacy bug: only the first (symbol, side) buy + first sell survive
    assert len(filled) == 2, f"legacy path should drop the rebuy, got:\n{filled}"
