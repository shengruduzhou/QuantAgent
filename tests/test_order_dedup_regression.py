"""INC-E1 regression contract (EVALUATOR_ORDER_DEDUP_BUG.md).

A backtest must be able to re-trade the same (symbol, side) on later days.
Before the fix it could not: OrderManager's idempotency dedupe used a
deterministic per-(symbol, side) client_order_id and a never-cleared history,
so the day-3 rebuy below was silently dropped.

The trusted evaluator now also enforces signal-date -> next-session execution.
That timing contract must not reintroduce the old cross-day suppression, even
when the historical ``fix_cross_day_order_dedup`` compatibility flag is false.
The flag remains a forensic compatibility knob, not permission to violate the
current production timing/idempotency invariant.
"""
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    EXECUTION_TIMING_SEMANTICS,
    simulate_ashare_target_weights,
)


def _fixture():
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
    return dates, sym, panel, tw


def _filled(sim):
    return sim.order_audit[
        sim.order_audit["filled_quantity"].astype(float).abs() > 0
    ]


def test_buy_cut_rebuy_all_three_orders_fill(tmp_path):
    dates, sym, panel, tw = _fixture()
    cfg = AShareExecutionSimulationConfig(
        initial_cash=1_000_000.0, audit_log_dir=str(tmp_path)
    )
    sim = simulate_ashare_target_weights(tw, panel, cfg)
    filled = _filled(sim)

    # Signals on d1/d2/d3 execute on d2/d3/d4 respectively. The final d4 signal
    # is right-censored because no next session exists in the bounded panel.
    assert len(filled) == 3, f"expected buy/sell/rebuy, got:\n{filled}"
    assert list(filled["execution_date"]) == list(dates[1:4])
    sides = list(filled["side"]) if "side" in filled.columns else []
    assert sides.count("buy") == 2 and sides.count("sell") == 1
    assert sim.config["execution_timing_semantics"] == EXECUTION_TIMING_SEMANTICS


def test_legacy_flag_cannot_reintroduce_cross_day_order_drop(tmp_path):
    """The old forensic flag cannot weaken the current executable invariant.

    Historical artifacts remain readable as historical evidence, but regenerating
    a knowingly wrong economic path under the current production simulator would
    create a second, contradictory execution contract. The compatibility flag may
    change internal replay plumbing; it may not silently suppress the d3 rebuy.
    """
    dates, sym, panel, tw = _fixture()
    cfg = AShareExecutionSimulationConfig(
        initial_cash=1_000_000.0,
        audit_log_dir=str(tmp_path),
        fix_cross_day_order_dedup=False,
    )
    sim = simulate_ashare_target_weights(tw, panel, cfg)
    filled = _filled(sim)

    assert len(filled) == 3, f"compatibility mode must not drop a valid rebuy:\n{filled}"
    assert list(filled["execution_date"]) == list(dates[1:4])
    assert sim.config["execution_trace_ok"] is True
