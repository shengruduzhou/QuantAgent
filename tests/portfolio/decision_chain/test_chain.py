"""Tests for the Stage 5.3 14-step decision chain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.portfolio.decision_chain import (
    GATE_ORDER,
    Candidate,
    DecisionChainConfig,
    DecisionContext,
    run_decision_chain,
    run_decision_chain_batch,
    traces_to_frame,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def _candidate(symbol: str = "600000.SH", alpha: float = 0.5, setup: str = "breakout", weight: float = 0.02):
    return Candidate(
        trade_date=pd.Timestamp("2024-03-01"),
        symbol=symbol,
        alpha_score=alpha,
        setup_label=setup,
        target_weight=weight,
    )


def _liquid_market(symbol: str = "600000.SH", amount: float = 200_000_000.0, daily_ret: float = 0.005) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2024-03-01"),
                "symbol": symbol,
                "amount": amount,
                "suspension": False,
                "daily_return": daily_ret,
            }
        ]
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_all_skip_chain_passes_with_defaults():
    """With no context inputs, every gate that needs context skips → eligible."""
    cand = _candidate()
    trace = run_decision_chain(cand, DecisionContext())
    assert trace.final_decision == "eligible"
    assert trace.failed_gate is None
    # All 14 gates appear in the trace
    assert [g.gate_name for g in trace.gate_results] == list(GATE_ORDER)
    assert all(g.passed for g in trace.gate_results)


def test_fully_populated_happy_path():
    cand = _candidate(symbol="600000.SH")
    ctx = DecisionContext(
        market_panel=_liquid_market(),
        st_flags=pd.DataFrame(
            [{"trade_date": pd.Timestamp("2024-03-01"), "symbol": "600000.SH", "is_st": False}]
        ),
        sector_map=pd.DataFrame([{"symbol": "600000.SH", "sector_level_1": "Bank"}]),
        sector_pool=pd.DataFrame([{"sector_level_1": "Bank", "pool_tier": "core"}]),
        hard_gate_frame=pd.DataFrame(
            [{"trade_date": pd.Timestamp("2024-03-01"), "hard_gate_active": False}]
        ),
        regime_state=pd.Series(["normal"], index=[pd.Timestamp("2024-03-01")]),
        fundamental_ranker=pd.DataFrame([{"symbol": "600000.SH", "composite_rank": 0.80}]),
        policy_signal_by_sector=pd.DataFrame(
            [{"trade_date": pd.Timestamp("2024-03-01"), "sector": "Bank", "policy_signal": 0.30}]
        ),
        broker_consensus=pd.DataFrame(
            [{"trade_date": pd.Timestamp("2024-03-01"), "symbol": "600000.SH", "broker_consensus_score": 0.60}]
        ),
        stock_drawdown=pd.DataFrame(
            [{"trade_date": pd.Timestamp("2024-03-01"), "symbol": "600000.SH", "dd_20d": -0.05}]
        ),
        current_weights={"600000.SH": 0.0},
        sector_weights={"Bank": 0.10},
    )
    trace = run_decision_chain(cand, ctx)
    assert trace.final_decision == "eligible"


# ---------------------------------------------------------------------------
# Each gate FAIL short-circuits in turn
# ---------------------------------------------------------------------------

def test_alpha_threshold_fail_short_circuits():
    cand = _candidate(alpha=-0.5)
    cfg = DecisionChainConfig(min_alpha=0.0)
    trace = run_decision_chain(cand, DecisionContext(), cfg)
    assert trace.final_decision == "rejected"
    assert trace.failed_gate == "alpha_threshold"
    # Short-circuit: only 1 gate evaluated
    assert len(trace.gate_results) == 1


def test_liquidity_fail():
    cand = _candidate()
    ctx = DecisionContext(market_panel=_liquid_market(amount=10_000_000))  # < 5000 万
    trace = run_decision_chain(cand, ctx)
    assert trace.failed_gate == "liquidity"


def test_tradeable_today_fail_on_suspension():
    cand = _candidate()
    panel = _liquid_market()
    panel["suspension"] = True
    trace = run_decision_chain(cand, DecisionContext(market_panel=panel))
    assert trace.failed_gate == "tradeable_today"


def test_price_limit_up_block():
    cand = _candidate()
    panel = _liquid_market(daily_ret=0.10)
    trace = run_decision_chain(cand, DecisionContext(market_panel=panel))
    assert trace.failed_gate == "price_limit_block"


def test_price_limit_down_block():
    cand = _candidate()
    panel = _liquid_market(daily_ret=-0.10)
    trace = run_decision_chain(cand, DecisionContext(market_panel=panel))
    assert trace.failed_gate == "price_limit_block"


def test_st_status_blocks_when_st_and_allow_false():
    cand = _candidate()
    st = pd.DataFrame([{"trade_date": pd.Timestamp("2024-03-01"), "symbol": "600000.SH", "is_st": True}])
    trace = run_decision_chain(cand, DecisionContext(st_flags=st))
    assert trace.failed_gate == "st_status"


def test_st_status_passes_when_allow_st():
    cand = _candidate()
    st = pd.DataFrame([{"trade_date": pd.Timestamp("2024-03-01"), "symbol": "600000.SH", "is_st": True}])
    cfg = DecisionChainConfig(allow_st=True)
    trace = run_decision_chain(cand, DecisionContext(st_flags=st), cfg)
    assert trace.final_decision == "eligible"


def test_sector_pool_blocks_excluded_tier():
    cand = _candidate()
    ctx = DecisionContext(
        sector_map=pd.DataFrame([{"symbol": "600000.SH", "sector_level_1": "Bank"}]),
        sector_pool=pd.DataFrame([{"sector_level_1": "Bank", "pool_tier": "excluded"}]),
    )
    trace = run_decision_chain(cand, ctx)
    assert trace.failed_gate == "sector_pool"


def test_hard_market_gate_active_blocks():
    cand = _candidate()
    hg = pd.DataFrame([{"trade_date": pd.Timestamp("2024-03-01"), "hard_gate_active": True}])
    trace = run_decision_chain(cand, DecisionContext(hard_gate_frame=hg))
    assert trace.failed_gate == "hard_market_gate"


def test_regime_alignment_crisis_blocks_all():
    cand = _candidate(setup="lowbuy")
    regimes = pd.Series(["crisis"], index=[pd.Timestamp("2024-03-01")])
    trace = run_decision_chain(cand, DecisionContext(regime_state=regimes))
    assert trace.failed_gate == "regime_alignment"


def test_regime_alignment_breakout_blocked_in_bear():
    cand = _candidate(setup="breakout")
    regimes = pd.Series(["bear"], index=[pd.Timestamp("2024-03-01")])
    trace = run_decision_chain(cand, DecisionContext(regime_state=regimes))
    assert trace.failed_gate == "regime_alignment"


def test_regime_alignment_lowbuy_allowed_in_bear():
    cand = _candidate(setup="lowbuy")
    regimes = pd.Series(["bear"], index=[pd.Timestamp("2024-03-01")])
    trace = run_decision_chain(cand, DecisionContext(regime_state=regimes))
    assert trace.final_decision == "eligible"


def test_fundamental_filter_low_rank_blocks():
    cand = _candidate()
    fr = pd.DataFrame([{"symbol": "600000.SH", "composite_rank": 0.10}])
    cfg = DecisionChainConfig(fundamental_rank_min_pct=0.30)
    trace = run_decision_chain(cand, DecisionContext(fundamental_ranker=fr), cfg)
    assert trace.failed_gate == "fundamental_filter"


def test_policy_aligned_negative_signal_blocks():
    cand = _candidate()
    ctx = DecisionContext(
        sector_map=pd.DataFrame([{"symbol": "600000.SH", "sector_level_1": "RealEstate"}]),
        policy_signal_by_sector=pd.DataFrame(
            [{"trade_date": pd.Timestamp("2024-03-01"), "sector": "RealEstate", "policy_signal": -0.50}]
        ),
    )
    cfg = DecisionChainConfig(policy_signal_min=-0.20)
    trace = run_decision_chain(cand, ctx, cfg)
    assert trace.failed_gate == "policy_aligned"


def test_broker_consensus_strong_sell_blocks():
    cand = _candidate()
    bc = pd.DataFrame(
        [{"trade_date": pd.Timestamp("2024-03-01"), "symbol": "600000.SH", "broker_consensus_score": -0.80}]
    )
    cfg = DecisionChainConfig(broker_consensus_min=-0.30)
    trace = run_decision_chain(cand, DecisionContext(broker_consensus=bc), cfg)
    assert trace.failed_gate == "broker_consensus"


def test_drawdown_kill_blocks_severe_drawdown():
    cand = _candidate()
    dd = pd.DataFrame([{"trade_date": pd.Timestamp("2024-03-01"), "symbol": "600000.SH", "dd_20d": -0.35}])
    cfg = DecisionChainConfig(stock_drawdown_kill_pct=-0.20)
    trace = run_decision_chain(cand, DecisionContext(stock_drawdown=dd), cfg)
    assert trace.failed_gate == "drawdown_kill"


def test_concentration_limit_blocks_over_sector_cap():
    cand = _candidate(weight=0.10)
    ctx = DecisionContext(
        sector_map=pd.DataFrame([{"symbol": "600000.SH", "sector_level_1": "Bank"}]),
        sector_weights={"Bank": 0.25},  # already at 25%, candidate +10% → 35% > 30% cap
    )
    cfg = DecisionChainConfig(max_sector_weight=0.30)
    trace = run_decision_chain(cand, ctx, cfg)
    assert trace.failed_gate == "concentration_limit"


def test_risk_budget_blocks_over_name_cap():
    cand = _candidate(weight=0.10)
    cfg = DecisionChainConfig(max_name_weight=0.03)
    trace = run_decision_chain(cand, DecisionContext(), cfg)
    assert trace.failed_gate == "risk_budget"


# ---------------------------------------------------------------------------
# Disabled gates
# ---------------------------------------------------------------------------

def test_disabled_gates_skip_execution():
    cand = _candidate(alpha=-1.0)
    # Disable the alpha gate → should pass through to next gates
    cfg = DecisionChainConfig(
        enabled_gates=tuple(g for g in GATE_ORDER if g != "alpha_threshold")
    )
    trace = run_decision_chain(cand, DecisionContext(), cfg)
    gate_names = [g.gate_name for g in trace.gate_results]
    assert "alpha_threshold" not in gate_names
    assert trace.final_decision == "eligible"  # nothing else can block


# ---------------------------------------------------------------------------
# Batch + frame
# ---------------------------------------------------------------------------

def test_batch_returns_one_trace_per_candidate():
    candidates = [_candidate(symbol=f"60000{i}.SH", alpha=0.5) for i in range(5)]
    traces = run_decision_chain_batch(candidates, DecisionContext())
    assert len(traces) == 5
    assert all(t.final_decision == "eligible" for t in traces)


def test_traces_to_frame_long_form():
    cand1 = _candidate(symbol="A.SH", alpha=0.5)
    cand2 = _candidate(symbol="B.SH", alpha=-1.0)  # fails alpha
    traces = run_decision_chain_batch([cand1, cand2], DecisionContext())
    df = traces_to_frame(traces)
    # All gates run for A (no early reject) + 1 row for B (short-circuit).
    assert (df["symbol"] == "A.SH").sum() == len(GATE_ORDER)
    assert (df["symbol"] == "B.SH").sum() == 1
    assert {"candidate_id", "gate_name", "gate_passed", "gate_reason"}.issubset(df.columns)


# ---------------------------------------------------------------------------
# Trace serialisation
# ---------------------------------------------------------------------------

def test_trace_to_dict_serialisable():
    import json
    cand = _candidate(alpha=-1.0)
    trace = run_decision_chain(cand, DecisionContext())
    payload = trace.to_dict()
    # Round-trips through JSON
    json.dumps(payload)
    assert payload["final_decision"] == "rejected"
    assert payload["failed_gate"] == "alpha_threshold"
    assert payload["symbol"] == "600000.SH"


# ---------------------------------------------------------------------------
# 14 gate count
# ---------------------------------------------------------------------------

def test_chain_has_expected_gate_count():
    """Stage 5.3 baseline = 14 gates; Stage 5.5 adds gross_exposure_budget."""
    assert len(GATE_ORDER) == 15
    assert "gross_exposure_budget" in GATE_ORDER


# ---------------------------------------------------------------------------
# Gross exposure budget gate (spec section 7)
# ---------------------------------------------------------------------------

def test_exposure_budget_allows_buy_under_default_cap():
    cand = _candidate(weight=0.02)
    ctx = DecisionContext(current_gross_exposure=0.50, global_conviction=0.40)
    trace = run_decision_chain(cand, ctx)
    assert trace.final_decision == "eligible"
    gates = {g.gate_name: g for g in trace.gate_results}
    assert gates["gross_exposure_budget"].passed
    assert gates["gross_exposure_budget"].reason == "within_default_cap"


def test_exposure_budget_rejects_buy_pushing_above_60pct_without_conviction():
    cand = _candidate(weight=0.02)
    ctx = DecisionContext(
        current_gross_exposure=0.59,   # +0.02 → 0.61 > 60%
        global_conviction=0.40,         # below high-conviction floor
    )
    trace = run_decision_chain(cand, ctx)
    assert trace.final_decision == "rejected"
    assert trace.failed_gate == "gross_exposure_budget"


def test_exposure_budget_allows_extension_to_80_with_conviction_and_normal_regime():
    cand = _candidate(weight=0.02)
    regime = pd.Series(
        ["normal"], index=[pd.Timestamp("2024-03-01")], name="regime"
    )
    ctx = DecisionContext(
        current_gross_exposure=0.65,   # +0.02 → 0.67 in the 60–80% band
        global_conviction=0.90,
        regime_state=regime,
    )
    trace = run_decision_chain(cand, ctx)
    assert trace.final_decision == "eligible"
    gates = {g.gate_name: g for g in trace.gate_results}
    assert gates["gross_exposure_budget"].reason == "high_conviction_extension_allowed"


def test_exposure_budget_rejects_extension_in_bear_regime_even_with_conviction():
    # bear regime allows "lowbuy" setup only; use that so regime_alignment passes.
    cand = _candidate(weight=0.02, setup="lowbuy")
    regime = pd.Series(
        ["bear"], index=[pd.Timestamp("2024-03-01")], name="regime"
    )
    ctx = DecisionContext(
        current_gross_exposure=0.65,
        global_conviction=0.95,
        regime_state=regime,
    )
    trace = run_decision_chain(cand, ctx)
    assert trace.failed_gate == "gross_exposure_budget"


def test_exposure_budget_rejects_anything_above_80():
    # weight stays within max_name_weight (0.03) so it gets to the budget gate.
    cand = _candidate(weight=0.025)
    regime = pd.Series(
        ["normal"], index=[pd.Timestamp("2024-03-01")], name="regime"
    )
    ctx = DecisionContext(
        current_gross_exposure=0.79,   # +0.025 → 0.815 > 80% hard cap
        global_conviction=0.95,
        regime_state=regime,
    )
    trace = run_decision_chain(cand, ctx)
    assert trace.failed_gate == "gross_exposure_budget"


def test_exposure_budget_passes_sells_regardless_of_exposure():
    cand = _candidate(weight=-0.02)  # sell
    ctx = DecisionContext(current_gross_exposure=0.95, global_conviction=0.10)
    trace = run_decision_chain(cand, ctx)
    gates = {g.gate_name: g for g in trace.gate_results}
    assert gates["gross_exposure_budget"].passed
    assert gates["gross_exposure_budget"].reason == "passed_sell_or_trim"
