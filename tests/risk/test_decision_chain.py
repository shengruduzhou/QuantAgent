"""Unit tests for the v8 15-gate decision chain."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.risk.decision_chain import (
    DecisionChainConfig,
    GATE_NAMES,
    run_decision_chain,
)


def _make_inputs(n_dates: int = 5, n_symbols: int = 50, seed: int = 7):
    dates = pd.date_range("2024-01-02", periods=n_dates, freq="B")
    symbols = [f"S{i:03d}.SZ" for i in range(n_symbols)]
    rng = np.random.default_rng(seed)
    composite = pd.DataFrame(
        [
            {"trade_date": d, "symbol": s, "composite_score": rng.normal()}
            for d in dates
            for s in symbols
        ]
    )
    panel = pd.DataFrame(
        [
            {
                "trade_date": d, "symbol": s,
                "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
                "volume": 1e7, "amount": 1e9,
                "is_suspended": (s == "S001.SZ"),
                "is_st": (s == "S002.SZ"),
                "is_limit_up": (s == "S003.SZ"),
                "is_limit_down": False,
            }
            for d in dates for s in symbols
        ]
    )
    sector_map = pd.DataFrame(
        [{"symbol": s, "sector_level_1": f"sector_{i % 5}"} for i, s in enumerate(symbols)]
    )
    return composite, panel, sector_map


def test_gate_names_are_16():
    assert len(GATE_NAMES) == 16
    assert "kill_switch" in GATE_NAMES
    assert "is_st" in GATE_NAMES
    assert "limit_up_no_buy" in GATE_NAMES
    assert "sector_concentration" in GATE_NAMES
    assert "old_dealer_block" in GATE_NAMES


def test_basic_chain_accepts_top_k_per_day():
    composite, panel, sector_map = _make_inputs(n_dates=5, n_symbols=50)
    cfg = DecisionChainConfig(top_k=10, min_avg_amount_yuan=0.0)
    res = run_decision_chain(composite=composite, market_panel=panel, sector_map=sector_map, config=cfg)
    # 5 days × 10 expected, but 3 syms can be rejected (suspended / ST / limit-up) per day
    assert res.summary["n_dates"] == 5
    assert res.summary["n_accepted"] >= 30
    assert res.summary["n_accepted"] <= 50
    # The three structurally-rejected symbols should never be in the accepted set
    accepted = res.target_weights.columns
    assert "S001.SZ" not in accepted or (res.target_weights["S001.SZ"] == 0.0).all()
    assert "S002.SZ" not in accepted or (res.target_weights["S002.SZ"] == 0.0).all()


def test_old_dealer_risk_blocks_candidate():
    d = pd.Timestamp("2024-01-02")
    composite = pd.DataFrame([
        {"trade_date": d, "symbol": "OLD.SZ", "composite_score": 2.0},
        {"trade_date": d, "symbol": "OK.SZ", "composite_score": 1.0},
    ])
    panel = pd.DataFrame([
        {"trade_date": d, "symbol": "OLD.SZ",
         "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0,
         "volume": 1e7, "amount": 1e9,
         "old_dealer_risk_score": 0.9, "old_dealer_block": True,
         "is_suspended": False, "is_st": False, "is_limit_up": False, "is_limit_down": False},
        {"trade_date": d, "symbol": "OK.SZ",
         "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0,
         "volume": 1e7, "amount": 1e9,
         "old_dealer_risk_score": 0.1, "old_dealer_block": False,
         "is_suspended": False, "is_st": False, "is_limit_up": False, "is_limit_down": False},
    ])
    cfg = DecisionChainConfig(top_k=2, min_avg_amount_yuan=0.0, old_dealer_risk_max=0.7)

    res = run_decision_chain(composite=composite, market_panel=panel, config=cfg)

    gates = res.decision_traces.set_index("symbol")["rejected_gate"].to_dict()
    assert gates["OLD.SZ"] == "old_dealer_block"
    assert gates["OK.SZ"] is None


def test_unknown_sector_does_not_collapse_portfolio():
    """When the sector_map is all-NaN, the sector_concentration gate
    must not reject all but the first 6 names — every unknown symbol
    should get its own bucket so the cap effectively becomes a no-op
    for unknown sectors.
    """
    composite, panel, _ = _make_inputs(n_dates=3, n_symbols=30)
    sector_map = pd.DataFrame(
        [{"symbol": f"S{i:03d}.SZ", "sector_level_1": None} for i in range(30)]
    )
    cfg = DecisionChainConfig(top_k=15, min_avg_amount_yuan=0.0, max_sector_weight=0.20)
    res = run_decision_chain(composite=composite, market_panel=panel, sector_map=sector_map, config=cfg)
    # With unknown sectors treated independently we should reach near-top_k per day
    assert res.summary["n_accepted"] >= 30  # 3 days × ~10+ per day


def test_consecutive_limit_up_blocks_after_cap():
    """A name that prints limit-up two days running should be blocked
    on day 3 by the consecutive_limit_up_cap gate.
    """
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    symbol = "S999.SZ"
    composite = pd.DataFrame(
        [{"trade_date": d, "symbol": symbol, "composite_score": 1.0} for d in dates]
    )
    panel = pd.DataFrame(
        [
            {"trade_date": dates[0], "symbol": symbol,
             "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
             "volume": 1e7, "amount": 1e9,
             "is_suspended": False, "is_st": False, "is_limit_up": True, "is_limit_down": False},
            {"trade_date": dates[1], "symbol": symbol,
             "open": 11.0, "high": 11.5, "low": 10.5, "close": 11.0,
             "volume": 1e7, "amount": 1e9,
             "is_suspended": False, "is_st": False, "is_limit_up": True, "is_limit_down": False},
            {"trade_date": dates[2], "symbol": symbol,
             "open": 12.0, "high": 12.5, "low": 11.5, "close": 12.0,
             "volume": 1e7, "amount": 1e9,
             "is_suspended": False, "is_st": False, "is_limit_up": False, "is_limit_down": False},
            {"trade_date": dates[3], "symbol": symbol,
             "open": 13.0, "high": 13.5, "low": 12.5, "close": 13.0,
             "volume": 1e7, "amount": 1e9,
             "is_suspended": False, "is_st": False, "is_limit_up": False, "is_limit_down": False},
        ]
    )
    # New behaviour: a *regular* limit-up (with intraday range) is fillable
    # at a small capped position, not hard-blocked. The consecutive-streak
    # gate still fires on day 3 after two prior limit-ups.
    cfg = DecisionChainConfig(top_k=5, min_avg_amount_yuan=0.0, max_consecutive_limit_up=2)
    res = run_decision_chain(composite=composite, market_panel=panel, config=cfg)
    gates = res.decision_traces.set_index("trade_date")["rejected_gate"].to_dict()
    # Day 0: regular limit-up accepted (small position) → no rejection
    assert gates[dates[0]] is None
    # Day 1: prior streak 1 < 2 → still accepted at cap
    assert gates[dates[1]] is None
    # Day 2: prior streak 2 ≥ 2 → blocked by consecutive_limit_up_cap
    assert gates[dates[2]] == "consecutive_limit_up_cap"
    # The accepted limit-up positions are capped at limit_up_position_cap
    accepted = res.decision_traces[res.decision_traces["accepted"]]
    assert (accepted["weight"] <= cfg.limit_up_position_cap + 1e-9).all()
    assert bool(accepted["limit_up_capped"].iloc[0])


def test_one_word_limit_up_is_blocked_but_regular_is_capped():
    """一字板 (high==low==close, no range) is unfillable → blocked.
    A regular limit-up (intraday range present) is accepted at a small cap.
    """
    d = pd.Timestamp("2024-01-02")
    composite = pd.DataFrame([
        {"trade_date": d, "symbol": "ONEWORD.SZ", "composite_score": 2.0},
        {"trade_date": d, "symbol": "REGULAR.SZ", "composite_score": 1.0},
    ])
    panel = pd.DataFrame([
        # one-word board: high == low == close (no intraday range)
        {"trade_date": d, "symbol": "ONEWORD.SZ",
         "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0,
         "volume": 1e6, "amount": 1e8,
         "is_suspended": False, "is_st": False, "is_limit_up": True, "is_limit_down": False},
        # regular limit-up: has intraday range
        {"trade_date": d, "symbol": "REGULAR.SZ",
         "open": 10.2, "high": 11.0, "low": 10.0, "close": 11.0,
         "volume": 1e7, "amount": 1e9,
         "is_suspended": False, "is_st": False, "is_limit_up": True, "is_limit_down": False},
    ])
    cfg = DecisionChainConfig(top_k=5, min_avg_amount_yuan=0.0)
    res = run_decision_chain(composite=composite, market_panel=panel, config=cfg)
    gates = res.decision_traces.set_index("symbol")["rejected_gate"].to_dict()
    assert gates["ONEWORD.SZ"] == "one_word_limit_up_no_buy"
    assert gates["REGULAR.SZ"] is None  # accepted at small position


def test_two_stage_pool_selects_at_most_top_k():
    """candidate_pool_size caps the pool; final selection ≤ top_k."""
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    symbols = [f"P{i:03d}.SZ" for i in range(60)]
    rng = np.random.default_rng(11)
    composite = pd.DataFrame([
        {"trade_date": dt, "symbol": s, "composite_score": rng.normal()}
        for dt in dates for s in symbols
    ])
    panel = pd.DataFrame([
        {"trade_date": dt, "symbol": s,
         "open": 10.0, "high": 10.4, "low": 9.8, "close": 10.1,
         "volume": 1e7, "amount": 1e9,
         "is_suspended": False, "is_st": False, "is_limit_up": False, "is_limit_down": False}
        for dt in dates for s in symbols
    ])
    cfg = DecisionChainConfig(top_k=10, candidate_pool_size=40, min_avg_amount_yuan=0.0)
    res = run_decision_chain(composite=composite, market_panel=panel, config=cfg)
    per_day = res.decision_traces[res.decision_traces["accepted"]].groupby("trade_date").size()
    assert (per_day <= 10).all()


def test_kill_switch_rejects_entire_day():
    composite, panel, sector_map = _make_inputs(n_dates=2, n_symbols=10)
    from quantagent.risk.kill_switch import KillSwitch

    ks = KillSwitch()
    ks.trigger("test_failure")
    cfg = DecisionChainConfig(top_k=5, min_avg_amount_yuan=0.0)
    res = run_decision_chain(
        composite=composite, market_panel=panel, sector_map=sector_map,
        config=cfg, kill_switch=ks,
    )
    assert res.summary["n_accepted"] == 0
    assert any(
        evt["event_type"] == "kill_switch_triggered"
        for evt in res.risk_events
    )


def test_market_regime_scales_down_in_bear():
    """牛市满仓 / 熊市空仓: after MA warmup, position scale collapses in a
    sustained downtrend with narrow breadth (crisis → 0)."""
    from quantagent.risk.decision_chain import (
        _compute_avg_amount, _compute_trend_quality, _compute_market_regime,
    )
    dates = pd.date_range("2023-01-02", periods=150, freq="B")
    syms = [f"S{i:02d}.SZ" for i in range(40)]
    rows = []
    for di, d in enumerate(dates):
        base = 10 * (1.006 ** di) if di < 75 else 10 * (1.006 ** 75) * (0.992 ** (di - 75))
        for i, s in enumerate(syms):
            px = base * (1 + 0.01 * ((i % 5) - 2))
            rows.append({"trade_date": d, "symbol": s, "open": px, "high": px * 1.005,
                         "low": px * 0.995, "close": px, "volume": 1e7, "amount": 1e9,
                         "is_limit_up": False})
    panel = _compute_trend_quality(_compute_avg_amount(pd.DataFrame(rows)))
    reg = _compute_market_regime(panel, config=DecisionChainConfig(regime_position_scaling=True)).dropna()
    warm = reg.iloc[60:]
    bull = warm[warm.index < dates[75]]["position_scale"].mean()
    bear = warm[warm.index >= dates[80]]["position_scale"].mean()
    assert bull > 0.8       # near full in the uptrend
    assert bear < 0.3       # near cash in the downtrend
    assert bull > bear + 0.3
