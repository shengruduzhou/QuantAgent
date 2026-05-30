"""PositionPolicy tests (spec section 7)."""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.portfolio.position_policy import (
    HeldPosition,
    PositionCandidate,
    PositionClass,
    PositionPolicy,
    PositionPolicyConfig,
    compute_consecutive_limit_up_count,
)


def _cand(**kwargs) -> PositionCandidate:
    defaults = dict(
        symbol="600000.SH",
        proposed_weight=0.02,
        proposed_class=PositionClass.MID,
        confidence=0.70,
        consecutive_limit_up_count=0,
        is_st=False,
        is_suspended=False,
        is_one_word_board=False,
        is_t_zero_sell=False,
    )
    defaults.update(kwargs)
    return PositionCandidate(**defaults)


def _held(**kwargs) -> HeldPosition:
    defaults = dict(
        symbol="600000.SH",
        weight=0.05,
        position_class=PositionClass.MID,
        cost_basis=10.0,
        open_date=pd.Timestamp("2024-02-01"),
        available_shares=1000,
        same_day_acquired=0,
    )
    defaults.update(kwargs)
    return HeldPosition(**defaults)


# ---------------------------------------------------------------------------
# Hard exclusions
# ---------------------------------------------------------------------------

def test_st_candidate_blocked_by_default():
    policy = PositionPolicy()
    verdict = policy.evaluate_candidate(_cand(is_st=True))
    assert not verdict.allowed
    rules = {v.rule for v in verdict.violations}
    assert "block_st" in rules


def test_suspended_candidate_blocked_by_default():
    policy = PositionPolicy()
    verdict = policy.evaluate_candidate(_cand(is_suspended=True))
    assert not verdict.allowed


def test_one_word_board_blocks_buy_only_not_sell():
    policy = PositionPolicy()
    buy_blocked = policy.evaluate_candidate(_cand(is_one_word_board=True, proposed_weight=0.02))
    sell_allowed = policy.evaluate_candidate(_cand(is_one_word_board=True, proposed_weight=-0.02))
    assert not buy_blocked.allowed
    assert "block_one_word_board" in {v.rule for v in buy_blocked.violations}
    # sell side may still have other issues but block_one_word_board should not fire
    rules = {v.rule for v in sell_allowed.violations}
    assert "block_one_word_board" not in rules


# ---------------------------------------------------------------------------
# Limit-up chase
# ---------------------------------------------------------------------------

def test_limit_up_chase_block_at_threshold():
    policy = PositionPolicy()
    verdict = policy.evaluate_candidate(
        _cand(consecutive_limit_up_count=3, proposed_weight=0.02)
    )
    assert not verdict.allowed
    assert "limit_up_chase" in {v.rule for v in verdict.violations}


def test_limit_up_chase_does_not_block_sells():
    policy = PositionPolicy()
    verdict = policy.evaluate_candidate(
        _cand(consecutive_limit_up_count=5, proposed_weight=-0.02)
    )
    assert "limit_up_chase" not in {v.rule for v in verdict.violations}


# ---------------------------------------------------------------------------
# T+0 enforcement
# ---------------------------------------------------------------------------

def test_t0_sell_blocked_when_same_day_acquired_shares_exist():
    policy = PositionPolicy()
    verdict = policy.evaluate_candidate(
        _cand(is_t_zero_sell=True, proposed_weight=-0.02),
        held=[_held(same_day_acquired=500)],
    )
    assert not verdict.allowed
    assert "t_plus_one_violation" in {v.rule for v in verdict.violations}


def test_t0_sell_allowed_when_only_legacy_shares_held():
    policy = PositionPolicy()
    verdict = policy.evaluate_candidate(
        _cand(is_t_zero_sell=True, proposed_weight=-0.02),
        held=[_held(same_day_acquired=0, available_shares=1000)],
    )
    assert "t_plus_one_violation" not in {v.rule for v in verdict.violations}


# ---------------------------------------------------------------------------
# Cross-class transition
# ---------------------------------------------------------------------------

def test_transition_allowed_with_high_confidence():
    policy = PositionPolicy(PositionPolicyConfig(transition_min_confidence=0.60))
    held = [_held(position_class=PositionClass.SHORT)]
    verdict = policy.evaluate_candidate(
        _cand(proposed_class=PositionClass.MID, confidence=0.80),
        held=held,
    )
    assert verdict.allowed


def test_transition_blocked_with_low_confidence():
    policy = PositionPolicy(PositionPolicyConfig(transition_min_confidence=0.70))
    held = [_held(position_class=PositionClass.SHORT)]
    verdict = policy.evaluate_candidate(
        _cand(proposed_class=PositionClass.MID, confidence=0.50),
        held=held,
    )
    assert not verdict.allowed
    assert "class_transition_low_confidence" in {v.rule for v in verdict.violations}


def test_same_class_does_not_trigger_transition_rule():
    policy = PositionPolicy(PositionPolicyConfig(transition_min_confidence=0.99))
    held = [_held(position_class=PositionClass.MID)]
    verdict = policy.evaluate_candidate(
        _cand(proposed_class=PositionClass.MID, confidence=0.10),
        held=held,
    )
    # confidence 0.1 < 0.99 but classes match → no transition rule fires
    assert "class_transition_low_confidence" not in {v.rule for v in verdict.violations}


# ---------------------------------------------------------------------------
# Gross exposure cap
# ---------------------------------------------------------------------------

def test_default_60pct_cap_enforced_without_high_conviction():
    policy = PositionPolicy()
    held = [_held(weight=0.59)]
    verdict = policy.evaluate_candidate(
        _cand(proposed_weight=0.03), held=held,
        global_conviction=0.40, regime="normal",
    )
    assert not verdict.allowed
    assert "gross_exposure_cap" in {v.rule for v in verdict.violations}


def test_high_conviction_can_extend_to_80pct_in_friendly_regime():
    policy = PositionPolicy()
    held = [_held(weight=0.65)]
    verdict = policy.evaluate_candidate(
        _cand(proposed_weight=0.05), held=held,
        global_conviction=0.85, regime="normal",
    )
    assert verdict.allowed


def test_no_extension_in_bear_regime_even_with_high_conviction():
    policy = PositionPolicy()
    held = [_held(weight=0.65)]
    verdict = policy.evaluate_candidate(
        _cand(proposed_weight=0.05), held=held,
        global_conviction=0.95, regime="bear_capitulation",
    )
    assert not verdict.allowed


def test_cap_above_80_always_blocks():
    policy = PositionPolicy()
    held = [_held(weight=0.78)]
    verdict = policy.evaluate_candidate(
        _cand(proposed_weight=0.05), held=held,
        global_conviction=0.95, regime="normal",
    )
    assert not verdict.allowed


# ---------------------------------------------------------------------------
# Cash buffer
# ---------------------------------------------------------------------------

def test_low_cash_buffer_warns_but_does_not_block():
    # buffer threshold 0.30 → fail when gross ≥ 0.70; cap 0.80 → still passes cap
    policy = PositionPolicy(PositionPolicyConfig(min_cash_buffer=0.30))
    held = [_held(weight=0.72)]
    verdict = policy.evaluate_candidate(
        _cand(proposed_weight=0.005), held=held,
        global_conviction=0.95, regime="normal",
    )
    rules = {v.rule for v in verdict.violations}
    assert "min_cash_buffer" in rules
    # warn-only — the gross_exposure_cap is the blocker if any
    cash_buf_violation = next(v for v in verdict.violations if v.rule == "min_cash_buffer")
    assert cash_buf_violation.severity == "warn"
    # ensure no blocking violation
    blocks = [v for v in verdict.violations if v.severity == "block"]
    assert blocks == []
    assert verdict.allowed


# ---------------------------------------------------------------------------
# Per-name cap
# ---------------------------------------------------------------------------

def test_per_name_cap_blocks_oversized_candidate():
    policy = PositionPolicy(PositionPolicyConfig(max_position_per_name=0.05))
    verdict = policy.evaluate_candidate(_cand(proposed_weight=0.10))
    assert not verdict.allowed
    assert "max_position_per_name" in {v.rule for v in verdict.violations}


# ---------------------------------------------------------------------------
# Consecutive limit-up counter
# ---------------------------------------------------------------------------

def test_compute_consecutive_limit_up_counts_until_break():
    dates = pd.bdate_range("2024-03-01", periods=6)
    # A.SH: 4 consecutive limit-ups ending today
    # B.SH: 2 limit-ups then a non-limit-up
    # C.SH: no limit-up
    rows = [
        {"trade_date": d, "symbol": "A.SH", "daily_return": 0.10}
        for d in dates[2:6]
    ] + [
        {"trade_date": dates[0], "symbol": "A.SH", "daily_return": 0.02},
        {"trade_date": dates[1], "symbol": "A.SH", "daily_return": 0.03},
    ] + [
        {"trade_date": dates[0], "symbol": "B.SH", "daily_return": 0.10},
        {"trade_date": dates[1], "symbol": "B.SH", "daily_return": 0.10},
        {"trade_date": dates[2], "symbol": "B.SH", "daily_return": 0.02},
        {"trade_date": dates[3], "symbol": "B.SH", "daily_return": 0.10},
        {"trade_date": dates[4], "symbol": "B.SH", "daily_return": 0.10},
        {"trade_date": dates[5], "symbol": "B.SH", "daily_return": 0.02},  # break
    ] + [
        {"trade_date": d, "symbol": "C.SH", "daily_return": 0.005}
        for d in dates
    ]
    counts = compute_consecutive_limit_up_count(
        pd.DataFrame(rows), as_of_date=dates[5]
    )
    assert counts["A.SH"] == 4
    assert counts["B.SH"] == 0   # breaks immediately on as_of date
    assert counts["C.SH"] == 0


def test_batch_evaluation_returns_one_verdict_per_candidate():
    policy = PositionPolicy()
    cands = [_cand(symbol=f"60000{i}.SH", proposed_weight=0.02) for i in range(3)]
    verdicts = policy.evaluate_batch(cands)
    assert len(verdicts) == 3
    assert all(v.allowed for v in verdicts)
