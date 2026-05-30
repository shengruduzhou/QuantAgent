"""Tests for the position state machine.

Each test pins a single transition rule from the spec to a concrete
``PositionContext`` and asserts the resulting ``PositionDecision``.
"""

from __future__ import annotations

import pytest

from quantagent.portfolio.state_machine import (
    PositionContext,
    PositionState,
    PositionStateMachine,
    StateMachineConfig,
)


def _ctx(**overrides):
    base = dict(
        symbol="A.SZ",
        trade_date="2024-01-02",
        current_state=PositionState.WATCH,
        current_weight=0.0,
        entry_weight=0.0,
        entry_prediction=None,
        current_prediction=None,
        pred_quantile=None,
        days_held=0,
        unrealized_return=None,
        unrealized_drawdown=None,
        is_suspended=False,
        is_st=False,
        is_limit_up_at_close=False,
        is_high_chase=False,
        market_regime="normal",
    )
    base.update(overrides)
    return PositionContext(**base)


def test_ban_stays_banned_while_st_active():
    """After review fix #5 BAN persists only while is_st (or
    is_suspended) is still True. The OLD permanent-BAN behavior was
    a bug — this test pins the new conditional semantics."""
    sm = PositionStateMachine()
    decision = sm.transition(_ctx(current_state=PositionState.BAN, is_st=True, pred_quantile=0.99))
    assert decision.target_state == PositionState.BAN
    assert decision.weight_action == "skip"


def test_suspended_with_position_holds_existing():
    sm = PositionStateMachine()
    decision = sm.transition(_ctx(current_weight=0.05, days_held=3, is_suspended=True))
    assert decision.target_state == PositionState.HOLD_SHORT
    assert decision.weight_action == "hold"


def test_suspended_no_position_stays_watch():
    sm = PositionStateMachine()
    decision = sm.transition(_ctx(current_weight=0.0, is_suspended=True))
    assert decision.target_state == PositionState.WATCH
    assert decision.weight_action == "skip"


def test_st_with_position_exits():
    sm = PositionStateMachine()
    decision = sm.transition(_ctx(current_weight=0.03, is_st=True, days_held=10))
    assert decision.target_state == PositionState.EXIT
    assert decision.weight_action == "exit"


def test_st_no_position_is_banned():
    sm = PositionStateMachine()
    decision = sm.transition(_ctx(current_weight=0.0, is_st=True))
    assert decision.target_state == PositionState.BAN


def test_stop_loss_when_dd_exceeds_threshold():
    sm = PositionStateMachine()
    decision = sm.transition(_ctx(
        current_state=PositionState.HOLD_SHORT,
        current_weight=0.03,
        days_held=4,
        unrealized_drawdown=-0.10,
    ))
    assert decision.target_state == PositionState.STOP_LOSS
    assert decision.weight_action == "exit"


def test_age_promotes_short_to_mid_to_long():
    sm = PositionStateMachine(StateMachineConfig(
        short_hold_max_days=5,
        mid_hold_max_days=30,
        long_hold_max_days=120,
    ))
    short_d = sm.transition(_ctx(current_weight=0.03, days_held=3))
    mid_d = sm.transition(_ctx(current_weight=0.03, days_held=15))
    long_d = sm.transition(_ctx(current_weight=0.03, days_held=60))
    assert short_d.target_state == PositionState.HOLD_SHORT
    assert mid_d.target_state == PositionState.HOLD_MID
    assert long_d.target_state == PositionState.HOLD_LONG


def test_age_beyond_long_triggers_take_profit():
    sm = PositionStateMachine(StateMachineConfig(long_hold_max_days=120))
    decision = sm.transition(_ctx(current_weight=0.03, days_held=200))
    assert decision.target_state == PositionState.TAKE_PROFIT
    assert decision.weight_action == "exit"


def test_pred_drop_take_profit_when_in_profit():
    sm = PositionStateMachine(StateMachineConfig(take_profit_pred_drop=0.40))
    decision = sm.transition(_ctx(
        current_weight=0.03,
        entry_prediction=1.0,
        current_prediction=0.3,  # 70% drop
        unrealized_return=0.05,  # in profit
        days_held=10,
    ))
    assert decision.target_state == PositionState.TAKE_PROFIT


def test_pred_drop_reduce_when_below_take_profit_threshold():
    sm = PositionStateMachine(StateMachineConfig(reduce_pred_drop=0.20, take_profit_pred_drop=0.50))
    decision = sm.transition(_ctx(
        current_weight=0.03,
        entry_prediction=1.0,
        current_prediction=0.7,  # 30% drop
        unrealized_return=0.02,
        days_held=10,
    ))
    assert decision.target_state == PositionState.REDUCE
    assert decision.weight_action == "reduce"
    assert decision.target_weight_multiplier == 0.5


def test_do_t_when_pred_in_top_quantile_and_holding():
    sm = PositionStateMachine(StateMachineConfig(do_t_pred_threshold=0.95))
    decision = sm.transition(_ctx(
        current_weight=0.03,
        days_held=5,
        pred_quantile=0.98,
    ))
    assert decision.target_state == PositionState.HOLD_SHORT
    assert decision.weight_action == "do_t"


def test_low_buy_ready_when_no_position_and_top_quantile():
    sm = PositionStateMachine(StateMachineConfig(low_buy_pred_quantile=0.10))
    decision = sm.transition(_ctx(
        current_weight=0.0,
        pred_quantile=0.95,  # top 5% → above 1-0.10=0.90 threshold
    ))
    assert decision.target_state == PositionState.LOW_BUY_READY
    assert decision.weight_action == "buy"


def test_no_entry_in_crisis_regime():
    sm = PositionStateMachine()
    decision = sm.transition(_ctx(
        current_weight=0.0,
        pred_quantile=0.99,
        market_regime="crisis",
    ))
    assert decision.target_state == PositionState.WATCH
    assert decision.weight_action == "skip"


def test_no_entry_when_high_chase():
    sm = PositionStateMachine()
    decision = sm.transition(_ctx(
        current_weight=0.0,
        pred_quantile=0.99,
        is_high_chase=True,
    ))
    assert decision.target_state == PositionState.WATCH


def test_no_entry_when_limit_up_at_close():
    sm = PositionStateMachine()
    decision = sm.transition(_ctx(
        current_weight=0.0,
        pred_quantile=0.99,
        is_limit_up_at_close=True,
    ))
    assert decision.target_state == PositionState.WATCH


def test_stop_loss_takes_precedence_over_state():
    """Even if days_held says HOLD_LONG, stop-loss kicks first."""
    sm = PositionStateMachine(StateMachineConfig(stop_loss_unrealized_dd=0.08))
    decision = sm.transition(_ctx(
        current_state=PositionState.HOLD_LONG,
        current_weight=0.03,
        days_held=80,
        unrealized_drawdown=-0.12,
    ))
    assert decision.target_state == PositionState.STOP_LOSS


def test_low_buy_below_quantile_threshold_stays_watch():
    sm = PositionStateMachine(StateMachineConfig(low_buy_pred_quantile=0.10))
    decision = sm.transition(_ctx(current_weight=0.0, pred_quantile=0.50))
    assert decision.target_state == PositionState.WATCH


def test_ban_clears_when_st_flag_removed():
    """Review fix #5: BAN should not be permanent. When the cause
    (is_st or is_suspended) clears, the state machine must return to
    WATCH so future signals can be acted on."""
    sm = PositionStateMachine()
    # ST flag has cleared but state still says BAN — should re-evaluate.
    decision = sm.transition(_ctx(
        current_state=PositionState.BAN,
        is_st=False,
        is_suspended=False,
        pred_quantile=0.99,
    ))
    assert decision.target_state == PositionState.WATCH


def test_ban_persists_while_st_flag_active():
    """Review fix #5 — sanity: BAN does still stick while ST is True."""
    sm = PositionStateMachine()
    decision = sm.transition(_ctx(
        current_state=PositionState.BAN,
        is_st=True,
        pred_quantile=0.99,
    ))
    assert decision.target_state == PositionState.BAN


def test_negative_entry_prediction_handled_without_skip():
    """Review fix #6: negative entry_prediction must not silently
    skip deterioration checks. The fraction drop is computed against
    |entry| so the take-profit / reduce logic still works."""
    sm = PositionStateMachine(StateMachineConfig(reduce_pred_drop=0.20))
    decision = sm.transition(_ctx(
        current_weight=0.03,
        entry_prediction=-0.10,   # negative entry
        current_prediction=-0.40, # got even more negative
        unrealized_return=-0.02,
        days_held=5,
    ))
    # drop = (-0.10 - -0.40) / 0.10 = 3.0 → REDUCE (and would TAKE_PROFIT if in profit)
    assert decision.target_state == PositionState.REDUCE
