from quantagent.portfolio.position_state import PositionSnapshot, PositionStatus, StopLossConfig, evaluate_position_state


def _snapshot(**overrides) -> PositionSnapshot:
    values = dict(
        symbol="A",
        entry_price=10.0,
        current_price=10.5,
        highest_price=11.0,
        holding_days=5,
        atr=0.5,
        volatility=0.2,
        flow_score=0.2,
        regime_score=0.2,
        event_risk_score=0.1,
        fundamental_risk_score=0.1,
        liquidity_score=0.8,
        current_drawdown=-0.02,
        expected_alpha_remaining=0.02,
        transaction_cost=0.002,
        sellable_today=True,
        is_limit_down=False,
    )
    values.update(overrides)
    return PositionSnapshot(**values)


def test_price_recovers_to_breakeven_triggers_exit():
    decision = evaluate_position_state(_snapshot(current_price=10.01, highest_price=10.8))
    assert decision.status == PositionStatus.BREAKEVEN_EXIT
    assert decision.should_exit


def test_trailing_stop_triggers():
    decision = evaluate_position_state(_snapshot(current_price=10.7, highest_price=12.0))
    assert decision.status == PositionStatus.PROFIT_PROTECT
    assert decision.should_exit


def test_t_plus_one_prevents_same_day_sell():
    decision = evaluate_position_state(_snapshot(current_price=9.0, sellable_today=False))
    assert decision.blocked_exit
    assert not decision.should_exit


def test_limit_down_blocks_exit():
    decision = evaluate_position_state(_snapshot(current_price=9.0, is_limit_down=True))
    assert decision.blocked_exit


def test_time_stop_triggers():
    config = StopLossConfig(max_holding_days=10)
    decision = evaluate_position_state(_snapshot(holding_days=12, expected_alpha_remaining=-0.01), config)
    assert decision.status == PositionStatus.TIME_STOP


def test_event_stop_overrides_normal_hold():
    decision = evaluate_position_state(_snapshot(event_risk_score=0.95))
    assert decision.status == PositionStatus.EVENT_STOP
    assert decision.should_exit

