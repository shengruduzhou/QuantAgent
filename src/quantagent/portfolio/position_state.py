from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PositionStatus(str, Enum):
    NEW = "new"
    NORMAL = "normal"
    PROFIT_PROTECT = "profit_protect"
    PULLBACK_HOLD = "pullback_hold"
    BREAKEVEN_EXIT = "breakeven_exit"
    SOFT_STOP = "soft_stop"
    HARD_STOP = "hard_stop"
    TIME_STOP = "time_stop"
    EVENT_STOP = "event_stop"
    LIQUIDITY_EXIT = "liquidity_exit"
    EXITED = "exited"


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    entry_price: float
    current_price: float
    highest_price: float
    holding_days: int
    atr: float
    volatility: float
    flow_score: float
    regime_score: float
    event_risk_score: float
    fundamental_risk_score: float
    liquidity_score: float
    current_drawdown: float
    expected_alpha_remaining: float
    transaction_cost: float
    sellable_today: bool = True
    is_limit_down: bool = False
    status: PositionStatus = PositionStatus.NORMAL


@dataclass(frozen=True)
class StopLossConfig:
    hard_stop_pct: float = 0.08
    hard_stop_atr_multiple: float = 2.5
    soft_stop_pct: float = 0.05
    breakeven_profit_threshold: float = 0.06
    trailing_stop_pct: float = 0.10
    trailing_activation_profit: float = 0.10
    max_holding_days: int = 20
    event_risk_threshold: float = 0.8
    fundamental_risk_threshold: float = 0.85
    liquidity_exit_threshold: float = 0.2
    alpha_remaining_threshold: float = 0.0


@dataclass(frozen=True)
class StopDecision:
    symbol: str
    status: PositionStatus
    should_exit: bool
    blocked_exit: bool
    reason: str
    risk_penalty: float


def evaluate_position_state(snapshot: PositionSnapshot, config: StopLossConfig | None = None) -> StopDecision:
    config = config or StopLossConfig()
    if snapshot.entry_price <= 0 or snapshot.current_price <= 0:
        return StopDecision(snapshot.symbol, PositionStatus.NORMAL, False, False, "invalid_price", 0.0)

    pnl = snapshot.current_price / snapshot.entry_price - 1.0
    peak_profit = snapshot.highest_price / snapshot.entry_price - 1.0 if snapshot.highest_price > 0 else pnl
    drawdown_from_peak = snapshot.current_price / max(snapshot.highest_price, snapshot.current_price) - 1.0
    atr_loss = (snapshot.entry_price - snapshot.current_price) / max(snapshot.atr, 1e-12)

    status = PositionStatus.NORMAL
    should_exit = False
    reason = "hold"
    risk_penalty = 0.0

    if snapshot.event_risk_score >= config.event_risk_threshold or snapshot.fundamental_risk_score >= config.fundamental_risk_threshold:
        status = PositionStatus.EVENT_STOP
        should_exit = True
        reason = "event_or_fundamental_risk"
        risk_penalty = max(snapshot.event_risk_score, snapshot.fundamental_risk_score)
    elif pnl <= -config.hard_stop_pct or atr_loss >= config.hard_stop_atr_multiple:
        status = PositionStatus.HARD_STOP
        should_exit = True
        reason = "hard_stop"
        risk_penalty = min(1.0, abs(pnl) / max(config.hard_stop_pct, 1e-12))
    elif snapshot.liquidity_score <= config.liquidity_exit_threshold:
        status = PositionStatus.LIQUIDITY_EXIT
        should_exit = True
        reason = "liquidity_exit"
        risk_penalty = 1.0 - snapshot.liquidity_score
    elif peak_profit >= config.breakeven_profit_threshold and pnl <= snapshot.transaction_cost:
        status = PositionStatus.BREAKEVEN_EXIT
        should_exit = True
        reason = "breakeven_stop"
        risk_penalty = 0.5
    elif peak_profit >= config.trailing_activation_profit and drawdown_from_peak <= -config.trailing_stop_pct:
        status = PositionStatus.PROFIT_PROTECT
        should_exit = True
        reason = "trailing_stop"
        risk_penalty = min(1.0, abs(drawdown_from_peak) / max(config.trailing_stop_pct, 1e-12))
    elif snapshot.holding_days >= config.max_holding_days and snapshot.expected_alpha_remaining <= config.alpha_remaining_threshold:
        status = PositionStatus.TIME_STOP
        should_exit = True
        reason = "time_stop"
        risk_penalty = 0.4
    elif pnl <= -config.soft_stop_pct and snapshot.flow_score < 0 and snapshot.regime_score < 0:
        status = PositionStatus.SOFT_STOP
        should_exit = True
        reason = "soft_stop_confirmed_by_flow_and_regime"
        risk_penalty = min(1.0, abs(pnl) / max(config.soft_stop_pct, 1e-12))
    elif peak_profit >= config.trailing_activation_profit and drawdown_from_peak < 0:
        status = PositionStatus.PULLBACK_HOLD
        reason = "profit_pullback_without_stop"
        risk_penalty = min(0.5, abs(drawdown_from_peak))
    elif peak_profit >= config.breakeven_profit_threshold:
        status = PositionStatus.PROFIT_PROTECT
        reason = "profit_protection_active"
        risk_penalty = 0.1

    blocked_exit = bool(should_exit and (snapshot.is_limit_down or not snapshot.sellable_today))
    if blocked_exit:
        reason = f"{reason}; exit_blocked_by_limit_down_or_t_plus_one"
    return StopDecision(
        symbol=snapshot.symbol,
        status=status,
        should_exit=should_exit and not blocked_exit,
        blocked_exit=blocked_exit,
        reason=reason,
        risk_penalty=float(max(0.0, min(1.0, risk_penalty))),
    )

