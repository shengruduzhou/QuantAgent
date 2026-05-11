from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quantagent.portfolio.position_state import StopDecision
from quantagent.risk.kill_switch import KillSwitch


@dataclass(frozen=True)
class KillSwitchLimits:
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    max_position_breach: float = 0.12
    max_gross_exposure: float = 1.0
    max_strategy_loss_pct: float = 0.05
    max_reject_rate: float = 0.30
    stale_data_minutes: int = 10


@dataclass(frozen=True)
class KillSwitchVerdict:
    triggered: bool
    reason: str


def evaluate_kill_switch(
    daily_pnl_pct: float,
    rolling_drawdown_pct: float,
    max_single_position_weight: float,
    gross_exposure: float,
    limits: KillSwitchLimits | None = None,
) -> KillSwitchVerdict:
    """Compatibility wrapper around the unified V6 KillSwitch."""
    limits = limits or KillSwitchLimits()
    switch = KillSwitch()
    switch.evaluate(
        daily_loss=daily_pnl_pct,
        drawdown=rolling_drawdown_pct,
        turnover=0.0,
        max_daily_loss=limits.max_daily_loss_pct,
        max_drawdown=limits.max_drawdown_pct,
    )
    if max_single_position_weight > limits.max_position_breach:
        return KillSwitchVerdict(True, f"position_breach:{max_single_position_weight:.4f}")
    if gross_exposure > limits.max_gross_exposure:
        return KillSwitchVerdict(True, f"gross_exposure_breach:{gross_exposure:.4f}")
    if switch.triggered:
        return KillSwitchVerdict(True, switch.reasons[0])
    return KillSwitchVerdict(False, "ok")


def evaluate_stop_loss_kill_switch(
    stop_decisions: list[StopDecision],
    max_blocked_exits: int = 3,
    max_hard_stops: int = 5,
) -> KillSwitchVerdict:
    blocked = sum(decision.blocked_exit for decision in stop_decisions)
    hard = sum(decision.status.value == "hard_stop" for decision in stop_decisions)
    if blocked >= max_blocked_exits:
        return KillSwitchVerdict(True, f"blocked_exit_cluster:{blocked}")
    if hard >= max_hard_stops:
        return KillSwitchVerdict(True, f"hard_stop_cluster:{hard}")
    return KillSwitchVerdict(False, "ok")


def evaluate_v4_kill_switch(
    daily_pnl_pct: float = 0.0,
    rolling_drawdown_pct: float = 0.0,
    strategy_loss_pct: float = 0.0,
    gross_exposure: float = 0.0,
    broker_connected: bool = True,
    market_halted: bool = False,
    stale_data: bool = False,
    reject_rate: float = 0.0,
    manual_lock_file: str | None = None,
    limits: KillSwitchLimits | None = None,
) -> KillSwitchVerdict:
    limits = limits or KillSwitchLimits()
    if manual_lock_file and Path(manual_lock_file).exists():
        return KillSwitchVerdict(True, "manual_lock_file")
    if not broker_connected:
        return KillSwitchVerdict(True, "broker_disconnected")
    if market_halted:
        return KillSwitchVerdict(True, "market_halted")
    if stale_data:
        return KillSwitchVerdict(True, "stale_data")
    if reject_rate > limits.max_reject_rate:
        switch = KillSwitch()
        switch.evaluate(rejection_rate=reject_rate, max_rejection_rate=limits.max_reject_rate)
        return KillSwitchVerdict(True, switch.reasons[0] if switch.reasons else f"abnormal_reject_rate:{reject_rate:.4f}")
    if strategy_loss_pct <= -limits.max_strategy_loss_pct:
        return KillSwitchVerdict(True, f"strategy_loss_breached:{strategy_loss_pct:.4f}")
    base = evaluate_kill_switch(daily_pnl_pct, rolling_drawdown_pct, 0.0, gross_exposure, limits)
    return base
