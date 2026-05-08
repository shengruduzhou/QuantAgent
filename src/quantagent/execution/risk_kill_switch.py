from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KillSwitchLimits:
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    max_position_breach: float = 0.12
    max_gross_exposure: float = 1.0


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
    """Hard cutoff: if any threshold breaches, stop new orders for the day."""
    limits = limits or KillSwitchLimits()
    if daily_pnl_pct <= -limits.max_daily_loss_pct:
        return KillSwitchVerdict(True, f"daily_loss_breached:{daily_pnl_pct:.4f}")
    if rolling_drawdown_pct <= -limits.max_drawdown_pct:
        return KillSwitchVerdict(True, f"drawdown_breached:{rolling_drawdown_pct:.4f}")
    if max_single_position_weight > limits.max_position_breach:
        return KillSwitchVerdict(True, f"position_breach:{max_single_position_weight:.4f}")
    if gross_exposure > limits.max_gross_exposure:
        return KillSwitchVerdict(True, f"gross_exposure_breach:{gross_exposure:.4f}")
    return KillSwitchVerdict(False, "ok")
