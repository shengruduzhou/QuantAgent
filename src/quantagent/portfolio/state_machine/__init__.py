"""Per-position state machine for short / mid / long horizon trading.

Implements the user's v4 spec §7-8 state-machine: each symbol carries a
PositionState (WATCH / LOW_BUY_READY / OPEN_POSITION / HOLD_SHORT/MID/LONG
/ DO_T / REDUCE / TAKE_PROFIT / STOP_LOSS / EXIT / BAN) and transitions
between them based on prediction / current weight / age / drawdown /
regime signals.

The machine itself is pure (no I/O, no global state); callers supply
``PositionContext`` per (date, symbol) and receive a ``PositionDecision``
that names the target state and recommended weight action.
"""

from quantagent.portfolio.state_machine.machine import (
    PositionContext,
    PositionDecision,
    PositionState,
    PositionStateMachine,
    StateMachineConfig,
)

__all__ = [
    "PositionContext",
    "PositionDecision",
    "PositionState",
    "PositionStateMachine",
    "StateMachineConfig",
]
