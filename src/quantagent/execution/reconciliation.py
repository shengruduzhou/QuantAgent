from __future__ import annotations

from dataclasses import dataclass

from quantagent.execution.broker_base import Position


@dataclass(frozen=True)
class BrokerSnapshot:
    positions: tuple[Position, ...]
    account_value: float
    connected: bool = True


@dataclass(frozen=True)
class ReconciliationResult:
    passed: bool
    reason: str
    snapshot: BrokerSnapshot


def reconcile_broker_state(snapshot: BrokerSnapshot, min_account_value: float = 0.0) -> ReconciliationResult:
    if not snapshot.connected:
        return ReconciliationResult(False, "broker_disconnected", snapshot)
    if snapshot.account_value < min_account_value:
        return ReconciliationResult(False, "account_value_below_minimum", snapshot)
    return ReconciliationResult(True, "ok", snapshot)
