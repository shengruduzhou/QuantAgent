"""Reconstruct paper state by replaying the event ledger.

There is no snapshot. Cash, positions, open orders, fills, realised and
unrealised P&L, and the kill-switch state are all rebuilt from events, which is
the only way a recovery can be *verified*: if state were stored separately, a
successful recovery would only prove the snapshot was readable, not that it
agreed with what actually happened.

Recovery refuses to proceed on a broken chain. A ledger that fails verification
describes a history that was edited or truncated, and rebuilding a portfolio
from it would produce confident numbers with no provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from quantagent.paper import ledger as lg
from quantagent.paper.orders import (
    CANCELLED,
    FILLED,
    PARTIALLY_FILLED,
    REJECTED,
    Fill,
    Order,
)
from quantagent.paper.portfolio import Portfolio, Position


class RecoveryRefused(RuntimeError):
    """Raised when the ledger cannot be trusted enough to rebuild from."""


@dataclass
class RecoveredState:
    portfolio: Portfolio
    orders: dict[str, Order] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    killed: bool = False
    kill_reason: str | None = None
    events_replayed: int = 0
    sessions_closed: int = 0
    chain_valid: bool = True

    def open_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if o.is_open]

    def to_dict(self, prices: Mapping[str, float] | None = None) -> dict[str, Any]:
        return {
            "portfolio": self.portfolio.to_dict(prices or {}),
            "open_orders": [o.to_dict() for o in self.open_orders()],
            "orders_total": len(self.orders),
            "fills": len(self.fills),
            "killed": self.killed,
            "kill_reason": self.kill_reason,
            "events_replayed": self.events_replayed,
            "sessions_closed": self.sessions_closed,
            "chain_valid": self.chain_valid,
        }


def recover(
    event_ledger: lg.EventLedger,
    *,
    portfolio_id: str,
    initial_cash: float,
    require_valid_chain: bool = True,
) -> RecoveredState:
    """Rebuild full paper state from the ledger.

    ``initial_cash`` seeds the replay; every subsequent change comes from
    events, so a mismatch between the replayed cash and the last CASH_CHANGED
    event is detectable rather than papered over.
    """
    verification = event_ledger.verify()
    if require_valid_chain and not verification["valid"]:
        raise RecoveryRefused(
            f"ledger chain verification failed: {verification['error']}. "
            "Rebuilding from an edited or truncated history would produce "
            "confident numbers with no provenance."
        )

    portfolio = Portfolio(
        portfolio_id=portfolio_id, cash=initial_cash, initial_cash=initial_cash
    )
    state = RecoveredState(portfolio=portfolio, chain_valid=verification["valid"])

    for event in event_ledger.read():
        state.events_replayed += 1
        payload = event.payload

        if event.event_type == lg.ORDER_CREATED:
            order = _order_from_payload(payload.get("order", {}))
            if order is not None:
                state.orders[order.order_id] = order

        elif event.event_type == lg.ORDER_ACCEPTED:
            order = state.orders.get(payload.get("order_id", ""))
            if order is not None and order.state == "NEW":
                order.transition("ACCEPTED")
                if "quantity" in payload:
                    order.quantity = float(payload["quantity"])

        elif event.event_type == lg.ORDER_REJECTED:
            order = state.orders.get(
                (payload.get("order") or {}).get("order_id", "")
            )
            if order is not None and order.is_open:
                order.reject_reason = payload.get("reason")
                if order.state == "NEW":
                    order.transition(REJECTED)

        elif event.event_type in (lg.ORDER_FILLED, lg.ORDER_PARTIALLY_FILLED):
            fill = _fill_from_payload(payload.get("fill", {}))
            if fill is None:
                continue
            state.fills.append(fill)
            portfolio.apply_fill(fill)
            order = state.orders.get(fill.order_id)
            if order is not None:
                order.filled_quantity += fill.quantity
                order.filled_notional += fill.notional
                order.fees_paid += fill.fees
                target = FILLED if event.event_type == lg.ORDER_FILLED else PARTIALLY_FILLED
                if target in _allowed_from(order.state):
                    order.transition(target)

        elif event.event_type == lg.ORDER_CANCELLED:
            order = state.orders.get(payload.get("order_id", ""))
            if order is not None and order.is_open:
                if "CANCEL_REQUESTED" in _allowed_from(order.state):
                    order.transition("CANCEL_REQUESTED")
                if CANCELLED in _allowed_from(order.state):
                    order.transition(CANCELLED)

        elif event.event_type == lg.CORPORATE_ACTION_APPLIED and payload.get("applied"):
            portfolio.apply_corporate_action(
                payload["symbol"],
                share_ratio=float(payload.get("share_ratio", 1.0)),
                cash_per_share=0.0,  # cash already recorded in the payload
            )
            portfolio.cash += float(payload.get("cash_received", 0.0))

        elif event.event_type == lg.SESSION_CLOSED:
            portfolio.settle()
            state.sessions_closed += 1

        elif event.event_type == lg.KILL_SWITCH_TRIGGERED:
            state.killed = True
            state.kill_reason = payload.get("reason")

    return state


def _allowed_from(current: str) -> frozenset[str]:
    from quantagent.paper.orders import ALLOWED_TRANSITIONS

    return ALLOWED_TRANSITIONS.get(current, frozenset())


def _order_from_payload(payload: Mapping[str, Any]) -> Order | None:
    if not payload.get("order_id"):
        return None
    try:
        return Order(
            symbol=payload["symbol"], side=payload["side"],
            quantity=float(payload["quantity"]),
            order_type=payload.get("order_type", "LIMIT"),
            limit_price=payload.get("limit_price"),
            board=payload.get("board", "SH_Main"),
            order_id=payload["order_id"],
            parent_id=payload.get("parent_id"),
            strategy_id=payload.get("strategy_id"),
            is_full_liquidation=bool(payload.get("is_full_liquidation", False)),
        )
    except (KeyError, ValueError):
        return None


def _fill_from_payload(payload: Mapping[str, Any]) -> Fill | None:
    if not payload.get("order_id"):
        return None
    try:
        return Fill(
            order_id=payload["order_id"], symbol=payload["symbol"],
            side=payload["side"], quantity=float(payload["quantity"]),
            price=float(payload["price"]), notional=float(payload["notional"]),
            commission=float(payload.get("commission", 0.0)),
            stamp_duty=float(payload.get("stamp_duty", 0.0)),
            transfer_fee=float(payload.get("transfer_fee", 0.0)),
            market_time=payload.get("market_time"),
            fill_id=payload.get("fill_id", ""),
            partial=bool(payload.get("partial", False)),
        )
    except (KeyError, ValueError):
        return None


def reconcile(live: Any, recovered: RecoveredState, *,
              tolerance: float = 1e-6) -> dict[str, Any]:
    """Compare a running broker's state against a fresh ledger replay.

    This is the check that makes the ledger meaningful: if the two disagree,
    either an action bypassed the ledger or the replay is wrong, and both are
    incidents rather than rounding.
    """
    problems: list[str] = []

    if abs(live.portfolio.cash - recovered.portfolio.cash) > tolerance:
        problems.append(
            f"cash mismatch: live {live.portfolio.cash:.6f} vs replay "
            f"{recovered.portfolio.cash:.6f}"
        )

    live_positions = {s: p.total for s, p in live.portfolio.positions.items()
                      if not p.is_flat}
    replay_positions = {s: p.total for s, p in recovered.portfolio.positions.items()
                        if not p.is_flat}
    for symbol in set(live_positions) | set(replay_positions):
        a, b = live_positions.get(symbol, 0.0), replay_positions.get(symbol, 0.0)
        if abs(a - b) > tolerance:
            problems.append(f"position mismatch {symbol}: live {a} vs replay {b}")

    if len(live.fills) != len(recovered.fills):
        problems.append(
            f"fill count mismatch: live {len(live.fills)} vs replay "
            f"{len(recovered.fills)}"
        )

    return {
        "passed": not problems,
        "problems": problems,
        "live_cash": live.portfolio.cash,
        "replayed_cash": recovered.portfolio.cash,
        "live_fills": len(live.fills),
        "replayed_fills": len(recovered.fills),
        "chain_valid": recovered.chain_valid,
    }
