"""Reconstruct paper state by replaying the event ledger.

There is no snapshot. Cash, positions, open orders, fills, realised and
unrealised P&L, and the kill-switch state are rebuilt from events, which is
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
    TERMINAL_STATES,
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
    """Rebuild full paper state from the legacy operational ledger."""
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
                cash_per_share=0.0,
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
    """Compare a running broker's state against a fresh ledger replay."""
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


def recover_from_canonical(
    canonical_ledger_path: str,
    *,
    portfolio_id: str,
    initial_cash: float,
    as_of_trade_date: str | None = None,
) -> RecoveredState:
    """Rebuild paper state from the canonical ledger — the record of account.

    ``as_of_trade_date`` is the market session for which the mutable paper view
    will be used. Sellability must be evaluated against that session, not against
    the latest date already present in the ledger: after a restart, Monday's buy
    is sellable on Tuesday even when no Tuesday economic event has been appended
    yet. The canonical lots keep their acquisition dates, so this is derived
    state rather than a settlement snapshot.
    """
    from quantagent.domain.ledger import CanonicalLedger

    ledger = CanonicalLedger(canonical_ledger_path)
    verification = ledger.verify()
    if not verification["valid"]:
        raise RecoveryRefused(
            f"canonical chain verification failed at record {verification.get('brokenAt')}. "
            "Rebuilding from an edited or truncated history would produce "
            "confident numbers with no provenance."
        )

    book, account = ledger.replay(initial_cash=initial_cash)
    portfolio = Portfolio(
        portfolio_id=portfolio_id, cash=account.cash, initial_cash=initial_cash
    )
    state = RecoveredState(portfolio=portfolio, chain_valid=True)
    state.events_replayed = len(ledger)

    sessions = [r.trade_date for r in ledger.read() if r.trade_date]
    latest_session = max(sessions) if sessions else ""
    valuation_session = str(as_of_trade_date or latest_session)
    if latest_session and valuation_session and valuation_session < latest_session:
        raise RecoveryRefused(
            f"cannot recover canonical account as of {valuation_session}: "
            f"ledger already contains later session {latest_session}"
        )

    for symbol, lots in account.lots.items():
        total = float(sum(lot.quantity for lot in lots))
        if total <= 0:
            continue
        sellable = (
            float(account.sellable(symbol, valuation_session))
            if valuation_session
            else 0.0
        )
        cost = account.cost_basis.get(symbol, 0.0)
        portfolio.positions[symbol] = Position(
            symbol=symbol,
            total=total,
            sellable=sellable,
            pending_settlement=total - sellable,
            average_cost=float(cost),
            realised_pnl=0.0,
        )
    portfolio.realised_pnl = float(account.realised_pnl)
    portfolio.fees_paid = float(account.total_fees)

    for canonical in book.orders():
        order = Order(
            symbol=canonical.symbol,
            side=canonical.side.value,
            quantity=float(canonical.quantity),
            limit_price=canonical.limit_price or 1.0,
            order_id=canonical.order_id,
        )
        order.state = _PAPER_STATE_OF[canonical.status]
        order.filled_quantity = float(canonical.cumulative_quantity)
        order.filled_notional = sum(f.quantity * f.price for f in canonical.fills)
        order.fees_paid = canonical.total_fees
        order.reject_reason = canonical.reason
        state.orders[order.order_id] = order

    state.fills = [
        Fill(
            order_id=fill.order_id, symbol=fill.symbol, side=fill.side.value,
            quantity=float(fill.quantity), price=float(fill.price),
            notional=float(fill.gross), commission=float(fill.commission),
            stamp_duty=float(fill.stamp_duty), transfer_fee=float(fill.transfer_fee),
            market_time=fill.filled_at, fill_id=fill.execution_id,
        )
        for fill in book.fills()
    ]
    return state


#: Canonical status -> paper's vocabulary, the inverse of paper.orders._TO_CANONICAL.
_PAPER_STATE_OF: dict[Any, str] = {}


def _build_state_map() -> None:
    from quantagent.paper.orders import _TO_CANONICAL

    _PAPER_STATE_OF.update({status: name for name, status in _TO_CANONICAL.items()})


_build_state_map()
