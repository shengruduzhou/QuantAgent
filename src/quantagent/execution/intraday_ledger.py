"""T+1 inventory ledger for A-share intraday Do-T research.

The ledger is deliberately small and explicit.  It tracks only the quantities
that matter for legal intraday round trips:

* carried shares are yesterday-settled inventory and can be sold today;
* today buys increase the economic position but are not sellable today;
* every open Do-T pair is closed through the opposite leg or marked as an EOD
  restore event, never silently counted as a successful trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quantagent.execution.broker_base import OrderSide


@dataclass
class IntradayPair:
    pair_id: str
    action: str
    quantity: int
    price: float
    time: str = ""
    cost_paid: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LedgerEvent:
    event_type: str
    action: str
    quantity: int
    price: float
    time: str = ""
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    cost: float = 0.0
    message: str = ""


@dataclass
class IntradayLedger:
    symbol: str
    date: str

    carried_shares: int
    target_shares: int
    cash: float

    today_sold: int = 0
    today_bought: int = 0

    open_sell_pairs: list[IntradayPair] = field(default_factory=list)
    open_buy_pairs: list[IntradayPair] = field(default_factory=list)

    realized_gross_pnl: float = 0.0
    realized_net_pnl: float = 0.0
    cost_paid: float = 0.0
    slippage_paid: float = 0.0
    events: list[LedgerEvent] = field(default_factory=list)

    @property
    def sellable_shares(self) -> int:
        return max(0, int(self.carried_shares) - int(self.today_sold))

    @property
    def current_position(self) -> int:
        return int(self.carried_shares) - int(self.today_sold) + int(self.today_bought)

    @property
    def position_gap_to_target(self) -> int:
        return self.current_position - int(self.target_shares)

    def can_sell(self, quantity: int) -> bool:
        return int(quantity) > 0 and int(quantity) <= self.sellable_shares

    def can_buy(self, quantity: int, price: float, cost: float = 0.0) -> bool:
        notional = int(quantity) * float(price)
        return int(quantity) > 0 and float(price) > 0 and self.cash + 1e-9 >= notional + float(cost)

    def open_sell_high(
        self,
        *,
        pair_id: str,
        quantity: int,
        price: float,
        time: str = "",
        cost: float = 0.0,
        slippage: float = 0.0,
    ) -> LedgerEvent:
        """Sell the first leg of a reverse-T pair from carried sellable shares."""
        quantity = int(quantity)
        if not self.can_sell(quantity):
            raise ValueError("SELL_HIGH violates T+1 sellable inventory")
        self.today_sold += quantity
        self.cash += quantity * float(price) - float(cost)
        self.cost_paid += float(cost)
        self.slippage_paid += float(slippage)
        self.open_sell_pairs.append(
            IntradayPair(pair_id, "SELL_HIGH", quantity, float(price), time, float(cost))
        )
        event = LedgerEvent("OPEN_PAIR", "SELL_HIGH", quantity, float(price), time, cost=float(cost))
        self.events.append(event)
        return event

    def close_sell_pair_buyback(
        self,
        *,
        quantity: int,
        price: float,
        time: str = "",
        cost: float = 0.0,
        slippage: float = 0.0,
    ) -> LedgerEvent:
        """Buy back against an existing SELL_HIGH pair.

        The buy creates today's bought shares.  They restore the economic
        position but remain unsellable until the next session.
        """
        quantity = int(quantity)
        if quantity <= 0 or not self.open_sell_pairs:
            raise ValueError("BUY_BACK requires an open sell pair")
        remaining = quantity
        gross = 0.0
        matched_cost = 0.0
        while remaining > 0 and self.open_sell_pairs:
            pair = self.open_sell_pairs[0]
            take = min(remaining, pair.quantity)
            gross += (pair.price - float(price)) * take
            matched_cost += pair.cost_paid * take / max(pair.quantity, 1)
            pair.quantity -= take
            remaining -= take
            if pair.quantity <= 0:
                self.open_sell_pairs.pop(0)
        if remaining > 0:
            raise ValueError("BUY_BACK quantity exceeds open sell pairs")
        if not self.can_buy(quantity, price, cost):
            raise ValueError("BUY_BACK violates cash constraint")
        self.today_bought += quantity
        self.cash -= quantity * float(price) + float(cost)
        self.cost_paid += float(cost)
        self.slippage_paid += float(slippage)
        net = gross - matched_cost - float(cost)
        self.realized_gross_pnl += gross
        self.realized_net_pnl += net
        event = LedgerEvent("CLOSE_PAIR", "BUY_BACK", quantity, float(price), time, gross, net, float(cost))
        self.events.append(event)
        return event

    def open_buy_low(
        self,
        *,
        pair_id: str,
        quantity: int,
        price: float,
        time: str = "",
        cost: float = 0.0,
        slippage: float = 0.0,
    ) -> LedgerEvent:
        """Open a positive-T pair by buying low.

        The later SELL_AFTER_BUY leg must still sell carried shares, not the
        shares bought by this method.
        """
        quantity = int(quantity)
        if not self.can_buy(quantity, price, cost):
            raise ValueError("BUY_LOW violates cash constraint")
        self.today_bought += quantity
        self.cash -= quantity * float(price) + float(cost)
        self.cost_paid += float(cost)
        self.slippage_paid += float(slippage)
        self.open_buy_pairs.append(
            IntradayPair(pair_id, "BUY_LOW", quantity, float(price), time, float(cost))
        )
        event = LedgerEvent("OPEN_PAIR", "BUY_LOW", quantity, float(price), time, cost=float(cost))
        self.events.append(event)
        return event

    def close_buy_pair_sell_after_buy(
        self,
        *,
        quantity: int,
        price: float,
        time: str = "",
        cost: float = 0.0,
        slippage: float = 0.0,
    ) -> LedgerEvent:
        """Sell carried shares to close a BUY_LOW pair and restore target exposure."""
        quantity = int(quantity)
        if not self.can_sell(quantity):
            raise ValueError("SELL_AFTER_BUY violates T+1 sellable inventory")
        if quantity <= 0 or not self.open_buy_pairs:
            raise ValueError("SELL_AFTER_BUY requires an open buy pair")
        remaining = quantity
        gross = 0.0
        matched_cost = 0.0
        while remaining > 0 and self.open_buy_pairs:
            pair = self.open_buy_pairs[0]
            take = min(remaining, pair.quantity)
            gross += (float(price) - pair.price) * take
            matched_cost += pair.cost_paid * take / max(pair.quantity, 1)
            pair.quantity -= take
            remaining -= take
            if pair.quantity <= 0:
                self.open_buy_pairs.pop(0)
        if remaining > 0:
            raise ValueError("SELL_AFTER_BUY quantity exceeds open buy pairs")
        self.today_sold += quantity
        self.cash += quantity * float(price) - float(cost)
        self.cost_paid += float(cost)
        self.slippage_paid += float(slippage)
        net = gross - matched_cost - float(cost)
        self.realized_gross_pnl += gross
        self.realized_net_pnl += net
        event = LedgerEvent(
            "CLOSE_PAIR", "SELL_AFTER_BUY", quantity, float(price), time, gross, net, float(cost)
        )
        self.events.append(event)
        return event

    def mark_eod_restore(self, *, price: float, time: str = "", cost: float = 0.0) -> LedgerEvent | None:
        """Record an end-of-day restore event when the position misses target."""
        gap = self.position_gap_to_target
        if gap == 0 and not self.open_sell_pairs and not self.open_buy_pairs:
            return None
        action = "EOD_RESTORE_BUY" if gap < 0 or self.open_sell_pairs else "EOD_RESTORE_SELL"
        quantity = abs(gap)
        if quantity == 0:
            quantity = sum(p.quantity for p in self.open_sell_pairs + self.open_buy_pairs)
        self.cost_paid += float(cost)
        event = LedgerEvent(
            "EOD_RESTORE",
            action,
            int(quantity),
            float(price),
            time,
            net_pnl=-float(cost),
            cost=float(cost),
            message="open pair did not close intraday; restore is a risk event",
        )
        self.open_sell_pairs.clear()
        self.open_buy_pairs.clear()
        self.events.append(event)
        return event


def side_from_action(action: str) -> OrderSide:
    if action in {"BUY_BACK", "BUY_LOW", "EOD_RESTORE_BUY"}:
        return OrderSide.BUY
    if action in {"SELL_HIGH", "SELL_AFTER_BUY", "EOD_RESTORE_SELL"}:
        return OrderSide.SELL
    raise ValueError(f"Unsupported intraday action: {action}")


__all__ = [
    "IntradayLedger",
    "IntradayPair",
    "LedgerEvent",
    "side_from_action",
]
