"""Cash, positions and P&L for the local paper portfolio.

Every number here is *derived* from the ledger, never stored as authoritative
mutable state. That is deliberate: a cached balance that can drift from its own
history is how a reconciliation passes while the book is wrong.

The T+1 model is the part most simulators get wrong. A-share shares bought today
cannot be sold today, so a position carries ``sellable`` (settled, disposable
now) separately from ``total`` (everything held). The daily settlement step
promotes today's purchases into sellable — and nothing else may.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Iterable, Mapping

from quantagent.domain.accounting import UnpriceablePosition
from quantagent.paper.orders import BUY, SELL, Fill


class InsufficientCash(RuntimeError):
    """Raised when a buy would overdraw the paper account."""


class InsufficientSellable(RuntimeError):
    """Raised when a sell exceeds the T+1-settled position."""


@dataclass
class Position:
    symbol: str
    total: float = 0.0
    #: Settled shares that may be sold today. Always <= total under T+1.
    sellable: float = 0.0
    #: Shares acquired today; become sellable at the next settlement.
    pending_settlement: float = 0.0
    average_cost: float = 0.0
    realised_pnl: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.total) < 1e-9

    def market_value(self, price: float) -> float:
        return self.total * price

    def unrealised_pnl(self, price: float) -> float:
        return (price - self.average_cost) * self.total

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Portfolio:
    portfolio_id: str
    cash: float
    initial_cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    realised_pnl: float = 0.0
    fees_paid: float = 0.0

    # -- queries ----------------------------------------------------------
    def position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position(symbol=symbol))

    def sellable(self, symbol: str) -> float:
        return self.positions[symbol].sellable if symbol in self.positions else 0.0

    def unpriceable(self, prices: Mapping[str, float]) -> tuple[str, ...]:
        """Held symbols with no mark. The delisted-and-still-held case."""
        return tuple(
            sorted(
                symbol for symbol, position in self.positions.items()
                if not position.is_flat and symbol not in prices
            )
        )

    def market_value(self, prices: Mapping[str, float]) -> float:
        """Raises when a held position has no mark.

        The `if s in prices` filter this replaces silently *excluded* unpriced
        positions, which claims the holding does not exist — the same class of
        defect as marking it at zero, and equally invisible because the resulting
        equity figure looks perfectly ordinary (DEF-021).
        """
        missing = self.unpriceable(prices)
        if missing:
            raise UnpriceablePosition(missing)
        return sum(
            p.market_value(prices[s]) for s, p in self.positions.items()
            if not p.is_flat
        )

    def equity(self, prices: Mapping[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def unrealised_pnl(self, prices: Mapping[str, float]) -> float:
        missing = self.unpriceable(prices)
        if missing:
            raise UnpriceablePosition(missing)
        return sum(
            p.unrealised_pnl(prices[s]) for s, p in self.positions.items()
            if not p.is_flat
        )

    def gross_exposure(self, prices: Mapping[str, float]) -> float:
        missing = self.unpriceable(prices)
        if missing:
            raise UnpriceablePosition(missing)
        return sum(
            abs(p.market_value(prices[s])) for s, p in self.positions.items()
            if not p.is_flat
        )

    # -- mutation (only ever driven by a Fill) ----------------------------
    def can_afford(self, fill: Fill) -> bool:
        return fill.side != BUY or (self.cash + fill.cash_delta) >= -1e-9

    def apply_fill(self, fill: Fill) -> None:
        """Apply a fill, enforcing cash and T+1 sellable constraints."""
        position = self.position(fill.symbol)

        if fill.side == BUY:
            if not self.can_afford(fill):
                raise InsufficientCash(
                    f"buy of {fill.quantity:.0f} {fill.symbol} needs "
                    f"{-fill.cash_delta:.2f} but only {self.cash:.2f} is available"
                )
            new_total = position.total + fill.quantity
            # Average cost includes fees: excluding them understates the entry
            # price and flatters every subsequent P&L number.
            position.average_cost = (
                (position.average_cost * position.total + fill.notional + fill.fees)
                / new_total
            ) if new_total else 0.0
            position.total = new_total
            # T+1: today's purchase is NOT sellable until settlement.
            position.pending_settlement += fill.quantity
        else:
            if fill.quantity - position.sellable > 1e-9:
                raise InsufficientSellable(
                    f"sell of {fill.quantity:.0f} {fill.symbol} exceeds the "
                    f"T+1-settled {position.sellable:.0f}"
                )
            realised = (fill.price - position.average_cost) * fill.quantity - fill.fees
            position.realised_pnl += realised
            self.realised_pnl += realised
            position.total -= fill.quantity
            position.sellable -= fill.quantity
            if position.is_flat:
                position.total = 0.0
                position.sellable = 0.0
                position.average_cost = 0.0

        self.cash += fill.cash_delta
        self.fees_paid += fill.fees

    def settle(self) -> dict[str, float]:
        """Promote pending purchases to sellable. The only path that may do so."""
        settled: dict[str, float] = {}
        for symbol, position in self.positions.items():
            if position.pending_settlement:
                settled[symbol] = position.pending_settlement
                position.sellable += position.pending_settlement
                position.pending_settlement = 0.0
        return settled

    def apply_corporate_action(
        self, symbol: str, *, share_ratio: float = 1.0, cash_per_share: float = 0.0
    ) -> dict[str, Any]:
        """Apply a split/bonus/dividend to a held position.

        Share count scales and average cost scales inversely, so the position's
        book value is unchanged by a pure share adjustment -- the alternative
        silently manufactures or destroys P&L on the ex-date.
        """
        position = self.position(symbol)
        if position.is_flat:
            return {"symbol": symbol, "applied": False, "reason": "no position"}

        cash_received = position.total * cash_per_share
        before = position.total
        position.total *= share_ratio
        position.sellable *= share_ratio
        position.pending_settlement *= share_ratio
        if share_ratio != 1.0:
            position.average_cost /= share_ratio
        self.cash += cash_received
        # Dividend income is realised return, not a free cash increase. On the ex
        # date the mark drops by the dividend, so cash rises and market value falls
        # by the same amount; booking the cash without the income left realised PnL
        # understated by exactly the dividend, and disagreeing with the canonical
        # replay that does book it (DEF-020).
        position.realised_pnl += cash_received
        self.realised_pnl += cash_received
        return {
            "symbol": symbol, "applied": True, "share_ratio": share_ratio,
            "shares_before": before, "shares_after": position.total,
            "cash_received": cash_received,
        }

    def to_dict(self, prices: Mapping[str, float] | None = None) -> dict[str, Any]:
        """Snapshot. Valuation fields are None when a holding cannot be marked."""
        prices = prices or {}
        unpriceable = self.unpriceable(prices)
        valuation: dict[str, Any] = (
            {"market_value": None, "equity": None, "unrealised_pnl": None}
            if unpriceable
            else {
                "market_value": self.market_value(prices),
                "equity": self.equity(prices),
                "unrealised_pnl": self.unrealised_pnl(prices),
            }
        )
        return {
            "portfolio_id": self.portfolio_id,
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "realised_pnl": self.realised_pnl,
            "fees_paid": self.fees_paid,
            "positions": {s: p.to_dict() for s, p in self.positions.items()
                          if not p.is_flat},
            "unpriceable_symbols": list(unpriceable),
        } | valuation
