"""Economic state derived purely by folding canonical order events.

Deliberately absent: a stored `frozen_cash`. It used to be a field here with
`freeze_cash`/`release_cash` helpers and no production caller, so on any replayed
account it read 0.0 and every "reserved cash matches" comparison passed without
measuring anything (DEF-007). Cash committed to working buy orders is a *function
of the order book* — a working order is the commitment — and it is computed as
`reserved_cash` by `reconciliation.snapshot.EconomicSnapshot`. Storing the same
fact separately would be the second record of truth this module exists to remove.

Nothing here reads a broker, a dataframe or a cached balance. Cash, lots,
realised PnL and NAV are *functions of the event log*, which is what makes
ledger replay a real proof rather than a comparison of two hand-maintained
copies of the same numbers. If a value cannot be reconstructed from events, it
does not belong in this class.

Every mutation returns a new `AccountState`. The invariants in `check` run after
each applied event, so a violation is attributed to the event that caused it
rather than discovered later as an unexplained drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import (
    CorporateAction,
    Fill,
    OrderEvent,
    OrderEventType,
    PositionLot,
    Side,
    same_execution,
    sellable_quantity,
    total_quantity,
)


class UnpriceablePosition(RuntimeError):
    """A held position has no mark, so no NAV can be reported for it.

    Refused rather than defaulted, because both plausible defaults are specific
    and usually false claims. Marking at zero says the holding is worthless;
    excluding it from market value says the holding does not exist. Measured cost
    of the zero default (DEF-021): 1,000 shares carried at 10.0051 with no mark
    understated NAV by 10,000.00 and fabricated a 10,005.10 loss — and the
    accounting identity *still held*, because cash and the mark were consistently
    wrong together. That is what makes this class of defect so quiet: the internal
    consistency check passes.

    A delisted or long-suspended symbol is the case that matters. Its value is
    genuinely unknown, and `unknown` is the answer the programme requires — never
    zero, never a stale last close carried forward silently.
    """

    def __init__(self, symbols: Sequence[str]) -> None:
        listed = ", ".join(sorted(symbols))
        super().__init__(
            f"no mark for held position(s): {listed}. A missing price is unknown, not "
            "zero: marking at zero claims the holding is worthless and excluding it "
            "claims it does not exist. Supply a mark, or read `unpriceable()` and "
            "report NAV as unavailable."
        )
        self.symbols = tuple(sorted(symbols))


class InvariantViolation(RuntimeError):
    """An accounting identity that must always hold did not.

    Raised eagerly: a book that has already broken an invariant produces
    numbers nobody can trust, and every later figure inherits the error.
    """


@dataclass(frozen=True, slots=True)
class AccountState:
    """Cash, inventory and realised PnL, all folded from events."""

    cash: float
    initial_cash: float
    lots: Mapping[str, tuple[PositionLot, ...]] = field(default_factory=dict)
    realised_pnl: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    #: Weighted average cost per symbol, needed to realise PnL on a sell.
    cost_basis: Mapping[str, float] = field(default_factory=dict)
    #: Execution ids the log reported more than once and replay refused to apply
    #: twice. Provenance, not economics: it is deliberately absent from
    #: `content_hash`, because a ledger carrying a re-delivered callback and one
    #: carrying it once describe the *same* economic state.
    duplicate_executions: tuple[str, ...] = ()

    # -- construction -------------------------------------------------------
    @classmethod
    def opening(cls, initial_cash: float) -> "AccountState":
        return cls(cash=float(initial_cash), initial_cash=float(initial_cash))

    # -- queries ------------------------------------------------------------
    def position(self, symbol: str) -> int:
        return total_quantity(self.lots.get(symbol, ()))

    def sellable(self, symbol: str, trade_date: str) -> int:
        """T+1: only lots acquired on an earlier session may be sold."""
        return sellable_quantity(self.lots.get(symbol, ()), trade_date)

    def unpriceable(self, prices: Mapping[str, float]) -> tuple[str, ...]:
        """Held symbols with no mark. Empty when every position can be valued."""
        return tuple(
            sorted(
                symbol for symbol in self.lots
                if self.position(symbol) != 0 and symbol not in prices
            )
        )

    def market_value(self, prices: Mapping[str, float]) -> float:
        missing = self.unpriceable(prices)
        if missing:
            raise UnpriceablePosition(missing)
        return sum(
            self.position(symbol) * float(prices[symbol]) for symbol in self.lots
        )

    def nav(self, prices: Mapping[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def unrealised_pnl(self, prices: Mapping[str, float]) -> float:
        """Marked against the same basis a sale would realise against.

        Measured from `cost_basis` rather than by summing per-lot cost prices:
        a sale realises against the weighted average, so marking the remainder
        per-lot would make `realised + unrealised` depend on which lots happened
        to be consumed. `identity_residual` is the check that keeps the two
        definitions from drifting apart again.
        """
        missing = self.unpriceable(prices)
        if missing:
            raise UnpriceablePosition(missing)
        total = 0.0
        for symbol, lots in self.lots.items():
            held = total_quantity(lots)
            if held == 0:
                continue
            basis = self.cost_basis.get(symbol, 0.0)
            total += (float(prices[symbol]) - basis) * held
        return total

    def identity_residual(self, prices: Mapping[str, float]) -> float:
        """`realised + unrealised - (NAV - initial cash)`. Must be zero.

        The identity that ties the PnL split to the money: every yuan of profit
        has to show up either as cash that arrived or as inventory that is worth
        more than it cost. It failed by exactly the capitalised entry fees while
        cost basis excluded them (DEF-009) — cash was right and the PnL split was
        not, which is the hardest kind of error to notice from a NAV curve.
        """
        return (self.realised_pnl + self.unrealised_pnl(prices)) - (
            self.nav(prices) - self.initial_cash
        )

    # -- folding ------------------------------------------------------------
    def apply_fill(self, fill: Fill, trade_date: str) -> "AccountState":
        """The only path by which inventory and cash change."""
        if fill.side is Side.BUY:
            return self._apply_buy(fill, trade_date)
        return self._apply_sell(fill, trade_date)

    def _apply_buy(self, fill: Fill, trade_date: str) -> "AccountState":
        lot = PositionLot.from_fill(fill, trade_date)
        existing = self.lots.get(fill.symbol, ())
        held = total_quantity(existing)
        prior_cost = self.cost_basis.get(fill.symbol, 0.0)
        new_held = held + fill.quantity
        # Weighted average cost *including* entry fees. Capitalising them is not
        # a presentational preference: with fees excluded, `realised + unrealised`
        # exceeded `NAV - initial cash` by exactly the entry costs, so the book
        # reported profit that no cash or inventory backed (DEF-009). Cash was
        # never wrong, which is why the error survived a NAV-level review.
        blended = (
            (prior_cost * held + fill.gross + fill.fees) / new_held if new_held else 0.0
        )
        return replace(
            self,
            cash=self.cash + fill.cash_delta,
            lots={**self.lots, fill.symbol: (*existing, lot)},
            cost_basis={**self.cost_basis, fill.symbol: blended},
            total_fees=self.total_fees + fill.fees,
            total_slippage=self.total_slippage + fill.slippage,
        )

    def _apply_sell(self, fill: Fill, trade_date: str) -> "AccountState":
        existing = list(self.lots.get(fill.symbol, ()))
        available = sellable_quantity(existing, trade_date)
        if fill.quantity > available:
            raise InvariantViolation(
                f"{fill.symbol}: sell of {fill.quantity} exceeds T+1 sellable {available} "
                f"on {trade_date}"
            )
        # FIFO across settled lots. Unsettled lots are skipped, never consumed.
        remaining = fill.quantity
        kept: list[PositionLot] = []
        for lot in existing:
            if remaining <= 0 or not lot.sellable_on(trade_date):
                kept.append(lot)
                continue
            take = min(remaining, lot.quantity)
            remaining -= take
            if take < lot.quantity:
                kept.append(replace(lot, quantity=lot.quantity - take))
        basis = self.cost_basis.get(fill.symbol, fill.price)
        realised = (fill.price - basis) * fill.quantity - fill.fees
        lots = {**self.lots}
        if kept:
            lots[fill.symbol] = tuple(kept)
        else:
            lots.pop(fill.symbol, None)
        return replace(
            self,
            cash=self.cash + fill.cash_delta,
            lots=lots,
            realised_pnl=self.realised_pnl + realised,
            total_fees=self.total_fees + fill.fees,
            total_slippage=self.total_slippage + fill.slippage,
        )

    def apply_corporate_action(self, action: CorporateAction) -> "AccountState":
        """Fold a split, bonus issue or cash dividend.

        A pure share adjustment moves no PnL: the position scales and cost basis
        scales inversely, and the price adjusts by the same factor on the ex date.
        A cash dividend *is* income and goes to realised PnL — on the ex date the
        mark drops by the dividend, so cash rises and market value falls by the
        same amount and NAV is unchanged. Leaving the dividend out of realised PnL
        would therefore break `realised + unrealised == NAV - initial cash` by
        exactly the amount received, which is the identity DEF-009 was fixed to
        preserve.
        """
        lots = self.lots.get(action.symbol, ())
        if not lots:
            # Nothing held on the ex date. A corporate action on a position that
            # does not exist is not an error, it just has no effect.
            return self

        held = total_quantity(lots)
        cash_received = held * action.cash_per_share
        adjusted: list[PositionLot] = []
        for lot in lots:
            scaled = lot.quantity * action.share_ratio
            if abs(scaled - round(scaled)) > 1e-9:
                raise InvariantViolation(
                    f"{action.symbol}: ratio {action.share_ratio} on a {lot.quantity}-share "
                    f"lot yields {scaled} shares. Fractional entitlements need an explicit "
                    "cash-in-lieu policy, which this build does not have — inventing one "
                    "would silently create or destroy value on the ex date."
                )
            adjusted.append(
                replace(
                    lot,
                    quantity=int(round(scaled)),
                    cost_price=lot.cost_price / action.share_ratio,
                )
            )
        basis = self.cost_basis.get(action.symbol)
        cost_basis = dict(self.cost_basis)
        if basis is not None:
            cost_basis[action.symbol] = basis / action.share_ratio
        return replace(
            self,
            cash=self.cash + cash_received,
            realised_pnl=self.realised_pnl + cash_received,
            lots={**self.lots, action.symbol: tuple(adjusted)},
            cost_basis=cost_basis,
        )

    # -- invariants ---------------------------------------------------------
    def check(self) -> None:
        for symbol, lots in self.lots.items():
            for lot in lots:
                if lot.quantity <= 0:
                    raise InvariantViolation(f"{symbol}: lot {lot.position_lot_id} has quantity {lot.quantity}")

    # -- identity -----------------------------------------------------------
    def content_hash(self) -> str:
        """Stable digest of the economic state, for before/after comparison."""
        payload = {
            "cash": round(self.cash, 6),
            "realisedPnl": round(self.realised_pnl, 6),
            "totalFees": round(self.total_fees, 6),
            "totalSlippage": round(self.total_slippage, 6),
            "lots": {
                symbol: sorted(
                    (lot.position_lot_id, lot.quantity, round(lot.cost_price, 6), lot.acquired_on)
                    for lot in lots
                )
                for symbol, lots in sorted(self.lots.items())
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "realisedPnl": self.realised_pnl,
            "totalFees": self.total_fees,
            "totalSlippage": self.total_slippage,
            "positions": {symbol: self.position(symbol) for symbol in sorted(self.lots)},
            "duplicateExecutions": list(self.duplicate_executions),
            "contentHash": self.content_hash(),
        }

    def valuation(self, prices: Mapping[str, float]) -> dict[str, Any]:
        """NAV and PnL, or an explicit statement that they are unavailable.

        For callers that must answer without knowing whether every held symbol has
        a mark — an API route, a report. `nav` is `None` rather than a number when
        anything is unpriceable, which is the difference between "we do not know"
        and "it is worth nothing".
        """
        missing = self.unpriceable(prices)
        if missing:
            return {
                "nav": None,
                "marketValue": None,
                "unrealisedPnl": None,
                "identityResidual": None,
                "unpriceableSymbols": list(missing),
                "reason": (
                    "no mark for "
                    + ", ".join(missing)
                    + ": NAV is unknown rather than zero"
                ),
            }
        return {
            "nav": self.nav(prices),
            "marketValue": self.market_value(prices),
            "unrealisedPnl": self.unrealised_pnl(prices),
            "identityResidual": self.identity_residual(prices),
            "unpriceableSymbols": [],
            "reason": None,
        }


def replay_account(
    events: Iterable[OrderEvent],
    *,
    initial_cash: float,
    trade_date_of: Mapping[str, str] | None = None,
    corporate_actions: Iterable[tuple[int, CorporateAction]] = (),
) -> AccountState:
    """Rebuild economic state from nothing but the event log.

    `trade_date_of` maps an *execution id* — falling back to an order id — to the
    session that fill settles against; when neither is present the fill's own
    timestamp date is used. Execution first, because an order that trades in one
    session and is cancelled in the next has two dates and only the fill's is a
    settlement date (DEF-016). Fills are the only events with an economic effect;
    everything else is lifecycle bookkeeping.

    `corporate_actions` carries `(position, action)` pairs, where the position is
    the index in `events` *before* which the action applies. Interleaving matters:
    a dividend paid before a sale and one paid after it produce different realised
    PnL, so applying them all at the end would be a different scenario.
    """
    state = AccountState.opening(initial_cash)
    pending_actions: dict[int, list[CorporateAction]] = {}
    for position, action in corporate_actions:
        pending_actions.setdefault(int(position), []).append(action)
    #: One execution id may move money once. The order book refuses a duplicate
    #: at write time, but replay reads whatever the file holds — including records
    #: appended before that guard existed — so the identity is enforced again on
    #: the read path rather than assumed.
    applied: dict[str, Fill] = {}
    duplicates: list[str] = []
    for index, event in enumerate(events):
        for action in pending_actions.pop(index, ()):
            state = state.apply_corporate_action(action)
            state.check()
        if event.event_type not in {OrderEventType.PARTIAL_FILL, OrderEventType.FILL}:
            continue
        if event.fill is None:
            raise InvariantViolation(f"{event.event_type.value} event carried no fill")
        seen = applied.get(event.fill.execution_id)
        if seen is not None:
            if not same_execution(seen, event.fill):
                raise InvariantViolation(
                    f"execution {event.fill.execution_id} appears twice with different "
                    f"economics: {seen.quantity}@{seen.price} then "
                    f"{event.fill.quantity}@{event.fill.price}"
                )
            duplicates.append(event.fill.execution_id)
            continue
        applied[event.fill.execution_id] = event.fill
        known = trade_date_of or {}
        trade_date = (
            known.get(event.fill.execution_id)
            or known.get(event.order_id)
            or event.fill.filled_at[:10]
        )
        state = state.apply_fill(event.fill, trade_date)
        state.check()
    # Anything positioned at or after the end of the event stream.
    for position in sorted(pending_actions):
        for action in pending_actions[position]:
            state = state.apply_corporate_action(action)
            state.check()
    return replace(state, duplicate_executions=tuple(duplicates))


__all__ = [
    "AccountState",
    "InvariantViolation",
    "UnpriceablePosition",
    "replay_account",
]
