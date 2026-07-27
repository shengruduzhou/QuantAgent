"""Event-driven A-share microstructure simulator with enforced fidelity.

The simulator's distinguishing property is that **fidelity is a constraint, not
a setting**. It is constructed from a
:class:`~quantagent.data.microstructure.fidelity.FidelityDecision` derived from
the data, and every fill model it will run is checked against what that level
licenses. Asking for queue-position fills on snapshot data raises rather than
quietly degrading, because a silent degrade is how a Level-B result ends up
described as Level A in a report.

Fill models by level:

``LEVEL_A``  order-event replay. Queue position is tracked from the order
             stream: an order joins behind the volume resting at its price and
             advances as trades and cancels ahead of it clear.
``LEVEL_B``  snapshot replay. Depth is consumed level by level; queue position
             is *not* claimed, so a passive order fills only once the visible
             size at its price has traded through.
``LEVEL_C``  tick replay. Marketable orders cross the spread; passive orders
             fill on volume participation.
``LEVEL_D``  bar simulation, for daily and minute strategies only. Refuses to
             run at all when asked to evaluate a sub-daily strategy.

Every A-share rule comes from :mod:`quantagent.backtest.ashare_rules` -- T+1,
per-board price limits, ST bands, lot sizes, sell-side stamp duty. Nothing here
reuses an FX or futures execution model.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quantagent.backtest import ashare_rules as rules
from quantagent.data.microstructure import contracts as mc
from quantagent.data.microstructure import fidelity as fid

BUY = "BUY"
SELL = "SELL"

#: Order styles the simulator understands.
MARKETABLE = "MARKETABLE"
PASSIVE = "PASSIVE"


class SimulationRefused(RuntimeError):
    """Raised when the data does not license the requested simulation."""


@dataclass
class OrderIntent:
    symbol: str
    side: str
    shares: float
    style: str = MARKETABLE
    limit_price: float | None = None
    #: Wall-clock the strategy released the order, Asia/Shanghai "HH:MM:SS".
    release_time: str = "09:30:00"
    board: str = rules.SH_MAIN
    is_full_liquidation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Fill:
    symbol: str
    side: str
    shares: float
    price: float
    notional_cny: float
    costs: dict[str, float]
    fill_time: str
    fidelity_level: str
    model: str
    #: Populated only at Level A. Null elsewhere, and the absence is the point.
    queue_position_shares: float | None = None
    partial: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RejectedOrder:
    symbol: str
    side: str
    shares: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResult:
    fidelity_level: str
    permitted_claims: list[str]
    downgrades: list[str]
    fills: list[Fill] = field(default_factory=list)
    rejected: list[RejectedOrder] = field(default_factory=list)
    unfilled_shares: float = 0.0
    latency_assumption_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def filled_shares(self) -> float:
        return float(sum(f.shares for f in self.fills))

    @property
    def total_costs(self) -> float:
        return float(sum(f.costs.get("total", 0.0) for f in self.fills))

    def average_price(self) -> float | None:
        if not self.fills:
            return None
        notional = sum(f.notional_cny for f in self.fills)
        shares = self.filled_shares
        return notional / shares if shares else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fidelity_level": self.fidelity_level,
            "permitted_claims": self.permitted_claims,
            "downgrades": self.downgrades,
            "latency_assumption_ms": self.latency_assumption_ms,
            "filled_shares": self.filled_shares,
            "unfilled_shares": self.unfilled_shares,
            "average_price": self.average_price(),
            "total_costs": self.total_costs,
            "fills": [f.to_dict() for f in self.fills],
            "rejected": [r.to_dict() for r in self.rejected],
            "notes": self.notes,
        }


class AShareMicrostructureSimulator:
    """Replays canonical events against order intents under A-share rules."""

    def __init__(
        self,
        decision: fid.FidelityDecision,
        *,
        latency_ms: float = 50.0,
        commission_rate: float = rules.DEFAULT_COMMISSION_RATE,
        participation_cap: float = 0.10,
    ) -> None:
        if decision.level == fid.NOT_SIMULATABLE:
            raise SimulationRefused(
                "the supplied data licenses no simulation at all: "
                f"{decision.reasons}"
            )
        self.decision = decision
        self.latency_ms = float(latency_ms)
        self.commission_rate = commission_rate
        #: Cap on the share of observed volume a single order may take. Without
        #: it, a backtest happily fills an order larger than the day's turnover.
        self.participation_cap = float(participation_cap)

    # -- guards ------------------------------------------------------------
    def _require(self, *claims: str) -> None:
        fid.assert_claims_permitted(self.decision, list(claims))

    def _check_tradability(
        self, intent: OrderIntent, state: Mapping[str, Any]
    ) -> str | None:
        verdict = rules.tradability(
            is_suspended=bool(state.get("is_suspended", False)),
            at_limit_up=bool(state.get("at_limit_up", False)),
            at_limit_down=bool(state.get("at_limit_down", False)),
            is_delisting_period=bool(state.get("is_delisting_period", False)),
            holding_acquired_today=bool(state.get("holding_acquired_today", False)),
        )
        if intent.side == BUY and not verdict.can_buy:
            return "; ".join(verdict.reasons) or "buy not permitted"
        if intent.side == SELL and not verdict.can_sell:
            return "; ".join(verdict.reasons) or "sell not permitted"
        return None

    def _sized(self, intent: OrderIntent) -> int:
        return rules.round_to_lot(
            intent.shares, board=intent.board, side=intent.side,
            is_full_liquidation=intent.is_full_liquidation,
        )

    def _cost(self, notional: float, side: str, trade_date: Any) -> dict[str, float]:
        costs = rules.trading_costs(
            notional_cny=notional, side=side, trade_date=trade_date,
            commission_rate=self.commission_rate,
        )
        return costs.to_dict()

    # -- entry point -------------------------------------------------------
    def simulate(
        self,
        intents: Sequence[OrderIntent],
        events: pd.DataFrame,
        *,
        trade_date: str,
        state: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> SimulationResult:
        """Run every intent against ``events`` at the licensed fidelity."""
        result = SimulationResult(
            fidelity_level=self.decision.level,
            permitted_claims=list(self.decision.permitted_claims),
            downgrades=list(self.decision.downgrades),
            latency_assumption_ms=self.latency_ms,
        )
        state = state or {}

        if self.decision.level == fid.LEVEL_D:
            result.notes.append(
                "Level D: bar simulation only. Intraday order placement is not "
                "modelled and sub-daily results must not be reported."
            )

        for intent in intents:
            symbol_state = state.get(intent.symbol, {})
            blocked = self._check_tradability(intent, symbol_state)
            if blocked:
                result.rejected.append(
                    RejectedOrder(intent.symbol, intent.side, intent.shares, blocked)
                )
                continue

            shares = self._sized(intent)
            if shares <= 0:
                result.rejected.append(RejectedOrder(
                    intent.symbol, intent.side, intent.shares,
                    f"size {intent.shares} rounds below the {intent.board} "
                    f"minimum lot {rules.LOT_RULES.get(intent.board)}",
                ))
                continue

            symbol_events = events[events["symbol"] == intent.symbol]
            if symbol_events.empty:
                result.rejected.append(RejectedOrder(
                    intent.symbol, intent.side, shares,
                    "no market events for this symbol on this day",
                ))
                continue

            fills = self._fill(intent, shares, symbol_events, trade_date=trade_date)
            result.fills.extend(fills)
            result.unfilled_shares += max(0.0, shares - sum(f.shares for f in fills))

        return result

    # -- fill models -------------------------------------------------------
    def _fill(
        self, intent: OrderIntent, shares: int, events: pd.DataFrame, *, trade_date: str
    ) -> list[Fill]:
        level = self.decision.level
        if level == fid.LEVEL_A:
            return self._fill_level_a(intent, shares, events, trade_date=trade_date)
        if level == fid.LEVEL_B:
            return self._fill_level_b(intent, shares, events, trade_date=trade_date)
        if level == fid.LEVEL_C:
            return self._fill_level_c(intent, shares, events, trade_date=trade_date)
        return self._fill_level_d(intent, shares, events, trade_date=trade_date)

    def _eligible_events(self, events: pd.DataFrame, intent: OrderIntent) -> pd.DataFrame:
        """Events after the order is released, inside a phase that can trade."""
        frame = events.copy()
        if "exchange_time" in frame.columns:
            times = pd.to_datetime(frame["exchange_time"], errors="coerce")
            release = pd.to_datetime(
                f"{times.dt.strftime('%Y-%m-%d').iloc[0]} {intent.release_time}",
                errors="coerce",
            )
            latency = pd.Timedelta(milliseconds=self.latency_ms)
            frame = frame.loc[times >= (release + latency)]
            if not frame.empty:
                clock = pd.to_datetime(frame["exchange_time"]).dt.strftime("%H:%M")
                phases = clock.map(
                    lambda hhmm: mc.session_phase(hhmm, board=intent.board)
                )
                frame = frame.loc[phases.isin(mc.CONTINUOUS_PHASES)]
        order = "ingest_sequence" if "ingest_sequence" in frame.columns else "event_time_ns"
        return frame.sort_values(order, kind="mergesort") if not frame.empty else frame

    def _fill_level_c(
        self, intent: OrderIntent, shares: int, events: pd.DataFrame, *, trade_date: str
    ) -> list[Fill]:
        """Tick replay: cross the spread, or participate in observed volume."""
        self._require("spread_crossing", "volume_participation", "latency")
        eligible = self._eligible_events(events, intent)
        if eligible.empty:
            return []

        remaining = float(shares)
        fills: list[Fill] = []
        for _, event in eligible.iterrows():
            if remaining <= 0:
                break
            price = float(event.get("price", np.nan))
            volume = float(event.get("volume_shares", 0.0) or 0.0)
            if not np.isfinite(price) or volume <= 0:
                continue
            if intent.style == PASSIVE and intent.limit_price is not None:
                if intent.side == BUY and price > intent.limit_price:
                    continue
                if intent.side == SELL and price < intent.limit_price:
                    continue
            available = volume * self.participation_cap
            take = min(remaining, available)
            if take <= 0:
                continue
            notional = take * price
            fills.append(Fill(
                symbol=intent.symbol, side=intent.side, shares=take, price=price,
                notional_cny=notional,
                costs=self._cost(notional, intent.side, trade_date),
                fill_time=str(event.get("exchange_time")),
                fidelity_level=self.decision.level,
                model="tick_participation",
                partial=True,
            ))
            remaining -= take

        if fills and remaining <= 0:
            fills[-1].partial = False
        return fills

    def _fill_level_b(
        self, intent: OrderIntent, shares: int, events: pd.DataFrame, *, trade_date: str
    ) -> list[Fill]:
        """Snapshot replay: consume visible depth, claim no queue position."""
        self._require("visible_depth", "depth_depletion", "approximate_partial_fills")
        eligible = self._eligible_events(events, intent)
        if eligible.empty:
            return []

        price_column = "ask_price" if intent.side == BUY else "bid_price"
        size_column = "ask_volume_shares" if intent.side == BUY else "bid_volume_shares"
        if price_column not in eligible.columns:
            return self._fill_level_c(intent, shares, events, trade_date=trade_date)

        remaining = float(shares)
        fills: list[Fill] = []
        for _, level_row in eligible.iterrows():
            if remaining <= 0:
                break
            price = float(level_row.get(price_column, np.nan))
            visible = float(level_row.get(size_column, 0.0) or 0.0)
            if not np.isfinite(price) or visible <= 0:
                continue
            if intent.limit_price is not None:
                if intent.side == BUY and price > intent.limit_price:
                    continue
                if intent.side == SELL and price < intent.limit_price:
                    continue
            take = min(remaining, visible)
            notional = take * price
            fills.append(Fill(
                symbol=intent.symbol, side=intent.side, shares=take, price=price,
                notional_cny=notional,
                costs=self._cost(notional, intent.side, trade_date),
                fill_time=str(level_row.get("exchange_time")),
                fidelity_level=self.decision.level,
                model="snapshot_depth_consumption",
                # Explicitly null: Level B cannot observe queue position.
                queue_position_shares=None,
                partial=True,
            ))
            remaining -= take

        if fills and remaining <= 0:
            fills[-1].partial = False
        return fills

    def _fill_level_a(
        self, intent: OrderIntent, shares: int, events: pd.DataFrame, *, trade_date: str
    ) -> list[Fill]:
        """Order-event replay: track queue position from the order stream.

        A passive order joins behind whatever size already rests at its price.
        It advances only as that size actually leaves -- traded through or
        cancelled -- and fills once the queue ahead is exhausted. This is the
        only level where ``queue_position_shares`` is populated, because it is
        the only level where the order stream makes it observable.
        """
        self._require("queue_position", "price_time_priority", "partial_fills")
        eligible = self._eligible_events(events, intent)
        if eligible.empty:
            return []

        limit = intent.limit_price
        if limit is None or intent.style == MARKETABLE:
            return self._fill_level_c(intent, shares, events, trade_date=trade_date)

        # Size resting ahead of us at our price when the order arrives.
        resting = eligible[
            (eligible.get("event_action", "INSERT") == "INSERT")
            & (pd.to_numeric(eligible.get("price"), errors="coerce") == limit)
            & (eligible.get("side") == intent.side)
        ]
        queue_ahead = float(
            pd.to_numeric(resting.get("volume_shares"), errors="coerce").sum()
        )

        remaining = float(shares)
        fills: list[Fill] = []
        for _, event in eligible.iterrows():
            if remaining <= 0:
                break
            price = pd.to_numeric(pd.Series([event.get("price")]), errors="coerce").iloc[0]
            if not np.isfinite(price) or float(price) != limit:
                continue
            volume = float(event.get("volume_shares", 0.0) or 0.0)
            action = str(event.get("event_action", ""))
            counter_side = event.get("side") != intent.side

            if action == "CANCEL" or counter_side:
                # Both a cancellation ahead of us and a trade against the other
                # side reduce the queue in front of the order.
                consumed = min(queue_ahead, volume)
                queue_ahead -= consumed
                leftover = volume - consumed
                if leftover > 0 and queue_ahead <= 0:
                    take = min(remaining, leftover)
                    notional = take * limit
                    fills.append(Fill(
                        symbol=intent.symbol, side=intent.side, shares=take,
                        price=float(limit), notional_cny=notional,
                        costs=self._cost(notional, intent.side, trade_date),
                        fill_time=str(event.get("exchange_time")),
                        fidelity_level=self.decision.level,
                        model="order_queue_replay",
                        queue_position_shares=0.0, partial=True,
                    ))
                    remaining -= take
            elif action == "INSERT":
                # Orders arriving after ours queue behind it; no effect.
                continue

        if fills and remaining <= 0:
            fills[-1].partial = False
        elif not fills:
            return []
        return fills

    def _fill_level_d(
        self, intent: OrderIntent, shares: int, events: pd.DataFrame, *, trade_date: str
    ) -> list[Fill]:
        """Bar simulation: one fill at the bar close, no intraday claims."""
        self._require("bar_close_execution")
        price_column = "close" if "close" in events.columns else "price"
        prices = pd.to_numeric(events[price_column], errors="coerce").dropna()
        if prices.empty:
            return []
        price = float(prices.iloc[-1])
        notional = shares * price
        return [Fill(
            symbol=intent.symbol, side=intent.side, shares=float(shares), price=price,
            notional_cny=notional, costs=self._cost(notional, intent.side, trade_date),
            fill_time=str(events.get("exchange_time", pd.Series(["bar_close"])).iloc[-1]),
            fidelity_level=self.decision.level, model="bar_close",
        )]


def simulator_for(
    events_by_class: Mapping[str, pd.DataFrame],
    *,
    integrity_reports: Sequence[Any] = (),
    has_bars: bool = True,
    **kwargs: Any,
) -> AShareMicrostructureSimulator:
    """Build a simulator whose fidelity is derived from the data supplied."""
    decision = fid.decide_fidelity(
        data_classes=list(events_by_class),
        integrity_reports=list(integrity_reports),
        has_bars=has_bars,
    )
    return AShareMicrostructureSimulator(decision, **kwargs)
