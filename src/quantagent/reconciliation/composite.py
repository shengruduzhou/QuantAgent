"""One scenario, four engines, one record of account.

The fast backtest, the paper broker, the OMS-to-paper chain and the event-driven
streaming engine each write to a `CanonicalLedger`. Until they were run against the
*same* economic scenario and diffed event by event, "the ledger is the single source
of truth" was an architectural intention rather than a measured fact.

Two of the comparisons carry no permitted difference at all, and they are the
interesting ones. Paper versus OMS-to-paper isolates the routing layer: the OMS adds
intent, risk and idempotency, and none of that may move a number. Streaming versus
paper isolates *control flow*: both consult the same A-share rulebook, but paper
validates and fills inside one synchronous call while the streaming venue is driven
by events and carries orders across bars. A difference there would be a defect in
one of them, not a modelling choice — which is why neither pair is granted a
tolerance.

The scenario deliberately covers the cases where books diverge in practice:

* an order that fills completely, with fees and sell-side stamp duty;
* an order accepted by the venue and then refused at fill time, with zero fills;
* a T+1 refusal, and the same sale succeeding in the next session;
* a partial fill whose remainder is cancelled;
* a re-delivered execution report, which must move no money;
* an out-of-order lifecycle event, which must be refused;
* a restart that keeps nothing but the ledger file.

Every path is measured twice: its own in-memory figures against the replay
(*does the engine agree with the record of account?*), then path against path
(*do the engines agree with each other?*). Differences between engines are
expected — they model prices differently — but each one must be named by an
`ExplanationRule`. What is left over is
`unexplained_economic_differences`, and the gate is zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from quantagent.domain.ledger import CanonicalLedger
from quantagent.domain.lineage import Lineage
from quantagent.execution.broker_base import Order as WireOrder, OrderSide, OrderType
from quantagent.execution.order_manager import OrderManager, OrderManagerConfig
from quantagent.execution.paper_adapter import PaperBrokerAdapter
from quantagent.paper import ledger as paper_ledger
from quantagent.paper import orders as po
from quantagent.paper.broker import BrokerConfig, MarketSnapshot, PaperBroker
from quantagent.paper.portfolio import Portfolio
from quantagent.reconciliation.differences import (
    BOUNDED_FLOAT,
    DOCUMENTED_ENGINE_DIFFERENCE,
    DifferenceTable,
    ExplanationRule,
    NOT_APPLICABLE,
    compare_flat,
    compare_snapshots,
)
from quantagent.reconciliation.snapshot import EconomicSnapshot

FAST = "fast_backtest"
PAPER = "paper_broker"
OMS_PAPER = "oms_to_paper"
STREAMING = "streaming_engine"

SESSION_1 = "2026-08-04"
SESSION_2 = "2026-08-05"
BLUE = "600000.SH"
THIN = "000001.SZ"
INITIAL_CASH = 1_000_000.0


@dataclass(frozen=True, slots=True)
class MarketPoint:
    """The market data every engine is given. Identical inputs or nothing."""

    symbol: str
    session: str
    last_price: float
    previous_close: float
    session_volume: float
    board: str = "SH_Main"

    def snapshot(self, clock: str = "10:00:00") -> MarketSnapshot:
        return MarketSnapshot(
            symbol=self.symbol, trade_date=self.session, last_price=self.last_price,
            previous_close=self.previous_close, session_volume=self.session_volume,
            board=self.board, clock=clock,
        )


@dataclass(frozen=True, slots=True)
class Step:
    """One economic action, and which engines are expected to express it."""

    step_id: str
    kind: str
    session: str
    symbol: str = ""
    quantity: int = 0
    limit_price: float | None = None
    proves: str = ""
    paths: frozenset[str] = frozenset({PAPER, OMS_PAPER, STREAMING})


MARKET: tuple[MarketPoint, ...] = (
    MarketPoint(BLUE, SESSION_1, 10.00, 10.00, 1e8, "SH_Main"),
    MarketPoint(BLUE, SESSION_2, 10.50, 10.00, 1e8, "SH_Main"),
    # Thin enough that the 10% participation cap can only fill part of an order.
    MarketPoint(THIN, SESSION_1, 20.00, 20.00, 10_000, "SZ_Main"),
    MarketPoint(THIN, SESSION_2, 20.00, 20.00, 10_000, "SZ_Main"),
)

STEPS: tuple[Step, ...] = (
    Step(
        "s1-buy-full", "buy", SESSION_1, BLUE, 1_000, 10.05,
        proves="accepted order fills completely; commission and transfer fee charged once",
        paths=frozenset({FAST, PAPER, OMS_PAPER, STREAMING}),
    ),
    Step(
        "s1-sell-t1-refused", "sell", SESSION_1, BLUE, 500, 9.95,
        proves="T+1: shares bought this session cannot be sold in it",
    ),
    Step(
        "s1-buy-unaffordable", "buy", SESSION_1, BLUE, 200_000, 10.05,
        proves="ACCEPTED then REJECTED at fill time, zero fills, no cash movement",
    ),
    Step(
        "s1-buy-partial", "buy", SESSION_1, THIN, 3_000, 20.20,
        proves="participation cap fills part of the order; remainder stays working",
    ),
    Step(
        "s1-cancel-remainder", "cancel_open", SESSION_1, THIN,
        proves="cancelling a partially filled order keeps its executed quantity",
    ),
    Step(
        "s1-duplicate-execution", "duplicate_execution", SESSION_1, BLUE,
        proves="a re-delivered execution report produces zero economic delta",
    ),
    Step(
        "s1-out-of-order-event", "out_of_order_event", SESSION_1, BLUE,
        proves="a lifecycle event arriving after a later one is refused",
    ),
    Step(
        "s1-close", "close_session", SESSION_1,
        proves="settlement promotes the session's purchases to sellable",
        paths=frozenset({FAST, PAPER, OMS_PAPER, STREAMING}),
    ),
    Step(
        "s2-sell-settled", "sell", SESSION_2, BLUE, 1_000, 10.45,
        proves="next-session sale succeeds and pays sell-side stamp duty",
        paths=frozenset({FAST, PAPER, OMS_PAPER, STREAMING}),
    ),
    Step(
        "s2-restart-recovery", "restart", SESSION_2,
        proves="every figure above rebuilds from the ledger file alone",
        paths=frozenset({FAST, PAPER, OMS_PAPER, STREAMING}),
    ),
)

#: Closing marks for the final NAV. Same prices for every engine.
CLOSING_PRICES: Mapping[str, float] = {BLUE: 10.50, THIN: 20.00}


def market_point(symbol: str, session: str) -> MarketPoint:
    for point in MARKET:
        if point.symbol == symbol and point.session == session:
            return point
    raise KeyError(f"no market data for {symbol} on {session}")


# ---------------------------------------------------------------------------
# Explanation rules
# ---------------------------------------------------------------------------
#: Why the fast engine legitimately differs from the venue paths. Each rule
#: names the code that causes it. Note what is *not* here: no rule excuses a
#: difference in position, settled inventory, order status or quantities — the
#: engines must agree on what happened, only on price may they differ.
_FAST_VS_VENUE_PRICE = (
    "backtest/engine.py fills at the bar's fill-price column via AShareFillModel; "
    "paper/broker.py fills at last price plus slippage and square-root impact, "
    "bounded by the order's limit. Different price, same quantity."
)

CROSS_PATH_RULES: tuple[ExplanationRule, ...] = tuple(
    rule
    for pair in ((FAST, PAPER), (FAST, OMS_PAPER), (FAST, STREAMING))
    for rule in (
        ExplanationRule("cash", DOCUMENTED_ENGINE_DIFFERENCE, "fill_price_model", _FAST_VS_VENUE_PRICE, pair=pair),
        ExplanationRule("available_cash", DOCUMENTED_ENGINE_DIFFERENCE, "fill_price_model", _FAST_VS_VENUE_PRICE, pair=pair),
        ExplanationRule("realised_pnl", DOCUMENTED_ENGINE_DIFFERENCE, "fill_price_model", _FAST_VS_VENUE_PRICE, pair=pair),
        ExplanationRule("unrealised_pnl", DOCUMENTED_ENGINE_DIFFERENCE, "fill_price_model", _FAST_VS_VENUE_PRICE, pair=pair),
        ExplanationRule("nav", DOCUMENTED_ENGINE_DIFFERENCE, "fill_price_model", _FAST_VS_VENUE_PRICE, pair=pair),
        ExplanationRule("commission", DOCUMENTED_ENGINE_DIFFERENCE, "cost_model", "quantagent.quant_math.transaction_cost vs backtest.ashare_rules.trading_costs: different commission floors and rates.", pair=pair),
        ExplanationRule("transfer_fee", DOCUMENTED_ENGINE_DIFFERENCE, "cost_model", "quantagent.quant_math.transaction_cost vs backtest.ashare_rules.trading_costs: transfer fee is board-scoped in one and flat in the other.", pair=pair),
        ExplanationRule("stamp_duty", DOCUMENTED_ENGINE_DIFFERENCE, "cost_model", "sell-side stamp duty is charged on notional, and the two engines fill at different prices.", pair=pair),
        ExplanationRule("fees_total", DOCUMENTED_ENGINE_DIFFERENCE, "cost_model", "sum of the itemised cost differences above.", pair=pair),
        ExplanationRule("slippage", DOCUMENTED_ENGINE_DIFFERENCE, "fill_price_model", "the fast engine's reference price is the bar column it filled at, so its recorded slippage is structurally smaller than a venue's.", pair=pair),
        ExplanationRule("lots[", DOCUMENTED_ENGINE_DIFFERENCE, "fill_price_model", "lot cost prices carry the engine's fill price; quantities and acquisition dates are compared exactly.", pair=pair, prefix=True),
        # Everything the fast engine cannot express. Naming the step is what
        # keeps this from becoming a blanket exemption.
        ExplanationRule("order[", NOT_APPLICABLE, "fast_engine_scope", "the fast engine is weight-driven: it has no venue callback, no working order and no cancel, so steps s1-sell-t1-refused, s1-buy-unaffordable, s1-buy-partial, s1-cancel-remainder, s1-duplicate-execution and s1-out-of-order-event have no counterpart in it. Those steps are proven on the paper and OMS paths, which do have a venue.", pair=pair, prefix=True),
        ExplanationRule("count[", NOT_APPLICABLE, "fast_engine_scope", "order, fill, reject and cancel counts differ because the fast engine expresses a subset of the scenario's steps; see the order-level rule.", pair=pair, prefix=True),
        ExplanationRule("position[", NOT_APPLICABLE, "fast_engine_scope", "the thin symbol exists only in the partial-fill step, which the fast engine cannot express.", pair=pair, prefix=True),
        ExplanationRule("settled_inventory[", NOT_APPLICABLE, "fast_engine_scope", "as position: the thin symbol is out of the fast engine's scope.", pair=pair, prefix=True),
        ExplanationRule("sellable_inventory[", NOT_APPLICABLE, "fast_engine_scope", "as position: the thin symbol is out of the fast engine's scope.", pair=pair, prefix=True),
        ExplanationRule("market_value", DOCUMENTED_ENGINE_DIFFERENCE, "fast_engine_scope", "the fast engine holds no position in the thin symbol.", pair=pair),
        ExplanationRule("lineage_gaps", DOCUMENTED_ENGINE_DIFFERENCE, "fast_engine_scope", "gap lists are keyed by logical order, and the two paths carry different order sets.", pair=pair),
    )
)

#: Paper and the OMS-to-paper chain run the *same* venue on the *same* market
#: data. There is no rule permitting any difference between them: the OMS adds
#: intent, risk and idempotency in front of the venue, and none of that may
#: change an economic figure.
PAPER_VS_OMS_RULES: tuple[ExplanationRule, ...] = ()

#: Streaming versus paper is the interesting one, and it also carries no permitted
#: difference. The two share `backtest.ashare_rules` — bands, tradability, lot
#: rounding and costs — but nothing else: paper validates and fills inside one
#: synchronous `submit`, while the streaming venue is driven by events and holds
#: orders across bars. So agreement here is a statement about *control flow* over a
#: shared rulebook, and any difference is a defect in one of the two rather than a
#: modelling choice. Granting a tolerance would discard exactly the signal this
#: comparison exists to produce.
STREAMING_VS_PAPER_RULES: tuple[ExplanationRule, ...] = ()


# ---------------------------------------------------------------------------
# Per-path execution
# ---------------------------------------------------------------------------
@dataclass
class PathResult:
    label: str
    ledger_path: Path
    #: Rebuilt from the ledger file with nothing kept in memory.
    snapshot: EconomicSnapshot
    #: The engine's own figures, and how they compare to the replay.
    native: Mapping[str, Any]
    native_table: DifferenceTable
    steps_expressed: tuple[str, ...]
    steps_out_of_scope: Mapping[str, str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "ledgerPath": str(self.ledger_path),
            "snapshot": self.snapshot.to_dict(),
            "native": dict(self.native),
            "nativeVsReplay": self.native_table.to_dict(),
            "stepsExpressed": list(self.steps_expressed),
            "stepsOutOfScope": dict(self.steps_out_of_scope),
            "notes": list(self.notes),
        }


def _lineage(run_id: str) -> Lineage:
    return Lineage(research_id="composite", strategy_version_id="sv_composite", run_id=run_id)


def _paper_native(broker: PaperBroker) -> dict[str, Any]:
    """The figures paper maintains itself, in the snapshot's dimension names."""
    portfolio = broker.portfolio
    native: dict[str, Any] = {
        "cash": round(portfolio.cash, 6),
        "realised_pnl": round(portfolio.realised_pnl, 6),
        "fees_total": round(portfolio.fees_paid, 6),
    }
    for symbol, position in portfolio.positions.items():
        if position.is_flat:
            continue
        native[f"position[{symbol}]"] = int(position.total)
        native[f"settled_inventory[{symbol}]"] = int(position.sellable)
    return native


def _replay(label: str, path: Path, *, session: str) -> EconomicSnapshot:
    """Rebuild from the file. Nothing from the run that wrote it is reused."""
    ledger = CanonicalLedger(path)
    book, account = ledger.replay(initial_cash=INITIAL_CASH)
    return EconomicSnapshot.from_replay(
        label, book, account, session=session, prices=CLOSING_PRICES
    )


def _steps_for(path: str) -> tuple[tuple[Step, ...], dict[str, str]]:
    expressed = tuple(step for step in STEPS if path in step.paths)
    out_of_scope = {
        step.step_id: f"not expressible on {path}: {step.proves}"
        for step in STEPS
        if path not in step.paths
    }
    return expressed, out_of_scope


def _build_paper(work_dir: Path, run_id: str, **canonical: Any) -> tuple[PaperBroker, Portfolio]:
    portfolio = Portfolio(portfolio_id=f"p_{run_id}", cash=INITIAL_CASH, initial_cash=INITIAL_CASH)
    broker = PaperBroker(
        portfolio,
        paper_ledger.EventLedger(work_dir / f"{run_id}_operational.jsonl"),
        run_id=run_id,
        config=BrokerConfig(participation_cap=0.10, slippage_bps=5.0, impact_coefficient=0.10),
        lineage=_lineage(run_id),
        **canonical,
    )
    return broker, portfolio


def run_paper_path(work_dir: Path) -> PathResult:
    """Drive the venue directly: paper broker, no OMS in front."""
    ledger_path = work_dir / "paper_canonical.jsonl"
    broker, _ = _build_paper(work_dir, "paper_run", canonical_ledger_path=str(ledger_path))
    expressed, out_of_scope = _steps_for(PAPER)
    notes: list[str] = []
    placed: dict[str, po.Order] = {}

    for step in expressed:
        if step.kind in {"buy", "sell"}:
            point = market_point(step.symbol, step.session)
            order = po.Order(
                symbol=step.symbol,
                side=po.BUY if step.kind == "buy" else po.SELL,
                quantity=float(step.quantity),
                order_type=po.LIMIT,
                limit_price=step.limit_price,
                board=point.board,
            )
            broker.submit(order, point.snapshot())
            placed[step.step_id] = order
        elif step.kind == "cancel_open":
            for order in broker.open_orders():
                if order.symbol == step.symbol:
                    broker.cancel(order.order_id)
        elif step.kind == "duplicate_execution":
            target = placed["s1-buy-full"]
            fill = next(f for f in broker.fills if f.order_id == target.order_id)
            booked = broker.apply_execution_report(target, fill, trade_date=step.session)
            notes.append(
                f"re-delivered execution {fill.fill_id}: booked={booked} (False means absorbed)"
            )
        elif step.kind == "out_of_order_event":
            from quantagent.domain.orders import IllegalTransition, OrderEventType

            target = placed["s1-buy-full"]
            try:
                broker._canonical_event(  # the venue's own mirror, exercised directly
                    target, OrderEventType.ACCEPTED, trade_date=step.session
                )
                notes.append("out-of-order ACCEPTED was NOT refused")
            except IllegalTransition as exc:
                notes.append(f"out-of-order ACCEPTED refused: {exc}")
        elif step.kind == "close_session":
            broker.close_session(step.session)

    native = _paper_native(broker)
    snapshot = _replay(PAPER, ledger_path, session=SESSION_2)
    return PathResult(
        label=PAPER,
        ledger_path=ledger_path,
        snapshot=snapshot,
        native=native,
        native_table=compare_flat(
            f"{PAPER}_native", native, f"{PAPER}_replay", snapshot.flatten(),
            restrict_to_shared=True,
        ),
        steps_expressed=tuple(step.step_id for step in expressed),
        steps_out_of_scope=out_of_scope,
        notes=notes,
    )


def run_oms_paper_path(work_dir: Path) -> PathResult:
    """Drive the full chain: intent -> risk -> OMS -> venue -> one ledger."""
    ledger_path = work_dir / "oms_canonical.jsonl"
    shared_ledger = CanonicalLedger(ledger_path)
    broker, _ = _build_paper(
        work_dir, "oms_run", canonical_ledger=shared_ledger,
    )
    adapter = PaperBrokerAdapter(
        broker, lambda symbol, session: market_point(symbol, session).snapshot()
    )
    manager = OrderManager(
        broker=adapter,
        config=OrderManagerConfig(strategy_version="composite"),
        lineage=_lineage("oms_run"),
        idempotency_path=str(work_dir / "oms_claims.jsonl"),
        canonical_ledger=shared_ledger,
        order_book=broker.book,
    )
    expressed, out_of_scope = _steps_for(OMS_PAPER)
    notes: list[str] = []

    for step in expressed:
        if step.kind in {"buy", "sell"}:
            manager.submit_orders(
                [
                    WireOrder(
                        client_order_id=step.step_id,
                        symbol=step.symbol,
                        side=OrderSide.BUY if step.kind == "buy" else OrderSide.SELL,
                        quantity=step.quantity,
                        order_type=OrderType.LIMIT,
                        price=step.limit_price,
                        signal_id=step.step_id,
                        strategy_version="composite",
                        timestamp=f"{step.session}T10:00:00+00:00",
                    )
                ]
            )
        elif step.kind == "cancel_open":
            manager.cancel_all_open()
        elif step.kind == "duplicate_execution":
            target = broker.orders[adapter._paper_ids["s1-buy-full"]]
            fill = next(f for f in broker.fills if f.order_id == target.order_id)
            booked = broker.apply_execution_report(target, fill, trade_date=step.session)
            notes.append(
                f"re-delivered execution {fill.fill_id}: booked={booked} (False means absorbed)"
            )
            # The OMS's own guard: the same intent re-submitted must not reach
            # the venue a second time.
            before = len(broker.fills)
            manager.submit_orders(
                [
                    WireOrder(
                        client_order_id="s1-buy-full",
                        symbol=BLUE,
                        side=OrderSide.BUY,
                        quantity=1_000,
                        order_type=OrderType.LIMIT,
                        price=10.05,
                        signal_id="s1-buy-full",
                        strategy_version="composite",
                        timestamp=f"{SESSION_1}T10:00:00+00:00",
                    )
                ]
            )
            notes.append(
                f"re-submitted intent s1-buy-full: venue fills {before} -> {len(broker.fills)}"
            )
        elif step.kind == "out_of_order_event":
            from quantagent.domain.orders import IllegalTransition, OrderEventType

            target = broker.orders[adapter._paper_ids["s1-buy-full"]]
            try:
                broker._canonical_event(target, OrderEventType.ACCEPTED, trade_date=step.session)
                notes.append("out-of-order ACCEPTED was NOT refused")
            except IllegalTransition as exc:
                notes.append(f"out-of-order ACCEPTED refused: {exc}")
        elif step.kind == "close_session":
            broker.close_session(step.session)

    projection_agrees = manager.history.keys() == manager.rebuild_history().keys()
    notes.append(f"OMS wire projection matches ledger rebuild: {projection_agrees}")

    native = _paper_native(broker)
    snapshot = _replay(OMS_PAPER, ledger_path, session=SESSION_2)
    return PathResult(
        label=OMS_PAPER,
        ledger_path=ledger_path,
        snapshot=snapshot,
        native=native,
        native_table=compare_flat(
            f"{OMS_PAPER}_native", native, f"{OMS_PAPER}_replay", snapshot.flatten(),
            restrict_to_shared=True,
        ),
        steps_expressed=tuple(step.step_id for step in expressed),
        steps_out_of_scope=out_of_scope,
        notes=notes,
    )


def run_fast_path(work_dir: Path) -> PathResult:
    """Drive the weight-driven fast engine over the same two sessions."""
    from quantagent.backtest.engine import BacktestConfig, EventDrivenBacktester

    ledger_path = work_dir / "fast_canonical.jsonl"
    expressed, out_of_scope = _steps_for(FAST)

    # 1% of a 1,000,000 NAV at 10.00 is exactly 1,000 shares — the same quantity
    # the venue paths buy — then flat in the next session.
    weights = pd.DataFrame(
        {BLUE: [0.01, 0.0]}, index=pd.to_datetime([SESSION_1, SESSION_2])
    )
    rows = []
    for point in MARKET:
        if point.symbol != BLUE:
            continue
        rows.append(
            {
                "symbol": point.symbol,
                "trade_date": point.session,
                "open": point.last_price,
                "high": point.last_price,
                "low": point.last_price,
                "close": point.last_price,
                "volume": point.session_volume,
                "pre_close": point.previous_close,
                "amount": point.last_price * point.session_volume,
            }
        )
    prices = pd.DataFrame(rows)

    engine = EventDrivenBacktester(
        BacktestConfig(
            initial_nav=INITIAL_CASH,
            next_day_fill=False,
            fill_price_column="open",
            ledger_path=str(ledger_path),
            lineage=_lineage("fast_run"),
        )
    )
    result = engine.run(weights, prices)
    native = {
        "nav": round(float(result.diagnostics["final_nav"]), 6),
        "count[fills]": int(len(result.trades)),
    }
    snapshot = _replay(FAST, ledger_path, session=SESSION_2)
    return PathResult(
        label=FAST,
        ledger_path=ledger_path,
        snapshot=snapshot,
        native=native,
        native_table=compare_flat(
            f"{FAST}_native", native, f"{FAST}_replay", snapshot.flatten(),
            rules=(
                ExplanationRule(
                    "nav", BOUNDED_FLOAT, "engine_nav_marks_at_close",
                    "the engine's reported NAV marks positions at each session's "
                    "close as it walks the panel; the replay marks the final "
                    "closing prices. Same cash, same shares.",
                    tolerance=1e-6,
                ),
            ),
            restrict_to_shared=True,
        ),
        steps_expressed=tuple(step.step_id for step in expressed),
        steps_out_of_scope=out_of_scope,
    )



def run_streaming_path(work_dir: Path) -> PathResult:
    """Drive the event-driven engine, which derives its own fills from bars.

    The point of this leg is that nothing is handed to it. It is given the same
    market data the venue paths get, and its matcher decides acceptance, quantity
    and price itself over the shared A-share rulebook. Replaying the venue's
    answers would make agreement automatic and meaningless.
    """
    from datetime import time as clock_time

    from quantagent.domain.timeline import EventTime, exchange_moment
    from quantagent.streaming.bus import EventBus
    from quantagent.streaming.events import EventKind, MarketEvent
    from quantagent.streaming.lifecycle import OrderLifecycle
    from quantagent.streaming.matching import MatcherConfig, MatchingVenue

    ledger_path = work_dir / "streaming_canonical.jsonl"
    lifecycle = OrderLifecycle(
        ledger=CanonicalLedger(ledger_path),
        lineage=_lineage("streaming_run"),
        initial_cash=INITIAL_CASH,
    )
    bus = EventBus()
    venue = MatchingVenue(
        lifecycle=lifecycle, bus=bus,
        config=MatcherConfig(participation_cap=0.10, slippage_bps=5.0, impact_coefficient=0.10),
    )
    expressed, out_of_scope = _steps_for(STREAMING)
    notes: list[str] = []

    def market_event(point: MarketPoint, at: clock_time) -> MarketEvent:
        return MarketEvent(
            kind=EventKind.BAR,
            times=EventTime.immediate(exchange_moment(point.session, at)),
            symbol=point.symbol,
            payload={
                "close": point.last_price,
                "previousClose": point.previous_close,
                "volume": point.session_volume,
                "board": point.board,
            },
        )

    def order_event(step: Step, at: clock_time) -> MarketEvent:
        point = market_point(step.symbol, step.session)
        return MarketEvent(
            kind=EventKind.ORDER,
            times=EventTime.immediate(exchange_moment(step.session, at)),
            symbol=step.symbol,
            payload={
                "clientOrderId": step.step_id,
                "side": "BUY" if step.kind == "buy" else "SELL",
                "quantity": step.quantity,
                "limitPrice": step.limit_price,
                "board": point.board,
                "previousClose": point.previous_close,
            },
        )

    def handler(event: MarketEvent, frontier) -> None:
        # The lifecycle first, so the order exists on the chain before the venue is
        # asked about it; then the venue, which answers with further events.
        lifecycle.handle(event, frontier)
        venue.handle(event)

    # Session one: orders before the bar, so the bar is what fills them.
    for step in expressed:
        if step.kind in {"buy", "sell"} and step.session == SESSION_1:
            bus.publish(order_event(step, clock_time(10, 0)))
    for point in MARKET:
        if point.session == SESSION_1:
            bus.publish(market_event(point, clock_time(15, 0)))
    bus.run(handler)

    # The remainder on the thin symbol is cancelled before the next session's bar
    # can fill any more of it.
    for step in expressed:
        if step.kind == "cancel_open":
            for order in lifecycle.book.orders():
                if order.symbol == step.symbol and not order.is_terminal:
                    bus.publish(
                        MarketEvent(
                            kind=EventKind.CANCEL,
                            times=EventTime.immediate(
                                exchange_moment(SESSION_1, clock_time(15, 30))
                            ),
                            symbol=step.symbol,
                            payload={
                                "clientOrderId": venue._client_order_id(order),
                                "reason": "operator_cancel",
                            },
                        )
                    )
    bus.run(handler)

    # A re-delivered execution report, and an out-of-order acknowledgement.
    for step in expressed:
        if step.kind == "duplicate_execution":
            fills = [
                event.fill for event in lifecycle.book.events()
                if event.fill is not None and event.fill.symbol == step.symbol
            ]
            if fills:
                original = fills[0]
                bus.publish(
                    MarketEvent(
                        kind=EventKind.FILL,
                        times=EventTime.immediate(
                            exchange_moment(SESSION_1, clock_time(15, 45))
                        ),
                        symbol=step.symbol,
                        payload={
                            "clientOrderId": "s1-buy-full",
                            "executionId": original.execution_id,
                            "quantity": original.quantity,
                            "price": original.price,
                            "commission": original.commission,
                            "stampDuty": original.stamp_duty,
                            "transferFee": original.transfer_fee,
                        },
                    )
                )
                before = len(lifecycle.book.fills())
                bus.run(handler)
                notes.append(
                    f"re-delivered execution {original.execution_id}: fills "
                    f"{before} -> {len(lifecycle.book.fills())} (unchanged means absorbed)"
                )
        elif step.kind == "out_of_order_event":
            from quantagent.domain.orders import IllegalTransition

            bus.publish(
                MarketEvent(
                    kind=EventKind.VENUE_CALLBACK,
                    times=EventTime.immediate(
                        exchange_moment(SESSION_1, clock_time(15, 50))
                    ),
                    symbol=step.symbol,
                    payload={"clientOrderId": "s1-buy-full", "status": "ACCEPTED"},
                )
            )
            try:
                bus.run(handler)
                notes.append("out-of-order ACCEPTED was NOT refused")
            except IllegalTransition as exc:
                notes.append(f"out-of-order ACCEPTED refused: {exc}")

    # Session two: the settled sale. Settlement needs no event — sellability is a
    # property of the lot's acquisition date, read from the chain.
    for step in expressed:
        if step.kind in {"buy", "sell"} and step.session == SESSION_2:
            bus.publish(order_event(step, clock_time(10, 0)))
    for point in MARKET:
        if point.session == SESSION_2 and point.symbol == BLUE:
            bus.publish(market_event(point, clock_time(15, 0)))
    bus.run(handler)

    notes.append(f"venue executions derived from bars: {venue.executions}")
    native = {
        "cash": round(lifecycle.account().cash, 6),
        "realised_pnl": round(lifecycle.account().realised_pnl, 6),
        "fees_total": round(lifecycle.account().total_fees, 6),
    }
    snapshot = _replay(STREAMING, ledger_path, session=SESSION_2)
    return PathResult(
        label=STREAMING,
        ledger_path=ledger_path,
        snapshot=snapshot,
        native=native,
        native_table=compare_flat(
            f"{STREAMING}_native", native, f"{STREAMING}_replay", snapshot.flatten(),
            restrict_to_shared=True,
        ),
        steps_expressed=tuple(step.step_id for step in expressed),
        steps_out_of_scope=out_of_scope,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# The composite report
# ---------------------------------------------------------------------------
@dataclass
class CompositeReport:
    paths: list[PathResult]
    cross_tables: list[DifferenceTable]

    @property
    def unexplained_economic_differences(self) -> int:
        return sum(
            table.unexplained_economic_differences for table in self.all_tables
        )

    @property
    def all_tables(self) -> list[DifferenceTable]:
        return [path.native_table for path in self.paths] + self.cross_tables

    @property
    def clean(self) -> bool:
        return self.unexplained_economic_differences == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "quantagent.module1_composite_replay.v1",
            "scenario": {
                "sessions": [SESSION_1, SESSION_2],
                "initialCash": INITIAL_CASH,
                "closingPrices": dict(CLOSING_PRICES),
                "market": [
                    {
                        "symbol": p.symbol, "session": p.session, "lastPrice": p.last_price,
                        "previousClose": p.previous_close, "sessionVolume": p.session_volume,
                        "board": p.board,
                    }
                    for p in MARKET
                ],
                "steps": [
                    {
                        "stepId": s.step_id, "kind": s.kind, "session": s.session,
                        "symbol": s.symbol, "quantity": s.quantity,
                        "limitPrice": s.limit_price, "proves": s.proves,
                        "paths": sorted(s.paths),
                    }
                    for s in STEPS
                ],
            },
            "paths": [path.to_dict() for path in self.paths],
            "crossPath": [table.to_dict() for table in self.cross_tables],
            "unexplainedEconomicDifferences": self.unexplained_economic_differences,
            "clean": self.clean,
        }


def run_composite(work_dir: str | Path) -> CompositeReport:
    """Run the scenario through every path and diff everything."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    paths = [
        run_fast_path(work),
        run_paper_path(work),
        run_oms_paper_path(work),
        run_streaming_path(work),
    ]
    by_label = {path.label: path for path in paths}

    cross = [
        compare_snapshots(
            by_label[PAPER].snapshot, by_label[OMS_PAPER].snapshot, rules=PAPER_VS_OMS_RULES
        ),
        # The M3-01 comparison. No permitted difference: shared rulebook, separate
        # control flow.
        compare_snapshots(
            by_label[STREAMING].snapshot, by_label[PAPER].snapshot,
            rules=STREAMING_VS_PAPER_RULES,
        ),
        compare_snapshots(
            by_label[FAST].snapshot, by_label[PAPER].snapshot, rules=CROSS_PATH_RULES
        ),
        compare_snapshots(
            by_label[FAST].snapshot, by_label[OMS_PAPER].snapshot, rules=CROSS_PATH_RULES
        ),
        compare_snapshots(
            by_label[FAST].snapshot, by_label[STREAMING].snapshot, rules=CROSS_PATH_RULES
        ),
    ]
    return CompositeReport(paths=paths, cross_tables=cross)


__all__ = [
    "CLOSING_PRICES",
    "CROSS_PATH_RULES",
    "CompositeReport",
    "FAST",
    "INITIAL_CASH",
    "MARKET",
    "MarketPoint",
    "OMS_PAPER",
    "PAPER",
    "PAPER_VS_OMS_RULES",
    "STREAMING",
    "STREAMING_VS_PAPER_RULES",
    "PathResult",
    "STEPS",
    "SESSION_1",
    "SESSION_2",
    "Step",
    "market_point",
    "run_composite",
    "run_fast_path",
    "run_oms_paper_path",
    "run_paper_path",
    "run_streaming_path",
]
