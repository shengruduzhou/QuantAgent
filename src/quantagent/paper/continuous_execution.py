"""Consume pending T-close targets on the next observed paper session.

The consumer is deliberately conservative:

* a signal is executable only on the exact next proven session;
* canonical and operational account state must reconcile before an order is sent;
* one economic order flows through the existing OrderManager -> PaperBrokerAdapter
  -> PaperBroker chain and the shared canonical ledger;
* an execution-start record is durable before broker interaction, so a crash is
  resolved as indeterminate rather than retried blindly;
* caller-supplied or observed-panel session sets may drive research timing but
  are never promoted to authoritative shadow-calendar evidence without a
  provenance-backed exchange calendar artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from quantagent.backtest import ashare_rules as market_rules
from quantagent.domain.ledger import CanonicalLedger
from quantagent.domain.lineage import Lineage
from quantagent.execution.broker_base import Order as ExecutionOrder, OrderType
from quantagent.execution.order_manager import OrderManager, OrderManagerConfig
from quantagent.execution.paper_adapter import PaperBrokerAdapter
from quantagent.paper.broker import BrokerConfig, MarketSnapshot, PaperBroker
from quantagent.paper.execution_journal import PendingExecutionJournal
from quantagent.paper.ledger import EventLedger
from quantagent.paper.pending_signal import PendingPaperSignal, PendingPaperSignalStore
from quantagent.paper.recovery import recover, recover_from_canonical
from quantagent.quant_math.ashare import AshareRuleEngine


@dataclass(frozen=True)
class ContinuousPaperExecutionConfig:
    pending_signal_dir: str
    execution_journal_path: str
    canonical_ledger_path: str
    operational_ledger_path: str
    idempotency_path: str
    portfolio_id: str = "v7-paper"
    initial_cash: float = 1_000_000.0
    lot_size: int = 100
    min_order_value_yuan: float = 100.0
    max_participation_rate: float = 0.05
    execution_clock: str = "14:59:00+08:00"
    strategy_version: str = "v7_continuous_paper_v1"


@dataclass(frozen=True)
class ContinuousPaperExecutionResult:
    signal_date: str
    execution_date: str
    status: str
    pending_payload_sha256: str
    order_count: int
    fill_count: int
    nav_before: float | None
    nav_after: float | None
    calendar_assurance: str
    shadow_acceptance_calendar_eligible: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "signal_date": self.signal_date,
            "execution_date": self.execution_date,
            "status": self.status,
            "pending_payload_sha256": self.pending_payload_sha256,
            "order_count": self.order_count,
            "fill_count": self.fill_count,
            "nav_before": self.nav_before,
            "nav_after": self.nav_after,
            "calendar_assurance": self.calendar_assurance,
            "shadow_acceptance_calendar_eligible": self.shadow_acceptance_calendar_eligible,
            "reasons": list(self.reasons),
        }


class ContinuousPaperExecutionBlocked(RuntimeError):
    pass


def _normalise_sessions(values: Iterable[object]) -> tuple[str, ...]:
    parsed = pd.to_datetime(pd.Series(list(values)), errors="coerce").dropna()
    return tuple(sorted({pd.Timestamp(value).date().isoformat() for value in parsed}))


def _next_session(signal_date: str, sessions: Sequence[str]) -> str | None:
    for session in sessions:
        if session > signal_date:
            return session
    return None


def _pending_signals(store: PendingPaperSignalStore) -> list[PendingPaperSignal]:
    if not store.root.exists():
        return []
    signals: list[PendingPaperSignal] = []
    for path in sorted(store.root.glob("*.json")):
        signal = store.read(path.stem)
        if signal is not None:
            signals.append(signal)
    return signals


def _position_quantities(portfolio) -> dict[str, float]:
    return {
        str(symbol): float(position.total)
        for symbol, position in portfolio.positions.items()
        if not position.is_flat
    }


def _assert_recovered_account_consistent(
    canonical_state,
    operational_state,
    *,
    tolerance: float = 1e-6,
) -> None:
    # The operational ledger is legitimately empty before the first economic order.
    if operational_state.events_replayed == 0:
        return
    if (
        abs(
            float(canonical_state.portfolio.cash)
            - float(operational_state.portfolio.cash)
        )
        > tolerance
    ):
        raise ContinuousPaperExecutionBlocked(
            "canonical/operational paper cash reconciliation failed"
        )
    left = _position_quantities(canonical_state.portfolio)
    right = _position_quantities(operational_state.portfolio)
    for symbol in sorted(set(left) | set(right)):
        if abs(left.get(symbol, 0.0) - right.get(symbol, 0.0)) > tolerance:
            raise ContinuousPaperExecutionBlocked(
                f"canonical/operational paper position reconciliation failed for {symbol}"
            )


def _snapshot_clock(execution_clock: str) -> str:
    clock = str(execution_clock).strip()
    if not clock:
        raise ContinuousPaperExecutionBlocked("execution clock is empty")
    local_clock = clock.split("+", 1)[0]
    if len(local_clock) != 8 or local_clock[2] != ":" or local_clock[5] != ":":
        raise ContinuousPaperExecutionBlocked(
            f"invalid execution clock {execution_clock!r}; expected HH:MM:SS[+offset]"
        )
    return local_clock


def _execution_timestamp(execution_date: str, execution_clock: str) -> str:
    clock = str(execution_clock).strip()
    _snapshot_clock(clock)
    return f"{execution_date}T{clock}"


def _paper_board(symbol: str, rule_engine: AshareRuleEngine) -> str:
    """Translate portfolio-rule board names to paper microstructure board names.

    The portfolio rule engine and paper fill model deliberately use different
    vocabularies. Passing ``main_board`` or ``star`` straight into
    ``backtest.ashare_rules`` silently yields UNKNOWN_BOARD and disables the
    intended price-limit semantics, so the translation is explicit and
    unsupported non-equity instruments fail closed.
    """

    text = str(symbol).upper()
    board = rule_engine.infer_board(text)
    if board == "star":
        return market_rules.STAR
    if board == "chinext":
        return market_rules.CHINEXT
    if board == "bse":
        return market_rules.BSE
    if board == "main_board":
        if text.endswith(".SH"):
            return market_rules.SH_MAIN
        if text.endswith(".SZ"):
            return market_rules.SZ_MAIN
        raise ContinuousPaperExecutionBlocked(
            f"main-board symbol lacks an explicit SH/SZ exchange suffix: {symbol}"
        )
    raise ContinuousPaperExecutionBlocked(
        f"continuous A-share paper execution does not support instrument class "
        f"{board!r} for {symbol}"
    )


def _optional_finite(row: pd.Series, name: str) -> float | None:
    if name not in row.index or pd.isna(row[name]):
        return None
    value = float(pd.to_numeric(row[name], errors="coerce"))
    return value if np.isfinite(value) else None


def _market_state(
    market_panel: pd.DataFrame,
    *,
    execution_date: str,
    execution_clock: str,
    required_symbols: set[str],
) -> tuple[dict[tuple[str, str], MarketSnapshot], pd.Series]:
    data = market_panel.copy()
    required_columns = {"trade_date", "symbol", "close", "volume", "amount"}
    missing = sorted(required_columns - set(data.columns))
    if missing:
        raise ContinuousPaperExecutionBlocked(f"market panel missing columns: {missing}")

    data["trade_date"] = pd.to_datetime(
        data["trade_date"], errors="coerce"
    ).dt.normalize()
    data["symbol"] = data["symbol"].astype(str)
    data = data.dropna(subset=["trade_date", "symbol"]).sort_values(
        ["symbol", "trade_date"]
    )
    if data.duplicated(["trade_date", "symbol"]).any():
        raise ContinuousPaperExecutionBlocked("market panel has duplicate execution bars")

    day = data[data["trade_date"] == pd.Timestamp(execution_date)].copy()
    if day.empty:
        raise ContinuousPaperExecutionBlocked("execution session bar set is empty")
    available = set(day["symbol"])
    missing_symbols = sorted(required_symbols - available)
    if missing_symbols:
        raise ContinuousPaperExecutionBlocked(
            f"execution session missing required symbols: {missing_symbols[:10]}"
        )

    rule_engine = AshareRuleEngine()
    snapshots: dict[tuple[str, str], MarketSnapshot] = {}
    prices: dict[str, float] = {}
    clock = _snapshot_clock(execution_clock)

    for symbol in sorted(required_symbols):
        row = day.loc[day["symbol"] == symbol].iloc[0]
        close = float(pd.to_numeric(row["close"], errors="coerce"))
        volume = float(pd.to_numeric(row["volume"], errors="coerce"))
        amount = float(pd.to_numeric(row["amount"], errors="coerce"))
        if (
            not np.isfinite(close)
            or close <= 0
            or not np.isfinite(volume)
            or volume < 0
            or not np.isfinite(amount)
            or amount < 0
        ):
            raise ContinuousPaperExecutionBlocked(
                f"invalid execution market values for {symbol}"
            )

        prior = data[
            (data["symbol"] == symbol)
            & (data["trade_date"] < pd.Timestamp(execution_date))
        ]
        if prior.empty:
            raise ContinuousPaperExecutionBlocked(
                f"previous close unavailable for execution symbol {symbol}"
            )
        previous_close = float(
            pd.to_numeric(prior.iloc[-1]["close"], errors="coerce")
        )
        if not np.isfinite(previous_close) or previous_close <= 0:
            raise ContinuousPaperExecutionBlocked(
                f"invalid previous close for {symbol}"
            )

        sessions_since_listing = None
        if (
            "sessions_since_listing" in row.index
            and pd.notna(row["sessions_since_listing"])
        ):
            sessions_since_listing = int(row["sessions_since_listing"])

        snapshot = MarketSnapshot(
            symbol=symbol,
            trade_date=execution_date,
            last_price=close,
            previous_close=previous_close,
            session_volume=volume,
            board=_paper_board(symbol, rule_engine),
            clock=clock,
            is_suspended=bool(row.get("is_suspended", False)),
            is_st=bool(row.get("is_st", False)),
            sessions_since_listing=sessions_since_listing,
            high=_optional_finite(row, "high"),
            low=_optional_finite(row, "low"),
        )
        snapshots[(symbol, execution_date)] = snapshot
        prices[symbol] = close

    return snapshots, pd.Series(prices, dtype=float)


def _execution_orders(
    manager: OrderManager,
    *,
    target_weights: pd.Series,
    prices: pd.Series,
    nav: float,
    signal_id: str,
    execution_date: str,
    execution_clock: str,
) -> list[ExecutionOrder]:
    # target_weights_to_order_intents queries the adapter's recovered positions,
    # so these are true target deltas, not repeated gross target purchases.
    intents = manager.target_weights_to_order_intents(
        target_weights,
        prices,
        nav,
        signal_id=signal_id,
        risk_check_result="paper_simulator_admissibility_only",
    )
    timestamp = _execution_timestamp(execution_date, execution_clock)
    orders: list[ExecutionOrder] = []
    for intent in intents:
        stamped = replace(intent, timestamp=timestamp)
        orders.append(
            ExecutionOrder(
                client_order_id=stamped.intent_id,
                symbol=stamped.symbol,
                side=stamped.side,
                quantity=stamped.quantity,
                order_type=OrderType.LIMIT,
                price=stamped.reference_price,
                note="continuous_paper_target_reconcile",
                signal_id=stamped.signal_id,
                model_version=stamped.model_version,
                feature_version=stamped.feature_version,
                strategy_version=stamped.strategy_version,
                risk_check_result=stamped.risk_check_result,
                timestamp=stamped.timestamp,
            )
        )
    return orders


def execute_pending_for_session(
    as_of_date: str,
    market_panel: pd.DataFrame,
    *,
    config: ContinuousPaperExecutionConfig,
    authoritative_sessions: Sequence[object] | None = None,
) -> list[ContinuousPaperExecutionResult]:
    """Consume eligible pending signals for one actually observed session.

    ``authoritative_sessions`` is retained for API compatibility. It may define
    candidate shadow timing, but it is deliberately non-certifying until a
    versioned exchange-calendar artifact with provenance is wired into this path.
    """

    as_of = pd.Timestamp(as_of_date).date().isoformat()
    observed_sessions = _normalise_sessions(
        market_panel["trade_date"] if "trade_date" in market_panel.columns else ()
    )

    if authoritative_sessions is not None:
        sessions = _normalise_sessions(authoritative_sessions)
        calendar_assurance = "caller_supplied_session_set_unverified"
        acceptance_calendar_eligible = False
        if as_of not in sessions:
            raise ContinuousPaperExecutionBlocked(
                f"as_of date {as_of} is not in the supplied session set"
            )
        if as_of not in observed_sessions:
            raise ContinuousPaperExecutionBlocked(
                f"supplied session {as_of} has no observed market panel"
            )
    else:
        sessions = observed_sessions
        calendar_assurance = "observed_market_panel_only"
        acceptance_calendar_eligible = False
        if as_of not in sessions:
            return []

    store = PendingPaperSignalStore(config.pending_signal_dir)
    journal = PendingExecutionJournal(config.execution_journal_path)
    if not journal.verify():
        raise ContinuousPaperExecutionBlocked(
            "pending execution journal verification failed"
        )

    results: list[ContinuousPaperExecutionResult] = []
    for pending in _pending_signals(store):
        terminal = journal.terminal(pending.payload_sha256)
        if terminal is not None:
            continue

        next_date = _next_session(pending.signal_date, sessions)
        if next_date is None or next_date > as_of:
            continue

        if journal.has_unresolved_start(pending.payload_sha256):
            journal.append(
                pending_payload_sha256=pending.payload_sha256,
                signal_date=pending.signal_date,
                execution_date=next_date,
                status="execution_indeterminate",
                details={
                    "reason": (
                        "prior execution_started has no terminal outcome; "
                        "automatic retry prohibited"
                    )
                },
            )
            results.append(
                ContinuousPaperExecutionResult(
                    signal_date=pending.signal_date,
                    execution_date=next_date,
                    status="execution_indeterminate",
                    pending_payload_sha256=pending.payload_sha256,
                    order_count=0,
                    fill_count=0,
                    nav_before=None,
                    nav_after=None,
                    calendar_assurance=calendar_assurance,
                    shadow_acceptance_calendar_eligible=acceptance_calendar_eligible,
                    reasons=("unresolved prior execution attempt",),
                )
            )
            continue

        if next_date < as_of:
            journal.append(
                pending_payload_sha256=pending.payload_sha256,
                signal_date=pending.signal_date,
                execution_date=next_date,
                status="missed_execution_session",
                details={
                    "observed_as_of": as_of,
                    "reason": (
                        "consumer did not run on exact next session; "
                        "retroactive shadow fill prohibited"
                    ),
                },
            )
            results.append(
                ContinuousPaperExecutionResult(
                    signal_date=pending.signal_date,
                    execution_date=next_date,
                    status="missed_execution_session",
                    pending_payload_sha256=pending.payload_sha256,
                    order_count=0,
                    fill_count=0,
                    nav_before=None,
                    nav_after=None,
                    calendar_assurance=calendar_assurance,
                    shadow_acceptance_calendar_eligible=acceptance_calendar_eligible,
                    reasons=("retroactive execution prohibited",),
                )
            )
            continue

        canonical = CanonicalLedger(config.canonical_ledger_path)
        if not canonical.verify()["valid"]:
            raise ContinuousPaperExecutionBlocked(
                "canonical paper ledger verification failed"
            )
        canonical_state = recover_from_canonical(
            config.canonical_ledger_path,
            portfolio_id=config.portfolio_id,
            initial_cash=config.initial_cash,
            as_of_session=as_of,
        )
        operational_ledger = EventLedger(config.operational_ledger_path)
        operational_state = recover(
            operational_ledger,
            portfolio_id=config.portfolio_id,
            initial_cash=config.initial_cash,
        )
        _assert_recovered_account_consistent(canonical_state, operational_state)

        if canonical_state.open_orders() or operational_state.open_orders():
            raise ContinuousPaperExecutionBlocked(
                "recovered paper account has unresolved open orders; reconciliation required"
            )
        if operational_state.killed:
            raise ContinuousPaperExecutionBlocked(
                f"paper kill switch is active: "
                f"{operational_state.kill_reason or 'unknown'}"
            )

        held_symbols = set(_position_quantities(canonical_state.portfolio))
        target_symbols = {
            symbol
            for symbol, weight in pending.target_weights.items()
            if abs(float(weight)) > 1e-12
        }
        required_symbols = held_symbols | target_symbols
        snapshots, prices = _market_state(
            market_panel,
            execution_date=as_of,
            execution_clock=config.execution_clock,
            required_symbols=required_symbols,
        )
        nav_before = canonical_state.portfolio.equity(prices.to_dict())
        if not np.isfinite(nav_before) or nav_before <= 0:
            raise ContinuousPaperExecutionBlocked(
                "recovered paper account NAV is invalid"
            )

        book = canonical.replay_book() if len(canonical) else None
        run_id = f"continuous-paper:{config.portfolio_id}"
        lineage = Lineage(
            run_id=run_id,
            strategy_version_id=config.strategy_version,
        )
        broker = PaperBroker(
            event_ledger=operational_ledger,
            portfolio=canonical_state.portfolio,
            run_id=run_id,
            canonical_ledger=canonical,
            book=book,
            lineage=lineage,
            config=BrokerConfig(
                participation_cap=config.max_participation_rate,
            ),
        )
        market_source = lambda symbol, trade_date: snapshots.get(
            (str(symbol), str(trade_date))
        )
        adapter = PaperBrokerAdapter(
            broker=broker,
            market_source=market_source,
        )
        manager = OrderManager(
            broker=adapter,
            lineage=lineage,
            idempotency_path=config.idempotency_path,
            canonical_ledger=canonical,
            order_book=broker.book,
            config=OrderManagerConfig(
                lot_size=config.lot_size,
                min_order_value_yuan=config.min_order_value_yuan,
                max_participation_rate=config.max_participation_rate,
                strategy_version=config.strategy_version,
            ),
        )

        target = (
            pd.Series(pending.target_weights, dtype=float)
            .reindex(prices.index)
            .fillna(0.0)
        )
        orders = _execution_orders(
            manager,
            target_weights=target,
            prices=prices,
            nav=nav_before,
            signal_id=f"pending:{pending.payload_sha256}",
            execution_date=as_of,
            execution_clock=config.execution_clock,
        )

        journal.append(
            pending_payload_sha256=pending.payload_sha256,
            signal_date=pending.signal_date,
            execution_date=as_of,
            status="execution_started",
            details={
                "order_count": len(orders),
                "canonical_records_before": len(canonical),
                "calendar_assurance": calendar_assurance,
                "shadow_acceptance_calendar_eligible": acceptance_calendar_eligible,
                "execution_clock": config.execution_clock,
                "production_pretrade_risk_certified": False,
                "risk_scope": "paper_simulator_admissibility_only",
            },
        )

        fills_before = len(broker.fills)
        states = manager.submit_orders(orders)
        manager.cancel_all_open()
        broker.mark_to_market(prices.to_dict(), market_time=as_of)
        nav_after = broker.portfolio.equity(prices.to_dict())
        fill_count = len(broker.fills) - fills_before
        statuses = [state.status.value for state in states]

        # Session settlement is part of the economic outcome. It must happen
        # before the terminal journal record; otherwise a crash after "observed"
        # but before close_session would make the retry guard skip an unsettled
        # account forever.
        broker.close_session(as_of)

        terminal_status = "execution_observed"
        if orders and fill_count == 0 and statuses and all(
            status.lower() in {"rejected", "cancelled"} for status in statuses
        ):
            terminal_status = "execution_blocked"

        journal.append(
            pending_payload_sha256=pending.payload_sha256,
            signal_date=pending.signal_date,
            execution_date=as_of,
            status=terminal_status,
            details={
                "order_count": len(orders),
                "fill_count": fill_count,
                "order_statuses": statuses,
                "nav_before": nav_before,
                "nav_after": nav_after,
                "calendar_assurance": calendar_assurance,
                "shadow_acceptance_calendar_eligible": acceptance_calendar_eligible,
                "session_closed": True,
                "execution_clock": config.execution_clock,
                "production_pretrade_risk_certified": False,
                "risk_scope": "paper_simulator_admissibility_only",
            },
        )
        results.append(
            ContinuousPaperExecutionResult(
                signal_date=pending.signal_date,
                execution_date=as_of,
                status=terminal_status,
                pending_payload_sha256=pending.payload_sha256,
                order_count=len(orders),
                fill_count=fill_count,
                nav_before=float(nav_before),
                nav_after=float(nav_after),
                calendar_assurance=calendar_assurance,
                shadow_acceptance_calendar_eligible=acceptance_calendar_eligible,
                reasons=(),
            )
        )

    return results


__all__ = [
    "ContinuousPaperExecutionConfig",
    "ContinuousPaperExecutionResult",
    "ContinuousPaperExecutionBlocked",
    "execute_pending_for_session",
]
