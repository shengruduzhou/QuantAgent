"""Consume pending T-close targets on the next observed paper session.

The consumer is deliberately conservative:

* a signal is executable only on the exact next proven session;
* canonical and operational account state must reconcile before an order is sent;
* one economic order flows through OrderManager -> PaperBrokerAdapter -> PaperBroker;
* ``execution_started`` is durable before broker interaction;
* every terminal outcome is bound to a freshly reopened canonical-ledger prefix;
* unresolved or indeterminate attempts freeze the whole economic account even if
  the pending-signal artifact has disappeared;
* an explicit append-only reconciliation record is the only way to clear an
  indeterminate freeze without rewriting history;
* caller/observed session sets remain non-authoritative shadow-calendar evidence;
* every non-terminal signal must match the immutable paper-account identity;
* exported economic execution/reconciliation boundaries own the same canonical-
  account cross-process lock used by target freezing, so non-CLI callers cannot
  bypass serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from quantagent.backtest import ashare_rules as market_rules
from quantagent.data.intraday_sessions import (
    ExecutionPriceProvenanceError,
    assert_raw_execution_prices,
)
from quantagent.domain.ledger import CanonicalLedger
from quantagent.domain.lineage import Lineage
from quantagent.execution.broker_base import Order as ExecutionOrder, OrderType
from quantagent.execution.order_manager import OrderManager, OrderManagerConfig
from quantagent.execution.paper_adapter import PaperBrokerAdapter
from quantagent.paper.account_identity import (
    PaperAccountIdentityError,
    ensure_paper_account_identity,
)
from quantagent.paper.account_lock import paper_account_lock
from quantagent.paper.broker import BrokerConfig, MarketSnapshot, PaperBroker
from quantagent.paper.canonical_receipt import (
    CanonicalPrefixReceiptError,
    build_canonical_prefix_index,
    build_canonical_prefix_receipt,
    canonical_snapshot,
    verify_canonical_prefix_receipt,
)
from quantagent.paper.execution_journal import (
    LEGACY_BINDING_STATUS,
    RECONCILIATION_STATUS,
    TERMINAL_OUTCOMES,
    PendingExecutionJournal,
)
from quantagent.paper.legacy_terminal_binding import (
    LegacyTerminalBindingError,
    verify_legacy_terminal_binding,
)
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
    account_identity_path: str | None = None
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
    operational_positions = _position_quantities(operational_state.portfolio)
    initial_cash = float(getattr(operational_state.portfolio, "initial_cash", 0.0))
    operational_cash = float(operational_state.portfolio.cash)
    has_operational_economics = bool(
        operational_state.orders
        or operational_state.fills
        or operational_positions
        or abs(operational_cash - initial_cash) > tolerance
    )
    if not has_operational_economics:
        return

    if abs(float(canonical_state.portfolio.cash) - operational_cash) > tolerance:
        raise ContinuousPaperExecutionBlocked(
            "canonical/operational paper cash reconciliation failed"
        )
    left = _position_quantities(canonical_state.portfolio)
    right = operational_positions
    for symbol in sorted(set(left) | set(right)):
        if abs(left.get(symbol, 0.0) - right.get(symbol, 0.0)) > tolerance:
            raise ContinuousPaperExecutionBlocked(
                f"canonical/operational paper position reconciliation failed for {symbol}"
            )


def _account_state_sha256(state) -> str:
    payload = {
        "cash": format(float(state.portfolio.cash), ".10f"),
        "positions": {
            symbol: format(quantity, ".10f")
            for symbol, quantity in sorted(_position_quantities(state.portfolio).items())
        },
        "open_order_ids": sorted(
            str(getattr(order, "order_id", getattr(order, "client_order_id", "")))
            for order in state.open_orders()
        ),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def _required_market_flag(row: pd.Series, name: str) -> bool:
    if name not in row.index or pd.isna(row[name]):
        raise ContinuousPaperExecutionBlocked(
            f"execution market row missing explicit {name}"
        )
    value = row[name]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        raise ContinuousPaperExecutionBlocked(
            f"invalid boolean execution flag {name}={value!r}"
        )
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and np.isfinite(float(numeric)) and float(numeric) in {0.0, 1.0}:
        return bool(int(float(numeric)))
    raise ContinuousPaperExecutionBlocked(
        f"invalid boolean execution flag {name}={value!r}"
    )


def _assert_execution_price_provenance(rows: pd.DataFrame, *, symbol: str) -> None:
    try:
        assert_raw_execution_prices(rows)
    except ExecutionPriceProvenanceError as exc:
        raise ContinuousPaperExecutionBlocked(
            f"execution price provenance failed for {symbol}: {exc}"
        ) from exc


def _market_state(
    market_panel: pd.DataFrame,
    *,
    execution_date: str,
    execution_clock: str,
    required_symbols: set[str],
) -> tuple[dict[tuple[str, str], MarketSnapshot], pd.Series]:
    data = market_panel.copy()
    required_columns = {
        "trade_date",
        "symbol",
        "close",
        "volume",
        "amount",
        "is_suspended",
        "is_st",
    }
    missing = sorted(required_columns - set(data.columns))
    if missing:
        raise ContinuousPaperExecutionBlocked(f"market panel missing columns: {missing}")

    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    data["symbol"] = data["symbol"].astype(str)
    data = data.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"])
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
        previous_row = prior.iloc[-1]
        previous_close = float(pd.to_numeric(previous_row["close"], errors="coerce"))
        if not np.isfinite(previous_close) or previous_close <= 0:
            raise ContinuousPaperExecutionBlocked(f"invalid previous close for {symbol}")

        _assert_execution_price_provenance(pd.DataFrame([previous_row, row]), symbol=symbol)
        is_suspended = _required_market_flag(row, "is_suspended")
        is_st = _required_market_flag(row, "is_st")
        sessions_since_listing = None
        if "sessions_since_listing" in row.index and pd.notna(row["sessions_since_listing"]):
            sessions_since_listing = int(row["sessions_since_listing"])

        snapshot = MarketSnapshot(
            symbol=symbol,
            trade_date=execution_date,
            last_price=close,
            previous_close=previous_close,
            session_volume=volume,
            board=_paper_board(symbol, rule_engine),
            clock=clock,
            is_suspended=is_suspended,
            is_st=is_st,
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


def _verify_terminal_canonical_binding(
    *,
    terminal,
    legacy_binding,
    canonical_ledger_path: str,
    expected_target_weights_sha256: str | None = None,
    expected_paper_account_identity_sha256: str | None = None,
):
    receipt = dict(terminal.details or {}).get("canonical_prefix_receipt")
    try:
        verification = verify_canonical_prefix_receipt(
            receipt,
            ledger_or_path=canonical_ledger_path,
            expected_target_weights_sha256=expected_target_weights_sha256,
            expected_paper_account_identity_sha256=expected_paper_account_identity_sha256,
        )
    except CanonicalPrefixReceiptError as exc:
        raise ContinuousPaperExecutionBlocked(
            "terminal paper execution evidence no longer matches the canonical "
            f"economic ledger: {exc}"
        ) from exc
    if verification.bound:
        return verification
    if expected_paper_account_identity_sha256 is None:
        raise ContinuousPaperExecutionBlocked(
            "legacy terminal verification requires the immutable paper-account identity"
        )
    try:
        verify_legacy_terminal_binding(
            terminal,
            legacy_binding,
            prefix_index=build_canonical_prefix_index(canonical_ledger_path),
            expected_paper_account_identity_sha256=expected_paper_account_identity_sha256,
            expected_target_weights_sha256=expected_target_weights_sha256,
        )
    except LegacyTerminalBindingError as exc:
        raise ContinuousPaperExecutionBlocked(
            "legacy terminal is not covered by a valid append-only operator binding: "
            f"{exc}"
        ) from exc
    return verification


def _reconciliation_is_valid(
    *,
    terminal,
    reconciliation,
    canonical_ledger_path: str,
    paper_account_identity_sha256: str,
) -> bool:
    if reconciliation is None:
        return False
    details = dict(reconciliation.details or {})
    if str(details.get("indeterminate_record_sha256") or "") != terminal.record_sha256:
        return False
    if str(details.get("paper_account_identity_sha256") or "") != paper_account_identity_sha256:
        return False
    try:
        record_count = int(details["canonical_records"])
        head = str(details["canonical_head"])
        index = build_canonical_prefix_index(canonical_ledger_path)
        return index.head_at(record_count) == head
    except (CanonicalPrefixReceiptError, KeyError, TypeError, ValueError):
        return False


def _assert_account_execution_state_resolved(
    journal: PendingExecutionJournal,
    *,
    canonical_ledger_path: str,
    paper_account_identity_sha256: str,
) -> None:
    """Freeze on any unresolved start or unreconciled indeterminate outcome."""

    records = journal.records()
    by_payload: dict[str, list[object]] = {}
    for record in records:
        by_payload.setdefault(record.pending_payload_sha256, []).append(record)

    for payload, history in by_payload.items():
        starts = [row for row in history if row.status == "execution_started"]
        terminals = [row for row in history if row.status in TERMINAL_OUTCOMES]
        if starts and not terminals:
            raise ContinuousPaperExecutionBlocked(
                "paper account has an unresolved execution_started record for "
                f"{payload}; pending artifact presence is irrelevant. Explicit "
                "account reconciliation is required before later signals can trade"
            )
        if len(terminals) > 1:
            raise ContinuousPaperExecutionBlocked(
                f"paper account has multiple terminal outcomes for {payload}"
            )
        if not terminals:
            continue
        terminal = terminals[0]
        if terminal.status == "execution_indeterminate":
            reconciliation = journal.reconciliation(payload)
            if not _reconciliation_is_valid(
                terminal=terminal,
                reconciliation=reconciliation,
                canonical_ledger_path=canonical_ledger_path,
                paper_account_identity_sha256=paper_account_identity_sha256,
            ):
                raise ContinuousPaperExecutionBlocked(
                    "paper account has an unreconciled execution_indeterminate outcome; "
                    "explicit canonical/operational reconciliation is required before "
                    "any later signal can trade"
                )
        _verify_terminal_canonical_binding(
            terminal=terminal,
            legacy_binding=journal.legacy_binding(payload),
            canonical_ledger_path=canonical_ledger_path,
            expected_paper_account_identity_sha256=paper_account_identity_sha256,
        )


def _same_prefix_receipt(
    *,
    canonical_ledger_path: str,
    target_weights_sha256: str,
    paper_account_identity_sha256: str,
) -> dict[str, object]:
    before_records, before_head = canonical_snapshot(canonical_ledger_path)
    return build_canonical_prefix_receipt(
        ledger=canonical_ledger_path,
        canonical_before_records=before_records,
        canonical_before_head=before_head,
        target_weights_sha256=target_weights_sha256,
        paper_account_identity_sha256=paper_account_identity_sha256,
    )


def bind_legacy_terminal_account(
    *,
    config: ContinuousPaperExecutionConfig,
    pending_payload_sha256: str,
    as_of_date: str,
    reason: str,
) -> dict[str, object]:
    """Append a lower-assurance binding without rewriting the legacy terminal."""

    with paper_account_lock(config.canonical_ledger_path):
        return _bind_legacy_terminal_account_locked(
            config=config,
            pending_payload_sha256=pending_payload_sha256,
            as_of_date=as_of_date,
            reason=reason,
        )


def _bind_legacy_terminal_account_locked(
    *,
    config: ContinuousPaperExecutionConfig,
    pending_payload_sha256: str,
    as_of_date: str,
    reason: str,
) -> dict[str, object]:
    binding_reason = str(reason).strip()
    if not binding_reason:
        raise ContinuousPaperExecutionBlocked("legacy binding reason must be non-empty")
    payload = str(pending_payload_sha256).strip()
    if len(payload) != 64:
        raise ContinuousPaperExecutionBlocked(
            "legacy binding requires a 64-character pending payload digest"
        )
    as_of = pd.Timestamp(as_of_date).date().isoformat()
    try:
        identity = ensure_paper_account_identity(
            canonical_ledger_path=config.canonical_ledger_path,
            portfolio_id=config.portfolio_id,
            initial_cash=config.initial_cash,
            identity_path=config.account_identity_path,
        )
    except PaperAccountIdentityError as exc:
        raise ContinuousPaperExecutionBlocked(
            f"paper account identity verification failed: {exc}"
        ) from exc
    journal = PendingExecutionJournal(config.execution_journal_path)
    if not journal.verify():
        raise ContinuousPaperExecutionBlocked("pending execution journal verification failed")
    terminal = journal.terminal(payload)
    if terminal is None:
        raise ContinuousPaperExecutionBlocked(
            f"no terminal execution evidence exists for pending payload {payload}"
        )
    if dict(terminal.details or {}).get("canonical_prefix_receipt") is not None:
        raise ContinuousPaperExecutionBlocked(
            "terminal already carries an original execution-time canonical-prefix receipt"
        )
    canonical_state = recover_from_canonical(
        config.canonical_ledger_path,
        portfolio_id=identity.portfolio_id,
        initial_cash=identity.initial_cash,
        as_of_session=as_of,
    )
    operational_state = recover(
        EventLedger(config.operational_ledger_path),
        portfolio_id=identity.portfolio_id,
        initial_cash=identity.initial_cash,
    )
    _assert_recovered_account_consistent(canonical_state, operational_state)
    if canonical_state.open_orders() or operational_state.open_orders():
        raise ContinuousPaperExecutionBlocked(
            "cannot bind legacy terminal while unresolved open orders remain"
        )
    target_sha = str(dict(terminal.details or {}).get("target_weights_sha256") or "")
    if not target_sha:
        pending = PendingPaperSignalStore(config.pending_signal_dir).read(terminal.signal_date)
        if pending is not None and pending.payload_sha256 == payload:
            target_sha = pending.target_weights_sha256
    if len(target_sha) != 64:
        raise ContinuousPaperExecutionBlocked(
            "legacy terminal cannot be safely bound without its exact target digest"
        )
    prefix_index = build_canonical_prefix_index(config.canonical_ledger_path)
    details = {
        "terminal_record_sha256": terminal.record_sha256,
        "paper_account_identity_sha256": identity.payload_sha256,
        "target_weights_sha256": target_sha,
        "canonical_records": prefix_index.record_count,
        "canonical_head": prefix_index.current_head,
        "account_state_sha256": _account_state_sha256(canonical_state),
        "reconciled_as_of": as_of,
        "reason": binding_reason,
        "assurance": "operator_reconciled_legacy_terminal_v1",
    }
    existing = journal.legacy_binding(payload)
    if existing is not None:
        try:
            verify_legacy_terminal_binding(
                terminal,
                existing,
                prefix_index=prefix_index,
                expected_paper_account_identity_sha256=identity.payload_sha256,
                expected_target_weights_sha256=target_sha,
            )
        except LegacyTerminalBindingError as exc:
            raise ContinuousPaperExecutionBlocked(
                f"existing legacy terminal binding is invalid: {exc}"
            ) from exc
        return existing.to_dict()
    return journal.append(
        pending_payload_sha256=payload,
        signal_date=terminal.signal_date,
        execution_date=terminal.execution_date,
        status=LEGACY_BINDING_STATUS,
        details=details,
    ).to_dict()


def reconcile_indeterminate_account(
    *,
    config: ContinuousPaperExecutionConfig,
    as_of_date: str,
    reason: str,
) -> list[dict[str, object]]:
    """Reconcile uncertain economics under the canonical-account lock."""

    with paper_account_lock(config.canonical_ledger_path):
        return _reconcile_indeterminate_account_locked(
            config=config,
            as_of_date=as_of_date,
            reason=reason,
        )


def _reconcile_indeterminate_account_locked(
    *,
    config: ContinuousPaperExecutionConfig,
    as_of_date: str,
    reason: str,
) -> list[dict[str, object]]:
    """Explicitly reconcile and append evidence that can clear account freeze.

    The public wrapper owns the account-wide cross-process lock. This inner
    function never deletes/replaces an indeterminate record. It first proves
    canonical and operational paper state agree and have no open orders. Any
    dangling ``execution_started`` is converted to ``execution_indeterminate``;
    each indeterminate terminal then receives one append-only
    ``execution_reconciled`` record bound to the terminal hash, current account
    identity, canonical prefix and reconciled account-state digest.
    """

    reconciliation_reason = str(reason).strip()
    if not reconciliation_reason:
        raise ContinuousPaperExecutionBlocked("reconciliation reason must be non-empty")
    as_of = pd.Timestamp(as_of_date).date().isoformat()
    try:
        identity = ensure_paper_account_identity(
            canonical_ledger_path=config.canonical_ledger_path,
            portfolio_id=config.portfolio_id,
            initial_cash=config.initial_cash,
            identity_path=config.account_identity_path,
        )
    except PaperAccountIdentityError as exc:
        raise ContinuousPaperExecutionBlocked(
            f"paper account identity verification failed: {exc}"
        ) from exc

    journal = PendingExecutionJournal(config.execution_journal_path)
    if not journal.verify():
        raise ContinuousPaperExecutionBlocked("pending execution journal verification failed")

    canonical_state = recover_from_canonical(
        config.canonical_ledger_path,
        portfolio_id=identity.portfolio_id,
        initial_cash=identity.initial_cash,
        as_of_session=as_of,
    )
    operational_state = recover(
        EventLedger(config.operational_ledger_path),
        portfolio_id=identity.portfolio_id,
        initial_cash=identity.initial_cash,
    )
    _assert_recovered_account_consistent(canonical_state, operational_state)
    if canonical_state.open_orders() or operational_state.open_orders():
        raise ContinuousPaperExecutionBlocked(
            "cannot reconcile indeterminate account while open orders remain"
        )

    records = journal.records()
    by_payload: dict[str, list[object]] = {}
    for record in records:
        by_payload.setdefault(record.pending_payload_sha256, []).append(record)

    appended: list[dict[str, object]] = []
    for payload, history in by_payload.items():
        starts = [row for row in history if row.status == "execution_started"]
        terminals = [row for row in history if row.status in TERMINAL_OUTCOMES]
        if starts and not terminals:
            start = starts[-1]
            details = dict(start.details or {})
            target_sha = str(details.get("target_weights_sha256") or "")
            before_records = details.get("canonical_records_before")
            before_head = str(details.get("canonical_head_before") or "")
            if not target_sha or before_records is None or not before_head:
                raise ContinuousPaperExecutionBlocked(
                    "dangling execution_started lacks canonical/target evidence "
                    f"required for reconciliation: {payload}"
                )
            try:
                receipt = build_canonical_prefix_receipt(
                    ledger=config.canonical_ledger_path,
                    canonical_before_records=int(before_records),
                    canonical_before_head=before_head,
                    target_weights_sha256=target_sha,
                    paper_account_identity_sha256=identity.payload_sha256,
                )
            except CanonicalPrefixReceiptError as exc:
                raise ContinuousPaperExecutionBlocked(
                    f"cannot bind dangling execution start {payload}: {exc}"
                ) from exc
            terminal = journal.append(
                pending_payload_sha256=payload,
                signal_date=start.signal_date,
                execution_date=start.execution_date,
                status="execution_indeterminate",
                details={
                    "paper_account_identity_sha256": identity.payload_sha256,
                    "target_weights_sha256": target_sha,
                    "canonical_prefix_receipt": receipt,
                    "reason": "explicit reconciliation converted dangling execution_started",
                },
            )
            terminals = [terminal]

        if not terminals:
            continue
        terminal = terminals[0]
        if terminal.status != "execution_indeterminate":
            continue
        existing = journal.reconciliation(payload)
        if existing is not None:
            if not _reconciliation_is_valid(
                terminal=terminal,
                reconciliation=existing,
                canonical_ledger_path=config.canonical_ledger_path,
                paper_account_identity_sha256=identity.payload_sha256,
            ):
                raise ContinuousPaperExecutionBlocked(
                    f"existing reconciliation evidence is invalid for {payload}"
                )
            continue

        canonical_records, canonical_head = canonical_snapshot(config.canonical_ledger_path)
        state_sha = _account_state_sha256(canonical_state)
        reconciliation = journal.append(
            pending_payload_sha256=payload,
            signal_date=terminal.signal_date,
            execution_date=terminal.execution_date,
            status=RECONCILIATION_STATUS,
            details={
                "indeterminate_record_sha256": terminal.record_sha256,
                "paper_account_identity_sha256": identity.payload_sha256,
                "canonical_records": canonical_records,
                "canonical_head": canonical_head,
                "account_state_sha256": state_sha,
                "reconciled_as_of": as_of,
                "reason": reconciliation_reason,
            },
        )
        appended.append(reconciliation.to_dict())

    _assert_account_execution_state_resolved(
        journal,
        canonical_ledger_path=config.canonical_ledger_path,
        paper_account_identity_sha256=identity.payload_sha256,
    )
    return appended


def execute_pending_for_session(
    as_of_date: str,
    market_panel: pd.DataFrame,
    *,
    config: ContinuousPaperExecutionConfig,
    authoritative_sessions: Sequence[object] | None = None,
) -> list[ContinuousPaperExecutionResult]:
    """Execute one session under the canonical-account cross-process lock.

    Lock ownership lives here rather than in a CLI adapter so direct library,
    daemon or future API callers cannot mutate the economic account outside the
    same critical section used by daily target freezing.
    """

    with paper_account_lock(config.canonical_ledger_path):
        return _execute_pending_for_session_locked(
            as_of_date,
            market_panel,
            config=config,
            authoritative_sessions=authoritative_sessions,
        )


def _execute_pending_for_session_locked(
    as_of_date: str,
    market_panel: pd.DataFrame,
    *,
    config: ContinuousPaperExecutionConfig,
    authoritative_sessions: Sequence[object] | None = None,
) -> list[ContinuousPaperExecutionResult]:
    as_of = pd.Timestamp(as_of_date).date().isoformat()
    try:
        account_identity = ensure_paper_account_identity(
            canonical_ledger_path=config.canonical_ledger_path,
            portfolio_id=config.portfolio_id,
            initial_cash=config.initial_cash,
            identity_path=config.account_identity_path,
        )
    except PaperAccountIdentityError as exc:
        raise ContinuousPaperExecutionBlocked(
            f"paper account identity verification failed: {exc}"
        ) from exc

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
        raise ContinuousPaperExecutionBlocked("pending execution journal verification failed")
    _assert_account_execution_state_resolved(
        journal,
        canonical_ledger_path=config.canonical_ledger_path,
        paper_account_identity_sha256=account_identity.payload_sha256,
    )

    results: list[ContinuousPaperExecutionResult] = []
    for pending in _pending_signals(store):
        terminal = journal.terminal(pending.payload_sha256)
        if terminal is not None:
            _verify_terminal_canonical_binding(
                terminal=terminal,
                legacy_binding=journal.legacy_binding(pending.payload_sha256),
                canonical_ledger_path=config.canonical_ledger_path,
                expected_target_weights_sha256=pending.target_weights_sha256,
                expected_paper_account_identity_sha256=account_identity.payload_sha256,
            )
            continue

        pending_identity_sha = str(
            pending.source_lineage.get("paper_account_identity_sha256", "")
        )
        if pending_identity_sha != account_identity.payload_sha256:
            raise ContinuousPaperExecutionBlocked(
                "pending signal is missing or mismatched paper-account identity; "
                "regenerate/reconcile the signal instead of executing it"
            )

        next_date = _next_session(pending.signal_date, sessions)
        if next_date is None or next_date > as_of:
            continue

        # This should already have been caught by the account-wide scan above;
        # keep the local guard so a future refactor cannot reintroduce a bypass.
        if journal.has_unresolved_start(pending.payload_sha256):
            raise ContinuousPaperExecutionBlocked(
                "unresolved execution_started requires explicit account reconciliation"
            )

        if next_date < as_of:
            try:
                receipt = _same_prefix_receipt(
                    canonical_ledger_path=config.canonical_ledger_path,
                    target_weights_sha256=pending.target_weights_sha256,
                    paper_account_identity_sha256=account_identity.payload_sha256,
                )
            except CanonicalPrefixReceiptError as exc:
                raise ContinuousPaperExecutionBlocked(
                    f"cannot certify missed-session canonical prefix: {exc}"
                ) from exc
            journal.append(
                pending_payload_sha256=pending.payload_sha256,
                signal_date=pending.signal_date,
                execution_date=next_date,
                status="missed_execution_session",
                details={
                    "paper_account_identity_sha256": account_identity.payload_sha256,
                    "target_weights_sha256": pending.target_weights_sha256,
                    "canonical_prefix_receipt": receipt,
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

        try:
            canonical_before_records, canonical_before_head = canonical_snapshot(
                config.canonical_ledger_path
            )
        except CanonicalPrefixReceiptError as exc:
            raise ContinuousPaperExecutionBlocked(
                f"canonical paper ledger verification failed: {exc}"
            ) from exc
        canonical = CanonicalLedger(config.canonical_ledger_path)
        canonical_state = recover_from_canonical(
            config.canonical_ledger_path,
            portfolio_id=account_identity.portfolio_id,
            initial_cash=account_identity.initial_cash,
            as_of_session=as_of,
        )
        operational_ledger = EventLedger(config.operational_ledger_path)
        operational_state = recover(
            operational_ledger,
            portfolio_id=account_identity.portfolio_id,
            initial_cash=account_identity.initial_cash,
        )
        _assert_recovered_account_consistent(canonical_state, operational_state)

        if canonical_state.open_orders() or operational_state.open_orders():
            raise ContinuousPaperExecutionBlocked(
                "recovered paper account has unresolved open orders; reconciliation required"
            )
        if operational_state.killed:
            raise ContinuousPaperExecutionBlocked(
                f"paper kill switch is active: {operational_state.kill_reason or 'unknown'}"
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
            raise ContinuousPaperExecutionBlocked("recovered paper account NAV is invalid")

        book = canonical.replay_book() if len(canonical) else None
        run_id = f"continuous-paper:{account_identity.portfolio_id}"
        lineage = Lineage(run_id=run_id, strategy_version_id=config.strategy_version)
        broker = PaperBroker(
            event_ledger=operational_ledger,
            portfolio=canonical_state.portfolio,
            run_id=run_id,
            canonical_ledger=canonical,
            book=book,
            lineage=lineage,
            config=BrokerConfig(participation_cap=config.max_participation_rate),
        )
        market_source = lambda symbol, trade_date: snapshots.get((str(symbol), str(trade_date)))
        adapter = PaperBrokerAdapter(broker=broker, market_source=market_source)
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

        target = pd.Series(pending.target_weights, dtype=float).reindex(prices.index).fillna(0.0)
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
                "canonical_records_before": canonical_before_records,
                "canonical_head_before": canonical_before_head,
                "target_weights_sha256": pending.target_weights_sha256,
                "paper_account_identity_sha256": account_identity.payload_sha256,
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
        broker.close_session(as_of)

        terminal_status = "execution_observed"
        if orders and fill_count == 0 and statuses and all(
            status.lower() in {"rejected", "cancelled"} for status in statuses
        ):
            terminal_status = "execution_blocked"

        try:
            # Reopen durable bytes after broker/session close. Do not seal a
            # terminal from the long-lived pre-execution CanonicalLedger cache.
            canonical_receipt = build_canonical_prefix_receipt(
                ledger=config.canonical_ledger_path,
                canonical_before_records=canonical_before_records,
                canonical_before_head=canonical_before_head,
                target_weights_sha256=pending.target_weights_sha256,
                paper_account_identity_sha256=account_identity.payload_sha256,
            )
        except CanonicalPrefixReceiptError as exc:
            raise ContinuousPaperExecutionBlocked(
                f"canonical terminal receipt creation failed: {exc}"
            ) from exc

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
                "paper_account_identity_sha256": account_identity.payload_sha256,
                "target_weights_sha256": pending.target_weights_sha256,
                "canonical_prefix_receipt": canonical_receipt,
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
    "bind_legacy_terminal_account",
    "execute_pending_for_session",
    "reconcile_indeterminate_account",
]
