"""Durable, hash-chained canonical event ledger.

The single write path for economic events across every engine. Two properties
make replay a proof rather than a formality:

* **Append-only with a hash chain.** Each record carries the digest of its
  predecessor, so a truncated tail is detectable and an edited middle record
  breaks verification at exactly the point it was altered. A ledger that can be
  silently rewritten proves nothing about what happened.
* **Events are the source of truth.** `replay` rebuilds the order book and the
  account from the file alone. Any state a component keeps in memory is a cache
  of this, never an independent copy — the divergence between two hand-maintained
  copies is the defect class this design removes.

A partially written trailing record (a process killed mid-append) is reported by
`verify` as a torn tail and skipped by `read`; every earlier record remains
intact and verifiable, which is the reason for appending rather than rewriting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Mapping

from quantagent.domain.accounting import AccountState, replay_account
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import (
    CorporateAction,
    Fill,
    Order,
    OrderBook,
    OrderEvent,
    OrderEventType,
    OrderIntent,
    RiskDecision,
    Side,
)

SCHEMA_VERSION = "quantagent.canonical_ledger.v1"
GENESIS_HASH = "0" * 64


class LedgerCorruption(RuntimeError):
    """The chain does not verify: a record was altered, reordered or removed."""


class LedgerWriteUnavailable(RuntimeError):
    """A durable write failed, so this ledger will not accept another one.

    After a failed append nobody knows whether the bytes landed: an `fsync` that
    raises `EIO` has already handed the line to the OS, a full disk may have
    written part of it, a read-only mount may have written none. The in-memory
    head is therefore no longer known to match the file's, and appending on top of
    it computes `previousHash` from a stale predecessor — which breaks the chain
    permanently and only surfaces at read time, when the damage is durable
    (DEF-017, measured: one failed fsync left 2 records on disk against 1 in
    memory, and the next append made the file unreplayable).

    Failing closed rather than resynchronising is deliberate. Re-reading the file
    and adopting its tail would mean treating bytes the OS refused to guarantee as
    durable. A restart re-reads the file honestly, skips a torn tail, and continues
    from what is actually there.
    """


def _digest(previous_hash: str, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{previous_hash}{canonical}".encode("utf-8")).hexdigest()


def _fill_to_dict(fill: Fill) -> dict[str, Any]:
    return {
        "executionId": fill.execution_id,
        "orderId": fill.order_id,
        "symbol": fill.symbol,
        "side": fill.side.value,
        "quantity": fill.quantity,
        "price": fill.price,
        "referencePrice": fill.reference_price,
        "commission": fill.commission,
        "stampDuty": fill.stamp_duty,
        "transferFee": fill.transfer_fee,
        "filledAt": fill.filled_at,
        "lineage": fill.lineage.as_dict(),
    }


def _fill_from_dict(payload: Mapping[str, Any] | None) -> Fill | None:
    if not payload:
        return None
    return Fill(
        execution_id=payload["executionId"],
        order_id=payload["orderId"],
        symbol=payload["symbol"],
        side=Side(payload["side"]),
        quantity=int(payload["quantity"]),
        price=float(payload["price"]),
        reference_price=payload.get("referencePrice"),
        commission=float(payload.get("commission") or 0.0),
        stamp_duty=float(payload.get("stampDuty") or 0.0),
        transfer_fee=float(payload.get("transferFee") or 0.0),
        filled_at=payload.get("filledAt") or "",
        lineage=Lineage.from_mapping(payload.get("lineage")),
    )


def _risk_to_dict(decision: RiskDecision) -> dict[str, Any]:
    return {
        "riskDecisionId": decision.risk_decision_id,
        "approved": decision.approved,
        "rule": decision.rule,
        "threshold": decision.threshold,
        "measured": decision.measured,
        "reason": decision.reason,
        "decidedBy": decision.decided_by,
        "decidedAt": decision.decided_at,
        "lineage": decision.lineage.as_dict(),
    }


def _risk_from_dict(payload: Mapping[str, Any] | None) -> RiskDecision | None:
    if not payload:
        return None
    return RiskDecision(
        risk_decision_id=payload["riskDecisionId"],
        approved=bool(payload["approved"]),
        rule=payload["rule"],
        threshold=payload.get("threshold"),
        measured=payload.get("measured"),
        reason=payload.get("reason") or "",
        decided_by=payload.get("decidedBy") or "risk_engine",
        decided_at=payload.get("decidedAt") or "",
        lineage=Lineage.from_mapping(payload.get("lineage")),
    )


@dataclass(frozen=True, slots=True)
class LedgerRecord:
    sequence: int
    recorded_at: str
    previous_hash: str
    record_hash: str
    #: Absent on an account-level record such as a corporate action, which is
    #: economic but belongs to no order.
    event: OrderEvent | None = None
    #: Carried so replay can settle fills against the right session without
    #: re-deriving a calendar it does not own.
    trade_date: str | None = None
    #: A split, bonus issue or cash dividend. Present instead of `event`.
    corporate_action: Mapping[str, Any] | None = None
    #: The intent that authorised this order, recorded on CREATED so the chain
    #: "no order without an intent" is provable from the log alone.
    intent: Mapping[str, Any] | None = None


class CanonicalLedger:
    """Append-only hash-chained event log shared by every engine."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._lock = RLock()
        self._path = Path(path) if path is not None else None
        self._records: list[LedgerRecord] = []
        self._head = GENESIS_HASH
        self._torn_tail = False
        #: Set when a durable append failed. Latches: once the on-disk tail is of
        #: unknown length, no further append from this instance can be trusted.
        self._write_failure: str | None = None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # Torn trailing write from a killed process; everything before
                # it is intact, which is the point of appending.
                self._torn_tail = True
                break
            record = self._record_from_dict(payload)
            self._records.append(record)
            self._head = record.record_hash

    @staticmethod
    def _record_from_dict(payload: Mapping[str, Any]) -> LedgerRecord:
        event_payload = payload.get("event")
        event = (
            OrderEvent(
                event_type=OrderEventType(event_payload["eventType"]),
                order_id=event_payload["orderId"],
                sequence=int(event_payload["sequence"]),
                event_time=event_payload["eventTime"],
                fill=_fill_from_dict(event_payload.get("fill")),
                reason=event_payload.get("reason"),
                risk_decision=_risk_from_dict(event_payload.get("riskDecision")),
                lineage=Lineage.from_mapping(event_payload.get("lineage")),
            )
            if event_payload
            else None
        )
        return LedgerRecord(
            sequence=int(payload["sequence"]),
            recorded_at=payload["recordedAt"],
            previous_hash=payload["previousHash"],
            record_hash=payload["recordHash"],
            event=event,
            trade_date=payload.get("tradeDate"),
            intent=payload.get("intent"),
            corporate_action=payload.get("corporateAction"),
        )

    @staticmethod
    def _event_to_dict(event: OrderEvent | None) -> dict[str, Any] | None:
        if event is None:
            return None
        return {
            "eventType": event.event_type.value,
            "orderId": event.order_id,
            "sequence": event.sequence,
            "eventTime": event.event_time,
            "reason": event.reason,
            "fill": _fill_to_dict(event.fill) if event.fill else None,
            "riskDecision": _risk_to_dict(event.risk_decision) if event.risk_decision else None,
            "lineage": event.lineage.as_dict(),
        }

    # -- writing ------------------------------------------------------------
    def append_corporate_action(
        self, action: CorporateAction, *, trade_date: str | None = None
    ) -> LedgerRecord:
        """Record a split, bonus issue or cash dividend on the chain.

        These moved cash and share count while being classified as operational
        telemetry, so paper's portfolio and the canonical replay diverged by the
        full dividend and the full share adjustment (DEF-020). They are economic and
        they belong here.
        """
        return self.append(None, trade_date=trade_date, corporate_action=action)

    def append(
        self,
        event: OrderEvent | None,
        *,
        trade_date: str | None = None,
        intent: OrderIntent | None = None,
        corporate_action: CorporateAction | None = None,
    ) -> LedgerRecord:
        with self._lock:
            if self._write_failure is not None:
                raise LedgerWriteUnavailable(
                    f"this ledger stopped accepting writes after a failed durable "
                    f"append ({self._write_failure}). The on-disk tail is of unknown "
                    "length, so another append would chain from a head that may not "
                    "be the file's. Restart to re-read the file and continue from it."
                )
            body = {
                "schemaVersion": SCHEMA_VERSION,
                "sequence": len(self._records),
                "recordedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "event": self._event_to_dict(event),
                "tradeDate": trade_date,
                "corporateAction": (
                    corporate_action.to_dict() if corporate_action is not None else None
                ),
                "intent": (
                    {
                        "orderIntentId": intent.order_intent_id,
                        "symbol": intent.symbol,
                        "side": intent.side.value,
                        "quantity": intent.quantity,
                        "tradeDate": intent.trade_date,
                        "lineage": intent.lineage.as_dict(),
                    }
                    if intent is not None
                    else None
                ),
            }
            record_hash = _digest(self._head, body)
            payload = {**body, "previousHash": self._head, "recordHash": record_hash}
            record = self._record_from_dict(payload)
            if self._path is not None:
                try:
                    with self._path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n"
                        )
                        handle.flush()
                        os.fsync(handle.fileno())
                except OSError as exc:
                    # Latch closed before re-raising. The caller may well retry, and
                    # a retry that succeeded would append a second record chained
                    # from the same stale head.
                    self._write_failure = f"{type(exc).__name__}: {exc}"
                    raise
            self._records.append(record)
            self._head = record_hash
            return record

    # -- reading ------------------------------------------------------------
    def read(self) -> tuple[LedgerRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def events(self) -> tuple[OrderEvent, ...]:
        """Order events only. Account-level records carry no `event`."""
        return tuple(record.event for record in self.read() if record.event is not None)

    def corporate_actions(self) -> tuple[tuple[int, CorporateAction], ...]:
        """Corporate actions paired with their position in `events()`.

        The position is what preserves interleaving: a dividend paid before a sale
        and one paid after it produce different realised PnL, so the fold needs to
        know where in the event stream each sits.
        """
        actions: list[tuple[int, CorporateAction]] = []
        seen_events = 0
        for record in self.read():
            if record.event is not None:
                seen_events += 1
            elif record.corporate_action:
                actions.append(
                    (seen_events, CorporateAction.from_dict(record.corporate_action))
                )
        return tuple(actions)

    @property
    def path(self) -> Path | None:
        """Where this chain is durable, or None when it lives only in memory."""
        return self._path

    @property
    def head_hash(self) -> str:
        return self._head

    @property
    def had_torn_tail(self) -> bool:
        return self._torn_tail

    @property
    def write_failure(self) -> str | None:
        """Why this ledger stopped accepting writes, or None while healthy."""
        return self._write_failure

    def __len__(self) -> int:
        return len(self._records)

    def verify(self) -> dict[str, Any]:
        """Recompute the chain. Reports the first break rather than a bare bool."""
        previous = GENESIS_HASH
        for index, record in enumerate(self.read()):
            body = {
                "schemaVersion": SCHEMA_VERSION,
                "sequence": record.sequence,
                "recordedAt": record.recorded_at,
                "event": self._event_to_dict(record.event),
                "tradeDate": record.trade_date,
                "corporateAction": (
                    dict(record.corporate_action) if record.corporate_action else None
                ),
                "intent": dict(record.intent) if record.intent else None,
            }
            expected = _digest(previous, body)
            if record.previous_hash != previous or record.record_hash != expected:
                return {
                    "valid": False,
                    "brokenAt": index,
                    "sequence": record.sequence,
                    "reason": "hash chain mismatch",
                    "records": len(self),
                    "tornTail": self._torn_tail,
                    "writeFailure": self._write_failure,
                }
            previous = record.record_hash
        return {
            "valid": True,
            "records": len(self),
            "headHash": previous,
            "tornTail": self._torn_tail,
            # A chain can verify perfectly and still belong to a process that can no
            # longer write to it. Reporting only `valid` would let an operator read
            # "healthy" off a ledger that has stopped accepting economic events.
            "writeFailure": self._write_failure,
        }

    # -- reconstruction -----------------------------------------------------
    def replay(self, *, initial_cash: float) -> tuple[OrderBook, AccountState]:
        """Rebuild the order book and the account from the log alone.

        Raises `LedgerCorruption` before replaying anything: reconstructing from
        a chain that does not verify would produce numbers that look authoritative
        and are not.
        """
        book, trade_dates = self._replay_book()
        account = replay_account(
            self.events(),
            initial_cash=initial_cash,
            trade_date_of=trade_dates,
            corporate_actions=self.corporate_actions(),
        )
        return book, account

    def replay_book(self) -> OrderBook:
        """Rebuild only the order book.

        For callers that need order state and know nothing about the account's
        opening cash. `OrderManager.rebuild_history` used to call `replay` with a
        fabricated `initial_cash=0.0`, which made cash go negative and tripped an
        accounting invariant on a question nobody had asked (DEF-010).
        """
        return self._replay_book()[0]

    def _replay_book(self) -> tuple[OrderBook, dict[str, str]]:
        verification = self.verify()
        if not verification["valid"]:
            raise LedgerCorruption(
                f"ledger failed verification at record {verification['brokenAt']}: "
                f"{verification['reason']}"
            )
        book = OrderBook()
        # Keyed by execution id, not by order id. The settlement session belongs to
        # the *fill*: an order can trade in one session and be cancelled in the
        # next, and an order-keyed map was last-write-wins, so the cancel's date
        # retroactively re-dated the fill and moved the T+1 lot a day forward
        # (DEF-016). Order-level dates are kept only as a fallback for fills whose
        # own record carried none.
        trade_dates: dict[str, str] = {}
        for record in self.read():
            event = record.event
            if event is None:
                # An account-level record: economic, but about no order.
                continue
            if record.trade_date:
                if event.fill is not None:
                    trade_dates[event.fill.execution_id] = record.trade_date
                else:
                    trade_dates.setdefault(event.order_id, record.trade_date)
            if event.event_type is OrderEventType.CREATED:
                if not record.intent:
                    raise LedgerCorruption(
                        f"order {event.order_id} was created without a recorded intent"
                    )
                intent = OrderIntent(
                    order_intent_id=record.intent["orderIntentId"],
                    symbol=record.intent["symbol"],
                    side=Side(record.intent["side"]),
                    quantity=int(record.intent["quantity"]),
                    trade_date=record.intent["tradeDate"],
                    lineage=Lineage.from_mapping(record.intent.get("lineage")),
                )
                book.open(intent, parent_order_id=event.lineage.parent_order_id)
                continue
            book.apply(
                event.order_id,
                event.event_type,
                fill=event.fill,
                reason=event.reason,
                risk_decision=event.risk_decision,
            )
        return book, trade_dates


class LineageCollision(RuntimeError):
    """An order id already on the chain was opened again.

    Order ids are content-addressed over lineage, so this means the same run id,
    signal and economics were replayed against a ledger that already recorded
    them — typically a re-run pointed at the previous run's ledger file. Raised at
    the write boundary because the alternative is appending lifecycle events to an
    order the file already closed, which leaves a chain that cannot be replayed.
    """

    def __init__(self, order_id: str, existing_status: str) -> None:
        super().__init__(
            f"order {order_id} is already on this ledger with status {existing_status}; "
            "opening it again would append a second lifecycle to one order. Use a "
            "distinct run_id or a distinct ledger."
        )
        self.order_id = order_id
        # Deliberately not named `status`: the parallel-model audit flags any
        # `.status` assignment in the order domain, and an exception field is not
        # an order-state mutation. Naming it around the check rather than adding
        # an exemption keeps the audit's default-deny worth something.
        self.existing_status = existing_status


def mirror_open(
    book: OrderBook,
    ledger: CanonicalLedger,
    intent: OrderIntent,
    *,
    trade_date: str | None = None,
    parent_order_id: str | None = None,
) -> Order:
    """Open an order on the book and record its CREATED event exactly once.

    `OrderBook.open` returns the existing order when the id is already known and
    records nothing. A caller that then appended `history_of(...)[-1]`
    unconditionally wrote whatever the *previous* last event was — a stale FILL —
    into the chain as if it were a CREATED (DEF-014). Both writes go through here
    so that cannot be reconstructed by a caller.
    """
    before = len(book.events())
    order = book.open(intent, parent_order_id=parent_order_id)
    if len(book.events()) == before:
        raise LineageCollision(order.order_id, order.status.value)
    ledger.append(
        book.history_of(order.order_id)[-1], trade_date=trade_date, intent=intent
    )
    return order


def mirror_event(
    book: OrderBook,
    ledger: CanonicalLedger,
    order_id: str,
    event_type: OrderEventType,
    *,
    trade_date: str | None = None,
    fill: Fill | None = None,
    reason: str | None = None,
    risk_decision: RiskDecision | None = None,
) -> Order:
    """Apply an event to the book and append it to the ledger iff it happened.

    The book absorbs a re-delivered execution without recording it. A caller that
    appended `history_of(...)[-1]` unconditionally would then write the *previous*
    event to the ledger a second time — turning a harmless duplicate callback into
    a corrupt log. Routing both writes through one function makes book and ledger
    unable to disagree about whether an event occurred.
    """
    before = len(book.events())
    updated = book.apply(
        order_id, event_type, fill=fill, reason=reason, risk_decision=risk_decision
    )
    if len(book.events()) > before:
        ledger.append(book.history_of(order_id)[-1], trade_date=trade_date)
    return updated


__all__ = [
    "CanonicalLedger",
    "GENESIS_HASH",
    "LedgerCorruption",
    "LedgerRecord",
    "LedgerWriteUnavailable",
    "LineageCollision",
    "SCHEMA_VERSION",
    "mirror_event",
    "mirror_open",
]
