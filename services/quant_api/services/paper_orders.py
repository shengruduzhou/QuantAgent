"""The paper order submission path: HTTP -> queue -> worker -> OMS -> one ledger.

Module One proved idempotency at the `OrderManager` level. It could not prove it
at the *entry point*, because there was no entry point: the API exposed research,
backtest and job routes only, and nothing in `jobs.COMMANDS` reached an order
manager or a broker. This module is that missing path, and it is deliberately the
narrowest one that can be economically real.

Four properties carry the safety argument:

* **Live intent is refused before anything else happens.** `reject_live_intent`
  runs on the raw request, and the broker this path reaches has no connector, no
  credential and no network call. `LIVE_DISABLED` stays a terminal policy state;
  `PAPER` is the mode this path declares and asserts.
* **Two guards, two questions, one durable file.** `req_*` keys answer "has this
  HTTP delivery been processed?"; the OMS's `idem_*` keys answer "has this economic
  intent been submitted?" A client retrying with the same key is answered from the
  first. A client that lost its key and retried with a fresh one is stopped by the
  second — which is why `signalId` is required and is *not* derived from the
  idempotency key: deriving it would make the economic guard depend on delivery
  identity, and defaulting it to a constant would collapse two legitimate sleeve
  orders that happen to want the same trade (the INC-E1 defect class). Both stores
  must point at the same file; one constructed without a path is an in-memory guard
  wearing a durable one's name (DEF-015).
* **The ledger decides whether an economic action happened — not the queue.**
  Recovery reads `CanonicalLedger`, because a claim record proves only that a
  worker intended to act.
* **A crash between claiming and submitting fails closed, permanently.** Such a
  request is reported `interrupted` and is never retried automatically. Retrying
  past a claim is precisely how a duplicate order gets created; the operator must
  submit a new request with a new key, which is a visible act rather than a silent
  one.

Deliberately absent: any fabricated price. Without market data a submission is
rejected with `market_data_unavailable`. A default price would make the fill a
property of this module rather than of the market.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
from threading import RLock
from typing import Any, Callable, Iterable, Mapping

from quantagent.domain.idempotency import IdempotencyStore
from quantagent.domain.ledger import CanonicalLedger
from quantagent.domain.lineage import Lineage, content_id
from quantagent.domain.orders import OrderStatus, Signal
from quantagent.execution.broker_base import (
    Order as WireOrder,
    OrderSide,
    OrderType,
    VenueRefusal,
)
from quantagent.execution.order_manager import (
    IdempotencyConflict,
    MissingIdempotencyLineage,
    OrderManager,
    OrderManagerConfig,
)
from quantagent.execution.paper_adapter import PaperBrokerAdapter
from quantagent.paper import ledger as paper_ledger
from quantagent.paper.account_identity import (
    PaperAccountIdentityError,
    PaperAccountIdentityStore,
    account_identity_path_for_canonical,
)
from quantagent.paper.broker import (
    BrokerConfig,
    InvalidMarketSnapshot,
    MarketSnapshot,
    PaperBroker,
)
from quantagent.paper.risk import RiskEngine, RiskLimits, load_industry_map
from quantagent.paper.portfolio import Portfolio
from quantagent.safety.operating_mode import (
    OperatingModeState,
    PAPER,
    reject_live_intent,
)

SCHEMA_VERSION = "quantagent.paper_order_submission.v1"

#: Submission states. `interrupted` is terminal and deliberately not retryable.
QUEUED = "queued"
EXECUTED = "executed"
INTERRUPTED = "interrupted"

#: Reasons a submission never reaches the venue. All fail closed.
MISSING_IDEMPOTENCY_KEY = "missing_idempotency_key"
MISSING_LINEAGE = "missing_lineage"
MARKET_DATA_UNAVAILABLE = "market_data_unavailable"
MARKET_DATA_INVALID = "market_data_invalid"

MarketSource = Callable[[str, str], MarketSnapshot | None]

#: Identity of this machine, recorded in the writer lock. `flock` is per-host, so
#: the hostname is what lets a second *host* be refused.
HOST = socket.gethostname()
#: How long a writer's heartbeat stays authoritative. Long enough that a busy
#: writer is never mistaken for a dead one, short enough that a genuinely dead
#: host does not hold the account hostage.
WRITER_HEARTBEAT_STALE_SECONDS = 300.0


def _age_seconds(timestamp: str | None) -> float:
    """Seconds since `timestamp`, or infinity when it cannot be read.

    Unparseable means "no evidence anyone is alive", which must read as stale
    rather than as a live holder — the opposite would let one corrupt byte lock
    the account permanently.
    """
    if not timestamp:
        return float("inf")
    try:
        moment = datetime.fromisoformat(str(timestamp))
    except ValueError:
        return float("inf")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - moment).total_seconds()


class SubmissionRejected(ValueError):
    """The request cannot be accepted. Carries a machine-readable reason."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class MarketDataRefused(SubmissionRejected, VenueRefusal):
    """No usable market snapshot for the order's session.

    A `SubmissionRejected` for the HTTP layer and a `VenueRefusal` for the order
    manager: if the lookup fails inside the venue rather than before it, the OMS
    terminates the canonical order instead of leaving it SUBMITTED.
    """


class WriterLockUnavailable(RuntimeError):
    """Another process already holds the single-writer lock for this ledger.

    The enforced deployment contract is *single host, single writer*. Two
    processes writing one paper account would each compute available cash from
    their own in-memory portfolio and could both approve a buy the account cannot
    fund — the durable claim store prevents duplicate *orders*, not concurrent
    distinct ones over-committing the same money. Refusing to start is the honest
    response; the alternative is a second writer that looks fine until it isn't.
    """


@dataclass(frozen=True, slots=True)
class PaperOrderRequest:
    """One economic instruction, as it arrived."""

    idempotency_key: str
    run_id: str
    symbol: str
    side: str
    quantity: int
    limit_price: float
    trade_date: str
    strategy_version_id: str = ""
    signal_id: str = ""

    def validate(self) -> None:
        """Fail closed on anything that would make de-duplication impossible."""
        if not str(self.idempotency_key).strip():
            raise SubmissionRejected(
                MISSING_IDEMPOTENCY_KEY,
                "an economic submission requires an idempotency key: without one a "
                "retry cannot be told apart from a new order",
            )
        if not str(self.run_id).strip():
            raise SubmissionRejected(
                MISSING_LINEAGE,
                "an economic submission requires a run id: canonical lineage is what "
                "makes the resulting order traceable and de-duplicable",
            )
        if int(self.quantity) <= 0:
            raise SubmissionRejected("invalid_quantity", "quantity must be positive")
        if float(self.limit_price) <= 0:
            raise SubmissionRejected(
                "missing_price_bound",
                "every paper order carries a worst price; this path exposes no "
                "unbounded market order",
            )
        if str(self.side).upper() not in {"BUY", "SELL"}:
            raise SubmissionRejected("invalid_side", f"unknown side {self.side!r}")
        if not str(self.signal_id).strip():
            raise SubmissionRejected(
                MISSING_LINEAGE,
                "an economic submission requires a signal id: it is the economic "
                "identity the order-intent guard de-duplicates on, and without it a "
                "client that lost its idempotency key could trade the same intent "
                "twice",
            )

    @property
    def normalised_side(self) -> str:
        return str(self.side).upper()

    def request_key(self) -> str:
        """Identity of this HTTP *delivery*, and only that.

        Deliberately excludes the economics. If the payload were folded in, a
        client reusing one key for a different order would get a *new* key and be
        cheerfully accepted; keying on the client's key alone is what lets the
        stored `fingerprint` be compared against and the reuse reported as a
        conflict. It also lets `status` and `cancel` reconstruct the key from the
        two identifiers a caller actually still has after a retry.
        """
        return content_id(
            "req",
            idempotency_key=self.idempotency_key,
            run_id=self.run_id,
        )

    def fingerprint(self) -> str:
        """Digest of everything that makes the request economically distinct."""
        return content_id(
            "fp",
            symbol=self.symbol,
            side=self.normalised_side,
            quantity=int(self.quantity),
            limit_price=round(float(self.limit_price), 6),
            trade_date=self.trade_date,
            strategy_version_id=self.strategy_version_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "idempotencyKey": self.idempotency_key,
            "runId": self.run_id,
            "symbol": self.symbol,
            "side": self.normalised_side,
            "quantity": int(self.quantity),
            "limitPrice": float(self.limit_price),
            "tradeDate": self.trade_date,
            "strategyVersionId": self.strategy_version_id,
            "signalId": self.signal_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PaperOrderRequest":
        return cls(
            idempotency_key=str(payload.get("idempotencyKey") or ""),
            run_id=str(payload.get("runId") or ""),
            symbol=str(payload.get("symbol") or ""),
            side=str(payload.get("side") or ""),
            quantity=int(payload.get("quantity") or 0),
            limit_price=float(payload.get("limitPrice") or 0.0),
            trade_date=str(payload.get("tradeDate") or ""),
            strategy_version_id=str(payload.get("strategyVersionId") or ""),
            signal_id=str(payload.get("signalId") or ""),
        )


class PaperOrderService:
    """The one place an economic paper order can enter the system."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        market_source: MarketSource | None = None,
        initial_cash: float = 1_000_000.0,
        broker_config: BrokerConfig | None = None,
        risk_limits: RiskLimits | None = None,
        acquire_writer_lock: bool = True,
        industry_map: Mapping[str, str] | None = None,
        sector_map_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.root = Path(root)
        # Where the venue's industry map came from, or why there is none. An
        # unreadable map must not take the API down; it leaves the map absent,
        # so BUYs are refused `industry_unmeasured`, and says so.
        self.industry_map_source: str | None = "injected" if industry_map else None
        self.industry_map_error: str | None = None
        if industry_map is None and sector_map_path is not None:
            try:
                industry_map = load_industry_map(sector_map_path)
                self.industry_map_source = str(sector_map_path)
            except (OSError, ValueError, ImportError) as exc:
                self.industry_map_error = f"sector map {sector_map_path} unreadable: {exc}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.broker_config = broker_config or BrokerConfig()
        self.mode = OperatingModeState(mode=PAPER)
        self.mode.require_order_simulation()

        self._lock = RLock()
        self._writer_lock_handle = None
        self.writer_lock_error: str | None = None
        if acquire_writer_lock:
            self._acquire_writer_lock()

        self.market_source = market_source
        self.initial_cash = float(initial_cash)
        self.claims = IdempotencyStore(self.root / "claims.jsonl")
        self.arrivals = self.root / "arrivals.jsonl"
        self.ledger_path = self.root / "canonical.jsonl"

        # One chain. The venue appends its lifecycle events to the order the OMS
        # opened, rather than opening a second one — see M1-22.
        self.ledger = CanonicalLedger(self.ledger_path)
        portfolio = Portfolio(
            portfolio_id="paper_api",
            cash=self.initial_cash,
            initial_cash=self.initial_cash,
        )
        self.broker = PaperBroker(
            portfolio,
            paper_ledger.EventLedger(self.root / "operational.jsonl"),
            run_id="paper_api",
            config=self.broker_config,
            canonical_ledger=self.ledger,
            lineage=Lineage(run_id="paper_api"),
            # Portfolio-level limits. Without this the venue enforced only
            # instrument-level rules and no single-name weight, industry
            # concentration, gross exposure, daily loss or drawdown limit
            # applied to anything submitted through the HTTP path.
            #
            # The pre-trade participation limit and the venue's
            # `participation_cap` are deliberately NOT the same number. The cap
            # meters how much of a bar one order may consume per fill, which is
            # what produces a legitimate partial fill; the pre-trade limit
            # refuses an order that is grossly oversized for the book at all.
            # Setting both to 0.10 makes a partial fill unreachable -- every
            # order large enough to leave a remainder is rejected before the
            # venue can meter it -- so the gate defaults to a full bar and the
            # cap keeps doing the metering.
            risk_engine=RiskEngine(
                limits=(
                    risk_limits
                    if risk_limits is not None
                    else RiskLimits(max_participation=1.0)
                ),
                run_id="paper_api",
            ),
            # Without a map, a BUY is refused `industry_unmeasured` while the
            # industry limit is below 1.0: unmeasured concentration is not
            # "within limit".
            industry_map=industry_map,
        )
        self.adapter = PaperBrokerAdapter(self.broker, self._market_for)
        self.manager = OrderManager(
            broker=self.adapter,
            config=OrderManagerConfig(strategy_version="paper_api"),
            lineage=Lineage(run_id="paper_api"),
            canonical_ledger=self.ledger,
            order_book=self.broker.book,
            # The *same file* this service's request-level guard uses. Omitting it
            # left the OMS with `IdempotencyStore(None)` — an in-memory guard that
            # forgets everything on restart and is invisible to a second process,
            # so the economic-intent question had no durable answer and only the
            # request-level claim was actually protecting anything (DEF-015). The
            # two live in one file under distinct key prefixes (`req_` and `idem_`).
            idempotency_path=str(self.root / "claims.jsonl"),
        )
        # Rebuild economic state from the chain, so a restart continues the same
        # account rather than starting a fresh one next to it.
        if len(self.ledger):
            self._restore_portfolio()

    # -- single-writer contract ---------------------------------------------
    def _acquire_writer_lock(self) -> None:
        """Take the exclusive writer lock, or record that we could not.

        Deliberately non-fatal. This service is constructed while the API process
        starts, and killing the whole API — including every read-only research
        page — because a second instance is running would be a worse outcome than
        a process that can still answer questions but not move money. So the lock
        is *enforced on the writes*, not on the import, and the failure is
        reported by `/paper/policy` and `/paper/account` rather than hidden.
        """
        path = self.root / "writer.lock"
        handle = path.open("r+" if path.exists() else "w+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            self.writer_lock_error = (
                f"another process on this host holds the paper writer lock at {path}. "
                "The enforced contract is single host, single writer: a second writer "
                "on one paper account sizes orders from its own in-memory portfolio "
                "and can over-commit the same cash, so this instance is read-only. "
                "Stop the other process, or point this one at a different root."
            )
            return

        # `flock` is advisory and per-host: on a shared filesystem it does not
        # arbitrate between two machines, so taking it proves nothing about a
        # second host. The occupancy record below is what refuses a *distributed*
        # writer. It is best effort by construction — the honest fix for multi-host
        # is shared transactional uniqueness, which this build does not have — so
        # multi-host deployment is unsupported and refused, not solved.
        occupant = self._read_occupancy(handle)
        if (
            occupant
            and occupant.get("host") != HOST
            and _age_seconds(occupant.get("heartbeatAt")) < WRITER_HEARTBEAT_STALE_SECONDS
        ):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            self.writer_lock_error = (
                f"host {occupant.get('host')!r} holds the paper writer lock at {path} "
                f"(heartbeat {occupant.get('heartbeatAt')}). Advisory locks do not "
                "arbitrate across hosts, so this build refuses distributed writers "
                "rather than pretending to coordinate them: run exactly one writer, "
                "on one host. If that host is gone, its heartbeat goes stale in "
                f"{WRITER_HEARTBEAT_STALE_SECONDS}s and this instance may take over."
            )
            return

        self._writer_lock_handle = handle
        self._write_occupancy()

    def _read_occupancy(self, handle) -> dict[str, Any]:
        try:
            handle.seek(0)
            return json.loads(handle.read() or "{}")
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_occupancy(self) -> None:
        """Record who holds the lock, so another host can see it is taken."""
        handle = self._writer_lock_handle
        if handle is None:
            return
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = {
            "host": HOST,
            "pid": os.getpid(),
            "acquiredAt": getattr(self, "_lock_acquired_at", None) or now,
            "heartbeatAt": now,
        }
        self._lock_acquired_at = payload["acquiredAt"]
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    @property
    def writable(self) -> bool:
        """Whether this instance holds the single-writer lock."""
        return self._writer_lock_handle is not None

    def _require_writer(self) -> None:
        if not self.writable:
            raise WriterLockUnavailable(
                self.writer_lock_error or "this instance does not hold the writer lock"
            )
        # Refresh the heartbeat on every economic write. A record that only carried
        # an acquisition time would go stale under a long-running writer and invite
        # another host to take over an account that is actively trading.
        self._write_occupancy()

    def close(self) -> None:
        """Release the writer lock. Safe to call more than once."""
        handle, self._writer_lock_handle = self._writer_lock_handle, None
        if handle is not None:
            try:
                # Clear the occupancy record: a clean shutdown should hand over at
                # once rather than making the next host wait out a heartbeat window.
                handle.seek(0)
                handle.truncate()
                handle.flush()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def __enter__(self) -> "PaperOrderService":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- market data --------------------------------------------------------
    def _market_for(self, symbol: str, trade_date: str) -> MarketSnapshot:
        try:
            snapshot = self.market_source(symbol, trade_date) if self.market_source else None
        except InvalidMarketSnapshot as exc:
            # A NaN volume used to fill the whole order uncapped at zero impact.
            raise MarketDataRefused(
                MARKET_DATA_INVALID,
                f"market data for {symbol} on {trade_date} is not a measurement: {exc}",
            ) from exc
        if snapshot is None:
            raise MarketDataRefused(
                MARKET_DATA_UNAVAILABLE,
                f"no market data for {symbol} on {trade_date}. This path fills at a "
                "real observed price or not at all; inventing one would make the fill "
                "a property of the server rather than of the market.",
            )
        return snapshot

    def _roll_session(self, trade_date: str) -> None:
        """Close the venue's previous session before trading a later one.

        Closing settles T+1 purchases. A long-lived API process never closed a
        session, so shares bought through this endpoint stayed pending forever
        and became sellable only after a restart rebuilt them from the ledger.
        """
        current = self.broker.session
        if current is not None and str(trade_date)[:10] > current:
            self.broker.close_session(current)

    def _mark_held_positions(self, trade_date: str) -> None:
        """Give the venue a mark for every held name on the order's session.

        The risk engine values the whole book. A held name the venue has never
        seen a quote for (e.g. after a restart) leaves the book unvaluable, and
        a BUY is then refused `book_priceable` rather than sized against a
        partial book. A missing quote here is left missing, never invented.
        """
        if self.market_source is None:
            return
        for symbol, position in list(self.broker.portfolio.positions.items()):
            if position.is_flat:
                continue
            try:
                snapshot = self.market_source(symbol, trade_date)
            except InvalidMarketSnapshot:
                continue
            if snapshot is not None:
                self.broker.observe(snapshot)

    def _restore_portfolio(self) -> None:
        _, account = self.ledger.replay(initial_cash=self.initial_cash)
        portfolio = self.broker.portfolio
        portfolio.cash = account.cash
        portfolio.realised_pnl = account.realised_pnl
        portfolio.fees_paid = account.total_fees
        for symbol in account.lots:
            position = portfolio.position(symbol)
            position.total = float(account.position(symbol))
            # Sellability is resolved against the latest session on the chain, the
            # same rule paper/recovery.py uses.
            latest = max(
                (record.trade_date or "" for record in self.ledger.read()), default=""
            )
            position.sellable = float(account.sellable(symbol, latest))
            position.pending_settlement = position.total - position.sellable
            position.average_cost = float(account.cost_basis.get(symbol, 0.0))

    # -- arrival log --------------------------------------------------------
    def _record_arrival(self, payload: Mapping[str, Any]) -> None:
        """Append-only audit of everything the endpoint was asked to do.

        Includes refusals and conflicts. A log of only the accepted requests
        cannot answer "did the client ever ask?", which is the first question
        after a disputed order.
        """
        line = json.dumps(
            {
                "schemaVersion": SCHEMA_VERSION,
                "recordedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **dict(payload),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        with self.arrivals.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # -- submission ---------------------------------------------------------
    def submit(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Accept (or refuse) one delivery. Does not execute it.

        Executing here would make "crash after the HTTP response" and "crash
        before execution" the same window. The worker is what executes, and
        `drain` is what runs the worker.
        """
        # Before validation, before lineage, before anything is recorded.
        reject_live_intent(payload, where="paper order submission")
        self.mode.require_order_simulation()
        self._require_writer()

        request = PaperOrderRequest.from_dict(payload)
        try:
            request.validate()
        except SubmissionRejected as exc:
            self._record_arrival(
                {"outcome": "refused", "reason": exc.reason, "request": request.to_dict()}
            )
            raise

        key = request.request_key()
        fingerprint = request.fingerprint()
        with self._lock:
            claim = self.claims.claim(
                key,
                payload={
                    "fingerprint": fingerprint,
                    "state": QUEUED,
                    "request": request.to_dict(),
                },
            )
            if not claim.granted:
                stored = str(claim.record.payload.get("fingerprint") or "")
                if stored and stored != fingerprint:
                    # Same key, different economics. Returning the original would
                    # answer a question this caller did not ask.
                    self._record_arrival(
                        {
                            "outcome": "conflict", "requestKey": key,
                            "storedFingerprint": stored, "attemptedFingerprint": fingerprint,
                            "request": request.to_dict(),
                        }
                    )
                    raise IdempotencyConflict(key, stored, fingerprint)
                self._record_arrival(
                    {"outcome": "duplicate", "requestKey": key, "request": request.to_dict()}
                )
                return self._status_of(claim.record.key, claim.record.payload, duplicate=True)

            self._record_arrival(
                {"outcome": "accepted", "requestKey": key, "request": request.to_dict()}
            )
            return self._status_of(key, claim.record.payload, duplicate=False)

    # -- worker -------------------------------------------------------------
    def pending(self) -> list[str]:
        """Request keys still queued. The claim store *is* the queue."""
        with self._lock:
            self.claims._load()  # refresh: another process may have queued work
            return [
                record.key
                for record in self.claims
                if record.payload.get("state") == QUEUED
            ]

    def drain(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Execute queued submissions. Safe to run concurrently.

        Concurrency safety does not come from this method: it comes from the
        `OrderManager`'s durable claim on the *economic intent*, which lets exactly
        one caller through per intent no matter how many workers pick the same
        queue entry up. That guarantee is the one proven for threads and processes
        in M1-10; this method inherits it rather than reimplementing it.
        """
        self._require_writer()
        results: list[dict[str, Any]] = []
        for key in self.pending()[: limit if limit is not None else None]:
            results.append(self._execute(key))
        return results

    def _execute(self, key: str) -> dict[str, Any]:
        record = self.claims.get(key)
        if record is None or record.payload.get("state") != QUEUED:
            return self._status_of(key, (record.payload if record else {}), duplicate=True)
        request = PaperOrderRequest.from_dict(record.payload.get("request") or {})

        wire = WireOrder(
            client_order_id=key,
            symbol=request.symbol,
            side=OrderSide.BUY if request.normalised_side == "BUY" else OrderSide.SELL,
            quantity=int(request.quantity),
            order_type=OrderType.LIMIT,
            price=float(request.limit_price),
            note="paper_api_submission",
            signal_id=request.signal_id,
            strategy_version="paper_api",
            timestamp=f"{request.trade_date}T10:00:00+00:00",
        )

        try:
            # Resolve the session's market data before the order manager opens a
            # canonical order. Risk approval without the snapshot the venue will
            # price against approves nothing, and a refusal discovered inside the
            # venue used to leave CREATED + RISK_APPROVED + SUBMITTED behind with
            # no working order anywhere.
            self._market_for(request.symbol, request.trade_date)
            self._roll_session(request.trade_date)
            self._mark_held_positions(request.trade_date)
            states = self.manager.submit_orders([wire])
        except SubmissionRejected as exc:
            # No market data: nothing economic happened, and saying so is more
            # useful than a generic failure.
            resolved = self.claims.resolve(
                key,
                outcome=exc.reason,
                payload={"state": EXECUTED, "venueStatus": "REFUSED", "reason": exc.reason},
            )
            return self._status_of(key, resolved.payload, duplicate=False)
        except (MissingIdempotencyLineage, IdempotencyConflict):
            raise

        if not states:
            # The OMS's durable guard already handled this intent. Whether an
            # order exists is a question for the ledger, not for this worker.
            return self._resolve_from_ledger(key, request)

        state = states[0]
        resolved = self.claims.resolve(
            key,
            outcome=state.client_order_id,
            payload={
                "state": EXECUTED,
                "venueStatus": state.status.value.upper(),
                "filledQuantity": int(state.filled_quantity),
                "averagePrice": float(state.avg_price),
                "brokerOrderId": state.broker_order_id,
                "reason": state.last_message or None,
            },
        )
        return self._status_of(key, resolved.payload, duplicate=False)

    def _resolve_from_ledger(self, key: str, request: PaperOrderRequest) -> dict[str, Any]:
        """Decide a queued request's fate from the record of account.

        A claim proves a worker *intended* to act; only the ledger says whether it
        did. When the ledger has no matching order the execution was interrupted
        between the claim and the submission — so nothing economic happened, and
        the request is marked terminally `interrupted` rather than retried.
        Retrying would mean submitting past a claim, which is exactly how a
        duplicate economic order gets created.
        """
        book = CanonicalLedger(self.ledger_path).replay_book()
        expected = self._expected_signal_id(request)
        match = next(
            (
                order
                for order in book.orders()
                if order.lineage.signal_id == expected
            ),
            None,
        )
        if match is None:
            resolved = self.claims.resolve(
                key,
                outcome=INTERRUPTED,
                payload={
                    "state": INTERRUPTED,
                    "venueStatus": "NONE",
                    "reason": (
                        "the economic intent was claimed but no order reached the "
                        "ledger; nothing was executed. Resubmit under a new "
                        "idempotency key — retrying past a claim is how duplicates "
                        "are created."
                    ),
                },
            )
            return self._status_of(key, resolved.payload, duplicate=False)
        resolved = self.claims.resolve(
            key,
            outcome=match.order_id,
            payload={
                "state": EXECUTED,
                "venueStatus": match.status.value,
                "filledQuantity": int(match.cumulative_quantity),
                "canonicalOrderId": match.order_id,
                "reason": match.reason,
            },
        )
        return self._status_of(key, resolved.payload, duplicate=False)

    def _expected_signal_id(self, request: PaperOrderRequest) -> str:
        """The signal id the OMS would derive for this request.

        Recovery has to find *this* request's order, and matching on economics is
        not good enough: two requests with different idempotency keys and identical
        economics would each match the other's order, so an interrupted submission
        could be reported as executed on the strength of a different request's fill.
        Built by calling `Signal.create` with the same arguments the OMS uses rather
        than re-deriving the digest, so the two cannot drift apart.
        """
        signal = Signal.create(
            symbol=request.symbol,
            trade_date=f"{request.trade_date}-{request.signal_id}",
            score=0.0,
            lineage=self.manager.lineage,
        )
        return signal.lineage.signal_id or ""

    def recover(self) -> list[dict[str, Any]]:
        """Settle every queued request against the ledger after a restart."""
        self._require_writer()
        settled: list[dict[str, Any]] = []
        for key in self.pending():
            record = self.claims.get(key)
            request = PaperOrderRequest.from_dict((record.payload if record else {}).get("request") or {})
            settled.append(self._resolve_from_ledger(key, request))
        return settled

    # -- cancellation -------------------------------------------------------
    def cancel(self, idempotency_key: str, run_id: str) -> dict[str, Any]:
        """Cancel a working order the API submitted."""
        self.mode.require_order_simulation()
        self._require_writer()
        key = PaperOrderRequest(
            idempotency_key=idempotency_key, run_id=run_id, symbol="", side="BUY",
            quantity=1, limit_price=1.0, trade_date="",
        ).request_key()
        record = self.claims.get(key)
        if record is None:
            raise SubmissionRejected(
                "unknown_submission", f"no submission is recorded for key {idempotency_key!r}"
            )
        paper_id = self.adapter._paper_ids.get(key)
        if paper_id is None:
            raise SubmissionRejected(
                "not_cancellable",
                "this submission never reached the venue in this process, so there is "
                "no working order to cancel",
            )
        self.broker.cancel(paper_id)
        return self.status(idempotency_key, run_id)

    # -- reading ------------------------------------------------------------
    def status(self, idempotency_key: str, run_id: str) -> dict[str, Any] | None:
        key = PaperOrderRequest(
            idempotency_key=idempotency_key, run_id=run_id, symbol="", side="BUY",
            quantity=1, limit_price=1.0, trade_date="",
        ).request_key()
        record = self.claims.get(key)
        return self._status_of(key, record.payload, duplicate=False) if record else None

    def _status_of(
        self, key: str, payload: Mapping[str, Any], *, duplicate: bool
    ) -> dict[str, Any]:
        return {
            "requestKey": key,
            "state": payload.get("state") or QUEUED,
            "venueStatus": payload.get("venueStatus"),
            "filledQuantity": payload.get("filledQuantity"),
            "averagePrice": payload.get("averagePrice"),
            "canonicalOrderId": payload.get("canonicalOrderId"),
            "reason": payload.get("reason"),
            "duplicate": duplicate,
            "request": payload.get("request"),
        }

    def orders(self) -> list[dict[str, Any]]:
        """Every order, projected from the ledger file — never from memory."""
        book = CanonicalLedger(self.ledger_path).replay_book()
        return [
            {
                "canonicalOrderId": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "status": order.status.value,
                "cumulativeQuantity": order.cumulative_quantity,
                "leavesQuantity": order.leaves_quantity,
                "lastQuantity": order.last_quantity,
                "averageFillPrice": order.average_fill_price,
                "fees": order.total_fees,
                "reason": order.reason,
                "tradeDate": order.trade_date,
                "lineage": order.lineage.as_dict(),
            }
            for order in book.orders()
        ]

    def account(self, prices: Mapping[str, float] | None = None) -> dict[str, Any]:
        """Cash, positions, PnL and NAV, all replayed from the ledger.

        NAV comes back as `None` with `unpriceableSymbols` populated when a held
        position has no mark. This route used to pass `{}` and report a NAV that
        silently valued every holding at zero (DEF-021) — for an account holding
        1,000 shares at 10.0051 that understated NAV by 10,000.00 and reported a
        10,005.10 loss that had not happened. No market source is wired in
        production, so that was the *normal* answer, not an edge case.
        """
        _, state = CanonicalLedger(self.ledger_path).replay(
            initial_cash=self.initial_cash
        )
        return state.to_dict() | state.valuation(dict(prices or {})) | {
            "initialCash": self.initial_cash,
            "mode": self.mode.to_dict(),
            "writable": self.writable,
            "writerLockError": self.writer_lock_error,
            "accountIdentity": self.account_identity(),
            "riskState": self.risk_state(),
        }

    def account_identity(self) -> dict[str, Any]:
        """Which paper account this is. Absent identity fields are null with a reason."""
        path = account_identity_path_for_canonical(self.ledger_path)
        reasons: dict[str, str] = {}
        identity = None
        try:
            identity = PaperAccountIdentityStore(path).read()
        except PaperAccountIdentityError as exc:
            reasons["accountInstanceId"] = reasons["identitySha256"] = (
                f"identity_record_unreadable: {exc}"
            )
        if identity is None and not reasons:
            reasons["accountInstanceId"] = reasons["identitySha256"] = (
                "no_identity_record: this HTTP paper account has no immutable identity "
                "file; portfolioId and initialCash are process configuration"
            )
        return {
            "portfolioId": self.broker.portfolio.portfolio_id,
            "initialCash": self.initial_cash,
            "accountInstanceId": identity.account_instance_id if identity else None,
            "identitySha256": identity.payload_sha256 if identity else None,
            "identityPath": str(path),
            "reasons": reasons,
        }

    def risk_state(self) -> dict[str, Any]:
        """What the venue's risk engine sees, measured from its own last marks.

        Every value the engine has not measured is null with an entry in
        `reasons` — never 0, which would read as "no drawdown" or "no turnover".
        """
        broker = self.broker
        engine = broker.risk_engine
        reasons: dict[str, str] = {}
        portfolio = broker.portfolio
        marks = dict(broker.marks)
        unpriceable = list(broker.unpriceable_symbols())
        nav = None if unpriceable else float(portfolio.equity(marks))
        if nav is None:
            reasons["nav"] = "unpriceable_positions: " + ",".join(unpriceable)

        switches = [
            {"scope": r["scope"], "key": r.get("key"), "reason": r["reason"],
             "triggeredAt": r.get("triggered_at")}
            for r in broker.kill_switch.active()
        ]
        first = switches[0] if switches else None
        kill_switch = {
            "active": bool(switches),
            "scope": first["scope"] if first else None,
            "reason": first["reason"] if first else None,
            "triggeredAt": first["triggeredAt"] if first else None,
            "reduceOnly": bool(switches) and all(s["scope"] == "PORTFOLIO" for s in switches),
            "switches": switches,
        }
        if not switches:
            reasons["killSwitch"] = "no_active_kill_switch"

        if engine is None:
            reasons["riskEngine"] = "no_risk_engine_attached: no portfolio limit is applied"
            return {
                "riskEngineAttached": False, "limits": None, "killSwitch": kill_switch,
                "peakEquity": None, "drawdownFromPeak": None, "nav": nav,
                "sessionTurnover": None, "sessionStartEquity": None,
                "unpriceableSymbols": unpriceable, "industryLimitEnforced": False,
                "industryLimit": None, "lastPortfolioCheck": None,
                "valuationMarks": {"source": "venue_last_observed", "count": len(marks)},
                "reasons": reasons,
            }

        peak = engine.peak_equity
        if peak is None:
            reasons["peakEquity"] = "no_equity_observation"
        drawdown = None
        if nav is not None and peak:
            drawdown = max(0.0, 1.0 - nav / peak)
        else:
            reasons["drawdownFromPeak"] = (
                "nav_unavailable" if nav is None else "peak_equity_unavailable"
            )

        session_turnover = None
        if engine.session_date is None:
            reasons["sessionTurnover"] = "no_session_observed_by_this_process"
        else:
            session_turnover = {
                "session": engine.session_date,
                "notional": float(engine.session_turnover),
                "fractionOfNav": (float(engine.session_turnover) / nav) if nav else None,
                "limit": engine.limits.max_daily_turnover,
            }
        if engine.session_start_equity is None:
            reasons["sessionStartEquity"] = "no_session_start_valuation"

        limit = float(engine.limits.max_industry_weight)
        map_loaded = bool(broker.industry_map)
        if limit >= 1.0:
            mode = "opt_out"
        elif map_loaded:
            mode = "measured_with_map"
        else:
            mode = "refuse_unmeasured"
            reasons["industryLimit"] = (
                "no industry map: every BUY is refused industry_unmeasured"
                + (f" ({self.industry_map_error})" if self.industry_map_error else "")
            )
        decision = engine.last_portfolio_decision
        if decision is None:
            reasons["lastPortfolioCheck"] = "no_portfolio_check_run_in_this_process"
        return {
            "riskEngineAttached": True,
            "limits": engine.limits.to_dict(),
            "killSwitch": kill_switch,
            "peakEquity": peak,
            "drawdownFromPeak": drawdown,
            "nav": nav,
            "sessionTurnover": session_turnover,
            "sessionStartEquity": engine.session_start_equity,
            "unpriceableSymbols": unpriceable,
            "industryLimitEnforced": limit < 1.0,
            "industryLimit": {
                "limit": limit,
                "mode": mode,
                "mapLoaded": map_loaded,
                "mapSymbolCount": len(broker.industry_map),
                "source": self.industry_map_source,
                "error": self.industry_map_error,
            },
            "lastPortfolioCheck": decision.to_dict() if decision is not None else None,
            "valuationMarks": {"source": "venue_last_observed", "count": len(marks)},
            "reasons": reasons,
        }


__all__ = [
    "EXECUTED",
    "INTERRUPTED",
    "MARKET_DATA_INVALID",
    "MARKET_DATA_UNAVAILABLE",
    "MarketDataRefused",
    "MISSING_IDEMPOTENCY_KEY",
    "MISSING_LINEAGE",
    "PaperOrderRequest",
    "PaperOrderService",
    "QUEUED",
    "SCHEMA_VERSION",
    "SubmissionRejected",
    "WRITER_HEARTBEAT_STALE_SECONDS",
    "WriterLockUnavailable",
]
