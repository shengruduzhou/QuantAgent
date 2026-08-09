"""Crash-safe next-observed-session execution for pending paper targets.

A target produced from T-close information is only a signal.  This module may
turn it into paper/shadow economic evidence only when a later run observes the
first market session strictly after T.

The execution chain is deliberately the repository's normal paper chain:

    pending signal -> whole-signal idempotency -> OMS -> paper adapter
    -> paper venue -> one canonical ledger

The outer signal claim prevents an entire target from being replayed after a
crash.  The OMS adds its own durable per-order claims and canonical risk/order
lifecycle.  The paper venue owns exchange-rule validation and fills.  A receipt
is written only after the canonical chain verifies and replays successfully.

One target gets one observed-session attempt.  Any residual working quantity is
cancelled at the end of that session; carrying it forward would be a different
execution strategy.  ST/risk-warning new buys remain blocked until historical
PIT risk-warning identity and the account-level 500k-share daily buy cap are
governed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.domain.idempotency import IdempotencyStore
from quantagent.domain.ledger import CanonicalLedger, GENESIS_HASH
from quantagent.domain.lineage import Lineage
from quantagent.execution.broker_base import (
    Order as WireOrder,
    OrderSide,
    OrderStatus,
    OrderType,
)
from quantagent.execution.order_manager import OrderManager, OrderManagerConfig
from quantagent.execution.paper_adapter import PaperBrokerAdapter
from quantagent.market_rules import ashare as rules
from quantagent.paper import ledger as operational_ledger
from quantagent.paper.broker import BrokerConfig, MarketSnapshot, PaperBroker
from quantagent.paper.pending_signal import (
    PendingPaperSignal,
    PendingPaperSignalStore,
    verify_pending_signal,
)
from quantagent.paper.recovery import recover_from_canonical


EXECUTION_RECEIPT_SCHEMA_VERSION = "paper_pending_execution_receipt_v1"
PENDING_EXECUTION_STRATEGY_VERSION = "paper_pending_next_session_v1"


class PendingExecutionRefused(RuntimeError):
    """The pending signal cannot be executed with trustworthy observed evidence."""


class AmbiguousPendingExecution(PendingExecutionRefused):
    """A durable signal claim exists without a verified receipt; never resubmit."""


class ExecutionReceiptCorruption(PendingExecutionRefused):
    """A persisted receipt or canonical-ledger binding no longer verifies."""


@dataclass(frozen=True)
class PendingExecutionConfig:
    initial_cash: float = 1_000_000.0
    portfolio_id: str = "v7-paper-shadow"
    participation_cap: float = 0.10
    commission_rate: float = rules.DEFAULT_COMMISSION_RATE
    slippage_bps: float = 5.0
    impact_coefficient: float = 0.10
    max_limit_slippage_fraction: float = 0.01
    min_order_value_yuan: float = 100.0
    allow_st_buy: bool = False
    require_explicit_market_flags: bool = True


@dataclass(frozen=True)
class PendingExecutionReceipt:
    schema_version: str
    signal_date: str
    execution_date: str
    execution_timing_semantics: str
    pending_payload_sha256: str
    target_weights_sha256: str
    outcome: str
    canonical_before_records: int
    canonical_before_head: str
    canonical_after_records: int
    canonical_after_head: str
    account_content_hash: str
    cash_after: float
    positions_after: dict[str, int]
    order_results: tuple[dict[str, object], ...]
    created_at: str
    payload_sha256: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["order_results"] = [dict(item) for item in self.order_results]
        return payload


@dataclass(frozen=True)
class PendingExecutionBatchResult:
    as_of_date: str
    executed_receipts: tuple[str, ...] = ()
    still_pending: tuple[str, ...] = ()
    skipped_existing_receipts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _payload_sha(payload: Mapping[str, object]) -> str:
    material = dict(payload)
    material.pop("payload_sha256", None)
    return sha256(_canonical_json(material)).hexdigest()


def _signal_claim_key(signal: PendingPaperSignal, execution_date: str) -> str:
    body = {
        "kind": "paper_pending_signal_execution_v1",
        "pending_payload_sha256": signal.payload_sha256,
        "execution_date": execution_date,
        "timing": EXECUTION_TIMING_SEMANTICS,
    }
    return "paper-signal:" + sha256(_canonical_json(body)).hexdigest()


def _wire_order_id(
    signal: PendingPaperSignal,
    execution_date: str,
    symbol: str,
    side: OrderSide,
) -> str:
    body = {
        "pending_payload_sha256": signal.payload_sha256,
        "execution_date": execution_date,
        "symbol": symbol,
        "side": side.value,
    }
    return "pending-" + sha256(_canonical_json(body)).hexdigest()[:24]


def _ledger_prefix_head(ledger: CanonicalLedger, record_count: int) -> str:
    if record_count == 0:
        return GENESIS_HASH
    records = ledger.read()
    if record_count < 0 or record_count > len(records):
        raise ExecutionReceiptCorruption(
            f"canonical prefix length {record_count} outside current ledger length {len(records)}"
        )
    return records[record_count - 1].record_hash


def verify_execution_receipt(
    receipt: PendingExecutionReceipt,
    *,
    signal: PendingPaperSignal,
    ledger: CanonicalLedger,
) -> None:
    verify_pending_signal(signal)
    if receipt.schema_version != EXECUTION_RECEIPT_SCHEMA_VERSION:
        raise ExecutionReceiptCorruption(
            f"unsupported execution receipt schema {receipt.schema_version!r}"
        )
    if receipt.execution_timing_semantics != EXECUTION_TIMING_SEMANTICS:
        raise ExecutionReceiptCorruption("execution receipt timing semantics mismatch")
    if receipt.signal_date != signal.signal_date:
        raise ExecutionReceiptCorruption("execution receipt signal-date mismatch")
    if receipt.pending_payload_sha256 != signal.payload_sha256:
        raise ExecutionReceiptCorruption("execution receipt pending-signal digest mismatch")
    if receipt.target_weights_sha256 != signal.target_weights_sha256:
        raise ExecutionReceiptCorruption("execution receipt target-weight digest mismatch")
    if pd.Timestamp(receipt.execution_date) <= pd.Timestamp(receipt.signal_date):
        raise ExecutionReceiptCorruption("execution date is not strictly after signal date")
    if receipt.canonical_after_records < receipt.canonical_before_records:
        raise ExecutionReceiptCorruption("canonical receipt record count moved backwards")
    verification = ledger.verify()
    if not verification.get("valid"):
        raise ExecutionReceiptCorruption(
            f"canonical ledger no longer verifies: {verification}"
        )
    if len(ledger) < receipt.canonical_after_records:
        raise ExecutionReceiptCorruption(
            "canonical ledger is shorter than the receipt-bound execution prefix"
        )
    if _ledger_prefix_head(ledger, receipt.canonical_before_records) != receipt.canonical_before_head:
        raise ExecutionReceiptCorruption("canonical pre-execution prefix no longer matches receipt")
    if _ledger_prefix_head(ledger, receipt.canonical_after_records) != receipt.canonical_after_head:
        raise ExecutionReceiptCorruption("canonical post-execution prefix no longer matches receipt")
    if _payload_sha(receipt.to_dict()) != receipt.payload_sha256:
        raise ExecutionReceiptCorruption("execution receipt payload digest mismatch")


class PendingExecutionReceiptStore:
    """Immutable execution receipts keyed by original signal date."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, signal_date: str) -> Path:
        safe = pd.Timestamp(signal_date).date().isoformat()
        return self.root / f"{safe}.json"

    def read(self, signal_date: str) -> PendingExecutionReceipt | None:
        path = self.path_for(signal_date)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["order_results"] = tuple(payload.get("order_results") or ())
            return PendingExecutionReceipt(**payload)
        except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
            raise ExecutionReceiptCorruption(
                f"cannot parse execution receipt {path}: {exc}"
            ) from exc

    def write(self, receipt: PendingExecutionReceipt) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(receipt.signal_date)
        if path.exists():
            existing = self.read(receipt.signal_date)
            if existing == receipt:
                return path
            raise ExecutionReceiptCorruption(
                f"execution receipt already exists for {receipt.signal_date} with different content"
            )
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        data = json.dumps(receipt.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        with temp.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        return path


def _pending_dates(store: PendingPaperSignalStore) -> list[str]:
    if not store.root.exists():
        return []
    dates: list[str] = []
    for path in store.root.glob("*.json"):
        try:
            dates.append(pd.Timestamp(path.stem).date().isoformat())
        except (ValueError, TypeError) as exc:
            raise PendingExecutionRefused(
                f"unexpected file in pending-signal store: {path}"
            ) from exc
    return sorted(set(dates))


def _prepare_market_panel(market_panel: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    if market_panel is None or market_panel.empty:
        raise PendingExecutionRefused(
            "market panel is empty; next observed session cannot be established"
        )
    required = {"trade_date", "symbol", "close", "volume"}
    missing = required - set(market_panel.columns)
    if missing:
        raise PendingExecutionRefused(
            f"market panel missing execution columns: {sorted(missing)}"
        )
    data = market_panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["symbol"] = data["symbol"].astype(str)
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
    data = data.dropna(subset=["trade_date", "symbol"])
    data = data[data["trade_date"] <= pd.Timestamp(as_of_date)]
    duplicated = data.duplicated(["trade_date", "symbol"], keep=False)
    if bool(duplicated.any()):
        examples = data.loc[duplicated, ["trade_date", "symbol"]].head(5).to_dict("records")
        raise PendingExecutionRefused(f"duplicate execution market rows: {examples}")
    return data.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _next_observed_session(data: pd.DataFrame, signal_date: str) -> str | None:
    sessions = pd.DatetimeIndex(data["trade_date"].dropna().unique()).sort_values()
    later = sessions[sessions > pd.Timestamp(signal_date)]
    return later[0].date().isoformat() if len(later) else None


def _bool_market_flag(row: pd.Series, name: str, *, required: bool) -> bool:
    if name not in row.index or pd.isna(row[name]):
        if required:
            raise PendingExecutionRefused(f"execution market row missing explicit {name}")
        return False
    value = row[name]
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise PendingExecutionRefused(f"invalid boolean market flag {name}={value!r}")
    return bool(value)


def _snapshot_for(
    data: pd.DataFrame,
    *,
    symbol: str,
    execution_date: str,
    require_explicit_flags: bool,
) -> MarketSnapshot:
    date = pd.Timestamp(execution_date)
    day = data[(data["trade_date"] == date) & (data["symbol"] == symbol)]
    if len(day) != 1:
        raise PendingExecutionRefused(
            f"need exactly one observed market row for {symbol} on {execution_date}; got {len(day)}"
        )
    row = day.iloc[0]
    close = float(row["close"])
    volume = float(row["volume"])
    if not np.isfinite(close) or close <= 0:
        raise PendingExecutionRefused(
            f"invalid close for {symbol} on {execution_date}: {close}"
        )
    if not np.isfinite(volume) or volume < 0:
        raise PendingExecutionRefused(
            f"invalid volume for {symbol} on {execution_date}: {volume}"
        )

    previous: float | None = None
    for column in ("previous_close", "prev_close", "pre_close"):
        if column in row.index and pd.notna(row[column]):
            candidate = float(row[column])
            if np.isfinite(candidate) and candidate > 0:
                previous = candidate
                break
    if previous is None:
        history = data[(data["symbol"] == symbol) & (data["trade_date"] < date)]
        history = history[pd.to_numeric(history["close"], errors="coerce").notna()]
        if not history.empty:
            candidate = float(history.iloc[-1]["close"])
            if np.isfinite(candidate) and candidate > 0:
                previous = candidate
    if previous is None:
        raise PendingExecutionRefused(
            f"previous observed close unavailable for {symbol} on {execution_date}"
        )

    is_st = _bool_market_flag(row, "is_st", required=require_explicit_flags)
    is_suspended = _bool_market_flag(row, "is_suspended", required=require_explicit_flags)
    sessions_since_listing = None
    if "sessions_since_listing" in row.index and pd.notna(row["sessions_since_listing"]):
        sessions_since_listing = int(row["sessions_since_listing"])
    return MarketSnapshot(
        symbol=symbol,
        trade_date=execution_date,
        last_price=close,
        previous_close=previous,
        session_volume=volume,
        board=rules.exchange_board_for_symbol(symbol),
        clock="15:00:00",
        is_suspended=is_suspended,
        is_st=is_st,
        sessions_since_listing=sessions_since_listing,
        high=(
            float(row["high"])
            if "high" in row.index and pd.notna(row["high"])
            else None
        ),
        low=(
            float(row["low"])
            if "low" in row.index and pd.notna(row["low"])
            else None
        ),
    )


def _marketable_limit(snapshot: MarketSnapshot, side: OrderSide, fraction: float) -> float:
    limits = snapshot.limits()
    if side is OrderSide.BUY:
        bound = snapshot.last_price * (1.0 + max(0.0, fraction))
        if limits.limit_up is not None:
            bound = min(bound, limits.limit_up)
    else:
        bound = snapshot.last_price * (1.0 - max(0.0, fraction))
        if limits.limit_down is not None:
            bound = max(bound, limits.limit_down)
    return round(max(bound, 0.01), 2)


def _target_orders(
    *,
    signal: PendingPaperSignal,
    execution_date: str,
    recovered,
    snapshots: Mapping[str, MarketSnapshot],
    config: PendingExecutionConfig,
) -> list[WireOrder]:
    gross_target = sum(float(value) for value in signal.target_weights.values())
    if gross_target > 1.0 + 1e-8:
        raise PendingExecutionRefused(
            "cash-account target weights exceed 100% gross long exposure"
        )

    current_symbols = {
        symbol
        for symbol, position in recovered.portfolio.positions.items()
        if not position.is_flat
    }
    all_symbols = sorted(current_symbols | set(signal.target_weights))
    prices = {symbol: snapshots[symbol].last_price for symbol in all_symbols}
    try:
        equity = float(recovered.portfolio.equity(prices))
    except Exception as exc:  # noqa: BLE001
        raise PendingExecutionRefused(
            f"cannot value recovered paper account: {exc}"
        ) from exc
    if not np.isfinite(equity) or equity <= 0:
        raise PendingExecutionRefused(f"invalid recovered account equity: {equity}")

    sells: list[WireOrder] = []
    buys: list[WireOrder] = []
    signal_id = f"pending:{signal.signal_date}:{signal.payload_sha256[:16]}"
    timestamp = f"{execution_date}T15:00:00+08:00"

    for symbol in all_symbols:
        snapshot = snapshots[symbol]
        current = int(round(recovered.portfolio.position(symbol).total))
        weight = float(signal.target_weights.get(symbol, 0.0))
        if weight < -1e-12:
            raise PendingExecutionRefused(
                f"negative target weight in cash account: {symbol}={weight}"
            )
        raw_target = max(0.0, weight) * equity / snapshot.last_price
        target = rules.round_to_lot(raw_target, board=snapshot.board, side="BUY")
        delta = target - current
        if delta == 0:
            continue

        if delta < 0:
            full_liquidation = target == 0
            quantity = rules.round_to_lot(
                abs(delta),
                board=snapshot.board,
                side="SELL",
                is_full_liquidation=full_liquidation,
            )
            side = OrderSide.SELL
        else:
            quantity = rules.round_to_lot(
                delta,
                board=snapshot.board,
                side="BUY",
            )
            side = OrderSide.BUY

        if quantity <= 0:
            continue
        if quantity * snapshot.last_price < config.min_order_value_yuan:
            continue

        order = WireOrder(
            client_order_id=_wire_order_id(signal, execution_date, symbol, side),
            symbol=symbol,
            side=side,
            quantity=int(quantity),
            order_type=OrderType.LIMIT,
            price=_marketable_limit(
                snapshot,
                side,
                config.max_limit_slippage_fraction,
            ),
            note="pending_target_next_observed_session",
            signal_id=signal_id,
            model_version=str(
                signal.source_lineage.get("model_version")
                or signal.source_lineage.get("model")
                or "unknown"
            ),
            feature_version=str(
                signal.source_lineage.get("feature_version")
                or signal.source_lineage.get("dataset")
                or "unknown"
            ),
            strategy_version=PENDING_EXECUTION_STRATEGY_VERSION,
            risk_check_result="not_checked",
            timestamp=timestamp,
        )
        (sells if side is OrderSide.SELL else buys).append(order)
    return sells + buys


def _receipt_outcome(results: list[dict[str, object]]) -> str:
    if not results:
        return "no_rebalance_needed"
    filled = sum(float(item.get("filled_quantity") or 0.0) for item in results)
    rejected = sum(1 for item in results if item.get("state") == "rejected")
    if filled <= 0:
        return "execution_blocked"
    if rejected or any(item.get("state") != "filled" for item in results):
        return "execution_partial"
    return "execution_observed"


def _build_receipt(
    *,
    signal: PendingPaperSignal,
    execution_date: str,
    outcome: str,
    before_records: int,
    before_head: str,
    ledger: CanonicalLedger,
    account,
    order_results: list[dict[str, object]],
) -> PendingExecutionReceipt:
    after_records = len(ledger)
    base: dict[str, object] = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
        "signal_date": signal.signal_date,
        "execution_date": execution_date,
        "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
        "pending_payload_sha256": signal.payload_sha256,
        "target_weights_sha256": signal.target_weights_sha256,
        "outcome": outcome,
        "canonical_before_records": before_records,
        "canonical_before_head": before_head,
        "canonical_after_records": after_records,
        "canonical_after_head": ledger.head_hash,
        "account_content_hash": account.content_hash(),
        "cash_after": float(account.cash),
        "positions_after": {
            symbol: int(account.position(symbol))
            for symbol in sorted(account.lots)
            if int(account.position(symbol)) != 0
        },
        "order_results": tuple(dict(item) for item in order_results),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "payload_sha256": "",
    }
    base["payload_sha256"] = _payload_sha(base)
    return PendingExecutionReceipt(**base)


def _lineage_for(signal: PendingPaperSignal, execution_date: str) -> Lineage:
    return Lineage(
        research_id=str(
            signal.source_lineage.get("research_id")
            or f"pending_{signal.payload_sha256[:16]}"
        ),
        strategy_version_id=PENDING_EXECUTION_STRATEGY_VERSION,
        model_version_id=str(
            signal.source_lineage.get("model_version")
            or signal.source_lineage.get("model")
            or "unknown"
        ),
        run_id=f"pending-{signal.signal_date}-on-{execution_date}",
        signal_id=f"pending:{signal.signal_date}:{signal.payload_sha256[:16]}",
    )


def _final_order_results(
    manager: OrderManager,
    submitted: list[WireOrder],
    initial_states,
) -> list[dict[str, object]]:
    state_by_id = {state.client_order_id: state for state in initial_states}
    for state in manager.cancel_all_open():
        state_by_id[state.client_order_id] = state

    results: list[dict[str, object]] = []
    for order in submitted:
        state = state_by_id.get(order.client_order_id)
        if state is None:
            raise PendingExecutionRefused(
                f"OMS produced no state for submitted order {order.client_order_id}"
            )
        results.append(
            {
                "client_order_id": order.client_order_id,
                "broker_order_id": state.broker_order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "requested_quantity": int(order.quantity),
                "filled_quantity": int(state.filled_quantity),
                "average_price": float(state.avg_price),
                "state": state.status.value,
                "message": state.last_message,
            }
        )
    return results


def execute_due_pending_signals(
    *,
    as_of_date: str,
    market_panel: pd.DataFrame,
    pending_store: PendingPaperSignalStore,
    receipt_store: PendingExecutionReceiptStore,
    canonical_ledger_path: str | Path,
    operational_ledger_path: str | Path,
    idempotency_path: str | Path,
    config: PendingExecutionConfig | None = None,
) -> PendingExecutionBatchResult:
    """Consume due pending signals in chronological order.

    Signals without a later observed session remain pending.  All deterministic
    data validation and sizing occurs before the whole-signal claim.  After that
    claim, every economic order routes through ``OrderManager`` and
    ``PaperBrokerAdapter`` into the existing canonical paper venue.
    """
    cfg = config or PendingExecutionConfig()
    cutoff = pd.Timestamp(as_of_date).date().isoformat()
    data = _prepare_market_panel(market_panel, cutoff)
    whole_signal_claims = IdempotencyStore(idempotency_path)
    op_ledger = operational_ledger.EventLedger(operational_ledger_path)

    executed: list[str] = []
    pending: list[str] = []
    skipped: list[str] = []

    for signal_date in _pending_dates(pending_store):
        if pd.Timestamp(signal_date) > pd.Timestamp(cutoff):
            pending.append(signal_date)
            continue
        signal = pending_store.read(signal_date)
        if signal is None:
            continue
        verify_pending_signal(signal)

        ledger = CanonicalLedger(canonical_ledger_path)
        existing_receipt = receipt_store.read(signal_date)
        if existing_receipt is not None:
            verify_execution_receipt(existing_receipt, signal=signal, ledger=ledger)
            skipped.append(signal_date)
            continue

        execution_date = _next_observed_session(data, signal_date)
        if execution_date is None:
            pending.append(signal_date)
            continue

        recovered = recover_from_canonical(
            str(canonical_ledger_path),
            portfolio_id=cfg.portfolio_id,
            initial_cash=cfg.initial_cash,
            as_of_trade_date=execution_date,
        )
        if recovered.open_orders():
            raise PendingExecutionRefused(
                "canonical account has pre-existing open orders; resolve them before consuming another pending target"
            )

        relevant = sorted(
            set(signal.target_weights)
            | {
                symbol
                for symbol, position in recovered.portfolio.positions.items()
                if not position.is_flat
            }
        )
        snapshots = {
            symbol: _snapshot_for(
                data,
                symbol=symbol,
                execution_date=execution_date,
                require_explicit_flags=cfg.require_explicit_market_flags,
            )
            for symbol in relevant
        }
        wire_orders = _target_orders(
            signal=signal,
            execution_date=execution_date,
            recovered=recovered,
            snapshots=snapshots,
            config=cfg,
        )

        claim_key = _signal_claim_key(signal, execution_date)
        existing_claim = whole_signal_claims.get(claim_key)
        if existing_claim is not None:
            raise AmbiguousPendingExecution(
                f"pending signal {signal_date} was already claimed at "
                f"{existing_claim.claimed_at} with outcome {existing_claim.outcome!r} "
                "but has no verified receipt; reconcile the canonical ledger instead of resubmitting"
            )

        before_records = len(ledger)
        before_head = ledger.head_hash
        claim = whole_signal_claims.claim(
            claim_key,
            payload={
                "signal_date": signal_date,
                "execution_date": execution_date,
                "pending_payload_sha256": signal.payload_sha256,
                "canonical_before_records": before_records,
                "canonical_before_head": before_head,
            },
        )
        if not claim.granted:
            raise AmbiguousPendingExecution(
                f"pending signal {signal_date} lost its whole-signal idempotency claim"
            )

        lineage = _lineage_for(signal, execution_date)
        book = ledger.replay_book()
        paper_broker = PaperBroker(
            recovered.portfolio,
            op_ledger,
            run_id=str(lineage.run_id),
            config=BrokerConfig(
                participation_cap=cfg.participation_cap,
                commission_rate=cfg.commission_rate,
                slippage_bps=cfg.slippage_bps,
                impact_coefficient=cfg.impact_coefficient,
                allow_st_buy=cfg.allow_st_buy,
            ),
            canonical_ledger=ledger,
            book=book,
            lineage=lineage,
        )

        def market_source(symbol: str, session: str) -> MarketSnapshot:
            if session[:10] != execution_date:
                raise PendingExecutionRefused(
                    f"OMS requested market session {session!r}; expected {execution_date}"
                )
            try:
                return snapshots[symbol]
            except KeyError as exc:
                raise PendingExecutionRefused(
                    f"no validated execution snapshot for {symbol} on {execution_date}"
                ) from exc

        adapter = PaperBrokerAdapter(paper_broker, market_source)
        manager = OrderManager(
            broker=adapter,
            config=OrderManagerConfig(
                min_order_value_yuan=cfg.min_order_value_yuan,
                strategy_version=PENDING_EXECUTION_STRATEGY_VERSION,
            ),
            lineage=lineage,
            idempotency_path=str(idempotency_path),
            canonical_ledger=ledger,
            order_book=paper_broker.book,
        )

        initial_states = manager.submit_orders(wire_orders)
        order_results = _final_order_results(manager, wire_orders, initial_states)

        verification = ledger.verify()
        if not verification.get("valid"):
            raise PendingExecutionRefused(
                f"canonical ledger failed verification after execution: {verification}"
            )
        _, account = ledger.replay(initial_cash=cfg.initial_cash)
        receipt = _build_receipt(
            signal=signal,
            execution_date=execution_date,
            outcome=_receipt_outcome(order_results),
            before_records=before_records,
            before_head=before_head,
            ledger=ledger,
            account=account,
            order_results=order_results,
        )
        verify_execution_receipt(receipt, signal=signal, ledger=ledger)
        receipt_path = receipt_store.write(receipt)

        # Receipt durability precedes resolving the outer claim.  A crash in this
        # window is safe: restart verifies the receipt and never reaches submit.
        whole_signal_claims.resolve(
            claim_key,
            outcome="completed",
            payload={
                "receipt_path": str(receipt_path),
                "receipt_payload_sha256": receipt.payload_sha256,
                "canonical_after_records": receipt.canonical_after_records,
                "canonical_after_head": receipt.canonical_after_head,
                "outcome": receipt.outcome,
            },
        )
        executed.append(signal_date)

    return PendingExecutionBatchResult(
        as_of_date=cutoff,
        executed_receipts=tuple(executed),
        still_pending=tuple(pending),
        skipped_existing_receipts=tuple(skipped),
    )


__all__ = [
    "EXECUTION_RECEIPT_SCHEMA_VERSION",
    "PENDING_EXECUTION_STRATEGY_VERSION",
    "AmbiguousPendingExecution",
    "ExecutionReceiptCorruption",
    "PendingExecutionBatchResult",
    "PendingExecutionConfig",
    "PendingExecutionReceipt",
    "PendingExecutionReceiptStore",
    "PendingExecutionRefused",
    "execute_due_pending_signals",
    "verify_execution_receipt",
]
