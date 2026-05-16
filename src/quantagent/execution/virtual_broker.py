from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from quantagent.execution.audit import AuditLogger
from quantagent.execution.broker_base import BrokerBase, Order, OrderSide, OrderState, OrderStatus, Position, TradeFill
from quantagent.execution.cost_model import AShareCostModel
from quantagent.execution.fill_simulator import FillSimulator
from quantagent.execution.position_ledger import PositionLedger
from quantagent.config.paths import quant_paths


@dataclass
class VirtualBroker(BrokerBase):
    user_id: str = "simulated_user_001"
    initial_cash: float = 1_000_000.0
    dry_run: bool = True
    audit_log_dir: str | Path = field(default_factory=lambda: quant_paths().logs / "v6")
    fill_simulator: FillSimulator = field(default_factory=FillSimulator)
    cost_model: AShareCostModel = field(default_factory=AShareCostModel)
    market_state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ledger = PositionLedger(cash=float(self.initial_cash))
        self.orders: dict[str, OrderState] = {}
        self.order_objects: dict[str, Order] = {}
        self.callbacks: list[object] = []
        self.audit = AuditLogger(self.audit_log_dir, "virtual_broker_audit.jsonl")
        self.audit.write("virtual_broker_initialized", {"user_id": self.user_id, "dry_run": self.dry_run})

    def submit(self, order: Order) -> OrderState:
        reject = self._reject_reason(order)
        if reject:
            state = OrderState(order.client_order_id, None, OrderStatus.REJECTED, 0, 0.0, reject)
            self._record(order, state, "order_rejected")
            return state
        fill = self.fill_simulator.simulate(order, available_volume=self._available_volume(order.symbol))
        if fill.quantity <= 0:
            state = OrderState(order.client_order_id, None, OrderStatus.REJECTED, 0, fill.price, fill.message)
            self._record(order, state, "order_rejected")
            return state
        costs = self.cost_model.calculate(order.side, fill.quantity, fill.price)
        if order.side == OrderSide.BUY and self.ledger.cash < fill.quantity * fill.price + costs["total"]:
            state = OrderState(order.client_order_id, None, OrderStatus.REJECTED, 0, fill.price, "insufficient_cash")
            self._record(order, state, "order_rejected")
            return state
        trade = TradeFill(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            fill_quantity=fill.quantity,
            fill_price=fill.price,
            fill_time=_now(),
            commission=costs["commission"],
            stamp_duty=costs["stamp_duty"],
            transfer_fee=costs["transfer_fee"],
        )
        self.ledger.apply_fill(trade)
        status = OrderStatus.FILLED if fill.quantity == order.quantity else OrderStatus.PARTIAL
        state = OrderState(order.client_order_id, f"VB-{uuid4().hex[:12]}", status, fill.quantity, fill.price, fill.message)
        self._record(order, state, "order_filled", trade)
        for callback in self.callbacks:
            callback(trade)
        return state

    def cancel(self, client_order_id: str) -> OrderState:
        existing = self.orders.get(client_order_id)
        if existing is None:
            state = OrderState(client_order_id, None, OrderStatus.REJECTED, 0, 0.0, "unknown_order")
        elif existing.status in {OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED}:
            state = existing
        else:
            state = OrderState(client_order_id, existing.broker_order_id, OrderStatus.CANCELLED, existing.filled_quantity, existing.avg_price, "cancelled")
        self.orders[client_order_id] = state
        self.audit.write("order_cancel", asdict(state))
        return state

    def query_order(self, client_order_id: str) -> OrderState:
        return self.orders.get(client_order_id, OrderState(client_order_id, None, OrderStatus.REJECTED, 0, 0.0, "unknown_order"))

    def query_positions(self) -> list[Position]:
        return list(self.ledger.snapshot())

    def query_account_value(self) -> float:
        position_value = sum((p.available_shares + p.frozen_shares) * p.avg_cost for p in self.ledger.snapshot())
        return float(self.ledger.cash + position_value)

    def on_trade(self, callback) -> None:
        self.callbacks.append(callback)

    def set_market_state(self, rows: list[dict[str, object]]) -> None:
        self.market_state = {str(row["symbol"]): row for row in rows}

    def advance_trading_day(self) -> None:
        self.ledger.release_frozen_shares()
        self.audit.write("advance_trading_day", {"user_id": self.user_id})

    def _reject_reason(self, order: Order) -> str | None:
        row = self.market_state.get(order.symbol, {})
        if bool(row.get("is_suspended", row.get("suspended", False))):
            return "suspended"
        if order.side == OrderSide.BUY and bool(row.get("is_limit_up", False)):
            return "limit_up_no_buy"
        if order.side == OrderSide.SELL and bool(row.get("is_limit_down", False)):
            return "limit_down_no_sell"
        if order.side == OrderSide.SELL:
            current = self.ledger.positions.get(order.symbol, Position(order.symbol, 0, 0, 0.0))
            if current.available_shares < order.quantity:
                return "t_plus_one_or_insufficient_available_shares"
        return None

    def _available_volume(self, symbol: str) -> float | None:
        row = self.market_state.get(symbol, {})
        volume = row.get("volume")
        return float(volume) if volume is not None else None

    def _record(self, order: Order, state: OrderState, event_type: str, fill: TradeFill | None = None) -> None:
        self.order_objects[order.client_order_id] = order
        self.orders[order.client_order_id] = state
        payload = {"order": asdict(order), "state": asdict(state)}
        if fill is not None:
            payload["fill"] = asdict(fill)
        self.audit.write(event_type, payload)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
