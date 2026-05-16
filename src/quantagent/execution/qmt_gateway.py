"""QMT / MiniQMT gateway with dry-run safety by default."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from quantagent.execution.broker_base import (
    BrokerBase,
    Order,
    OrderState,
    OrderStatus,
    Position,
)
from quantagent.execution.audit import AuditLogger
from quantagent.config.paths import quant_paths


@dataclass
class QMTConfig:
    account_id: str = ""
    mini_qmt_path: str = ""
    session_id: int = 0
    auto_reconnect: bool = True
    dry_run: bool = True
    live_trading_enabled: bool = False
    timeout_seconds: float = 5.0
    audit_log_dir: str = field(default_factory=lambda: str(quant_paths().logs / "execution"))


@dataclass
class QMTGateway(BrokerBase):
    config: QMTConfig = field(default_factory=QMTConfig)
    _client: object | None = field(default=None, repr=False)
    _trade_handlers: list[object] = field(default_factory=list, repr=False)
    _orders: dict[str, OrderState] = field(default_factory=dict, repr=False)
    _audit: AuditLogger | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._audit = AuditLogger(self.config.audit_log_dir, "qmt_gateway.jsonl")

    def connect(self) -> None:
        if self.config.dry_run:
            self._client = "dry_run"
            self._write_audit("connect", {"mode": "dry_run"})
            return
        if not self.config.live_trading_enabled:
            raise RuntimeError("Live QMT trading is disabled; set live_trading_enabled=true and dry_run=false explicitly.")
        try:
            from xtquant import xttrader  # type: ignore
            from xtquant.xttype import StockAccount  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime-only dependency
            raise RuntimeError(
                "xtquant (QMT) not installed; this gateway runs only on the trading host."
            ) from exc
        client = xttrader.XtQuantTrader(self.config.mini_qmt_path, self.config.session_id)
        client.start()
        client.connect()
        client.subscribe(StockAccount(self.config.account_id))
        self._client = client

    def submit(self, order: Order) -> OrderState:
        if order.client_order_id in self._orders:
            return self._orders[order.client_order_id]
        if self.config.dry_run:
            state = OrderState(
                client_order_id=order.client_order_id,
                broker_order_id=f"dry-{order.client_order_id}",
                status=OrderStatus.SUBMITTED,
                filled_quantity=0,
                avg_price=0.0,
                last_message="dry_run_not_submitted_to_broker",
            )
            self._orders[order.client_order_id] = state
            self._write_audit("submit_dry_run", order.__dict__)
            return state
        if not self.config.live_trading_enabled:
            raise RuntimeError("Live QMT submit blocked by configuration.")
        raise NotImplementedError("Live xttrader.order_stock wiring must be enabled on a controlled QMT host.")

    def cancel(self, client_order_id: str) -> OrderState:
        current = self._orders.get(client_order_id)
        state = OrderState(
            client_order_id=client_order_id,
            broker_order_id=current.broker_order_id if current else None,
            status=OrderStatus.CANCELLED,
            filled_quantity=current.filled_quantity if current else 0,
            avg_price=current.avg_price if current else 0.0,
            last_message="dry_run_cancelled" if self.config.dry_run else "cancel_requested",
        )
        self._orders[client_order_id] = state
        self._write_audit("cancel", state.__dict__)
        return state

    def query_order(self, client_order_id: str) -> OrderState:
        return self._orders.get(
            client_order_id,
            OrderState(client_order_id, None, OrderStatus.PENDING, 0, 0.0, "unknown_order"),
        )

    def query_orders(self) -> list[OrderState]:
        return list(self._orders.values())

    def query_trades(self) -> list[object]:
        return []

    def query_positions(self) -> list[Position]:
        return []

    def query_account_value(self) -> float:
        return 0.0 if self.config.dry_run else float("nan")

    def reconnect(self) -> None:
        self.disconnect()
        self.connect()

    def on_trade(self, callback) -> None:
        self._trade_handlers.append(callback)

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                if hasattr(self._client, "stop"):
                    self._client.stop()  # type: ignore[attr-defined]
            finally:
                self._client = None

    @staticmethod
    def map_status(qmt_status: int) -> OrderStatus:
        """Translate xtquant order status code to OrderStatus enum."""
        return {
            48: OrderStatus.PENDING,
            49: OrderStatus.SUBMITTED,
            50: OrderStatus.PARTIAL,
            51: OrderStatus.FILLED,
            52: OrderStatus.CANCELLED,
            53: OrderStatus.REJECTED,
        }.get(qmt_status, OrderStatus.PENDING)

    def _write_audit(self, event_type: str, payload: dict) -> None:
        if self._audit is not None:
            payload = {"gateway_time": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), **payload}
            self._audit.write(event_type, payload)
