"""QMT / MiniQMT broker gateway.

The gateway is dry-run by default.  Live mode is deliberately fail-closed and
requires all of the following before an economic submission can leave the
process:

* ``dry_run=False`` and ``live_trading_enabled=True``;
* a successful XtQuantTrader connection + account subscription;
* startup preflight/state synchronisation (unless explicitly disabled for a
  controlled test harness);
* an upstream risk decision marked approved on the order;
* a client order id which is written into ``order_remark`` so broker state can
  be reconciled after a restart.

QMT callbacks are treated as the authority for final order state.  In
particular, a successful cancel *request* is not reported as CANCELLED until a
broker order snapshot/callback confirms the terminal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from quantagent.config.paths import quant_paths
from quantagent.execution.audit import AuditLogger
from quantagent.execution.broker_base import (
    BrokerBase,
    Order,
    OrderSide,
    OrderState,
    OrderStatus,
    OrderType,
    Position,
    TradeFill,
)


_TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}
_APPROVED_RISK = {"approved", "approve", "pass", "passed", "ok"}


@dataclass
class QMTConfig:
    account_id: str = ""
    mini_qmt_path: str = ""
    session_id: int = 0
    auto_reconnect: bool = True
    dry_run: bool = True
    live_trading_enabled: bool = False
    timeout_seconds: float = 5.0
    strategy_name: str = "QuantAgent"
    order_remark_prefix: str = "qa"
    require_preflight: bool = True
    require_risk_approval: bool = True
    audit_log_dir: str = field(default_factory=lambda: str(quant_paths().logs / "execution"))


@dataclass
class QMTGateway(BrokerBase):
    config: QMTConfig = field(default_factory=QMTConfig)
    _client: object | None = field(default=None, repr=False)
    _account: object | None = field(default=None, repr=False)
    _xt_const: object | None = field(default=None, repr=False)
    _callback: object | None = field(default=None, repr=False)
    _trade_handlers: list[object] = field(default_factory=list, repr=False)
    _orders: dict[str, OrderState] = field(default_factory=dict, repr=False)
    _broker_to_client: dict[str, str] = field(default_factory=dict, repr=False)
    _audit: AuditLogger | None = field(default=None, repr=False)
    _connected: bool = field(default=False, repr=False)
    _preflight_ok: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self._audit = AuditLogger(self.config.audit_log_dir, "qmt_gateway.jsonl")

    # ------------------------------------------------------------------
    # Connection / readiness
    # ------------------------------------------------------------------
    def connect(self) -> None:
        if self.config.dry_run:
            self._client = "dry_run"
            self._connected = True
            self._preflight_ok = True
            self._write_audit("connect", {"mode": "dry_run"})
            return
        if not self.config.live_trading_enabled:
            raise RuntimeError(
                "Live QMT trading is disabled; set live_trading_enabled=true and dry_run=false explicitly."
            )
        if not self.config.account_id.strip():
            raise RuntimeError("Live QMT requires a non-empty account_id.")
        if not self.config.mini_qmt_path.strip():
            raise RuntimeError("Live QMT requires mini_qmt_path/userdata path.")
        try:
            from xtquant import xtconstant, xttrader  # type: ignore
            from xtquant.xttype import StockAccount  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime-only dependency
            raise RuntimeError(
                "xtquant (QMT) not installed; this gateway runs only on the controlled trading host."
            ) from exc

        gateway = self

        class _Callback(xttrader.XtQuantTraderCallback):  # type: ignore[misc]
            def on_connected(self) -> None:
                gateway._connected = True
                gateway._write_audit("broker_connected", {})

            def on_disconnected(self) -> None:
                gateway._connected = False
                gateway._preflight_ok = False
                gateway._write_audit("broker_disconnected", {})

            def on_stock_order(self, broker_order: object) -> None:
                gateway._ingest_order_snapshot(broker_order, source="callback")

            def on_stock_trade(self, broker_trade: object) -> None:
                gateway._ingest_trade_snapshot(broker_trade, source="callback")

            def on_stock_position(self, position: object) -> None:
                gateway._write_audit(
                    "position_callback",
                    {"symbol": str(getattr(position, "stock_code", ""))},
                )

            def on_stock_asset(self, asset: object) -> None:
                gateway._write_audit(
                    "asset_callback",
                    {"account_id": str(getattr(asset, "account_id", ""))},
                )

            def on_order_error(self, order_error: object) -> None:
                gateway._ingest_order_error(order_error)

        client = xttrader.XtQuantTrader(self.config.mini_qmt_path, int(self.config.session_id))
        callback = _Callback()
        client.register_callback(callback)
        client.start()
        connect_result = client.connect()
        if connect_result != 0:
            try:
                client.stop()
            finally:
                raise RuntimeError(f"QMT connect failed with code {connect_result}")

        account = StockAccount(self.config.account_id)
        subscribe_result = client.subscribe(account)
        if subscribe_result != 0:
            try:
                client.stop()
            finally:
                raise RuntimeError(f"QMT account subscribe failed with code {subscribe_result}")

        self._bind_live_client(client, account, xtconstant, callback)
        self._write_audit(
            "connect",
            {"mode": "live", "account_id": self.config.account_id, "session_id": self.config.session_id},
        )
        if self.config.require_preflight:
            report = self.preflight()
            if not report["ok"]:
                self.disconnect()
                raise RuntimeError(f"QMT preflight failed: {report['errors']}")

    def _bind_live_client(
        self,
        client: object,
        account: object,
        constants: object,
        callback: object | None = None,
    ) -> None:
        """Bind an already-connected live client; also used by contract tests."""
        self._client = client
        self._account = account
        self._xt_const = constants
        self._callback = callback
        self._connected = True

    def preflight(self) -> dict[str, object]:
        """Query broker state before allowing live submissions.

        Startup synchronisation is intentionally query-based even when callbacks
        are registered: callbacks only describe events observed *after* this
        process attached, whereas open orders/trades/positions may pre-date the
        current process after a restart.
        """
        if self.config.dry_run:
            self._preflight_ok = True
            return {"ok": True, "mode": "dry_run", "errors": []}
        self._ensure_connected(require_preflight=False)
        errors: list[str] = []
        try:
            asset = self._client.query_stock_asset(self._account)  # type: ignore[attr-defined]
            if asset is None:
                errors.append("asset_query_empty")
            else:
                broker_account = str(getattr(asset, "account_id", "") or "")
                if broker_account and broker_account != self.config.account_id:
                    errors.append(
                        f"account_mismatch:{broker_account}!={self.config.account_id}"
                    )
        except Exception as exc:
            errors.append(f"asset_query_failed:{type(exc).__name__}:{exc}")

        for label, method in (
            ("orders", self.query_orders),
            ("trades", self.query_trades),
            ("positions", self.query_positions),
        ):
            try:
                method()
            except Exception as exc:
                errors.append(f"{label}_sync_failed:{type(exc).__name__}:{exc}")

        self._preflight_ok = not errors
        report = {
            "ok": self._preflight_ok,
            "connected": self._connected,
            "account_id": self.config.account_id,
            "orders_synced": len(self._orders),
            "errors": errors,
            "checked_at": _utc_now(),
        }
        self._write_audit("preflight", report)
        return report

    def health(self) -> dict[str, object]:
        """Read-only liveness snapshot suitable for a trading-host heartbeat."""
        if self.config.dry_run:
            return {
                "ok": True,
                "mode": "dry_run",
                "connected": self._connected,
                "preflight_ok": self._preflight_ok,
                "checked_at": _utc_now(),
            }
        errors: list[str] = []
        asset_value: float | None = None
        try:
            asset_value = self.query_account_value()
            self.query_orders()
        except Exception as exc:
            errors.append(f"heartbeat_query_failed:{type(exc).__name__}:{exc}")
        return {
            "ok": bool(self._connected and self._preflight_ok and not errors),
            "mode": "live",
            "connected": self._connected,
            "preflight_ok": self._preflight_ok,
            "account_value": asset_value,
            "errors": errors,
            "checked_at": _utc_now(),
        }

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------
    def submit(self, order: Order) -> OrderState:
        if order.client_order_id in self._orders:
            # Idempotent replay in-process; startup preflight reconstructs the
            # same mapping from order_remark after a restart.
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

        self._ensure_connected()
        if self.config.require_risk_approval and order.risk_check_result.strip().lower() not in _APPROVED_RISK:
            raise RuntimeError(
                f"Live QMT submit blocked: order {order.client_order_id} has "
                f"risk_check_result={order.risk_check_result!r}; explicit upstream approval is required."
            )

        side = (
            getattr(self._xt_const, "STOCK_BUY")
            if order.side == OrderSide.BUY
            else getattr(self._xt_const, "STOCK_SELL")
        )
        if order.order_type == OrderType.LIMIT:
            price_type = getattr(self._xt_const, "FIX_PRICE")
            if order.price is None or float(order.price) <= 0:
                raise ValueError("QMT LIMIT order requires price > 0")
            price = float(order.price)
        else:
            price_type = getattr(self._xt_const, "MARKET_PEER_PRICE_FIRST")
            price = 0.0

        remark = self._encode_remark(order.client_order_id)
        broker_order_id = self._client.order_stock(  # type: ignore[attr-defined]
            self._account,
            order.symbol,
            side,
            int(order.quantity),
            price_type,
            price,
            self.config.strategy_name,
            remark,
        )
        broker_key = str(broker_order_id)
        if broker_order_id is None or int(broker_order_id) <= 0:
            state = OrderState(
                order.client_order_id,
                None,
                OrderStatus.REJECTED,
                0,
                0.0,
                f"broker_submit_rejected:{broker_order_id}",
            )
        else:
            self._broker_to_client[broker_key] = order.client_order_id
            state = OrderState(
                order.client_order_id,
                broker_key,
                OrderStatus.SUBMITTED,
                0,
                0.0,
                "broker_submit_accepted",
            )
        self._orders[order.client_order_id] = state
        self._write_audit(
            "submit_live",
            {
                "client_order_id": order.client_order_id,
                "broker_order_id": state.broker_order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "order_type": order.order_type.value,
                "price": order.price,
                "risk_check_result": order.risk_check_result,
                "status": state.status.value,
            },
        )
        return state

    def cancel(self, client_order_id: str) -> OrderState:
        current = self._orders.get(client_order_id)
        if self.config.dry_run:
            state = OrderState(
                client_order_id=client_order_id,
                broker_order_id=current.broker_order_id if current else None,
                status=OrderStatus.CANCELLED,
                filled_quantity=current.filled_quantity if current else 0,
                avg_price=current.avg_price if current else 0.0,
                last_message="dry_run_cancelled",
            )
            self._orders[client_order_id] = state
            self._write_audit("cancel", state.__dict__)
            return state

        self._ensure_connected()
        if current is None:
            self.query_orders()
            current = self._orders.get(client_order_id)
        if current is None:
            return OrderState(client_order_id, None, OrderStatus.PENDING, 0, 0.0, "unknown_order")
        if current.status in _TERMINAL:
            return current
        if not current.broker_order_id:
            return OrderState(
                client_order_id,
                None,
                current.status,
                current.filled_quantity,
                current.avg_price,
                "cancel_blocked_missing_broker_order_id",
            )

        broker_id: int | str = current.broker_order_id
        if str(broker_id).lstrip("-").isdigit():
            broker_id = int(str(broker_id))
        result = self._client.cancel_order_stock(self._account, broker_id)  # type: ignore[attr-defined]
        state = OrderState(
            client_order_id,
            current.broker_order_id,
            current.status,
            current.filled_quantity,
            current.avg_price,
            "cancel_requested" if result == 0 else f"cancel_request_failed:{result}",
        )
        self._orders[client_order_id] = state
        self._write_audit(
            "cancel_request",
            {"client_order_id": client_order_id, "broker_order_id": current.broker_order_id, "result": result},
        )
        return state

    def query_order(self, client_order_id: str) -> OrderState:
        if not self.config.dry_run:
            self.query_orders()
        return self._orders.get(
            client_order_id,
            OrderState(client_order_id, None, OrderStatus.PENDING, 0, 0.0, "unknown_order"),
        )

    def query_orders(self) -> list[OrderState]:
        if self.config.dry_run:
            return list(self._orders.values())
        self._ensure_connected(require_preflight=False)
        rows = self._client.query_stock_orders(self._account, False)  # type: ignore[attr-defined]
        for row in rows or []:
            self._ingest_order_snapshot(row, source="query")
        return list(self._orders.values())

    def query_trades(self) -> list[TradeFill]:
        if self.config.dry_run:
            return []
        self._ensure_connected(require_preflight=False)
        rows = self._client.query_stock_trades(self._account)  # type: ignore[attr-defined]
        fills: list[TradeFill] = []
        for row in rows or []:
            fill = self._trade_from_snapshot(row)
            if fill is not None:
                fills.append(fill)
        return fills

    def query_positions(self) -> list[Position]:
        if self.config.dry_run:
            return []
        self._ensure_connected(require_preflight=False)
        rows = self._client.query_stock_positions(self._account)  # type: ignore[attr-defined]
        positions: list[Position] = []
        for row in rows or []:
            total = _int_attr(row, "volume", "total_volume", default=0)
            available = _int_attr(row, "can_use_volume", "available_volume", default=0)
            frozen = _int_attr(row, "frozen_volume", default=max(0, total - available))
            positions.append(
                Position(
                    symbol=str(getattr(row, "stock_code", "")),
                    available_shares=max(0, available),
                    frozen_shares=max(0, frozen),
                    avg_cost=_float_attr(row, "open_price", "avg_price", "cost_price", default=0.0),
                )
            )
        return positions

    def query_account_value(self) -> float:
        if self.config.dry_run:
            return 0.0
        self._ensure_connected(require_preflight=False)
        asset = self._client.query_stock_asset(self._account)  # type: ignore[attr-defined]
        if asset is None:
            raise RuntimeError("QMT asset query returned no account snapshot")
        return _float_attr(asset, "total_asset", "asset", default=float("nan"))

    # ------------------------------------------------------------------
    # Callback/query normalisation
    # ------------------------------------------------------------------
    def _ingest_order_snapshot(self, row: object, *, source: str) -> OrderState | None:
        broker_id = str(getattr(row, "order_id", "") or "")
        client_id = self._client_id_for_snapshot(row, broker_id)
        if not client_id:
            # Do not invent a QuantAgent identity for unrelated manual/QMT
            # orders.  They remain visible to account-level reconciliation but
            # cannot be merged into this process' canonical order chain.
            self._write_audit(
                "unowned_broker_order",
                {"broker_order_id": broker_id, "source": source},
            )
            return None
        if broker_id:
            self._broker_to_client[broker_id] = client_id
        state = OrderState(
            client_order_id=client_id,
            broker_order_id=broker_id or None,
            status=self._map_broker_status(getattr(row, "order_status", None)),
            filled_quantity=_int_attr(row, "traded_volume", "filled_volume", default=0),
            avg_price=_float_attr(row, "traded_price", "avg_price", "price", default=0.0),
            last_message=f"broker_{source}",
        )
        self._orders[client_id] = state
        self._write_audit(
            "order_snapshot",
            {
                "source": source,
                "client_order_id": client_id,
                "broker_order_id": broker_id,
                "status": state.status.value,
                "filled_quantity": state.filled_quantity,
                "avg_price": state.avg_price,
            },
        )
        return state

    def _ingest_trade_snapshot(self, row: object, *, source: str) -> None:
        fill = self._trade_from_snapshot(row)
        if fill is None:
            self._write_audit(
                "unowned_broker_trade",
                {"broker_order_id": str(getattr(row, "order_id", "")), "source": source},
            )
            return
        self._write_audit(
            "trade_snapshot",
            {
                "source": source,
                "client_order_id": fill.client_order_id,
                "symbol": fill.symbol,
                "side": fill.side.value,
                "fill_quantity": fill.fill_quantity,
                "fill_price": fill.fill_price,
            },
        )
        for handler in list(self._trade_handlers):
            handler(fill)  # type: ignore[operator]

    def _trade_from_snapshot(self, row: object) -> TradeFill | None:
        broker_id = str(getattr(row, "order_id", "") or "")
        client_id = self._client_id_for_snapshot(row, broker_id)
        if not client_id:
            return None
        side_value = getattr(row, "order_type", None)
        if side_value == getattr(self._xt_const, "STOCK_SELL", object()):
            side = OrderSide.SELL
        else:
            side = OrderSide.BUY
        return TradeFill(
            client_order_id=client_id,
            symbol=str(getattr(row, "stock_code", "")),
            side=side,
            fill_quantity=_int_attr(row, "traded_volume", "fill_volume", default=0),
            fill_price=_float_attr(row, "traded_price", "price", default=0.0),
            fill_time=str(getattr(row, "traded_time", getattr(row, "trade_time", _utc_now()))),
            # XtQuant trade callbacks do not expose the final broker statement
            # fee breakdown consistently.  Keep zero here and reconcile fees
            # from the canonical execution/cash ledger instead of fabricating.
            commission=0.0,
            stamp_duty=0.0,
            transfer_fee=0.0,
        )

    def _ingest_order_error(self, error: object) -> None:
        broker_id = str(getattr(error, "order_id", "") or "")
        client_id = self._broker_to_client.get(broker_id)
        message = str(getattr(error, "error_msg", getattr(error, "error_message", "broker_order_error")))
        if client_id:
            current = self._orders.get(client_id)
            self._orders[client_id] = OrderState(
                client_order_id=client_id,
                broker_order_id=broker_id or (current.broker_order_id if current else None),
                status=OrderStatus.REJECTED,
                filled_quantity=current.filled_quantity if current else 0,
                avg_price=current.avg_price if current else 0.0,
                last_message=message,
            )
        self._write_audit(
            "order_error",
            {"broker_order_id": broker_id, "client_order_id": client_id, "message": message},
        )

    def _client_id_for_snapshot(self, row: object, broker_id: str) -> str | None:
        if broker_id and broker_id in self._broker_to_client:
            return self._broker_to_client[broker_id]
        remark = str(getattr(row, "order_remark", "") or "")
        prefix = f"{self.config.order_remark_prefix}:"
        if remark.startswith(prefix) and len(remark) > len(prefix):
            return remark[len(prefix):]
        return None

    def _encode_remark(self, client_order_id: str) -> str:
        return f"{self.config.order_remark_prefix}:{client_order_id}"

    def _map_broker_status(self, status: object) -> OrderStatus:
        c = self._xt_const
        mapping = {
            getattr(c, "ORDER_UNREPORTED", object()): OrderStatus.PENDING,
            getattr(c, "ORDER_WAIT_REPORTING", object()): OrderStatus.PENDING,
            getattr(c, "ORDER_REPORTED", object()): OrderStatus.SUBMITTED,
            getattr(c, "ORDER_PART_SUCC", object()): OrderStatus.PARTIAL,
            getattr(c, "ORDER_SUCCEEDED", object()): OrderStatus.FILLED,
            getattr(c, "ORDER_PART_CANCEL", object()): OrderStatus.CANCELLED,
            getattr(c, "ORDER_CANCELED", object()): OrderStatus.CANCELLED,
            getattr(c, "ORDER_JUNK", object()): OrderStatus.REJECTED,
        }
        if status in mapping:
            return mapping[status]
        try:
            return self.map_status(int(status))
        except (TypeError, ValueError):
            return OrderStatus.PENDING

    def _ensure_connected(self, *, require_preflight: bool = True) -> None:
        if self.config.dry_run:
            return
        if not self.config.live_trading_enabled:
            raise RuntimeError("Live QMT operation blocked by configuration.")
        if not self._connected or self._client is None or self._account is None or self._xt_const is None:
            raise RuntimeError("QMT live client/account is not connected and subscribed.")
        if require_preflight and self.config.require_preflight and not self._preflight_ok:
            raise RuntimeError("QMT live operation blocked until startup preflight/state sync succeeds.")

    def reconnect(self) -> None:
        self.disconnect()
        self.connect()

    def on_trade(self, callback) -> None:
        self._trade_handlers.append(callback)

    def disconnect(self) -> None:
        if self._client is not None and self._client != "dry_run":
            try:
                if hasattr(self._client, "stop"):
                    self._client.stop()  # type: ignore[attr-defined]
            finally:
                self._client = None
        else:
            self._client = None
        self._account = None
        self._xt_const = None
        self._callback = None
        self._connected = False
        self._preflight_ok = False

    @staticmethod
    def map_status(qmt_status: int) -> OrderStatus:
        """Legacy numeric fallback for older xtquant status encodings.

        Live mode prefers symbolic ``xtconstant.ORDER_*`` values through
        ``_map_broker_status`` so a library release can change numeric encodings
        without silently changing QuantAgent semantics.
        """
        return {
            48: OrderStatus.PENDING,
            49: OrderStatus.SUBMITTED,
            50: OrderStatus.PARTIAL,
            51: OrderStatus.FILLED,
            52: OrderStatus.CANCELLED,
            53: OrderStatus.REJECTED,
        }.get(qmt_status, OrderStatus.PENDING)

    def _write_audit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._audit is not None:
            payload = {"gateway_time": _utc_now(), **payload}
            self._audit.write(event_type, payload)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _int_attr(obj: object, *names: str, default: int = 0) -> int:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return int(default)


def _float_attr(obj: object, *names: str, default: float = 0.0) -> float:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return float(default)
