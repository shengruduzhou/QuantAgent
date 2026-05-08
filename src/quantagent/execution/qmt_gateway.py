"""QMT / MiniQMT gateway. Connects xtdata + xttrader to BrokerBase.

This module is the seam between QuantAgent's research stack and live trading.
Concrete xtdata / xttrader imports are deferred to runtime so research-only
installs do not need them. The class is a stub: only when QMT is actually
deployed should `submit / cancel / query` reach the live trading client.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quantagent.execution.broker_base import (
    BrokerBase,
    Order,
    OrderState,
    OrderStatus,
    Position,
)


@dataclass
class QMTConfig:
    account_id: str = ""
    mini_qmt_path: str = ""
    session_id: int = 0
    auto_reconnect: bool = True


@dataclass
class QMTGateway(BrokerBase):
    config: QMTConfig
    _client: object | None = field(default=None, repr=False)
    _trade_handlers: list[object] = field(default_factory=list, repr=False)

    def connect(self) -> None:
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
        raise NotImplementedError("Wire xttrader.order_stock here once running on a QMT host.")

    def cancel(self, client_order_id: str) -> OrderState:
        raise NotImplementedError("Wire xttrader.cancel_order_stock here.")

    def query_order(self, client_order_id: str) -> OrderState:
        raise NotImplementedError("Wire xttrader.query_stock_order here.")

    def query_positions(self) -> list[Position]:
        raise NotImplementedError("Wire xttrader.query_stock_positions here.")

    def query_account_value(self) -> float:
        raise NotImplementedError("Wire xttrader.query_stock_asset here.")

    def on_trade(self, callback) -> None:
        self._trade_handlers.append(callback)

    def disconnect(self) -> None:
        if self._client is not None:
            try:
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
