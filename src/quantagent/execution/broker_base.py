from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Order:
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: float | None = None
    note: str = ""


@dataclass(frozen=True)
class OrderState:
    client_order_id: str
    broker_order_id: str | None
    status: OrderStatus
    filled_quantity: int
    avg_price: float
    last_message: str = ""


@dataclass(frozen=True)
class Position:
    symbol: str
    available_shares: int
    frozen_shares: int
    avg_cost: float


@dataclass(frozen=True)
class TradeFill:
    client_order_id: str
    symbol: str
    side: OrderSide
    fill_quantity: int
    fill_price: float
    fill_time: str
    commission: float
    stamp_duty: float
    transfer_fee: float


class BrokerBase(ABC):
    """Minimum contract a broker adapter must satisfy."""

    @abstractmethod
    def submit(self, order: Order) -> OrderState: ...

    @abstractmethod
    def cancel(self, client_order_id: str) -> OrderState: ...

    @abstractmethod
    def query_order(self, client_order_id: str) -> OrderState: ...

    @abstractmethod
    def query_positions(self) -> list[Position]: ...

    @abstractmethod
    def query_account_value(self) -> float: ...

    @abstractmethod
    def on_trade(self, callback) -> None: ...
