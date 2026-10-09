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
    signal_id: str = ""
    model_version: str = ""
    feature_version: str = ""
    strategy_version: str = ""
    risk_check_result: str = "not_checked"
    timestamp: str = ""


@dataclass(frozen=True)
class OrderIntent:
    intent_id: str
    symbol: str
    side: OrderSide
    quantity: int
    target_weight: float
    reference_price: float
    signal_id: str
    model_version: str
    feature_version: str
    strategy_version: str
    risk_check_result: str
    timestamp: str


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
    #: Square-root market-impact charge (yuan).  Defaults to 0.0 so a venue
    #: that cannot observe day volume records "no impact charged" explicitly
    #: rather than by omission; ``AShareCostModel.impact_alpha_bps`` is
    #: published in the trusted-backtest certificate, so a venue that can
    #: observe participation must charge it here.
    impact_cost: float = 0.0


class VenueRefusal(RuntimeError):
    """The venue declined an order before acknowledging it.

    Raised from ``BrokerBase.submit`` when nothing economic happened at the
    venue -- no market data for the session, or market data that is not a
    measurement. It is a declared outcome, not a crash: the order manager turns
    it into a terminal canonical REJECTED event, so the record of account never
    shows a working order that no venue holds. Any other exception keeps its
    crash semantics and leaves the order for explicit recovery.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


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
