from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable
from uuid import uuid4

import pandas as pd

from quantagent.execution.broker_base import (
    BrokerBase,
    Order,
    OrderSide,
    OrderState,
    OrderStatus,
    OrderType,
)
from quantagent.quant_math.constraints import weights_to_lot_shares


@dataclass(frozen=True)
class OrderManagerConfig:
    lot_size: int = 100
    max_orders_per_symbol_per_day: int = 5
    block_buy_limit_up: bool = True
    block_sell_limit_down: bool = True
    max_participation_rate: float = 0.05


@dataclass
class OrderRecord:
    order: Order
    state: OrderState
    submitted_at: str
    last_updated_at: str


@dataclass
class OrderManager:
    """Idempotent order router. Generates per-symbol delta orders from target weights."""

    broker: BrokerBase
    config: OrderManagerConfig = field(default_factory=OrderManagerConfig)
    history: dict[str, OrderRecord] = field(default_factory=dict)
    counts_today: dict[str, int] = field(default_factory=dict)

    def reset_daily_counters(self) -> None:
        self.counts_today.clear()

    def reconcile(self, target_weights: pd.Series, prices: pd.Series, nav: float) -> list[OrderState]:
        target_shares = weights_to_lot_shares(target_weights, nav, prices, self.config.lot_size)
        positions = {p.symbol: p for p in self.broker.query_positions()}
        orders: list[Order] = []
        for symbol, desired in target_shares.items():
            current = positions.get(symbol)
            current_shares = current.available_shares + current.frozen_shares if current else 0
            delta = int(desired) - current_shares
            if delta == 0:
                continue
            if self.counts_today.get(symbol, 0) >= self.config.max_orders_per_symbol_per_day:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(
                Order(
                    client_order_id=self._make_id(symbol, side),
                    symbol=symbol,
                    side=side,
                    quantity=abs(delta),
                    order_type=OrderType.LIMIT,
                    price=float(prices.loc[symbol]) if symbol in prices.index else None,
                    note=f"target_weight_reconcile_{datetime.utcnow().isoformat()}",
                )
            )
        return list(self._submit_all(orders))

    def cancel_all_open(self) -> list[OrderState]:
        results: list[OrderState] = []
        for record in self.history.values():
            if record.state.status in {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL}:
                state = self.broker.cancel(record.order.client_order_id)
                self._update(record.order, state)
                results.append(state)
        return results

    def _submit_all(self, orders: Iterable[Order]) -> Iterable[OrderState]:
        for order in orders:
            if order.client_order_id in self.history:
                continue
            state = self.broker.submit(order)
            self._update(order, state)
            self.counts_today[order.symbol] = self.counts_today.get(order.symbol, 0) + 1
            yield state

    def _update(self, order: Order, state: OrderState) -> None:
        now = datetime.utcnow().isoformat()
        record = self.history.get(order.client_order_id)
        if record is None:
            self.history[order.client_order_id] = OrderRecord(order=order, state=state, submitted_at=now, last_updated_at=now)
        else:
            self.history[order.client_order_id] = OrderRecord(order=order, state=state, submitted_at=record.submitted_at, last_updated_at=now)

    @staticmethod
    def _make_id(symbol: str, side: OrderSide) -> str:
        return f"{symbol}-{side.value}-{uuid4().hex[:10]}"
