from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from typing import Iterable
from uuid import uuid4

import pandas as pd

from quantagent.execution.broker_base import (
    BrokerBase,
    Order,
    OrderIntent,
    OrderSide,
    OrderState,
    OrderStatus,
    OrderType,
)
from quantagent.quant_math.ashare import AshareRuleEngine


@dataclass(frozen=True)
class OrderManagerConfig:
    lot_size: int = 100
    min_order_value_yuan: float = 5_000.0
    allow_odd_lot_sell_only_for_full_liquidation: bool = True
    max_orders_per_symbol_per_day: int = 5
    block_buy_limit_up: bool = True
    block_sell_limit_down: bool = True
    max_participation_rate: float = 0.05
    strategy_version: str = "v4.0"


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
    rule_engine: AshareRuleEngine = field(default_factory=AshareRuleEngine)
    history: dict[str, OrderRecord] = field(default_factory=dict)
    counts_today: dict[str, int] = field(default_factory=dict)
    last_skipped_orders: list[dict[str, object]] = field(default_factory=list)
    skipped_orders: list[dict[str, object]] = field(default_factory=list)

    def reset_daily_counters(self) -> None:
        self.counts_today.clear()

    def reconcile(self, target_weights: pd.Series, prices: pd.Series, nav: float) -> list[OrderState]:
        positions = {p.symbol: p for p in self.broker.query_positions()}
        intents = self.target_weights_to_order_intents(target_weights, prices, nav, positions=positions)
        orders: list[Order] = []
        for intent in intents:
            if self.counts_today.get(intent.symbol, 0) >= self.config.max_orders_per_symbol_per_day:
                continue
            orders.append(
                Order(
                    client_order_id=intent.intent_id,
                    symbol=intent.symbol,
                    side=intent.side,
                    quantity=intent.quantity,
                    order_type=OrderType.LIMIT,
                    price=intent.reference_price,
                    note="target_weight_reconcile",
                    signal_id=intent.signal_id,
                    model_version=intent.model_version,
                    feature_version=intent.feature_version,
                    strategy_version=intent.strategy_version,
                    risk_check_result=intent.risk_check_result,
                    timestamp=intent.timestamp,
                )
            )
        return list(self._submit_all(orders))

    def target_weights_to_order_intents(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        nav: float,
        positions: dict[str, object] | None = None,
        signal_id: str = "manual",
        model_version: str = "unknown",
        feature_version: str = "unknown",
        risk_check_result: str = "not_checked",
    ) -> list[OrderIntent]:
        positions = positions or {p.symbol: p for p in self.broker.query_positions()}
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        intents: list[OrderIntent] = []
        self.last_skipped_orders = []
        for symbol, weight in target_weights.reindex(prices.index).fillna(0.0).items():
            price = float(prices.loc[symbol])
            if price <= 0 or nav <= 0:
                continue
            current = positions.get(str(symbol))
            current_shares = int(getattr(current, "available_shares", 0) + getattr(current, "frozen_shares", 0)) if current else 0
            raw_target = float(weight) * nav / price
            delta_shares = raw_target - current_shares
            delta_value = abs(delta_shares) * price
            if raw_target >= current_shares:
                side = OrderSide.BUY
                quantity = self.rule_engine.round_order_quantity(str(symbol), "buy", delta_shares)
                if quantity <= 0:
                    self._skip(str(symbol), side, 0, float(weight), price, "skipped_invalid_lot", now, delta_value)
                    continue
                if quantity * price < self.config.min_order_value_yuan:
                    self._skip(str(symbol), side, quantity, float(weight), price, "skipped_small_order", now, quantity * price)
                    continue
            else:
                side = OrderSide.SELL
                desired_sell = current_shares - raw_target
                full_liquidation = current_shares < self.config.lot_size and float(weight) <= 1e-6
                if full_liquidation and self.config.allow_odd_lot_sell_only_for_full_liquidation:
                    quantity = current_shares
                else:
                    quantity = int(desired_sell // self.config.lot_size * self.config.lot_size)
                if quantity <= 0:
                    reason = (
                        "skipped_not_full_odd_lot_liquidation"
                        if current_shares < self.config.lot_size and not full_liquidation
                        else "skipped_invalid_lot"
                    )
                    self._skip(str(symbol), side, 0, float(weight), price, reason, now, delta_value)
                    continue
                if quantity * price < self.config.min_order_value_yuan and not full_liquidation:
                    self._skip(str(symbol), side, quantity, float(weight), price, "skipped_small_order", now, quantity * price)
                    continue
            if quantity <= 0:
                continue
            intent_id = self._make_id(str(symbol), side, signal_id=signal_id, model_version=model_version)
            intents.append(
                OrderIntent(
                    intent_id=intent_id,
                    symbol=str(symbol),
                    side=side,
                    quantity=quantity,
                    target_weight=float(weight),
                    reference_price=price,
                    signal_id=signal_id,
                    model_version=model_version,
                    feature_version=feature_version,
                    strategy_version=self.config.strategy_version,
                    risk_check_result=risk_check_result,
                    timestamp=now,
                )
            )
        return intents

    def _skip(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        target_weight: float,
        reference_price: float,
        reason: str,
        timestamp: str,
        delta_value: float,
    ) -> None:
        row = {
            "symbol": symbol,
            "side": side.value,
            "quantity": int(quantity),
            "target_weight": float(target_weight),
            "reference_price": float(reference_price),
            "reason": reason,
            "delta_value": float(delta_value),
            "timestamp": timestamp,
        }
        self.last_skipped_orders.append(row)
        self.skipped_orders.append(row)

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
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        record = self.history.get(order.client_order_id)
        if record is None:
            self.history[order.client_order_id] = OrderRecord(order=order, state=state, submitted_at=now, last_updated_at=now)
        else:
            self.history[order.client_order_id] = OrderRecord(order=order, state=state, submitted_at=record.submitted_at, last_updated_at=now)

    @staticmethod
    def _make_id(symbol: str, side: OrderSide, signal_id: str = "", model_version: str = "") -> str:
        seed = f"{symbol}-{side.value}-{signal_id}-{model_version}"
        suffix = uuid4().hex[:10] if not signal_id else sha1(seed.encode("utf-8")).hexdigest()[:10]
        return f"{symbol}-{side.value}-{suffix}"
