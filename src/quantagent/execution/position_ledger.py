from __future__ import annotations

from dataclasses import dataclass, field

from quantagent.execution.broker_base import OrderSide, Position, TradeFill


@dataclass
class PositionLedger:
    positions: dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    fills: list[TradeFill] = field(default_factory=list)

    def apply_fill(self, fill: TradeFill) -> None:
        current = self.positions.get(fill.symbol, Position(fill.symbol, 0, 0, fill.fill_price))
        value = fill.fill_quantity * fill.fill_price
        cost = fill.commission + fill.stamp_duty + fill.transfer_fee
        if fill.side == OrderSide.BUY:
            total_shares = current.available_shares + current.frozen_shares + fill.fill_quantity
            avg_cost = ((current.available_shares + current.frozen_shares) * current.avg_cost + value) / max(total_shares, 1)
            self.positions[fill.symbol] = Position(fill.symbol, current.available_shares, current.frozen_shares + fill.fill_quantity, avg_cost)
            self.cash -= value + cost
        else:
            sell_qty = min(fill.fill_quantity, current.available_shares)
            self.positions[fill.symbol] = Position(fill.symbol, current.available_shares - sell_qty, current.frozen_shares, current.avg_cost)
            self.cash += sell_qty * fill.fill_price - cost
        self.fills.append(fill)

    def snapshot(self) -> tuple[Position, ...]:
        return tuple(self.positions.values())

    def release_frozen_shares(self) -> None:
        for symbol, position in list(self.positions.items()):
            self.positions[symbol] = Position(symbol, position.available_shares + position.frozen_shares, 0, position.avg_cost)
