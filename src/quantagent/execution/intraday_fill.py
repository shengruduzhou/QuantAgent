"""Conservative intraday fill simulation for A-share Do-T research."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import floor
from typing import Mapping

import pandas as pd

from quantagent.execution.broker_base import OrderSide


class FillMode(str, Enum):
    CONSERVATIVE = "conservative"
    NORMAL = "normal"
    OPTIMISTIC = "optimistic"


@dataclass(frozen=True)
class CostConfig:
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_tax_sell: float = 0.0005
    transfer_fee: float = 0.00001
    slippage_bps: float = 8.0
    spread_bps: float = 6.0


@dataclass(frozen=True)
class IntradayFill:
    side: str
    requested_qty: int
    filled_qty: int
    fill_price: float
    fill_time: str
    mode: str
    status: str
    reason: str
    participation_rate: float
    volume_capacity_ratio: float
    costs: dict[str, float] = field(default_factory=dict)

    @property
    def filled(self) -> bool:
        return self.filled_qty > 0 and self.status in {"filled", "partial"}


def trade_cost_breakdown(
    side: OrderSide | str,
    quantity: int,
    price: float,
    config: CostConfig | None = None,
    *,
    slippage_bps: float | None = None,
    spread_bps: float | None = None,
) -> dict[str, float]:
    cfg = config or CostConfig()
    side_value = side.value if isinstance(side, OrderSide) else str(side).lower()
    quantity = max(0, int(quantity))
    price = max(0.0, float(price))
    notional = quantity * price
    if notional <= 0:
        return {
            "gross_pnl_bps": 0.0,
            "commission_bps": 0.0,
            "stamp_tax_bps": 0.0,
            "transfer_fee_bps": 0.0,
            "slippage_bps": 0.0,
            "spread_cost_bps": 0.0,
            "net_pnl_bps": 0.0,
            "commission": 0.0,
            "stamp_tax": 0.0,
            "transfer_fee": 0.0,
            "slippage_cost": 0.0,
            "spread_cost": 0.0,
            "total": 0.0,
        }
    slip_bps = cfg.slippage_bps if slippage_bps is None else float(slippage_bps)
    spr_bps = cfg.spread_bps if spread_bps is None else float(spread_bps)
    commission = max(float(cfg.min_commission), notional * float(cfg.commission_rate))
    stamp = notional * float(cfg.stamp_tax_sell) if side_value == "sell" else 0.0
    transfer = notional * float(cfg.transfer_fee)
    slippage = notional * slip_bps / 10_000.0
    spread = notional * spr_bps / 10_000.0
    total = commission + stamp + transfer + slippage + spread
    return {
        "gross_pnl_bps": 0.0,
        "commission_bps": commission / notional * 10_000.0,
        "stamp_tax_bps": stamp / notional * 10_000.0,
        "transfer_fee_bps": transfer / notional * 10_000.0,
        "slippage_bps": slip_bps,
        "spread_cost_bps": spr_bps,
        "net_pnl_bps": -total / notional * 10_000.0,
        "commission": commission,
        "stamp_tax": stamp,
        "transfer_fee": transfer,
        "slippage_cost": slippage,
        "spread_cost": spread,
        "total": total,
    }


@dataclass(frozen=True)
class IntradayFillSimulator:
    cost_config: CostConfig = field(default_factory=CostConfig)
    round_lot: int = 100
    conservative_participation: float = 0.05
    normal_participation: float = 0.10
    optimistic_participation: float = 0.20
    near_limit_bps: float = 20.0

    def simulate(
        self,
        bars: pd.DataFrame,
        *,
        signal_index: int,
        side: OrderSide | str,
        quantity: int,
        limit_price: float | None = None,
        mode: FillMode | str = FillMode.CONSERVATIVE,
    ) -> IntradayFill:
        mode_value = mode.value if isinstance(mode, FillMode) else str(mode)
        side_enum = side if isinstance(side, OrderSide) else OrderSide(str(side).lower())
        if bars is None or bars.empty:
            return self._empty(side_enum, quantity, mode_value, "no_bars")
        b = bars.reset_index(drop=True)
        idx = self._execution_index(signal_index, mode_value, len(b))
        if idx is None:
            return self._empty(side_enum, quantity, mode_value, "no_next_bar")
        row = b.iloc[idx]
        if self._blocked_by_limit(row, side_enum):
            return self._empty(side_enum, quantity, mode_value, "near_price_limit", row=row)
        price = self._fill_price(row, side_enum, limit_price, mode_value)
        if price is None or price <= 0:
            return self._empty(side_enum, quantity, mode_value, "limit_not_reached", row=row)
        participation = self._participation(mode_value)
        available_qty = self._capacity(row, participation)
        filled_qty = min(int(quantity), available_qty)
        filled_qty = self._round_lot(filled_qty)
        if filled_qty <= 0:
            return self._empty(side_enum, quantity, mode_value, "no_liquidity", row=row)
        status = "filled" if filled_qty == int(quantity) else "partial"
        cost = trade_cost_breakdown(side_enum, filled_qty, price, self.cost_config)
        volume = max(0.0, float(row.get("volume", 0.0) or 0.0))
        return IntradayFill(
            side=side_enum.value,
            requested_qty=int(quantity),
            filled_qty=filled_qty,
            fill_price=round(float(price), 6),
            fill_time=str(row.get("trade_time", row.get("datetime", ""))),
            mode=mode_value,
            status=status,
            reason="filled" if status == "filled" else "capacity_partial",
            participation_rate=(filled_qty / volume) if volume > 0 else 0.0,
            volume_capacity_ratio=(int(quantity) / max(volume * participation, 1.0)) if volume > 0 else float("inf"),
            costs=cost,
        )

    def _execution_index(self, signal_index: int, mode: str, n: int) -> int | None:
        idx = int(signal_index)
        if mode == FillMode.OPTIMISTIC.value:
            return idx if 0 <= idx < n else None
        idx += 1
        return idx if 0 <= idx < n else None

    def _participation(self, mode: str) -> float:
        if mode == FillMode.NORMAL.value:
            return self.normal_participation
        if mode == FillMode.OPTIMISTIC.value:
            return self.optimistic_participation
        return self.conservative_participation

    def _capacity(self, row: Mapping[str, object], participation: float) -> int:
        volume = max(0.0, float(row.get("volume", 0.0) or 0.0))
        return self._round_lot(floor(volume * participation))

    def _round_lot(self, quantity: int) -> int:
        lot = max(1, int(self.round_lot))
        return int(max(0, int(quantity)) // lot * lot)

    def _blocked_by_limit(self, row: Mapping[str, object], side: OrderSide) -> bool:
        open_px = float(row.get("open", row.get("close", 0.0)) or 0.0)
        close_px = float(row.get("close", open_px) or open_px)
        limit_up = row.get("limit_up")
        limit_down = row.get("limit_down")
        guard = self.near_limit_bps / 10_000.0
        if side == OrderSide.BUY and limit_up is not None and pd.notna(limit_up):
            lu = float(limit_up)
            return lu > 0 and max(open_px, close_px) >= lu * (1.0 - guard)
        if side == OrderSide.SELL and limit_down is not None and pd.notna(limit_down):
            ld = float(limit_down)
            return ld > 0 and min(open_px, close_px) <= ld * (1.0 + guard)
        return False

    def _fill_price(
        self,
        row: Mapping[str, object],
        side: OrderSide,
        limit_price: float | None,
        mode: str,
    ) -> float | None:
        open_px = float(row.get("open", row.get("close", 0.0)) or 0.0)
        high = float(row.get("high", open_px) or open_px)
        low = float(row.get("low", open_px) or open_px)
        slip = self.cost_config.slippage_bps / 10_000.0
        spread = self.cost_config.spread_bps / 10_000.0
        limit = None if limit_price is None else float(limit_price)

        if mode == FillMode.CONSERVATIVE.value:
            penalty = slip + spread
            price = open_px * (1.0 + penalty) if side == OrderSide.BUY else open_px * (1.0 - penalty)
            if side == OrderSide.BUY and limit is not None and price > limit:
                return None
            if side == OrderSide.SELL and limit is not None and price < limit:
                return None
            return price

        if mode == FillMode.NORMAL.value:
            half_spread = spread * 0.5
            if side == OrderSide.BUY:
                target = open_px if limit is None else limit
                if low > target:
                    return None
                return min(target, open_px) * (1.0 + slip + half_spread)
            target = open_px if limit is None else limit
            if high < target:
                return None
            return max(target, open_px) * (1.0 - slip - half_spread)

        if side == OrderSide.BUY:
            if limit is not None and low > limit:
                return None
            target = low if limit is None else min(limit, open_px)
            return target * (1.0 + spread * 0.25)
        if limit is not None and high < limit:
            return None
        target = high if limit is None else max(limit, open_px)
        return target * (1.0 - spread * 0.25)

    def _empty(
        self,
        side: OrderSide,
        quantity: int,
        mode: str,
        reason: str,
        *,
        row: Mapping[str, object] | None = None,
    ) -> IntradayFill:
        return IntradayFill(
            side=side.value,
            requested_qty=int(quantity),
            filled_qty=0,
            fill_price=0.0,
            fill_time=str(row.get("trade_time", row.get("datetime", ""))) if row is not None else "",
            mode=mode,
            status="rejected",
            reason=reason,
            participation_rate=0.0,
            volume_capacity_ratio=float("inf"),
            costs=trade_cost_breakdown(side, 0, 0.0, self.cost_config),
        )


__all__ = [
    "CostConfig",
    "FillMode",
    "IntradayFill",
    "IntradayFillSimulator",
    "trade_cost_breakdown",
]
