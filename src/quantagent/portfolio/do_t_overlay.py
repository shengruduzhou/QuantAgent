"""A-share intraday Do-T execution overlay.

The overlay is intentionally separate from the daily alpha model. It consumes
minute bars plus yesterday-available inventory and emits *simulated* intraday
round trips. It never creates live orders.

A-share T+1 legality:

* A sell leg may only use shares that were already available before today's
  session.
* A buy leg executed today cannot be sold as today's sell source.
* Therefore both sequences are legal when a base position exists:
  ``sell high -> buy back low`` and ``buy low -> sell old shares high``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.execution.broker_base import OrderSide
from quantagent.execution.cost_model import AShareCostModel


@dataclass(frozen=True)
class DoTOverlayConfig:
    """Configuration for minute-level Do-T opportunity detection."""

    trade_fraction: float = 0.30
    min_edge_pct: float = 0.025
    min_minutes_between_legs: int = 5
    round_lot: int = 100
    max_trades_per_day: int = 50


def simulate_do_t_overlay(
    minute_panel: pd.DataFrame,
    available_inventory: pd.DataFrame,
    *,
    config: DoTOverlayConfig | None = None,
    cost_model: AShareCostModel | None = None,
) -> pd.DataFrame:
    """Find legal Do-T round trips from minute bars and available shares.

    Parameters
    ----------
    minute_panel
        Long frame with ``symbol, trade_date, datetime, close``. ``high``/``low``
        are optional; ``close`` is used as the executable proxy.
    available_inventory
        Long frame with ``trade_date, symbol, available_shares``. These shares
        must be yesterday-settled inventory available for today's sell leg.
    """
    cfg = config or DoTOverlayConfig()
    cm = cost_model or AShareCostModel()
    if minute_panel is None or minute_panel.empty:
        return _empty_result()
    if available_inventory is None or available_inventory.empty:
        return _empty_result()

    inv = available_inventory[["trade_date", "symbol", "available_shares"]].copy()
    inv["trade_date"] = pd.to_datetime(inv["trade_date"], errors="coerce")
    inv["symbol"] = inv["symbol"].astype(str)
    inv["available_shares"] = pd.to_numeric(inv["available_shares"], errors="coerce").fillna(0.0)
    inv = inv[inv["available_shares"] >= cfg.round_lot]
    if inv.empty:
        return _empty_result()

    panel = minute_panel[["trade_date", "symbol", "datetime", "close"]].copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel["datetime"] = pd.to_datetime(panel["datetime"], errors="coerce")
    panel["symbol"] = panel["symbol"].astype(str)
    panel["close"] = pd.to_numeric(panel["close"], errors="coerce")
    panel = panel.dropna(subset=["trade_date", "datetime", "symbol", "close"])
    if panel.empty:
        return _empty_result()

    inv_idx = inv.set_index(["trade_date", "symbol"])["available_shares"]
    rows: list[dict[str, object]] = []
    for (date, symbol), g in panel.groupby(["trade_date", "symbol"], sort=False):
        available = float(inv_idx.get((date, symbol), 0.0))
        if available < cfg.round_lot:
            continue
        qty = _round_lot(available * cfg.trade_fraction, cfg.round_lot)
        if qty <= 0:
            continue
        trade = _best_round_trip(g.sort_values("datetime"), qty, cfg, cm)
        if trade is not None:
            rows.append({
                "trade_date": date,
                "symbol": symbol,
                "available_shares": available,
                **trade,
            })
    if not rows:
        return _empty_result()
    out = pd.DataFrame(rows).sort_values(["trade_date", "net_pnl"], ascending=[True, False])
    if cfg.max_trades_per_day > 0:
        out = (
            out.groupby("trade_date", group_keys=False)
            .head(int(cfg.max_trades_per_day))
            .reset_index(drop=True)
        )
    return out


def _best_round_trip(
    bars: pd.DataFrame,
    quantity: int,
    config: DoTOverlayConfig,
    cost_model: AShareCostModel,
) -> dict[str, object] | None:
    prices = bars["close"].to_numpy(dtype="float64")
    times = bars["datetime"].to_numpy()
    if len(prices) < config.min_minutes_between_legs + 2:
        return None
    best: dict[str, object] | None = None
    min_gap = max(1, int(config.min_minutes_between_legs))
    for i in range(0, len(prices) - min_gap):
        p0 = float(prices[i])
        if not np.isfinite(p0) or p0 <= 0:
            continue
        tail = prices[i + min_gap:]
        if tail.size == 0:
            continue
        low_j_rel = int(np.nanargmin(tail))
        high_j_rel = int(np.nanargmax(tail))
        candidates = [
            _candidate(
                "sell_high_buy_low",
                sell_idx=i,
                buy_idx=i + min_gap + low_j_rel,
                prices=prices,
                times=times,
                quantity=quantity,
                cost_model=cost_model,
            ),
            _candidate(
                "buy_low_sell_old_high",
                buy_idx=i,
                sell_idx=i + min_gap + high_j_rel,
                prices=prices,
                times=times,
                quantity=quantity,
                cost_model=cost_model,
            ),
        ]
        for cand in candidates:
            if cand is None:
                continue
            if cand["edge_pct"] < config.min_edge_pct:
                continue
            if best is None or float(cand["net_pnl"]) > float(best["net_pnl"]):
                best = cand
    return best


def _candidate(
    mode: str,
    *,
    buy_idx: int,
    sell_idx: int,
    prices: np.ndarray,
    times: np.ndarray,
    quantity: int,
    cost_model: AShareCostModel,
) -> dict[str, object] | None:
    buy_price = float(prices[buy_idx])
    sell_price = float(prices[sell_idx])
    if not all(np.isfinite([buy_price, sell_price])) or buy_price <= 0 or sell_price <= 0:
        return None
    edge_pct = sell_price / buy_price - 1.0
    if edge_pct <= 0:
        return None
    buy_cost = float(cost_model.calculate(OrderSide.BUY, quantity, buy_price)["total"])
    sell_cost = float(cost_model.calculate(OrderSide.SELL, quantity, sell_price)["total"])
    gross = float((sell_price - buy_price) * quantity)
    net = gross - buy_cost - sell_cost
    return {
        "mode": mode,
        "buy_time": pd.Timestamp(times[buy_idx]),
        "sell_time": pd.Timestamp(times[sell_idx]),
        "buy_price": buy_price,
        "sell_price": sell_price,
        "quantity": int(quantity),
        "edge_pct": float(edge_pct),
        "gross_pnl": gross,
        "cost": buy_cost + sell_cost,
        "net_pnl": net,
        "t1_legal": True,
    }


def _round_lot(quantity: float, lot: int) -> int:
    lot = max(1, int(lot))
    return int(quantity // lot * lot)


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "trade_date", "symbol", "available_shares", "mode", "buy_time",
        "sell_time", "buy_price", "sell_price", "quantity", "edge_pct",
        "gross_pnl", "cost", "net_pnl", "t1_legal",
    ])


__all__ = ["DoTOverlayConfig", "simulate_do_t_overlay"]
