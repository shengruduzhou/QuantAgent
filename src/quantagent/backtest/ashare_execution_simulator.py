"""Production-grade A-share execution simulation around target_weights and OrderManager."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.execution.fill_simulator import FillSimulator
from quantagent.execution.order_manager import OrderManager, OrderManagerConfig
from quantagent.execution.virtual_broker import VirtualBroker


@dataclass(frozen=True)
class AShareExecutionSimulationConfig:
    initial_cash: float = 1_000_000.0
    lot_size: int = 100
    min_order_value_yuan: float = 100.0
    allow_odd_lot_sell_only_for_full_liquidation: bool = True
    volume_participation_cap: float = 0.10
    slippage_bps: float = 8.0
    block_st_buy: bool = True
    max_st_weight: float = 0.0
    audit_log_dir: str | None = None


@dataclass(frozen=True)
class AShareExecutionSimulationResult:
    nav: pd.Series
    order_audit: pd.DataFrame
    position_history: pd.DataFrame
    failed_order_audit: pd.DataFrame
    skipped_order_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: dict[str, object] = field(default_factory=dict)


def simulate_ashare_target_weights(
    target_weight_history: pd.DataFrame,
    market_panel: pd.DataFrame,
    config: AShareExecutionSimulationConfig | None = None,
) -> AShareExecutionSimulationResult:
    config = config or AShareExecutionSimulationConfig()
    audit_log_dir = config.audit_log_dir or str(quant_paths().logs / "v7_backtest")
    if target_weight_history is None or target_weight_history.empty:
        return AShareExecutionSimulationResult(pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), asdict(config))
    market = market_panel.copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")
    market = market.dropna(subset=["trade_date", "symbol"]).sort_values(["trade_date", "symbol"])
    target = target_weight_history.copy()
    target.index = pd.to_datetime(target.index, errors="coerce")
    target = target[~target.index.isna()].sort_index()

    broker = VirtualBroker(
        initial_cash=config.initial_cash,
        dry_run=True,
        audit_log_dir=audit_log_dir,
        fill_simulator=FillSimulator(
            participation_rate=config.volume_participation_cap,
            slippage_bps=config.slippage_bps,
        ),
    )
    manager = OrderManager(
        broker=broker,
        config=OrderManagerConfig(
            lot_size=config.lot_size,
            min_order_value_yuan=config.min_order_value_yuan,
            allow_odd_lot_sell_only_for_full_liquidation=config.allow_odd_lot_sell_only_for_full_liquidation,
            max_participation_rate=config.volume_participation_cap,
            strategy_version="v7_ashare_simulation",
        ),
    )
    nav_rows: list[tuple[pd.Timestamp, float]] = []
    order_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []

    for date, weights in target.iterrows():
        day_market = market[market["trade_date"] == date]
        if day_market.empty:
            continue
        broker.advance_trading_day()
        broker.set_market_state(day_market.to_dict("records"))
        prices = day_market.set_index("symbol")["close"].astype(float)
        current_weights = _current_weights(broker, prices)
        adjusted = _apply_st_policy(weights.astype(float), current_weights, day_market, config)
        nav = _mark_to_market_nav(broker, prices)
        states = manager.reconcile(adjusted, prices, nav)
        for skipped in manager.last_skipped_orders:
            skipped_rows.append({"trade_date": date, **skipped})
        for state in states:
            order = broker.order_objects.get(state.client_order_id)
            row = {
                "trade_date": date,
                "client_order_id": state.client_order_id,
                "status": state.status.value,
                "filled_quantity": state.filled_quantity,
                "avg_price": state.avg_price,
                "last_message": state.last_message,
            }
            if order is not None:
                row |= {
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": order.quantity,
                    "reference_price": order.price,
                }
            order_rows.append(row)
        nav_after = _mark_to_market_nav(broker, prices)
        nav_rows.append((date, nav_after))
        for position in broker.query_positions():
            price = float(prices.get(position.symbol, position.avg_cost))
            position_rows.append(
                {
                    "trade_date": date,
                    "symbol": position.symbol,
                    "available_shares": position.available_shares,
                    "frozen_shares": position.frozen_shares,
                    "market_value": (position.available_shares + position.frozen_shares) * price,
                }
            )

    order_audit = pd.DataFrame(order_rows)
    failed = order_audit[order_audit["status"].isin(["rejected", "cancelled"])] if not order_audit.empty else pd.DataFrame()
    return AShareExecutionSimulationResult(
        nav=pd.Series(dict(nav_rows), name="nav").sort_index(),
        order_audit=order_audit,
        position_history=pd.DataFrame(position_rows),
        failed_order_audit=failed.reset_index(drop=True),
        skipped_order_audit=pd.DataFrame(skipped_rows),
        config=asdict(config),
    )


def _current_weights(broker: VirtualBroker, prices: pd.Series) -> pd.Series:
    positions = broker.query_positions()
    nav = _mark_to_market_nav(broker, prices)
    values = {
        position.symbol: (position.available_shares + position.frozen_shares) * float(prices.get(position.symbol, position.avg_cost))
        for position in positions
    }
    return pd.Series(values, dtype=float).div(nav).fillna(0.0) if nav > 0 else pd.Series(dtype=float)


def _mark_to_market_nav(broker: VirtualBroker, prices: pd.Series) -> float:
    cash = float(broker.ledger.cash)
    value = 0.0
    for position in broker.query_positions():
        shares = position.available_shares + position.frozen_shares
        value += shares * float(prices.get(position.symbol, position.avg_cost))
    return cash + value


def _apply_st_policy(
    target_weights: pd.Series,
    current_weights: pd.Series,
    day_market: pd.DataFrame,
    config: AShareExecutionSimulationConfig,
) -> pd.Series:
    if "is_st" not in day_market.columns:
        return target_weights
    st_symbols = set(day_market.loc[day_market["is_st"].fillna(False).astype(bool), "symbol"].astype(str))
    adjusted = target_weights.copy()
    for symbol in st_symbols:
        current = float(current_weights.get(symbol, 0.0))
        desired = float(adjusted.get(symbol, 0.0))
        if config.block_st_buy and desired > current:
            adjusted.loc[symbol] = current
        if config.max_st_weight >= 0 and desired > config.max_st_weight:
            adjusted.loc[symbol] = min(float(adjusted.get(symbol, 0.0)), max(current, config.max_st_weight))
    return adjusted
