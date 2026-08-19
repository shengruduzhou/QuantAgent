"""Production-grade A-share execution simulation around signal-dated target weights.

The target-weight index is a **signal date**, never an execution date.  Under the
canonical executable-label contract a signal produced at session T close can
only be executed at the close of the next market session.  The simulator emits
a machine-readable trace so strict evaluation can prove that timing rather than
trusting a summary boolean.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import pandas as pd

from quantagent.backtest.execution_timing import (
    EXECUTION_TIMING_SEMANTICS,
    TRACE_SCHEMA_VERSION,
)
from quantagent.config.paths import quant_paths
from quantagent.domain.lineage import Lineage
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
    # Target-weight rows are signal-dated. There is intentionally no supported
    # same-session execution mode at this production-grade boundary.
    execution_timing_semantics: str = EXECUTION_TIMING_SEMANTICS
    # INC-E1 (EVALUATOR_ORDER_DEDUP_BUG.md): PROMOTED to trusted-evaluator
    # default 2026-07-06 with user approval. False is retained only for the
    # isolated forensic harness; the facade separately prevents unsafe brokers.
    fix_cross_day_order_dedup: bool = True

    def __post_init__(self) -> None:
        if self.execution_timing_semantics != EXECUTION_TIMING_SEMANTICS:
            raise ValueError(
                "production A-share simulator only supports "
                f"{EXECUTION_TIMING_SEMANTICS}; got {self.execution_timing_semantics!r}"
            )


@dataclass(frozen=True)
class AShareExecutionSimulationResult:
    nav: pd.Series
    order_audit: pd.DataFrame
    position_history: pd.DataFrame
    failed_order_audit: pd.DataFrame
    skipped_order_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    risk_events: list[dict] = field(default_factory=list)
    config: dict[str, object] = field(default_factory=dict)
    execution_trace: pd.DataFrame = field(default_factory=pd.DataFrame)

    def write_risk_events(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.risk_events, indent=2, default=str), encoding="utf-8"
        )
        return target

    def write_execution_trace(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.execution_trace.to_csv(target, index=False)
        return target


def simulate_ashare_target_weights(
    target_weight_history: pd.DataFrame,
    market_panel: pd.DataFrame,
    config: AShareExecutionSimulationConfig | None = None,
) -> AShareExecutionSimulationResult:
    """Replay signal-dated targets on the next available market-session close.

    Invalid timing/data never falls back to same-day execution.  The generic
    simulator records an ``execution_error`` and skips that signal atomically;
    :func:`run_strict_backtest_v8` validates the trace and rejects such a run.
    """
    config = config or AShareExecutionSimulationConfig()
    audit_log_dir = config.audit_log_dir or str(quant_paths().logs / "v7_backtest")
    if target_weight_history is None or target_weight_history.empty:
        return AShareExecutionSimulationResult(
            nav=pd.Series(dtype=float),
            order_audit=pd.DataFrame(),
            position_history=pd.DataFrame(),
            failed_order_audit=pd.DataFrame(),
            skipped_order_audit=pd.DataFrame(),
            risk_events=[],
            config=asdict(config),
            execution_trace=pd.DataFrame(),
        )
    if market_panel is None or market_panel.empty:
        return AShareExecutionSimulationResult(
            nav=pd.Series(dtype=float),
            order_audit=pd.DataFrame(),
            position_history=pd.DataFrame(),
            failed_order_audit=pd.DataFrame(),
            skipped_order_audit=pd.DataFrame(),
            risk_events=[],
            config=asdict(config),
            execution_trace=_unmapped_trace_for_all_signals(
                target_weight_history,
                reason="market_panel_empty",
            ),
        )

    market = market_panel.copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce").dt.normalize()
    market = market.dropna(subset=["trade_date", "symbol"]).sort_values(["trade_date", "symbol"])
    target = target_weight_history.copy()
    target.index = pd.to_datetime(target.index, errors="coerce").normalize()
    target = target[~target.index.isna()].sort_index()
    if target.index.duplicated().any():
        duplicates = sorted({str(pd.Timestamp(value).date()) for value in target.index[target.index.duplicated(keep=False)]})
        raise ValueError(f"target weights contain duplicate signal dates: {duplicates[:5]}")

    sessions = pd.DatetimeIndex(sorted(pd.unique(market["trade_date"])))
    session_set = set(sessions)

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
        lineage=Lineage(run_id=f"ashare_sim_{id(config):x}", strategy_version_id="v7_ashare_simulation"),
        forensic_replay=not config.fix_cross_day_order_dedup,
        config=OrderManagerConfig(
            lot_size=config.lot_size,
            min_order_value_yuan=config.min_order_value_yuan,
            allow_odd_lot_sell_only_for_full_liquidation=config.allow_odd_lot_sell_only_for_full_liquidation,
            # `volume_participation_cap` is not passed to the order manager: it
            # meters fills, and the order manager does not match orders.  It
            # reaches the only component that can act on it, the FillSimulator
            # above.  Until Round 24 it was also written onto OrderManagerConfig,
            # where nothing read it -- a caller setting the cap here believed it
            # had constrained routing as well, and it never had.
            strategy_version="v7_ashare_simulation",
        ),
    )
    nav_rows: list[tuple[pd.Timestamp, float]] = []
    order_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    risk_events: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []

    for signal_date, weights in target.iterrows():
        signal_date = pd.Timestamp(signal_date).normalize()
        execution_date = _next_market_session(signal_date, sessions) if signal_date in session_set else None
        if execution_date is None:
            reason = (
                "signal_date_not_market_session"
                if signal_date not in session_set
                else "no_next_market_session"
            )
            trace_rows.append(_trace_row(
                record_type="session_mapping",
                signal_date=signal_date,
                execution_date=None,
                status="unmapped",
                reason=reason,
                price_source="close",
            ))
            continue

        trace_rows.append(_trace_row(
            record_type="session_mapping",
            signal_date=signal_date,
            execution_date=execution_date,
            status="mapped",
            reason="",
            price_source="close",
        ))

        day_market = market[market["trade_date"] == execution_date]
        if day_market.empty:
            trace_rows.append(_trace_row(
                record_type="execution_error",
                signal_date=signal_date,
                execution_date=execution_date,
                status="blocked",
                reason="execution_session_missing",
                price_source="close",
            ))
            continue
        if day_market["symbol"].astype(str).duplicated().any():
            trace_rows.append(_trace_row(
                record_type="execution_error",
                signal_date=signal_date,
                execution_date=execution_date,
                status="blocked",
                reason="duplicate_symbol_execution_bar",
                price_source="close",
            ))
            continue
        if "close" not in day_market.columns:
            trace_rows.append(_trace_row(
                record_type="execution_error",
                signal_date=signal_date,
                execution_date=execution_date,
                status="blocked",
                reason="execution_close_column_missing",
                price_source="close",
            ))
            continue

        market_symbols = set(day_market["symbol"].astype(str))
        held_symbols = {
            str(position.symbol)
            for position in broker.query_positions()
            if int(position.available_shares) + int(position.frozen_shares) > 0
        }
        desired_active = {
            str(symbol)
            for symbol, weight in weights.items()
            if pd.notna(weight) and abs(float(weight)) > manager.config.negligible_weight
        }
        required_symbols = held_symbols | desired_active
        missing_bars = sorted(required_symbols - market_symbols)
        if missing_bars:
            for symbol in missing_bars:
                trace_rows.append(_trace_row(
                    record_type="execution_error",
                    signal_date=signal_date,
                    execution_date=execution_date,
                    status="blocked",
                    reason="missing_execution_bar",
                    symbol=symbol,
                    price_source="close",
                ))
            continue

        close_by_symbol = pd.to_numeric(
            day_market.set_index(day_market["symbol"].astype(str))["close"],
            errors="coerce",
        )
        invalid_required = sorted(
            symbol for symbol in required_symbols
            if symbol not in close_by_symbol.index
            or pd.isna(close_by_symbol.loc[symbol])
            or float(close_by_symbol.loc[symbol]) <= 0
        )
        if invalid_required:
            for symbol in invalid_required:
                trace_rows.append(_trace_row(
                    record_type="execution_error",
                    signal_date=signal_date,
                    execution_date=execution_date,
                    status="blocked",
                    reason="invalid_execution_close",
                    symbol=symbol,
                    price_source="close",
                ))
            continue

        # The economic day advances only at the mapped execution session. This
        # releases T+1-frozen shares before a later-session sell can be routed.
        broker.advance_trading_day()
        if config.fix_cross_day_order_dedup:
            manager.reset_daily_counters()
        broker.set_market_state(day_market.to_dict("records"))
        prices = close_by_symbol.dropna()
        prices = prices[prices > 0]
        if prices.empty:
            trace_rows.append(_trace_row(
                record_type="execution_error",
                signal_date=signal_date,
                execution_date=execution_date,
                status="blocked",
                reason="no_valid_execution_prices",
                price_source="close",
            ))
            continue

        current_weights = _current_weights(broker, prices)
        adjusted = _apply_st_policy(weights.astype(float), current_weights, day_market, config)
        nav = _mark_to_market_nav(broker, prices)
        day_signal_id = f"bt-sig-{signal_date:%Y%m%d}-exec-{execution_date:%Y%m%d}"
        states = manager.reconcile(adjusted, prices, nav, signal_id=day_signal_id)

        for skipped in manager.last_skipped_orders:
            skipped_row = {
                "trade_date": execution_date,
                "signal_date": signal_date,
                "execution_date": execution_date,
                "price_source": "close",
                **skipped,
            }
            skipped_rows.append(skipped_row)
            trace_rows.append(_trace_row(
                record_type="skip",
                signal_date=signal_date,
                execution_date=execution_date,
                status="skipped",
                reason=str(skipped.get("reason") or ""),
                symbol=str(skipped.get("symbol") or ""),
                client_order_id="",
                price_source="close",
                reference_price=skipped.get("reference_price"),
                target_weight=skipped.get("target_weight"),
            ))

        for state in states:
            order = broker.order_objects.get(state.client_order_id)
            row: dict[str, object] = {
                "trade_date": execution_date,
                "signal_date": signal_date,
                "execution_date": execution_date,
                "price_source": "close",
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
            trace_rows.append(_trace_row(
                record_type="order",
                signal_date=signal_date,
                execution_date=execution_date,
                status=state.status.value,
                reason=str(state.last_message or ""),
                symbol="" if order is None else str(order.symbol),
                client_order_id=str(state.client_order_id),
                price_source="close",
                reference_price=None if order is None else order.price,
                target_weight=None if order is None else float(adjusted.get(order.symbol, 0.0)),
            ))
            if state.status.value in ("rejected", "cancelled", "partial"):
                risk_events.append(
                    {
                        "trade_date": str(execution_date),
                        "signal_date": str(signal_date),
                        "execution_date": str(execution_date),
                        "event_type": f"order_{state.status.value}",
                        "client_order_id": state.client_order_id,
                        "symbol": getattr(order, "symbol", None) if order else None,
                        "reason": state.last_message,
                        "filled_quantity": state.filled_quantity,
                    }
                )
        for skipped in manager.last_skipped_orders:
            risk_events.append(
                {
                    "trade_date": str(execution_date),
                    "signal_date": str(signal_date),
                    "execution_date": str(execution_date),
                    "event_type": "order_skipped",
                    "symbol": skipped.get("symbol"),
                    "side": skipped.get("side"),
                    "reason": skipped.get("reason"),
                    "target_weight": skipped.get("target_weight"),
                }
            )

        nav_after = _mark_to_market_nav(broker, prices)
        nav_rows.append((execution_date, nav_after))
        for position in broker.query_positions():
            price = _position_price(position, prices)
            position_rows.append(
                {
                    "trade_date": execution_date,
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "symbol": position.symbol,
                    "available_shares": position.available_shares,
                    "frozen_shares": position.frozen_shares,
                    "market_value": (position.available_shares + position.frozen_shares) * price,
                }
            )

    order_audit = pd.DataFrame(order_rows)
    failed = (
        order_audit[order_audit["status"].isin(["rejected", "cancelled"])]
        if not order_audit.empty
        else pd.DataFrame()
    )
    metadata = asdict(config)
    metadata["execution_trace_schema"] = TRACE_SCHEMA_VERSION
    metadata["target_index_semantics"] = "signal_date"
    metadata["execution_price_source"] = "close"
    return AShareExecutionSimulationResult(
        nav=pd.Series(dict(nav_rows), name="nav").sort_index(),
        order_audit=order_audit,
        position_history=pd.DataFrame(position_rows),
        failed_order_audit=failed.reset_index(drop=True),
        skipped_order_audit=pd.DataFrame(skipped_rows),
        risk_events=risk_events,
        config=metadata,
        execution_trace=pd.DataFrame(trace_rows),
    )


def _next_market_session(signal_date: pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Timestamp | None:
    position = int(sessions.searchsorted(pd.Timestamp(signal_date), side="right"))
    if position >= len(sessions):
        return None
    return pd.Timestamp(sessions[position]).normalize()


def _unmapped_trace_for_all_signals(
    target_weight_history: pd.DataFrame,
    *,
    reason: str,
) -> pd.DataFrame:
    rows = []
    for value in pd.to_datetime(target_weight_history.index, errors="coerce"):
        if pd.isna(value):
            continue
        rows.append(_trace_row(
            record_type="session_mapping",
            signal_date=pd.Timestamp(value).normalize(),
            execution_date=None,
            status="unmapped",
            reason=reason,
            price_source="close",
        ))
    return pd.DataFrame(rows)


def _trace_row(
    *,
    record_type: str,
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp | None,
    status: str,
    reason: str,
    symbol: str = "",
    client_order_id: str = "",
    price_source: str = "close",
    reference_price: object = None,
    target_weight: object = None,
) -> dict[str, object]:
    return {
        "record_type": record_type,
        "signal_date": pd.Timestamp(signal_date).normalize(),
        "execution_date": None if execution_date is None else pd.Timestamp(execution_date).normalize(),
        "status": status,
        "reason": reason,
        "symbol": symbol,
        "client_order_id": client_order_id,
        "price_source": price_source,
        "reference_price": reference_price,
        "target_weight": target_weight,
        "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
        "trace_schema": TRACE_SCHEMA_VERSION,
    }


def _current_weights(broker: VirtualBroker, prices: pd.Series) -> pd.Series:
    positions = broker.query_positions()
    nav = _mark_to_market_nav(broker, prices)
    values = {
        position.symbol: (position.available_shares + position.frozen_shares) * _position_price(position, prices)
        for position in positions
    }
    return pd.Series(values, dtype=float).div(nav).fillna(0.0) if nav > 0 else pd.Series(dtype=float)


def _mark_to_market_nav(broker: VirtualBroker, prices: pd.Series) -> float:
    cash = float(broker.ledger.cash)
    value = 0.0
    for position in broker.query_positions():
        shares = position.available_shares + position.frozen_shares
        value += shares * _position_price(position, prices)
    return cash + value


def _position_price(position: object, prices: pd.Series) -> float:
    fallback = float(getattr(position, "avg_cost", 0.0))
    price_value = prices.get(getattr(position, "symbol", ""), fallback)
    return fallback if pd.isna(price_value) else float(price_value)


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
