from __future__ import annotations

import pandas as pd

from quantagent.execution.order_manager import OrderManager
from quantagent.execution.virtual_broker import VirtualBroker


def generate_dry_run_order_intents(
    target_weights: pd.Series,
    prices: pd.Series,
    nav: float = 1_000_000.0,
    audit_log_dir: str = "logs/execution",
) -> list[object]:
    broker = VirtualBroker(initial_cash=nav, dry_run=True, audit_log_dir=audit_log_dir)
    manager = OrderManager(broker)
    return manager.target_weights_to_order_intents(
        target_weights,
        prices,
        nav,
        signal_id="synthetic_v4",
        model_version="compat_unified_multitower",
        feature_version="synthetic_compat",
        risk_check_result="dry_run_pass",
    )


def paper_trade_v4(
    target_weights: pd.Series,
    prices: pd.Series,
    nav: float = 1_000_000.0,
    dry_run: bool = True,
    audit_log_dir: str = "logs/execution",
) -> list[object]:
    broker = VirtualBroker(initial_cash=nav, dry_run=dry_run, audit_log_dir=audit_log_dir)
    broker.set_market_state([{"symbol": str(symbol), "volume": 1_000_000, "is_suspended": False, "is_limit_up": False, "is_limit_down": False} for symbol in prices.index])
    manager = OrderManager(broker)
    return manager.reconcile(target_weights, prices, nav)
