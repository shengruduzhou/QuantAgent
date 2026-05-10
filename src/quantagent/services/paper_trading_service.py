from __future__ import annotations

import pandas as pd

from quantagent.execution.order_manager import OrderManager
from quantagent.execution.qmt_gateway import QMTConfig, QMTGateway


def generate_dry_run_order_intents(
    target_weights: pd.Series,
    prices: pd.Series,
    nav: float = 1_000_000.0,
    audit_log_dir: str = "logs/execution",
) -> list[object]:
    gateway = QMTGateway(QMTConfig(dry_run=True, audit_log_dir=audit_log_dir))
    manager = OrderManager(gateway)
    return manager.target_weights_to_order_intents(
        target_weights,
        prices,
        nav,
        signal_id="synthetic_v4",
        model_version="v4_multitower_tiny",
        feature_version="synthetic_v4",
        risk_check_result="dry_run_pass",
    )


def paper_trade_v4(
    target_weights: pd.Series,
    prices: pd.Series,
    nav: float = 1_000_000.0,
    dry_run: bool = True,
    audit_log_dir: str = "logs/execution",
) -> list[object]:
    gateway = QMTGateway(QMTConfig(dry_run=dry_run, live_trading_enabled=False, audit_log_dir=audit_log_dir))
    manager = OrderManager(gateway)
    return manager.reconcile(target_weights, prices, nav)
