import pandas as pd

from quantagent.execution.broker_base import Order, OrderSide, OrderType
from quantagent.execution.order_manager import OrderManager
from quantagent.execution.qmt_gateway import QMTGateway


def test_qmt_gateway_defaults_to_dry_run_without_xtquant():
    gateway = QMTGateway()
    gateway.connect()
    state = gateway.submit(Order("id-1", "600519.SH", OrderSide.BUY, 100, OrderType.LIMIT, 10.0))
    assert gateway.config.dry_run is True
    assert state.last_message == "dry_run_not_submitted_to_broker"


def test_order_manager_generates_metadata_rich_intents():
    manager = OrderManager(QMTGateway())
    intents = manager.target_weights_to_order_intents(
        pd.Series({"600519.SH": 0.1}),
        pd.Series({"600519.SH": 10.0}),
        nav=100_000.0,
        signal_id="sig",
        model_version="model",
        feature_version="feat",
        risk_check_result="pass",
    )
    assert intents[0].signal_id == "sig"
    assert intents[0].model_version == "model"
    assert intents[0].feature_version == "feat"
    assert intents[0].risk_check_result == "pass"
