import pandas as pd

from quantagent.execution.broker_base import OrderIntent, OrderSide
from quantagent.risk.risk_gate import RiskGate
from quantagent.risk.risk_limits import V6RiskLimits


def test_risk_gate_rejects_illegal_targets_and_orders():
    gate = RiskGate(V6RiskLimits(max_name_weight=0.05, max_order_value=1000))
    weights = pd.Series({"600000.SH": 0.20})
    market = pd.DataFrame({"symbol": ["600000.SH"], "is_suspended": [True], "is_limit_up": [False], "is_limit_down": [False], "is_st": [False]})
    target_result = gate.check_target_weights(weights, market_state=market)
    assert target_result.rejected_symbols["600000.SH"] in {"max_name_weight", "suspended"}
    assert target_result.checked_weights.loc["600000.SH"] == 0.0

    intent = OrderIntent("id1", "600000.SH", OrderSide.BUY, 100, 0.05, 20.0, "sig", "v6", "fv", "v6", "checked", "2026-01-02")
    order_result = gate.check_order_intents([intent], market_state=market, cash_available=100000)
    assert not order_result.passed
    assert order_result.rejected_symbols["id1"] == "suspended"

