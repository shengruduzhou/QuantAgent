import pandas as pd

from quantagent.execution.broker_base import Position
from quantagent.execution.reconciliation import reconcile_virtual_state


def test_v6_reconciliation_report_contains_required_fields():
    weights = pd.Series({"600000.SH": 0.01})
    prices = pd.Series({"600000.SH": 10.0})
    report = reconcile_virtual_state(weights, prices, [Position("600000.SH", 100, 0, 10.0)], nav=100000, cash_expected=99000, cash_actual=99000)
    assert report.expected_position["600000.SH"] == 100.0
    assert report.broker_position["600000.SH"] == 100.0
    assert report.unresolved_orders == 0
    assert report.fill_rate == 1.0

