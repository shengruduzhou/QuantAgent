import numpy as np
import pandas as pd

from quantagent.factors.dag import FactorDAG, FactorNode
from quantagent.factors.governance import factor_group_metrics
from quantagent.factors.lifecycle import build_factor_lifecycle_report, recommend_factor_status


def _panel():
    dates = pd.date_range("2026-01-01", periods=30)
    rows = []
    for j, symbol in enumerate(["A", "B", "C", "D", "E"]):
        for i, date in enumerate(dates):
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "close": 10 + i * 0.1 + j,
                    "amount": 1e7 + j * 1e6,
                    "factor": float(j),
                    "existing": float(j) * 0.5 + np.sin(i),
                    "future_return": j * 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_factor_dag_executes_dependencies_in_order():
    dag = FactorDAG()
    dag.add(FactorNode("base", lambda f: f["close"] * 2, required_columns=("close",)))
    dag.add(FactorNode("derived", lambda f: f["base"] + 1, dependencies=("base",), required_columns=("base",)))
    result = dag.execute(_panel(), selected=["derived"])
    assert result.execution_order == ("base", "derived")
    assert "derived" in result.frame.columns


def test_factor_lifecycle_statuses_and_group_metrics():
    frame = _panel()
    report = build_factor_lifecycle_report(frame, "factor", "future_return", existing_factor_columns=["existing"])
    assert report.recommended_status in {"active", "degraded", "retired", "watch"}
    assert recommend_factor_status(0.2, 0.8, 0.8) == "active"
    assert recommend_factor_status(-0.2, 0.2, 0.0) == "retired"
    metrics = factor_group_metrics([report], {"factor": "alpha"})
    assert metrics.loc[0, "group"] == "alpha"
