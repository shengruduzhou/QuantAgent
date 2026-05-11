import pandas as pd

from quantagent.factors.composite import combine_with_model_gate
from quantagent.factors.pipeline_v6 import FactorPipeline


def test_factor_gate_is_consumed_by_composite_weights():
    statistical = pd.Series({"momentum": 0.2, "quality": 0.8})
    gate = pd.Series({"momentum": 0.9, "quality": 0.1})
    combined = combine_with_model_gate(statistical, gate, gate_strength=0.75)
    assert combined["momentum"] > statistical["momentum"]
    assert abs(combined.sum() - 1.0) < 1e-12


def test_factor_pipeline_writes_gate_audit_payload():
    frame = pd.DataFrame({"trade_date": ["2026-01-02"], "symbol": ["600000.SH"], "momentum": [1.0], "quality": [0.5]})
    result = FactorPipeline().run(
        frame,
        ["momentum", "quality"],
        pd.Series({"momentum": 0.5, "quality": 0.5}),
        pd.Series({"momentum": 0.8, "quality": 0.2}),
    )
    assert "composite_factor_score" in result.frame.columns
    assert "model_gate" in result.audit

