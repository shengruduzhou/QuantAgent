from __future__ import annotations

import pandas as pd

from quantagent.quant_math.optimizer import V4PortfolioResult
from quantagent.portfolio.v6_portfolio_service import build_v6_portfolio as build_unified_portfolio


def build_portfolio_v4(signals: pd.DataFrame, mode: str = "long_only_enhancement") -> V4PortfolioResult:
    """Compatibility wrapper that delegates portfolio construction to V6."""
    if {"alpha_5d", "q_low", "q_high", "risk_score", "conformal_confidence"}.issubset(signals.columns):
        model_outputs = signals.copy()
    else:
        model_outputs = signals.rename(columns={"alpha": "alpha_5d"}).copy()
        model_outputs["alpha_1d"] = model_outputs["alpha_5d"] / 3.0
        model_outputs["alpha_20d"] = model_outputs["alpha_5d"] * 1.8
        model_outputs["q_low"] = model_outputs["alpha_5d"] - 0.01
        model_outputs["q_high"] = model_outputs["alpha_5d"] + 0.01
        model_outputs["conformal_confidence"] = model_outputs.get("confidence", 0.5)
        model_outputs["risk_score"] = 0.0
        model_outputs["regime"] = "range_bound"
    result = build_unified_portfolio(
        model_outputs,
        evidence=[],
        config={"portfolio": {"max_name_weight": 0.05, "max_sector_weight": 0.30, "max_turnover": 0.30}},
    )
    optimizer = result.optimizer_result
    return V4PortfolioResult(
        target_weights=result.target_weights,
        expected_turnover=optimizer.expected_turnover,
        expected_cost=optimizer.expected_cost,
        active_risk_proxy=optimizer.active_risk_proxy,
        constraint_diagnostics=optimizer.constraint_diagnostics | {"compatibility_entrypoint": "build_portfolio_v4"},
        rejected_symbols=result.risk_result.rejected_symbols | optimizer.rejected_symbols,
        status=optimizer.status,
    )
