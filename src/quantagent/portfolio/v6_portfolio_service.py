from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantagent.agents.agent_reliability import AgentReliability
from quantagent.agents.agent_router import AgentRouter
from quantagent.agents.bayesian_arbitration import reliability_weighted_posterior
from quantagent.agents.views_schema import EvidenceRecord
from quantagent.quant_math.optimizer import V4PortfolioConfig, V4PortfolioResult
from quantagent.quant_math.regime_aware_optimizer import RegimeAwareConfig, solve_v5_portfolio
from quantagent.risk.risk_gate import RiskGate, RiskGateResult
from quantagent.risk.risk_limits import V6RiskLimits


@dataclass(frozen=True)
class V6PortfolioBuildResult:
    target_weights: pd.Series
    posterior_alpha: pd.Series
    optimizer_result: V4PortfolioResult
    risk_result: RiskGateResult
    agent_view_count: int
    diagnostics: dict[str, object]


def build_v6_portfolio(
    model_outputs: pd.DataFrame,
    evidence: list[EvidenceRecord] | None = None,
    market_state: pd.DataFrame | None = None,
    current_weights: pd.Series | None = None,
    config: dict[str, Any] | None = None,
    reliability: AgentReliability | None = None,
    risk_gate: RiskGate | None = None,
) -> V6PortfolioBuildResult:
    cfg = config or {}
    if model_outputs.empty:
        empty = pd.Series(dtype=float)
        optimizer_result = V4PortfolioResult(empty, 0.0, 0.0, 0.0, {}, {}, "empty")
        risk_result = RiskGateResult(True, checked_weights=empty)
        return V6PortfolioBuildResult(empty, empty, optimizer_result, risk_result, 0, {"status": "empty"})
    data = model_outputs.copy()
    alpha = data.set_index("symbol")["alpha_5d"].astype(float)
    conformal = data.set_index("symbol").get("conformal_confidence", pd.Series(1.0, index=alpha.index)).astype(float)
    confidence = data.set_index("symbol").get("confidence", pd.Series(1.0, index=alpha.index)).astype(float)
    risk_score = data.set_index("symbol").get("risk_score", pd.Series(0.0, index=alpha.index)).astype(float)
    adjusted_alpha = alpha * conformal.clip(0.0, 1.0) * confidence.clip(0.0, 1.0) * (1.0 - risk_score.clip(0.0, 1.0) * 0.5)
    router = AgentRouter(reliability=reliability)
    routed = router.route(evidence or [], adjusted_alpha.index)
    posterior = reliability_weighted_posterior(adjusted_alpha, routed.views)
    symbols = posterior.index
    covariance = pd.DataFrame(np.eye(len(symbols)) * 0.04, index=symbols, columns=symbols)
    current = current_weights.reindex(symbols).fillna(0.0) if current_weights is not None else pd.Series(0.0, index=symbols)
    cost = pd.Series(0.0008, index=symbols)
    sector = _sector_series(market_state, symbols)
    portfolio_cfg = cfg.get("portfolio", cfg)
    base_config = V4PortfolioConfig(
        max_name_weight=float(portfolio_cfg.get("max_name_weight", 0.05)),
        max_sector_weight=float(portfolio_cfg.get("max_sector_weight", 0.30)),
        max_turnover=float(portfolio_cfg.get("max_turnover", 0.30)),
    )
    regime = str(data.get("regime", pd.Series(["range_bound"])).iloc[0])
    optimizer_result = solve_v5_portfolio(
        posterior,
        covariance,
        current_weights=current,
        cost=cost,
        sector=sector,
        tradability=market_state,
        config=RegimeAwareConfig(regime=regime, base_config=base_config),
    )
    risk_cfg = cfg.get("risk", cfg.get("portfolio", {}))
    limits = V6RiskLimits(
        max_name_weight=float(risk_cfg.get("max_name_weight", base_config.max_name_weight)),
        max_sector_weight=float(risk_cfg.get("max_sector_weight", base_config.max_sector_weight)),
        max_turnover=float(risk_cfg.get("max_turnover", base_config.max_turnover)),
        min_data_quality_score=float(cfg.get("data", {}).get("min_data_quality_score", 0.85)) if "data" in cfg else float(risk_cfg.get("min_data_quality_score", 0.85)),
    )
    gate = risk_gate or RiskGate(limits)
    width = data.set_index("symbol")["q_high"].astype(float) - data.set_index("symbol")["q_low"].astype(float) if {"q_high", "q_low"}.issubset(data.columns) else None
    risk_result = gate.check_target_weights(
        optimizer_result.target_weights,
        current_weights=current,
        market_state=market_state,
        sector=sector,
        data_quality_score=float(cfg.get("data_quality_score", 1.0)),
        model_drift_score=float(cfg.get("model_drift_score", 0.0)),
        conformal_width=width,
    )
    checked = risk_result.checked_weights if risk_result.checked_weights is not None else optimizer_result.target_weights
    diagnostics = {
        "routed_views": len(routed.views),
        "risk_warnings": routed.risk_warnings,
        "optimizer_status": optimizer_result.status,
        "rejected_symbols": risk_result.rejected_symbols,
        "violations": risk_result.violations,
    }
    return V6PortfolioBuildResult(checked.sort_index(), posterior.sort_index(), optimizer_result, risk_result, len(routed.views), diagnostics)


def _sector_series(market_state: pd.DataFrame | None, symbols: pd.Index) -> pd.Series | None:
    if market_state is None or market_state.empty or "sector" not in market_state.columns:
        return None
    return market_state.drop_duplicates("symbol").set_index("symbol")["sector"].reindex(symbols)

