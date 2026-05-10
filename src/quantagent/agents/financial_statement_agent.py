from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.domain.schemas import AgentSignal, EvidenceItem
from quantagent.fundamental.forensic_accounting import fraud_risk_composite
from quantagent.fundamental.target_price import TargetPriceEstimate, final_target_price_band
from quantagent.fundamental.valuation import DCFInputs


@dataclass(frozen=True)
class FinancialStatementAgentOutput:
    signal: AgentSignal
    target_price: TargetPriceEstimate
    thesis: str
    red_flags: tuple[str, ...]


class FinancialStatementAgent:
    def run(
        self,
        symbol: str,
        statements: pd.DataFrame,
        current_price: float,
        dcf_inputs: DCFInputs,
        news_evidence: tuple[EvidenceItem, ...] = (),
        relative_price: float | None = None,
    ) -> FinancialStatementAgentOutput:
        data = statements.copy()
        latest = data.sort_values("report_date").tail(1)
        quality = _quality_from_latest(latest)
        fraud_frame = fraud_risk_composite(data)
        fraud_risk = float(fraud_frame["fraud_risk_composite"].dropna().tail(1).iloc[0]) if fraud_frame["fraud_risk_composite"].dropna().shape[0] else 0.5
        target = final_target_price_band(
            symbol=symbol,
            current_price=current_price,
            dcf_inputs=dcf_inputs,
            relative_price=relative_price,
            fraud_risk=fraud_risk,
            quality_score=quality,
        )
        strength = float(np.tanh(target.expected_upside * 2.0)) if np.isfinite(target.expected_upside) else 0.0
        risk_penalty = float(np.clip(fraud_risk + max(0.0, 40.0 - quality) / 100.0, 0.0, 1.0))
        signal = AgentSignal(
            agent_name="financial_statement_agent",
            symbol=symbol,
            horizon_days=120,
            signal_strength=strength,
            confidence=target.confidence,
            evidence_quality=0.65 if news_evidence else 0.55,
            risk_penalty=risk_penalty,
            evidence=news_evidence,
            tags=("fundamental", "target_price"),
        )
        red_flags = target.risk_flags
        thesis = _thesis(target.expected_upside, quality, fraud_risk)
        return FinancialStatementAgentOutput(signal=signal, target_price=target, thesis=thesis, red_flags=red_flags)


def _quality_from_latest(latest: pd.DataFrame) -> float:
    if latest.empty:
        return 50.0
    row = latest.iloc[0]
    metrics = []
    if "roe" in latest.columns:
        metrics.append(float(np.clip(row["roe"] / 0.15, 0.0, 1.0)))
    if "roic" in latest.columns:
        metrics.append(float(np.clip(row["roic"] / 0.12, 0.0, 1.0)))
    if "gross_margin" in latest.columns:
        metrics.append(float(np.clip(row["gross_margin"], 0.0, 1.0)))
    if not metrics:
        return 50.0
    return float(100.0 * np.mean(metrics))


def _thesis(expected_upside: float, quality: float, fraud_risk: float) -> str:
    if expected_upside > 0.2 and quality >= 60 and fraud_risk < 0.5:
        return "valuation upside is supported by quality and clean accounting signals"
    if fraud_risk >= 0.7:
        return "fundamental upside is constrained by accounting risk"
    if quality < 40:
        return "fundamental quality is weak despite valuation output"
    return "fundamental signal is balanced and should be blended with factor evidence"

