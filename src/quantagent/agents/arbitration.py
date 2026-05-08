from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.domain.schemas import AgentSignal


def agent_reliability_weights(
    agent_stats: pd.DataFrame,
    agent_column: str = "agent_name",
    ir_column: str = "ir",
    evidence_quality_column: str = "evidence_quality",
    error_column: str = "error",
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> pd.Series:
    """Softmax weights based on historical IR, evidence quality, and error."""
    required = {agent_column, ir_column, evidence_quality_column, error_column}
    missing = required.difference(agent_stats.columns)
    if missing:
        raise ValueError(f"Missing agent stats columns: {sorted(missing)}")
    score = (
        alpha * agent_stats[ir_column].fillna(0.0).to_numpy()
        + beta * agent_stats[evidence_quality_column].fillna(0.0).to_numpy()
        - gamma * agent_stats[error_column].fillna(0.0).to_numpy()
    )
    stable = score - np.max(score)
    raw = np.exp(stable)
    names = agent_stats[agent_column].to_numpy()
    if raw.sum() <= 0:
        return pd.Series(np.full(len(names), 1.0 / len(names)), index=names)
    return pd.Series(raw / raw.sum(), index=names)


def aggregate_agent_signals(
    signals: list[AgentSignal],
    agent_weights: pd.Series | None = None,
) -> pd.Series:
    """Aggregate structured Agent signals into per-symbol scores."""
    if not signals:
        return pd.Series(dtype=float)
    rows = []
    for signal in signals:
        agent_weight = 1.0 if agent_weights is None else float(agent_weights.get(signal.agent_name, 0.0))
        rows.append(
            {
                "symbol": signal.symbol,
                "score": agent_weight
                * signal.signal_strength
                * signal.confidence
                * signal.evidence_quality
                - signal.risk_penalty,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.groupby("symbol")["score"].sum()
