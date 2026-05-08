from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.domain.schemas import AgentSignal, EvidenceItem


@dataclass(frozen=True)
class PolicyEvent:
    published_at: str
    headline: str
    sectors: tuple[str, ...]
    polarity: float
    source: str = "policy_release"


def policy_signals(
    events: list[PolicyEvent],
    sector_map: pd.Series,
    horizon_days: int = 20,
    decay_lambda: float = 0.05,
    reference_date: pd.Timestamp | None = None,
) -> list[AgentSignal]:
    """Map decay-weighted policy events to per-symbol AgentSignals via sector map."""
    if not events:
        return []
    ref = reference_date or pd.Timestamp.utcnow().normalize()
    signals: list[AgentSignal] = []
    bucket: dict[tuple[str, str], list[tuple[float, PolicyEvent]]] = {}
    for event in events:
        days = max(0.0, (ref - pd.Timestamp(event.published_at)).days)
        weight = float(np.exp(-decay_lambda * days))
        for sector in event.sectors:
            symbols_in = sector_map[sector_map == sector].index
            for sym in symbols_in:
                bucket.setdefault((sym, sector), []).append((weight, event))
    for (sym, sector), items in bucket.items():
        weights = np.array([w for w, _ in items])
        polarities = np.array([e.polarity for _, e in items])
        score = float(np.tanh(np.dot(weights, polarities)))
        evidence = tuple(
            EvidenceItem(
                source=e.source,
                title=e.headline,
                published_at=e.published_at,
            )
            for _, e in items[:5]
        )
        signals.append(
            AgentSignal(
                agent_name="policy_agent",
                symbol=sym,
                horizon_days=horizon_days,
                signal_strength=score,
                confidence=min(1.0, 0.3 + 0.1 * len(items)),
                evidence_quality=0.6,
                risk_penalty=max(0.0, -score) * 0.2,
                evidence=evidence,
                tags=("policy", sector),
            )
        )
    return signals
