from __future__ import annotations

import pandas as pd

from quantagent.agents.views_schema import AgentView


def reliability_weighted_posterior(
    prior_alpha: pd.Series,
    views: list[AgentView],
    max_view_impact: float = 0.03,
) -> pd.Series:
    """Apply a compact BL-like posterior update from routed agent views."""
    posterior = prior_alpha.astype(float).copy()
    if not views:
        return posterior
    for view in views:
        reliability = float(view.constraints.get("reliability", 1.0))
        precision = 1.0 / max(float(view.omega), 1e-9)
        total_abs = sum(abs(v) for v in view.exposure.values()) or 1.0
        for symbol, exposure in view.exposure.items():
            if symbol not in posterior.index:
                continue
            impact = float(view.q) * float(exposure) / total_abs
            scaled = max(-max_view_impact, min(max_view_impact, impact * reliability * precision / (precision + 1_000.0)))
            posterior.loc[symbol] += scaled
    return posterior

