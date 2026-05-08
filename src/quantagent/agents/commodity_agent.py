from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.domain.schemas import AgentSignal


COMMODITY_SECTOR_BETA = {
    "crude_oil": {"oil_gas": 0.8, "petrochem": 0.5, "transport": -0.3, "airline": -0.6},
    "copper": {"non_ferrous_metals": 0.9, "power_grid": 0.4, "construction": 0.3},
    "rebar": {"steel": 0.9, "real_estate": 0.5, "machinery": 0.3},
    "thermal_coal": {"coal": 0.9, "power": -0.4, "cement": -0.2},
    "soybean": {"food_beverage": 0.4, "agri_processing": 0.7, "livestock": -0.3},
}


def commodity_shock_signals(
    commodity_returns: pd.Series,
    sector_map: pd.Series,
    horizon_days: int = 10,
    threshold: float = 0.02,
) -> list[AgentSignal]:
    """Generate sector-level signals from significant commodity moves."""
    signals: list[AgentSignal] = []
    for commodity, ret in commodity_returns.items():
        if abs(ret) < threshold or commodity not in COMMODITY_SECTOR_BETA:
            continue
        beta_map = COMMODITY_SECTOR_BETA[commodity]
        for sector, beta in beta_map.items():
            symbols_in = sector_map[sector_map == sector].index
            strength = float(np.tanh(beta * ret * 20.0))
            for sym in symbols_in:
                signals.append(
                    AgentSignal(
                        agent_name=f"commodity_agent::{commodity}",
                        symbol=sym,
                        horizon_days=horizon_days,
                        signal_strength=strength,
                        confidence=min(1.0, abs(ret) / max(threshold, 1e-6)),
                        evidence_quality=0.5,
                        risk_penalty=max(0.0, -strength) * 0.3,
                        tags=("commodity", commodity, sector),
                    )
                )
    return signals
