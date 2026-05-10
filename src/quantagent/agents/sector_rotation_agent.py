from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from quantagent.domain.schemas import AgentSignal

MarketState = Literal["main_trend", "rotation", "diffusion", "exhaustion", "defensive", "crash"]


@dataclass(frozen=True)
class SectorRotationDecision:
    market_state: MarketState
    sector_signals: tuple[AgentSignal, ...]
    symbol_signals: tuple[AgentSignal, ...]


class SectorRotationAgent:
    def __init__(self, score_column: str = "sector_rotation_score") -> None:
        self.score_column = score_column

    def classify_market_state(self, sector_factors: pd.DataFrame) -> MarketState:
        return classify_market_state(sector_factors, self.score_column)

    def generate_signals(
        self,
        sector_factors: pd.DataFrame,
        symbol_sector_map: pd.Series | dict[str, str] | None = None,
        horizon_days: int = 10,
    ) -> SectorRotationDecision:
        data = _latest(sector_factors)
        state = self.classify_market_state(sector_factors)
        sector_signals: list[AgentSignal] = []
        for _, row in data.iterrows():
            score = _finite(row.get(self.score_column, 0.0))
            breadth = _finite(row.get("sector_breadth", 0.5))
            flow = _finite(row.get("sector_money_flow_share", 0.0))
            confidence = float(np.clip(0.35 + 0.25 * abs(score) + 0.25 * max(breadth, 0.0) + 0.15 * max(flow, 0.0), 0.0, 1.0))
            risk_penalty = float(np.clip(-score, 0.0, 1.0)) if state in {"exhaustion", "crash", "defensive"} else 0.0
            sector = str(row.get("sector"))
            sector_signals.append(
                AgentSignal(
                    agent_name="sector_rotation_agent",
                    symbol=f"SECTOR:{sector}",
                    horizon_days=horizon_days,
                    signal_strength=float(np.tanh(score)),
                    confidence=confidence,
                    evidence_quality=float(np.clip(0.5 + abs(flow), 0.0, 1.0)),
                    risk_penalty=risk_penalty,
                    tags=("sector_rotation", state, sector),
                )
            )

        symbol_signals: list[AgentSignal] = []
        if symbol_sector_map is not None:
            mapping = pd.Series(symbol_sector_map)
            sector_by_name = {str(signal.symbol).replace("SECTOR:", ""): signal for signal in sector_signals}
            for symbol, sector in mapping.items():
                sector_signal = sector_by_name.get(str(sector))
                if sector_signal is None:
                    continue
                symbol_signals.append(
                    AgentSignal(
                        agent_name="sector_rotation_agent",
                        symbol=str(symbol),
                        horizon_days=horizon_days,
                        signal_strength=sector_signal.signal_strength,
                        confidence=sector_signal.confidence * 0.85,
                        evidence_quality=sector_signal.evidence_quality,
                        risk_penalty=sector_signal.risk_penalty,
                        tags=("sector_rotation", state, str(sector)),
                    )
                )
        return SectorRotationDecision(state, tuple(sector_signals), tuple(symbol_signals))


def classify_market_state(sector_factors: pd.DataFrame, score_column: str = "sector_rotation_score") -> MarketState:
    data = _latest(sector_factors)
    if data.empty or score_column not in data.columns:
        return "defensive"
    score = data[score_column].replace([np.inf, -np.inf], np.nan)
    breadth = data["sector_breadth"].replace([np.inf, -np.inf], np.nan) if "sector_breadth" in data.columns else pd.Series(np.nan, index=data.index)
    flow = data["sector_money_flow_share"].replace([np.inf, -np.inf], np.nan) if "sector_money_flow_share" in data.columns else pd.Series(np.nan, index=data.index)
    mean_score = score.mean()
    score_std = score.std(ddof=0)
    positive_breadth = (breadth > 0.55).mean()
    if mean_score < -1.0 and positive_breadth < 0.25:
        return "crash"
    if mean_score < -0.25:
        return "defensive"
    if score.max() > 1.0 and positive_breadth < 0.35:
        return "exhaustion"
    if positive_breadth > 0.65 and flow.mean() > 0:
        return "diffusion"
    if score_std > 0.8:
        return "rotation"
    return "main_trend"


def sector_rotation_signals(
    sector_factors: pd.DataFrame,
    symbol_sector_map: pd.Series | dict[str, str] | None = None,
    horizon_days: int = 10,
) -> SectorRotationDecision:
    return SectorRotationAgent().generate_signals(sector_factors, symbol_sector_map, horizon_days)


def _latest(sector_factors: pd.DataFrame) -> pd.DataFrame:
    if sector_factors.empty:
        return sector_factors.copy()
    data = sector_factors.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    latest_date = data["trade_date"].max()
    return data.loc[data["trade_date"] == latest_date].copy()


def _finite(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if np.isfinite(numeric) else 0.0

