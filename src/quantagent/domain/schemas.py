from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TradeAction(str, Enum):
    BUY = "buy"
    HOLD = "hold"
    REDUCE = "reduce"
    EXIT = "exit"
    BLOCK = "block"


@dataclass(frozen=True)
class ModelScores:
    """Normalized model scores on a 0-100 scale."""

    ticker: str
    short_score: float
    long_score: float
    news_score: float = 50.0
    llm_score: float = 50.0
    risk_score: float = 50.0
    confidence: float = 0.5


@dataclass(frozen=True)
class AlphaPrediction:
    """Probabilistic model output used by statistical validation and optimization."""

    symbol: str
    horizon: str
    expected_return: float
    expected_excess_return: float
    volatility_forecast: float
    downside_risk: float
    confidence: float
    rank_score: float
    regime_adjusted_score: float | None = None


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    title: str
    url: str | None = None
    published_at: str | None = None
    excerpt: str | None = None


@dataclass(frozen=True)
class AgentSignal:
    """Structured Agent output. Agents never emit orders."""

    agent_name: str
    symbol: str
    horizon_days: int
    signal_strength: float
    confidence: float
    evidence_quality: float
    risk_penalty: float = 0.0
    evidence: tuple[EvidenceItem, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetWeight:
    """Weight-centric interface shared by research, backtest, paper, and live execution."""

    symbol: str
    target_weight: float
    horizon_days: int
    confidence: float
    source: str
    reason: str = ""


@dataclass(frozen=True)
class RiskLimits:
    allow_buy_min_long_score: float = 75.0
    allow_buy_min_short_score: float = 65.0
    max_risk_score_for_buy: float = 40.0
    force_reduce_risk_score: float = 70.0
    block_long_score_below: float = 50.0


@dataclass(frozen=True)
class SignalDecision:
    ticker: str
    action: TradeAction
    final_score: float
    target_weight: float
    reason: str
