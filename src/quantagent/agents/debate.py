from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from quantagent.domain.schemas import AgentSignal, EvidenceItem


@dataclass(frozen=True)
class DebateRound:
    round_index: int
    role: str
    position: str
    confidence: float
    rationale: str
    rebuttal_to: int | None = None
    evidence: tuple[EvidenceItem, ...] = ()


@dataclass(frozen=True)
class DebateOutcome:
    symbol: str
    horizon_days: int
    final_position: str
    final_confidence: float
    rounds: tuple[DebateRound, ...]
    arbitrator: str
    reason: str


@dataclass
class DebateSession:
    symbol: str
    horizon_days: int
    rounds: list[DebateRound] = field(default_factory=list)

    def add(self, round_: DebateRound) -> None:
        self.rounds.append(round_)

    def outcome(self) -> DebateOutcome:
        if not self.rounds:
            return DebateOutcome(
                symbol=self.symbol,
                horizon_days=self.horizon_days,
                final_position="hold",
                final_confidence=0.0,
                rounds=(),
                arbitrator="empty",
                reason="no rounds recorded",
            )
        bull_conf = sum(r.confidence for r in self.rounds if r.position == "bull")
        bear_conf = sum(r.confidence for r in self.rounds if r.position == "bear")
        if bull_conf > bear_conf:
            position = "bull"
            confidence = bull_conf / max(bull_conf + bear_conf, 1e-6)
        elif bear_conf > bull_conf:
            position = "bear"
            confidence = bear_conf / max(bull_conf + bear_conf, 1e-6)
        else:
            position = "hold"
            confidence = 0.0
        return DebateOutcome(
            symbol=self.symbol,
            horizon_days=self.horizon_days,
            final_position=position,
            final_confidence=float(confidence),
            rounds=tuple(self.rounds),
            arbitrator="confidence_sum",
            reason=f"bull={bull_conf:.3f} bear={bear_conf:.3f}",
        )

    def to_signal(self, agent_name: str = "debate_arbitrator") -> AgentSignal:
        outcome = self.outcome()
        sign = 1.0 if outcome.final_position == "bull" else -1.0 if outcome.final_position == "bear" else 0.0
        evidence = tuple(item for r in outcome.rounds for item in r.evidence)
        return AgentSignal(
            agent_name=agent_name,
            symbol=outcome.symbol,
            horizon_days=outcome.horizon_days,
            signal_strength=sign * outcome.final_confidence,
            confidence=outcome.final_confidence,
            evidence_quality=min(1.0, 0.4 + 0.1 * len(outcome.rounds)),
            risk_penalty=0.0,
            evidence=evidence,
            tags=("debate", outcome.final_position),
        )


def persist_debate(outcome: DebateOutcome, root: Path, trade_date: str) -> Path:
    """Append-only audit log: data/decisions/yyyymmdd/symbol.jsonl."""
    target = root / trade_date / f"{outcome.symbol}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(outcome)
    payload["recorded_at"] = datetime.utcnow().isoformat()
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return target
