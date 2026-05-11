from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1

import numpy as np
import pandas as pd

from quantagent.agents.agent_reliability import AgentReliability
from quantagent.agents.views_schema import AgentView, EvidenceRecord


@dataclass(frozen=True)
class RoutedAgentOutput:
    views: list[AgentView]
    constraints: list[dict[str, object]]
    risk_warnings: list[dict[str, object]]
    no_trade_flags: list[dict[str, object]]


class AgentRouter:
    """Map structured evidence into BL views, constraints, and risk flags."""

    def __init__(
        self,
        base_view_scale: float = 0.03,
        min_omega: float = 1e-6,
        reliability: AgentReliability | None = None,
        base_omega: float | None = None,
    ) -> None:
        self.base_view_scale = base_view_scale
        self.min_omega = min_omega
        self.reliability = reliability or AgentReliability()
        self.base_omega = base_omega if base_omega is not None else base_view_scale**2

    def route(self, evidence: list[EvidenceRecord], universe: pd.Index | list[str]) -> RoutedAgentOutput:
        universe_index = pd.Index(universe)
        views: list[AgentView] = []
        constraints: list[dict[str, object]] = []
        risk_warnings: list[dict[str, object]] = []
        no_trade_flags: list[dict[str, object]] = []
        for record in evidence:
            symbols = self._resolve_symbols(record, universe_index)
            if not symbols:
                continue
            direction = float(np.sign(record.direction)) if record.direction != 0 else 0.0
            confidence = float(np.clip(record.confidence, 0.0, 1.0))
            if record.event_type in {"risk", "regulatory_penalty", "fraud"} or record.direction < -0.8:
                risk_warnings.append({"symbol": record.symbol, "sector": record.sector, "source": record.source, "confidence": confidence})
            if record.event_type in {"halt", "suspension", "manual_no_trade"}:
                no_trade_flags.append({"symbols": tuple(symbols), "reason": record.event_type, "source": record.source})
                continue
            if direction == 0.0:
                constraints.append({"symbols": tuple(symbols), "type": "neutral_view", "source": record.source})
                continue
            exposure = {symbol: direction / len(symbols) for symbol in symbols}
            reliability = float(self.reliability.score(record.source))
            q = direction * abs(record.magnitude) * confidence * self.base_view_scale * reliability
            omega = max(self.base_omega * (1.0 - confidence + 1e-3) / max(reliability, 0.1), self.min_omega)
            view = AgentView(
                view_id=self._view_id(record, symbols),
                symbols=tuple(symbols),
                exposure=exposure,
                q=float(q),
                omega=float(omega),
                confidence=confidence,
                constraints={"reliability": reliability},
                expires_at=None,
                evidence=(record,),
            )
            views.append(view)
        return RoutedAgentOutput(views=views, constraints=constraints, risk_warnings=risk_warnings, no_trade_flags=no_trade_flags)

    def _resolve_symbols(self, record: EvidenceRecord, universe: pd.Index) -> list[str]:
        if record.symbol and record.symbol in universe:
            return [record.symbol]
        if record.sector is None:
            return []
        sector_members = [symbol for symbol in universe if str(symbol).startswith(record.sector)]
        return sector_members

    def _view_id(self, record: EvidenceRecord, symbols: list[str]) -> str:
        raw = "|".join([record.source, record.timestamp, ",".join(symbols), record.event_type, str(record.direction)])
        return sha1(raw.encode("utf-8")).hexdigest()[:16]
