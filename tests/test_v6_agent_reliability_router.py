import pandas as pd

from quantagent.agents.agent_reliability import AgentReliability
from quantagent.agents.agent_router import AgentRouter
from quantagent.agents.views_schema import EvidenceRecord


def test_agent_reliability_changes_q_and_omega():
    evidence = [
        EvidenceRecord(
            source="policy_agent",
            timestamp="2026-01-02",
            symbol="600000.SH",
            event_type="policy",
            direction=1.0,
            magnitude=1.0,
            confidence=0.8,
        )
    ]
    low = AgentReliability(initial_score=0.2)
    high = AgentReliability(initial_score=1.2)
    low_view = AgentRouter(reliability=low).route(evidence, pd.Index(["600000.SH"])).views[0]
    high_view = AgentRouter(reliability=high).route(evidence, pd.Index(["600000.SH"])).views[0]
    assert high_view.q > low_view.q
    assert high_view.omega < low_view.omega
    assert high_view.constraints["reliability"] == 1.2

