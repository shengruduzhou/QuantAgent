import pandas as pd

from quantagent.agents.agent_reliability import AgentReliability
from quantagent.agents.sentiment_agent import SentimentAgent


def test_agent_reliability_rises_with_correct_calls():
    tracker = AgentReliability(halflife=5)
    base = tracker.score("policy_agent")
    for _ in range(20):
        tracker.update("policy_agent", predicted_direction=1.0, realized_return=0.02)
    new_score = tracker.score("policy_agent")
    assert new_score > base


def test_agent_reliability_falls_with_wrong_calls():
    tracker = AgentReliability(halflife=5)
    base = tracker.score("flow_agent")
    for _ in range(20):
        tracker.update("flow_agent", predicted_direction=1.0, realized_return=-0.03)
    new_score = tracker.score("flow_agent")
    assert new_score < base


def test_sentiment_agent_separates_positive_and_negative_text():
    frame = pd.DataFrame(
        {
            "symbol": ["600000.SH", "600001.SH"],
            "timestamp": ["2026-05-11", "2026-05-11"],
            "text": [
                "公司业绩大幅增长，新签订单超预期，盈利改善显著。",
                "公司业绩下滑，遭立案调查并被处罚，存在退市风险。",
            ],
        }
    )
    agent = SentimentAgent()
    records = agent.run(frame)
    assert len(records) == 2
    pos = next(r for r in records if r.symbol == "600000.SH")
    neg = next(r for r in records if r.symbol == "600001.SH")
    assert pos.direction > 0
    assert neg.direction < 0
