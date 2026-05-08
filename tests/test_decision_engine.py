from quantagent.domain.schemas import ModelScores, TradeAction
from quantagent.strategy.decision_engine import decide_trade


def test_strong_signal_allows_buy_with_nonzero_weight():
    decision = decide_trade(
        ModelScores(
            ticker="NVDA",
            short_score=82,
            long_score=88,
            news_score=70,
            llm_score=65,
            risk_score=30,
            confidence=0.8,
        )
    )

    assert decision.action == TradeAction.BUY
    assert decision.target_weight > 0


def test_high_risk_forces_reduction():
    decision = decide_trade(
        ModelScores(
            ticker="AAPL",
            short_score=90,
            long_score=90,
            risk_score=75,
            confidence=0.9,
        )
    )

    assert decision.action == TradeAction.REDUCE


def test_low_long_score_exits_holding():
    decision = decide_trade(
        ModelScores(
            ticker="XYZ",
            short_score=95,
            long_score=40,
            risk_score=20,
            confidence=0.9,
        )
    )

    assert decision.action == TradeAction.EXIT
    assert decision.target_weight == 0
