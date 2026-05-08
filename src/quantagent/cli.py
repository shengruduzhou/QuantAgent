from __future__ import annotations

import typer

from quantagent.domain.schemas import ModelScores
from quantagent.strategy.decision_engine import decide_trade

app = typer.Typer(help="QuantAgent research and decision CLI.")


@app.command()
def demo_decision(
    ticker: str = "NVDA",
    short_score: float = 82.0,
    long_score: float = 86.0,
    news_score: float = 70.0,
    llm_score: float = 68.0,
    risk_score: float = 32.0,
    confidence: float = 0.72,
) -> None:
    """Run the deterministic decision layer with normalized scores."""
    decision = decide_trade(
        ModelScores(
            ticker=ticker,
            short_score=short_score,
            long_score=long_score,
            news_score=news_score,
            llm_score=llm_score,
            risk_score=risk_score,
            confidence=confidence,
        )
    )
    typer.echo(decision)


if __name__ == "__main__":
    app()
