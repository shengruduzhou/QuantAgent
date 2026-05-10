import pandas as pd

from quantagent.agents.sector_rotation_agent import SectorRotationAgent
from quantagent.factors.sector_rotation import compute_sector_rotation_factors


def _sector_panel() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=30)
    rows = []
    for symbol, sector, drift in [
        ("A", "tech", 0.02),
        ("B", "tech", 0.015),
        ("C", "bank", 0.001),
        ("D", "bank", -0.001),
    ]:
        for i, date in enumerate(dates):
            close = 10 + i * drift + (1 if sector == "tech" else 0)
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "sector": sector,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000,
                    "amount": close * 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_sector_rotation_factors_and_agent_emit_signals():
    factors = compute_sector_rotation_factors(_sector_panel(), window=5)
    assert "sector_rotation_score" in factors.columns
    decision = SectorRotationAgent().generate_signals(factors, {"A": "tech", "C": "bank"})
    assert decision.market_state in {"main_trend", "rotation", "diffusion", "exhaustion", "defensive", "crash"}
    assert decision.sector_signals
    assert decision.symbol_signals

