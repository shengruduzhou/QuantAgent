import pandas as pd

from quantagent.agents.financial_statement_agent import FinancialStatementAgent
from quantagent.fundamental.target_price import TargetPriceEstimate, final_target_price_band
from quantagent.fundamental.valuation import DCFInputs


def _dcf() -> DCFInputs:
    return DCFInputs(
        fcff=1_000_000_000,
        growth_rate=0.06,
        terminal_growth_rate=0.025,
        wacc=0.09,
        years=5,
        net_debt=500_000_000,
        shares_outstanding=100_000_000,
    )


def test_target_price_returns_bear_base_bull_band():
    estimate = final_target_price_band("A", 50.0, _dcf(), fraud_risk=0.2, quality_score=70)
    assert isinstance(estimate, TargetPriceEstimate)
    assert estimate.bear_price < estimate.base_price < estimate.bull_price
    assert estimate.current_price == 50.0


def test_financial_statement_agent_outputs_signal_not_order():
    statements = pd.DataFrame(
        {
            "symbol": ["A"] * 5,
            "report_date": pd.date_range("2025-01-01", periods=5, freq="QE"),
            "revenue": [100, 110, 120, 130, 145],
            "cogs": [50, 54, 58, 62, 68],
            "receivables": [10, 11, 12, 13, 14],
            "inventory": [8, 8.5, 9, 9.2, 9.5],
            "net_income": [10, 11, 12, 13, 15],
            "operating_cash_flow": [12, 12, 13, 14, 16],
            "total_assets": [200, 205, 210, 220, 230],
            "capex": [-5, -5, -6, -6, -7],
            "roe": [0.10, 0.11, 0.12, 0.13, 0.14],
            "roic": [0.09, 0.10, 0.11, 0.12, 0.13],
            "gross_margin": [0.50, 0.51, 0.52, 0.53, 0.53],
        }
    )
    output = FinancialStatementAgent().run("A", statements, 50.0, _dcf())
    assert output.signal.symbol == "A"
    assert "fundamental" in output.signal.tags
