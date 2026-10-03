"""R10-F05: dividends were credited gross of the A-share holding-period tax."""

from __future__ import annotations

import tempfile

import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    simulate_ashare_target_weights,
)
from quantagent.backtest.dividend_tax import DividendTaxLots, holding_period_rate
from quantagent.data.ashare import execution_panel as ep


def test_rates_follow_the_holding_period():
    bought = pd.Timestamp("2024-01-15")
    assert holding_period_rate(bought, "2024-02-15") == 0.20   # 含1个月
    assert holding_period_rate(bought, "2024-02-16") == 0.10
    assert holding_period_rate(bought, "2025-01-15") == 0.10   # 含1年
    assert holding_period_rate(bought, "2025-01-16") == 0.0


def test_fifo_lots_and_bonus_shares_inherit_the_acquisition_date():
    lots = DividendTaxLots()
    lots.buy("X", "2024-01-02", 1000)
    lots.buy("X", "2024-06-03", 1000)
    lots.dividend("X", 0.5)                       # 1,000 CNY gross on 2,000 shares
    lots.bonus("X", 1.0)                          # 2,000 -> 4,000 shares
    # Sell the first (old) lot's 2,000 shares after 13 months: exempt.
    assert lots.sell("X", "2025-02-10", 2000) == pytest.approx(0.0)
    # The June lot (2,000 shares, 500 CNY accrued) sold within a year: 10%.
    assert lots.sell("X", "2025-02-10", 2000) == pytest.approx(50.0)
    assert lots.dividends_gross == pytest.approx(1000.0)


def test_latent_tax_is_reported_for_open_lots():
    lots = DividendTaxLots()
    lots.buy("X", "2024-01-02", 1000)
    lots.dividend("X", 1.0)
    assert lots.latent_tax("2024-01-20") == pytest.approx(200.0)


def test_simulator_charges_the_short_holding_tax_at_sale():
    sessions = pd.bdate_range("2024-03-04", periods=6)
    closes = [10.0, 10.0, 9.5, 9.5, 9.5, 9.5]
    bars = pd.DataFrame({
        "symbol": "600000.SH", "trade_date": sessions, "open": closes, "high": closes,
        "low": closes, "close": closes, "volume": 1e7, "amount": [c * 1e7 for c in closes],
        "available_at": sessions + pd.Timedelta(hours=15), "serving_provider": "tickflow",
        "mask_is_suspended": "FALSE", "mask_is_st": "FALSE", "mask_limit_up": "FALSE",
        "mask_limit_down": "FALSE",
    })
    factors = pd.DataFrame([
        {"symbol": "600000.SH", "effective_date": pd.Timestamp("2010-01-01"), "hfq_factor": 1.0},
        {"symbol": "600000.SH", "effective_date": sessions[2], "hfq_factor": 10.0 / 9.5},
    ])
    ca = pd.DataFrame([{"symbol": "600000.SH", "ex_date": sessions[2],
                        "cash_dividend_per_share": 0.5, "stock_dividend_ratio": 0.0,
                        "rights_ratio": 0.0}])
    panel, _ = ep.build_execution_panel(bars, session_gaps=None, factors=factors,
                                        corporate_actions=ca)
    weights = pd.DataFrame({"600000.SH": [0.5, 0.0]}, index=sessions[[0, 3]])
    with tempfile.TemporaryDirectory() as folder:
        result = simulate_ashare_target_weights(
            weights, panel, AShareExecutionSimulationConfig(slippage_bps=0, audit_log_dir=folder))
    audit = result.corporate_action_audit
    shares = int(audit.loc[audit["basis"] == "corporate_action", "shares_before"].iloc[0])
    tax_rows = audit[audit["basis"] == "dividend_tax"]
    assert len(tax_rows) == 1
    assert -tax_rows["cash_credit"].iloc[0] == pytest.approx(shares * 0.5 * 0.20)
    disclosed = result.config["dividend_tax"]
    assert disclosed["tax_paid_at_sale_cny"] == pytest.approx(shares * 0.5 * 0.20, abs=0.01)
    assert disclosed["latent_tax_on_open_lots_cny"] == pytest.approx(0.0)
