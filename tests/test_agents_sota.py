import numpy as np
import pandas as pd

from quantagent.agents.bl_views import BLViewConfig, posterior_alpha_from_agents
from quantagent.agents.commodity_agent import commodity_shock_signals
from quantagent.agents.debate import DebateRound, DebateSession
from quantagent.agents.flow_agent import dragon_tiger_signals, northbound_flow_signals
from quantagent.agents.policy_agent import PolicyEvent, policy_signals
from quantagent.domain.schemas import AgentSignal
from quantagent.fundamental.scores import (
    AltmanInputs,
    BeneishInputs,
    PiotroskiInputs,
    altman_z_score,
    beneish_m_score,
    piotroski_f_score,
)


def test_bl_views_shift_alpha_in_view_direction():
    universe = pd.Index(["A", "B", "C"])
    prior = pd.Series([0.0, 0.0, 0.0], index=universe)
    cov = pd.DataFrame(np.eye(3) * 0.04, index=universe, columns=universe)
    signals = [AgentSignal("technical", "B", 5, 0.8, 0.9, 0.8)]
    posterior = posterior_alpha_from_agents(
        prior,
        cov,
        signals,
        expected_volatility=pd.Series([0.2, 0.2, 0.2], index=universe),
        config=BLViewConfig(tau=0.05),
    )
    assert posterior.loc["B"] > posterior.loc["A"]


def test_debate_session_picks_majority_position():
    session = DebateSession(symbol="000300.SH", horizon_days=5)
    session.add(DebateRound(0, "bull_analyst", "bull", 0.7, "macro tailwind"))
    session.add(DebateRound(1, "bear_analyst", "bear", 0.4, "valuation high", rebuttal_to=0))
    outcome = session.outcome()
    assert outcome.final_position == "bull"
    signal = session.to_signal()
    assert signal.signal_strength > 0


def test_piotroski_f_score_accumulates_points():
    f = piotroski_f_score(
        PiotroskiInputs(
            net_income=100, operating_cash_flow=120,
            roa=0.1, roa_prev=0.08,
            leverage=0.4, leverage_prev=0.5,
            current_ratio=2.0, current_ratio_prev=1.8,
            shares_outstanding=1e8, shares_outstanding_prev=1e8,
            gross_margin=0.5, gross_margin_prev=0.45,
            asset_turnover=1.1, asset_turnover_prev=1.0,
        )
    )
    assert f == 9


def test_altman_z_score_distress_threshold():
    z = altman_z_score(
        AltmanInputs(
            working_capital=-50, retained_earnings=-100,
            ebit=-10, market_cap=20,
            total_liabilities=200, sales=300, total_assets=500,
        )
    )
    assert z < 1.81


def test_beneish_m_score_returns_finite():
    m = beneish_m_score(
        BeneishInputs(
            receivables_curr=100, receivables_prev=80,
            sales_curr=1200, sales_prev=1000,
            cogs_curr=600, cogs_prev=520,
            current_assets_curr=300, current_assets_prev=280,
            ppe_curr=400, ppe_prev=380,
            total_assets_curr=1000, total_assets_prev=950,
            depreciation_curr=20, depreciation_prev=18,
            sga_curr=120, sga_prev=100,
            leverage_curr=0.5, leverage_prev=0.45,
            net_income_curr=80, operating_cash_flow_curr=60,
        )
    )
    assert np.isfinite(m)


def test_policy_signals_emit_for_mapped_sector():
    sector_map = pd.Series({"600519.SH": "food_beverage", "688981.SH": "semi"})
    events = [PolicyEvent(
        published_at="2026-05-01",
        headline="boost",
        sectors=("food_beverage",),
        polarity=0.5,
    )]
    signals = policy_signals(events, sector_map, reference_date=pd.Timestamp("2026-05-09"))
    assert any(s.symbol == "600519.SH" for s in signals)


def test_commodity_shock_emits_signals():
    sector_map = pd.Series({"601857.SH": "oil_gas"})
    moves = pd.Series({"crude_oil": 0.05})
    signals = commodity_shock_signals(moves, sector_map)
    assert len(signals) > 0


def test_northbound_flow_signal_filters_low_z():
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range("2026-01-01", periods=30):
        rows.append({"symbol": "600519.SH", "trade_date": d, "holding_value": 1e9 + rng.normal(scale=1e6)})
    rows.append({"symbol": "600519.SH", "trade_date": pd.Timestamp("2026-02-01"), "holding_value": 5e9})
    flow = pd.DataFrame(rows)
    signals = northbound_flow_signals(flow, z_threshold=1.5, window=20)
    assert len(signals) == 1
    assert signals[0].signal_strength > 0


def test_dragon_tiger_signal_imbalance():
    frame = pd.DataFrame(
        [
            {"symbol": "300750.SZ", "inst_buy": 1e8, "inst_sell": 2e7, "retail_buy": 5e7, "retail_sell": 8e7},
        ]
    )
    signals = dragon_tiger_signals(frame)
    assert signals[0].signal_strength > 0
