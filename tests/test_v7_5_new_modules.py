"""Smoke tests for the v7.6 additions:

- Dynamic industry-chain reasoner derives nodes/edges from evidence (no template)
- Long-horizon factor library produces 22+ factor columns per symbol
- Intrinsic valuation engine returns DCF/relative/asset components and a haircut confidence
- Economic analyzer scores macro, industries, and per-company economics
- Retail/HFT risk module penalises elevated structural risk
- Deep alpha model emits MultiHorizonAlpha for every universe member
- Long-short allocator returns a sleeve dict with risk-off shifting to cash
- Pipeline end-to-end still produces expected new keys
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
from quantagent.factors.long_horizon_factors import (
    LONG_HORIZON_FACTORS,
    compute_long_horizon_factors,
    long_horizon_alpha_score,
)
from quantagent.fundamental.economic_analyzer import (
    analyze_industries,
    analyze_macro,
    industry_snapshots_to_company_frame,
)
from quantagent.fundamental.intrinsic_valuation import value_universe
from quantagent.models.v7_deep_alpha import V7DeepAlphaConfig, predict_v7_deep_alpha
from quantagent.portfolio.hedge_decision_engine import decide_v7_hedge
from quantagent.portfolio.strategic_tactical_allocator import construct_v7_portfolio
from quantagent.risk.retail_hft_risk import RetailHFTRiskConfig, score_retail_hft_risk
from quantagent.services.v7_pipeline_service import run_daily_v7_research
from quantagent.strategy.long_short_allocator import allocate_long_short
from quantagent.themes.industry_chain_reasoner import (
    IndustryChainReasonerConfig,
    reason_industry_chain,
)
from quantagent.themes.policy_crawler import local_policy_documents
from quantagent.themes.policy_parser import parse_policy_document
from quantagent.themes.theme_extractor import discover_themes
from quantagent.themes.theme_universe_builder import build_thematic_universe
from quantagent.v7.schemas import (
    ChainRelationType,
    MarketRegime,
    MarketRegimeSnapshot,
    SleeveType,
    ThematicUniverseMember,
    UniverseBucket,
)


@pytest.fixture()
def synthetic_evidence_and_theme():
    docs = local_policy_documents(
        [
            {
                "document_id": "p_ai_compute",
                "title": "AI compute infrastructure plan",
                "body": "Support GPU, AI server, optical module, PCB, advanced packaging, "
                "HBM, liquid cooling, data center and power equipment. "
                "国产替代 in domestic GPU and semiconductor equipment is a 卡脖子 priority. "
                "Critical bottleneck on advanced packaging.",
                "source": "ministry",
                "source_level": "ministry",
                "published_at": "2026-05-14",
            }
        ]
    )
    parsed = [parse_policy_document(doc) for doc in docs]
    themes, evidence = discover_themes(parsed, "2026-05-14")
    return themes[0], evidence


def test_industry_chain_reasoner_uses_evidence_not_template(synthetic_evidence_and_theme):
    theme, evidence = synthetic_evidence_and_theme
    config = IndustryChainReasonerConfig(use_llm_refinement=False)
    result = reason_industry_chain(theme, evidence, config, LLMSkillClient(LLMSkillConfig()))
    assert result.theme == theme.theme_name
    assert result.nodes, "expected at least one node inferred from evidence"
    assert result.chain_confidence > 0.0
    relation_types = {edge.relation_type for edge in result.edges}
    assert relation_types.issubset(set(ChainRelationType)), "all edges have a known relation type"


def test_long_horizon_factors_returns_full_factor_set():
    fundamentals = pd.DataFrame(
        [
            {
                "symbol": "600001.SH", "report_date": "2025-09-30", "industry": "server",
                "revenue": 100.0, "net_income": 8.0, "roe": 0.10, "roa": 0.05,
                "gross_margin": 0.22, "operating_cash_flow": 12.0, "capex": -8.0,
                "total_assets": 200.0, "debt_to_asset": 0.40, "pe_ttm": 32.0, "pb": 3.0,
                "ps_ttm": 2.4, "ev_ebitda": 17.0, "peg": 1.0,
                "order_visibility_score": 70.0, "capacity_release_score": 65.0,
                "fraud_risk_score": 40.0, "margin_of_safety": 0.20,
                "valuation_bubble_score": 35.0,
            },
            {
                "symbol": "600001.SH", "report_date": "2025-12-31", "industry": "server",
                "revenue": 115.0, "net_income": 10.0, "roe": 0.11, "roa": 0.06,
                "gross_margin": 0.23, "operating_cash_flow": 14.0, "capex": -9.0,
                "total_assets": 210.0, "debt_to_asset": 0.39, "pe_ttm": 30.0, "pb": 2.9,
                "ps_ttm": 2.3, "ev_ebitda": 16.0, "peg": 0.9,
                "order_visibility_score": 73.0, "capacity_release_score": 68.0,
                "fraud_risk_score": 38.0, "margin_of_safety": 0.22,
                "valuation_bubble_score": 32.0,
            },
            {
                "symbol": "600001.SH", "report_date": "2026-03-31", "industry": "server",
                "revenue": 125.0, "net_income": 12.0, "roe": 0.12, "roa": 0.07,
                "gross_margin": 0.24, "operating_cash_flow": 16.0, "capex": -10.0,
                "total_assets": 220.0, "debt_to_asset": 0.38, "pe_ttm": 28.0, "pb": 2.8,
                "ps_ttm": 2.2, "ev_ebitda": 15.0, "peg": 0.8,
                "order_visibility_score": 75.0, "capacity_release_score": 70.0,
                "fraud_risk_score": 36.0, "margin_of_safety": 0.25,
                "valuation_bubble_score": 30.0,
            },
        ]
    )
    market_state = pd.DataFrame(
        [{"symbol": "600001.SH", "trade_date": "2026-05-14", "market_cap": 1.0e10,
          "sector_rotation_score": 60.0}]
    )
    price_panel = pd.DataFrame(
        [
            {"trade_date": f"2026-{(month):02d}-{day:02d}", "symbol": "600001.SH",
             "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0 + 0.05 * day,
             "volume": 1_000_000, "amount": 1.0e7 + day * 1e5}
            for month in (3, 4, 5) for day in (1, 5, 10, 15, 20, 25)
        ]
    )
    frame = compute_long_horizon_factors(fundamentals, market_state, price_panel)
    assert not frame.empty
    missing = [factor for factor in LONG_HORIZON_FACTORS if factor not in frame.columns]
    assert not missing, f"missing long-horizon factors: {missing}"
    alpha_frame = long_horizon_alpha_score(frame)
    assert not alpha_frame.empty
    assert {"symbol", "long_horizon_alpha_120d", "long_horizon_confidence"}.issubset(alpha_frame.columns)


def test_intrinsic_valuation_reports_fair_value_and_haircut_confidence():
    fundamentals = pd.DataFrame(
        [
            {
                "symbol": "600001.SH", "report_date": "2026-03-31", "industry": "server",
                "revenue": 120.0, "net_income": 12.0, "operating_cash_flow": 18.0,
                "capex": -8.0, "total_shares": 1.0e9, "price": 12.0, "market_cap": 1.2e10,
                "revenue_growth": 0.18, "profit_growth": 0.16, "pe_ttm": 28.0,
                "pb": 2.5, "ps_ttm": 2.0, "ev_ebitda": 15.0,
                "eps": 0.012, "book_value_per_share": 4.8, "revenue_per_share": 0.12,
                "fraud_risk_score": 35.0, "wacc": 0.10,
            },
            {
                "symbol": "999999.SH", "report_date": "2026-03-31", "industry": "server",
                "revenue": 100.0, "net_income": -5.0, "operating_cash_flow": -2.0,
                "capex": -3.0, "total_shares": 5.0e8, "price": 20.0, "market_cap": 1.0e10,
                "revenue_growth": -0.05, "profit_growth": -0.20, "pe_ttm": 80.0,
                "pb": 5.0, "ps_ttm": 6.0, "ev_ebitda": 40.0,
                "eps": -0.01, "book_value_per_share": 4.0, "revenue_per_share": 0.2,
                "fraud_risk_score": 75.0, "wacc": 0.12, "audit_opinion": "qualified",
            },
        ]
    )
    market_state = pd.DataFrame(
        [
            {"symbol": "600001.SH", "trade_date": "2026-05-14", "close": 12.0, "amount": 5e8},
            {"symbol": "999999.SH", "trade_date": "2026-05-14", "close": 20.0, "amount": 5e8},
        ]
    )
    reports = value_universe(fundamentals, market_state, "2026-05-14")
    assert len(reports) == 2
    by_symbol = {report.symbol: report for report in reports}
    healthy = by_symbol["600001.SH"]
    risky = by_symbol["999999.SH"]
    assert healthy.confidence > risky.confidence, "fraud-flagged company must have lower confidence"
    assert "non_standard_audit_opinion" in risky.flags or "valuation_bubble" in risky.flags
    assert healthy.method_weights, "healthy company should produce at least one valuation method"


def test_economic_analyzer_produces_macro_industry_and_company_panels():
    macro_indicators = pd.DataFrame(
        [
            {"as_of_date": "2026-04-30", "lpr": 3.20, "rrr": 7.0, "cny_usd": 7.05,
             "commodity_index_zscore": -0.5, "cpi_yoy": 0.012, "credit_impulse": 0.05,
             "pmi": 50.8, "budget_deficit_ratio": 0.040},
        ]
    )
    snapshot = analyze_macro(macro_indicators, "2026-05-14")
    assert snapshot.monetary_stance in {"easing", "neutral", "tightening"}
    fundamentals = pd.DataFrame(
        [
            {"symbol": "600001.SH", "report_date": "2026-03-31", "industry": "server",
             "revenue": 100.0, "net_income": 8.0, "gross_margin": 0.22,
             "revenue_growth": 0.18, "capex": -8.0, "total_assets": 200.0,
             "inventory": 18.0, "cogs": 70.0, "order_visibility_score": 70.0,
             "capacity_release_score": 70.0, "roa": 0.05},
            {"symbol": "002371.SZ", "report_date": "2026-03-31", "industry": "server",
             "revenue": 80.0, "net_income": 5.0, "gross_margin": 0.20,
             "revenue_growth": 0.12, "capex": -4.0, "total_assets": 150.0,
             "inventory": 16.0, "cogs": 60.0, "order_visibility_score": 65.0,
             "capacity_release_score": 65.0, "roa": 0.04},
            {"symbol": "300750.SZ", "report_date": "2026-03-31", "industry": "server",
             "revenue": 200.0, "net_income": 20.0, "gross_margin": 0.30,
             "revenue_growth": 0.20, "capex": -25.0, "total_assets": 400.0,
             "inventory": 50.0, "cogs": 130.0, "order_visibility_score": 75.0,
             "capacity_release_score": 75.0, "roa": 0.06},
        ]
    )
    snapshots = analyze_industries(fundamentals, theme_profiles=[], macro_snapshot=snapshot)
    assert snapshots and snapshots[0].industry == "server"
    company_frame = industry_snapshots_to_company_frame(fundamentals, snapshots, "2026-05-14")
    assert {"symbol", "industry", "supply_demand_balance", "monetary_tailwind"}.issubset(company_frame.columns)


def test_retail_hft_risk_penalises_volume_spike():
    base_dates = pd.date_range("2026-01-01", periods=80, freq="B")
    rows = []
    for date in base_dates:
        rows.append({"trade_date": date, "symbol": "600001.SH", "open": 10.0,
                     "high": 10.2, "low": 9.8, "close": 10.0, "volume": 1_000_000,
                     "amount": 1.0e7, "is_limit_up": False, "is_limit_down": False,
                     "turnover_ratio": 0.05})
    rows[-1]["amount"] = 5.0e8  # spike
    rows[-1]["turnover_ratio"] = 0.35
    rows[-1]["is_limit_up"] = True
    panel = pd.DataFrame(rows)
    state = pd.DataFrame([{"symbol": "600001.SH", "is_limit_up": True, "is_limit_down": False,
                           "is_suspended": False, "is_st": False, "block_trade_share": 0.45}])
    reports = score_retail_hft_risk(panel, state, RetailHFTRiskConfig())
    assert reports
    report = reports[0]
    assert report.penalty_score >= 0.40
    assert report.extra_slippage_bps > 15.0


def _make_member(symbol: str, theme: str, exposure: float, fraud: float) -> ThematicUniverseMember:
    return ThematicUniverseMember(
        symbol=symbol,
        company_name=symbol,
        theme=theme,
        sub_theme=theme,
        chain_node=theme,
        exposure_type=ChainRelationType.DIRECT_EXPOSURE,
        exposure_score=exposure,
        revenue_exposure_estimate=exposure / 100.0,
        profit_exposure_estimate=exposure / 100.0,
        evidence_count=3,
        source_confidence=0.75,
        fundamental_score=70.0,
        valuation_score=60.0,
        quality_score=65.0,
        fraud_risk_score=fraud,
        liquidity_score=70.0,
        market_attention_score=65.0,
        theme_lifecycle_stage=None,  # type: ignore[arg-type]
        entry_date="2026-05-14",
        expiry_date="2026-11-14",
        last_validated_at="2026-05-14",
        watchlist_status=UniverseBucket.CORE_BENEFICIARY,
    )


def test_deep_alpha_model_emits_multi_horizon_predictions():
    from quantagent.v7.schemas import ThemeLifecycleStage

    members = [
        _make_member("600001.SH", "ai_compute", 80.0, 30.0),
        _make_member("002371.SZ", "ai_compute", 70.0, 35.0),
    ]
    members = [m.__class__(**{**m.__dict__, "theme_lifecycle_stage": ThemeLifecycleStage.CAPITAL_INFLOW}) for m in members]
    factor_frame = pd.DataFrame(
        [
            {"symbol": "600001.SH", "trade_date": "2026-05-14",
             "ret_1d": 0.01, "ret_5d": 0.03, "ret_20d": 0.05,
             "theme_strength": 60.0, "policy_strength": 70.0,
             "exposure_score": 80.0, "fundamental_score": 70.0,
             "quality_score": 65.0, "valuation_score": 60.0,
             "margin_of_safety": 0.20, "volatility_20d": 0.22,
             "growth_revenue_yoy_120d": 0.4, "quality_roe_persistence_120d": 0.5,
             "valuation_margin_of_safety_120d": 0.20, "risk_fraud_haircut_120d": 0.85},
            {"symbol": "002371.SZ", "trade_date": "2026-05-14",
             "ret_1d": 0.005, "ret_5d": 0.02, "ret_20d": 0.04,
             "theme_strength": 55.0, "policy_strength": 65.0,
             "exposure_score": 70.0, "fundamental_score": 65.0,
             "quality_score": 60.0, "valuation_score": 55.0,
             "margin_of_safety": 0.15, "volatility_20d": 0.20,
             "growth_revenue_yoy_120d": 0.3, "quality_roe_persistence_120d": 0.4,
             "valuation_margin_of_safety_120d": 0.15, "risk_fraud_haircut_120d": 0.80},
        ]
    )
    alphas = predict_v7_deep_alpha(factor_frame, members, [], V7DeepAlphaConfig())
    assert set(alphas) == {"600001.SH", "002371.SZ"}
    for alpha in alphas.values():
        assert alpha.alpha_120d != 0.0 or alpha.alpha_60d != 0.0
        assert 0.0 <= alpha.confidence <= 1.0


def test_long_short_allocator_shifts_to_cash_under_risk_off():
    from quantagent.v7.schemas import HedgeDecision, MultiHorizonAlpha, ThemeLifecycleStage

    alpha = MultiHorizonAlpha(
        symbol="600001.SH",
        alpha_1d=0.02, alpha_5d=0.03, alpha_20d=0.05,
        alpha_60d=0.07, alpha_120d=0.10, alpha_126d=0.10,
        expected_return=0.06, expected_excess_return=0.04,
        volatility_forecast=0.25, downside_risk=0.10,
        confidence=0.70, conformal_confidence=0.65,
        prediction_interval_low=-0.05, prediction_interval_high=0.15,
        rank_score=80.0, regime_adjusted_score=60.0,
        factor_contribution={}, evidence_contribution={},
        risk_penalty=0.10, final_alpha_score=60.0,
    )
    hedge = HedgeDecision(
        hedge_need_score=0.85,
        hedge_type="cash_and_exposure_reduction",
        hedge_weight=0.20,
        reduce_exposure_amount=0.30,
        cash_buffer_target=0.40,
        affected_positions=(),
        rationale="risk_off",
        reactivation_condition="risk_off<0.45",
    )
    market = MarketRegimeSnapshot(
        market_regime=MarketRegime.RISK_OFF,
        sector_regime={},
        risk_on_score=0.20, risk_off_score=0.80,
        liquidity_score=0.40, breadth_score=0.30,
        volatility_score=0.70, drawdown_risk=0.50,
        sector_rotation_score={"tech": 0.30},
        recommended_gross_exposure=0.40,
        recommended_cash_weight=0.45,
        hedge_need_score=0.80,
    )
    allocation = allocate_long_short({"600001.SH": alpha}, [], market, hedge)
    sleeves = allocation.sleeve_weights
    assert sleeves[SleeveType.CASH_BUFFER] >= 0.20
    assert sleeves[SleeveType.SHORT_EVENT] <= 0.20
    assert sleeves[SleeveType.HEDGE] > 0.05


def test_v7_pipeline_includes_new_keys():
    result = run_daily_v7_research("configs/v7.mock.yaml", "2026-05-14")
    for key in (
        "industry_chain_reasoner",
        "long_horizon_factors",
        "long_horizon_alpha",
        "intrinsic_valuation",
        "economics_macro",
        "economics_industries",
        "retail_hft_risk",
        "long_short_allocation",
    ):
        assert key in result, f"missing key {key}"
    assert isinstance(result["industry_chain_reasoner"], dict)
    assert isinstance(result["long_horizon_factors"], list)
    if result["long_horizon_factors"]:
        record = result["long_horizon_factors"][0]
        assert "symbol" in record
        assert any(field.endswith("_120d") for field in record)
