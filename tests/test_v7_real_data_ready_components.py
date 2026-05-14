from dataclasses import replace

import pandas as pd

from quantagent.credibility.news_credibility_agent import score_news_credibility
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.providers.policy_web_provider import PolicyWebProvider
from quantagent.data.providers.tradingview_provider import TradingViewPublicProvider
from quantagent.data.providers.v7_research_provider import LocalV7ResearchProvider
from quantagent.data.v7_datahub import V7DataHub, V7DataQualityError
from quantagent.factors.factor_applicability_agent import validate_factor_applicability
from quantagent.fundamental.due_diligence import build_fundamental_due_diligence
from quantagent.models.v7_multi_horizon import predict_v7_multi_horizon_alpha
from quantagent.portfolio.strategic_tactical_allocator import construct_v7_portfolio
from quantagent.themes.company_exposure_mapper import map_company_exposures
from quantagent.themes.industry_chain_graph import build_industry_chain_graph
from quantagent.themes.policy_crawler import local_policy_documents
from quantagent.themes.policy_schema_extractor import extract_policy_schema_evidence
from quantagent.themes.stock_pool_selector import build_stock_pool_selection
from quantagent.v7.schemas import MarketRegime, MarketRegimeSnapshot, MultiHorizonAlpha, ThematicUniverseMember, ThemeLifecycleStage, ThemeProfile, UniverseBucket, ChainRelationType
from quantagent.v7.schemas import FactorApplicability, FundamentalScore, FraudRiskScore, InvestmentHorizonBucket


def _theme_profile() -> ThemeProfile:
    return ThemeProfile(
        theme_name="ai_compute",
        theme_category="policy_industry",
        theme_strength=0.8,
        policy_strength=0.8,
        market_strength=0.6,
        industry_fundamental_strength=0.6,
        capital_flow_strength=0.5,
        news_sentiment_strength=0.5,
        lifecycle_stage=ThemeLifecycleStage.FUNDAMENTAL_VALIDATION,
        expected_horizon_days=120,
        theme_confidence=0.8,
        bubble_risk=0.3,
        crowding_score=0.4,
        expiry_date="2026-09-01",
        update_frequency="daily",
    )


def _member(symbol: str, theme: str = "ai_compute", bucket: UniverseBucket = UniverseBucket.CORE_BENEFICIARY) -> ThematicUniverseMember:
    return ThematicUniverseMember(
        symbol=symbol,
        company_name=symbol,
        theme=theme,
        sub_theme="server",
        chain_node="server",
        exposure_type=ChainRelationType.DIRECT_EXPOSURE,
        exposure_score=80.0,
        revenue_exposure_estimate=0.4,
        profit_exposure_estimate=0.3,
        evidence_count=3,
        source_confidence=0.8,
        fundamental_score=75.0,
        valuation_score=55.0,
        quality_score=70.0,
        fraud_risk_score=25.0,
        liquidity_score=70.0,
        market_attention_score=60.0,
        theme_lifecycle_stage=ThemeLifecycleStage.FUNDAMENTAL_VALIDATION,
        entry_date="2026-05-01",
        expiry_date="2026-09-01",
        last_validated_at="2026-05-14",
        watchlist_status=bucket,
    )


def test_local_v7_research_provider_filters_future_rows(tmp_path):
    root = tmp_path / "v7"
    root.mkdir()
    (root / "policies.csv").write_text(
        "document_id,title,body,source,source_level,published_at\n"
        "p1,AI policy,Support GPU,gov,ministry,2026-05-14\n"
        "p2,Future policy,Future leak,gov,ministry,2026-06-01\n",
        encoding="utf-8",
    )
    provider = LocalV7ResearchProvider(root)
    bundle = provider.load_bundle(ProviderRequest("2026-05-01", "2026-05-31"), "2026-05-14")

    assert len(bundle.policies.frame) == 1
    assert bundle.policies.frame.iloc[0]["document_id"] == "p1"
    assert "missing_v7_file" in bundle.fundamentals.warnings[0]


def test_v7_datahub_strict_requires_core_pit_tables(tmp_path):
    root = tmp_path / "v7"
    root.mkdir()
    (root / "policies.csv").write_text(
        "document_id,title,body,source,source_level,published_at\n"
        "p1,AI policy,Support GPU,gov,ministry,2026-05-14\n",
        encoding="utf-8",
    )
    hub = V7DataHub({"v7_root": str(root), "provider_mode": "strict_local", "allow_synthetic_fallback": False})

    try:
        hub.load(ProviderRequest("2026-05-01", "2026-05-31"), "2026-05-14")
    except V7DataQualityError as exc:
        assert "base_universe" in str(exc)
        assert "market_state" in str(exc)
    else:
        raise AssertionError("strict V7 DataHub must fail on missing core PIT tables")


def test_policy_web_provider_degrades_when_network_disabled():
    result = PolicyWebProvider(allow_network=False).fetch_policy_documents(
        ProviderRequest("2026-05-01", "2026-05-31"),
        ["https://www.gov.cn/example.html"],
        as_of_date="2026-05-14",
    )

    assert result.frame.empty
    assert result.quality_score == 0.0
    assert "disabled" in result.warnings[0]


def test_tradingview_public_provider_is_disabled_by_default():
    result = TradingViewPublicProvider(allow_network=False).fetch_public_pages(
        ProviderRequest("2026-05-01", "2026-05-31"),
        ["https://www.tradingview.com/symbols/SSE-600000/"],
        as_of_date="2026-05-14",
    )

    assert result.frame.empty
    assert result.quality_score == 0.0
    assert "disabled" in result.warnings[0]


def test_remote_policy_schema_extraction_is_optional_and_disabled():
    docs = local_policy_documents(
        [
            {
                "document_id": "p1",
                "title": "AI compute policy",
                "body": "Support GPU and server infrastructure.",
                "source": "gov",
                "source_level": "central",
                "published_at": "2026-05-14",
            }
        ]
    )
    evidence, warnings = extract_policy_schema_evidence(docs, "2026-05-14", {"enabled": False})

    assert evidence == []
    assert "disabled" in warnings[0]


def test_company_exposure_mapper_infers_chain_node_from_profile_text():
    nodes, _ = build_industry_chain_graph(_theme_profile())
    profiles = pd.DataFrame(
        [
            {
                "symbol": "600001.SH",
                "company_name": "AI Server Co",
                "business_scope": "AI server and data center infrastructure",
                "server_revenue_exposure": 0.45,
                "announcement_id": "a1",
            }
        ]
    )
    mapping = map_company_exposures(profiles, "ai_compute", nodes, as_of_date="2026-05-14")

    assert mapping.iloc[0]["chain_node"] == "server"
    assert mapping.iloc[0]["exposure_score"] > 70
    assert mapping.iloc[0]["source_confidence"] > 0.5


def test_news_credibility_penalizes_rumor_and_duplicate_reposts():
    news = pd.DataFrame(
        [
            {"news_id": "n1", "source_type": "company_announcement", "source": "exchange", "title": "Contract order confirmed", "symbol": "600001.SH", "theme": "ai_compute"},
            {"news_id": "n2", "source_type": "social_media", "source": "forum", "title": "Contract order confirmed", "symbol": "600001.SH", "theme": "ai_compute", "rumor_risk": 0.9},
        ]
    )
    scores = {item.news_id: item for item in score_news_credibility(news)}

    assert scores["n1"].confidence > scores["n2"].confidence
    assert scores["n2"].rumor_risk >= 0.9


def test_factor_applicability_validates_by_theme_and_horizon():
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    symbols = [f"S{i}" for i in range(6)]
    rows = []
    for date_index, date in enumerate(dates):
        for score, symbol in enumerate(symbols):
            close = 10.0 * (1.0 + 0.01 * score) ** date_index
            rows.append({"trade_date": date, "symbol": symbol, "close": close, "amount": 1_000_000 + score, "theme_momentum": score})
    frame = pd.DataFrame(rows)
    members = [_member(symbol, theme="ai_compute" if i < 3 else "semiconductor_domestic_substitution") for i, symbol in enumerate(symbols)]

    reports = validate_factor_applicability(frame, ["theme_momentum"], members, MarketRegime.POLICY_DRIVEN, config=None)

    assert reports
    assert any(report.factor_lifecycle_stage in {"production", "validation"} for report in reports)
    assert any(report.horizon_days == 1 for report in reports)
    assert all(report.validation_method.startswith("walk_forward_pit_embargo") for report in reports)
    assert all(report.validation_sample_count > 0 for report in reports)


def test_v7_multi_horizon_model_uses_feature_inputs_for_long_horizon():
    member = _member("600001.SH")
    features = pd.DataFrame(
        [
            {
                "trade_date": "2026-05-14",
                "symbol": "600001.SH",
                "close": 10.0,
                "amount": 1_000_000,
                "policy_strength": 85.0,
                "industry_fundamental_strength": 75.0,
                "fundamental_score": 80.0,
                "exposure_score": 88.0,
                "valuation_score": 55.0,
                "ret_1d": 0.01,
                "ret_5d": 0.03,
                "ret_20d": 0.08,
            }
        ]
    )
    output = predict_v7_multi_horizon_alpha(features, [member])

    alpha = output["600001.SH"]
    assert alpha.alpha_120d > alpha.alpha_1d
    assert alpha.confidence > 0
    assert "long_fundamental" in alpha.factor_contribution


def test_stock_pool_selection_reports_horizon_relation_and_factor_scope():
    members = [
        _member("600001.SH", bucket=UniverseBucket.CORE_BENEFICIARY),
        replace(
            _member("002371.SZ", bucket=UniverseBucket.STRONG_CORRELATION),
            exposure_type=ChainRelationType.INFRASTRUCTURE_DEPENDENCY,
        ),
        replace(
            _member("000858.SZ", bucket=UniverseBucket.EXCLUSION),
            exposure_type=ChainRelationType.FALSE_ASSOCIATION,
            fraud_risk_score=85.0,
        ),
    ]
    factors = [
        FactorApplicability(
            factor_name="long_quality_value",
            factor_category="quality",
            applicable_universe=("core_beneficiary_pool",),
            applicable_sector=("server",),
            applicable_theme=("ai_compute",),
            applicable_market_regime=(MarketRegime.POLICY_DRIVEN,),
            horizon_days=120,
            decay_half_life=60.0,
            rank_ic=0.05,
            rank_icir=0.20,
            hit_rate=0.58,
            turnover=0.10,
            capacity=10_000_000.0,
            crowding_score=0.30,
            factor_lifecycle_stage="validation",
            last_validated_at="2026-05-14",
            invalidation_condition="latest sliced ICIR turns negative",
        )
    ]

    reports = build_stock_pool_selection(members, [_theme_profile()], factors, as_of_date="2026-05-14")

    report = reports[0]
    assert report.horizon_bucket == InvestmentHorizonBucket.LONG_TERM
    assert report.core_symbols == ("600001.SH",)
    assert "002371.SZ" in report.strong_relation_symbols
    assert "000858.SZ" in report.false_association_symbols
    assert report.applicable_factor_names == ("long_quality_value",)


def test_fundamental_due_diligence_calculates_market_cap_and_discounts_fraud():
    financials = pd.DataFrame(
        [
            {
                "symbol": "600001.SH",
                "report_date": "2026-03-31",
                "close": 20.0,
                "total_share_capital": 1_000_000_000.0,
                "revenue": 100.0,
                "net_income": 8.0,
                "operating_cash_flow": 2.0,
                "total_assets": 180.0,
            }
        ]
    )
    fundamental = FundamentalScore(
        symbol="600001.SH",
        fundamental_score=70.0,
        quality_score=65.0,
        growth_score=60.0,
        valuation_score=45.0,
        earnings_visibility_score=62.0,
        fraud_risk_score=85.0,
        management_risk_score=40.0,
        margin_of_safety=-0.10,
        investment_horizon=120,
        confidence=0.80,
        rationale="synthetic report",
    )
    fraud = FraudRiskScore(
        symbol="600001.SH",
        beneish_m_score=None,
        piotroski_f_score=None,
        altman_z_score=None,
        accruals_quality_score=30.0,
        cashflow_quality_score=25.0,
        receivables_risk_score=80.0,
        inventory_risk_score=40.0,
        related_party_risk_score=20.0,
        regulatory_penalty_score=90.0,
        audit_opinion_score=10.0,
        overall_fraud_risk_score=85.0,
        risk_flags=("high_fraud_risk",),
    )

    report = build_fundamental_due_diligence(financials, [fundamental], [fraud], "2026-05-14")[0]

    assert report.market_cap == 20_000_000_000.0
    assert report.estimated_intrinsic_value_per_share == 18.0
    assert report.confidence < 0.20
    assert "high_fraud_risk" in report.fraud_flags


def test_v7_portfolio_enforces_sector_and_theme_caps():
    members = [
        replace(_member(f"S{i}"), sector="technology", theme="ai_compute")
        for i in range(8)
    ]
    alphas = {
        member.symbol: MultiHorizonAlpha(
            symbol=member.symbol,
            alpha_1d=0.3,
            alpha_5d=0.4,
            alpha_20d=0.5,
            alpha_60d=0.6,
            alpha_120d=0.7,
            alpha_126d=0.7,
            expected_return=0.1,
            expected_excess_return=0.08,
            volatility_forecast=0.2,
            downside_risk=0.08,
            confidence=0.8,
            conformal_confidence=0.7,
            prediction_interval_low=-0.05,
            prediction_interval_high=0.15,
            rank_score=80,
            regime_adjusted_score=70,
            risk_penalty=0.1,
            final_alpha_score=75,
        )
        for member in members
    }
    market = MarketRegimeSnapshot(
        market_regime=MarketRegime.POLICY_DRIVEN,
        sector_regime={"technology": "capital_inflow"},
        risk_on_score=0.6,
        risk_off_score=0.2,
        liquidity_score=0.7,
        breadth_score=0.6,
        volatility_score=0.3,
        drawdown_risk=0.1,
        sector_rotation_score={"technology": 0.7},
        recommended_gross_exposure=0.7,
        recommended_cash_weight=0.2,
        hedge_need_score=0.2,
    )
    portfolio = construct_v7_portfolio(members, alphas, market, max_sector_weight=0.12, max_theme_weight=0.15)

    assert portfolio.sector_weights["technology"] <= 0.1200001
    assert portfolio.theme_weights["ai_compute"] <= 0.1500001
    assert portfolio.sleeve_target_weights
    assert any(note.startswith("sector_cap_applied") for note in portfolio.constraint_notes)
