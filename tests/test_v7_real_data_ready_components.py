import pandas as pd

from quantagent.credibility.news_credibility_agent import score_news_credibility
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.providers.v7_research_provider import LocalV7ResearchProvider
from quantagent.factors.factor_applicability_agent import validate_factor_applicability
from quantagent.models.v7_multi_horizon import predict_v7_multi_horizon_alpha
from quantagent.themes.company_exposure_mapper import map_company_exposures
from quantagent.themes.industry_chain_graph import build_industry_chain_graph
from quantagent.v7.schemas import MarketRegime, ThematicUniverseMember, ThemeLifecycleStage, ThemeProfile, UniverseBucket, ChainRelationType


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
