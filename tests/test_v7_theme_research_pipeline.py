import pandas as pd

from quantagent.fundamental.confidence_adjuster import adjust_confidence
from quantagent.fundamental.financial_statement_agent import score_financial_statements
from quantagent.fundamental.fraud_risk_agent import score_fraud_risk
from quantagent.data.v7_datahub import V7DataQualityError
from quantagent.services.v7_pipeline_service import run_daily_v7_research, validate_v7
from quantagent.themes.industry_chain_graph import build_industry_chain_graph
from quantagent.themes.policy_crawler import local_policy_documents
from quantagent.themes.policy_parser import parse_policy_document
from quantagent.themes.theme_extractor import discover_themes
from quantagent.themes.theme_universe_builder import build_thematic_universe
from quantagent.v7.schemas import UniverseBucket


def test_policy_to_theme_to_industry_chain_pipeline_builds_real_pool():
    docs = local_policy_documents(
        [
            {
                "document_id": "p1",
                "title": "AI compute infrastructure pilot policy",
                "body": "Support GPU, AI server, optical module, PCB, liquid cooling and data center projects in 2026.",
                "source": "ministry",
                "source_level": "ministry",
                "published_at": "2026-05-14",
            }
        ]
    )
    parsed = [parse_policy_document(doc) for doc in docs]
    themes, evidence = discover_themes(parsed, "2026-05-14")
    ai_theme = themes[0]
    nodes, edges = build_industry_chain_graph(ai_theme)

    assert ai_theme.theme_name == "ai_compute"
    assert any(node.node_id == "gpu" for node in nodes)
    assert any(node.node_id == "optical_module" for node in nodes)
    assert edges
    assert evidence[0].hash


def test_theme_universe_distinguishes_core_from_false_association():
    docs = local_policy_documents(
        [
            {
                "document_id": "p1",
                "title": "AI compute infrastructure policy",
                "body": "Support GPU server and PCB for data center.",
                "source": "ministry",
                "source_level": "ministry",
                "published_at": "2026-05-14",
            }
        ]
    )
    themes, _ = discover_themes([parse_policy_document(doc) for doc in docs], "2026-05-14")
    nodes, _ = build_industry_chain_graph(themes[0])
    fundamental_rows = pd.DataFrame(
        [
            {"symbol": "600001.SH", "report_date": "2026-03-31", "theme_revenue_exposure": 80, "revenue_growth": 0.25, "profit_growth": 0.20, "roe": 0.12, "roa": 0.06, "gross_margin": 0.25, "operating_cash_flow": 10, "net_income": 8, "debt_to_asset": 0.4, "order_visibility_score": 80, "capacity_release_score": 75, "customer_validation_score": 75, "receivables": 10, "revenue": 100, "inventory": 15, "cogs": 70, "total_assets": 180, "capex": -8},
            {"symbol": "000858.SZ", "report_date": "2026-03-31", "theme_revenue_exposure": 0, "revenue_growth": 0.05, "profit_growth": 0.04, "roe": 0.18, "roa": 0.12, "gross_margin": 0.70, "operating_cash_flow": 20, "net_income": 16, "debt_to_asset": 0.2, "order_visibility_score": 5, "capacity_release_score": 5, "customer_validation_score": 5, "receivables": 4, "revenue": 120, "inventory": 12, "cogs": 35, "total_assets": 220, "capex": -5},
        ]
    )
    fraud = {item.symbol: item for item in score_fraud_risk(fundamental_rows)}
    fundamental_rows["fraud_risk_score"] = fundamental_rows["symbol"].map(lambda symbol: fraud[symbol].overall_fraud_risk_score)
    fundamentals = {item.symbol: item for item in score_financial_statements(fundamental_rows)}
    members = build_thematic_universe(
        pd.DataFrame(
            [
                {"symbol": "600001.SH", "company_name": "Synthetic Server", "liquidity_score": 70},
                {"symbol": "000858.SZ", "company_name": "Synthetic Unrelated", "liquidity_score": 80},
            ]
        ),
        pd.DataFrame(
            [
                {"symbol": "600001.SH", "theme": "ai_compute", "chain_node": "server", "exposure_type": "direct_exposure", "exposure_score": 85, "source_confidence": 0.8, "evidence_count": 4},
                {"symbol": "000858.SZ", "theme": "ai_compute", "chain_node": "cloud_application", "exposure_type": "false_association", "exposure_score": 15, "source_confidence": 0.1, "evidence_count": 0},
            ]
        ),
        themes,
        nodes,
        fundamentals,
        as_of_date="2026-05-14",
    )

    by_symbol = {member.symbol: member for member in members}
    assert by_symbol["600001.SH"].watchlist_status in {UniverseBucket.CORE_BENEFICIARY, UniverseBucket.STRONG_CORRELATION}
    assert by_symbol["000858.SZ"].watchlist_status == UniverseBucket.EXCLUSION


def test_v7_daily_service_returns_closed_loop_without_orders():
    validation = validate_v7("configs/v7.default.yaml")
    result = run_daily_v7_research("configs/v7.mock.yaml", as_of_date="2026-05-14")

    assert validation["status"] == "passed"
    assert result["data_mode"]["provider_mode"] == "mock"
    assert result["theme_ranking"]
    assert result["thematic_universe"]
    assert result["portfolio_plan"]["target_weights"]
    assert len(result["selected_themes"]) >= 2
    assert len(result["industry_chain"]["by_theme"]) >= 2
    assert "OrderIntent" not in str(result)
    assert "OrderManager" in result["order_boundary"]


def test_v7_default_daily_refuses_synthetic_fallback_when_data_missing():
    try:
        run_daily_v7_research("configs/v7.default.yaml", as_of_date="2026-05-14")
    except V7DataQualityError as exc:
        assert "refusing synthetic fallback" in str(exc)
    else:
        raise AssertionError("strict_local default must not fall back to synthetic data")


def test_confidence_adjuster_penalizes_high_fraud_risk():
    clean = adjust_confidence(0.8, fraud_risk_score=20, news_confidence=0.8, data_quality=1.0)
    risky = adjust_confidence(0.8, fraud_risk_score=85, news_confidence=0.8, data_quality=1.0)

    assert risky < clean
    assert risky <= 0.2
