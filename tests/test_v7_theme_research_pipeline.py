import pandas as pd
import yaml

from quantagent.fundamental.financial_statement_agent import score_financial_statements
from quantagent.fundamental.fraud_risk_agent import score_fraud_risk
from quantagent.data.v7_datahub import V7DataQualityError
from quantagent.services.v7_pipeline_service import run_daily_v7_research, validate_v7
from quantagent.themes.policy_crawler import local_policy_documents
from quantagent.themes.policy_parser import parse_policy_document
from quantagent.themes.theme_extractor import discover_themes
from quantagent.themes.theme_universe_builder import build_thematic_universe
from quantagent.v7.schemas import UniverseBucket






def test_v7_daily_service_returns_closed_loop_without_orders():
    validation = validate_v7("configs/v7.default.yaml")
    result = run_daily_v7_research("configs/v7.mock.yaml", as_of_date="2026-05-14")

    assert validation["status"] == "passed"
    assert result["data_mode"]["provider_mode"] == "mock"
    assert result["theme_ranking"]
    assert result["stock_pool_selection"]
    assert result["thematic_universe"]
    assert result["fundamental_due_diligence"]
    assert result["portfolio_plan"]["target_weights"]
    assert result["portfolio_plan"]["sleeve_target_weights"]
    assert len(result["selected_themes"]) >= 2
    assert len(result["industry_chain"]["by_theme"]) >= 2
    assert "OrderIntent" not in str(result)
    assert "OrderManager" in result["order_boundary"]


def test_v7_daily_service_stops_when_stock_pool_gate_is_empty():
    config = yaml.safe_load(open("configs/v7.mock.yaml", encoding="utf-8")) or {}
    config["stock_pool_gate"] = {
        "enabled": True,
        "allow_satellite_if_confidence_above": 0.99,
        "require_factor_coverage": True,
        "block_false_association": True,
    }
    config["factor_applicability"] = {
        "hard_gate": True,
        "production_stages": ["disabled_stage_for_gate_regression"],
    }

    result = run_daily_v7_research(config, as_of_date="2026-05-14")

    assert result["stock_pool_gate"]["gate_failed"] is True
    assert result["stock_pool_gate"]["audit_reason"] == "empty_after_stock_pool_hard_gate"
    assert result["multi_horizon_alpha"] == {}
    assert result["portfolio_plan"]["target_weights"] == {}
    assert result["risk_report"]["risk_passed"] is False
    assert "stock_pool_gate_failed" in result["risk_report"]["risk_warnings"]
    assert result["audit_log"]["final_decision_reason"] == "empty_after_stock_pool_hard_gate"


def test_v7_default_daily_refuses_synthetic_fallback_when_data_missing():
    try:
        run_daily_v7_research("configs/v7.default.yaml", as_of_date="2026-05-14")
    except V7DataQualityError as exc:
        assert "refusing synthetic fallback" in str(exc)
    else:
        raise AssertionError("strict_local default must not fall back to synthetic data")


