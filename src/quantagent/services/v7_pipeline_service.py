from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quantagent.backtest.event_driven_theme_backtester import EventDrivenThemeBacktester
from quantagent.credibility.news_credibility_agent import score_news_credibility
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.providers.v7_research_provider import LocalV7ResearchProvider
from quantagent.factors.factor_applicability_agent import validate_factor_applicability
from quantagent.fundamental.financial_statement_agent import score_financial_statements
from quantagent.fundamental.fraud_risk_agent import score_fraud_risk
from quantagent.models.v7_multi_horizon import predict_v7_multi_horizon_alpha
from quantagent.portfolio.hedge_decision_engine import decide_v7_hedge
from quantagent.portfolio.strategic_tactical_allocator import construct_v7_portfolio
from quantagent.themes.company_exposure_mapper import map_company_exposures
from quantagent.themes.industry_chain_graph import build_industry_chain_graph
from quantagent.themes.policy_crawler import local_policy_documents
from quantagent.themes.policy_parser import parse_policy_document
from quantagent.themes.theme_extractor import discover_themes
from quantagent.themes.theme_universe_builder import build_thematic_universe
from quantagent.v7.dag import validate_dag
from quantagent.v7.schemas import (
    AuditLogRecord,
    BacktestAttributionReport,
    ExecutionConstraintReport,
    MarketRegime,
    MarketRegimeSnapshot,
    MultiHorizonAlpha,
    RiskGateReport,
    TechnicalTimingPlan,
)
from quantagent.v7.scoring import execution_feasibility_score


def load_v7_config(config: str | Path | dict[str, Any] | None = None) -> dict[str, Any]:
    if config is None:
        path = Path("configs/v7.default.yaml")
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(config, dict):
        return dict(config)
    path = Path(config)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def validate_v7(config: str | Path | dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_v7_config(config)
    dag_errors = validate_dag()
    safety = cfg.get("safety", {})
    violations = []
    if safety.get("agents_can_emit_orders", False):
        violations.append("agents_can_emit_orders_must_be_false")
    if safety.get("optimizer_output") != "target_weights":
        violations.append("optimizer_must_output_target_weights")
    if not safety.get("dry_run", True):
        violations.append("dry_run_default_must_be_true")
    if not safety.get("virtual_broker_only", True):
        violations.append("virtual_broker_only_default_must_be_true")
    status = "passed" if not dag_errors and not violations else "failed"
    return {
        "status": status,
        "dag_errors": dag_errors,
        "safety_violations": violations,
        "supported_horizons_days": cfg.get("data", {}).get("supported_horizons_days", [1, 5, 20, 60, 120, 126]),
        "agent_count": len(cfg.get("agents", {})),
    }


def run_daily_v7_research(config: str | Path | dict[str, Any] | None = None, as_of_date: str = "2026-05-14") -> dict[str, Any]:
    cfg = load_v7_config(config)
    bundle = _load_local_bundle(cfg, as_of_date)
    policy_rows = _frame_or_records(bundle.policies.frame, cfg.get("synthetic_policy_documents", _synthetic_policy_records(as_of_date)))
    documents = local_policy_documents(policy_rows)
    parsed = [parse_policy_document(document) for document in documents]
    news_scores = score_news_credibility(bundle.news.frame)
    theme_profiles, evidence = discover_themes(
        parsed,
        as_of_date,
        _frame_or(bundle.theme_metrics.frame, _synthetic_market_theme_metrics()),
    )
    primary_theme = max(theme_profiles, key=lambda item: item.theme_strength)
    chain_nodes, chain_edges = build_industry_chain_graph(primary_theme)

    financials_raw = _frame_or(bundle.fundamentals.frame, _synthetic_financials())
    fraud_scores = score_fraud_risk(financials_raw)
    fraud_by_symbol = {score.symbol: score for score in fraud_scores}
    financials = financials_raw.copy()
    financials["fraud_risk_score"] = financials["symbol"].map(lambda symbol: fraud_by_symbol[str(symbol)].overall_fraud_risk_score)
    fundamental_scores = score_financial_statements(financials)
    fundamentals = {score.symbol: score for score in fundamental_scores}

    base_universe = _frame_or(bundle.base_universe.frame, _synthetic_base_universe())
    company_theme_map = bundle.company_theme_map.frame
    if company_theme_map.empty:
        profiles = _frame_or(bundle.company_profiles.frame, pd.DataFrame())
        if not profiles.empty:
            company_theme_map = map_company_exposures(profiles, primary_theme.theme_name, chain_nodes, evidence, as_of_date=as_of_date)
    company_theme_map = _frame_or(company_theme_map, _synthetic_company_theme_map(as_of_date))
    market_state = _frame_or(bundle.market_state.frame, _synthetic_market_state())
    universe_members = build_thematic_universe(
        base_universe=base_universe,
        company_theme_map=company_theme_map,
        theme_profiles=theme_profiles,
        chain_nodes=chain_nodes,
        fundamentals=fundamentals,
        market_state=market_state,
        as_of_date=as_of_date,
    )
    market = _synthetic_market_regime()
    factor_frame = _feature_frame_for_v7(bundle, universe_members, theme_profiles, financials, market_state, as_of_date)
    factor_columns = _factor_columns(factor_frame)
    factor_applicability = validate_factor_applicability(factor_frame, factor_columns, universe_members, market.market_regime) if factor_columns else []
    alphas = predict_v7_multi_horizon_alpha(factor_frame, universe_members, factor_applicability)
    if not alphas:
        alphas = _build_synthetic_alphas(universe_members)
    timing = _build_timing(universe_members)
    portfolio = construct_v7_portfolio(universe_members, alphas, market, timing)
    hedge = decide_v7_hedge(market, portfolio, theme_crowding_score=primary_theme.crowding_score)
    execution_reports = _execution_reports(portfolio)
    risk_report = _risk_gate_report(universe_members, portfolio, execution_reports, hedge)
    backtest = _run_synthetic_theme_backtest(portfolio)
    audit = AuditLogRecord(
        decision_id=f"v7-{as_of_date}",
        timestamp=as_of_date,
        input_data_versions={
            "policy": bundle.policies.source,
            "market": bundle.market_panel.source,
            "financials": bundle.fundamentals.source,
            "company_theme_map": bundle.company_theme_map.source,
        },
        model_version="v7.synthetic.multi_horizon",
        feature_version="v7.synthetic.features",
        evidence_hashes=tuple(record.hash for record in evidence if record.hash),
        risk_gate_result="passed" if risk_report.risk_passed else "failed",
        final_decision_reason="V7 synthetic daily research run; no live orders emitted.",
    )
    return {
        "market_summary": {
            "market_regime": market.market_regime.value,
            "risk_off_score": market.risk_off_score,
            "recommended_gross_exposure": market.recommended_gross_exposure,
            "recommended_cash_weight": market.recommended_cash_weight,
            "hedge_need_score": hedge.hedge_need_score,
        },
        "theme_ranking": [_to_dict(profile) for profile in sorted(theme_profiles, key=lambda item: item.theme_strength, reverse=True)],
        "industry_chain": {
            "nodes": [_to_dict(node) for node in chain_nodes],
            "edges": [_to_dict(edge) for edge in chain_edges],
        },
        "thematic_universe": [_to_dict(member) for member in universe_members],
        "multi_horizon_alpha": {symbol: _to_dict(alpha) for symbol, alpha in alphas.items()},
        "factor_applicability": [_to_dict(item) for item in factor_applicability],
        "news_credibility": [_to_dict(item) for item in news_scores],
        "portfolio_plan": _to_dict(portfolio),
        "hedge_decision": _to_dict(hedge),
        "execution_constraints": [_to_dict(report) for report in execution_reports],
        "risk_report": _to_dict(risk_report),
        "backtest_attribution": _to_dict(backtest),
        "audit_log": _to_dict(audit),
        "order_boundary": "agents_and_optimizer_emit_no_orders; OrderManager is the only order-intent owner",
    }


def _load_local_bundle(cfg: dict[str, Any], as_of_date: str):
    data_cfg = cfg.get("data", {})
    root = data_cfg.get("v7_root", "data/v7")
    request = ProviderRequest(
        start_date=str(data_cfg.get("start_date", "1900-01-01")),
        end_date=str(data_cfg.get("end_date", as_of_date)),
        symbols=tuple(data_cfg.get("symbols", ())),
        universe=data_cfg.get("universe"),
    )
    return LocalV7ResearchProvider(root).load_bundle(request, as_of_date)


def _frame_or(frame: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    return frame if frame is not None and not frame.empty else fallback


def _frame_or_records(frame: pd.DataFrame, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return frame.to_dict("records") if frame is not None and not frame.empty else fallback


def _feature_frame_for_v7(bundle, universe_members: list, theme_profiles: list, financials: pd.DataFrame, market_state: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    base = bundle.factors.frame if not bundle.factors.frame.empty else bundle.market_panel.frame
    if base is None or base.empty:
        base = pd.DataFrame({"trade_date": [as_of_date for _ in universe_members], "symbol": [member.symbol for member in universe_members], "close": [10.0 + i for i, _ in enumerate(universe_members)], "amount": [1_000_000.0 for _ in universe_members]})
    data = base.copy()
    if "trade_date" not in data.columns:
        data["trade_date"] = as_of_date
    member_frame = pd.DataFrame(
        [
            {
                "symbol": member.symbol,
                "theme": member.theme,
                "theme_strength": next((profile.theme_strength for profile in theme_profiles if profile.theme_name == member.theme), 0.0) * 100.0,
                "policy_strength": next((profile.policy_strength for profile in theme_profiles if profile.theme_name == member.theme), 0.0) * 100.0,
                "industry_fundamental_strength": next((profile.industry_fundamental_strength for profile in theme_profiles if profile.theme_name == member.theme), 0.0) * 100.0,
                "exposure_score": member.exposure_score,
                "fundamental_score": member.fundamental_score,
                "quality_score": member.quality_score,
                "valuation_score": member.valuation_score,
                "fraud_risk_score": member.fraud_risk_score,
            }
            for member in universe_members
        ]
    )
    data = data.merge(member_frame, on="symbol", how="left", suffixes=("", "_member"))
    data = data.merge(market_state[["symbol", "market_attention_score", "liquidity_score"]].drop_duplicates("symbol"), on="symbol", how="left") if not market_state.empty and "symbol" in market_state.columns else data
    if "close" in data.columns:
        data = data.sort_values(["symbol", "trade_date"])
        data["ret_1d"] = data.groupby("symbol")["close"].pct_change().fillna(0.0)
        data["ret_5d"] = data.groupby("symbol")["close"].pct_change(5).fillna(data["ret_1d"])
        data["ret_20d"] = data.groupby("symbol")["close"].pct_change(20).fillna(data["ret_5d"])
        data["momentum_20d"] = data["ret_20d"]
        data["volatility_20d"] = data.groupby("symbol")["ret_1d"].transform(lambda item: item.rolling(20, min_periods=2).std()).fillna(0.20)
    if "sector_rotation_score" not in data.columns:
        data["sector_rotation_score"] = data.get("market_attention_score", pd.Series(50.0, index=data.index)).fillna(50.0) / 100.0
    return data


def _factor_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "trade_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "theme",
        "sector",
        "watchlist_status",
    }
    return [column for column in frame.select_dtypes("number").columns if column not in excluded and not column.startswith("forward_return_")]


def _synthetic_policy_records(as_of_date: str) -> list[dict[str, Any]]:
    return [
        {
            "document_id": "policy-ai-compute-001",
            "title": "Action plan for artificial intelligence compute infrastructure and power coordination",
            "body": "Support AI compute, GPU, server, optical module, CPO, liquid cooling, data center, power equipment and energy storage. Target year 2026. Pilot projects and procurement support.",
            "source": "ministry_joint_release",
            "source_level": "ministry",
            "published_at": as_of_date,
        },
        {
            "document_id": "policy-chip-001",
            "title": "Integrated circuit domestic substitution and advanced packaging support",
            "body": "Support semiconductor equipment, wafer foundry, advanced packaging, memory, EDA and materials. Target year 2027.",
            "source": "state_council",
            "source_level": "central",
            "published_at": as_of_date,
        },
    ]


def _synthetic_market_theme_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"theme": "ai_compute", "market_strength": 0.62, "industry_fundamental_strength": 0.58, "capital_flow_strength": 0.60, "news_sentiment_strength": 0.55, "bubble_risk": 0.35, "crowding_score": 0.48},
            {"theme": "semiconductor_domestic_substitution", "market_strength": 0.55, "industry_fundamental_strength": 0.62, "capital_flow_strength": 0.50, "news_sentiment_strength": 0.45, "bubble_risk": 0.30, "crowding_score": 0.40},
        ]
    )


def _synthetic_base_universe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "600001.SH", "company_name": "Synthetic AI Server", "sector": "technology", "industry": "server", "liquidity_score": 75.0, "is_st": False},
            {"symbol": "002371.SZ", "company_name": "Synthetic PCB", "sector": "electronics", "industry": "pcb", "liquidity_score": 68.0, "is_st": False},
            {"symbol": "300750.SZ", "company_name": "Synthetic Energy Storage", "sector": "power", "industry": "energy_storage", "liquidity_score": 82.0, "is_st": False},
            {"symbol": "688981.SH", "company_name": "Synthetic Foundry", "sector": "semiconductor", "industry": "foundry", "liquidity_score": 88.0, "is_st": False},
            {"symbol": "000858.SZ", "company_name": "Synthetic Unrelated", "sector": "consumer", "industry": "liquor", "liquidity_score": 80.0, "is_st": False},
        ]
    )


def _synthetic_company_theme_map(as_of_date: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "600001.SH", "theme": "ai_compute", "sub_theme": "server", "chain_node": "server", "exposure_type": "direct_exposure", "exposure_score": 82.0, "revenue_exposure_estimate": 0.48, "profit_exposure_estimate": 0.42, "source_confidence": 0.78, "evidence_count": 4, "entry_date": as_of_date},
            {"symbol": "002371.SZ", "theme": "ai_compute", "sub_theme": "pcb", "chain_node": "pcb", "exposure_type": "upstream_supplier", "exposure_score": 70.0, "revenue_exposure_estimate": 0.30, "profit_exposure_estimate": 0.28, "source_confidence": 0.70, "evidence_count": 3, "entry_date": as_of_date},
            {"symbol": "300750.SZ", "theme": "ai_compute", "sub_theme": "energy_storage", "chain_node": "energy_storage", "exposure_type": "infrastructure_dependency", "exposure_score": 52.0, "revenue_exposure_estimate": 0.12, "profit_exposure_estimate": 0.10, "source_confidence": 0.55, "evidence_count": 2, "entry_date": as_of_date},
            {"symbol": "688981.SH", "theme": "semiconductor_domestic_substitution", "sub_theme": "foundry", "chain_node": "foundry", "exposure_type": "critical_bottleneck", "exposure_score": 88.0, "revenue_exposure_estimate": 0.70, "profit_exposure_estimate": 0.62, "source_confidence": 0.82, "evidence_count": 4, "entry_date": as_of_date},
            {"symbol": "000858.SZ", "theme": "ai_compute", "sub_theme": "weak_concept", "chain_node": "cloud_application", "exposure_type": "false_association", "exposure_score": 15.0, "revenue_exposure_estimate": 0.00, "profit_exposure_estimate": 0.00, "source_confidence": 0.15, "evidence_count": 0, "entry_date": as_of_date},
        ]
    )


def _synthetic_financials() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "600001.SH", "report_date": "2026-03-31", "theme_revenue_exposure": 76.0, "revenue_growth": 0.28, "profit_growth": 0.24, "roe": 0.13, "roa": 0.07, "gross_margin": 0.24, "operating_cash_flow": 18.0, "net_income": 16.0, "debt_to_asset": 0.42, "order_visibility_score": 78.0, "capacity_release_score": 70.0, "customer_validation_score": 72.0, "pe_ttm": 32.0, "pb": 3.5, "receivables": 34.0, "revenue": 120.0, "inventory": 28.0, "cogs": 91.0, "total_assets": 260.0, "capex": -20.0},
            {"symbol": "002371.SZ", "report_date": "2026-03-31", "theme_revenue_exposure": 58.0, "revenue_growth": 0.18, "profit_growth": 0.20, "roe": 0.11, "roa": 0.06, "gross_margin": 0.22, "operating_cash_flow": 10.0, "net_income": 9.0, "debt_to_asset": 0.48, "order_visibility_score": 64.0, "capacity_release_score": 58.0, "customer_validation_score": 60.0, "pe_ttm": 28.0, "pb": 2.9, "receivables": 24.0, "revenue": 90.0, "inventory": 22.0, "cogs": 70.0, "total_assets": 190.0, "capex": -12.0},
            {"symbol": "300750.SZ", "report_date": "2026-03-31", "theme_revenue_exposure": 35.0, "revenue_growth": 0.12, "profit_growth": 0.10, "roe": 0.17, "roa": 0.09, "gross_margin": 0.26, "operating_cash_flow": 30.0, "net_income": 24.0, "debt_to_asset": 0.44, "order_visibility_score": 54.0, "capacity_release_score": 60.0, "customer_validation_score": 52.0, "pe_ttm": 24.0, "pb": 4.1, "receivables": 40.0, "revenue": 180.0, "inventory": 46.0, "cogs": 130.0, "total_assets": 420.0, "capex": -50.0},
            {"symbol": "688981.SH", "report_date": "2026-03-31", "theme_revenue_exposure": 80.0, "revenue_growth": 0.20, "profit_growth": 0.18, "roe": 0.08, "roa": 0.05, "gross_margin": 0.20, "operating_cash_flow": 25.0, "net_income": 18.0, "debt_to_asset": 0.35, "order_visibility_score": 70.0, "capacity_release_score": 76.0, "customer_validation_score": 68.0, "pe_ttm": 42.0, "pb": 3.2, "receivables": 55.0, "revenue": 210.0, "inventory": 60.0, "cogs": 160.0, "total_assets": 560.0, "capex": -80.0},
            {"symbol": "000858.SZ", "report_date": "2026-03-31", "theme_revenue_exposure": 0.0, "revenue_growth": 0.05, "profit_growth": 0.04, "roe": 0.20, "roa": 0.15, "gross_margin": 0.70, "operating_cash_flow": 40.0, "net_income": 35.0, "debt_to_asset": 0.20, "order_visibility_score": 10.0, "capacity_release_score": 10.0, "customer_validation_score": 10.0, "pe_ttm": 20.0, "pb": 5.0, "receivables": 8.0, "revenue": 200.0, "inventory": 45.0, "cogs": 60.0, "total_assets": 500.0, "capex": -8.0},
        ]
    )


def _synthetic_market_state() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "600001.SH", "liquidity_score": 75.0, "market_attention_score": 72.0, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"symbol": "002371.SZ", "liquidity_score": 68.0, "market_attention_score": 65.0, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"symbol": "300750.SZ", "liquidity_score": 82.0, "market_attention_score": 60.0, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"symbol": "688981.SH", "liquidity_score": 88.0, "market_attention_score": 70.0, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
            {"symbol": "000858.SZ", "liquidity_score": 80.0, "market_attention_score": 35.0, "is_limit_up": False, "is_limit_down": False, "is_suspended": False},
        ]
    )


def _synthetic_market_regime() -> MarketRegimeSnapshot:
    return MarketRegimeSnapshot(
        market_regime=MarketRegime.POLICY_DRIVEN,
        sector_regime={"technology": "capital_inflow", "semiconductor": "fundamental_validation"},
        risk_on_score=0.58,
        risk_off_score=0.32,
        liquidity_score=0.62,
        breadth_score=0.56,
        volatility_score=0.42,
        drawdown_risk=0.28,
        sector_rotation_score={"technology": 0.65, "semiconductor": 0.60},
        recommended_gross_exposure=0.68,
        recommended_cash_weight=0.22,
        hedge_need_score=0.35,
    )


def _build_synthetic_alphas(universe_members: list) -> dict[str, MultiHorizonAlpha]:
    alphas: dict[str, MultiHorizonAlpha] = {}
    for member in universe_members:
        base = max(0.0, min(1.0, (member.exposure_score * 0.45 + member.fundamental_score * 0.35 + member.valuation_score * 0.20 - member.fraud_risk_score * 0.25) / 100.0))
        alphas[member.symbol] = MultiHorizonAlpha(
            symbol=member.symbol,
            alpha_1d=base * 0.35,
            alpha_5d=base * 0.55,
            alpha_20d=base * 0.75,
            alpha_60d=base * 0.90,
            alpha_120d=base,
            alpha_126d=base * 0.95,
            expected_return=base * 0.12,
            expected_excess_return=base * 0.08,
            volatility_forecast=0.22,
            downside_risk=0.10 + member.fraud_risk_score / 800.0,
            confidence=max(0.05, min(0.95, member.source_confidence * (1.0 - member.fraud_risk_score / 180.0))),
            conformal_confidence=0.72,
            prediction_interval_low=-0.08,
            prediction_interval_high=0.16,
            rank_score=base * 100.0,
            regime_adjusted_score=base * 85.0,
            factor_contribution={"theme": base * 0.4, "fundamental": base * 0.35, "timing": base * 0.25},
            evidence_contribution={member.theme: base},
            risk_penalty=member.fraud_risk_score / 100.0,
            final_alpha_score=base * 100.0,
        )
    return alphas


def _build_timing(universe_members: list) -> dict[str, TechnicalTimingPlan]:
    return {
        member.symbol: TechnicalTimingPlan(
            symbol=member.symbol,
            timing_score=max(20.0, min(85.0, 45.0 + member.market_attention_score * 0.25 - member.fraud_risk_score * 0.10)),
            entry_zone=None,
            add_position_zone=None,
            reduce_zone=None,
            stop_loss_level=None,
            take_profit_level=None,
            invalidation_level=None,
            max_chase_risk=max(0.0, member.market_attention_score / 100.0 - 0.50),
            current_position_action="watch" if member.watchlist_status.value == "watchlist_pool" else "eligible_for_target_weight",
            rationale="Synthetic timing uses attention, fraud penalty, and V7 universe bucket.",
        )
        for member in universe_members
    }


def _execution_reports(portfolio) -> list[ExecutionConstraintReport]:
    state = _synthetic_market_state().set_index("symbol")
    reports: list[ExecutionConstraintReport] = []
    for symbol in portfolio.target_weights:
        row = state.loc[symbol] if symbol in state.index else pd.Series(dtype=object)
        feasibility = execution_feasibility_score(
            bool(row.get("is_suspended", False)),
            bool(row.get("is_limit_up", False)),
            bool(row.get("is_limit_down", False)),
            float(row.get("liquidity_score", 50.0)),
            0.05,
        )
        reports.append(
            ExecutionConstraintReport(
                symbol=symbol,
                can_buy=feasibility > 0.2 and not bool(row.get("is_limit_up", False)),
                can_sell=feasibility > 0.2 and not bool(row.get("is_limit_down", False)),
                t_plus_one_blocked=False,
                limit_up_no_buy=bool(row.get("is_limit_up", False)),
                limit_down_no_sell=bool(row.get("is_limit_down", False)),
                suspended_no_trade=bool(row.get("is_suspended", False)),
                st_blocked=False,
                min_lot_size=100,
                volume_participation_cap=0.10,
                slippage_bps=5.0,
                impact_bps=8.0,
                feasibility_score=feasibility,
                rejection_reason=None if feasibility > 0.2 else "low_execution_feasibility",
            )
        )
    return reports


def _risk_gate_report(universe_members: list, portfolio, execution_reports: list[ExecutionConstraintReport], hedge) -> RiskGateReport:
    member_by_symbol = {member.symbol: member for member in universe_members}
    rejected = {}
    reduced = {}
    blocked = {}
    warnings = []
    max_allowed = {}
    for symbol, weight in portfolio.target_weights.items():
        member = member_by_symbol[symbol]
        max_allowed[symbol] = min(portfolio.max_single_name_weight, 0.02 if member.fraud_risk_score >= 60.0 else portfolio.max_single_name_weight)
        if member.fraud_risk_score > 80.0:
            blocked[symbol] = "high_fraud_risk"
        if weight > max_allowed[symbol]:
            reduced[symbol] = max_allowed[symbol]
    for report in execution_reports:
        if report.rejection_reason:
            rejected[report.symbol] = report.rejection_reason
    if hedge.hedge_need_score > 0.55:
        warnings.append("hedge_need_elevated")
    return RiskGateReport(
        risk_passed=not blocked and not rejected,
        rejected_symbols=rejected,
        reduced_symbols=reduced,
        blocked_symbols=blocked,
        risk_warnings=tuple(warnings),
        max_allowed_position=max_allowed,
        required_cash_buffer=max(portfolio.cash_weight, hedge.cash_buffer_target),
        kill_switch_triggered=False,
        rationale="V7 synthetic risk gate checked fraud, execution feasibility, and hedge need.",
    )


def _run_synthetic_theme_backtest(portfolio) -> BacktestAttributionReport:
    dates = pd.date_range("2026-05-01", periods=8, freq="B")
    symbols = list(portfolio.target_weights) or ["600001.SH"]
    price_rows = []
    for i, date in enumerate(dates):
        for j, symbol in enumerate(symbols):
            close = 10.0 + i * (0.05 + j * 0.01)
            price_rows.append({"trade_date": date, "symbol": symbol, "open": close * 0.99, "high": close * 1.01, "low": close * 0.98, "close": close, "volume": 1_000_000 + j * 100_000, "amount": close * (1_000_000 + j * 100_000)})
    prices = pd.DataFrame(price_rows)
    weights = pd.DataFrame(0.0, index=dates, columns=symbols)
    for symbol, weight in portfolio.target_weights.items():
        weights.loc[dates[2]:, symbol] = weight
    membership = pd.DataFrame({"symbol": symbols, "theme": ["ai_compute" for _ in symbols]})
    result = EventDrivenThemeBacktester().run(weights, prices, membership)
    report = result.base_result.report
    return BacktestAttributionReport(
        annual_return=float(report.get("annualized_return", 0.0)),
        cumulative_return=float(result.base_result.diagnostics.get("total_return", 0.0)),
        sharpe=float(report.get("sharpe", 0.0)),
        sortino=float(report.get("sortino", 0.0)),
        max_drawdown=float(report.get("max_drawdown", 0.0)),
        calmar=float(report.get("calmar", 0.0)),
        volatility=float(report.get("volatility", 0.0)),
        hit_rate=0.0,
        win_loss_ratio=0.0,
        turnover=float(report.get("turnover", 0.0)),
        transaction_cost=float(report.get("cost_attribution", 0.0)),
        alpha=0.0,
        beta=0.0,
        information_ratio=0.0,
        rank_ic=0.0,
        rank_icir=0.0,
        factor_decay={},
        capacity=float(report.get("capacity_proxy", 0.0)),
        tail_risk=0.0,
        drawdown_recovery_days=0,
        theme_contribution=result.theme_contribution,
        factor_contribution={},
        agent_contribution={"theme_discovery_agent": 0.35, "fundamental_due_diligence_agent": 0.30, "multi_horizon_alpha_agent": 0.35},
    )


def _to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_dict(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(_to_dict(key)): _to_dict(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_dict(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value
