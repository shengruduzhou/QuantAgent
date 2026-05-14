from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from dataclasses import asdict as _asdict_dataclass

from quantagent.agents.llm_orchestrator import (
    EconomicsOverlay,
    ForensicsOverlay,
    LLMOrchestrator,
    NewsCredibilityAIScore,
    PolicyAnalysis,
    SentimentAIResult,
    SkillToggles,
    ValuationOverlay,
)
from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
from quantagent.backtest.event_driven_theme_backtester import EventDrivenThemeBacktester
from quantagent.credibility.news_credibility_agent import news_scores_to_evidence, score_news_credibility
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.v7_datahub import V7DataHub, V7DataHubResult
from quantagent.factors.factor_applicability_agent import validate_factor_applicability
from quantagent.factors.long_horizon_factors import (
    LongHorizonFactorConfig,
    compute_long_horizon_factors,
    long_horizon_alpha_score,
)
from quantagent.fundamental.due_diligence import build_fundamental_due_diligence
from quantagent.fundamental.economic_analyzer import (
    EconomicAnalyzerConfig,
    analyze_industries,
    analyze_macro,
    industry_snapshots_to_company_frame,
)
from quantagent.fundamental.financial_statement_agent import score_financial_statements
from quantagent.fundamental.fraud_risk_agent import score_fraud_risk
from quantagent.fundamental.intrinsic_valuation import (
    IntrinsicValuationConfig,
    value_universe,
)
from quantagent.fundamental.order_contract_agent import order_contract_evidence
from quantagent.models.v7_deep_alpha import V7DeepAlphaConfig, predict_v7_deep_alpha
from quantagent.models.v7_multi_horizon import predict_v7_multi_horizon_alpha
from quantagent.portfolio.hedge_decision_engine import decide_v7_hedge
from quantagent.portfolio.strategic_tactical_allocator import construct_v7_portfolio
from quantagent.risk.retail_hft_risk import (
    RetailHFTRiskConfig,
    apply_retail_hft_penalty,
    score_retail_hft_risk,
)
from quantagent.strategy.long_short_allocator import (
    LongShortAllocatorConfig,
    allocate_long_short,
)
from quantagent.themes.company_exposure_mapper import map_company_exposures
from quantagent.themes.industry_chain_graph import build_industry_chain_graph
from quantagent.themes.industry_chain_reasoner import (
    IndustryChainReasonerConfig,
    reason_industry_chain_for_themes,
)
from quantagent.themes.policy_crawler import local_policy_documents
from quantagent.themes.policy_parser import parse_policy_document
from quantagent.themes.policy_schema_extractor import extract_policy_schema_evidence
from quantagent.themes.stock_pool_selector import build_stock_pool_selection
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
    ThemeLifecycleStage,
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
    hub_result = _load_datahub_result(cfg, as_of_date)
    bundle = hub_result.bundle
    allow_synthetic = hub_result.allow_synthetic_fallback
    policy_rows = _frame_or_records(
        bundle.policies.frame,
        cfg.get("synthetic_policy_documents", _synthetic_policy_records(as_of_date)),
        allow_synthetic,
    )
    documents = local_policy_documents(policy_rows)
    llm_client = _build_llm_client(cfg)
    orchestrator = _build_orchestrator(cfg, llm_client)
    policy_analyses = orchestrator.analyze_policies(documents)
    parsed = [
        parse_policy_document(document, analysis=analysis)
        for document, analysis in zip(documents, policy_analyses)
    ]
    schema_evidence, schema_warnings = extract_policy_schema_evidence(
        documents,
        as_of_date,
        cfg.get("policy_extraction", {}) | {"allow_network": bool(cfg.get("data", {}).get("allow_network", False))},
    )
    ai_news_scores = orchestrator.score_news_batch(bundle.news.frame)
    news_scores = _merge_news_credibility(score_news_credibility(bundle.news.frame), ai_news_scores)
    announcement_evidence = order_contract_evidence(bundle.announcements.frame, as_of_date)
    news_evidence = news_scores_to_evidence(news_scores, as_of_date)
    theme_profiles, evidence = discover_themes(
        parsed,
        as_of_date,
        _frame_or(bundle.theme_metrics.frame, _synthetic_market_theme_metrics(), allow_synthetic),
        extra_evidence=schema_evidence + announcement_evidence + news_evidence,
    )
    selected_themes = _select_theme_profiles(theme_profiles, cfg)
    reasoner_cfg = cfg.get("industry_chain_reasoner", {}) or {}
    use_dynamic_reasoner = bool(reasoner_cfg.get("use_dynamic_reasoner", True))
    strict_no_template = bool(reasoner_cfg.get("strict_no_template_fallback", True))
    chain_reasoner_results: dict[str, Any] = {}
    if use_dynamic_reasoner:
        reasoner_config = IndustryChainReasonerConfig(
            min_evidence_count_for_node=int(reasoner_cfg.get("min_evidence_count_for_node", 1)),
            min_evidence_count_for_strong_edge=int(reasoner_cfg.get("min_evidence_count_for_strong_edge", 2)),
            weak_association_max_evidence=int(reasoner_cfg.get("weak_association_max_evidence", 1)),
            use_llm_refinement=bool(reasoner_cfg.get("use_llm_refinement", False))
            and bool(cfg.get("llm_skills", {}).get("enabled_skills", {}).get("industry_chain_reasoner", True)),
            strict_no_template_fallback=strict_no_template,
        )
        chain_reasoner_results = reason_industry_chain_for_themes(selected_themes, evidence, reasoner_config, llm_client)
        chain_by_theme = {
            theme: (list(result.nodes), list(result.edges)) for theme, result in chain_reasoner_results.items()
        }
        if not strict_no_template:
            for theme, (nodes, edges) in chain_by_theme.items():
                if not nodes:
                    template_nodes, template_edges = build_industry_chain_graph(
                        next(profile for profile in selected_themes if profile.theme_name == theme),
                        evidence,
                    )
                    chain_by_theme[theme] = (template_nodes, template_edges)
    else:
        chain_by_theme = {profile.theme_name: build_industry_chain_graph(profile, evidence) for profile in selected_themes}
    all_chain_nodes = [node for nodes, _ in chain_by_theme.values() for node in nodes]
    all_chain_edges = [edge for _, edges in chain_by_theme.values() for edge in edges]

    financials_raw = _frame_or(bundle.fundamentals.frame, _synthetic_financials(), allow_synthetic)
    financials = _fundamental_input_frame(financials_raw, market_state=bundle.market_state.frame, market_panel=bundle.market_panel.frame)
    forensics_overlays: dict[str, ForensicsOverlay] = {}
    if financials.empty:
        fraud_scores = []
        fundamental_scores = []
    else:
        fraud_scores = score_fraud_risk(financials)
        forensics_overlays = _run_forensics_overlays(orchestrator, financials)
        fraud_scores = _apply_forensics_overlays(fraud_scores, forensics_overlays)
        fraud_by_symbol = {score.symbol: score for score in fraud_scores}
        financials["fraud_risk_score"] = financials["symbol"].map(
            lambda symbol: fraud_by_symbol[str(symbol)].overall_fraud_risk_score if str(symbol) in fraud_by_symbol else 50.0
        )
        fundamental_scores = score_financial_statements(financials)
    fundamental_due_diligence = build_fundamental_due_diligence(
        financials,
        fundamental_scores,
        fraud_scores,
        as_of_date,
    )
    fundamentals = {score.symbol: score for score in fundamental_scores}

    base_universe = _frame_or(bundle.base_universe.frame, _synthetic_base_universe(), allow_synthetic)
    company_theme_map = bundle.company_theme_map.frame
    if company_theme_map.empty:
        profiles = _frame_or(bundle.company_profiles.frame, pd.DataFrame(), False)
        mapped_frames = []
        if not profiles.empty:
            for profile in selected_themes:
                chain_nodes, _ = chain_by_theme[profile.theme_name]
                mapped = map_company_exposures(profiles, profile.theme_name, chain_nodes, evidence, as_of_date=as_of_date)
                if not mapped.empty:
                    mapped_frames.append(mapped)
        company_theme_map = pd.concat(mapped_frames, ignore_index=True) if mapped_frames else company_theme_map
    company_theme_map = _frame_or(company_theme_map, _synthetic_company_theme_map(as_of_date), allow_synthetic)
    if allow_synthetic:
        company_theme_map = _align_synthetic_company_theme_map(company_theme_map, selected_themes)
    market_state = _frame_or(bundle.market_state.frame, _synthetic_market_state(), allow_synthetic)
    universe_members = _build_multi_theme_universe(
        base_universe=base_universe,
        company_theme_map=company_theme_map,
        theme_profiles=selected_themes,
        chain_by_theme=chain_by_theme,
        fundamentals=fundamentals,
        market_state=market_state,
        as_of_date=as_of_date,
    )
    market = _market_regime_from_data(bundle.market_panel.frame, market_state, universe_members, selected_themes, allow_synthetic)

    macro_indicators = _maybe_frame(getattr(bundle, "macro_indicators", None))
    economic_cfg = cfg.get("economic_analyzer", {}) or {}
    macro_snapshot = analyze_macro(macro_indicators, as_of_date, EconomicAnalyzerConfig(
        capacity_utilization_target=float(economic_cfg.get("capacity_utilization_target", 0.80)),
        inventory_days_warning=float(economic_cfg.get("inventory_days_warning", 90.0)),
    )) if bool(economic_cfg.get("enabled", True)) else None
    industry_snapshots = (
        analyze_industries(financials, selected_themes, macro_snapshot)
        if macro_snapshot is not None and not financials.empty
        else []
    )
    economics_overlays = _run_economics_overlays(orchestrator, industry_snapshots, macro_snapshot, selected_themes)
    industry_snapshots = _apply_economics_overlays(industry_snapshots, economics_overlays)
    economics_company_frame = (
        industry_snapshots_to_company_frame(financials, industry_snapshots, as_of_date)
        if industry_snapshots
        else pd.DataFrame()
    )

    valuation_cfg = cfg.get("intrinsic_valuation", {}) or {}
    valuation_reports = (
        value_universe(
            financials,
            market_state,
            as_of_date,
            IntrinsicValuationConfig(
                default_terminal_growth=float(valuation_cfg.get("default_terminal_growth", 0.025)),
                default_forecast_years=int(valuation_cfg.get("default_forecast_years", 5)),
                fraud_confidence_haircut_threshold=float(valuation_cfg.get("fraud_confidence_haircut_threshold", 60.0)),
                fraud_confidence_haircut_strength=float(valuation_cfg.get("fraud_confidence_haircut_strength", 0.60)),
            ),
        )
        if bool(valuation_cfg.get("enabled", True)) and not financials.empty
        else []
    )
    valuation_overlays = _run_valuation_overlays(orchestrator, financials, market_state)
    valuation_reports = _apply_valuation_overlays(valuation_reports, valuation_overlays)
    theme_sentiments = _run_theme_sentiments(orchestrator, selected_themes, evidence)

    chain_features_frame = _chain_features_frame(universe_members, chain_by_theme)
    long_horizon_cfg = cfg.get("long_horizon_factors", {}) or {}
    long_horizon_factor_frame = (
        compute_long_horizon_factors(
            fundamentals=financials,
            market_state=market_state,
            price_panel=bundle.market_panel.frame,
            chain_features=chain_features_frame,
            economics_features=economics_company_frame,
            config=LongHorizonFactorConfig(
                fraud_haircut_threshold=float(long_horizon_cfg.get("fraud_haircut_threshold", 60.0)),
                fraud_haircut_strength=float(long_horizon_cfg.get("fraud_haircut_strength", 0.50)),
                policy_decay_half_life_days=float(long_horizon_cfg.get("policy_decay_half_life_days", 90.0)),
            ),
        )
        if bool(long_horizon_cfg.get("enabled", True))
        else pd.DataFrame()
    )
    long_horizon_alpha_frame = long_horizon_alpha_score(long_horizon_factor_frame) if not long_horizon_factor_frame.empty else pd.DataFrame()

    factor_frame = _feature_frame_for_v7(bundle, universe_members, theme_profiles, financials, market_state, as_of_date, allow_synthetic)
    factor_frame = _merge_long_horizon_factors(factor_frame, long_horizon_factor_frame, economics_company_frame)
    factor_columns = _factor_columns(factor_frame)
    factor_applicability = validate_factor_applicability(factor_frame, factor_columns, universe_members, market.market_regime) if factor_columns else []
    factor_hard_gate = bool(cfg.get("factor_applicability", {}).get("hard_gate", True))
    production_stages = tuple(cfg.get("factor_applicability", {}).get("production_stages", ("production", "validation")))
    production_applicability = (
        [item for item in factor_applicability if item.factor_lifecycle_stage in production_stages]
        if factor_hard_gate
        else list(factor_applicability)
    )
    stock_pool_selection = build_stock_pool_selection(universe_members, selected_themes, production_applicability, as_of_date)

    use_deep_alpha = bool(cfg.get("deep_alpha_model", {}).get("enabled", True))
    deep_cfg = cfg.get("deep_alpha_model", {}) or {}
    if use_deep_alpha:
        alphas = predict_v7_deep_alpha(
            factor_frame,
            universe_members,
            production_applicability,
            V7DeepAlphaConfig(
                hidden_size=int(deep_cfg.get("hidden_size", 16)),
                seed=int(deep_cfg.get("seed", 1729)),
                use_torch_if_available=bool(deep_cfg.get("use_torch_if_available", True)),
                fraud_penalty_weight=float(deep_cfg.get("fraud_penalty_weight", 0.30)),
            ),
        )
    else:
        alphas = predict_v7_multi_horizon_alpha(factor_frame, universe_members, production_applicability)
    if not alphas and allow_synthetic:
        alphas = _build_synthetic_alphas(universe_members)

    retail_cfg = cfg.get("retail_hft_risk", {}) or {}
    if bool(retail_cfg.get("enabled", True)):
        retail_hft_reports = score_retail_hft_risk(
            bundle.market_panel.frame,
            market_state,
            RetailHFTRiskConfig(
                base_penalty=float(retail_cfg.get("base_penalty", 0.20)),
                max_penalty=float(retail_cfg.get("max_penalty", 0.85)),
                institutional_volume_zscore_warning=float(retail_cfg.get("institutional_volume_zscore_warning", 2.0)),
            ),
        )
        alphas = apply_retail_hft_penalty(alphas, retail_hft_reports)
    else:
        retail_hft_reports = []
    timing = _build_timing(universe_members)
    portfolio_cfg = cfg.get("portfolio", {})
    portfolio = construct_v7_portfolio(
        universe_members,
        alphas,
        market,
        timing,
        current_weights=_current_weights(bundle.positions.frame),
        max_single_name_weight=float(portfolio_cfg.get("max_single_name_weight", 0.06)),
        max_sector_weight=float(portfolio_cfg.get("max_sector_weight", 0.30)),
        max_theme_weight=float(portfolio_cfg.get("max_theme_weight", 0.35)),
        turnover_limit=float(portfolio_cfg.get("max_turnover", portfolio_cfg.get("turnover_limit", 0.35))),
    )
    theme_crowding = _average_theme_crowding(selected_themes)
    hedge = decide_v7_hedge(market, portfolio, theme_crowding_score=theme_crowding)
    allocator_cfg = cfg.get("long_short_allocator", {}) or {}
    long_short_allocation = (
        allocate_long_short(
            alphas,
            universe_members,
            market,
            hedge,
            LongShortAllocatorConfig(
                long_horizon_confidence_threshold=float(allocator_cfg.get("long_horizon_confidence_threshold", 0.45)),
                short_horizon_confidence_threshold=float(allocator_cfg.get("short_horizon_confidence_threshold", 0.55)),
                hedge_scale=float(allocator_cfg.get("hedge_scale", 0.30)),
            ),
        )
        if bool(allocator_cfg.get("enabled", True))
        else None
    )
    execution_reports = _execution_reports(portfolio, market_state, bundle.positions.frame, cfg.get("execution", {}))
    risk_report = _risk_gate_report(universe_members, portfolio, execution_reports, hedge)
    backtest = _run_theme_backtest(portfolio, bundle.market_panel.frame, universe_members, allow_synthetic)
    audit = AuditLogRecord(
        decision_id=f"v7-{as_of_date}",
        timestamp=as_of_date,
        input_data_versions={
            "policy": bundle.policies.source,
            "market": bundle.market_panel.source,
            "financials": bundle.fundamentals.source,
            "company_theme_map": bundle.company_theme_map.source,
        },
        model_version="v7.mock.multi_horizon" if allow_synthetic else "v7.strict.multi_horizon",
        feature_version="v7.mock.features" if allow_synthetic else "v7.pit.features",
        evidence_hashes=tuple(record.hash for record in evidence if record.hash),
        risk_gate_result="passed" if risk_report.risk_passed else "failed",
        final_decision_reason="V7 research run emitted target_weights only; no live orders emitted.",
    )
    return {
        "data_mode": {
            "provider_mode": hub_result.provider_mode,
            "allow_synthetic_fallback": allow_synthetic,
            "warnings": list(hub_result.warnings + schema_warnings),
        },
        "market_summary": {
            "market_regime": market.market_regime.value,
            "risk_off_score": market.risk_off_score,
            "recommended_gross_exposure": market.recommended_gross_exposure,
            "recommended_cash_weight": market.recommended_cash_weight,
            "hedge_need_score": hedge.hedge_need_score,
        },
        "selected_themes": [profile.theme_name for profile in selected_themes],
        "theme_ranking": [_to_dict(profile) for profile in sorted(theme_profiles, key=lambda item: item.theme_strength, reverse=True)],
        "industry_chain": {
            "nodes": [_to_dict(node) for node in all_chain_nodes],
            "edges": [_to_dict(edge) for edge in all_chain_edges],
            "by_theme": {
                theme: {
                    "nodes": [_to_dict(node) for node in nodes],
                    "edges": [_to_dict(edge) for edge in edges],
                }
                for theme, (nodes, edges) in chain_by_theme.items()
            },
        },
        "stock_pool_selection": [_to_dict(report) for report in stock_pool_selection],
        "thematic_universe": [_to_dict(member) for member in universe_members],
        "fundamental_due_diligence": [_to_dict(report) for report in fundamental_due_diligence],
        "intrinsic_valuation": [_to_dict(report) for report in valuation_reports],
        "valuation_overlays": [_to_dict(overlay) for overlay in valuation_overlays.values()],
        "forensics_overlays": [_to_dict(overlay) for overlay in forensics_overlays.values()],
        "policy_ai_analyses": [_to_dict(analysis) for analysis in policy_analyses],
        "ai_news_credibility": [_to_dict(score) for score in ai_news_scores],
        "theme_sentiments": [_to_dict(sentiment) for sentiment in theme_sentiments],
        "economics_macro": _to_dict(macro_snapshot) if macro_snapshot is not None else {},
        "economics_industries": [_to_dict(snapshot) for snapshot in industry_snapshots],
        "economics_overlays": [_to_dict(overlay) for overlay in economics_overlays.values()],
        "long_horizon_factors": _frame_to_records(long_horizon_factor_frame),
        "long_horizon_alpha": _frame_to_records(long_horizon_alpha_frame),
        "industry_chain_reasoner": {
            theme: {
                "chain_confidence": result.chain_confidence,
                "used_llm": result.used_llm,
                "rationale": result.rationale,
                "nodes": [_to_dict(node) for node in result.nodes],
                "edges": [_to_dict(edge) for edge in result.edges],
            }
            for theme, result in chain_reasoner_results.items()
        },
        "multi_horizon_alpha": {symbol: _to_dict(alpha) for symbol, alpha in alphas.items()},
        "factor_applicability": [_to_dict(item) for item in factor_applicability],
        "news_credibility": [_to_dict(item) for item in news_scores],
        "retail_hft_risk": [_to_dict(report) for report in retail_hft_reports],
        "long_short_allocation": _to_dict(long_short_allocation) if long_short_allocation is not None else {},
        "portfolio_plan": _to_dict(portfolio),
        "hedge_decision": _to_dict(hedge),
        "execution_constraints": [_to_dict(report) for report in execution_reports],
        "risk_report": _to_dict(risk_report),
        "backtest_attribution": _to_dict(backtest),
        "audit_log": _to_dict(audit),
        "order_boundary": "agents_and_optimizer_emit_no_orders; OrderManager is the only order-intent owner",
    }


def _load_datahub_result(cfg: dict[str, Any], as_of_date: str) -> V7DataHubResult:
    data_cfg = cfg.get("data", {})
    request = ProviderRequest(
        start_date=str(data_cfg.get("start_date", "1900-01-01")),
        end_date=str(data_cfg.get("end_date", as_of_date)),
        symbols=tuple(data_cfg.get("symbols", ())),
        universe=data_cfg.get("universe"),
    )
    return V7DataHub(data_cfg).load(request, as_of_date)


def _frame_or(frame: pd.DataFrame, fallback: pd.DataFrame, allow_fallback: bool) -> pd.DataFrame:
    if frame is not None and not frame.empty:
        return frame
    return fallback if allow_fallback else pd.DataFrame()


def _frame_or_records(frame: pd.DataFrame, fallback: list[dict[str, Any]], allow_fallback: bool) -> list[dict[str, Any]]:
    if frame is not None and not frame.empty:
        return frame.to_dict("records")
    return fallback if allow_fallback else []


def _select_theme_profiles(theme_profiles: list, cfg: dict[str, Any]) -> list:
    if not theme_profiles:
        return []
    threshold = float(cfg.get("themes", {}).get("min_theme_strength", 0.20))
    inactive = {ThemeLifecycleStage.DECAY, ThemeLifecycleStage.INVALIDATED}
    selected = [
        profile
        for profile in theme_profiles
        if profile.theme_strength >= threshold and profile.lifecycle_stage not in inactive
    ]
    return selected or sorted(theme_profiles, key=lambda item: item.theme_strength, reverse=True)[:1]


def _build_multi_theme_universe(
    base_universe: pd.DataFrame,
    company_theme_map: pd.DataFrame,
    theme_profiles: list,
    chain_by_theme: dict[str, tuple[list, list]],
    fundamentals: dict[str, Any],
    market_state: pd.DataFrame,
    as_of_date: str,
) -> list:
    members = []
    if base_universe.empty or company_theme_map.empty:
        return members
    for profile in theme_profiles:
        if "theme" not in company_theme_map.columns:
            continue
        theme_map = company_theme_map[company_theme_map["theme"].astype(str) == profile.theme_name]
        if theme_map.empty:
            continue
        chain_nodes, _ = chain_by_theme.get(profile.theme_name, ([], []))
        members.extend(
            build_thematic_universe(
                base_universe=base_universe,
                company_theme_map=theme_map,
                theme_profiles=[profile],
                chain_nodes=chain_nodes,
                fundamentals=fundamentals,
                market_state=market_state,
                as_of_date=as_of_date,
            )
        )
    return _dedupe_members(members)


def _align_synthetic_company_theme_map(frame: pd.DataFrame, theme_profiles: list) -> pd.DataFrame:
    """When the orchestrator falls back to token-derived theme names, the
    synthetic ``company_theme_map`` (which is itself just a unit-test stub) must
    track those names — otherwise the mock universe is empty. In production
    ``map_company_exposures`` builds this frame dynamically and this step is a
    no-op.
    """

    if frame is None or frame.empty or "theme" not in frame.columns:
        return frame
    available = [profile.theme_name for profile in theme_profiles if profile.theme_name]
    if not available:
        return frame
    existing = list(dict.fromkeys(frame["theme"].astype(str)))
    if any(theme in set(available) for theme in existing):
        return frame
    mapping = {old: available[index % len(available)] for index, old in enumerate(existing)}
    data = frame.copy()
    data["theme"] = data["theme"].astype(str).map(lambda value: mapping.get(value, value))
    return data


def _dedupe_members(members: list) -> list:
    best_by_key: dict[tuple[str, str], Any] = {}
    for member in members:
        key = (member.symbol, member.theme)
        current = best_by_key.get(key)
        if current is None or member.exposure_score > current.exposure_score:
            best_by_key[key] = member
    return sorted(best_by_key.values(), key=lambda item: (item.theme, item.watchlist_status.value, -item.exposure_score, item.symbol))


def _fundamental_input_frame(financials: pd.DataFrame, market_state: pd.DataFrame, market_panel: pd.DataFrame) -> pd.DataFrame:
    if financials is None or financials.empty:
        return pd.DataFrame()
    data = financials.copy()
    latest_market = _latest_symbol_rows(market_panel)
    latest_state = _latest_symbol_rows(market_state)
    for source in (latest_market, latest_state):
        if source.empty or "symbol" not in source.columns:
            continue
        add_columns = [
            column
            for column in (
                "symbol",
                "close",
                "price",
                "total_share_capital",
                "total_shares",
                "free_float_shares",
                "float_shares",
                "market_cap",
                "free_float_market_cap",
                "industry",
                "sector",
            )
            if (column == "symbol" and column in source.columns) or (column in source.columns and column not in data.columns)
        ]
        if len(add_columns) > 1:
            data = data.merge(source[add_columns].drop_duplicates("symbol"), on="symbol", how="left", suffixes=("", "_market"))
    return data


def _latest_symbol_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "symbol" not in frame.columns:
        return pd.DataFrame()
    data = frame.copy()
    date_column = "trade_date" if "trade_date" in data.columns else "available_at" if "available_at" in data.columns else None
    if date_column is None:
        return data.drop_duplicates("symbol", keep="last")
    data[date_column] = pd.to_datetime(data[date_column])
    return data.sort_values(["symbol", date_column]).groupby("symbol", sort=False).tail(1)


def _feature_frame_for_v7(bundle, universe_members: list, theme_profiles: list, financials: pd.DataFrame, market_state: pd.DataFrame, as_of_date: str, allow_synthetic: bool) -> pd.DataFrame:
    base = bundle.factors.frame if not bundle.factors.frame.empty else bundle.market_panel.frame
    if base is None or base.empty:
        if not allow_synthetic:
            return pd.DataFrame()
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
    if financials is not None and not financials.empty and "symbol" in financials.columns:
        financial_latest = financials.copy()
        if "report_date" in financial_latest.columns:
            financial_latest["report_date"] = pd.to_datetime(financial_latest["report_date"])
            financial_latest = financial_latest.sort_values(["symbol", "report_date"]).groupby("symbol", sort=False).tail(1)
        financial_columns = [
            column
            for column in (
                "symbol",
                "market_cap",
                "free_float_market_cap",
                "pe_ttm",
                "pb",
                "ps_ttm",
                "ev_ebitda",
                "peg",
                "industry_valuation_percentile",
                "history_valuation_percentile",
                "valuation_bubble_score",
                "margin_of_safety",
                "order_visibility_score",
                "capacity_release_score",
                "customer_validation_score",
            )
            if column in financial_latest.columns
        ]
        data = data.merge(financial_latest[financial_columns].drop_duplicates("symbol"), on="symbol", how="left", suffixes=("", "_financial"))
    if not market_state.empty and "symbol" in market_state.columns:
        market_columns = [column for column in ("symbol", "market_attention_score", "liquidity_score") if column in market_state.columns]
        data = data.merge(market_state[market_columns].drop_duplicates("symbol"), on="symbol", how="left")
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


def _market_regime_from_data(
    market_panel: pd.DataFrame,
    market_state: pd.DataFrame,
    universe_members: list,
    theme_profiles: list,
    allow_synthetic: bool,
) -> MarketRegimeSnapshot:
    if market_panel is None or market_panel.empty:
        if allow_synthetic:
            return _synthetic_market_regime()
        return _market_regime_from_state(market_state, universe_members, theme_profiles)
    data = market_panel.copy()
    if "trade_date" in data.columns:
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        data = data.sort_values(["symbol", "trade_date"])
    if "close" in data.columns:
        data["ret_1d"] = data.groupby("symbol")["close"].pct_change().fillna(0.0)
        data["ret_20d"] = data.groupby("symbol")["close"].pct_change(20).fillna(data["ret_1d"])
        data["drawdown"] = data.groupby("symbol")["close"].transform(lambda item: item / item.cummax() - 1.0).fillna(0.0)
        data["volatility_20d"] = data.groupby("symbol")["ret_1d"].transform(lambda item: item.rolling(20, min_periods=2).std()).fillna(0.0)
    latest = data.groupby("symbol", sort=False).tail(1) if "symbol" in data.columns else data.tail(1)
    liquidity = _score_from_columns(latest, ("liquidity_score",), default=0.55)
    if "amount" in latest.columns:
        amount_score = float(np.clip(np.log1p(latest["amount"].fillna(0.0).mean()) / np.log1p(1e10), 0.0, 1.0))
        liquidity = max(liquidity, amount_score)
    breadth = float((latest.get("ret_20d", pd.Series(dtype=float)).fillna(0.0) > 0).mean()) if "ret_20d" in latest.columns else 0.50
    volatility = float(np.clip(latest.get("volatility_20d", pd.Series([0.20])).fillna(0.20).mean() / 0.04, 0.0, 1.0))
    drawdown = float(np.clip(abs(latest.get("drawdown", pd.Series([0.0])).fillna(0.0).min()) / 0.20, 0.0, 1.0))
    crowding = _average_theme_crowding(theme_profiles)
    risk_off = float(np.clip(0.35 * volatility + 0.35 * drawdown + 0.20 * (1.0 - breadth) + 0.10 * crowding, 0.0, 1.0))
    risk_on = float(np.clip(0.40 * breadth + 0.30 * liquidity + 0.20 * (1.0 - risk_off) + 0.10 * _average_theme_strength(theme_profiles), 0.0, 1.0))
    regime = _classify_market_regime(risk_on, risk_off, volatility, breadth, theme_profiles)
    return MarketRegimeSnapshot(
        market_regime=regime,
        sector_regime=_sector_regime_from_members(universe_members),
        risk_on_score=risk_on,
        risk_off_score=risk_off,
        liquidity_score=liquidity,
        breadth_score=breadth,
        volatility_score=volatility,
        drawdown_risk=drawdown,
        sector_rotation_score=_sector_rotation_scores(universe_members, market_state),
        recommended_gross_exposure=float(np.clip(0.75 - 0.35 * risk_off + 0.10 * risk_on, 0.25, 0.85)),
        recommended_cash_weight=float(np.clip(0.15 + 0.35 * risk_off + 0.10 * (1.0 - liquidity), 0.10, 0.60)),
        hedge_need_score=float(np.clip(0.20 + 0.50 * risk_off + 0.20 * crowding, 0.0, 1.0)),
    )


def _market_regime_from_state(market_state: pd.DataFrame, universe_members: list, theme_profiles: list) -> MarketRegimeSnapshot:
    state = market_state if market_state is not None else pd.DataFrame()
    liquidity = _score_from_columns(state, ("liquidity_score",), default=0.50)
    attention = _score_from_columns(state, ("market_attention_score",), default=0.50)
    limit_pressure = _score_from_bool_columns(state, ("is_limit_up", "is_limit_down", "is_suspended"))
    crowding = _average_theme_crowding(theme_profiles)
    risk_off = float(np.clip(0.25 + 0.30 * limit_pressure + 0.20 * (1.0 - liquidity) + 0.25 * crowding, 0.0, 1.0))
    risk_on = float(np.clip(0.25 + 0.35 * attention + 0.25 * liquidity + 0.15 * _average_theme_strength(theme_profiles), 0.0, 1.0))
    return MarketRegimeSnapshot(
        market_regime=_classify_market_regime(risk_on, risk_off, 0.35, attention, theme_profiles),
        sector_regime=_sector_regime_from_members(universe_members),
        risk_on_score=risk_on,
        risk_off_score=risk_off,
        liquidity_score=liquidity,
        breadth_score=attention,
        volatility_score=0.35,
        drawdown_risk=risk_off * 0.50,
        sector_rotation_score=_sector_rotation_scores(universe_members, state),
        recommended_gross_exposure=float(np.clip(0.70 - 0.35 * risk_off + 0.10 * risk_on, 0.25, 0.85)),
        recommended_cash_weight=float(np.clip(0.18 + 0.35 * risk_off, 0.10, 0.60)),
        hedge_need_score=float(np.clip(0.20 + 0.45 * risk_off + 0.20 * crowding, 0.0, 1.0)),
    )


def _classify_market_regime(risk_on: float, risk_off: float, volatility: float, breadth: float, theme_profiles: list) -> MarketRegime:
    if risk_off >= 0.70:
        return MarketRegime.RISK_OFF
    if volatility >= 0.75:
        return MarketRegime.HIGH_VOLATILITY
    if risk_on >= 0.62 and breadth >= 0.55:
        return MarketRegime.RISK_ON
    if _average_theme_strength(theme_profiles) >= 0.45:
        return MarketRegime.POLICY_DRIVEN
    return MarketRegime.RANGE_BOUND


def _score_from_columns(frame: pd.DataFrame, columns: tuple[str, ...], default: float) -> float:
    if frame is None or frame.empty:
        return default
    values = []
    for column in columns:
        if column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not series.empty:
                value = float(series.mean())
                values.append(value / 100.0 if value > 1.0 else value)
    return float(np.clip(np.mean(values), 0.0, 1.0)) if values else default


def _score_from_bool_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> float:
    if frame is None or frame.empty:
        return 0.0
    values = []
    for column in columns:
        if column in frame.columns:
            values.append(float(frame[column].fillna(False).astype(bool).mean()))
    return float(np.clip(np.mean(values), 0.0, 1.0)) if values else 0.0


def _sector_regime_from_members(universe_members: list) -> dict[str, str]:
    sectors: dict[str, list[float]] = {}
    for member in universe_members:
        sector = member.sector or "unknown"
        sectors.setdefault(sector, []).append(member.market_attention_score)
    return {
        sector: "capital_inflow" if np.mean(scores) >= 60.0 else "neutral"
        for sector, scores in sectors.items()
    }


def _sector_rotation_scores(universe_members: list, market_state: pd.DataFrame) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    state_by_symbol = {}
    if market_state is not None and not market_state.empty and "symbol" in market_state.columns:
        state_by_symbol = {str(row["symbol"]): row.to_dict() for _, row in market_state.iterrows()}
    for member in universe_members:
        sector = member.sector or "unknown"
        row = state_by_symbol.get(member.symbol, {})
        value = float(row.get("sector_rotation_score", row.get("market_attention_score", member.market_attention_score)))
        scores.setdefault(sector, []).append(value / 100.0 if value > 1.0 else value)
    return {sector: float(np.clip(np.mean(values), 0.0, 1.0)) for sector, values in scores.items()}


def _average_theme_crowding(theme_profiles: list) -> float:
    if not theme_profiles:
        return 0.0
    return float(np.clip(np.mean([profile.crowding_score for profile in theme_profiles]), 0.0, 1.0))


def _average_theme_strength(theme_profiles: list) -> float:
    if not theme_profiles:
        return 0.0
    return float(np.clip(np.mean([profile.theme_strength for profile in theme_profiles]), 0.0, 1.0))


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


def _current_weights(positions: pd.DataFrame) -> dict[str, float]:
    if positions is None or positions.empty or "symbol" not in positions.columns:
        return {}
    weight_column = "current_weight" if "current_weight" in positions.columns else "weight" if "weight" in positions.columns else None
    if weight_column is None:
        return {}
    return {str(row["symbol"]): float(row.get(weight_column, 0.0)) for _, row in positions.iterrows()}


def _execution_reports(portfolio, market_state: pd.DataFrame, positions: pd.DataFrame, execution_cfg: dict[str, Any]) -> list[ExecutionConstraintReport]:
    state = market_state.set_index("symbol") if market_state is not None and not market_state.empty and "symbol" in market_state.columns else pd.DataFrame()
    position_state = positions.set_index("symbol") if positions is not None and not positions.empty and "symbol" in positions.columns else pd.DataFrame()
    current_weights = _current_weights(positions)
    lot_size = int(execution_cfg.get("lot_size", 100))
    volume_cap = float(execution_cfg.get("volume_participation_cap", 0.10))
    reports: list[ExecutionConstraintReport] = []
    for symbol in portfolio.target_weights:
        row = state.loc[symbol] if symbol in state.index else pd.Series(dtype=object)
        position = position_state.loc[symbol] if symbol in position_state.index else pd.Series(dtype=object)
        target_weight = float(portfolio.target_weights.get(symbol, 0.0))
        current_weight = float(current_weights.get(symbol, 0.0))
        available_shares = float(position.get("available_shares", position.get("sellable_shares", 0.0)))
        total_shares = float(position.get("total_shares", position.get("shares", available_shares)))
        reducing_position = target_weight < current_weight
        t_plus_one_blocked = bool(execution_cfg.get("t_plus_one", True)) and reducing_position and total_shares > 0 and available_shares <= 0
        is_limit_up = bool(row.get("is_limit_up", False))
        is_limit_down = bool(row.get("is_limit_down", False))
        is_suspended = bool(row.get("is_suspended", False))
        is_st = bool(row.get("is_st", False))
        feasibility = execution_feasibility_score(
            is_suspended,
            is_limit_up,
            is_limit_down,
            float(row.get("liquidity_score", 50.0)),
            min(volume_cap, 0.20),
        )
        if t_plus_one_blocked or (is_st and bool(execution_cfg.get("block_st", True))):
            feasibility = min(feasibility, 0.10)
        rejection_reason = _execution_rejection_reason(
            is_suspended,
            is_limit_up,
            is_limit_down,
            is_st,
            t_plus_one_blocked,
            feasibility,
            execution_cfg,
        )
        reports.append(
            ExecutionConstraintReport(
                symbol=symbol,
                can_buy=feasibility > 0.2 and not is_limit_up and not is_suspended and not is_st,
                can_sell=feasibility > 0.2 and not is_limit_down and not is_suspended and not t_plus_one_blocked,
                t_plus_one_blocked=t_plus_one_blocked,
                limit_up_no_buy=is_limit_up,
                limit_down_no_sell=is_limit_down,
                suspended_no_trade=is_suspended,
                st_blocked=is_st and bool(execution_cfg.get("block_st", True)),
                min_lot_size=lot_size,
                volume_participation_cap=volume_cap,
                slippage_bps=_slippage_bps(row, target_weight),
                impact_bps=_impact_bps(row, target_weight, volume_cap),
                feasibility_score=feasibility,
                rejection_reason=rejection_reason,
            )
        )
    return reports


def _execution_rejection_reason(
    is_suspended: bool,
    is_limit_up: bool,
    is_limit_down: bool,
    is_st: bool,
    t_plus_one_blocked: bool,
    feasibility: float,
    execution_cfg: dict[str, Any],
) -> str | None:
    if is_suspended and bool(execution_cfg.get("block_suspended", True)):
        return "suspended_no_trade"
    if is_st and bool(execution_cfg.get("block_st", True)):
        return "st_blocked"
    if is_limit_up and bool(execution_cfg.get("block_buy_limit_up", True)):
        return "limit_up_no_buy"
    if is_limit_down and bool(execution_cfg.get("block_sell_limit_down", True)):
        return "limit_down_no_sell"
    if t_plus_one_blocked:
        return "t_plus_one_no_sellable_shares"
    if feasibility <= 0.2:
        return "low_execution_feasibility"
    return None


def _slippage_bps(row: pd.Series, target_weight: float) -> float:
    liquidity = float(row.get("liquidity_score", 50.0))
    return float(np.clip(12.0 - liquidity / 12.0 + target_weight * 100.0, 2.0, 35.0))


def _impact_bps(row: pd.Series, target_weight: float, volume_cap: float) -> float:
    amount = float(row.get("amount", row.get("turnover_amount", 0.0)) or 0.0)
    amount_penalty = 20.0 if amount <= 0 else float(np.clip(1e8 / max(amount, 1.0), 0.0, 25.0))
    return float(np.clip(5.0 + amount_penalty + target_weight * 80.0 + volume_cap * 20.0, 5.0, 60.0))


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
    kill_switch = hedge.hedge_need_score >= 0.85 or len(blocked) >= max(3, len(portfolio.target_weights))
    return RiskGateReport(
        risk_passed=not blocked and not rejected and not kill_switch,
        rejected_symbols=rejected,
        reduced_symbols=reduced,
        blocked_symbols=blocked,
        risk_warnings=tuple(warnings),
        max_allowed_position=max_allowed,
        required_cash_buffer=max(portfolio.cash_weight, hedge.cash_buffer_target),
        kill_switch_triggered=kill_switch,
        rationale="V7 risk gate checked fraud, A-share execution feasibility, concentration, and hedge need.",
    )


def _run_theme_backtest(portfolio, market_panel: pd.DataFrame, universe_members: list, allow_synthetic: bool) -> BacktestAttributionReport:
    if market_panel is not None and not market_panel.empty and {"trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"}.issubset(market_panel.columns):
        prices = market_panel.copy()
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        dates = sorted(prices["trade_date"].drop_duplicates())
        symbols = list(portfolio.target_weights)
        if dates and symbols:
            weights = pd.DataFrame(0.0, index=dates, columns=symbols)
            weights.iloc[-1] = pd.Series(portfolio.target_weights)
            membership = pd.DataFrame(
                [
                    {"symbol": member.symbol, "theme": member.theme}
                    for member in universe_members
                    if member.symbol in symbols
                ]
            ).sort_values(["symbol", "theme"]).drop_duplicates("symbol", keep="first")
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
                agent_contribution={"event_driven_backtester": 1.0},
            )
    if allow_synthetic:
        return _run_synthetic_theme_backtest(portfolio)
    return BacktestAttributionReport(
        annual_return=0.0,
        cumulative_return=0.0,
        sharpe=0.0,
        sortino=0.0,
        max_drawdown=0.0,
        calmar=0.0,
        volatility=0.0,
        hit_rate=0.0,
        win_loss_ratio=0.0,
        turnover=0.0,
        transaction_cost=0.0,
        alpha=0.0,
        beta=0.0,
        information_ratio=0.0,
        rank_ic=0.0,
        rank_icir=0.0,
        factor_decay={},
        capacity=0.0,
        tail_risk=0.0,
        drawdown_recovery_days=0,
        theme_contribution={},
        factor_contribution={},
        agent_contribution={"backtest_data_missing": 1.0},
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


def _build_llm_client(cfg: dict[str, Any]) -> LLMSkillClient:
    skills_cfg = cfg.get("llm_skills", {}) or {}
    return LLMSkillClient(
        LLMSkillConfig(
            enabled=bool(skills_cfg.get("enabled", False)),
            allow_network=bool(skills_cfg.get("allow_network", False)),
            endpoint=str(skills_cfg.get("endpoint", "https://api.openai.com/v1/chat/completions")),
            model=str(skills_cfg.get("model", "gpt-4.1-mini")),
            api_key_env=str(skills_cfg.get("api_key_env", "OPENAI_API_KEY")),
            timeout_seconds=float(skills_cfg.get("timeout_seconds", 30.0)),
            max_input_chars=int(skills_cfg.get("max_input_chars", 16000)),
            temperature=float(skills_cfg.get("temperature", 0.0)),
        )
    )


def _build_orchestrator(cfg: dict[str, Any], client: LLMSkillClient) -> LLMOrchestrator:
    skills_cfg = cfg.get("llm_skills", {}) or {}
    toggles = SkillToggles(
        policy_analyst=bool(skills_cfg.get("enabled_skills", {}).get("policy_analyst", True)),
        industry_chain_reasoner=bool(skills_cfg.get("enabled_skills", {}).get("industry_chain_reasoner", True)),
        news_credibility_agent=bool(skills_cfg.get("enabled_skills", {}).get("news_credibility_agent", True)),
        sentiment_agent=bool(skills_cfg.get("enabled_skills", {}).get("sentiment_agent", False)),
        valuation_agent=bool(skills_cfg.get("enabled_skills", {}).get("valuation_agent", False)),
        financial_forensics_agent=bool(skills_cfg.get("enabled_skills", {}).get("financial_forensics_agent", False)),
        economics_agent=bool(skills_cfg.get("enabled_skills", {}).get("economics_agent", False)),
    )
    return LLMOrchestrator(client, toggles)


def _merge_news_credibility(base_scores: list, ai_scores: list[NewsCredibilityAIScore]) -> list:
    if not ai_scores:
        return base_scores
    ai_by_id = {score.news_id: score for score in ai_scores if score.used_llm}
    if not ai_by_id:
        return base_scores
    merged = []
    for score in base_scores:
        ai = ai_by_id.get(getattr(score, "news_id", ""))
        if ai is None:
            merged.append(score)
            continue
        merged.append(
            score.__class__(
                **{
                    field: getattr(score, field)
                    for field in score.__dataclass_fields__
                    if field
                    not in {
                        "source_reliability",
                        "is_primary_source",
                        "is_official",
                        "cross_validation_count",
                        "sentiment_score",
                        "fundamental_impact_score",
                        "short_term_impact_score",
                        "medium_term_impact_score",
                        "confidence",
                        "decay_half_life",
                        "horizon_days",
                        "rumor_risk",
                        "rationale",
                    }
                },
                source_reliability=ai.source_reliability,
                is_primary_source=ai.is_primary_source,
                is_official=ai.is_official,
                cross_validation_count=ai.cross_validation_count,
                sentiment_score=ai.sentiment_score,
                fundamental_impact_score=ai.fundamental_impact,
                short_term_impact_score=ai.short_term_impact,
                medium_term_impact_score=ai.medium_term_impact,
                confidence=ai.confidence,
                decay_half_life=ai.decay_half_life,
                horizon_days=ai.horizon_days,
                rumor_risk=ai.rumor_risk,
                rationale=f"ai_overlay:{ai.rationale}",
            )
        )
    return merged


def _run_forensics_overlays(orchestrator: LLMOrchestrator, financials: pd.DataFrame) -> dict[str, ForensicsOverlay]:
    if financials is None or financials.empty or not orchestrator.toggles.financial_forensics_agent:
        return {}
    overlays: dict[str, ForensicsOverlay] = {}
    latest = (
        financials.assign(report_date=pd.to_datetime(financials.get("report_date"), errors="coerce"))
        .sort_values(["symbol", "report_date"])
        .groupby("symbol", sort=False)
        .tail(1)
    )
    columns = [
        "symbol",
        "revenue",
        "net_income",
        "operating_cash_flow",
        "receivables",
        "inventory",
        "cogs",
        "total_assets",
        "debt_to_asset",
        "gross_margin",
        "audit_opinion",
        "recent_restatement",
        "related_party_revenue_share",
        "fraud_risk_score",
    ]
    for _, row in latest.iterrows():
        symbol = str(row.get("symbol", ""))
        if not symbol:
            continue
        blob = "\n".join(f"{column}: {row.get(column)}" for column in columns if column in row.index)
        overlay = orchestrator.overlay_forensics(symbol, blob)
        if overlay is not None:
            overlays[symbol] = overlay
    return overlays


def _apply_forensics_overlays(fraud_scores: list, overlays: dict[str, ForensicsOverlay]) -> list:
    if not overlays:
        return fraud_scores
    merged = []
    for score in fraud_scores:
        overlay = overlays.get(getattr(score, "symbol", ""))
        if overlay is None:
            merged.append(score)
            continue
        if not hasattr(score, "overall_fraud_risk_score"):
            merged.append(score)
            continue
        weight = 0.55
        blended = float(
            (1.0 - weight) * float(getattr(score, "overall_fraud_risk_score", 50.0))
            + weight * overlay.fraud_risk_score
        )
        try:
            merged.append(
                score.__class__(
                    **{
                        field: getattr(score, field)
                        for field in score.__dataclass_fields__
                        if field != "overall_fraud_risk_score"
                    },
                    overall_fraud_risk_score=blended,
                )
            )
        except TypeError:
            merged.append(score)
    return merged


def _run_valuation_overlays(orchestrator: LLMOrchestrator, financials: pd.DataFrame, market_state: pd.DataFrame) -> dict[str, ValuationOverlay]:
    if financials is None or financials.empty or not orchestrator.toggles.valuation_agent:
        return {}
    overlays: dict[str, ValuationOverlay] = {}
    market_lookup = {
        str(row.get("symbol")): row.to_dict()
        for _, row in (market_state if market_state is not None else pd.DataFrame()).iterrows()
        if str(row.get("symbol"))
    }
    latest = (
        financials.assign(report_date=pd.to_datetime(financials.get("report_date"), errors="coerce"))
        .sort_values(["symbol", "report_date"])
        .groupby("symbol", sort=False)
        .tail(1)
    )
    columns = [
        "symbol",
        "industry",
        "revenue",
        "net_income",
        "operating_cash_flow",
        "capex",
        "total_assets",
        "debt_to_asset",
        "gross_margin",
        "revenue_growth",
        "profit_growth",
        "roe",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "ev_ebitda",
        "peg",
        "eps",
        "book_value_per_share",
        "dividend_per_share",
    ]
    for _, row in latest.iterrows():
        symbol = str(row.get("symbol", ""))
        if not symbol:
            continue
        market = market_lookup.get(symbol, {})
        market_blob = "\n".join(
            f"{key}: {market[key]}"
            for key in ("close", "market_cap", "free_float_market_cap", "liquidity_score")
            if key in market
        )
        fin_blob = "\n".join(f"{column}: {row.get(column)}" for column in columns if column in row.index)
        overlay = orchestrator.overlay_valuation(symbol, f"{fin_blob}\n{market_blob}".strip())
        if overlay is not None:
            overlays[symbol] = overlay
    return overlays


def _apply_valuation_overlays(reports: list, overlays: dict[str, ValuationOverlay]) -> list:
    if not overlays:
        return reports
    merged = []
    for report in reports:
        overlay = overlays.get(getattr(report, "symbol", ""))
        if overlay is None:
            merged.append(report)
            continue
        try:
            merged.append(
                report.__class__(
                    **{
                        field: getattr(report, field)
                        for field in report.__dataclass_fields__
                        if field
                        not in {
                            "fair_value_per_share",
                            "margin_of_safety_pct",
                            "valuation_score",
                            "bubble_risk_score",
                            "rationale",
                        }
                    },
                    fair_value_per_share=overlay.fair_value_per_share or report.fair_value_per_share,
                    margin_of_safety_pct=float(
                        0.50 * report.margin_of_safety_pct + 0.50 * overlay.margin_of_safety_pct
                    ),
                    valuation_score=float(0.50 * report.valuation_score + 0.50 * overlay.valuation_score),
                    bubble_risk_score=max(report.bubble_risk_score, overlay.bubble_risk_score),
                    rationale=f"{report.rationale}; ai={overlay.rationale}",
                )
            )
        except TypeError:
            merged.append(report)
    return merged


def _run_economics_overlays(
    orchestrator: LLMOrchestrator,
    snapshots: list,
    macro,
    theme_profiles: list,
) -> dict[str, EconomicsOverlay]:
    if not snapshots or not orchestrator.toggles.economics_agent:
        return {}
    macro_blob = (
        f"as_of={getattr(macro, 'as_of_date', '')}; "
        f"cycle={getattr(macro, 'business_cycle_stage', '')}; "
        f"monetary={getattr(macro, 'monetary_stance', '')}; "
        f"fiscal={getattr(macro, 'fiscal_stance', '')}; "
        f"credit_impulse={getattr(macro, 'credit_impulse', 0.0)}; "
        f"cny_strength={getattr(macro, 'cny_strength', 0.0)}; "
        f"commodity_z={getattr(macro, 'commodity_index_zscore', 0.0)}"
        if macro is not None
        else ""
    )
    theme_blob = "; ".join(
        f"{getattr(profile, 'theme_name', '')}:{getattr(profile, 'policy_strength', 0.0):.2f}"
        for profile in theme_profiles
    )
    overlays: dict[str, EconomicsOverlay] = {}
    for snapshot in snapshots:
        industry = getattr(snapshot, "industry", "")
        if not industry:
            continue
        blob = (
            f"industry: {industry}\n"
            f"current_stage: {getattr(snapshot, 'industry_cycle_stage', '')}\n"
            f"supply_demand: {getattr(snapshot, 'supply_demand_balance', 0.0)}\n"
            f"capacity_utilization: {getattr(snapshot, 'capacity_utilization', 0.0)}\n"
            f"inventory_days_zscore: {getattr(snapshot, 'inventory_days_zscore', 0.0)}\n"
            f"capex_intensity_trend: {getattr(snapshot, 'capex_intensity_trend', 0.0)}\n"
            f"pricing_power: {getattr(snapshot, 'pricing_power', 0.0)}\n"
            f"policy_support: {getattr(snapshot, 'policy_support_strength', 0.0)}\n"
            f"expected_growth: {getattr(snapshot, 'expected_industry_revenue_growth_yoy', 0.0)}\n"
            f"macro: {macro_blob}\n"
            f"theme_policy: {theme_blob}"
        )
        overlay = orchestrator.overlay_economics(str(industry), blob)
        if overlay is not None:
            overlays[str(industry)] = overlay
    return overlays


def _apply_economics_overlays(snapshots: list, overlays: dict[str, EconomicsOverlay]) -> list:
    if not overlays:
        return snapshots
    merged = []
    for snapshot in snapshots:
        overlay = overlays.get(getattr(snapshot, "industry", ""))
        if overlay is None:
            merged.append(snapshot)
            continue
        try:
            merged.append(
                snapshot.__class__(
                    **{
                        field: getattr(snapshot, field)
                        for field in snapshot.__dataclass_fields__
                        if field
                        not in {
                            "industry_cycle_stage",
                            "supply_demand_balance",
                            "pricing_power",
                            "capacity_utilization",
                            "capex_intensity_trend",
                            "monetary_tailwind",
                            "fx_pressure",
                            "commodity_cost_pressure",
                            "policy_support_strength",
                            "expected_industry_revenue_growth_yoy",
                            "expected_horizon_days",
                            "economic_thesis",
                            "rationale",
                        }
                    },
                    industry_cycle_stage=overlay.industry_cycle_stage,
                    supply_demand_balance=overlay.supply_demand_balance,
                    pricing_power=overlay.pricing_power,
                    capacity_utilization=overlay.capacity_utilization,
                    capex_intensity_trend=overlay.capex_intensity_trend,
                    monetary_tailwind=overlay.monetary_tailwind,
                    fx_pressure=overlay.fx_pressure,
                    commodity_cost_pressure=overlay.commodity_cost_pressure,
                    policy_support_strength=overlay.policy_support_strength,
                    expected_industry_revenue_growth_yoy=overlay.expected_industry_revenue_growth_yoy,
                    expected_horizon_days=overlay.expected_horizon_days,
                    economic_thesis=overlay.economic_thesis,
                    rationale=f"{getattr(snapshot, 'rationale', '')}; ai={overlay.rationale}",
                )
            )
        except TypeError:
            merged.append(snapshot)
    return merged


def _run_theme_sentiments(
    orchestrator: LLMOrchestrator,
    theme_profiles: list,
    evidence: list,
) -> list[SentimentAIResult]:
    if not theme_profiles or not orchestrator.toggles.sentiment_agent:
        return []
    by_theme: dict[str, list[str]] = {}
    for record in evidence:
        theme = getattr(record, "theme", None)
        if not theme:
            continue
        rationale = getattr(record, "rationale", "")
        if rationale:
            by_theme.setdefault(theme, []).append(rationale)
    results: list[SentimentAIResult] = []
    for profile in theme_profiles:
        rationales = by_theme.get(profile.theme_name, [])
        blob = "\n".join(rationales[:60])
        if not blob:
            continue
        results.append(orchestrator.assess_sentiment(profile.theme_name, blob))
    return results


def _maybe_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if hasattr(value, "frame"):
        return getattr(value, "frame", pd.DataFrame())
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame()


def _chain_features_frame(universe_members: list, chain_by_theme: dict[str, tuple[list, list]]) -> pd.DataFrame:
    rows = []
    nodes_by_id: dict[str, Any] = {}
    for nodes, _ in chain_by_theme.values():
        for node in nodes:
            existing = nodes_by_id.get(node.node_id)
            if existing is None or node.policy_support_score > existing.policy_support_score:
                nodes_by_id[node.node_id] = node
    for member in universe_members:
        node = nodes_by_id.get(member.chain_node)
        if node is None:
            continue
        rows.append(
            {
                "symbol": member.symbol,
                "as_of_date": member.last_validated_at or member.entry_date,
                "chain_centrality": float(node.dependency_strength),
                "bottleneck_score": float(node.bottleneck_score),
                "domestic_substitution_score": float(node.domestic_substitution_score),
                "policy_support_decay": float(node.policy_support_score),
                "demand_visibility": float(node.demand_visibility),
            }
        )
    return pd.DataFrame(rows)


def _merge_long_horizon_factors(
    factor_frame: pd.DataFrame,
    long_horizon_frame: pd.DataFrame,
    economics_frame: pd.DataFrame,
) -> pd.DataFrame:
    if factor_frame is None or factor_frame.empty:
        return factor_frame
    data = factor_frame.copy()
    if long_horizon_frame is not None and not long_horizon_frame.empty:
        merge_columns = [column for column in long_horizon_frame.columns if column not in data.columns or column == "symbol"]
        data = data.merge(long_horizon_frame[merge_columns], on="symbol", how="left")
    if economics_frame is not None and not economics_frame.empty:
        economics_columns = [
            "symbol",
            "supply_demand_balance",
            "pricing_power",
            "capacity_utilization",
            "credit_impulse_alignment",
            "monetary_tailwind",
            "fx_pressure",
            "commodity_cost_pressure",
            "policy_support_strength",
            "expected_industry_revenue_growth_yoy",
            "cobb_douglas_efficiency",
            "company_pricing_power",
            "company_capital_efficiency",
            "company_demand_visibility",
        ]
        present = [column for column in economics_columns if column in economics_frame.columns]
        if present:
            data = data.merge(economics_frame[present], on="symbol", how="left")
    return data


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [
        {key: _coerce_jsonable(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _coerce_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value
