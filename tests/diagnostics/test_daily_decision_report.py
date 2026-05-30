"""Daily decision report tests + LLM-isolation guard."""

from __future__ import annotations

import ast
from pathlib import Path
import pkgutil

import pandas as pd
import pytest

from quantagent.diagnostics.daily_decision_report import (
    DailyDecisionInputs,
    DailyDecisionReport,
    build_daily_decision_report,
)


# ---------------------------------------------------------------------------
# Empty / missing inputs degrade gracefully
# ---------------------------------------------------------------------------

def test_report_emits_markdown_even_with_zero_inputs():
    inputs = DailyDecisionInputs(as_of_date=pd.Timestamp("2024-03-01"))
    report = build_daily_decision_report(inputs)
    md = report.to_markdown()
    assert "Daily Decision Report" in md
    assert "2024-03-01" in md
    # Every section header still appears
    for section in (
        "## Summary",
        "## Sector picks",
        "## Stock picks",
        "## Position sizing",
        "## Rejected candidates",
        "## Risk view",
        "## Thesis corroboration",
    ):
        assert section in md


def test_report_summary_includes_regime_and_conviction():
    inputs = DailyDecisionInputs(
        as_of_date=pd.Timestamp("2024-03-01"),
        market_regime="bear",
        global_conviction=0.45,
        gross_exposure=0.55,
    )
    md = build_daily_decision_report(inputs).to_markdown()
    assert "bear" in md
    assert "0.450" in md
    assert "55.00%" in md


def test_report_sector_picks_grouped_by_sector_map():
    weights = pd.Series({"A.SH": 0.05, "B.SH": 0.03, "C.SH": 0.02}, name="w")
    sector_map = pd.DataFrame(
        [
            {"symbol": "A.SH", "sector_level_1": "Semi"},
            {"symbol": "B.SH", "sector_level_1": "Semi"},
            {"symbol": "C.SH", "sector_level_1": "Bank"},
        ]
    )
    pool = pd.DataFrame(
        [
            {"sector_level_1": "Semi", "pool_tier": "core"},
            {"sector_level_1": "Bank", "pool_tier": "watch"},
        ]
    )
    inputs = DailyDecisionInputs(
        as_of_date=pd.Timestamp("2024-03-01"),
        target_weights=weights,
        sector_map=sector_map,
        sector_pool=pool,
    )
    md = build_daily_decision_report(inputs).to_markdown()
    assert "Semi" in md
    assert "Bank" in md
    assert "core" in md
    assert "watch" in md
    # Semi total is 8% (A+B)
    assert "8.00%" in md


def test_report_stock_picks_show_delta_vs_prior():
    today = pd.Series({"A.SH": 0.05}, name="w")
    prior = pd.Series({"A.SH": 0.02}, name="w")
    inputs = DailyDecisionInputs(
        as_of_date=pd.Timestamp("2024-03-01"),
        target_weights=today,
        prior_weights=prior,
    )
    md = build_daily_decision_report(inputs).to_markdown()
    assert "+3.00%" in md


def test_report_position_sizing_warns_above_high_conviction_cap():
    inputs = DailyDecisionInputs(
        as_of_date=pd.Timestamp("2024-03-01"),
        gross_exposure=0.90,
    )
    md = build_daily_decision_report(inputs).to_markdown()
    assert "Above high-conviction cap" in md or "⚠️" in md


def test_report_rejected_candidates_groups_by_failed_gate():
    traces = pd.DataFrame(
        [
            {"final_decision": "rejected", "failed_gate": "liquidity"},
            {"final_decision": "rejected", "failed_gate": "liquidity"},
            {"final_decision": "rejected", "failed_gate": "st_status"},
            {"final_decision": "eligible", "failed_gate": None},
        ]
    )
    inputs = DailyDecisionInputs(
        as_of_date=pd.Timestamp("2024-03-01"), decision_traces=traces
    )
    md = build_daily_decision_report(inputs).to_markdown()
    assert "liquidity" in md
    assert "st_status" in md
    assert "| 2 |" in md
    assert "| 1 |" in md


def test_report_risk_view_counts_event_types():
    events = [
        {"event_type": "order_skipped"},
        {"event_type": "order_skipped"},
        {"event_type": "order_rejected"},
    ]
    inputs = DailyDecisionInputs(
        as_of_date=pd.Timestamp("2024-03-01"), risk_events=events
    )
    md = build_daily_decision_report(inputs).to_markdown()
    assert "order_skipped" in md
    assert "order_rejected" in md


def test_report_thesis_corroboration_shows_top_theses():
    theses = pd.DataFrame(
        [
            {
                "direction_kind": "theme",
                "direction_value": "Semi",
                "thesis_sign": 0.9,
                "confidence": 0.8,
                "contradiction_score": 0.0,
                "validation_status": "verified",
                "tradability_score": 0.85,
            }
        ]
    )
    inputs = DailyDecisionInputs(
        as_of_date=pd.Timestamp("2024-03-01"), capital_flow_theses=theses
    )
    md = build_daily_decision_report(inputs).to_markdown()
    assert "Semi" in md
    assert "verified" in md


def test_report_write_to_file(tmp_path):
    inputs = DailyDecisionInputs(as_of_date=pd.Timestamp("2024-03-01"))
    report = build_daily_decision_report(inputs)
    target = tmp_path / "daily.md"
    report.write(target)
    assert target.exists()
    assert "Daily Decision Report" in target.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM allowlist — spec section 10: OrderManager must not import any LLM
# ---------------------------------------------------------------------------

LLM_FORBIDDEN_MODULE_HINTS = (
    "anthropic",
    "openai",
    "llama",
    "transformers.pipelines",
    "langchain",
    "quantagent.agents",       # agents emit views only; never wire to OM
    "quantagent.themes.policy_parser",  # the LLM-driven extractor
    "quantagent.credibility.news_credibility_agent",
)


def _walk_imports(source: str) -> list[str]:
    """Return every top-level import target in a Python file."""
    out: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


def test_order_manager_does_not_import_any_llm_module():
    om_path = Path(__file__).resolve().parents[2] / "src" / "quantagent" / "execution" / "order_manager.py"
    source = om_path.read_text(encoding="utf-8")
    imports = _walk_imports(source)
    leaks = [
        imp for imp in imports
        if any(banned in imp for banned in LLM_FORBIDDEN_MODULE_HINTS)
    ]
    assert leaks == [], (
        "OrderManager must not import LLM-bearing modules — found: "
        + ", ".join(leaks)
    )


def test_risk_gate_does_not_import_any_llm_module():
    path = Path(__file__).resolve().parents[2] / "src" / "quantagent" / "risk" / "risk_gate.py"
    source = path.read_text(encoding="utf-8")
    imports = _walk_imports(source)
    leaks = [
        imp for imp in imports
        if any(banned in imp for banned in LLM_FORBIDDEN_MODULE_HINTS)
    ]
    assert leaks == [], (
        "RiskGate must not import LLM-bearing modules — found: "
        + ", ".join(leaks)
    )


def test_ashare_execution_simulator_does_not_import_any_llm_module():
    path = Path(__file__).resolve().parents[2] / "src" / "quantagent" / "backtest" / "ashare_execution_simulator.py"
    source = path.read_text(encoding="utf-8")
    imports = _walk_imports(source)
    leaks = [
        imp for imp in imports
        if any(banned in imp for banned in LLM_FORBIDDEN_MODULE_HINTS)
    ]
    assert leaks == [], (
        "ashare_execution_simulator must not import LLM-bearing modules — found: "
        + ", ".join(leaks)
    )
