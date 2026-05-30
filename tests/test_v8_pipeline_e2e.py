"""End-to-end smoke for the v8 spec pipeline.

Threads every new layer together on a tiny synthetic dataset:

    canonical evidence → PIT lint → capital_flow_thesis → validation →
    decision_chain (with gross_exposure_budget gate) →
    execution constraint DSL → backtest with cost model →
    risk_events.json → daily_decision_report.md

Each section asserts the *shape* of its output (not realistic returns
— this is integration, not a strategy test). If any stage breaks the
contract the downstream depends on, the test fails with a clear pointer
to which seam regressed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    simulate_ashare_target_weights,
)
from quantagent.data.evidence import (
    to_canonical_evidence_frame,
    validate_pit_safety,
)
from quantagent.data.thesis import (
    CapitalFlowThesisBuilder,
    validate_theses,
    theses_to_frame,
)
from quantagent.diagnostics.daily_decision_report import (
    DailyDecisionInputs,
    build_daily_decision_report,
)
from quantagent.execution.constraints import (
    ExecutionConstraintEvaluator,
    ExecutionConstraintSet,
    OrderIntentRecord,
)
from quantagent.optimization.multi_objective_loss import (
    compute_multi_objective_loss,
)
from quantagent.portfolio.decision_chain import (
    Candidate,
    DecisionContext,
    run_decision_chain_batch,
    traces_to_frame,
)


def _policy_silver(n: int = 3) -> pd.DataFrame:
    base = pd.Timestamp("2024-03-01")
    return pd.DataFrame(
        [
            {
                "event_id": f"p{i}",
                "source": "csrc",
                "url": f"https://csrc.gov.cn/{i}",
                "announced_at": base + pd.Timedelta(days=i),
                "effective_at": base + pd.Timedelta(days=i),
                "available_at": base + pd.Timedelta(days=i, hours=1),
                "fetched_at": base + pd.Timedelta(days=i, hours=1),
                "title": f"政策{i}",
                "body_summary": "",
                "themes": ["tech_innovation"],
                "sectors_hint": ["Semi"],
                "policy_strength": 0.7,
                "source_version": "v1",
            }
            for i in range(n)
        ]
    )


def _broker_silver(n: int = 3) -> pd.DataFrame:
    base = pd.Timestamp("2024-03-05")
    return pd.DataFrame(
        [
            {
                "event_id": f"br{i}",
                "symbol": "600519.SH",
                "broker": "中信证券",
                "broker_tier": "tier_1",
                "announced_at": base + pd.Timedelta(days=i),
                "available_at": base + pd.Timedelta(days=i, hours=1),
                "rating": "buy",
                "rating_change": "upgrade",
                "target_price": 1900.0,
                "prev_target_price": 1600.0,
                "target_price_pct_change": 0.1875,
                "summary": "",
                "broker_credibility": 0.85,
                "source": "wind",
                "source_version": "v1",
                "fetched_at": base + pd.Timedelta(days=i, hours=2),
            }
            for i in range(n)
        ]
    )


def test_v8_pipeline_full_smoke(tmp_path: Path):
    # ─── 1. canonical evidence + PIT lint ───────────────────────────────────
    canonical = to_canonical_evidence_frame(
        policy_events=_policy_silver(),
        broker_reports=_broker_silver(),
    )
    assert len(canonical) > 0
    pit_report = validate_pit_safety(canonical)
    assert pit_report.passed, pit_report.to_dict()

    # ─── 2. thesis builder ──────────────────────────────────────────────────
    builder = CapitalFlowThesisBuilder()
    theses = builder.build(canonical)
    assert theses, "thesis builder should emit ≥1 thesis from the synthetic data"

    # ─── 3. thesis validation: synthetic panel with positive excess ─────────
    base = pd.Timestamp("2024-03-08")
    dates = pd.bdate_range(base, periods=125)
    panel_rows = []
    for d in dates:
        panel_rows.append({
            "trade_date": d, "sector_level_1": "Semi", "theme": "Semi",
            "symbol": "600519.SH", "sector_return": 0.004,
            "forward_return": 0.004, "benchmark_return": 0.0005,
        })
        panel_rows.append({
            "trade_date": d, "sector_level_1": "Other", "theme": "Other",
            "symbol": "OTHER.SH", "sector_return": 0.0005,
            "forward_return": 0.0005, "benchmark_return": 0.0005,
        })
    val_panel = pd.DataFrame(panel_rows)
    updated, results = validate_theses(theses, val_panel)
    assert len(updated) == len(theses)
    assert any(r.new_status in ("verified", "partially_verified") for r in results), (
        "at least one thesis should pick up positive excess in the bullish panel"
    )

    # ─── 4. decision chain with gross_exposure_budget gate ──────────────────
    regime = pd.Series(["normal"], index=[pd.Timestamp("2024-03-15")], name="regime")
    candidates = [
        Candidate(trade_date=pd.Timestamp("2024-03-15"),
                  symbol=f"60000{i}.SH", alpha_score=0.5,
                  setup_label="breakout", target_weight=0.02)
        for i in range(5)
    ]
    ctx = DecisionContext(
        regime_state=regime,
        current_gross_exposure=0.55,
        global_conviction=0.85,
    )
    traces = run_decision_chain_batch(candidates, ctx)
    assert all(t.final_decision == "eligible" for t in traces)
    traces_frame = traces_to_frame(traces)
    assert "gross_exposure_budget" in set(traces_frame["gate_name"])

    # ─── 5. execution constraint DSL ────────────────────────────────────────
    intents = [
        OrderIntentRecord(
            intent_id=f"i{i}",
            symbol=f"60000{i}.SH",
            side="buy",
            quantity=200,
            price=50.0,
            timestamp=pd.Timestamp(f"2024-03-15 10:{30 + i}:00"),
            portfolio_nav=1_000_000.0,
        )
        for i in range(5)
    ]
    evaluator = ExecutionConstraintEvaluator(ExecutionConstraintSet())
    constraint_report = evaluator.evaluate(intents)
    assert constraint_report.passed, constraint_report.to_dict()

    # ─── 6. backtest + risk_events ──────────────────────────────────────────
    market_panel = pd.DataFrame(
        [
            {"trade_date": d, "symbol": "600000.SH",
             "close": 10.0, "volume": 1_000_000.0, "amount": 10_000_000.0,
             "is_suspended": False, "is_st": False,
             "is_limit_up": False, "is_limit_down": False}
            for d in pd.bdate_range("2024-03-01", periods=5)
        ]
    )
    targets = pd.DataFrame(
        {"600000.SH": [0.02] * 5},
        index=pd.bdate_range("2024-03-01", periods=5),
    )
    bt_cfg = AShareExecutionSimulationConfig(
        initial_cash=1_000_000.0, slippage_bps=0,
        audit_log_dir=str(tmp_path / "audit"),
    )
    bt_result = simulate_ashare_target_weights(targets, market_panel, bt_cfg)
    risk_path = tmp_path / "risk_events.json"
    bt_result.write_risk_events(risk_path)
    assert risk_path.exists()
    parsed = json.loads(risk_path.read_text(encoding="utf-8"))
    assert isinstance(parsed, list)

    # ─── 7. multi-objective loss (Stage 5 terms) ────────────────────────────
    daily_rets = bt_result.nav.pct_change().dropna()
    if daily_rets.empty:
        daily_rets = pd.Series([0.001] * 30)
    loss = compute_multi_objective_loss(
        daily_rets,
        transaction_cost_rate=0.02,
        concentration_score=0.20,
        illiquidity_score=0.05,
        st_exposure_rate=0.0,
        execution_unfilled_rate=0.10,
    )
    assert "transaction_cost" in loss.as_dict()
    assert loss.transaction_cost == 0.02
    assert loss.concentration == 0.20

    # ─── 8. daily decision report ───────────────────────────────────────────
    thesis_frame = theses_to_frame(updated)
    decision_inputs = DailyDecisionInputs(
        as_of_date=pd.Timestamp("2024-03-15"),
        target_weights=pd.Series({"600000.SH": 0.02}),
        sector_map=pd.DataFrame([
            {"symbol": "600000.SH", "sector_level_1": "Semi"},
        ]),
        capital_flow_theses=thesis_frame,
        decision_traces=traces_frame,
        risk_events=parsed,
        market_regime="normal",
        global_conviction=0.85,
        gross_exposure=0.57,
    )
    report = build_daily_decision_report(decision_inputs)
    md_path = tmp_path / "daily.md"
    report.write(md_path)
    md = md_path.read_text(encoding="utf-8")
    assert "Daily Decision Report" in md
    assert "Semi" in md
    assert "normal" in md
