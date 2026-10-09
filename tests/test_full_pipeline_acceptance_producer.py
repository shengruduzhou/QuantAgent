"""The full-pipeline acceptance producer must not supply measurements it never took.

Round 29 R9-F01: `_build_full_pipeline_acceptance_metrics` defaulted
`max_drawdown`, `sharpe`, `turnover_adjusted_net_return` and
`excess_return_after_costs` to 0.0. The acceptance gate had been hardened to
report `unknown` on a missing value (DEF-023), so the producer re-created the
defect from the other side: an empty paper summary reached the gate as a
*measured* drawdown of 0.0 and passed it. It also dropped the DEF-022 benchmark
coverage fields, so an incomplete benchmark was reported as "no benchmark".
"""

from __future__ import annotations

import pandas as pd

from quantagent.cli.v7_train import _build_full_pipeline_acceptance_metrics
from quantagent.data.v7_quality_gates import (
    GATE_PASS,
    GATE_UNKNOWN,
    V7ModelAcceptanceGateConfig,
    evaluate_model_acceptance_gates,
)


def _build(paper_summary: dict, benchmark_symbol: str | None = None) -> dict:
    return _build_full_pipeline_acceptance_metrics(
        training_metrics={"rank_ic_mean": 0.02},
        paper_summary=paper_summary,
        weight_diagnostics={},
        training_dataset=pd.DataFrame({"symbol": ["000001.SZ"], "trade_date": ["2026-01-05"]}),
        predictions=pd.DataFrame({"symbol": ["000001.SZ"]}),
        benchmark_symbol=benchmark_symbol,
    )


def _gate(report, name: str) -> dict:
    for gate in report.gates:
        if gate["name"] == name:
            return gate
    raise AssertionError(f"no gate named {name}: {[g['name'] for g in report.gates]}")


def test_an_empty_paper_summary_is_unmeasured_not_zero():
    built = _build({})
    for key in ("max_drawdown", "sharpe", "turnover_adjusted_net_return", "excess_return_after_costs"):
        assert built[key] is None, f"{key} was fabricated as {built[key]!r}"


def test_a_missing_drawdown_never_passes_the_drawdown_gate():
    report = evaluate_model_acceptance_gates(
        _build({}), V7ModelAcceptanceGateConfig(min_sharpe=0.0)
    )
    for name in ("max_drawdown", "sharpe", "turnover_adjusted_net_return", "excess_return_after_costs"):
        gate = _gate(report, name)
        assert gate["status"] == GATE_UNKNOWN, (name, gate)
    assert _gate(report, "max_drawdown")["status"] != GATE_PASS


def test_measured_values_still_flow_through():
    built = _build({"max_drawdown": -0.05, "sharpe": 1.2, "turnover_adjusted_net_return": 0.03,
                    "excess_return_after_costs": 0.01, "excess_return": 0.01})
    assert built["max_drawdown"] == -0.05
    assert built["sharpe"] == 1.2
    assert built["turnover_adjusted_net_return"] == 0.03
    assert built["excess_return_after_costs"] == 0.01


def test_an_incomplete_benchmark_is_reported_as_incomplete():
    summary = {
        "excess_return": None,
        "excess_return_after_costs": None,
        "benchmark_return": None,
        "benchmark_status": "incomplete",
        "benchmark_sessions_covered": 5,
        "benchmark_sessions_missing": 5,
        "max_drawdown": -0.05,
        "sharpe": 1.0,
        "turnover_adjusted_net_return": 0.01,
    }
    report = evaluate_model_acceptance_gates(_build(summary, "000300.SH"), V7ModelAcceptanceGateConfig())
    gate = _gate(report, "excess_return_after_costs")
    assert gate["status"] == GATE_UNKNOWN
    assert gate["benchmark_status"] == "incomplete"
    assert gate["benchmark_sessions_missing"] == 5
