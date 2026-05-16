"""Tests covering the live-readiness gate aggregation and CLI integration."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from quantagent.cli import app
from quantagent.data.v7_quality_gates import (
    V7ModelAcceptanceGateConfig,
    evaluate_model_acceptance_gates,
)


def _passing_metrics() -> dict[str, object]:
    return {
        "rank_ic_mean": 0.015,
        "rank_ic_stability": 0.55,
        "turnover_adjusted_net_return": 0.02,
        "max_drawdown": -0.10,
        "single_factor_dominance": 0.20,
        "adverse_regime_passed": True,
        "uses_mock_or_synthetic": False,
    }


def test_readiness_blocks_when_paper_report_missing(tmp_path):
    report = evaluate_model_acceptance_gates(
        _passing_metrics(),
        V7ModelAcceptanceGateConfig(),
        paper_report_path=tmp_path / "missing.json",
    )
    assert not report.passed
    assert "paper_trading_report_missing" in report.failures


def test_readiness_blocks_when_drawdown_too_large(tmp_path):
    metrics = _passing_metrics()
    metrics["max_drawdown"] = -0.40
    paper = tmp_path / "paper.json"
    paper.write_text("{}", encoding="utf-8")
    report = evaluate_model_acceptance_gates(metrics, paper_report_path=paper)
    assert not report.passed
    assert "max_drawdown_exceeded" in report.failures


def test_readiness_blocks_when_mock_data_detected(tmp_path):
    metrics = _passing_metrics()
    metrics["uses_mock_or_synthetic"] = True
    paper = tmp_path / "paper.json"
    paper.write_text("{}", encoding="utf-8")
    report = evaluate_model_acceptance_gates(metrics, paper_report_path=paper)
    assert not report.passed
    assert "mock_data_model_not_production_ready" in report.failures


def test_readiness_blocks_when_adverse_regime_missing(tmp_path):
    metrics = _passing_metrics()
    metrics["adverse_regime_passed"] = False
    paper = tmp_path / "paper.json"
    paper.write_text("{}", encoding="utf-8")
    report = evaluate_model_acceptance_gates(metrics, paper_report_path=paper)
    assert not report.passed
    assert "adverse_regime_not_validated" in report.failures


def test_readiness_passes_when_all_gates_satisfied(tmp_path):
    paper = tmp_path / "paper.json"
    paper.write_text("{}", encoding="utf-8")
    report = evaluate_model_acceptance_gates(_passing_metrics(), paper_report_path=paper)
    assert report.passed
    assert report.failures == ()


def test_cli_live_readiness_report_writes_safety_defaults(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_passing_metrics()), encoding="utf-8")
    paper = tmp_path / "paper.json"
    paper.write_text("{}", encoding="utf-8")
    output = tmp_path / "ready.json"
    runner = CliRunner()
    invocation = runner.invoke(
        app,
        [
            "v7-live-readiness-report",
            "--metrics",
            str(metrics_path),
            "--paper-report",
            str(paper),
            "--output",
            str(output),
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    payload = json.loads(Path(output).read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["safety_defaults"]["live_trading_enabled"] is False
    assert payload["safety_defaults"]["dry_run"] is True
    assert payload["safety_defaults"]["virtual_broker_only"] is True
