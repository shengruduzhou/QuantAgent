"""V8 CLI surface registration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from quantagent.cli import app


EXPECTED_V8_COMMANDS = {
    "ingest-policy-evidence-v8",
    "ingest-bond-flow-v8",
    "ingest-bank-financials-v8",
    "build-capital-flow-thesis-v8",
    "validate-capital-flow-thesis-v8",
    "build-sector-pool-v8",
    "build-fundamental-rank-v8",
    "build-technical-factors-v8",
    "train-horizon-models-v8",
    "optimize-ga-weights-v8",
    "build-target-weights-v8",
    "run-strict-a-share-backtest-v8",
    "run-paper-trading-v8",
    "generate-daily-decision-report-v8",
    "generate-risk-report-v8",
}


def _registered_names() -> set[str]:
    return {c.name for c in app.registered_commands}


def test_v8_commands_are_registered_on_main_app():
    found = _registered_names()
    missing = EXPECTED_V8_COMMANDS - found
    assert not missing, f"missing v8 commands: {missing}"


def test_v8_help_lists_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Typer prints commands in --help
    for name in EXPECTED_V8_COMMANDS:
        assert name in result.stdout, f"command {name} missing from CLI --help"


# ---------------------------------------------------------------------------
# Light end-to-end smoke for the new v8 commands
# ---------------------------------------------------------------------------

def test_build_capital_flow_thesis_v8_smoke(tmp_path):
    # write a tiny silver policy_events.parquet
    policy = pd.DataFrame(
        [
            {
                "event_id": f"p{i}",
                "source": "csrc",
                "url": f"https://csrc.gov.cn/{i}",
                "announced_at": pd.Timestamp("2024-03-01") + pd.Timedelta(days=i),
                "effective_at": pd.Timestamp("2024-03-01") + pd.Timedelta(days=i),
                "available_at": pd.Timestamp("2024-03-01") + pd.Timedelta(days=i, hours=1),
                "fetched_at": pd.Timestamp("2024-03-01") + pd.Timedelta(days=i, hours=2),
                "title": f"政策{i}",
                "body_summary": "",
                "themes": ["tech_innovation"],
                "sectors_hint": ["Semi"],
                "policy_strength": 0.7,
                "source_version": "v1",
            }
            for i in range(3)
        ]
    )
    policy_path = tmp_path / "policy_events.parquet"
    policy.to_parquet(policy_path, index=False)

    runner = CliRunner()
    out_root = tmp_path / "lake"
    result = runner.invoke(
        app,
        [
            "build-capital-flow-thesis-v8",
            "--policy-events", str(policy_path),
            "--output-root", str(out_root),
        ],
    )
    assert result.exit_code == 0, result.stdout
    thesis_path = out_root / "silver" / "capital_flow_thesis" / "capital_flow_thesis.parquet"
    assert thesis_path.exists()
    frame = pd.read_parquet(thesis_path)
    assert "thesis_id" in frame.columns
    assert len(frame) >= 1


def test_generate_risk_report_v8_smoke(tmp_path):
    events = [
        {"event_type": "order_skipped", "symbol": "600519.SH"},
        {"event_type": "order_rejected", "symbol": "000001.SZ"},
        {"event_type": "order_skipped", "symbol": "600519.SH"},
    ]
    events_path = tmp_path / "risk_events.json"
    events_path.write_text(json.dumps(events), encoding="utf-8")

    runner = CliRunner()
    out_md = tmp_path / "risk_report.md"
    result = runner.invoke(
        app,
        [
            "generate-risk-report-v8",
            "--risk-events-path", str(events_path),
            "--output-path", str(out_md),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out_md.exists()
    text = out_md.read_text(encoding="utf-8")
    assert "order_skipped" in text
    assert "600519.SH" in text


def test_ingest_bank_financials_v8_smoke(tmp_path):
    rows = pd.DataFrame([
        {"bank_code": "ICBC", "report_period": "2023-12-31",
         "available_at": "2024-03-30", "loans_total": 100.0},
        {"bank_code": "CCB", "report_period": "2023-12-31",
         "available_at": "2024-03-31", "loans_total": 90.0},
    ])
    src = tmp_path / "bank_in.csv"
    rows.to_csv(src, index=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["ingest-bank-financials-v8", "--raw-path", str(src),
         "--output-root", str(tmp_path / "lake")],
    )
    assert result.exit_code == 0, result.stdout
    out = tmp_path / "lake" / "silver" / "bank_financials" / "bank_financials.parquet"
    assert out.exists()


def test_build_target_weights_v8_smoke(tmp_path):
    dates = pd.bdate_range("2024-03-01", periods=3)
    preds = []
    for d in dates:
        for sym, score in (("A", 0.5), ("B", 0.3), ("C", 0.2)):
            preds.append({"trade_date": d, "symbol": sym, "alpha_score": score})
    src = tmp_path / "preds.csv"
    pd.DataFrame(preds).to_csv(src, index=False)
    runner = CliRunner()
    out_target = tmp_path / "weights.parquet"
    result = runner.invoke(
        app,
        ["build-target-weights-v8", "--predictions-path", str(src),
         "--output-path", str(out_target), "--top-k", "2"],
    )
    assert result.exit_code == 0, result.stdout
    assert out_target.exists()
    wide = pd.read_parquet(out_target)
    # Top-2 of 3 each day, equal weight 0.5 each
    assert (wide.sum(axis=1) <= 1.0 + 1e-9).all()


def test_generate_daily_decision_report_v8_smoke(tmp_path):
    runner = CliRunner()
    out_md = tmp_path / "daily.md"
    result = runner.invoke(
        app,
        [
            "generate-daily-decision-report-v8",
            "--as-of-date", "2024-03-01",
            "--market-regime", "normal",
            "--global-conviction", "0.7",
            "--gross-exposure", "0.55",
            "--output-path", str(out_md),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out_md.exists()
    text = out_md.read_text(encoding="utf-8")
    assert "Daily Decision Report" in text
    assert "normal" in text
