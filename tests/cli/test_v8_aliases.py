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
    "summarize-v8-results",
    "optimize-regime-aware-ensemble-v8",
    "search-regime-factor-experts-v8",
    "build-llm-hybrid-stock-pool-v8",
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


def test_llm_stock_selection_fallback_keeps_full_ranking_pool():
    from quantagent.cli.v8 import _agent_scores_from_analysis, _fallback_stock_selection_analysis
    from quantagent.factors.core_policy import CORE_FACTOR_PRIOR_WEIGHTS

    rows = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2024-03-01"),
                "symbol": f"S{i:03d}",
                "prediction": float(100 - i),
                "model_rank": i + 1,
                "old_dealer_risk_score": 0.0,
                "old_dealer_block": False,
                "dip_buy_flow_score": 0.0,
            }
            for i in range(50)
        ]
    )

    analysis = _fallback_stock_selection_analysis(rows, CORE_FACTOR_PRIOR_WEIGHTS)
    scores = _agent_scores_from_analysis(rows, analysis)

    assert len(analysis["candidates"]) == 50
    assert len(scores) == 50
    assert "model_rank" in scores.columns
    assert scores["model_rank"].max() == 50
    assert "agent_rank" in scores.columns


def test_build_llm_hybrid_stock_pool_v8_smoke(tmp_path, monkeypatch):
    monkeypatch.delenv("QUANTAGENT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("QUANTAGENT_LLM_ENABLED", raising=False)
    monkeypatch.delenv("QUANTAGENT_LLM_ALLOW_NETWORK", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    date = pd.Timestamp("2024-03-01")
    symbols = [f"00000{i}.SZ" for i in range(1, 7)]
    preds = pd.DataFrame(
        [{"trade_date": date, "symbol": sym, "prediction": 1.0 - i * 0.1} for i, sym in enumerate(symbols)]
    )
    pred_path = tmp_path / "preds.parquet"
    preds.to_parquet(pred_path, index=False)
    core = pd.DataFrame(
        [
            {
                "trade_date": date,
                "symbol": sym,
                "core_policy_score": 0.2,
                "core_sentiment_score": 0.1,
                "fundamental_quality_score": 0.3 - i * 0.02,
                "cicc_stock_selection_score": 0.2,
                "sector_resonance_score": 0.15,
                "dip_buy_flow_score": 0.1,
                "trend_strength_score": 0.2,
                "old_dealer_risk_score": 0.05 if i < 5 else 0.8,
                "old_dealer_block": 0 if i < 5 else 1,
            }
            for i, sym in enumerate(symbols)
        ]
    )
    core_path = tmp_path / "core.parquet"
    core.to_parquet(core_path, index=False)
    sector = pd.DataFrame(
        [{"symbol": sym, "sector_level_1": "Semi" if i < 4 else "Bank"} for i, sym in enumerate(symbols)]
    )
    sector_path = tmp_path / "sector.parquet"
    sector.to_parquet(sector_path, index=False)
    policy = pd.DataFrame(
        [
            {
                "event_id": "p0",
                "source": "gov",
                "url": "https://example.gov/p0",
                "announced_at": pd.Timestamp("2024-02-29"),
                "available_at": pd.Timestamp("2024-02-29 10:00"),
                "fetched_at": pd.Timestamp("2024-02-29 10:01"),
                "title": "support semi",
                "body_summary": "support semi",
                "themes": ["theme:chip"],
                "sectors_hint": ["sector:Semi"],
                "policy_strength": 0.8,
            }
        ]
    )
    policy_path = tmp_path / "policy.parquet"
    policy.to_parquet(policy_path, index=False)
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-llm-hybrid-stock-pool-v8",
            "--predictions-path", str(pred_path),
            "--core-dataset-path", str(core_path),
            "--sector-map-path", str(sector_path),
            "--policy-events", str(policy_path),
            "--as-of-date", "2024-03-01",
            "--candidate-pool-size", "30",
            "--stock-top-n", "10",
            "--output-dir", str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    hybrid = pd.read_parquet(out_dir / "hybrid_stock_pool.parquet")
    scores = pd.read_parquet(out_dir / "agent_scores.parquet")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["used_fallback"] is True
    assert not hybrid.empty
    assert not scores.empty
    assert "hybrid_score" in hybrid.columns
    assert "agent_stock_score" in scores.columns


def test_build_llm_hybrid_stock_pool_v8_require_llm_fails_without_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("QUANTAGENT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("QUANTAGENT_LLM_ENABLED", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    date = pd.Timestamp("2024-03-01")
    pred_path = tmp_path / "preds.parquet"
    pd.DataFrame([{"trade_date": date, "symbol": "000001.SZ", "prediction": 1.0}]).to_parquet(pred_path, index=False)
    core_path = tmp_path / "core.parquet"
    pd.DataFrame([{"trade_date": date, "symbol": "000001.SZ", "fundamental_quality_score": 0.1}]).to_parquet(core_path, index=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-llm-hybrid-stock-pool-v8",
            "--predictions-path", str(pred_path),
            "--core-dataset-path", str(core_path),
            "--as-of-date", "2024-03-01",
            "--candidate-pool-size", "30",
            "--require-llm",
        ],
    )
    assert result.exit_code == 2
    assert "llm_required_but_fallback_used" in result.stderr


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
