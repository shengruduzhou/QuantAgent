"""Tests for the Stage 5.4 per-trade post-mortem analyser."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quantagent.diagnostics.post_mortem import (
    PerTradePostMortem,
    analyze_blotter,
    analyze_trade,
    write_post_mortem_reports,
)


def _good_trace(alpha: float = 0.50, setup: str = "breakout") -> dict:
    return {
        "candidate_id": "2024-03-01|A.SH",
        "trade_date": "2024-03-01",
        "symbol": "A.SH",
        "final_decision": "eligible",
        "failed_gate": None,
        "gate_results": [
            {"gate_name": "alpha_threshold", "passed": True, "reason": "passed", "detail": {"alpha": alpha}},
            {"gate_name": "liquidity", "passed": True, "reason": "passed", "detail": {"amount_cny": 200_000_000}},
            {"gate_name": "regime_alignment", "passed": True, "reason": "passed", "detail": {"regime": "normal", "setup": setup}},
            {"gate_name": "fundamental_filter", "passed": True, "reason": "passed", "detail": {"composite_rank": 0.80}},
            {"gate_name": "policy_aligned", "passed": True, "reason": "passed", "detail": {"sector": "Bank", "signal": 0.30}},
            {"gate_name": "broker_consensus", "passed": True, "reason": "passed", "detail": {"score": 0.40}},
            {"gate_name": "drawdown_kill", "passed": True, "reason": "passed", "detail": {"dd_20d": -0.05}},
            {"gate_name": "concentration_limit", "passed": True, "reason": "passed", "detail": {"proposed": 0.15}},
            {"gate_name": "risk_budget", "passed": True, "reason": "passed", "detail": {"target_weight": 0.02}},
        ],
    }


# ---------------------------------------------------------------------------
# analyze_trade
# ---------------------------------------------------------------------------

def test_analyze_trade_realized_excess_computed_correctly():
    pm = analyze_trade(
        trade_id="t1",
        entry_date=pd.Timestamp("2024-03-01"),
        exit_date=pd.Timestamp("2024-04-01"),
        symbol="A.SH",
        entry_price=50.0,
        exit_price=55.0,
        benchmark_entry_price=4000.0,
        benchmark_exit_price=4080.0,
        entry_decision_trace=_good_trace(),
    )
    assert pm.realized_pnl_pct == pytest.approx(0.10)
    assert pm.benchmark_return_pct == pytest.approx(0.02)
    assert pm.excess_return_pct == pytest.approx(0.08)
    assert pm.holding_days == 31


def test_attribution_alpha_uses_entry_alpha():
    pm = analyze_trade(
        trade_id="t1",
        entry_date=pd.Timestamp("2024-03-01"),
        exit_date=pd.Timestamp("2024-03-08"),
        symbol="A.SH",
        entry_price=100.0,
        exit_price=105.0,
        benchmark_entry_price=4000.0,
        benchmark_exit_price=4040.0,
        entry_decision_trace=_good_trace(alpha=0.03),
    )
    # realized = 0.05; market = 0.01 ((4040/4000)-1); alpha = 0.03; residual = 0.05 - 0.01 - 0.03 = 0.01
    assert pm.attribution_alpha == pytest.approx(0.03)
    assert pm.attribution_market == pytest.approx(0.01)
    assert pm.attribution_residual == pytest.approx(0.01)


def test_setup_label_extracted_from_trace():
    pm = analyze_trade(
        trade_id="t1",
        entry_date=pd.Timestamp("2024-03-01"),
        exit_date=pd.Timestamp("2024-03-08"),
        symbol="A.SH",
        entry_price=100.0, exit_price=100.0,
        benchmark_entry_price=4000.0, benchmark_exit_price=4000.0,
        entry_decision_trace=_good_trace(setup="lowbuy"),
    )
    assert pm.setup_label == "lowbuy"


def test_nearest_passing_gate_identifies_thin_margin():
    # Override one gate to be very close to its threshold
    trace = _good_trace()
    for g in trace["gate_results"]:
        if g["gate_name"] == "drawdown_kill":
            g["detail"]["dd_20d"] = -0.18  # very close to the -0.20 kill threshold
    pm = analyze_trade(
        trade_id="t1",
        entry_date=pd.Timestamp("2024-03-01"),
        exit_date=pd.Timestamp("2024-03-08"),
        symbol="A.SH",
        entry_price=100.0, exit_price=100.0,
        benchmark_entry_price=4000.0, benchmark_exit_price=4000.0,
        entry_decision_trace=trace,
    )
    assert pm.nearest_passing_gate == "drawdown_kill"
    assert pm.nearest_passing_margin < 0.20


def test_negative_alpha_prediction_underperform_creates_note():
    trace = _good_trace(alpha=0.10)
    pm = analyze_trade(
        trade_id="t1",
        entry_date=pd.Timestamp("2024-03-01"),
        exit_date=pd.Timestamp("2024-04-01"),
        symbol="A.SH",
        entry_price=100.0,
        exit_price=98.0,            # realized -2%
        benchmark_entry_price=4000.0,
        benchmark_exit_price=4040.0,  # benchmark +1%
        entry_decision_trace=trace,
    )
    assert "alpha_predicted_positive_but_underperformed_bench" in pm.notes


def test_realized_loss_over_10pct_creates_note():
    pm = analyze_trade(
        trade_id="t1",
        entry_date=pd.Timestamp("2024-03-01"),
        exit_date=pd.Timestamp("2024-03-08"),
        symbol="A.SH",
        entry_price=100.0, exit_price=85.0,
        benchmark_entry_price=4000.0, benchmark_exit_price=4000.0,
        entry_decision_trace=_good_trace(),
    )
    assert "realized_loss_exceeded_10pct" in pm.notes


def test_to_dict_is_json_serialisable():
    pm = analyze_trade(
        trade_id="t1",
        entry_date=pd.Timestamp("2024-03-01"),
        exit_date=pd.Timestamp("2024-03-08"),
        symbol="A.SH",
        entry_price=100.0, exit_price=110.0,
        benchmark_entry_price=4000.0, benchmark_exit_price=4040.0,
        entry_decision_trace=_good_trace(),
    )
    json.dumps(pm.to_dict())


# ---------------------------------------------------------------------------
# analyze_blotter
# ---------------------------------------------------------------------------

def test_analyze_blotter_minimal():
    blotter = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "symbol": "A.SH",
                "entry_date": pd.Timestamp("2024-03-01"),
                "exit_date": pd.Timestamp("2024-03-08"),
                "entry_price": 100.0,
                "exit_price": 105.0,
            }
        ]
    )
    out = analyze_blotter(blotter)
    assert len(out) == 1
    assert out[0].realized_pnl_pct == pytest.approx(0.05)
    # benchmark missing → bench = 0
    assert out[0].benchmark_return_pct == 0.0


def test_analyze_blotter_with_benchmark_prices():
    blotter = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "symbol": "A.SH",
                "entry_date": pd.Timestamp("2024-03-01"),
                "exit_date": pd.Timestamp("2024-03-08"),
                "entry_price": 100.0, "exit_price": 105.0,
            }
        ]
    )
    bench = pd.Series(
        [4000.0, 4020.0, 4040.0],
        index=pd.bdate_range("2024-03-01", periods=3),
    )
    out = analyze_blotter(blotter, benchmark_prices=bench)
    # Approximate excess: (105/100 - 1) - (some bench return between 4000 and 4040)
    assert out[0].benchmark_return_pct != 0.0
    assert out[0].excess_return_pct != out[0].realized_pnl_pct


def test_analyze_blotter_with_decision_traces():
    blotter = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "symbol": "A.SH",
                "entry_date": pd.Timestamp("2024-03-01"),
                "exit_date": pd.Timestamp("2024-03-08"),
                "entry_price": 100.0, "exit_price": 105.0,
            }
        ]
    )
    traces_long = pd.DataFrame(
        [
            {
                "candidate_id": "2024-03-01|A.SH",
                "trade_date": pd.Timestamp("2024-03-01"),
                "symbol": "A.SH",
                "final_decision": "eligible",
                "failed_gate": None,
                "gate_name": "alpha_threshold",
                "gate_passed": True,
                "gate_reason": "passed",
            },
            {
                "candidate_id": "2024-03-01|A.SH",
                "trade_date": pd.Timestamp("2024-03-01"),
                "symbol": "A.SH",
                "final_decision": "eligible",
                "failed_gate": None,
                "gate_name": "liquidity",
                "gate_passed": True,
                "gate_reason": "passed",
            },
        ]
    )
    out = analyze_blotter(blotter, decision_traces=traces_long)
    assert out[0].entry_decision_trace["final_decision"] == "eligible"
    assert len(out[0].entry_decision_trace["gate_results"]) == 2


def test_analyze_blotter_missing_columns_raises():
    bad = pd.DataFrame([{"trade_id": "t1"}])
    with pytest.raises(ValueError, match="missing columns"):
        analyze_blotter(bad)


def test_analyze_blotter_empty():
    assert analyze_blotter(pd.DataFrame()) == []


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def test_writer_emits_per_trade_json_plus_summary(tmp_path):
    blotter = pd.DataFrame(
        [
            {"trade_id": "t1", "symbol": "A.SH", "entry_date": pd.Timestamp("2024-03-01"),
             "exit_date": pd.Timestamp("2024-03-08"), "entry_price": 100.0, "exit_price": 105.0},
            {"trade_id": "t2", "symbol": "B.SH", "entry_date": pd.Timestamp("2024-03-05"),
             "exit_date": pd.Timestamp("2024-03-12"), "entry_price": 50.0, "exit_price": 47.5},
        ]
    )
    post_mortems = analyze_blotter(blotter)
    summary = write_post_mortem_reports(post_mortems, tmp_path)
    # Per-trade JSON files
    assert (tmp_path / "trades" / "t1.json").exists()
    assert (tmp_path / "trades" / "t2.json").exists()
    # Summary CSV
    assert (tmp_path / "summary.csv").exists()
    # Aggregate JSON
    aggregate = json.loads((tmp_path / "aggregate_summary.json").read_text())
    assert aggregate["n_trades"] == 2
    assert aggregate["win_rate_realized"] == 0.5  # t1 wins, t2 loses
    assert summary["n_trades"] == 2


def test_writer_summary_contains_attribution_columns(tmp_path):
    blotter = pd.DataFrame(
        [
            {"trade_id": "t1", "symbol": "A.SH", "entry_date": pd.Timestamp("2024-03-01"),
             "exit_date": pd.Timestamp("2024-03-08"), "entry_price": 100.0, "exit_price": 105.0},
        ]
    )
    post_mortems = analyze_blotter(blotter)
    write_post_mortem_reports(post_mortems, tmp_path)
    df = pd.read_csv(tmp_path / "summary.csv")
    for col in (
        "realized_pnl_pct",
        "benchmark_return_pct",
        "excess_return_pct",
        "attribution_alpha",
        "attribution_market",
        "attribution_residual",
        "nearest_passing_gate",
        "nearest_passing_margin",
    ):
        assert col in df.columns
