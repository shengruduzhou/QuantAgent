from __future__ import annotations

import json

import pandas as pd
import pytest


def _large_predictions(count: int, value_mode: str = "ranked") -> tuple[pd.DataFrame, pd.DataFrame]:
    date = pd.Timestamp("2026-01-02")
    rows = []
    market = []
    for idx in range(count):
        symbol = f"{idx:06d}.SZ"
        prediction = 1.0 if value_mode == "flat" else float(idx)
        rows.append({"symbol": symbol, "trade_date": date, "prediction": prediction})
        market.append(
            {
                "symbol": symbol,
                "trade_date": date,
                "close": 10.0 + idx * 0.01,
                "amount": 10_000_000.0,
                "is_suspended": False,
                "is_st": False,
                "is_limit_up": False,
                "is_limit_down": False,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(market)


def test_top_k_ratio_limits_large_universe_to_ten_percent():
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

    preds, market = _large_predictions(100)
    result = build_v7_target_weights(
        preds,
        market,
        config=V7TargetWeightsConfig(selection_mode="top_k", top_k=30, top_k_ratio=0.10),
    )

    daily = result.diagnostics["daily_selection"][0]
    assert daily["selected_count"] == 10
    assert daily["effective_top_k"] == 10
    assert daily["selection_pressure"] == 10.0


def test_top_k_covering_universe_fails_when_ratio_disabled():
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

    preds, market = _large_predictions(30)
    with pytest.raises(ValueError, match="covers the eligible universe"):
        build_v7_target_weights(
            preds,
            market,
            config=V7TargetWeightsConfig(selection_mode="top_k", top_k=30, top_k_ratio=None, fail_if_top_k_covers_universe=True),
        )


def test_small_universe_ratio_selects_three_not_thirty():
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

    preds, market = _large_predictions(30)
    result = build_v7_target_weights(
        preds,
        market,
        config=V7TargetWeightsConfig(selection_mode="top_k", top_k=30, top_k_ratio=0.10),
    )

    assert result.diagnostics["daily_selection"][0]["selected_count"] == 3


def test_selection_pressure_below_threshold_fails():
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

    preds, market = _large_predictions(30)
    with pytest.raises(ValueError, match="selection_pressure"):
        build_v7_target_weights(
            preds,
            market,
            config=V7TargetWeightsConfig(selection_mode="top_k", top_k=30, top_k_ratio=0.50, min_selection_pressure=3.0),
        )


def test_selected_alpha_spread_warning_when_selection_has_no_alpha_edge():
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

    preds, market = _large_predictions(10, value_mode="flat")
    result = build_v7_target_weights(
        preds,
        market,
        config=V7TargetWeightsConfig(selection_mode="top_k", top_k=2, top_k_ratio=0.20, min_selection_pressure=1.0),
    )

    warnings = result.diagnostics.get("warnings", [])
    assert any(row.get("warning") == "selected_alpha_not_above_unselected_alpha" for row in warnings)


def test_paper_report_separates_generation_and_quant_acceptance(tmp_path):
    from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig, simulate_ashare_target_weights
    from quantagent.backtest.paper_report import PaperReportConfig, write_paper_report

    market = pd.DataFrame(
        [
            {"trade_date": "2026-01-02", "symbol": "000001.SZ", "close": 10.0, "volume": 1_000_000},
            {"trade_date": "2026-01-05", "symbol": "000001.SZ", "close": 10.2, "volume": 1_000_000},
        ]
    )
    weights = pd.DataFrame({"000001.SZ": [0.10, 0.0]}, index=pd.to_datetime(["2026-01-02", "2026-01-05"]))
    sim = simulate_ashare_target_weights(
        weights,
        market,
        AShareExecutionSimulationConfig(initial_cash=1_000_000, min_order_value_yuan=100.0),
    )
    write_paper_report(sim, market_panel=market, config=PaperReportConfig(output_dir=tmp_path))

    payload = json.loads((tmp_path / "paper_report.json").read_text(encoding="utf-8"))
    assert payload["report_generation_status"] == "passed"
    assert payload["status"] == "report_generation_passed"
    assert payload["quant_acceptance_status"] == "not_evaluated"
    assert "skipped_orders" in payload["files"]


def test_benchmark_missing_fails_production_acceptance(tmp_path):
    from quantagent.data.v7_quality_gates import evaluate_model_acceptance_gates

    paper = tmp_path / "paper.json"
    paper.write_text("{}", encoding="utf-8")
    report = evaluate_model_acceptance_gates(
        {
            "rank_ic_mean": 0.01,
            "rank_ic_stability": 0.5,
            "turnover_adjusted_net_return": 0.02,
            "max_drawdown": -0.10,
            "single_factor_dominance": 0.2,
            "adverse_regime_passed": True,
            "excess_return_after_costs": 0.01,
            "selection_pressure_min": 5.0,
            "training_dataset_symbol_count": 100,
            "prediction_symbol_count": 100,
            "eligible_symbol_count_min": 100,
            "uses_mock_or_synthetic": False,
            "pit_violation_count": 0,
        },
        paper_report_path=paper,
    )

    assert not report.passed
    assert "benchmark_missing_quant_alpha_not_validated" in report.failures
    assert report.gates


def test_order_manager_skips_small_orders_and_invalid_odd_lot_rebalances():
    from quantagent.execution.broker_base import Position
    from quantagent.execution.order_manager import OrderManager, OrderManagerConfig
    from quantagent.execution.virtual_broker import VirtualBroker

    broker = VirtualBroker(initial_cash=1_000_000, dry_run=True)
    manager = OrderManager(broker=broker, config=OrderManagerConfig(min_order_value_yuan=5_000.0))
    prices = pd.Series({"000001.SZ": 10.0})
    small_buy = manager.target_weights_to_order_intents(pd.Series({"000001.SZ": 0.001}), prices, 1_000_000)
    assert small_buy == []
    assert manager.last_skipped_orders[-1]["reason"] == "skipped_small_order"

    broker.ledger.positions["000001.SZ"] = Position("000001.SZ", 50, 0, 10.0)
    no_full_liquidation = manager.target_weights_to_order_intents(pd.Series({"000001.SZ": 0.0001}), prices, 1_000_000)
    assert no_full_liquidation == []
    assert manager.last_skipped_orders[-1]["reason"] == "skipped_not_full_odd_lot_liquidation"

    full_liquidation = manager.target_weights_to_order_intents(pd.Series({"000001.SZ": 0.0}), prices, 1_000_000)
    assert len(full_liquidation) == 1
    assert full_liquidation[0].quantity == 50
