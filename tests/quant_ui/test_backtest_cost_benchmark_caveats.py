"""Round-29 R8-F08: every backtest row states its cost basis and benchmark identity."""

import json

from services.quant_api.adapters.backtests import (
    COST_BASIS_ALL_FILLS,
    COST_BASIS_PRE_FIX,
    BacktestAdapter,
    _cost_and_benchmark,
)
from services.quant_api.runtime_indexer import RuntimeIndexer


def test_unlabelled_total_cost_is_read_as_the_pre_fix_fees_only_basis():
    out = _cost_and_benchmark({"total_cost": 64823.97}, has_benchmark_nav=False)
    assert out["totalCost"] == 64823.97
    assert out["totalCostBasis"] == COST_BASIS_PRE_FIX
    assert out["slippageCost"] is None
    assert out["benchmark"] is None


def test_post_fix_costs_carry_slippage_and_the_all_fills_basis():
    strict = _cost_and_benchmark(
        {"total_cost": 748971.0, "total_cost_basis": COST_BASIS_ALL_FILLS, "slippage_cost": 300000.0},
        has_benchmark_nav=False,
    )
    assert strict["totalCostBasis"] == COST_BASIS_ALL_FILLS
    assert strict["slippageCost"] == 300000.0
    exported = _cost_and_benchmark({"total_cost_cny": 1200.5, "slippage_cost_cny": 400.25}, has_benchmark_nav=False)
    assert exported["totalCost"] == 1200.5
    assert exported["totalCostBasis"] == COST_BASIS_ALL_FILLS


def test_missing_cost_is_not_recorded_rather_than_zero():
    out = _cost_and_benchmark({}, has_benchmark_nav=False)
    assert out["totalCost"] is None and out["totalCostBasis"] is None


def test_baseline_protocol_benchmark_is_named_universe_equal_weight_with_caveat():
    out = _cost_and_benchmark(
        {"benchmark_annualized_return": 0.522753, "benchmark_nav_gap_sessions": 0},
        has_benchmark_nav=True,
    )
    bench = out["benchmark"]
    assert bench["mode"] == "universe_equal_weight"
    assert bench["source"] == "baseline_protocol_export"
    assert "untradeable" in bench["caveat"] and "overstated" in bench["caveat"]
    assert bench["annualizedReturn"] == 0.522753


def test_unlabelled_benchmark_column_is_flagged():
    bench = _cost_and_benchmark({}, has_benchmark_nav=True)["benchmark"]
    assert bench["mode"] == "unlabelled"
    assert bench["caveat"]


def test_adapter_rows_expose_cost_basis_and_benchmark(quant_ui_settings):
    items = BacktestAdapter(quant_ui_settings, RuntimeIndexer(quant_ui_settings)).list()
    assert items
    for item in items:
        assert "totalCostBasis" in item and "benchmark" in item and "slippageCost" in item
        if item["totalCost"] is None:
            assert item["totalCostBasis"] is None
    json.dumps(items)
