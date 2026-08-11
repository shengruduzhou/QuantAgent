from __future__ import annotations

import pytest

from quantagent.execution.tca import (
    TCAFill,
    TransactionCostAnalysisError,
    calculate_parent_tca,
)


def test_complete_buy_implementation_shortfall_includes_fees() -> None:
    result = calculate_parent_tca(
        parent_id="p-buy",
        side="buy",
        parent_quantity=1_000,
        arrival_price=10.0,
        fills=[
            TCAFill("c1", 400, 10.02, fees=2.0),
            TCAFill("c2", 600, 10.04, fees=3.0),
        ],
        market_vwap=10.03,
        market_volume=20_000,
    )
    assert result.execution_vwap == pytest.approx(10.032)
    assert result.gross_execution_shortfall_cash == pytest.approx(32.0)
    assert result.fees_cash == pytest.approx(5.0)
    assert result.opportunity_cost_cash == 0.0
    assert result.implementation_shortfall_cash == pytest.approx(37.0)
    assert result.implementation_shortfall_bps == pytest.approx(37.0)
    assert result.execution_vs_arrival_bps == pytest.approx(32.0)
    assert result.execution_vs_market_vwap_bps == pytest.approx(1.994017946, rel=1e-6)
    assert result.realized_participation_rate == pytest.approx(0.05)
    assert result.production_certified is False


def test_sell_shortfall_has_correct_sign() -> None:
    result = calculate_parent_tca(
        parent_id="p-sell",
        side="sell",
        parent_quantity=1_000,
        arrival_price=10.0,
        fills=[TCAFill("c1", 1_000, 9.95, fees=0.0)],
    )
    assert result.gross_execution_shortfall_cash == pytest.approx(50.0)
    assert result.implementation_shortfall_bps == pytest.approx(50.0)
    assert result.execution_vs_arrival_bps == pytest.approx(50.0)


def test_incomplete_parent_requires_terminal_mark_for_total_is() -> None:
    without_mark = calculate_parent_tca(
        parent_id="p-partial",
        side="buy",
        parent_quantity=1_000,
        arrival_price=10.0,
        fills=[TCAFill("c1", 400, 10.05, fees=2.0)],
    )
    assert without_mark.remaining_quantity == 600
    assert without_mark.opportunity_cost_cash is None
    assert without_mark.implementation_shortfall_cash is None
    assert without_mark.implementation_shortfall_bps is None

    marked = calculate_parent_tca(
        parent_id="p-partial",
        side="buy",
        parent_quantity=1_000,
        arrival_price=10.0,
        fills=[TCAFill("c1", 400, 10.05, fees=2.0)],
        terminal_price=10.10,
    )
    assert marked.gross_execution_shortfall_cash == pytest.approx(20.0)
    assert marked.opportunity_cost_cash == pytest.approx(60.0)
    assert marked.implementation_shortfall_cash == pytest.approx(82.0)
    assert marked.implementation_shortfall_bps == pytest.approx(82.0)


def test_zero_market_volume_with_fills_is_rejected() -> None:
    with pytest.raises(TransactionCostAnalysisError, match="zero market volume"):
        calculate_parent_tca(
            parent_id="p",
            side="buy",
            parent_quantity=100,
            arrival_price=10.0,
            fills=[TCAFill("c", 100, 10.0)],
            market_volume=0,
        )


def test_fill_over_parent_is_rejected() -> None:
    with pytest.raises(TransactionCostAnalysisError, match="exceed parent"):
        calculate_parent_tca(
            parent_id="p",
            side="buy",
            parent_quantity=100,
            arrival_price=10.0,
            fills=[TCAFill("c", 200, 10.0)],
        )
