from __future__ import annotations

import pytest

from quantagent.execution.tca import (
    TCAFill,
    TransactionCostAnalysisError,
    calculate_parent_tca,
)


def _fill(fill_id: str, child_id: str, quantity: int, price: float, *, fees: float = 0.0):
    return TCAFill(
        fill_id=fill_id,
        child_id=child_id,
        quantity=quantity,
        price=price,
        fees=fees,
    )


def test_complete_buy_implementation_shortfall_includes_fees() -> None:
    result = calculate_parent_tca(
        parent_id="p-buy",
        side="buy",
        parent_quantity=1_000,
        arrival_price=10.0,
        fills=[
            _fill("f1", "c1", 400, 10.02, fees=2.0),
            _fill("f2", "c2", 600, 10.04, fees=3.0),
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
    assert result.realized_participation_rate == pytest.approx(0.05)
    assert result.implementation_shortfall_complete is True
    assert result.production_certified is False


def test_sell_shortfall_uses_directionally_correct_sign() -> None:
    result = calculate_parent_tca(
        parent_id="p-sell",
        side="sell",
        parent_quantity=1_000,
        arrival_price=10.0,
        fills=[_fill("f1", "c1", 1_000, 9.95)],
    )
    assert result.gross_execution_shortfall_cash == pytest.approx(50.0)
    assert result.implementation_shortfall_bps == pytest.approx(50.0)


def test_price_improvement_is_negative_cost() -> None:
    buy = calculate_parent_tca(
        parent_id="buy-good",
        side="buy",
        parent_quantity=100,
        arrival_price=10.0,
        fills=[_fill("f1", "c", 100, 9.90)],
    )
    sell = calculate_parent_tca(
        parent_id="sell-good",
        side="sell",
        parent_quantity=100,
        arrival_price=10.0,
        fills=[_fill("f2", "c", 100, 10.10)],
    )
    assert buy.implementation_shortfall_cash == pytest.approx(-10.0)
    assert sell.implementation_shortfall_cash == pytest.approx(-10.0)


def test_incomplete_parent_without_terminal_mark_does_not_fake_zero_opportunity_cost() -> None:
    incomplete = calculate_parent_tca(
        parent_id="p-partial",
        side="buy",
        parent_quantity=1_000,
        arrival_price=10.0,
        fills=[_fill("f1", "c1", 400, 10.05, fees=2.0)],
    )
    assert incomplete.remaining_quantity == 600
    assert incomplete.opportunity_cost_cash is None
    assert incomplete.implementation_shortfall_cash is None
    assert incomplete.implementation_shortfall_complete is False

    marked = calculate_parent_tca(
        parent_id="p-partial",
        side="buy",
        parent_quantity=1_000,
        arrival_price=10.0,
        fills=[_fill("f1", "c1", 400, 10.05, fees=2.0)],
        terminal_price=10.10,
    )
    assert marked.gross_execution_shortfall_cash == pytest.approx(20.0)
    assert marked.opportunity_cost_cash == pytest.approx(60.0)
    assert marked.implementation_shortfall_cash == pytest.approx(82.0)
    assert marked.implementation_shortfall_bps == pytest.approx(82.0)
    assert marked.implementation_shortfall_complete is True


def test_zero_fill_parent_with_terminal_mark_has_complete_is_evidence() -> None:
    result = calculate_parent_tca(
        parent_id="p-none",
        side="buy",
        parent_quantity=1_000,
        arrival_price=10.0,
        fills=[],
        terminal_price=10.25,
    )
    assert result.execution_vwap is None
    assert result.opportunity_cost_cash == pytest.approx(250.0)
    assert result.implementation_shortfall_bps == pytest.approx(250.0)
    assert result.implementation_shortfall_complete is True


def test_duplicate_fill_id_is_rejected_even_if_other_fields_differ() -> None:
    with pytest.raises(TransactionCostAnalysisError, match="duplicate fill_id"):
        calculate_parent_tca(
            parent_id="p",
            side="buy",
            parent_quantity=500,
            arrival_price=10.0,
            fills=[
                _fill("same", "c1", 100, 10.0),
                _fill("same", "c1", 100, 10.01),
            ],
        )


def test_missing_fill_identity_is_rejected() -> None:
    with pytest.raises(TransactionCostAnalysisError, match="fill_id and child_id"):
        calculate_parent_tca(
            parent_id="p",
            side="buy",
            parent_quantity=100,
            arrival_price=10.0,
            fills=[_fill("", "c", 100, 10.0)],
        )


def test_market_volume_evidence_must_be_economically_consistent() -> None:
    with pytest.raises(TransactionCostAnalysisError, match="zero market volume"):
        calculate_parent_tca(
            parent_id="p",
            side="buy",
            parent_quantity=100,
            arrival_price=10.0,
            fills=[_fill("f", "c", 100, 10.0)],
            market_volume=0,
        )
    with pytest.raises(TransactionCostAnalysisError, match="exceeds observed market volume"):
        calculate_parent_tca(
            parent_id="p",
            side="buy",
            parent_quantity=500,
            arrival_price=10.0,
            fills=[_fill("f", "c", 200, 10.0)],
            market_volume=100,
        )


def test_fill_over_parent_is_rejected() -> None:
    with pytest.raises(TransactionCostAnalysisError, match="exceed parent"):
        calculate_parent_tca(
            parent_id="p",
            side="buy",
            parent_quantity=100,
            arrival_price=10.0,
            fills=[_fill("f", "c", 200, 10.0)],
        )
