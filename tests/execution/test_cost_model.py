"""A-share cost model tests — commission, stamp tax, transfer, impact."""

from __future__ import annotations

import math

import pytest

from quantagent.execution.broker_base import OrderSide
from quantagent.execution.cost_model import AShareCostModel


def test_min_commission_applies_when_value_below_floor():
    m = AShareCostModel(min_commission=5.0, commission_rate=0.0003)
    # 1000 yuan * 0.0003 = 0.3 yuan → floor to 5
    out = m.calculate(OrderSide.BUY, quantity=10, price=100.0)
    assert out["commission"] == 5.0


def test_commission_scales_with_value_above_floor():
    m = AShareCostModel(min_commission=5.0, commission_rate=0.0003)
    # 100_000 * 0.0003 = 30
    out = m.calculate(OrderSide.BUY, quantity=1000, price=100.0)
    assert out["commission"] == pytest.approx(30.0)


def test_stamp_tax_applies_only_on_sell():
    m = AShareCostModel(stamp_tax_rate=0.0005)
    buy = m.calculate(OrderSide.BUY, quantity=1000, price=100.0)
    sell = m.calculate(OrderSide.SELL, quantity=1000, price=100.0)
    assert buy["stamp_duty"] == 0.0
    assert sell["stamp_duty"] == 100_000 * 0.0005  # 50


def test_transfer_fee_applies_both_sides():
    m = AShareCostModel(transfer_fee_rate=0.00001)
    buy = m.calculate(OrderSide.BUY, quantity=1000, price=100.0)
    sell = m.calculate(OrderSide.SELL, quantity=1000, price=100.0)
    assert buy["transfer_fee"] == pytest.approx(1.0)
    assert sell["transfer_fee"] == pytest.approx(1.0)


def test_impact_cost_zero_without_participation():
    m = AShareCostModel()
    out = m.calculate(OrderSide.BUY, quantity=1000, price=100.0)
    assert out["impact_cost"] == 0.0


def test_impact_cost_scales_with_sqrt_participation():
    m = AShareCostModel(impact_alpha_bps=10.0)
    # value 100_000, participation 4% → 10 * sqrt(0.04) = 2.0 bps → 20
    out = m.calculate(
        OrderSide.BUY, quantity=1000, price=100.0, participation_rate=0.04
    )
    assert out["impact_cost"] == pytest.approx(20.0)
    # participation 16% → 10 * sqrt(0.16) = 4.0 bps → 40
    out2 = m.calculate(
        OrderSide.BUY, quantity=1000, price=100.0, participation_rate=0.16
    )
    assert out2["impact_cost"] == pytest.approx(40.0)


def test_impact_cost_zero_when_alpha_disabled():
    m = AShareCostModel(impact_alpha_bps=0.0)
    out = m.calculate(
        OrderSide.BUY, quantity=1000, price=100.0, participation_rate=0.10
    )
    assert out["impact_cost"] == 0.0


def test_total_aggregates_every_component():
    m = AShareCostModel(
        commission_rate=0.0003,
        min_commission=5.0,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0.00001,
        impact_alpha_bps=10.0,
    )
    out = m.calculate(
        OrderSide.SELL, quantity=1000, price=100.0, participation_rate=0.04
    )
    # commission=30, stamp=50, transfer=1, impact=20
    assert out["total"] == pytest.approx(30 + 50 + 1 + 20)


def test_negative_participation_rate_clamped():
    m = AShareCostModel(impact_alpha_bps=10.0)
    out = m.calculate(
        OrderSide.BUY, quantity=1000, price=100.0, participation_rate=-0.10
    )
    assert out["impact_cost"] == 0.0
