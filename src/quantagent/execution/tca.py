"""Transaction-cost analysis for parent/child execution evidence.

The primary benchmark is implementation shortfall (arrival price). For a buy,
execution above arrival is a cost; for a sell, execution below arrival is a
cost. Fees are always costs. If a parent is incomplete and a terminal mark is
provided, the unfilled quantity contributes opportunity cost versus arrival.

This is an evidence calculator, not an execution optimiser. It intentionally
reports benchmark availability and never upgrades paper/shadow results to
production certification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable


class TransactionCostAnalysisError(ValueError):
    """TCA inputs are incomplete or economically inconsistent."""


@dataclass(frozen=True, slots=True)
class TCAFill:
    child_id: str
    quantity: int
    price: float
    fees: float = 0.0
    timestamp: str = ""


@dataclass(frozen=True, slots=True)
class ParentTCAResult:
    parent_id: str
    side: str
    parent_quantity: int
    filled_quantity: int
    remaining_quantity: int
    completion_ratio: float
    arrival_price: float
    execution_vwap: float | None
    market_vwap: float | None
    terminal_price: float | None
    gross_execution_shortfall_cash: float
    fees_cash: float
    opportunity_cost_cash: float | None
    implementation_shortfall_cash: float | None
    implementation_shortfall_bps: float | None
    execution_vs_arrival_bps: float | None
    execution_vs_market_vwap_bps: float | None
    realized_participation_rate: float | None
    production_certified: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _positive_finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise TransactionCostAnalysisError(f"{name} must be finite and > 0")
    return number


def _optional_positive_finite(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    return _positive_finite(name, value)


def _signed_cost(side: str, observed: float, benchmark: float) -> float:
    # Positive means worse execution / cost for both sides.
    return (observed - benchmark) if side == "buy" else (benchmark - observed)


def calculate_parent_tca(
    *,
    parent_id: str,
    side: str,
    parent_quantity: int,
    arrival_price: float,
    fills: Iterable[TCAFill],
    market_vwap: float | None = None,
    terminal_price: float | None = None,
    market_volume: int | None = None,
) -> ParentTCAResult:
    """Calculate arrival-price implementation shortfall and supporting metrics.

    ``terminal_price`` is required for a complete implementation-shortfall cash
    figure when the parent remains unfilled. Without it, realized execution cost
    is still reported but total IS is explicitly unavailable rather than treating
    the unfilled quantity as zero cost.
    """

    parent = str(parent_id).strip()
    if not parent:
        raise TransactionCostAnalysisError("parent_id must be non-empty")
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"buy", "sell"}:
        raise TransactionCostAnalysisError("side must be buy or sell")
    quantity = int(parent_quantity)
    if quantity <= 0:
        raise TransactionCostAnalysisError("parent_quantity must be positive")
    arrival = _positive_finite("arrival_price", arrival_price)
    benchmark_vwap = _optional_positive_finite("market_vwap", market_vwap)
    terminal = _optional_positive_finite("terminal_price", terminal_price)

    rows = list(fills)
    filled = 0
    notional = 0.0
    fees = 0.0
    for fill in rows:
        fill_quantity = int(fill.quantity)
        fill_price = _positive_finite("fill.price", fill.price)
        fill_fees = float(fill.fees)
        if fill_quantity <= 0:
            raise TransactionCostAnalysisError("fill quantity must be positive")
        if not math.isfinite(fill_fees) or fill_fees < 0.0:
            raise TransactionCostAnalysisError("fill fees must be finite and >= 0")
        filled += fill_quantity
        notional += fill_quantity * fill_price
        fees += fill_fees
    if filled > quantity:
        raise TransactionCostAnalysisError("fills exceed parent quantity")

    remaining = quantity - filled
    completion = filled / quantity
    execution_vwap = notional / filled if filled else None
    gross_execution_shortfall = (
        _signed_cost(normalized_side, execution_vwap, arrival) * filled
        if execution_vwap is not None
        else 0.0
    )
    execution_vs_arrival_bps = (
        _signed_cost(normalized_side, execution_vwap, arrival) / arrival * 10_000.0
        if execution_vwap is not None
        else None
    )
    execution_vs_market_vwap_bps = (
        _signed_cost(normalized_side, execution_vwap, benchmark_vwap)
        / benchmark_vwap
        * 10_000.0
        if execution_vwap is not None and benchmark_vwap is not None
        else None
    )

    opportunity_cost: float | None
    implementation_shortfall: float | None
    implementation_shortfall_bps: float | None
    if remaining == 0:
        opportunity_cost = 0.0
        implementation_shortfall = gross_execution_shortfall + fees
    elif terminal is None:
        opportunity_cost = None
        implementation_shortfall = None
    else:
        opportunity_cost = _signed_cost(normalized_side, terminal, arrival) * remaining
        implementation_shortfall = gross_execution_shortfall + fees + opportunity_cost

    if implementation_shortfall is None:
        implementation_shortfall_bps = None
    else:
        implementation_shortfall_bps = (
            implementation_shortfall / (arrival * quantity) * 10_000.0
        )

    if market_volume is None:
        participation = None
    else:
        volume = int(market_volume)
        if volume < 0:
            raise TransactionCostAnalysisError("market_volume must be >= 0")
        if volume == 0 and filled > 0:
            raise TransactionCostAnalysisError(
                "positive fills with zero market volume are inconsistent"
            )
        participation = filled / volume if volume else 0.0

    return ParentTCAResult(
        parent_id=parent,
        side=normalized_side,
        parent_quantity=quantity,
        filled_quantity=filled,
        remaining_quantity=remaining,
        completion_ratio=completion,
        arrival_price=arrival,
        execution_vwap=execution_vwap,
        market_vwap=benchmark_vwap,
        terminal_price=terminal,
        gross_execution_shortfall_cash=gross_execution_shortfall,
        fees_cash=fees,
        opportunity_cost_cash=opportunity_cost,
        implementation_shortfall_cash=implementation_shortfall,
        implementation_shortfall_bps=implementation_shortfall_bps,
        execution_vs_arrival_bps=execution_vs_arrival_bps,
        execution_vs_market_vwap_bps=execution_vs_market_vwap_bps,
        realized_participation_rate=participation,
        production_certified=False,
    )


__all__ = [
    "ParentTCAResult",
    "TCAFill",
    "TransactionCostAnalysisError",
    "calculate_parent_tca",
]
