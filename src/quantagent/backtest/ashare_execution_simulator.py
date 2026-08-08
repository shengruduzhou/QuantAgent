"""Strict A-share cash-account execution simulator facade.

The historical implementation is retained in
``ashare_execution_simulator_impl.py``.  This facade makes one broker capability
explicit: the simulator models a long-only cash stock account and therefore
cannot establish a negative stock position.

Research code may still construct hypothetical long-short weights, but those
weights must not silently pass through a cash-account simulator where the short
orders are rejected one-by-one and the remaining long leg is then mistaken for
the intended market-neutral strategy.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from quantagent.backtest import ashare_execution_simulator_impl as _impl


STRICT_CASH_ACCOUNT_SEMANTICS = "ashare_cash_long_only_v1_no_naked_stock_short"

AShareExecutionSimulationConfig = _impl.AShareExecutionSimulationConfig
AShareExecutionSimulationResult = _impl.AShareExecutionSimulationResult


class UnsupportedStockShortError(ValueError):
    """Raised when cash-account target weights require a negative stock position."""


def validate_cash_account_target_weights(
    target_weight_history: pd.DataFrame | None,
    *,
    tolerance: float = 1e-12,
) -> None:
    """Fail closed if final stock targets require naked short positions.

    Zero targets remain valid and may liquidate an existing long position in an
    execution engine that carries holdings.  A negative *final* stock weight is
    different: it requires an explicit securities-lending/margin capability,
    borrow inventory and financing/recall economics that this simulator does not
    model.
    """
    if target_weight_history is None or target_weight_history.empty:
        return
    numeric = target_weight_history.apply(pd.to_numeric, errors="coerce")
    negative = numeric < -abs(float(tolerance))
    if not bool(negative.to_numpy().any()):
        return
    locations = np.argwhere(negative.to_numpy())
    samples: list[str] = []
    for row_idx, col_idx in locations[:5]:
        date = target_weight_history.index[int(row_idx)]
        symbol = target_weight_history.columns[int(col_idx)]
        value = numeric.iat[int(row_idx), int(col_idx)]
        samples.append(f"{pd.Timestamp(date).date()}:{symbol}={float(value):.6g}")
    suffix = ", ".join(samples)
    raise UnsupportedStockShortError(
        "strict A-share cash-account simulation cannot establish negative stock "
        "weights; use an explicit securities-lending/margin simulator with "
        "borrow inventory/fees/recall rules or a separately modelled index-futures "
        f"hedge. offending targets: {suffix}"
    )


def simulate_ashare_target_weights(
    target_weight_history: pd.DataFrame,
    market_panel: pd.DataFrame,
    config: AShareExecutionSimulationConfig | None = None,
) -> AShareExecutionSimulationResult:
    validate_cash_account_target_weights(target_weight_history)
    result = _impl.simulate_ashare_target_weights(
        target_weight_history,
        market_panel,
        config,
    )
    metadata = dict(result.config or {})
    metadata["stock_shorting_capability"] = "cash_long_only"
    metadata["execution_semantics_version"] = STRICT_CASH_ACCOUNT_SEMANTICS
    return replace(result, config=metadata)


def __getattr__(name: str):
    # Preserve compatibility for private forensic helpers while keeping the
    # public simulator boundary governed above.
    return getattr(_impl, name)


__all__ = [
    "STRICT_CASH_ACCOUNT_SEMANTICS",
    "UnsupportedStockShortError",
    "AShareExecutionSimulationConfig",
    "AShareExecutionSimulationResult",
    "validate_cash_account_target_weights",
    "simulate_ashare_target_weights",
]
