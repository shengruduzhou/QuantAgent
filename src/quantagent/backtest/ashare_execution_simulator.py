"""Strict A-share cash-account execution simulator facade.

This facade owns three non-negotiable production contracts:

* a cash stock account cannot establish naked negative stock weights;
* target-weight rows presented to this boundary are signal dated, never already
  delayed execution dates;
* signal-dated target weights cannot execute until the next market session.

The historical implementation lives in ``ashare_execution_simulator_impl.py``;
all public callers pass through the validators here.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from quantagent.backtest import ashare_execution_simulator_impl as _impl
from quantagent.backtest.execution_timing import (
    EXECUTION_TIMING_SEMANTICS,
    execution_trace_sha256,
    validate_execution_trace,
)
from quantagent.market_rules.tradability_flags import ensure_tradability_flags


STRICT_CASH_ACCOUNT_SEMANTICS = "ashare_cash_long_only_v1_no_naked_stock_short"
TARGET_INDEX_SIGNAL_DATE = "signal_date"

AShareExecutionSimulationConfig = _impl.AShareExecutionSimulationConfig
AShareExecutionSimulationResult = _impl.AShareExecutionSimulationResult


class UnsupportedStockShortError(ValueError):
    """Raised when cash-account target weights require a negative stock position."""


class ExecutionTimingViolation(ValueError):
    """Raised when target/trace timing cannot prove the strict signal-date contract."""


def validate_execution_market_panel(
    target_weight_history: pd.DataFrame | None,
    market_panel: pd.DataFrame | None,
) -> pd.DataFrame:
    """Fail closed unless every production fill input is measured and finite."""
    if target_weight_history is None or target_weight_history.empty:
        return market_panel.copy() if market_panel is not None else pd.DataFrame()
    if market_panel is None or market_panel.empty:
        raise ValueError("A-share execution requires a non-empty measured market panel")

    required = {
        "trade_date", "symbol", "close", "volume", "amount",
        "is_suspended", "is_st", "is_limit_up", "is_limit_down",
    }
    missing = sorted(required.difference(market_panel.columns))
    if missing:
        raise ValueError(
            "A-share execution requires measured execution fields; "
            f"missing {missing}"
        )

    out, _ = ensure_tradability_flags(market_panel, require_measured=True)
    trade_dates = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    symbols = out["symbol"].astype("string").str.strip()
    close = pd.to_numeric(out["close"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")
    amount = pd.to_numeric(out["amount"], errors="coerce")
    if trade_dates.isna().any():
        raise ValueError("A-share execution market panel has invalid trade_date values")
    if symbols.isna().any() or symbols.eq("").any():
        raise ValueError("A-share execution market panel has invalid symbol values")
    if not np.isfinite(close.to_numpy(dtype=float)).all() or (close <= 0).any():
        raise ValueError("A-share execution requires finite positive close prices")
    if not np.isfinite(volume.to_numpy(dtype=float)).all() or (volume < 0).any():
        raise ValueError(
            "A-share execution requires finite non-negative measured volume"
        )
    if not np.isfinite(amount.to_numpy(dtype=float)).all() or (amount < 0).any():
        raise ValueError(
            "A-share execution requires finite non-negative measured amount"
        )
    duplicate = out.assign(
        _trade_date=trade_dates,
        _symbol=symbols,
    ).duplicated(["_trade_date", "_symbol"], keep=False)
    if duplicate.any():
        raise ValueError(
            "A-share execution market panel contains duplicate symbol/session rows"
        )
    out["trade_date"] = trade_dates
    out["symbol"] = symbols.astype(str)
    out["close"] = close
    out["volume"] = volume
    out["amount"] = amount
    return out


def validate_signal_dated_target_weights(
    target_weight_history: pd.DataFrame | None,
) -> None:
    """Reject matrices that declare an already-delayed execution-date index.

    Generic callers written before index metadata existed remain supported when
    the attr is absent; builders that *do* know their index semantics must not be
    allowed to contradict the strict simulator.  In particular, a hold-band
    matrix with ``delay_days=1`` is execution dated and passing it here would
    otherwise turn one intended T+1 delay into T+2.
    """
    if target_weight_history is None:
        return
    semantics = target_weight_history.attrs.get("target_index_semantics")
    if semantics is None:
        return
    normalized = str(semantics).strip().lower()
    if normalized == TARGET_INDEX_SIGNAL_DATE:
        return
    raise ExecutionTimingViolation(
        "strict A-share simulator requires signal-dated target weights because "
        "it owns the sole T-close -> next-session execution mapping; got "
        f"target_index_semantics={semantics!r}"
    )


def validate_cash_account_target_weights(
    target_weight_history: pd.DataFrame | None,
    *,
    tolerance: float = 1e-12,
) -> None:
    """Fail closed if final stock targets require naked short positions."""
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
    """Run the public production-grade simulator and verify its timing trace."""
    validate_signal_dated_target_weights(target_weight_history)
    validate_cash_account_target_weights(target_weight_history)
    measured_panel = validate_execution_market_panel(
        target_weight_history,
        market_panel,
    )
    result = _impl.simulate_ashare_target_weights(
        target_weight_history,
        measured_panel,
        config,
    )
    metadata = dict(result.config or {})
    metadata["stock_shorting_capability"] = "cash_long_only"
    metadata["execution_semantics_version"] = STRICT_CASH_ACCOUNT_SEMANTICS
    metadata["execution_timing_semantics"] = EXECUTION_TIMING_SEMANTICS
    declared_semantics = target_weight_history.attrs.get("target_index_semantics")
    metadata["target_input_index_semantics"] = (
        str(declared_semantics) if declared_semantics is not None else "undeclared_legacy_signal_date"
    )

    if target_weight_history is not None and not target_weight_history.empty:
        timing = validate_execution_trace(result.execution_trace)
        metadata["execution_trace_ok"] = timing.ok
        metadata["execution_trace_reasons"] = list(timing.reasons)
        metadata["execution_trace_mapped_signal_days"] = timing.mapped_signal_days
        metadata["execution_trace_order_records"] = timing.order_records
        metadata["execution_trace_skip_records"] = timing.skip_records
        metadata["execution_trace_sha256"] = execution_trace_sha256(result.execution_trace)
        if not timing.ok:
            raise ExecutionTimingViolation(
                "production A-share execution timing could not be proven: "
                + "; ".join(timing.reasons)
            )
    else:
        metadata["execution_trace_ok"] = True
        metadata["execution_trace_reasons"] = []
        metadata["execution_trace_mapped_signal_days"] = 0
        metadata["execution_trace_order_records"] = 0
        metadata["execution_trace_skip_records"] = 0
        metadata["execution_trace_sha256"] = None

    return replace(result, config=metadata)


def __getattr__(name: str):
    return getattr(_impl, name)


__all__ = [
    "STRICT_CASH_ACCOUNT_SEMANTICS",
    "TARGET_INDEX_SIGNAL_DATE",
    "EXECUTION_TIMING_SEMANTICS",
    "UnsupportedStockShortError",
    "ExecutionTimingViolation",
    "AShareExecutionSimulationConfig",
    "AShareExecutionSimulationResult",
    "validate_signal_dated_target_weights",
    "validate_cash_account_target_weights",
    "validate_execution_market_panel",
    "simulate_ashare_target_weights",
]
