"""Executable forward-return labels for governed factor evaluation.

The production A-share timing contract is explicit: a signal observed at the
close of session T can only enter at the close of the next market session.  A
factor that is promoted using close(T)->close(T+h) outcomes is therefore scored
on economics the strict simulator cannot execute.

This module is deliberately separate from ``factors.evaluation``'s historical
same-close helper.  Research callers may still use the legacy helper when they
name that choice explicitly; governed factor promotion uses the executable
contract here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.factors.evaluation import DecayResult, information_coefficient


FACTOR_LABEL_SEMANTICS = EXECUTION_TIMING_SEMANTICS
FACTOR_LABEL_SCHEMA_VERSION = "factor_executable_labels_v1"


@dataclass(frozen=True)
class ExecutableLabelBuildResult:
    frame: pd.DataFrame
    schema: dict[str, object]


def build_executable_forward_returns(
    frame: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 5, 10, 20),
    *,
    price_column: str = "close",
    entry_delay_sessions: int = 1,
    date_column: str = "trade_date",
    symbol_column: str = "symbol",
) -> ExecutableLabelBuildResult:
    """Build ``entry(T+d)->exit(T+d+h)`` returns per symbol.

    ``entry_delay_sessions=1`` is the only governed production setting because
    it matches :data:`EXECUTION_TIMING_SEMANTICS`.  Other positive delays are
    available for explicitly labelled research sensitivity analysis, but their
    schema does not claim production timing equivalence.
    """

    horizons = tuple(sorted({int(value) for value in horizons}))
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons must contain positive integers")
    if int(entry_delay_sessions) < 1:
        raise ValueError("entry_delay_sessions must be >= 1")
    required = {date_column, symbol_column, price_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"executable factor labels missing columns: {missing}")

    data = frame.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[price_column] = pd.to_numeric(data[price_column], errors="coerce")
    data = data.dropna(subset=[date_column, symbol_column]).sort_values(
        [symbol_column, date_column]
    )
    if data.duplicated([date_column, symbol_column]).any():
        raise ValueError("executable factor labels require unique trade_date/symbol rows")

    grouped = data.groupby(symbol_column, sort=False)
    entry_delay = int(entry_delay_sessions)
    entry_price = grouped[price_column].shift(-entry_delay)
    entry_date = grouped[date_column].shift(-entry_delay)
    data["factor_label_entry_date"] = entry_date

    label_columns: list[str] = []
    end_columns: list[str] = []
    for horizon in horizons:
        exit_shift = entry_delay + int(horizon)
        exit_price = grouped[price_column].shift(-exit_shift)
        exit_date = grouped[date_column].shift(-exit_shift)
        label = f"forward_executable_return_{horizon}d"
        end = f"factor_label_end_{horizon}d"
        valid = (
            entry_price.notna()
            & exit_price.notna()
            & np.isfinite(entry_price)
            & np.isfinite(exit_price)
            & (entry_price > 0)
            & (exit_price > 0)
        )
        data[label] = (exit_price / entry_price - 1.0).where(valid)
        data[end] = exit_date
        label_columns.append(label)
        end_columns.append(end)

    governed_semantics = (
        FACTOR_LABEL_SEMANTICS
        if entry_delay == 1
        else f"research_signal_t_close_entry_delay_{entry_delay}_sessions"
    )
    schema = {
        "schema_version": FACTOR_LABEL_SCHEMA_VERSION,
        "execution_timing_semantics": governed_semantics,
        "entry_delay_sessions": entry_delay,
        "horizons": list(horizons),
        "label_columns": label_columns,
        "label_end_columns": end_columns,
        "price_column": price_column,
        "pit_note": "future outcomes for evaluation only; never inference features",
    }
    return ExecutableLabelBuildResult(frame=data, schema=schema)


def executable_factor_decay_curve(
    frame: pd.DataFrame,
    factor_column: str,
    horizons: Iterable[int] = (1, 3, 5, 10, 20),
    *,
    price_column: str = "close",
    entry_delay_sessions: int = 1,
) -> DecayResult:
    """Rank-IC decay under the same executable entry delay as strict backtests."""

    horizons_tuple = tuple(sorted({int(value) for value in horizons}))
    built = build_executable_forward_returns(
        frame,
        horizons_tuple,
        price_column=price_column,
        entry_delay_sessions=entry_delay_sessions,
    )
    ic_values: dict[int, float] = {}
    rank_values: dict[int, float] = {}
    for horizon in horizons_tuple:
        result = information_coefficient(
            built.frame,
            factor_column,
            f"forward_executable_return_{horizon}d",
        )
        ic_values[horizon] = result.summary.mean_ic
        rank_values[horizon] = result.summary.mean_rank_ic
    return DecayResult(
        horizon_days=horizons_tuple,
        rank_ic=pd.Series(rank_values, dtype=float),
        ic=pd.Series(ic_values, dtype=float),
    )


__all__ = [
    "FACTOR_LABEL_SEMANTICS",
    "FACTOR_LABEL_SCHEMA_VERSION",
    "ExecutableLabelBuildResult",
    "build_executable_forward_returns",
    "executable_factor_decay_curve",
]
