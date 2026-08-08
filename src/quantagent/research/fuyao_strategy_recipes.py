"""Fuyao / Financial-API best-practice strategy recipes 13-15.

The implementation is kept in ``fuyao_strategy_recipes_impl``.  This facade
preserves the public API while installing one accounting invariant shared by
all recipes: dates with an empty target book are explicit zero-weight/cash
rows, not silently dropped dates.  Keeping those rows is required for warm-up
periods, cash-state diagnostics, T+1 alignment, and downstream backtests.
"""

from __future__ import annotations

import pandas as pd

from quantagent.research import fuyao_strategy_recipes_impl as _impl


def _dict_weights(
    daily: dict[pd.Timestamp, dict[str, float]],
    symbols: object,
) -> pd.DataFrame:
    columns = sorted({str(symbol) for symbol in symbols})
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(daily.keys()))))
    frame = pd.DataFrame.from_dict(daily, orient="index")
    frame.index = pd.to_datetime(frame.index)
    # DataFrame.from_dict drops entries whose value is an empty dict. Reindex
    # to the declared trading calendar so a genuine 100%-cash state remains
    # observable instead of disappearing from the strategy history.
    frame = frame.reindex(index=dates, columns=columns).fillna(0.0)
    frame.index.name = "trade_date"
    return frame.sort_index()


# The recipe functions are defined in the implementation module and resolve
# helpers from that module at call time, so replace its helper once here.
_impl._dict_weights = _dict_weights

BreakoutConfig = _impl.BreakoutConfig
MomentumConfig = _impl.MomentumConfig
RecipeResult = _impl.RecipeResult
ReversalConfig = _impl.ReversalConfig
price_volume_breakout_weights = _impl.price_volume_breakout_weights
short_term_reversal_weights = _impl.short_term_reversal_weights
time_series_momentum_weights = _impl.time_series_momentum_weights

__all__ = [
    "BreakoutConfig",
    "MomentumConfig",
    "RecipeResult",
    "ReversalConfig",
    "price_volume_breakout_weights",
    "short_term_reversal_weights",
    "time_series_momentum_weights",
]
