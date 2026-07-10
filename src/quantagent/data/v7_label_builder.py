"""Forward label generation for V7 model training.

NOTE (2026-07-10, EVALUATOR_VALIDITY_AUDIT_IC016.md): the PRODUCTION training
datasets (``*_exec_*``, incl. plus7clean/plus7clean_fund) do NOT use this
module's close(t)->close(t+h) convention. They are re-labelled by
``scripts/build_executable_labels_dataset.py`` to the DELAY-1 executable form
``close(t+1+h)/close(t+1) - 1`` with entry-infeasible rows dropped, and that
convention is regression-locked by tests/test_executable_label_convention.py.

This module emits three families of labels:

* ``forward_return_{h}d`` — raw close-to-close forward returns over a
  horizon of *h* trading bars.
* ``forward_excess_return_{h}d`` — cross-sectional excess vs the day's
  mean return (per ``trade_date``), used for ranking models.
* ``forward_rank_{h}d`` — per-date percentile rank of the raw label.

Optionally, ``forward_tradable_return_{h}d`` masks out horizons whose
exit bar would have been **untradable** (limit-up sell block, limit-down
buy block, suspended, ST). This avoids training the model to chase moves
it could not have captured under realistic A-share constraints.

All output columns sit alongside the original frame; nothing is dropped
unless every horizon's label is NaN. Labels are always future-looking
and must not be joined back into inference frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


V7_LABEL_HORIZONS: tuple[int, ...] = (1, 5, 20, 60, 120, 126)
TRADABILITY_BLOCK_COLUMNS: tuple[str, ...] = (
    "is_suspended",
    "is_st",
    "is_limit_up",
    "is_limit_down",
)


@dataclass(frozen=True)
class V7LabelBuildResult:
    frame: pd.DataFrame
    label_schema: dict[str, object]


def build_forward_return_labels(
    market_panel: pd.DataFrame,
    horizons: Iterable[int] = V7_LABEL_HORIZONS,
    price_column: str = "close",
    include_excess_return: bool = True,
    include_rank_label: bool = True,
    include_tradable_labels: bool = True,
    benchmark_column: str | None = None,
) -> V7LabelBuildResult:
    horizons = tuple(int(h) for h in horizons)
    if market_panel is None or market_panel.empty:
        return V7LabelBuildResult(pd.DataFrame(), {"horizons": list(horizons), "label_columns": []})
    required = {"symbol", "trade_date", price_column}
    missing = sorted(required - set(market_panel.columns))
    if missing:
        raise ValueError(f"market panel missing label columns: {missing}")

    data = market_panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "symbol", price_column]).sort_values(["symbol", "trade_date"])

    label_columns: list[str] = []
    excess_columns: list[str] = []
    rank_columns: list[str] = []
    tradable_columns: list[str] = []
    end_columns: list[str] = []

    grouped = data.groupby("symbol", sort=False)
    for horizon in horizons:
        label = f"forward_return_{horizon}d"
        end_col = f"label_end_{horizon}d"
        future_price = grouped[price_column].shift(-horizon)
        future_date = grouped["trade_date"].shift(-horizon)
        data[label] = future_price / data[price_column] - 1.0
        data[end_col] = future_date
        label_columns.append(label)
        end_columns.append(end_col)

        if include_excess_return:
            excess = f"forward_excess_return_{horizon}d"
            if benchmark_column and benchmark_column in data.columns:
                future_benchmark = grouped[benchmark_column].shift(-horizon)
                data[excess] = data[label] - (future_benchmark / data[benchmark_column] - 1.0)
            else:
                # Cross-sectional excess vs same-day mean keeps the label
                # cross-sectionally neutral even when no benchmark series
                # is available.
                date_mean = data.groupby("trade_date")[label].transform("mean")
                data[excess] = data[label] - date_mean
            excess_columns.append(excess)

        if include_rank_label:
            rank = f"forward_rank_{horizon}d"
            data[rank] = data.groupby("trade_date")[label].rank(pct=True)
            rank_columns.append(rank)

        if include_tradable_labels and any(col in data.columns for col in TRADABILITY_BLOCK_COLUMNS):
            tradable = f"forward_tradable_return_{horizon}d"
            untradable_at_exit = _untradable_at_exit(data, grouped, horizon)
            data[tradable] = data[label].where(~untradable_at_exit, other=np.nan)
            tradable_columns.append(tradable)

    drop_subset = label_columns
    data = data.dropna(subset=drop_subset, how="all").reset_index(drop=True)
    schema = {
        "horizons": list(horizons),
        "label_columns": label_columns,
        "excess_label_columns": excess_columns,
        "rank_label_columns": rank_columns,
        "tradable_label_columns": tradable_columns,
        "label_end_columns": end_columns,
        "price_column": price_column,
        "benchmark_column": benchmark_column,
        "pit_note": "labels are future outcomes for training only and must not be joined into inference frames",
    }
    return V7LabelBuildResult(data, schema)


def _untradable_at_exit(
    frame: pd.DataFrame,
    grouped: "pd.api.typing.DataFrameGroupBy",
    horizon: int,
) -> pd.Series:
    masks: list[pd.Series] = []
    for column in TRADABILITY_BLOCK_COLUMNS:
        if column not in frame.columns:
            continue
        series = frame[column].astype(bool)
        shifted = series.groupby(frame["symbol"]).shift(-horizon)
        masks.append(shifted.fillna(False))
    if not masks:
        return pd.Series(False, index=frame.index)
    combined = masks[0]
    for mask in masks[1:]:
        combined = combined | mask
    return combined
