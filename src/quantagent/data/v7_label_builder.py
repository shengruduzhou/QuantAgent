"""Forward label generation for V7 model training."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


V7_LABEL_HORIZONS: tuple[int, ...] = (1, 5, 20, 60, 120, 126)


@dataclass(frozen=True)
class V7LabelBuildResult:
    frame: pd.DataFrame
    label_schema: dict[str, object]


def build_forward_return_labels(
    market_panel: pd.DataFrame,
    horizons: tuple[int, ...] = V7_LABEL_HORIZONS,
    price_column: str = "close",
) -> V7LabelBuildResult:
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
    for horizon in horizons:
        label = f"forward_return_{horizon}d"
        end_col = f"label_end_{horizon}d"
        future_price = data.groupby("symbol", sort=False)[price_column].shift(-horizon)
        future_date = data.groupby("symbol", sort=False)["trade_date"].shift(-horizon)
        data[label] = future_price / data[price_column] - 1.0
        data[end_col] = future_date
        label_columns.append(label)
    data = data.dropna(subset=label_columns, how="all").reset_index(drop=True)
    schema = {
        "horizons": list(horizons),
        "label_columns": label_columns,
        "price_column": price_column,
        "pit_note": "labels are future outcomes for training only and must not be joined into inference frames",
    }
    return V7LabelBuildResult(data, schema)
