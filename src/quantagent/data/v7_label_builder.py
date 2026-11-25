"""Forward label generation for V7 model training.

NOTE (2026-07-10, EVALUATOR_VALIDITY_AUDIT_IC016.md (removed in Round 31; git show 2c4e296:EVALUATOR_VALIDITY_AUDIT_IC016.md)): the PRODUCTION training
datasets (``*_exec_*``, incl. plus7clean/plus7clean_fund) do NOT use this
module's close(t)->close(t+h) convention. They are re-labelled by
``scripts/build_executable_labels_dataset.py`` to the DELAY-1 executable form
``close(t+1+h)/close(t+1) - 1`` with entry-infeasible rows dropped, and that
convention is regression-locked by tests/test_executable_label_convention.py.

Label convention (ROADMAP P1-2): ``entry_delay_sessions`` selects the clock.
``0`` (the historical default, kept so existing callers' numbers do not move)
is ``close(t+h)/close(t) - 1`` -- id ``same_close_t_to_close_t_plus_h_v1``;
``1`` is ``close(S[t+1+h])/close(S[t+1]) - 1`` on an explicit global session
clock ``S`` (``market_sessions``, required): the label is NaN unless the
symbol's next row IS session t+1 and its exit row IS session t+1+h, so a
suspension can never stretch the label span past h sessions (the purge/embargo
contract). Id ``delay1_exact_global_session_t1_to_t1_plus_h_v1``, the
``_exec_`` clock. The id is written to ``label_schema['label_convention']`` and
``frame.attrs['label_convention']``; persist with
``label_contract.write_labels_file`` so the file carries it too.
``build-labels-v7`` requires the choice explicitly.

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
    entry_delay_sessions: int = 0,
    market_sessions: Iterable[object] | None = None,
) -> V7LabelBuildResult:
    from quantagent.data.label_contract import (
        CONVENTION_TEXT,
        EXEC_DELAY1_SESSION,
        LABEL_CONVENTION_ATTR,
        V7_SAME_CLOSE,
    )

    horizons = tuple(int(h) for h in horizons)
    delay = int(entry_delay_sessions)
    if delay not in (0, 1):
        raise ValueError("entry_delay_sessions must be 0 (same close) or 1 (delay-1 executable)")
    convention = EXEC_DELAY1_SESSION if delay == 1 else V7_SAME_CLOSE
    sessions = None
    if delay:
        from quantagent.factors.executable_labels import canonical_market_sessions

        sessions = canonical_market_sessions(market_sessions)
    if market_panel is None or market_panel.empty:
        empty = pd.DataFrame()
        empty.attrs[LABEL_CONVENTION_ATTR] = convention
        return V7LabelBuildResult(
            empty,
            {"horizons": list(horizons), "label_columns": [], "label_convention": convention,
             "entry_delay_sessions": delay},
        )
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
    entry_price = grouped[price_column].shift(-delay) if delay else data[price_column]
    if delay:
        session_of = _session_offsets(data["trade_date"], sessions)
        entry_ok = grouped["trade_date"].shift(-delay).eq(session_of(delay))
    for horizon in horizons:
        label = f"forward_return_{horizon}d"
        end_col = f"label_end_{horizon}d"
        future_price = grouped[price_column].shift(-(delay + horizon))
        future_date = grouped["trade_date"].shift(-(delay + horizon))
        data[label] = future_price / entry_price - 1.0
        if delay:
            # Exact global-session clock: the symbol's next row must BE session
            # t+1 and its (1+h)-th next row must BE session t+1+h; otherwise a
            # suspension stretched the window and the label is unavailable.
            expected_exit = session_of(delay + horizon)
            exact = entry_ok & future_date.eq(expected_exit)
            data[label] = data[label].where(exact)
            future_date = expected_exit
        data[end_col] = future_date
        label_columns.append(label)
        end_columns.append(end_col)

        if include_excess_return:
            excess = f"forward_excess_return_{horizon}d"
            if benchmark_column and benchmark_column in data.columns:
                future_benchmark = grouped[benchmark_column].shift(-(delay + horizon))
                entry_benchmark = (
                    grouped[benchmark_column].shift(-delay) if delay else data[benchmark_column]
                )
                data[excess] = data[label] - (future_benchmark / entry_benchmark - 1.0)
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
            untradable = _untradable_at_exit(data, grouped, delay + horizon)
            if delay:
                untradable = untradable | _unbuyable_at_entry(data, delay)
            data[tradable] = data[label].where(~untradable, other=np.nan)
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
        "label_convention": convention,
        "label_convention_text": CONVENTION_TEXT[convention],
        "entry_delay_sessions": delay,
        "market_session_count": int(len(sessions)) if sessions is not None else None,
        "pit_note": "labels are future outcomes for training only and must not be joined into inference frames",
    }
    data.attrs[LABEL_CONVENTION_ATTR] = convention
    return V7LabelBuildResult(data, schema)


def _session_offsets(trade_dates: pd.Series, sessions: pd.DatetimeIndex):
    """``offset -> Series`` of the global session ``offset`` sessions after each row."""
    position_of = {session: idx for idx, session in enumerate(sessions)}
    positions = pd.to_datetime(trade_dates).dt.normalize().map(position_of)
    if positions.isna().any():
        bad = (
            pd.to_datetime(trade_dates[positions.isna()]).dt.strftime("%Y-%m-%d").drop_duplicates().head(5).tolist()
        )
        raise ValueError(f"market panel has trade dates absent from market_sessions: {bad}")
    base = positions.astype("int64").to_numpy()
    values = sessions.to_numpy(dtype="datetime64[ns]")

    def offset(k: int) -> pd.Series:
        target = base + int(k)
        out = np.full(len(target), np.datetime64("NaT"), dtype="datetime64[ns]")
        inside = target < len(values)
        out[inside] = values[target[inside]]
        return pd.Series(out, index=trade_dates.index)

    return offset


def _unbuyable_at_entry(frame: pd.DataFrame, delay: int) -> pd.Series:
    """Entry at ``t+delay`` is impossible when suspended or sealed limit-up there."""
    masks: list[pd.Series] = []
    for column in ("is_suspended", "is_limit_up"):
        if column not in frame.columns:
            continue
        shifted = frame[column].astype(bool).groupby(frame["symbol"]).shift(-delay)
        masks.append(shifted.fillna(False).astype(bool))
    if not masks:
        return pd.Series(False, index=frame.index)
    combined = masks[0]
    for mask in masks[1:]:
        combined = combined | mask
    return combined


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
