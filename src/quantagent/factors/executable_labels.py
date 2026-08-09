"""Executable forward-return labels for governed factor evaluation.

The production A-share timing contract is explicit: a signal observed at the
close of market session T can only enter on a later **global market session**.
That clock is not the same thing as "the next row available for this symbol".
If a stock is suspended, delisted, or its bar is missing on mapped T+1, a
strictly governed label is unavailable; silently shifting to the stock's next
observed row would invent a variable execution delay the simulator does not use.

Governed labels therefore require an explicit, validated market-session
schedule.  Per-symbol data rows are exact-looked-up on the mapped entry/exit
sessions.  The schedule identity is hashed into the schema so lifecycle evidence
can bind to the same clock used to construct its outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.factors.evaluation import DecayResult, information_coefficient


FACTOR_LABEL_SEMANTICS = EXECUTION_TIMING_SEMANTICS
FACTOR_LABEL_SCHEMA_VERSION = "factor_executable_labels_v2_global_sessions"


@dataclass(frozen=True)
class ExecutableLabelBuildResult:
    frame: pd.DataFrame
    schema: dict[str, object]


def canonical_market_sessions(values: Iterable[object] | None) -> pd.DatetimeIndex:
    """Validate and canonicalise an explicit ordered market-session schedule.

    The input must already represent exchange sessions in chronological order.
    This helper deliberately does not synthesize weekdays or infer a calendar
    from the factor panel: both would be wrong around exchange holidays and
    sparse/suspended symbols.
    """

    if values is None:
        raise ValueError(
            "governed executable factor labels require explicit market_sessions; "
            "do not infer the execution clock from per-symbol bars"
        )
    raw = list(values)
    if not raw:
        raise ValueError("market_sessions must not be empty")
    parsed = pd.to_datetime(pd.Series(raw, dtype="object"), errors="coerce")
    if parsed.isna().any():
        bad = [str(raw[i]) for i, is_bad in enumerate(parsed.isna().tolist()) if is_bad][:5]
        raise ValueError(f"market_sessions contain invalid dates: {bad}")
    sessions = pd.DatetimeIndex(parsed)
    if sessions.tz is not None:
        # Preserve the market-local calendar date rather than converting it to
        # UTC and potentially moving a midnight timestamp to the prior date.
        sessions = sessions.tz_localize(None)
    sessions = sessions.normalize()
    if sessions.has_duplicates:
        duplicates = sessions[sessions.duplicated()].strftime("%Y-%m-%d").tolist()[:5]
        raise ValueError(f"market_sessions contain duplicate sessions: {duplicates}")
    if not sessions.is_monotonic_increasing:
        raise ValueError("market_sessions must be strictly increasing")
    return sessions


def market_session_schedule_sha256(values: Iterable[object] | None) -> str:
    sessions = canonical_market_sessions(values)
    material = "\n".join(sessions.strftime("%Y-%m-%d")) + "\n"
    return sha256(material.encode("utf-8")).hexdigest()


def _exact_price_lookup(
    data: pd.DataFrame,
    *,
    symbol_column: str,
    date_column: str,
    price_column: str,
    mapped_dates: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Exact same-symbol/session price lookup without next-observation fallback."""

    prices = data.set_index([symbol_column, date_column])[price_column]
    keys = pd.MultiIndex.from_arrays(
        [data[symbol_column].to_numpy(), mapped_dates.to_numpy()],
        names=[symbol_column, date_column],
    )
    looked_up = prices.reindex(keys)
    result = pd.Series(looked_up.to_numpy(dtype=float), index=data.index, dtype=float)
    observed = pd.Series(keys.isin(prices.index), index=data.index, dtype=bool)
    # NaT cannot be a valid mapped session even if MultiIndex membership ever
    # changes behavior across pandas versions.
    observed &= mapped_dates.notna()
    return result, observed


def _mapped_session_dates(
    session_positions: pd.Series,
    sessions: pd.DatetimeIndex,
    offset: int,
) -> pd.Series:
    target_positions = session_positions.to_numpy(dtype=np.int64) + int(offset)
    valid = (target_positions >= 0) & (target_positions < len(sessions))
    out = np.full(len(target_positions), np.datetime64("NaT"), dtype="datetime64[ns]")
    if bool(valid.any()):
        out[valid] = sessions.to_numpy(dtype="datetime64[ns]")[target_positions[valid]]
    return pd.Series(pd.to_datetime(out), index=session_positions.index)


def build_executable_forward_returns(
    frame: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 5, 10, 20),
    *,
    price_column: str = "close",
    entry_delay_sessions: int = 1,
    date_column: str = "trade_date",
    symbol_column: str = "symbol",
    market_sessions: Iterable[object] | None = None,
) -> ExecutableLabelBuildResult:
    """Build exact ``entry(T+d)->exit(T+d+h)`` returns on a global session clock.

    ``market_sessions`` is mandatory even when ``entry_delay_sessions`` differs
    from the governed value.  A sensitivity analysis still needs a defined
    exchange-session clock; per-symbol row shifts are never a valid substitute.

    Missing exact entry/exit rows or non-positive/invalid prices produce ``NaN``
    outcomes.  The intended mapped dates are retained in the frame so callers can
    distinguish right-censoring from missing symbol observations.
    """

    horizons = tuple(sorted({int(value) for value in horizons}))
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons must contain positive integers")
    entry_delay = int(entry_delay_sessions)
    if entry_delay < 1:
        raise ValueError("entry_delay_sessions must be >= 1")
    sessions = canonical_market_sessions(market_sessions)
    calendar_digest = market_session_schedule_sha256(sessions)

    required = {date_column, symbol_column, price_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"executable factor labels missing columns: {missing}")

    data = frame.copy()
    parsed_dates = pd.to_datetime(data[date_column], errors="coerce")
    if getattr(parsed_dates.dt, "tz", None) is not None:
        parsed_dates = parsed_dates.dt.tz_localize(None)
    data[date_column] = parsed_dates.dt.normalize()
    data[symbol_column] = data[symbol_column].astype(str)
    data[price_column] = pd.to_numeric(data[price_column], errors="coerce")
    data = data.dropna(subset=[date_column, symbol_column]).sort_values(
        [symbol_column, date_column]
    ).reset_index(drop=True)
    if data.duplicated([date_column, symbol_column]).any():
        raise ValueError("executable factor labels require unique trade_date/symbol rows")

    position_of = {session: idx for idx, session in enumerate(sessions)}
    positions = data[date_column].map(position_of)
    if positions.isna().any():
        examples = (
            data.loc[positions.isna(), date_column]
            .drop_duplicates()
            .dt.strftime("%Y-%m-%d")
            .head(5)
            .tolist()
        )
        raise ValueError(
            "factor frame contains trade dates absent from market_sessions: "
            f"{examples}"
        )
    session_positions = positions.astype(np.int64)

    entry_date = _mapped_session_dates(session_positions, sessions, entry_delay)
    entry_price, entry_observed = _exact_price_lookup(
        data,
        symbol_column=symbol_column,
        date_column=date_column,
        price_column=price_column,
        mapped_dates=entry_date,
    )
    data["factor_label_entry_date"] = entry_date
    data["factor_label_entry_observed"] = entry_observed

    label_columns: list[str] = []
    end_columns: list[str] = []
    observation_columns: list[str] = ["factor_label_entry_observed"]
    for horizon in horizons:
        exit_date = _mapped_session_dates(
            session_positions,
            sessions,
            entry_delay + int(horizon),
        )
        exit_price, exit_observed = _exact_price_lookup(
            data,
            symbol_column=symbol_column,
            date_column=date_column,
            price_column=price_column,
            mapped_dates=exit_date,
        )
        label = f"forward_executable_return_{horizon}d"
        end = f"factor_label_end_{horizon}d"
        observed_col = f"factor_label_end_observed_{horizon}d"
        valid = (
            entry_observed
            & exit_observed
            & entry_price.notna()
            & exit_price.notna()
            & np.isfinite(entry_price)
            & np.isfinite(exit_price)
            & (entry_price > 0)
            & (exit_price > 0)
        )
        data[label] = (exit_price / entry_price - 1.0).where(valid)
        data[end] = exit_date
        data[observed_col] = exit_observed
        label_columns.append(label)
        end_columns.append(end)
        observation_columns.append(observed_col)

    governed_semantics = (
        FACTOR_LABEL_SEMANTICS
        if entry_delay == 1
        else f"research_signal_t_close_entry_delay_{entry_delay}_global_sessions"
    )
    schema = {
        "schema_version": FACTOR_LABEL_SCHEMA_VERSION,
        "execution_timing_semantics": governed_semantics,
        "entry_delay_sessions": entry_delay,
        "horizons": list(horizons),
        "label_columns": label_columns,
        "label_end_columns": end_columns,
        "observation_columns": observation_columns,
        "price_column": price_column,
        "market_session_count": int(len(sessions)),
        "market_session_first": sessions[0].date().isoformat(),
        "market_session_last": sessions[-1].date().isoformat(),
        "market_session_schedule_sha256": calendar_digest,
        "missing_bar_semantics": (
            "exact mapped global session required; missing symbol entry/exit bar => NaN, never shift"
        ),
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
    market_sessions: Iterable[object] | None = None,
) -> DecayResult:
    """Rank-IC decay under the same explicit global clock as strict backtests."""

    horizons_tuple = tuple(sorted({int(value) for value in horizons}))
    built = build_executable_forward_returns(
        frame,
        horizons_tuple,
        price_column=price_column,
        entry_delay_sessions=entry_delay_sessions,
        market_sessions=market_sessions,
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
    "canonical_market_sessions",
    "market_session_schedule_sha256",
    "build_executable_forward_returns",
    "executable_factor_decay_curve",
]
