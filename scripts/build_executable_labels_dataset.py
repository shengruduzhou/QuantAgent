#!/usr/bin/env python3
"""Rebuild the v8 training dataset under the canonical executable-label clock.

Production label semantics are deliberately identical to the governed factor and
Stage-4 research contract:

    signal observed at close(T)
      -> enter on the next *global market session* T+1
      -> H-session outcome ends on global session T+1+H

"Global market session" is not the same as the next row available for one
symbol. A suspended/delisted/missing symbol bar on the mapped entry/exit session
makes the outcome unavailable; it must never shift to the symbol's next observed
row. The market clock therefore comes from an explicit independent calendar
artifact (U0 PIT calendar by default), whose identity is written into a manifest.

Execution-critical signal/entry flags also fail closed. Unknown suspension/ST/
limit-up state is not interpreted as tradable.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.factors.executable_labels import (  # noqa: E402
    FACTOR_LABEL_SEMANTICS,
    build_executable_forward_returns,
    canonical_market_sessions,
    market_session_schedule_sha256,
)

DEFAULT_INPUT = REPO / "runtime/data/v7/training/training_dataset_v8_ensemble.parquet"
DEFAULT_OUTPUT = REPO / "runtime/data/v7/training/training_dataset_v8_ensemble_exec.parquet"
DEFAULT_MARKET_CALENDAR = REPO / "runtime/data/u0/pit/trading_calendar.parquet"
CRITICAL_FLAGS = ("is_suspended", "is_st", "is_limit_up")
LABEL_SCHEMA = "quantagent.training-executable-labels.v3_global_sessions"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported tabular format: {path}")


def _load_market_sessions(path: Path) -> tuple[pd.DatetimeIndex, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"market calendar does not exist: {path}; executable labels require an independent global session clock"
        )
    frame = _read_table(path)
    if frame.empty:
        raise ValueError(f"market calendar is empty: {path}")
    date_column = next(
        (name for name in ("trade_date", "calendar_date", "date") if name in frame.columns),
        None,
    )
    if date_column is None:
        if len(frame.columns) != 1:
            raise ValueError(
                "market calendar requires trade_date/calendar_date/date or exactly one date column"
            )
        date_column = str(frame.columns[0])
    work = frame.copy()
    if "is_trading_day" in work.columns:
        flag = work["is_trading_day"].astype(str).str.strip().str.lower()
        work = work[flag.isin({"1", "true", "yes"})]
    sessions = canonical_market_sessions(work[date_column].tolist())
    return sessions, {
        "path": str(path.resolve()),
        "input_sha256": _sha256(path),
        "date_column": date_column,
        "session_count": int(len(sessions)),
        "market_session_schedule_sha256": market_session_schedule_sha256(sessions),
    }


def _coerce_flag(series: pd.Series) -> pd.Series:
    """Parse bool-like values while preserving unknowns as nullable NA."""
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna() & numeric.isin([0, 1])
    result.loc[numeric_mask] = numeric.loc[numeric_mask].astype(int).astype(bool)
    text = series.astype("string").str.strip().str.lower()
    true_mask = text.isin({"true", "t", "yes", "y", "1"})
    false_mask = text.isin({"false", "f", "no", "n", "0"})
    result.loc[true_mask] = True
    result.loc[false_mask] = False
    return result


def _explicitly_clear(series: pd.Series) -> pd.Series:
    parsed = _coerce_flag(series)
    return parsed.notna() & ~parsed.fillna(True).astype(bool)


def _exact_entry_state(frame: pd.DataFrame) -> pd.DataFrame:
    state = frame.set_index(["symbol", "trade_date"])[list(CRITICAL_FLAGS)]
    keys = pd.MultiIndex.from_arrays(
        [frame["symbol"].astype(str).to_numpy(), frame["factor_label_entry_date"].to_numpy()],
        names=["symbol", "trade_date"],
    )
    looked = state.reindex(keys)
    out = frame.copy()
    for column in CRITICAL_FLAGS:
        out[f"{column}_t1"] = looked[column].to_numpy()
    return out


def _mapped_dates(
    frame: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    offset: int,
) -> pd.Series:
    position_of = {session: idx for idx, session in enumerate(sessions)}
    positions = frame["trade_date"].map(position_of)
    if positions.isna().any():
        examples = (
            frame.loc[positions.isna(), "trade_date"]
            .drop_duplicates()
            .dt.strftime("%Y-%m-%d")
            .head(5)
            .tolist()
        )
        raise ValueError(f"training rows contain dates absent from market calendar: {examples}")
    target = positions.astype(np.int64).to_numpy() + int(offset)
    valid = (target >= 0) & (target < len(sessions))
    result = np.full(len(frame), np.datetime64("NaT"), dtype="datetime64[ns]")
    if bool(valid.any()):
        result[valid] = sessions.to_numpy(dtype="datetime64[ns]")[target[valid]]
    return pd.Series(pd.to_datetime(result), index=frame.index)


def _exact_price_lookup(
    frame: pd.DataFrame,
    mapped_dates: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    prices = frame.set_index(["symbol", "trade_date"])["close"]
    keys = pd.MultiIndex.from_arrays(
        [frame["symbol"].astype(str).to_numpy(), mapped_dates.to_numpy()],
        names=["symbol", "trade_date"],
    )
    looked = prices.reindex(keys)
    values = pd.Series(pd.to_numeric(looked, errors="coerce").to_numpy(), index=frame.index, dtype=float)
    observed = pd.Series(keys.isin(prices.index), index=frame.index, dtype=bool) & mapped_dates.notna()
    return values, observed


def _exact_holding_period_min_close(
    frame: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    holding_sessions: int,
) -> tuple[pd.Series, pd.Series]:
    """Minimum close on T+1..T+holding_sessions, all exact global sessions."""
    minimum = pd.Series(np.inf, index=frame.index, dtype=float)
    all_observed = pd.Series(True, index=frame.index, dtype=bool)
    for offset in range(1, int(holding_sessions) + 1):
        mapped = _mapped_dates(frame, sessions, offset)
        price, observed = _exact_price_lookup(frame, mapped)
        valid = observed & price.notna() & np.isfinite(price) & (price > 0)
        all_observed &= valid
        minimum = pd.Series(
            np.minimum(minimum.to_numpy(), price.fillna(np.inf).to_numpy()),
            index=frame.index,
            dtype=float,
        )
    return minimum.where(all_observed), all_observed


def build_executable_training_labels(
    source: pd.DataFrame,
    *,
    horizons: Iterable[int],
    market_sessions: Iterable[object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    horizons_tuple = tuple(sorted({int(value) for value in horizons}))
    if not horizons_tuple or any(value <= 0 for value in horizons_tuple):
        raise ValueError("horizons must contain positive integers")
    missing = sorted({"trade_date", "symbol", "close", *CRITICAL_FLAGS} - set(source.columns))
    if missing:
        raise ValueError(
            "production executable labels require close plus explicit execution-critical flags; "
            f"missing={missing}"
        )

    sessions = canonical_market_sessions(market_sessions)
    built = build_executable_forward_returns(
        source,
        horizons=horizons_tuple,
        price_column="close",
        entry_delay_sessions=1,
        market_sessions=sessions,
    )
    df = built.frame.copy()
    df = _exact_entry_state(df)

    signal_clear = pd.Series(True, index=df.index, dtype=bool)
    entry_clear = pd.Series(True, index=df.index, dtype=bool)
    for column in CRITICAL_FLAGS:
        signal_clear &= _explicitly_clear(df[column])
        entry_clear &= _explicitly_clear(df[f"{column}_t1"])
    close_valid = pd.to_numeric(df["close"], errors="coerce").gt(0)
    entry_observed = df["factor_label_entry_observed"].fillna(False).astype(bool)
    df["_signal_tradable"] = signal_clear & close_valid
    df["_entry_tradable"] = entry_clear & entry_observed
    df["_execution_tradable"] = df["_signal_tradable"] & df["_entry_tradable"]

    for horizon in horizons_tuple:
        canonical = f"forward_executable_return_{horizon}d"
        compatible = f"forward_return_{horizon}d"
        df[compatible] = pd.to_numeric(df[canonical], errors="coerce").where(
            df["_execution_tradable"]
        )

    if 5 in horizons_tuple:
        future_min, path_observed = _exact_holding_period_min_close(
            df,
            sessions,
            holding_sessions=5,
        )
        entry_close, exact_entry_observed = _exact_price_lookup(df, df["factor_label_entry_date"])
        dd_valid = (
            df["_execution_tradable"]
            & path_observed
            & exact_entry_observed
            & entry_close.notna()
            & np.isfinite(entry_close)
            & (entry_close > 0)
        )
        df["forward_max_drawdown_5d"] = (future_min / entry_close - 1.0).where(dd_valid)

    if "published_at" in df.columns:
        published = pd.to_datetime(df["published_at"], errors="coerce")
        df["has_fundamentals"] = (
            df["published_at"].notna()
            & published.notna()
            & (published <= pd.to_datetime(df["trade_date"]))
        ).astype(int)
    else:
        df["has_fundamentals"] = 0

    primary = "forward_return_5d" if "forward_return_5d" in df.columns else f"forward_return_{horizons_tuple[0]}d"
    df["_sample_weight"] = np.where(df[primary].notna(), 1.0, 0.0)
    df["_sample_weight"] *= np.where(df["_execution_tradable"], 1.0, 0.0)
    if "adv20_cny" in df.columns:
        liq = pd.to_numeric(df["adv20_cny"], errors="coerce").clip(lower=0)
        scale = float(liq.replace(0, np.nan).median())
        if np.isfinite(scale) and scale > 0:
            df["_sample_weight"] *= np.sqrt((liq / scale).clip(lower=0.25, upper=4.0)).fillna(0.0)

    schema = {
        "schema": LABEL_SCHEMA,
        "execution_timing_semantics": FACTOR_LABEL_SEMANTICS,
        "entry_delay_sessions": 1,
        "horizons": list(horizons_tuple),
        "market_session_schedule_sha256": market_session_schedule_sha256(sessions),
        "market_session_count": int(len(sessions)),
        "missing_bar_semantics": "exact global session required; missing entry/exit/path bar => NaN, never next-row shift",
        "critical_flag_semantics": "is_suspended/is_st/is_limit_up must be explicitly known false on signal and entry sessions",
        "canonical_builder_schema": built.schema,
    }
    return df, schema


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--market-calendar",
        type=Path,
        default=DEFAULT_MARKET_CALENDAR,
        help="Independent exchange-session artifact; defaults to U0 PIT trading calendar.",
    )
    parser.add_argument("--horizons", default="1,3,5,10,20,60,120")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input dataset does not exist: {args.input}")
    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    sessions, calendar_meta = _load_market_sessions(args.market_calendar)
    source = pd.read_parquet(args.input)
    rebuilt, schema = build_executable_training_labels(
        source,
        horizons=horizons,
        market_sessions=sessions,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.to_parquet(args.output, index=False)
    manifest_path = args.output.with_suffix(".labels.json")
    manifest = {
        **schema,
        "input_path": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "output_path": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "market_calendar": calendar_meta,
        "rows": int(len(rebuilt)),
        "symbols": int(rebuilt["symbol"].nunique()),
        "signal_date_min": str(pd.to_datetime(rebuilt["trade_date"]).min().date()),
        "signal_date_max": str(pd.to_datetime(rebuilt["trade_date"]).max().date()),
        "benchmarkSymbol": "000300.SH",
        "training_window": "<=2025-09-30",
        "final_holdout_window": "2025-10-01..2026-03-31",
        "production_note": (
            "T-close signal; exact next-global-session entry; no per-symbol next-row fallback; "
            "all forward_return_* labels are evaluation targets and never inference features"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"wrote {args.output} rows={len(rebuilt):,}")
    print(f"wrote {manifest_path}")
    for horizon in horizons:
        column = f"forward_return_{horizon}d"
        if column in rebuilt.columns:
            print(f"  {column}: {rebuilt[column].notna().sum():,} valid")
    print(f"  signal_tradable: {rebuilt['_signal_tradable'].mean():.3%}")
    print(f"  entry_tradable:  {rebuilt['_entry_tradable'].mean():.3%}")
    print(f"  calendar_sha256: {schema['market_session_schedule_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
