"""Reader for qlib 1-minute binary data → intraday OHLCV panel.

The qlib on-disk format stores each (instrument, field) as a little-endian
float32 ``.bin`` file whose **first float is the start index into the
calendar**; the remaining floats are consecutive minute values from that
index. This module decodes that format into a tidy long DataFrame the
intraday factor + execution layers consume.

Layout (``provider_uri`` root)::

    calendars/1min.txt                # one ISO minute per line
    instruments/all.txt               # SYMBOL\tstart\tend  (qlib SH/SZ prefix)
    features/<sh600000>/<field>.1min.bin

Prices are raw; multiply OHLC by ``factor`` for forward-adjusted (qfq)
series. ``volume`` is shares; per-minute amount ≈ ``volume * close``.

Note on coverage: the qlib *community free* 1-min release covers only
~2020-09 → 2021-06. Production intraday strategies need a recent 1-min feed
(tickflow / broker); this reader is feed-agnostic and works for either.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.data.v7_auto_range import from_qlib_instrument, to_qlib_instrument

_DEFAULT_FIELDS = ("open", "high", "low", "close", "volume", "factor")


@lru_cache(maxsize=8)
def load_calendar(root: str | Path) -> pd.DatetimeIndex:
    """Load the 1-minute calendar as a DatetimeIndex."""
    cal_path = Path(root) / "calendars" / "1min.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"no 1min calendar at {cal_path}")
    s = pd.read_csv(cal_path, header=None)[0]
    return pd.DatetimeIndex(pd.to_datetime(s))


def load_instruments(root: str | Path, universe: str = "all") -> pd.DataFrame:
    """Load an instrument list (canonical symbols + active window)."""
    path = Path(root) / "instruments" / f"{universe}.txt"
    if not path.exists():
        raise FileNotFoundError(f"no instruments file {path}")
    df = pd.read_csv(path, sep="\t", header=None, names=["qlib_symbol", "start", "end"])
    df["symbol"] = df["qlib_symbol"].astype(str).map(from_qlib_instrument)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    return df


def _read_bin(path: Path) -> tuple[int, np.ndarray]:
    """Return (start_index_into_calendar, float32 values) for a qlib .bin."""
    arr = np.fromfile(path, dtype="<f4")
    if arr.size == 0:
        return 0, np.array([], dtype="float32")
    return int(arr[0]), arr[1:]


def read_instrument_minutes(
    root: str | Path,
    symbol: str,
    *,
    fields: tuple[str, ...] = _DEFAULT_FIELDS,
    calendar: pd.DatetimeIndex | None = None,
    adjust: bool = True,
) -> pd.DataFrame:
    """Decode one instrument's minute bars into a tidy frame.

    Returns columns ``[symbol, datetime, trade_date, <fields...>, amount]``
    with NaN (auction / halt) rows dropped. OHLC are forward-adjusted when
    ``adjust`` and a ``factor`` field is present.
    """
    root = Path(root)
    cal = calendar if calendar is not None else load_calendar(root)
    qsym = to_qlib_instrument(symbol).lower()
    fdir = root / "features" / qsym
    if not fdir.exists():
        return pd.DataFrame(columns=["symbol", "datetime", "trade_date", *fields])

    series: dict[str, pd.Series] = {}
    for field in fields:
        fp = fdir / f"{field}.1min.bin"
        if not fp.exists():
            continue
        start, vals = _read_bin(fp)
        if vals.size == 0:
            continue
        idx = cal[start:start + len(vals)]
        series[field] = pd.Series(vals.astype("float64"), index=idx)
    if "close" not in series:
        return pd.DataFrame(columns=["symbol", "datetime", "trade_date", *fields])

    df = pd.DataFrame(series)
    df = df.dropna(subset=["close"])
    if adjust and "factor" in df.columns:
        for c in ("open", "high", "low", "close"):
            if c in df.columns:
                df[c] = df[c] * df["factor"]
    df.index.name = "datetime"
    df = df.reset_index()
    df["symbol"] = symbol
    df["trade_date"] = df["datetime"].dt.normalize()
    if "volume" in df.columns and "close" in df.columns:
        df["amount"] = df["volume"] * df["close"]
    cols = ["symbol", "datetime", "trade_date", *[f for f in fields if f in df.columns]]
    if "amount" in df.columns:
        cols.append("amount")
    return df[cols]


def build_intraday_panel(
    root: str | Path,
    symbols: list[str],
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    fields: tuple[str, ...] = _DEFAULT_FIELDS,
    adjust: bool = True,
) -> pd.DataFrame:
    """Stack many instruments' minute bars into one long intraday panel."""
    cal = load_calendar(root)
    frames = []
    for sym in symbols:
        f = read_instrument_minutes(root, sym, fields=fields, calendar=cal, adjust=adjust)
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["symbol", "datetime", "trade_date", *fields])
    panel = pd.concat(frames, ignore_index=True)
    if start is not None:
        panel = panel[panel["datetime"] >= pd.Timestamp(start)]
    if end is not None:
        panel = panel[panel["datetime"] <= pd.Timestamp(end)]
    return panel.sort_values(["symbol", "datetime"]).reset_index(drop=True)


__all__ = [
    "load_calendar",
    "load_instruments",
    "read_instrument_minutes",
    "build_intraday_panel",
]
