"""Supplier-neutral historical ST/risk-warning evidence contract.

The table is intentionally local/versioned rather than tied to one vendor. Two
representations are accepted:

* exact daily: ``symbol, trade_date, is_st, available_at``;
* explicit intervals: ``symbol, start_date, end_date, is_st, available_at``.

Interval tables must describe *both* ST and non-ST periods. Absence of an ST row
is never interpreted as ``False``. Every requested market row must resolve to
exactly one evidence row whose ``available_at`` timestamp is no later than the
A-share pre-open boundary (09:25 Asia/Shanghai) for that trade date.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

import pandas as pd


class HistoricalSTCoverageError(RuntimeError):
    """Historical ST evidence is incomplete, ambiguous or not point-in-time."""


@dataclass(frozen=True, slots=True)
class HistoricalSTEvidence:
    frame: pd.DataFrame
    mode: Literal["daily", "interval"]
    source_path: str
    source_sha256: str


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise HistoricalSTCoverageError(
        f"unsupported historical ST evidence format {suffix!r}; use parquet/csv/jsonl"
    )


def _strict_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes"}:
        return True
    if text in {"0", "false", "f", "no"}:
        return False
    raise HistoricalSTCoverageError(f"invalid is_st value {value!r}")


def _parse_available_at(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        raise HistoricalSTCoverageError("historical ST available_at contains invalid timestamps")
    return parsed


def _preopen_deadline_utc(trade_dates: pd.Series) -> pd.Series:
    date_text = pd.to_datetime(trade_dates, errors="coerce").dt.strftime("%Y-%m-%d")
    if date_text.isna().any():
        raise HistoricalSTCoverageError("historical ST evidence contains invalid trade dates")
    return pd.to_datetime(
        date_text + " 09:25:00+08:00",
        errors="coerce",
        utc=True,
    )


def load_historical_st_evidence(path: str | Path) -> HistoricalSTEvidence:
    source = Path(path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise HistoricalSTCoverageError(
            f"historical ST evidence file does not exist: {source}"
        )
    frame = _read_table(source)
    required = {"symbol", "is_st", "available_at"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HistoricalSTCoverageError(
            f"historical ST evidence missing required columns: {missing}"
        )
    has_daily = "trade_date" in frame.columns
    has_interval = "start_date" in frame.columns
    if has_daily == has_interval:
        raise HistoricalSTCoverageError(
            "historical ST evidence must use exactly one representation: trade_date or start_date/end_date"
        )

    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.strip()
    if out["symbol"].eq("").any():
        raise HistoricalSTCoverageError("historical ST evidence contains empty symbols")
    out["is_st"] = out["is_st"].map(_strict_bool).astype(bool)
    out["available_at"] = _parse_available_at(out["available_at"])
    if "point_in_time_valid" in out.columns:
        pit = out["point_in_time_valid"].map(_strict_bool)
        if not bool(pit.all()):
            raise HistoricalSTCoverageError(
                "historical ST evidence contains point_in_time_valid=False rows"
            )

    if has_daily:
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
        if out["trade_date"].isna().any():
            raise HistoricalSTCoverageError("historical ST evidence contains invalid trade_date")
        duplicates = out.duplicated(["symbol", "trade_date"], keep=False)
        if bool(duplicates.any()):
            raise HistoricalSTCoverageError(
                "historical ST daily evidence has duplicate (symbol, trade_date) rows"
            )
        deadline = _preopen_deadline_utc(out["trade_date"])
        if bool((out["available_at"] > deadline).any()):
            raise HistoricalSTCoverageError(
                "historical ST daily evidence contains status unavailable by 09:25 Asia/Shanghai"
            )
        out = out.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        mode: Literal["daily", "interval"] = "daily"
    else:
        if "end_date" not in out.columns:
            raise HistoricalSTCoverageError(
                "historical ST interval evidence requires end_date; open-ended intervals are not accepted"
            )
        out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce").dt.normalize()
        out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce").dt.normalize()
        if out[["start_date", "end_date"]].isna().any().any():
            raise HistoricalSTCoverageError(
                "historical ST interval evidence contains invalid start/end dates"
            )
        if bool((out["end_date"] < out["start_date"]).any()):
            raise HistoricalSTCoverageError(
                "historical ST interval evidence contains end_date before start_date"
            )
        deadline = _preopen_deadline_utc(out["start_date"])
        if bool((out["available_at"] > deadline).any()):
            raise HistoricalSTCoverageError(
                "historical ST interval evidence was not available by interval start pre-open"
            )
        ordered = out.sort_values(["symbol", "start_date", "end_date"]).reset_index(drop=True)
        prior_end = ordered.groupby("symbol", sort=False)["end_date"].shift(1)
        overlaps = prior_end.notna() & (ordered["start_date"] <= prior_end)
        if bool(overlaps.any()):
            raise HistoricalSTCoverageError(
                "historical ST interval evidence has overlapping intervals for a symbol"
            )
        out = ordered
        mode = "interval"

    return HistoricalSTEvidence(
        frame=out,
        mode=mode,
        source_path=str(source),
        source_sha256=_file_sha256(source),
    )


def attach_historical_st(
    market_rows: pd.DataFrame,
    evidence: HistoricalSTEvidence,
) -> pd.DataFrame:
    """Attach exact historical ``is_st`` state to every requested market row."""

    if market_rows is None or market_rows.empty:
        return pd.DataFrame() if market_rows is None else market_rows.copy()
    required = {"symbol", "trade_date"}
    missing = sorted(required - set(market_rows.columns))
    if missing:
        raise HistoricalSTCoverageError(
            f"market rows missing ST join keys: {missing}"
        )
    rows = market_rows.copy()
    rows["symbol"] = rows["symbol"].astype(str).str.strip()
    rows["trade_date"] = pd.to_datetime(rows["trade_date"], errors="coerce").dt.normalize()
    if rows["trade_date"].isna().any():
        raise HistoricalSTCoverageError("market rows contain invalid trade_date")
    rows["__st_order"] = range(len(rows))

    if evidence.mode == "daily":
        status = evidence.frame[
            ["symbol", "trade_date", "is_st", "available_at"]
        ].rename(columns={"available_at": "st_available_at"})
        merged = rows.merge(
            status,
            on=["symbol", "trade_date"],
            how="left",
            validate="many_to_one",
        )
    else:
        left = rows.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
        right = evidence.frame[
            ["symbol", "start_date", "end_date", "is_st", "available_at"]
        ].rename(columns={"available_at": "st_available_at"})
        right = right.sort_values(["start_date", "symbol"]).reset_index(drop=True)
        merged = pd.merge_asof(
            left,
            right,
            left_on="trade_date",
            right_on="start_date",
            by="symbol",
            direction="backward",
            allow_exact_matches=True,
        )
        invalid_interval = merged["end_date"].isna() | (
            merged["trade_date"] > merged["end_date"]
        )
        merged.loc[invalid_interval, ["is_st", "st_available_at"]] = pd.NA

    missing_status = merged["is_st"].isna() | merged["st_available_at"].isna()
    if bool(missing_status.any()):
        sample = merged.loc[missing_status, ["symbol", "trade_date"]].head(5)
        rendered = [
            f"{row.symbol}@{row.trade_date.date()}"
            for row in sample.itertuples(index=False)
        ]
        raise HistoricalSTCoverageError(
            "historical ST evidence does not fully cover requested market rows; "
            f"sample={rendered}"
        )

    deadline = _preopen_deadline_utc(merged["trade_date"])
    if bool((merged["st_available_at"] > deadline).any()):
        raise HistoricalSTCoverageError(
            "historical ST join would use status unavailable by the session pre-open boundary"
        )
    merged["is_st"] = merged["is_st"].astype(bool)
    merged["is_st_provenance"] = "dated_pit_st_evidence"
    merged["st_evidence_sha256"] = evidence.source_sha256
    merged = merged.sort_values("__st_order").drop(columns=["__st_order"])
    return merged.reset_index(drop=True)


__all__ = [
    "HistoricalSTCoverageError",
    "HistoricalSTEvidence",
    "attach_historical_st",
    "load_historical_st_evidence",
]
