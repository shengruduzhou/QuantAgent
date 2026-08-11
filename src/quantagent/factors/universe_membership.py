"""Point-in-time research-universe membership contracts.

A static symbol list is useful for smoke research, but it is not a historical
stock-pool reconstruction. This module keeps those semantics separate:

* ``point_in_time_membership`` consumes effective-dated membership evidence;
* ``research_universe_explicit_static`` is represented by the research command,
  never upgraded to PIT merely because the symbols were supplied explicitly.

PIT evidence is supplier-neutral and version-bound. Every interval carries a
source, source version, availability timestamp and membership reason. The file
SHA and canonical row digests make universe changes visible in research
manifests instead of allowing today's constituents to masquerade as history.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


PIT_UNIVERSE_MODE = "point_in_time_membership"
STATIC_UNIVERSE_MODE = "research_universe_explicit_static"


class UniverseMembershipError(RuntimeError):
    """Universe evidence is incomplete, ambiguous or temporally invalid."""


@dataclass(frozen=True, slots=True)
class UniverseMembershipEvidence:
    frame: pd.DataFrame
    universe_id: str
    source_path: str
    source_sha256: str
    source_versions: tuple[str, ...]


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
    raise UniverseMembershipError(
        f"unsupported universe evidence format {suffix!r}; use parquet/csv/jsonl"
    )


def _preopen_deadline_utc(dates: pd.Series) -> pd.Series:
    text = pd.to_datetime(dates, errors="coerce").dt.strftime("%Y-%m-%d")
    if text.isna().any():
        raise UniverseMembershipError("universe evidence contains invalid effective dates")
    return pd.to_datetime(text + " 09:25:00+08:00", errors="coerce", utc=True)


def _canonical_row_digest(row: pd.Series) -> str:
    end = row["effective_to"]
    payload = {
        "symbol": str(row["symbol"]),
        "effective_from": pd.Timestamp(row["effective_from"]).date().isoformat(),
        "effective_to": None if pd.isna(end) else pd.Timestamp(end).date().isoformat(),
        "universe_id": str(row["universe_id"]),
        "source": str(row["source"]),
        "source_version": str(row["source_version"]),
        "available_at": pd.Timestamp(row["available_at"]).isoformat(),
        "membership_reason": str(row["membership_reason"]),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def load_universe_membership(
    path: str | Path,
    *,
    universe_id: str | None = None,
) -> UniverseMembershipEvidence:
    """Load and validate effective-dated PIT membership evidence.

    ``effective_to`` may be null for an open interval. Membership must have been
    available by the 09:25 Asia/Shanghai pre-open boundary on ``effective_from``.
    Overlapping intervals for the same symbol/universe are rejected; absence of
    an interval means non-membership, not an implicit current-member backfill.
    """

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        raise UniverseMembershipError(
            f"universe membership evidence does not exist: {source_path}"
        )
    frame = _read_table(source_path)
    required = {
        "symbol",
        "effective_from",
        "effective_to",
        "universe_id",
        "source",
        "source_version",
        "available_at",
        "membership_reason",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise UniverseMembershipError(
            f"universe membership evidence missing required columns: {missing}"
        )

    out = frame.copy()
    for column in ("symbol", "universe_id", "source", "source_version", "membership_reason"):
        out[column] = out[column].astype("string").str.strip()
        if out[column].isna().any() or out[column].eq("").any():
            raise UniverseMembershipError(
                f"universe membership evidence contains empty {column}"
            )
    if universe_id is not None and str(universe_id).strip():
        selected_id = str(universe_id).strip()
        out = out[out["universe_id"] == selected_id].copy()
        if out.empty:
            raise UniverseMembershipError(
                f"universe_id {selected_id!r} has no membership rows"
            )
    else:
        ids = sorted(out["universe_id"].unique().tolist())
        if len(ids) != 1:
            raise UniverseMembershipError(
                "membership file contains multiple universe_id values; pass --universe-id explicitly"
            )
        selected_id = str(ids[0])

    out["effective_from"] = pd.to_datetime(
        out["effective_from"], errors="coerce"
    ).dt.normalize()
    out["effective_to"] = pd.to_datetime(
        out["effective_to"], errors="coerce"
    ).dt.normalize()
    if out["effective_from"].isna().any():
        raise UniverseMembershipError("membership evidence has invalid effective_from")
    bad_end = out["effective_to"].notna() & (
        out["effective_to"] < out["effective_from"]
    )
    if bool(bad_end.any()):
        raise UniverseMembershipError(
            "membership evidence has effective_to before effective_from"
        )

    out["available_at"] = pd.to_datetime(
        out["available_at"], errors="coerce", utc=True
    )
    if out["available_at"].isna().any():
        raise UniverseMembershipError("membership evidence has invalid available_at")
    deadline = _preopen_deadline_utc(out["effective_from"])
    if bool((out["available_at"] > deadline).any()):
        raise UniverseMembershipError(
            "membership evidence contains rows unavailable by 09:25 Asia/Shanghai on effective_from"
        )

    ordered = out.sort_values(
        ["universe_id", "symbol", "effective_from", "effective_to"],
        na_position="last",
    ).reset_index(drop=True)
    for (_uid, _symbol), group in ordered.groupby(
        ["universe_id", "symbol"], sort=False
    ):
        previous_end: pd.Timestamp | None = None
        for row in group.itertuples(index=False):
            start = pd.Timestamp(row.effective_from)
            end = None if pd.isna(row.effective_to) else pd.Timestamp(row.effective_to)
            if previous_end is None and group.index[0] != getattr(row, "Index", group.index[0]):
                pass
            if previous_end is not None and start <= previous_end:
                raise UniverseMembershipError(
                    "membership evidence has overlapping intervals for a symbol"
                )
            if end is None:
                # An open interval must be last; otherwise the next interval overlaps forever.
                previous_end = pd.Timestamp.max.normalize()
            else:
                previous_end = end

    computed = ordered.apply(_canonical_row_digest, axis=1)
    if "row_sha256" in ordered.columns:
        supplied = ordered["row_sha256"].astype("string").str.strip().str.lower()
        present = supplied.notna() & supplied.ne("")
        if bool((present & supplied.ne(computed)).any()):
            raise UniverseMembershipError(
                "membership evidence row_sha256 does not match canonical row contents"
            )
    ordered["row_sha256"] = computed

    return UniverseMembershipEvidence(
        frame=ordered.reset_index(drop=True),
        universe_id=selected_id,
        source_path=str(source_path),
        source_sha256=_file_sha256(source_path),
        source_versions=tuple(sorted(ordered["source_version"].unique().tolist())),
    )


def symbols_for_window(
    evidence: UniverseMembershipEvidence,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> tuple[str, ...]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise UniverseMembershipError("research window end precedes start")
    frame = evidence.frame
    overlaps = (frame["effective_from"] <= end) & (
        frame["effective_to"].isna() | (frame["effective_to"] >= start)
    )
    symbols = tuple(sorted(frame.loc[overlaps, "symbol"].astype(str).unique()))
    if not symbols:
        raise UniverseMembershipError(
            "PIT universe has no members overlapping the requested research window"
        )
    return symbols


def filter_market_by_membership(
    market: pd.DataFrame,
    evidence: UniverseMembershipEvidence,
) -> pd.DataFrame:
    """Keep only market rows whose symbol is a member on that trade date."""

    if market is None or market.empty:
        return pd.DataFrame() if market is None else market.copy()
    required = {"symbol", "trade_date"}
    missing = sorted(required - set(market.columns))
    if missing:
        raise UniverseMembershipError(f"market panel missing universe join keys: {missing}")
    rows = market.copy()
    rows["symbol"] = rows["symbol"].astype(str).str.strip()
    rows["trade_date"] = pd.to_datetime(rows["trade_date"], errors="coerce").dt.normalize()
    if rows["trade_date"].isna().any():
        raise UniverseMembershipError("market panel contains invalid trade_date")

    pieces: list[pd.DataFrame] = []
    membership = evidence.frame
    for symbol, group in rows.groupby("symbol", sort=False):
        intervals = membership[membership["symbol"] == symbol]
        if intervals.empty:
            continue
        mask = pd.Series(False, index=group.index)
        for interval in intervals.itertuples(index=False):
            start = pd.Timestamp(interval.effective_from)
            end = (
                pd.Timestamp.max.normalize()
                if pd.isna(interval.effective_to)
                else pd.Timestamp(interval.effective_to)
            )
            mask |= group["trade_date"].between(start, end, inclusive="both")
        selected = group.loc[mask]
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise UniverseMembershipError(
            "no market rows survive the configured PIT universe membership"
        )
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["trade_date", "symbol"]
    ).reset_index(drop=True)


def membership_artifact_for_window(
    evidence: UniverseMembershipEvidence,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    frame = evidence.frame.copy()
    overlaps = (frame["effective_from"] <= end) & (
        frame["effective_to"].isna() | (frame["effective_to"] >= start)
    )
    return frame.loc[overlaps].reset_index(drop=True)


def static_membership_artifact(
    symbols: Iterable[str],
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Represent an explicit static universe without pretending it is PIT."""

    canonical = sorted({str(symbol).strip() for symbol in symbols if str(symbol).strip()})
    if not canonical:
        raise UniverseMembershipError("explicit static research universe is empty")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    return pd.DataFrame(
        {
            "symbol": canonical,
            "effective_from": [start] * len(canonical),
            "effective_to": [end] * len(canonical),
            "universe_id": ["explicit_static_research_universe"] * len(canonical),
            "source": ["research_command"] * len(canonical),
            "source_version": ["unversioned_static_symbols"] * len(canonical),
            "available_at": [pd.NaT] * len(canonical),
            "membership_reason": [STATIC_UNIVERSE_MODE] * len(canonical),
            "row_sha256": [pd.NA] * len(canonical),
            "point_in_time_valid": [False] * len(canonical),
        }
    )


def dataframe_sha256(frame: pd.DataFrame) -> str:
    material = frame.copy()
    for column in material.columns:
        if pd.api.types.is_datetime64_any_dtype(material[column]):
            material[column] = material[column].astype("string")
    payload = material.sort_values(list(material.columns)).to_csv(index=False).encode("utf-8")
    return sha256(payload).hexdigest()


__all__ = [
    "PIT_UNIVERSE_MODE",
    "STATIC_UNIVERSE_MODE",
    "UniverseMembershipError",
    "UniverseMembershipEvidence",
    "dataframe_sha256",
    "filter_market_by_membership",
    "load_universe_membership",
    "membership_artifact_for_window",
    "static_membership_artifact",
    "symbols_for_window",
]
