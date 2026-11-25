"""Classify open-exchange sessions on which a listed symbol has no bar.

The v7 silver panel (like the U0 raw panel) keeps **traded sessions only**: a
suspended session is represented by omitting the row. A strict simulator cannot
value or refuse to trade a held name that has no row, so every missing
``(symbol, session)`` must be *classified with evidence* before an execution
step may use the panel:

``SUSPENDED``
    the session falls inside a dated vendor halt interval
    (``akshare.stock_tfp_em`` / Eastmoney ``RPT_CUSTOM_SUSPEND_DATA_INTERFACE``)
    **and** a halt snapshot for that session is on disk.
``DELISTED``
    the session is after a dated delisting.
``MISSING_UNEXPLAINED``
    anything else -- a vendor gap, a symbol outside the fetch seed, or a halt
    the snapshots never recorded. It is never filled: an execution consumer
    must fail closed on it.

Only ``SUSPENDED`` gaps become explicit execution rows
(:func:`suspended_execution_rows`): ``is_suspended = 1``, ``volume = amount =
0`` and OHLC equal to the last traded raw close, which is a **valuation mark**
(provenance ``gap_classification = SUSPENDED`` / ``price_basis =
carried_last_traded_close``), never a tradeable price. This is the convention
of the certified execution panel (:mod:`quantagent.data.ashare.execution_panel`),
except that unexplained gaps are *not* materialised here.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from quantagent.data.ashare.symbols import SymbolError, identify

GAP_SUSPENDED = "SUSPENDED"
GAP_DELISTED = "DELISTED"
GAP_UNEXPLAINED = "MISSING_UNEXPLAINED"
GAP_KIND_INTERIOR = "interior"
GAP_KIND_TRAILING = "trailing"
PRICE_BASIS_CARRIED = "carried_last_traded_close"
SUSPENSION_SOURCE = "akshare.stock_tfp_em"
SUSPENSION_ENDPOINT = "eastmoney RPT_CUSTOM_SUSPEND_DATA_INTERFACE"


class SessionGapError(RuntimeError):
    """Raised when gap evidence is internally inconsistent."""


def fold_suspension_snapshots(snapshots: pd.DataFrame, *, retrieved_at: str) -> pd.DataFrame:
    """Fold per-date halt snapshots into ``(symbol, start, end)`` intervals.

    A halt with no published resume date is closed at the last snapshot that
    still showed it -- never left open-ended into the future.
    """
    columns = ["symbol", "effective_start", "effective_end", "suspension_reason", "evidence",
               "source", "source_endpoint", "retrieved_at", "available_at", "quality_status"]
    if snapshots is None or snapshots.empty:
        return pd.DataFrame(columns=columns)
    snaps = snapshots.copy()
    symbols, keep = [], []
    for code in snaps["code"].astype(str):
        try:
            symbols.append(identify(code.zfill(6)).symbol)
            keep.append(True)
        except SymbolError:
            keep.append(False)
    snaps = snaps[keep].copy()
    snaps["symbol"] = symbols
    snaps["snapshot_date"] = pd.to_datetime(snaps["snapshot_date"]).dt.normalize()
    snaps["suspend_start"] = pd.to_datetime(snaps["suspend_start"], errors="coerce").dt.normalize()
    snaps["suspend_end"] = pd.to_datetime(snaps["suspend_end"], errors="coerce").dt.normalize()
    snaps["effective_start"] = snaps["suspend_start"].fillna(snaps["snapshot_date"])
    grouped = snaps.groupby(["symbol", "effective_start"], as_index=False).agg(
        effective_end=("suspend_end", "max"),
        last_seen=("snapshot_date", "max"),
        suspension_reason=("suspension_reason", "first"))
    grouped["effective_end"] = grouped["effective_end"].fillna(grouped["last_seen"])
    return pd.DataFrame({
        "symbol": grouped["symbol"],
        "effective_start": grouped["effective_start"],
        "effective_end": grouped["effective_end"],
        "suspension_reason": grouped["suspension_reason"],
        "evidence": "eastmoney_suspension_snapshot",
        "source": SUSPENSION_SOURCE,
        "source_endpoint": SUSPENSION_ENDPOINT,
        "retrieved_at": retrieved_at,
        "available_at": grouped["effective_start"].dt.strftime("%Y-%m-%d"),
        "quality_status": "OK",
    }).sort_values(["symbol", "effective_start"]).reset_index(drop=True)


def find_session_gaps(
    panel: pd.DataFrame,
    sessions: Iterable[Any],
    *,
    start: Any = None,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Every open session in ``[first bar, panel max]`` without a bar, per symbol.

    ``interior`` gaps lie between two bars of the symbol; ``trailing`` gaps lie
    after its last bar up to the panel's last session. Sessions are the exchange
    calendar -- never the panel's own dates, which would hide a market-wide hole.
    ``start`` limits the reported gaps (bars before it still anchor the span).
    """
    frame = panel[["symbol", "trade_date"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    if symbols is not None:
        frame = frame[frame["symbol"].astype(str).isin(set(map(str, symbols)))]
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", "gap_kind"])
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Index(list(sessions))).normalize().unique()))
    panel_max = frame["trade_date"].max()
    cal = cal[cal <= panel_max]
    off_calendar = set(frame["trade_date"].unique()) - set(cal)
    if off_calendar:
        raise SessionGapError(
            f"{len(off_calendar)} panel dates are not open sessions in the calendar, "
            f"e.g. {sorted(off_calendar)[:3]}")
    lo_bound = pd.Timestamp(start).normalize() if start is not None else None
    records: list[tuple[str, pd.Timestamp, str]] = []
    cal_values = cal.to_numpy()
    for symbol, dates in frame.groupby("symbol", sort=True)["trade_date"]:
        have = np.unique(dates.to_numpy())
        first, last = have[0], have[-1]
        span = cal_values[(cal_values > first) & (cal_values <= panel_max.to_datetime64())]
        missing = np.setdiff1d(span, have, assume_unique=True)
        if lo_bound is not None:
            missing = missing[missing >= lo_bound.to_datetime64()]
        for day in missing:
            kind = GAP_KIND_INTERIOR if day < last else GAP_KIND_TRAILING
            records.append((str(symbol), pd.Timestamp(day), kind))
    return pd.DataFrame(records, columns=["symbol", "trade_date", "gap_kind"])


def classify_session_gaps(
    gaps: pd.DataFrame,
    *,
    suspension_intervals: pd.DataFrame | None,
    snapshot_dates: Iterable[Any] | None,
    delisting_dates: Mapping[str, Any] | pd.Series | None = None,
) -> pd.DataFrame:
    """Attach ``classification`` and ``evidence`` to each gap.

    ``snapshot_dates`` are the sessions whose halt snapshot is on disk; a gap on
    a session outside it can never be SUSPENDED (no snapshot = no evidence),
    even if an interval from another snapshot spans it.
    """
    out = gaps.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
    out["classification"] = GAP_UNEXPLAINED
    out["evidence"] = "no_halt_interval"
    out["suspension_reason"] = ""
    if out.empty:
        return out
    covered = set(pd.to_datetime(pd.Index(list(snapshot_dates or []))).normalize())
    in_window = out["trade_date"].isin(covered).to_numpy()
    out.loc[~in_window, "evidence"] = "outside_halt_snapshot_window"

    if suspension_intervals is not None and len(suspension_intervals):
        iv = suspension_intervals[["symbol", "effective_start", "effective_end",
                                   "suspension_reason"]].copy()
        iv["effective_start"] = pd.to_datetime(iv["effective_start"]).dt.normalize()
        iv["effective_end"] = pd.to_datetime(iv["effective_end"]).dt.normalize()
        keyed = out.reset_index().merge(iv, on="symbol", how="inner", suffixes=("", "_iv"))
        hit = keyed[(keyed["trade_date"] >= keyed["effective_start"])
                    & (keyed["trade_date"] <= keyed["effective_end"])]
        hit = hit.drop_duplicates("index")
        idx = hit["index"].to_numpy()
        ok = in_window[idx]
        out.loc[idx[ok], "classification"] = GAP_SUSPENDED
        out.loc[idx[ok], "evidence"] = "eastmoney_suspension_snapshot"
        out.loc[idx[ok], "suspension_reason"] = hit["suspension_reason_iv"].to_numpy()[ok]

    if delisting_dates is not None:
        dated = pd.to_datetime(pd.Series(delisting_dates), errors="coerce").dropna()
        dated.index = dated.index.astype(str)
        # the U0 table repeats some rows verbatim; on a genuine conflict the
        # LATEST date wins (fewer gaps excused as delisted = fail closed)
        dated = dated.groupby(level=0).max()
        when = out["symbol"].map(dated)
        delisted = when.notna() & (out["trade_date"] > when)
        out.loc[delisted, "classification"] = GAP_DELISTED
        out.loc[delisted, "evidence"] = "dated_delisting"
    return out


def enforce_halt_chain(classified: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Demote SUSPENDED gaps whose carried mark would cross an unexplained session.

    The carried close is the last traded bar's close. It is a defensible mark
    only if *every* session between that bar and the gap is halt-covered; one
    unexplained session in between means we do not know what happened to the
    price, so the gap (and every later gap of the same run) is
    MISSING_UNEXPLAINED with ``evidence = halt_chain_broken``. ``classified``
    must contain all gaps since each symbol's prior bar (do not pre-filter by
    window start).
    """
    out = classified.copy()
    if out.empty:
        return out
    bars = panel[["symbol", "trade_date"]].copy()
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    bars["prior_bar"] = bars["trade_date"]
    left = out[["symbol", "trade_date"]].reset_index().sort_values("trade_date", kind="mergesort")
    hit = pd.merge_asof(left, bars.sort_values("trade_date", kind="mergesort"),
                        on="trade_date", by="symbol", direction="backward",
                        allow_exact_matches=False).set_index("index").reindex(out.index)
    out["_prior_bar"] = hit["prior_bar"]
    out = out.sort_values(["symbol", "trade_date"], kind="mergesort")
    not_halt = (out["classification"] != GAP_SUSPENDED).astype(int)
    seen_before = (not_halt.groupby([out["symbol"], out["_prior_bar"]], dropna=False).cumsum()
                   - not_halt)
    broken = (out["classification"] == GAP_SUSPENDED) & (seen_before > 0)
    out.loc[broken, "classification"] = GAP_UNEXPLAINED
    out.loc[broken, "evidence"] = "halt_chain_broken"
    return out.drop(columns=["_prior_bar"]).sort_index()


def suspended_execution_rows(panel: pd.DataFrame, classified: pd.DataFrame) -> pd.DataFrame:
    """Explicit untradeable rows for SUSPENDED gaps, valued at the carried close.

    ``panel`` must contain the bars preceding each gap. A SUSPENDED gap with no
    earlier traded bar cannot be valued and raises.
    """
    gaps = classified[classified["classification"] == GAP_SUSPENDED][
        ["symbol", "trade_date", "suspension_reason"]].copy()
    if gaps.empty:
        return pd.DataFrame(columns=list(panel.columns) + ["gap_classification", "price_basis"])
    bars = panel.copy()
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    bars = bars.sort_values("trade_date", kind="mergesort")
    gaps = gaps.sort_values("trade_date", kind="mergesort")
    prior = pd.merge_asof(
        gaps, bars.drop(columns=["suspension_reason"], errors="ignore"),
        on="trade_date", by="symbol", direction="backward", allow_exact_matches=False)
    if prior["close"].isna().any():
        bad = prior.loc[prior["close"].isna(), ["symbol", "trade_date"]].head(3).to_dict("records")
        raise SessionGapError(f"SUSPENDED gaps without a prior traded close, e.g. {bad}")
    rows = prior.copy()
    for column in ("open", "high", "low"):
        if column in rows.columns:
            rows[column] = rows["close"]
    for column in ("volume", "amount"):
        if column in rows.columns:
            rows[column] = 0.0
    if "is_suspended" in rows.columns:
        rows["is_suspended"] = 1.0
    for column in ("is_limit_up", "is_limit_down"):
        if column in rows.columns:
            rows[column] = 0.0
    if "available_at" in rows.columns:
        stamp = rows["trade_date"] + pd.Timedelta(hours=15)
        if pd.api.types.is_datetime64_any_dtype(panel["available_at"]):
            tz = getattr(panel["available_at"].dt, "tz", None)
            rows["available_at"] = stamp.dt.tz_localize(tz) if tz is not None else stamp
        else:
            rows["available_at"] = stamp.dt.strftime("%Y-%m-%d %H:%M:%S")
    if "source" in rows.columns:
        rows["source"] = "derived:suspension_gap"
    rows["gap_classification"] = GAP_SUSPENDED
    rows["price_basis"] = PRICE_BASIS_CARRIED
    rows = rows.drop(columns=["suspension_reason"])
    return rows.reindex(columns=list(panel.columns) + ["gap_classification", "price_basis"])


def gap_summary(classified: pd.DataFrame) -> dict[str, Any]:
    if classified.empty:
        return {"gaps": 0}
    out: dict[str, Any] = {"gaps": int(len(classified)),
                           "symbols": int(classified["symbol"].nunique())}
    for key in ("classification", "gap_kind", "evidence"):
        if key in classified.columns:
            out[f"by_{key}"] = {str(k): int(v) for k, v in
                                classified[key].value_counts().items()}
    if {"classification", "gap_kind"} <= set(classified.columns):
        out["by_kind_and_class"] = {
            f"{k}/{c}": int(v) for (k, c), v in
            classified.groupby(["gap_kind", "classification"]).size().items()}
    return out


def load_suspension_evidence(directories: Iterable[Any]) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Union of halt intervals and covered snapshot sessions from U0-style dirs.

    Each directory holds ``suspension_intervals.parquet`` and a ``_suspension/``
    staging dir with one ``YYYYMMDD.parquet`` per fetched snapshot (an empty file
    = fetched, no halts). Missing directories contribute nothing -- they never
    widen coverage.
    """
    from pathlib import Path

    frames: list[pd.DataFrame] = []
    covered: set[pd.Timestamp] = set()
    for raw in directories:
        directory = Path(raw)
        intervals = directory / "suspension_intervals.parquet"
        if intervals.exists():
            frames.append(pd.read_parquet(intervals))
        staging = directory / "_suspension"
        if staging.is_dir():
            for path in staging.glob("*.parquet"):
                try:
                    covered.add(pd.Timestamp(path.stem))
                except ValueError:
                    continue
    combined = (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame(columns=["symbol", "effective_start", "effective_end",
                                           "suspension_reason"]))
    if len(combined):
        combined = combined.drop_duplicates(["symbol", "effective_start", "effective_end"])
    return combined, sorted(covered)


def _classified_gaps(panel, sessions, suspension_intervals, snapshot_dates, delisting_dates,
                     start) -> pd.DataFrame:
    """Classify every gap since each symbol's first bar, enforce the halt chain,
    then trim to ``start`` (trimming first would hide a broken chain)."""
    gaps = find_session_gaps(panel, sessions)
    classified = classify_session_gaps(
        gaps, suspension_intervals=suspension_intervals,
        snapshot_dates=snapshot_dates, delisting_dates=delisting_dates)
    classified = enforce_halt_chain(classified, panel)
    if start is not None and len(classified):
        classified = classified[classified["trade_date"] >= pd.Timestamp(start).normalize()]
    return classified.reset_index(drop=True)


def augment_with_suspended_rows(
    panel: pd.DataFrame,
    *,
    sessions: Iterable[Any],
    suspension_intervals: pd.DataFrame | None,
    snapshot_dates: Iterable[Any] | None,
    delisting_dates: Mapping[str, Any] | pd.Series | None = None,
    start: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(panel + explicit SUSPENDED rows, classified gaps)``.

    Unexplained and delisted gaps are reported, not filled: a consumer holding
    or targeting such a name has no bar and must fail closed.
    """
    sessions = list(sessions)
    classified = _classified_gaps(panel, sessions, suspension_intervals, snapshot_dates,
                                  delisting_dates, start)
    rows = suspended_execution_rows(panel, classified)
    if rows.empty:
        return panel.copy(), classified
    rows = rows[list(panel.columns)]
    out = pd.concat([panel, rows.astype(panel.dtypes.to_dict())], ignore_index=True)
    out = out.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)
    if out.duplicated(["symbol", "trade_date"]).any():
        raise SessionGapError("suspended rows collide with traded bars")
    return out, classified


#: Per-date bar coverage below this fraction of the expected in-life set fails.
MIN_DATE_COVERAGE = 0.995
DATED_ST_PROVENANCE = "dated_pit_st_evidence"


def panel_integrity(
    panel: pd.DataFrame,
    *,
    sessions: Iterable[Any],
    suspension_intervals: pd.DataFrame | None,
    snapshot_dates: Iterable[Any] | None,
    delisting_dates: Mapping[str, Any] | pd.Series | None,
    start: Any,
) -> dict[str, Any]:
    """Whether a traded-sessions-only panel is complete on ``[start, max]``.

    "Current" (max date) is not "complete": this reports, from the panel itself,
    per-date coverage (bars + evidenced halts over bars + all non-delisted
    gaps), unexplained
    interior holes, trailing gaps of names that stopped printing without a dated
    delisting, and ST rows whose provenance is not dated PIT evidence. Any of
    them makes ``status = "INCOMPLETE"`` with a named problem list.
    """
    frame = panel.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    start = pd.Timestamp(start).normalize()
    classified = _classified_gaps(frame, sessions, suspension_intervals, snapshot_dates,
                                  delisting_dates, start)
    window = frame[frame["trade_date"] >= start]
    bars = window.groupby("trade_date").size()
    live_gaps = classified[classified["classification"] != GAP_DELISTED]
    expected = bars.add(live_gaps.groupby("trade_date").size(), fill_value=0)
    # a halted name legitimately has no bar: it counts as accounted for
    suspended = classified[classified["classification"] == GAP_SUSPENDED].groupby("trade_date").size()
    accounted = bars.reindex(expected.index).fillna(0).add(suspended, fill_value=0)
    coverage = (accounted.reindex(expected.index) / expected).sort_index()
    unexplained = classified[classified["classification"] == GAP_UNEXPLAINED]
    interior_unexplained = unexplained[unexplained["gap_kind"] == GAP_KIND_INTERIOR]
    trailing_unexplained = unexplained[unexplained["gap_kind"] == GAP_KIND_TRAILING]
    report: dict[str, Any] = {
        "window": [str(start.date()), str(frame["trade_date"].max().date())],
        "sessions": int(len(bars)),
        "min_date_coverage": round(float(coverage.min()), 6) if len(coverage) else None,
        "dates_below_coverage": [str(d.date()) for d in coverage[coverage < MIN_DATE_COVERAGE].index],
        "gaps": gap_summary(classified),
        "interior_unexplained": int(len(interior_unexplained)),
        "interior_unexplained_symbols": int(interior_unexplained["symbol"].nunique()),
        "trailing_unexplained_symbols": int(trailing_unexplained["symbol"].nunique()),
    }
    problems: list[str] = []
    if report["dates_below_coverage"]:
        problems.append(f"DATE_COVERAGE_BELOW_{MIN_DATE_COVERAGE} on "
                        f"{len(report['dates_below_coverage'])} sessions "
                        f"(min {report['min_date_coverage']})")
    if len(interior_unexplained):
        problems.append(f"INTERIOR_HOLES_UNEXPLAINED {len(interior_unexplained)} "
                        f"({report['interior_unexplained_symbols']} symbols)")
    if len(trailing_unexplained):
        problems.append(f"SYMBOLS_STOPPED_WITHOUT_EVIDENCE {report['trailing_unexplained_symbols']}")
    if "is_st_provenance" in window.columns:
        provenance = window["is_st_provenance"].astype("string").fillna("<NA>")
        bad = window[provenance != DATED_ST_PROVENANCE]
        report["st_provenance"] = {str(k): int(v) for k, v in provenance.value_counts().items()}
        if len(bad):
            report["st_provenance_unverified_by_exchange"] = {
                str(k): int(v) for k, v in
                bad["symbol"].astype(str).str[-2:].value_counts().items()}
            report["st_provenance_unverified_range"] = [
                str(bad["trade_date"].min().date()), str(bad["trade_date"].max().date())]
            problems.append(f"ST_PROVENANCE_UNVERIFIED {len(bad)} rows "
                            f"{report['st_provenance_unverified_range']}")
    else:
        problems.append("ST_PROVENANCE_COLUMN_MISSING")
    report["problems"] = problems
    report["status"] = "COMPLETE" if not problems else "INCOMPLETE"
    return report
