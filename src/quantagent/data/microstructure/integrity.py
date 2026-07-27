"""Integrity forensics for canonical microstructure events.

Every check returns a verdict rather than raising, because the caller -- the
Data Quality agent -- needs the *whole* picture to decide, and a dataset that
fails one check often fails several in a way that identifies the root cause.

Verdicts are deliberately three-valued:

``PASS``       the property holds on the data examined.
``WARN``       the property is violated in a way that has a benign explanation
               (a session boundary, a vendor's known snapshot cadence) but that
               a human should see.
``FAIL``       the property is violated in a way that invalidates downstream
               microstructure conclusions.
``NOT_RUN``    the input needed to evaluate the check is absent. **Never**
               treated as a pass -- this is the failure mode that produced the
               old "gates are literal True constants" bug in the U0 layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from quantagent.data.microstructure import contracts

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"


@dataclass
class CheckResult:
    check: str
    verdict: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IntegrityReport:
    family: str
    data_class: str
    rows: int
    symbols: int
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def verdict_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for check in self.checks:
            counts[check.verdict] = counts.get(check.verdict, 0) + 1
        return counts

    @property
    def failed(self) -> list[str]:
        return [c.check for c in self.checks if c.verdict == FAIL]

    @property
    def not_run(self) -> list[str]:
        return [c.check for c in self.checks if c.verdict == NOT_RUN]

    @property
    def usable(self) -> bool:
        """A dataset is usable only if nothing failed *and* nothing was skipped."""
        return not self.failed and not self.not_run

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "data_class": self.data_class,
            "rows": self.rows,
            "symbols": self.symbols,
            "verdict_counts": self.verdict_counts,
            "failed_checks": self.failed,
            "not_run_checks": self.not_run,
            "usable": self.usable,
            "checks": [c.to_dict() for c in self.checks],
        }


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
def check_schema(frame: pd.DataFrame, family: str) -> CheckResult:
    contract = contracts.contract_for(family)
    missing = contract.missing_columns(frame.columns)
    if missing:
        return CheckResult(
            "schema_columns", FAIL,
            f"{len(missing)} declared {family} columns absent",
            {"missing": missing},
        )
    return CheckResult(
        "schema_columns", PASS, "frame carries the declared event contract columns",
        {"columns": list(contract.columns)},
    )


def check_declared_semantics(frame: pd.DataFrame) -> CheckResult:
    if "data_class" not in frame.columns:
        return CheckResult(
            "declared_semantics", NOT_RUN, "frame carries no data_class column", {}
        )
    classes = sorted(set(frame["data_class"].dropna().astype(str)))
    unknown = [c for c in classes if c not in contracts.DATA_CLASSES]
    if unknown:
        return CheckResult(
            "declared_semantics", FAIL, "undeclared data_class values present",
            {"unknown": unknown},
        )
    if contracts.UNKNOWN_SEMANTICS in classes:
        return CheckResult(
            "declared_semantics", FAIL,
            "provider semantics are unknown; the dataset cannot be interpreted",
            {"classes": classes},
        )
    if len(classes) > 1:
        return CheckResult(
            "declared_semantics", WARN,
            "frame mixes multiple data classes; downstream must not aggregate them",
            {"classes": classes},
        )
    return CheckResult(
        "declared_semantics", PASS, f"single declared data class {classes[0]}",
        {"classes": classes},
    )


def check_timestamp_monotonicity(frame: pd.DataFrame) -> CheckResult:
    if "event_time_ns" not in frame.columns or frame.empty:
        return CheckResult("timestamp_monotonicity", NOT_RUN, "no event_time_ns", {})
    regressions = 0
    worst_ns = 0
    for _, group in frame.groupby("symbol", sort=False):
        series = group.sort_values("ingest_sequence")["event_time_ns"].to_numpy()
        deltas = np.diff(series)
        negative = deltas[deltas < 0]
        regressions += int(negative.size)
        if negative.size:
            worst_ns = max(worst_ns, int(-negative.min()))
    if regressions == 0:
        return CheckResult(
            "timestamp_monotonicity", PASS,
            "event time never moves backwards within a symbol", {"regressions": 0},
        )
    verdict = FAIL if worst_ns > 1_000_000_000 else WARN
    return CheckResult(
        "timestamp_monotonicity", verdict,
        f"{regressions} out-of-order events, worst regression {worst_ns / 1e9:.3f}s",
        {"regressions": regressions, "worst_regression_ns": worst_ns},
    )


def check_duplicate_events(frame: pd.DataFrame, family: str) -> CheckResult:
    contract = contracts.contract_for(family)
    identity = [c for c in ("trade_id", "order_id", "snapshot_sequence") if c in frame.columns]
    identity = [c for c in identity if frame[c].notna().any()]
    if not identity:
        return CheckResult(
            "duplicate_events", NOT_RUN,
            "source publishes no event identifier, so duplicates cannot be "
            "distinguished from genuinely repeated observations",
            {"contract": contract.family},
        )
    key = ["symbol", "trade_date", *identity]
    duplicated = int(frame.duplicated(subset=key).sum())
    if duplicated == 0:
        return CheckResult(
            "duplicate_events", PASS, "no repeated event identifiers", {"key": key}
        )
    return CheckResult(
        "duplicate_events", FAIL, f"{duplicated} duplicate event identifiers",
        {"key": key, "duplicates": duplicated},
    )


def check_sequence_gaps(frame: pd.DataFrame) -> CheckResult:
    if "sequence" not in frame.columns or frame["sequence"].isna().all():
        return CheckResult(
            "sequence_gaps", NOT_RUN,
            "provider publishes no exchange sequence number; completeness "
            "cannot be proven from the stream alone",
            {},
        )
    gaps: list[dict[str, Any]] = []
    for symbol, group in frame.groupby("symbol", sort=True):
        series = group["sequence"].dropna().astype("int64").sort_values().to_numpy()
        if series.size < 2:
            continue
        deltas = np.diff(series)
        missing = int(deltas[deltas > 1].sum() - (deltas > 1).sum())
        if missing:
            gaps.append({"symbol": str(symbol), "missing_sequences": missing,
                         "gap_count": int((deltas > 1).sum())})
    if not gaps:
        return CheckResult("sequence_gaps", PASS, "exchange sequences are contiguous", {})
    total = sum(g["missing_sequences"] for g in gaps)
    return CheckResult(
        "sequence_gaps", FAIL,
        f"{total} sequence numbers missing across {len(gaps)} symbols",
        {"symbols_with_gaps": gaps[:50], "total_missing": total},
    )


def check_manufactured_fields(frame: pd.DataFrame, family: str) -> CheckResult:
    """Catch identifiers that look synthesised rather than vendor-published.

    A dense 0..N-1 range in a column the contract marks ``never_manufactured``
    is the signature of someone filling in an id the source never gave.
    """
    contract = contracts.contract_for(family)
    suspects: list[str] = []
    for column in contract.never_manufactured:
        if column not in frame.columns:
            continue
        values = frame[column].dropna()
        if values.empty or not pd.api.types.is_numeric_dtype(values):
            continue
        as_int = values.astype("int64")
        if as_int.min() == 0 and as_int.max() == len(as_int) - 1 and as_int.nunique() == len(as_int):
            suspects.append(column)
    if suspects:
        return CheckResult(
            "manufactured_identifiers", FAIL,
            "columns look like a generated range rather than vendor identifiers",
            {"suspect_columns": suspects},
        )
    return CheckResult(
        "manufactured_identifiers", PASS,
        "no contract-protected identifier looks synthesised",
        {"inspected": list(contract.never_manufactured)},
    )


def check_side_provenance(frame: pd.DataFrame) -> CheckResult:
    if "side" not in frame.columns:
        return CheckResult("side_provenance", NOT_RUN, "family carries no side column", {})
    if "side_method" not in frame.columns:
        return CheckResult(
            "side_provenance", FAIL,
            "trade side present without side_method; an inferred direction "
            "must name the rule that produced it", {},
        )
    stated = frame["side"].notna()
    if not bool(stated.any()):
        return CheckResult(
            "side_provenance", PASS, "no trade direction claimed",
            {"observed_rows": 0, "inferred_rows": 0},
        )
    methods = frame.loc[stated, "side_method"]
    invalid = int((~methods.isin(contracts.SIDE_METHODS)).sum())
    if invalid:
        return CheckResult(
            "side_provenance", FAIL, f"{invalid} rows carry an unknown side_method",
            {"methods": sorted(set(methods.dropna().astype(str)))},
        )
    observed = int(methods.isin(contracts.OBSERVED_SIDE_METHODS).sum())
    inferred = int(stated.sum()) - observed
    verdict = PASS if inferred == 0 else WARN
    return CheckResult(
        "side_provenance", verdict,
        f"{observed} observed sides, {inferred} inferred sides"
        + ("" if inferred == 0 else " — inferred direction is not an observation"),
        {"observed_rows": observed, "inferred_rows": inferred,
         "methods": sorted(set(methods.dropna().astype(str)))},
    )


def check_session_boundaries(frame: pd.DataFrame, *, board: str | None = None) -> CheckResult:
    """Classify every event into a session phase, per-symbol board-aware.

    Grading is three-way on purpose:

    ``FAIL``  events land in the lunch break or fully outside the trading day.
              Both mean the timestamps do not describe an A-share session.
    ``WARN``  events land in the post-close window. They exist in the vendor
              stream, are absent from the exchange daily bar, and their nature
              is not established, so they are surfaced and quarantined rather
              than accepted or dismissed.
    ``PASS``  everything sits in a real trading phase.
    """
    if "exchange_time" not in frame.columns or frame.empty:
        return CheckResult("session_boundaries", NOT_RUN, "no exchange_time column", {})
    times = pd.to_datetime(frame["exchange_time"], errors="coerce")
    if times.isna().all():
        return CheckResult(
            "session_boundaries", NOT_RUN, "exchange_time is not parseable", {}
        )

    clock = times.dt.strftime("%H:%M")
    if board is None and "symbol" in frame.columns:
        # STAR has an after-hours window nothing else has, so a board-blind
        # classifier would mislabel legitimate STAR prints as unexplained.
        boards = frame["symbol"].astype(str).map(contracts.board_of)
        phases = pd.Series(
            [contracts.session_phase(c, board=b) for c, b in zip(clock, boards)],
            index=frame.index,
        )
    else:
        phases = clock.map(lambda hhmm: contracts.session_phase(hhmm, board=board))

    counts = phases.value_counts().to_dict()
    outside = int(counts.get(contracts.PHASE_CLOSED, 0))
    lunch = int(counts.get(contracts.PHASE_LUNCH_BREAK, 0))
    post_close = int(counts.get(contracts.PHASE_POST_CLOSE, 0))
    evidence = {
        "phase_counts": {str(k): int(v) for k, v in counts.items()},
        "outside_session_rows": outside,
        "lunch_break_rows": lunch,
        "post_close_rows": post_close,
    }
    if outside or lunch:
        return CheckResult(
            "session_boundaries", FAIL,
            f"{outside} events outside the trading day, {lunch} inside the lunch break",
            evidence,
        )
    if post_close:
        share = post_close / len(frame)
        return CheckResult(
            "session_boundaries", WARN,
            f"{post_close} events ({share:.3%}) fall in the post-close window; "
            "they are excluded from the exchange daily bar and their nature is "
            "not established, so they must not enter continuous-session features",
            evidence,
        )
    return CheckResult(
        "session_boundaries", PASS, "every event falls inside a declared session phase",
        evidence,
    )


def check_price_sanity(frame: pd.DataFrame) -> CheckResult:
    price_columns = [c for c in ("price", "bid_price", "ask_price", "last_price")
                     if c in frame.columns]
    if not price_columns:
        return CheckResult("price_sanity", NOT_RUN, "no price column", {})
    problems: dict[str, int] = {}
    for column in price_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = int(((values <= 0) & values.notna()).sum())
        if bad:
            problems[column] = bad
    if problems:
        return CheckResult(
            "price_sanity", FAIL, "non-positive traded or quoted prices present", problems
        )
    return CheckResult("price_sanity", PASS, "all prices strictly positive",
                       {"columns": price_columns})


def check_volume_sanity(frame: pd.DataFrame) -> CheckResult:
    volume_columns = [c for c in frame.columns if c.endswith("volume_shares")]
    if not volume_columns:
        return CheckResult("volume_sanity", NOT_RUN, "no share-volume column", {})
    problems: dict[str, int] = {}
    for column in volume_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = int(((values < 0) & values.notna()).sum())
        if bad:
            problems[column] = bad
    if problems:
        return CheckResult("volume_sanity", FAIL, "negative share volume present", problems)
    return CheckResult("volume_sanity", PASS, "share volume is non-negative",
                       {"columns": volume_columns})


def check_cumulative_monotonicity(frame: pd.DataFrame) -> CheckResult:
    columns = [c for c in ("cum_volume_shares", "cum_amount_cny") if c in frame.columns]
    if not columns:
        return CheckResult("cumulative_monotonicity", NOT_RUN, "no cumulative column", {})
    resets: dict[str, int] = {}
    for column in columns:
        count = 0
        for _, group in frame.groupby(["symbol", "trade_date"], sort=False):
            series = pd.to_numeric(
                group.sort_values("ingest_sequence")[column], errors="coerce"
            ).dropna().to_numpy()
            if series.size > 1:
                count += int((np.diff(series) < 0).sum())
        if count:
            resets[column] = count
    if resets:
        return CheckResult(
            "cumulative_monotonicity", FAIL,
            "session cumulative fields decrease intraday, which means either a "
            "vendor reset or events from two different sessions were merged",
            resets,
        )
    return CheckResult("cumulative_monotonicity", PASS,
                       "session cumulatives are non-decreasing", {"columns": columns})


def check_book_ordering(frame: pd.DataFrame) -> CheckResult:
    needed = {"level", "bid_price", "ask_price"}
    if not needed.issubset(frame.columns) or frame.empty:
        return CheckResult("book_ordering", NOT_RUN, "not a book-snapshot frame", {})
    crossed = 0
    misordered = 0
    group_keys = [c for c in ("symbol", "trade_date", "snapshot_sequence", "event_time_ns")
                  if c in frame.columns]
    for _, snapshot in frame.groupby(group_keys, sort=False):
        snapshot = snapshot.sort_values("level")
        bids = pd.to_numeric(snapshot["bid_price"], errors="coerce").dropna()
        asks = pd.to_numeric(snapshot["ask_price"], errors="coerce").dropna()
        # Level 1 is the inside market; deeper levels must be worse.
        if bids.size > 1 and bool((np.diff(bids.to_numpy()) > 0).any()):
            misordered += 1
        if asks.size > 1 and bool((np.diff(asks.to_numpy()) < 0).any()):
            misordered += 1
        if bids.size and asks.size:
            best_bid = bids.iloc[0]
            best_ask = asks.iloc[0]
            if best_bid > 0 and best_ask > 0 and best_bid >= best_ask:
                crossed += 1
    evidence = {"crossed_or_locked_snapshots": crossed, "misordered_levels": misordered}
    if crossed or misordered:
        return CheckResult(
            "book_ordering", FAIL,
            f"{crossed} crossed/locked snapshots, {misordered} snapshots with "
            "price levels out of order", evidence,
        )
    return CheckResult("book_ordering", PASS,
                       "price levels ordered and inside market never crossed", evidence)


def check_clock_drift(frame: pd.DataFrame) -> CheckResult:
    if not {"event_time_ns", "receive_time_ns"}.issubset(frame.columns) or frame.empty:
        return CheckResult("clock_drift", NOT_RUN, "need both event and receive time", {})
    event = pd.to_numeric(frame["event_time_ns"], errors="coerce")
    receive = pd.to_numeric(frame["receive_time_ns"], errors="coerce")
    delta = (receive - event).dropna()
    if delta.empty:
        return CheckResult("clock_drift", NOT_RUN, "no comparable timestamp pairs", {})
    negative = int((delta < 0).sum())
    evidence = {
        "p50_latency_ms": float(delta.quantile(0.50) / 1e6),
        "p95_latency_ms": float(delta.quantile(0.95) / 1e6),
        "p99_latency_ms": float(delta.quantile(0.99) / 1e6),
        "max_latency_ms": float(delta.max() / 1e6),
        "received_before_exchange_time_rows": negative,
    }
    if negative:
        return CheckResult(
            "clock_drift", FAIL,
            f"{negative} events were received before their exchange timestamp; "
            "the two clocks are not comparable", evidence,
        )
    return CheckResult("clock_drift", PASS,
                       "receive time follows exchange time on every event", evidence)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def run_integrity_checks(
    frame: pd.DataFrame,
    *,
    family: str,
    data_class: str = contracts.UNKNOWN_SEMANTICS,
    board: str | None = None,
) -> IntegrityReport:
    """Run every applicable check and return the collected verdicts."""
    checks: list[CheckResult] = [
        check_schema(frame, family),
        check_declared_semantics(frame),
        check_timestamp_monotonicity(frame),
        check_duplicate_events(frame, family),
        check_sequence_gaps(frame),
        check_manufactured_fields(frame, family),
        check_side_provenance(frame),
        check_session_boundaries(frame, board=board),
        check_price_sanity(frame),
        check_volume_sanity(frame),
        check_cumulative_monotonicity(frame),
        check_book_ordering(frame),
        check_clock_drift(frame),
    ]
    return IntegrityReport(
        family=family,
        data_class=data_class,
        rows=len(frame),
        symbols=int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0,
        checks=checks,
    )
