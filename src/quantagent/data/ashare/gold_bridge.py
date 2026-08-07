"""Bridge the raw U0 daily panel into a full-universe gold training dataset.

The U0 panel is deliberately raw: unadjusted traded prices, traded sessions
only, with PIT metadata living beside it rather than folded in. Training needs
the opposite -- one adjustment scale, eligibility resolved per security-day, and
labels that a book could actually have executed. This module is that
translation, and it is the only sanctioned one.

The failure this design exists to prevent is on record. A previous full-universe
panel mixed qfq and raw prices while declaring its adjustment as "none", and
passed its own audit because the gates were literal ``True`` constants. So:

**One adjustment method, declared and versioned.** The bridge takes exactly one
:data:`ADJUSTMENT_METHODS` value, applies it to every price column, and records
the factor-table content hash in the manifest. Mixing scales is not reachable
through this API -- there is no per-column adjustment argument.

**Eligibility is resolved, not assumed.** Suspension, ST, pre-listing,
post-delisting and new-listing seasoning each produce an explicit mask column.
A day that is missing from a source produces ``UNKNOWN``, never ``False``.

**Availability is a first-class column.** ``has_tick_events`` and friends state
whether a family was *observed* for that security-day. A row without tick data
must never be read as a row with zero order flow, and downstream features are
required to consult the indicator rather than the zero.

**Labels are delay-1 executable.** ``forward_return_{h}d = close(t+1+h) /
close(t+1) - 1``, matching the convention pinned by
``tests/test_executable_label_convention.py``, with entry-infeasible rows
dropped rather than silently kept at a price nobody could have paid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quantagent.data.ashare import contracts

#: The adjustment scales the bridge will apply. Exactly one per build.
ADJUSTMENT_METHODS: tuple[str, ...] = (
    contracts.ADJUST_NONE, contracts.ADJUST_QFQ, contracts.ADJUST_HFQ,
)

#: Price columns the adjustment scale applies to. Volume is *not* adjusted;
#: mixing an adjusted close with a raw volume is the original bug.
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

#: Tri-state mask values. UNKNOWN exists so a gap in a source cannot masquerade
#: as a negative -- "we have no ST register for this exchange" and "this name is
#: not ST" must not produce the same column value.
MASK_TRUE = "TRUE"
MASK_FALSE = "FALSE"
MASK_UNKNOWN = "UNKNOWN"

#: Master `status` values that positively assert the security still trades. Only
#: these turn a missing delisting date into a confident FALSE; anything else —
#: "delisted", "suspended", a blank, an unrecognised value — leaves it UNKNOWN,
#: because the alternative is asserting a security is alive on no evidence.
LISTED_STATUSES: frozenset[str] = frozenset({"listed", "active", "trading", "normal"})

#: Optional feature families. Each contributes a ``has_<family>`` indicator.
OPTIONAL_FAMILIES: tuple[str, ...] = (
    "tick_events", "level2_snapshot", "level2_order_events", "minute_bars",
)

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 20)
#: Trading days a newly listed name must season before it is trainable. The IPO
#: no-limit window plus a settling period; entering on day 1 backtests a price
#: regime that does not repeat.
DEFAULT_SEASONING_DAYS = 20


class GoldBridgeError(RuntimeError):
    """Raised when the bridge is asked to build something incoherent."""


@dataclass
class GoldBuildManifest:
    generated_at: str
    source_commit: str
    adjustment_method: str
    adjustment_factor_version: str
    rows: int
    symbols: int
    date_range: tuple[str, str]
    horizons: list[int]
    seasoning_days: int
    feature_columns: list[str] = field(default_factory=list)
    mask_columns: list[str] = field(default_factory=list)
    availability_columns: list[str] = field(default_factory=list)
    label_columns: list[str] = field(default_factory=list)
    label_convention: str = ""
    feature_coverage: dict[str, float] = field(default_factory=dict)
    mask_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    rows_dropped: dict[str, int] = field(default_factory=dict)
    content_hash: str = ""
    rebuild_command: str = ""
    inputs: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: str | Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()[:16]


def _frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.reindex(sorted(frame.columns), axis=1)
    payload = pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# adjustment
# ---------------------------------------------------------------------------
def apply_adjustment(
    panel: pd.DataFrame, factors: pd.DataFrame, *, method: str
) -> pd.DataFrame:
    """Apply exactly one adjustment scale to every price column.

    ``factors`` carries cumulative backward (hfq) factors keyed by
    ``(symbol, effective_date)``. The factor in force on a trade date is the
    most recent one at or before it -- a merge_asof, never an interpolation,
    because an adjustment factor is a step function.
    """
    if method not in ADJUSTMENT_METHODS:
        raise GoldBridgeError(
            f"unknown adjustment method {method!r}; expected one of {list(ADJUSTMENT_METHODS)}"
        )
    result = panel.copy()
    result["adjustment_method"] = method
    if method == contracts.ADJUST_NONE:
        result["adjust_factor"] = 1.0
        return result

    if factors.empty:
        raise GoldBridgeError(
            f"adjustment method {method!r} requested but no factor table supplied; "
            "refusing to emit prices whose declared scale is not the applied scale"
        )

    left = result.sort_values("trade_date")
    right = (
        factors[["symbol", "effective_date", "hfq_factor"]]
        .dropna(subset=["effective_date"])
        .sort_values("effective_date")
    )
    left["trade_date"] = pd.to_datetime(left["trade_date"])
    right["effective_date"] = pd.to_datetime(right["effective_date"])

    merged = pd.merge_asof(
        left, right, left_on="trade_date", right_on="effective_date",
        by="symbol", direction="backward",
    )
    merged["hfq_factor"] = merged["hfq_factor"].fillna(1.0)

    if method == contracts.ADJUST_HFQ:
        scale = merged["hfq_factor"]
    else:
        # qfq re-bases the backward series on each symbol's latest factor, so
        # the most recent price equals the traded price.
        latest = merged.groupby("symbol")["hfq_factor"].transform("last")
        scale = merged["hfq_factor"] / latest

    for column in PRICE_COLUMNS:
        if column in merged.columns:
            merged[column] = merged[column] * scale
    merged["adjust_factor"] = scale
    # Volume and amount are deliberately untouched: they are traded quantities,
    # and scaling them to match adjusted prices is the mixed-scale bug.
    return merged.drop(columns=["effective_date"], errors="ignore")


# ---------------------------------------------------------------------------
# eligibility masks
# ---------------------------------------------------------------------------
def _interval_mask(
    panel: pd.DataFrame, intervals: pd.DataFrame, *, available: bool
) -> pd.Series:
    """Tri-state membership of each panel row in a set of dated intervals."""
    if not available:
        return pd.Series(MASK_UNKNOWN, index=panel.index, dtype="object")
    mask = pd.Series(MASK_FALSE, index=panel.index, dtype="object")
    if intervals.empty:
        return mask

    dates = pd.to_datetime(panel["trade_date"])
    starts = pd.to_datetime(intervals["effective_start"])
    ends = pd.to_datetime(intervals["effective_end"]).fillna(pd.Timestamp.max)
    by_symbol: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for symbol, start, end in zip(intervals["symbol"], starts, ends):
        by_symbol.setdefault(str(symbol), []).append((start, end))

    for symbol, group in panel.groupby("symbol", sort=False):
        windows = by_symbol.get(str(symbol))
        if not windows:
            continue
        group_dates = dates.loc[group.index]
        hit = pd.Series(False, index=group.index)
        for start, end in windows:
            hit |= (group_dates >= start) & (group_dates <= end)
        mask.loc[group.index[hit]] = MASK_TRUE
    return mask


#: Master `source` values that identify a security fetched from an exchange's
#: *delisting* register rather than its listing register. Provenance is the only
#: place U0's master records the distinction: `status_end` (the delisting date) is
#: empty for all 5,888 rows, so the fact that a name is dead survives only in where
#: it was found.
DELISTING_REGISTER_SOURCES: frozenset[str] = frozenset(
    {"sz_delist", "sz_delist_retry", "sh_delist", "sh_delist_retry", "bj_delist"}
)

#: What `resolve_listing_status` concluded, and from which column.
LISTING_STATUS_LISTED = "listed"
LISTING_STATUS_DELISTED = "delisted"
LISTING_STATUS_UNKNOWN = "unknown"


def resolve_listing_status(master: pd.DataFrame) -> tuple[pd.Series, str]:
    """Decide, per security, whether the master says it is listed or dead.

    DEF-024 taught `build_masks` to consult a `status` column so that a missing
    delisting date would not be read as "confidently never delisted". The U0
    master has no `status` column, so on the real master that fix could only ever
    answer UNKNOWN — for all 5,888 names, including the 5,530 the master sources
    from live listing registers. Honest, but inert: it cannot tell the 358 names it
    exists to catch from the 5,530 it does not need to.

    The master does carry the distinction, in two agreeing columns:
    `status_end_blocked` (True for exactly the 358) and `source` (`sz_delist` /
    `sh_delist_retry`). Reading them turns the mask from "nothing is knowable" into
    "these 358 are dead and undated, the rest are alive" — which is the difference
    between a gate that can never pass and a gate that names what is missing.

    Returns the per-symbol status and the column it was derived from, so a caller
    can record *how* it knows rather than asserting the conclusion.
    """
    index = master["symbol"].astype(str)
    if "status" in master.columns:
        status = master["status"].astype("object").str.lower()
        resolved = pd.Series(LISTING_STATUS_UNKNOWN, index=index, dtype="object")
        resolved[status.isin(LISTED_STATUSES).to_numpy()] = LISTING_STATUS_LISTED
        resolved[status.astype(str).str.contains("delist", na=False).to_numpy()] = (
            LISTING_STATUS_DELISTED
        )
        return resolved, "status"

    from_delisting_register = (
        master["source"].astype(str).isin(DELISTING_REGISTER_SOURCES)
        if "source" in master.columns
        else pd.Series(False, index=master.index)
    )
    blocked = (
        master["status_end_blocked"].fillna(False).astype(bool)
        if "status_end_blocked" in master.columns
        else pd.Series(False, index=master.index)
    )
    if "source" not in master.columns and "status_end_blocked" not in master.columns:
        return pd.Series(LISTING_STATUS_UNKNOWN, index=index, dtype="object"), "none"

    dead = (from_delisting_register | blocked).to_numpy()
    resolved = pd.Series(LISTING_STATUS_LISTED, index=index, dtype="object")
    resolved[dead] = LISTING_STATUS_DELISTED
    basis = "+".join(
        name for name, present in
        (("source", "source" in master.columns),
         ("status_end_blocked", "status_end_blocked" in master.columns))
        if present
    )
    return resolved, basis


def build_masks(
    panel: pd.DataFrame,
    *,
    master: pd.DataFrame,
    suspension: pd.DataFrame | None = None,
    st: pd.DataFrame | None = None,
    st_available: bool = False,
    seasoning_days: int = DEFAULT_SEASONING_DAYS,
) -> pd.DataFrame:
    """Attach the tri-state eligibility masks to the panel.

    ``st_available`` is a deliberate parameter rather than an inference from
    whether the frame is empty: U0's ST register covers SZSE only, so a partial
    source must produce UNKNOWN for the exchanges it does not cover instead of
    a confident FALSE.
    """
    result = panel.copy()
    dates = pd.to_datetime(result["trade_date"])

    result["mask_is_suspended"] = _interval_mask(
        result, suspension if suspension is not None else pd.DataFrame(),
        available=suspension is not None,
    )
    result["mask_is_st"] = _interval_mask(
        result, st if st is not None else pd.DataFrame(), available=st_available
    )

    identity = master.set_index("symbol")
    listing = result["symbol"].map(identity.get("listing_date", pd.Series(dtype=object)))
    delisting = result["symbol"].map(identity.get("delisting_date", pd.Series(dtype=object)))
    listing = pd.to_datetime(listing, errors="coerce")
    delisting = pd.to_datetime(delisting, errors="coerce")

    result["mask_pre_listing"] = np.where(
        listing.isna(), MASK_UNKNOWN,
        np.where(dates < listing, MASK_TRUE, MASK_FALSE),
    )
    # A missing delisting date means two different things, and conflating them is
    # survivorship bias by default. If the master says the security is still
    # listed, no delisting date is a confident FALSE. If it says *delisted* and the
    # date was never captured, we do not know when the name stopped trading, so the
    # honest mask is UNKNOWN for its whole history — a FALSE there made a dead name
    # contribute exactly as many eligible training sessions as a live one
    # (DEF-024). Note the asymmetry this removes: `mask_pre_listing` already
    # returned UNKNOWN for a missing listing date.
    #
    # The status is *resolved* from whatever the master actually carries rather
    # than read from a column name assumed to exist — see `resolve_listing_status`.
    # Measured on U0's master: reading only `status` left all 5,888 names UNKNOWN,
    # because that column is not there; reading provenance identifies the 358 that
    # came from delisting registers and clears the 5,530 that did not.
    listing_status, status_basis = resolve_listing_status(master)
    status = result["symbol"].astype(str).map(listing_status)
    known_listed = (status == LISTING_STATUS_LISTED).to_numpy()
    result["mask_post_delisting"] = np.where(
        delisting.notna(),
        np.where(dates > delisting, MASK_TRUE, MASK_FALSE),
        np.where(known_listed, MASK_FALSE, MASK_UNKNOWN),
    )
    result.attrs["listing_status_basis"] = status_basis
    # Published as a column, not just consumed here. `mask_post_delisting` answers
    # "is this row after the security died", which is FALSE both for a live name and
    # for a dead name's rows *before* it died — so the mask alone cannot say whether
    # the panel contains dead names at all. That ambiguity made the survivorship
    # audit read a correctly-built panel as the most suspicious kind (DEF-028).
    result["listing_status"] = status.fillna(LISTING_STATUS_UNKNOWN)

    # Seasoning counts *trading sessions observed in the panel*, not calendar
    # days, so holidays cannot shorten the window.
    session_index = result.sort_values("trade_date").groupby("symbol").cumcount()
    seasoning = pd.Series(MASK_FALSE, index=result.index, dtype="object")
    seasoning.loc[session_index.index[session_index < seasoning_days]] = MASK_TRUE
    result["mask_seasoning"] = seasoning

    mask_columns = [
        "mask_is_suspended", "mask_is_st", "mask_pre_listing",
        "mask_post_delisting", "mask_seasoning",
    ]
    # `eligible_for_training` keeps its permissive meaning — "not *known* to be
    # ineligible" — because flipping UNKNOWN to ineligible would empty the universe
    # wherever a register has partial coverage (U0's ST data is SZSE-only). But a
    # single boolean that silently absorbs UNKNOWN either way is how a caller ends
    # up unable to tell "verified eligible" from "we could not check". So the
    # tri-state is published alongside it, and `unknown_masks` names which checks
    # could not be made, so a gate can enforce a policy on the difference rather
    # than inferring one.
    result["eligible_for_training"] = np.logical_and.reduce(
        [result[column] != MASK_TRUE for column in mask_columns]
    )
    any_true = np.logical_or.reduce([result[column] == MASK_TRUE for column in mask_columns])
    any_unknown = np.logical_or.reduce(
        [result[column] == MASK_UNKNOWN for column in mask_columns]
    )
    result["eligibility_status"] = np.where(
        any_true, MASK_FALSE, np.where(any_unknown, MASK_UNKNOWN, MASK_TRUE)
    )
    # Built column-wise. The row-wise `iterrows()` this replaces ran at ~30k
    # rows/s, i.e. ~6 minutes on the 10.9M-row full-universe panel — and wiring
    # survivorship into the training path (M5-02) makes that a cost paid on every
    # run rather than once at build time. Same output, verified by test.
    unknown_parts = pd.Series("", index=result.index, dtype="object")
    for column in mask_columns:
        name = column.removeprefix("mask_")
        is_unknown = result[column].to_numpy() == MASK_UNKNOWN
        unknown_parts = unknown_parts.where(
            ~is_unknown, unknown_parts.str.cat(pd.Series(name, index=result.index), sep=",")
        )
    result["unknown_masks"] = unknown_parts.str.lstrip(",")
    return result


# ---------------------------------------------------------------------------
# availability indicators
# ---------------------------------------------------------------------------
def attach_availability(
    panel: pd.DataFrame,
    observed: Mapping[str, pd.DataFrame] | None = None,
    families: Sequence[str] = OPTIONAL_FAMILIES,
) -> pd.DataFrame:
    """Mark, per security-day, which optional families were actually observed.

    This is the column that stops a missing tick day from being read as a quiet
    one. ``observed`` maps a family name to a frame carrying ``symbol`` and
    ``trade_date``; anything absent from it is ``False`` for that family, which
    here genuinely means "not observed" rather than "observed as zero".
    """
    result = panel.copy()
    observed = observed or {}
    keys = pd.Series(
        result["symbol"].astype(str) + "|"
        + pd.to_datetime(result["trade_date"]).dt.strftime("%Y-%m-%d"),
        index=result.index,
    )
    for family in families:
        frame = observed.get(family)
        column = f"has_{family}"
        if frame is None or frame.empty:
            result[column] = False
            continue
        seen = set(
            frame["symbol"].astype(str) + "|"
            + pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
        )
        result[column] = keys.isin(seen)
    return result


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
LABEL_CONVENTION = (
    "forward_return_{h}d = close(t+1+h) / close(t+1) - 1 (delay-1 executable); "
    "rows are dropped when entry at t+1 was infeasible"
)


def build_labels(
    panel: pd.DataFrame, *, horizons: Sequence[int] = DEFAULT_HORIZONS
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach delay-1 executable forward returns and drop infeasible entries.

    Entry is infeasible when the security is suspended at ``t`` or ``t+1``, is
    ST at ``t``, or is sealed at limit-up at ``t+1`` -- you cannot buy a locked
    limit-up, and pretending otherwise is where the old phantom alpha came from.
    """
    result = panel.sort_values(["symbol", "trade_date"]).copy()
    grouped = result.groupby("symbol", sort=False)

    entry_price = grouped["close"].shift(-1)
    result["entry_close_t1"] = entry_price

    for horizon in horizons:
        exit_price = grouped["close"].shift(-(1 + horizon))
        result[f"forward_return_{horizon}d"] = exit_price / entry_price - 1.0

    infeasible = pd.Series(False, index=result.index)
    reasons: dict[str, int] = {}

    if "mask_is_suspended" in result.columns:
        suspended_now = result["mask_is_suspended"] == MASK_TRUE
        suspended_next = grouped["mask_is_suspended"].shift(-1) == MASK_TRUE
        reasons["suspended_at_t"] = int(suspended_now.sum())
        reasons["suspended_at_t1"] = int(suspended_next.sum())
        infeasible |= suspended_now | suspended_next
    if "mask_is_st" in result.columns:
        is_st = result["mask_is_st"] == MASK_TRUE
        reasons["st_at_t"] = int(is_st.sum())
        infeasible |= is_st
    if "mask_limit_up" in result.columns:
        sealed = grouped["mask_limit_up"].shift(-1) == MASK_TRUE
        reasons["limit_up_at_t1"] = int(sealed.sum())
        infeasible |= sealed

    reasons["entry_price_missing"] = int(entry_price.isna().sum())
    infeasible |= entry_price.isna()

    result["entry_feasible"] = ~infeasible
    kept = result.loc[~infeasible].copy()
    reasons["rows_dropped_total"] = int(len(result) - len(kept))
    return kept, reasons


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def build_gold_dataset(
    panel: pd.DataFrame,
    *,
    master: pd.DataFrame,
    factors: pd.DataFrame | None = None,
    suspension: pd.DataFrame | None = None,
    st: pd.DataFrame | None = None,
    st_available: bool = False,
    observed_families: Mapping[str, pd.DataFrame] | None = None,
    adjustment_method: str = contracts.ADJUST_HFQ,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    seasoning_days: int = DEFAULT_SEASONING_DAYS,
    source_commit: str = "unknown",
    rebuild_command: str = "",
    inputs: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, GoldBuildManifest]:
    """Run the whole raw-to-gold bridge and return the dataset plus manifest."""
    if panel.empty:
        raise GoldBridgeError("cannot build a gold dataset from an empty panel")

    adjusted = apply_adjustment(
        panel, factors if factors is not None else pd.DataFrame(),
        method=adjustment_method,
    )
    masked = build_masks(
        adjusted, master=master, suspension=suspension, st=st,
        st_available=st_available, seasoning_days=seasoning_days,
    )
    available = attach_availability(masked, observed_families)
    labelled, dropped = build_labels(available, horizons=horizons)

    mask_columns = [c for c in labelled.columns if c.startswith("mask_")]
    availability_columns = [c for c in labelled.columns if c.startswith("has_")]
    label_columns = [c for c in labelled.columns if c.startswith("forward_return_")]
    reserved = set(mask_columns) | set(availability_columns) | set(label_columns) | {
        "symbol", "trade_date", "source", "source_endpoint", "retrieved_at",
        "available_at", "quality_status", "serving_provider", "adjustment_method",
        "adjust_factor", "hfq_factor", "entry_close_t1", "entry_feasible",
        "eligible_for_training",
    }
    feature_columns = [c for c in labelled.columns if c not in reserved]

    warnings: list[str] = []
    if not st_available:
        warnings.append(
            "ST intervals were not available as a complete dated register, so "
            "mask_is_st is UNKNOWN and no row can be excluded on ST grounds; "
            "any dataset built this way is NOT point-in-time complete"
        )
    if adjustment_method == contracts.ADJUST_NONE:
        warnings.append(
            "adjustment_method is 'none': prices are raw traded prices and "
            "cross-date returns will contain ex-rights jumps"
        )

    manifest = GoldBuildManifest(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_commit=source_commit,
        adjustment_method=adjustment_method,
        adjustment_factor_version=(
            _frame_hash(factors[["symbol", "effective_date", "hfq_factor"]])
            if factors is not None and not factors.empty else "none"
        ),
        rows=len(labelled),
        symbols=int(labelled["symbol"].nunique()),
        date_range=(
            str(pd.to_datetime(labelled["trade_date"]).min().date()),
            str(pd.to_datetime(labelled["trade_date"]).max().date()),
        ),
        horizons=list(horizons),
        seasoning_days=seasoning_days,
        feature_columns=feature_columns,
        mask_columns=mask_columns,
        availability_columns=availability_columns,
        label_columns=label_columns,
        label_convention=LABEL_CONVENTION,
        feature_coverage={
            column: float(labelled[column].notna().mean())
            for column in feature_columns
        },
        mask_distribution={
            column: {
                str(k): int(v) for k, v in labelled[column].value_counts().items()
            }
            for column in mask_columns
        },
        rows_dropped=dropped,
        content_hash=_frame_hash(labelled),
        rebuild_command=rebuild_command,
        inputs=dict(inputs or {}),
        warnings=warnings,
    )
    return labelled, manifest


@dataclass
class TrainingSliceCertificate:
    """Whether the produced dataset may be trained on, and why or why not."""

    generated_at: str
    dataset_content_hash: str
    training_permitted: bool
    decision: str
    blockers: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def certify_training_slice(
    manifest: GoldBuildManifest, *, u0_pit_certificate: Mapping[str, Any] | None = None
) -> TrainingSliceCertificate:
    """Gate training on real evidence, not on the build having succeeded.

    A dataset that builds cleanly is not a dataset that may be trained on. The
    U0 PIT certificate is the authority; when it withholds permission, so does
    this. Deriving the answer rather than restating a constant is the whole
    point -- the previous generation of these gates were literal ``True``.
    """
    blockers: list[str] = []
    evidence: dict[str, Any] = {"manifest_warnings": list(manifest.warnings)}

    if u0_pit_certificate is None:
        blockers.append(
            "no U0 PIT certificate supplied; absence of evidence is not evidence "
            "of readiness"
        )
    else:
        evidence["u0_decision"] = u0_pit_certificate.get("decision")
        evidence["u0_training_permitted"] = u0_pit_certificate.get("training_permitted")
        evidence["u0_blocked_pit_fields"] = u0_pit_certificate.get("blocked_pit_fields", [])
        if not u0_pit_certificate.get("training_permitted", False):
            blockers.append(
                f"U0 PIT gate withholds training permission: "
                f"{u0_pit_certificate.get('decision')} "
                f"(blocked fields: {u0_pit_certificate.get('blocked_pit_fields')})"
            )

    if any("NOT point-in-time complete" in w for w in manifest.warnings):
        blockers.append("gold build reported an incomplete PIT mask")
    if manifest.rows == 0:
        blockers.append("dataset is empty")

    permitted = not blockers
    return TrainingSliceCertificate(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset_content_hash=manifest.content_hash,
        training_permitted=permitted,
        decision="TRAINING_PERMITTED" if permitted else "TRAINING_BLOCKED",
        blockers=blockers,
        evidence=evidence,
    )
