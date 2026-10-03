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
    #: Per-exchange TRUE/FALSE/UNKNOWN counts on the masked panel *before* the
    #: label step drops infeasible rows -- what each register actually measured.
    mask_rows_by_exchange: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    st_coverage: list[str] = field(default_factory=list)
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
#: Exchange suffixes of mainland cash-equity symbols (``000001.SZ``).
EXCHANGE_SUFFIXES: tuple[str, ...] = ("SH", "SZ", "BJ")

#: Exchange names as U0 PIT manifests spell them -> symbol suffix.
EXCHANGE_NAME_TO_SUFFIX: dict[str, str] = {"SSE": "SH", "SZSE": "SZ", "BSE": "BJ"}

#: What a register may cover: every exchange (``True``), none (``False``), or a
#: named subset of symbol suffixes such as ``{"SZ"}``.
RegisterCoverage = bool | Iterable[str]


def exchange_suffix(symbols: pd.Series) -> pd.Series:
    """``000001.SZ`` -> ``SZ``. A symbol without a suffix maps to ``""``."""
    text = symbols.astype(str).str.upper()
    return text.str.rpartition(".")[2].where(text.str.contains(".", regex=False), "")


def normalise_coverage(available: RegisterCoverage | None) -> frozenset[str]:
    """Resolve a coverage declaration to the exchange suffixes it measures.

    ``True`` keeps its historical meaning (the register is complete for every
    exchange) and ``False``/``None`` means nothing is measured. A collection
    names the exchanges whose rows the register can answer for; exchange names
    (``SZSE``) and suffixes (``SZ``) are both accepted.
    """
    if available is None or available is False:
        return frozenset()
    if available is True:
        return frozenset(EXCHANGE_SUFFIXES)
    if isinstance(available, str):
        available = [available]
    resolved: set[str] = set()
    for item in available:
        token = str(item).strip().upper()
        token = EXCHANGE_NAME_TO_SUFFIX.get(token, token)
        if token not in EXCHANGE_SUFFIXES:
            raise GoldBridgeError(
                f"unknown exchange {item!r} in register coverage; expected one of "
                f"{list(EXCHANGE_SUFFIXES)} or {list(EXCHANGE_NAME_TO_SUFFIX)}"
            )
        resolved.add(token)
    return frozenset(resolved)


def _interval_mask(
    panel: pd.DataFrame, intervals: pd.DataFrame, *, available: RegisterCoverage | None
) -> pd.Series:
    """Tri-state membership of each panel row in a set of dated intervals.

    Only rows on an exchange the register covers can be answered. A partial
    register (U0's ST history is SZSE-only) answers TRUE/FALSE for its exchange
    and UNKNOWN for the rest; it must neither blank out the exchange it does
    cover nor lend a confident FALSE to the ones it does not.
    """
    covered = normalise_coverage(available)
    if not covered:
        return pd.Series(MASK_UNKNOWN, index=panel.index, dtype="object")
    on_covered = exchange_suffix(panel["symbol"]).isin(covered).to_numpy()
    mask = pd.Series(
        np.where(on_covered, MASK_FALSE, MASK_UNKNOWN), index=panel.index, dtype="object"
    )
    if intervals.empty:
        return mask
    intervals = intervals[exchange_suffix(intervals["symbol"]).isin(covered).to_numpy()]
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


def sessions_since_listing(panel: pd.DataFrame, master: pd.DataFrame) -> pd.Series:
    """Exact per-row trading-session count since listing, or NaN when unknowable.

    Counted on the panel the caller passes -- pass the *full* U0 history, then cut
    the date window, so a start-date filter cannot restart the count. A symbol
    whose first bar is after its listing date (provider history truncated) has no
    exact count and gets NaN; `build_masks` then falls back to a lower bound.
    """
    identity = master.set_index("symbol")
    listing = pd.to_datetime(
        panel["symbol"].map(identity.get("listing_date", pd.Series(dtype=object))),
        errors="coerce",
    )
    dates = pd.to_datetime(panel["trade_date"])
    known, _ = _sessions_since_listing(panel.drop(columns=["sessions_since_listing"],
                                                  errors="ignore"), listing, dates)
    return pd.Series(known, index=panel.index, dtype="float64")


def _sessions_since_listing(
    frame: pd.DataFrame, listing: pd.Series, dates: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """(exact sessions since listing or NaN, a guaranteed lower bound).

    Exact only when the panel holds the listing session itself (first bar on or
    before the listing date). Otherwise the stock traded at least once before the
    panel's first bar, so ``row_index + 1`` is a lower bound; with no listing
    date at all, ``row_index`` is.
    """
    order = np.lexsort((dates.to_numpy(), frame["symbol"].astype(str).to_numpy()))
    symbols = frame["symbol"].astype(str).to_numpy()[order]
    ordered_dates = dates.to_numpy()[order]
    ordered_listing = listing.to_numpy()[order]
    on_or_after = ~pd.isna(ordered_listing) & (ordered_dates >= ordered_listing)
    starts = np.r_[True, symbols[1:] != symbols[:-1]]
    group = np.cumsum(starts) - 1
    first_index = np.flatnonzero(starts)
    row_index = np.arange(len(symbols)) - first_index[group]
    after_count = np.cumsum(on_or_after.astype(np.int64))
    after_base = np.r_[0, after_count][first_index][group]
    count_after = after_count - after_base - 1  # index among on/after-listing rows
    first_date = ordered_dates[first_index][group]
    covers_listing = ~pd.isna(ordered_listing) & (first_date <= ordered_listing)

    exact = np.where(covers_listing & on_or_after, count_after, np.nan).astype(float)
    if "sessions_since_listing" in frame.columns:
        supplied = pd.to_numeric(frame["sessions_since_listing"], errors="coerce").to_numpy()[order]
        exact = np.where(np.isfinite(supplied), supplied, exact)
    lower = np.where(
        np.isfinite(exact), exact,
        np.where(~pd.isna(ordered_listing), row_index + 1, row_index),
    ).astype(float)

    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return exact[inverse], lower[inverse]


def _half_up_cents(value: np.ndarray) -> np.ndarray:
    """Round a CNY price (or a cents product) half-up to whole cents.

    Exchange price limits are rounded 四舍五入 to 0.01; Python's ``round`` is
    banker's rounding on binary floats and is not the exchange rule.
    """
    return np.floor(np.asarray(value, dtype=float) * 100.0 + 0.5 + 1e-6)


#: Codes used while combining possible worlds (ST vs not ST).
_FALSE, _UNKNOWN, _TRUE = 0, 1, 2
_CODE_TO_MASK = np.array([MASK_FALSE, MASK_UNKNOWN, MASK_TRUE], dtype=object)


def price_limit_masks(
    frame: pd.DataFrame,
    *,
    master: pd.DataFrame | None = None,
    sessions_known: np.ndarray | None = None,
    sessions_lower_bound: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Tri-state "closed at the limit-up / limit-down price" for every row.

    Computed on **raw** traded prices, the scale the exchange enforces:

    * raw close = adjusted close / ``adjust_factor`` (1.0 when the panel is raw);
    * the reference is the ex-rights reference price -- the previous traded
      close re-based to today's factor (``adjusted prev close / today's factor``),
      rounded to 0.01 as the exchange publishes it;
    * limit = half-up 0.01 rounding of reference x (1 +/- band), with the band
      from the dated board rules in :mod:`quantagent.market_rules.ashare`
      (ChiNext 10% before 2020-08-24, main-board ST 5% before 2026-07-06, IPO
      windows anchored on the listing session).

    UNKNOWN, never a guess, when: there is no previous close; the board is
    unknown; the IPO window cannot be ruled in or out; a legacy listing-day band
    whose reference (the issue price) is not in the panel; ST is UNKNOWN and the
    close sits between the ST and ordinary limits; the close lies *beyond* every
    possible band (the reference is wrong -- e.g. a vendor gap or a no-limit
    resumption); or an ex-rights day puts the close within one tick of a limit
    computed from a factor-derived reference. A known IPO no-limit session is
    FALSE: there is no limit to be sealed at.
    """
    from quantagent.market_rules import ashare as rules

    n = len(frame)
    stats: dict[str, int] = {}
    if n == 0:
        empty = np.array([], dtype=object)
        return empty, empty, stats
    if "close" not in frame.columns:
        unknown = np.full(n, MASK_UNKNOWN, dtype=object)
        return unknown, unknown.copy(), {"rows": int(n), "no_close_column": int(n)}
    if "adjust_factor" in frame.columns:
        factor = pd.to_numeric(frame["adjust_factor"], errors="coerce").to_numpy(dtype=float)
    else:
        methods = (
            set(frame["adjustment_method"].dropna().astype(str).unique())
            if "adjustment_method" in frame.columns else set()
        )
        if methods - {contracts.ADJUST_NONE}:
            raise GoldBridgeError(
                "price-limit masks need raw prices: the panel declares adjustment "
                f"{sorted(methods)} but carries no adjust_factor to undo it"
            )
        factor = np.ones(n)

    dates = pd.to_datetime(frame["trade_date"]).to_numpy()
    symbols = frame["symbol"].astype(str).to_numpy()
    order = np.lexsort((dates, symbols))
    sym_o = symbols[order]
    starts = np.r_[True, sym_o[1:] != sym_o[:-1]]
    adj_close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)[order]
    fac = factor[order]
    prev_adj_close = np.r_[np.nan, adj_close[:-1]]
    prev_fac = np.r_[np.nan, fac[:-1]]
    prev_adj_close[starts] = np.nan
    prev_fac[starts] = np.nan

    raw_close = adj_close / fac
    reference = prev_adj_close / fac
    close_c = _half_up_cents(raw_close)
    ref_c = _half_up_cents(reference)
    ex_rights = np.isfinite(prev_fac) & (np.abs(prev_fac / fac - 1.0) > 1e-9)

    if master is not None and "board" in master.columns:
        board_map = master.drop_duplicates("symbol").set_index("symbol")["board"].astype(str)
        board = pd.Series(sym_o).map(board_map).to_numpy(dtype=object)
    else:
        board = np.full(len(sym_o), None, dtype=object)
    missing_board = pd.isna(board) | ~np.isin(board.astype(str), list(rules.BOARDS))
    if missing_board.any():
        board[missing_board] = [rules.exchange_board_for_symbol(s) for s in sym_o[missing_board]]

    if sessions_known is None or sessions_lower_bound is None:
        identity = (master.set_index("symbol") if master is not None
                    else pd.DataFrame(columns=["listing_date"]))
        listing = pd.to_datetime(
            frame["symbol"].map(identity.get("listing_date", pd.Series(dtype=object))),
            errors="coerce",
        )
        sessions_known, sessions_lower_bound = _sessions_since_listing(
            frame, listing, pd.to_datetime(frame["trade_date"])
        )
    s_known = np.asarray(sessions_known, dtype=float)[order]
    s_lower = np.asarray(sessions_lower_bound, dtype=float)[order]
    window = np.array([rules.IPO_UNLIMITED_DAYS.get(b, 0) for b in board], dtype=float)
    # Sessions used to resolve the band. Rows provably past the IPO window use the
    # ordinary band; rows that may still be inside it are UNKNOWN unless exact.
    in_window_unknown = ~np.isfinite(s_known) & (s_lower < window)
    session_key = np.where(np.isfinite(s_known), np.minimum(s_known, 99), 99).astype(int)

    st_values = (frame["mask_is_st"].astype(str).to_numpy()[order]
                 if "mask_is_st" in frame.columns
                 else np.full(len(sym_o), MASK_UNKNOWN, dtype=object))

    keys = pd.DataFrame({
        "board": board.astype(str),
        "date": pd.to_datetime(dates[order]).normalize(),
        "session": np.minimum(session_key, 6),
    })
    unique = keys.drop_duplicates().reset_index(drop=True)
    resolved = {}
    for row in unique.itertuples(index=False):
        for is_st in (False, True):
            try:
                limits = rules.price_limits(
                    board=row.board, previous_close=1.0, trade_date=row.date.date(),
                    sessions_since_listing=int(row.session), is_st=is_st,
                )
                resolved[(row.board, row.date, row.session, is_st)] = (
                    limits.ratio, limits.regime)
            except ValueError:
                resolved[(row.board, row.date, row.session, is_st)] = (None, "UNKNOWN_BOARD")
    lookup = keys.merge(unique.assign(_k=np.arange(len(unique))), how="left",
                        on=["board", "date", "session"])["_k"].to_numpy()
    ratio_o = np.empty(len(unique)); regime_o = np.empty(len(unique), dtype=object)
    ratio_s = np.empty(len(unique)); regime_s = np.empty(len(unique), dtype=object)
    for i, row in enumerate(unique.itertuples(index=False)):
        r, g = resolved[(row.board, row.date, row.session, False)]
        ratio_o[i] = np.nan if r is None else r; regime_o[i] = g
        r, g = resolved[(row.board, row.date, row.session, True)]
        ratio_s[i] = np.nan if r is None else r; regime_s[i] = g
    ratio_o, regime_o = ratio_o[lookup], regime_o[lookup]
    ratio_s, regime_s = ratio_s[lookup], regime_s[lookup]

    tol = np.where(ex_rights, 1.0, 0.0)

    def world(ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        up_c = _half_up_cents(ref_c * (1.0 + ratio) / 100.0)
        dn_c = _half_up_cents(ref_c * (1.0 - ratio) / 100.0)
        up_diff = close_c - up_c
        dn_diff = dn_c - close_c
        valid = (up_diff <= tol) & (dn_diff <= tol) & np.isfinite(up_c)
        def code(diff: np.ndarray) -> np.ndarray:
            # diff > tol is outside this world (invalid); diff == 0 is at the
            # limit; on ex-rights days a one-tick miss is undecidable.
            out = np.full(len(diff), _FALSE, dtype=np.int8)
            out[(tol > 0) & (np.abs(diff) <= tol)] = _UNKNOWN
            out[diff == 0] = _TRUE
            return out
        return valid, code(up_diff), code(dn_diff)

    valid_o, up_o, dn_o = world(ratio_o)
    valid_s, up_s, dn_s = world(ratio_s)
    may_be_st = np.isin(st_values, [MASK_TRUE, MASK_UNKNOWN])
    may_be_ordinary = np.isin(st_values, [MASK_FALSE, MASK_UNKNOWN])
    use_o = may_be_ordinary & valid_o
    use_s = may_be_st & valid_s

    def combine(code_o: np.ndarray, code_s: np.ndarray) -> np.ndarray:
        out = np.full(len(code_o), _UNKNOWN, dtype=np.int8)
        only_o = use_o & ~use_s
        only_s = use_s & ~use_o
        both = use_o & use_s
        out[only_o] = code_o[only_o]
        out[only_s] = code_s[only_s]
        agree = both & (code_o == code_s)
        out[agree] = code_o[agree]
        return out

    up = combine(up_o, up_s)
    down = combine(dn_o, dn_s)

    no_reference = ~np.isfinite(ref_c) | ~np.isfinite(close_c)
    # SSE B shares quote in USD with a 0.001 tick; the cent arithmetic above is
    # not their exchange rounding, so their limit state is not claimed.
    sub_cent_tick = np.char.startswith(sym_o.astype(str), "900")
    no_limit = (regime_o == "IPO_NO_LIMIT_WINDOW") & np.isfinite(s_known)
    legacy_listing_day = regime_o == "IPO_LEGACY_APPROVAL_SYSTEM"
    unknown_board = (regime_o == "UNKNOWN_BOARD") | ~np.isfinite(ratio_o)
    contradiction = ~(may_be_ordinary & valid_o) & ~(may_be_st & valid_s) & ~no_reference
    for target in (up, down):
        target[no_reference | unknown_board | in_window_unknown | legacy_listing_day] = _UNKNOWN
        target[no_limit] = _FALSE
        target[sub_cent_tick] = _UNKNOWN

    stats = {
        "rows": int(n),
        "no_previous_close": int((no_reference & ~no_limit).sum()),
        "ipo_no_limit_window": int(no_limit.sum()),
        "ipo_window_undetermined": int((in_window_unknown & ~no_limit).sum()),
        "legacy_listing_day": int(legacy_listing_day.sum()),
        "close_beyond_every_band": int((contradiction & ~unknown_board & ~no_limit
                                        & ~in_window_unknown & ~legacy_listing_day).sum()),
        "ex_rights_rows": int(ex_rights.sum()),
        "sub_cent_tick_rows": int(sub_cent_tick.sum()),
        "limit_up_true": int((up == _TRUE).sum()),
        "limit_up_unknown": int((up == _UNKNOWN).sum()),
        "limit_down_true": int((down == _TRUE).sum()),
        "limit_down_unknown": int((down == _UNKNOWN).sum()),
    }
    inverse = np.empty_like(order)
    inverse[order] = np.arange(n)
    return _CODE_TO_MASK[up[inverse]], _CODE_TO_MASK[down[inverse]], stats


def no_trade_mask(frame: pd.DataFrame) -> np.ndarray:
    """Tri-state "the vendor bar records no trade" (zero volume, or zero amount).

    TickFlow prints a flat bar at the last close on sessions a stock did not
    trade. Those are not traded sessions: they cannot be entered or exited, and
    their "close" is a stale print, not a price anyone paid (R10-F01).
    UNKNOWN where volume is not measured.
    """
    if "volume" not in frame.columns:
        return np.full(len(frame), MASK_UNKNOWN, dtype=object)
    volume = pd.to_numeric(frame["volume"], errors="coerce").to_numpy(dtype=float)
    amount = (pd.to_numeric(frame["amount"], errors="coerce").to_numpy(dtype=float)
              if "amount" in frame.columns else np.full(len(frame), np.nan))
    no_trade = (volume <= 0) | (np.isfinite(amount) & (amount <= 0))
    out = np.where(no_trade, MASK_TRUE, np.where(np.isfinite(volume), MASK_FALSE, MASK_UNKNOWN))
    return out.astype(object)


def carry_no_trade_prices(frame: pd.DataFrame, no_trade: np.ndarray) -> pd.DataFrame:
    """Value no-trade bars at the last traded close, on the frame's price scale.

    On an adjusted (hfq) panel this keeps the series flat through a halt and
    re-bases across an ex-rights date inside it: the vendor's stale pre-ex
    print times the new factor is a jump that never traded (000836.SZ 10转20
    read as x3.0). A no-trade bar before any traded bar keeps its print.
    """
    if "close" not in frame.columns:
        return frame
    hit = np.asarray(no_trade) == MASK_TRUE
    if not hit.any():
        return frame
    result = frame
    order = np.lexsort((pd.to_datetime(result["trade_date"]).to_numpy(),
                        result["symbol"].astype(str).to_numpy()))
    close = pd.to_numeric(result["close"], errors="coerce").to_numpy(dtype=float)[order]
    hit_o = hit[order]
    symbols = result["symbol"].astype(str).to_numpy()[order]
    group = np.cumsum(np.r_[True, symbols[1:] != symbols[:-1]])
    traded_close = np.where(hit_o, np.nan, close)
    carried = pd.Series(traded_close).groupby(group).ffill().to_numpy()
    use = hit_o & np.isfinite(carried)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    carried_by_row = carried[inverse]
    use_by_row = use[inverse]
    for column in PRICE_COLUMNS:
        if column in result.columns:
            values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float).copy()
            values[use_by_row] = carried_by_row[use_by_row]
            result[column] = values
    return result


def build_masks(
    panel: pd.DataFrame,
    *,
    master: pd.DataFrame,
    suspension: pd.DataFrame | None = None,
    st: pd.DataFrame | None = None,
    st_available: RegisterCoverage = False,
    seasoning_days: int = DEFAULT_SEASONING_DAYS,
) -> pd.DataFrame:
    """Attach the tri-state eligibility masks to the panel.

    ``st_available`` is a deliberate parameter rather than an inference from
    whether the frame is empty: U0's ST register covers SZSE only, so a partial
    source must produce UNKNOWN for the exchanges it does not cover instead of
    a confident FALSE. Pass the covered exchanges (``{"SZ"}``) to measure those
    rows; ``True`` still means "complete for every exchange" and ``False`` means
    "nothing measured". Before Round 29 the only choices were all-or-nothing,
    so the SZSE register measured nothing and known ST names stayed trainable.
    """
    result = panel.copy()
    dates = pd.to_datetime(result["trade_date"])

    # Zero-volume vendor bars are no-trade sessions; their stale print is
    # replaced by the last traded close before anything reads a price.
    result["mask_no_trade"] = no_trade_mask(result)
    result = carry_no_trade_prices(result, result["mask_no_trade"].to_numpy())

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
    # days, so holidays cannot shorten the window. A caller that knows each row's
    # session count since listing (computed on the full history, see
    # `sessions_since_listing`) passes it as a column; otherwise the count restarts
    # at the panel's first row, which marks every old name's first sessions after a
    # start-date cut as "unseasoned".
    sessions_known, sessions_lower_bound = _sessions_since_listing(result, listing, dates)
    if "sessions_since_listing" in result.columns:
        seasoning_count = sessions_lower_bound
    else:
        seasoning_count = (
            result.assign(_d=dates).sort_values("_d").groupby("symbol").cumcount()
            .reindex(result.index).to_numpy(dtype=float)
        )
    result["mask_seasoning"] = np.where(
        seasoning_count < seasoning_days, MASK_TRUE, MASK_FALSE
    ).astype(object)

    # Price-limit state at the close, from raw prices and dated board rules. It is
    # an *entry-feasibility* fact consumed by `build_labels` at t+1, not an
    # eligibility mask at t, so it stays out of `eligible_for_training`.
    limit_up, limit_down, limit_stats = price_limit_masks(
        result, master=master, sessions_known=sessions_known,
        sessions_lower_bound=sessions_lower_bound,
    )
    result["mask_limit_up"] = limit_up
    result["mask_limit_down"] = limit_down
    result.attrs["price_limit_stats"] = limit_stats

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
    result.attrs["st_coverage"] = sorted(normalise_coverage(st_available))
    return result


def mask_rows_by_exchange(
    frame: pd.DataFrame, columns: Sequence[str]
) -> dict[str, dict[str, dict[str, int]]]:
    """``{mask: {exchange: {TRUE|FALSE|UNKNOWN: rows}}}`` -- measured, not presence.

    A mask column that exists but is UNKNOWN on every row of an exchange has
    measured nothing there. Counting rows per value per exchange is what lets a
    certificate tell "the SZSE register was applied" from "the column exists".
    """
    if frame.empty:
        return {}
    exchange = exchange_suffix(frame["symbol"])
    out: dict[str, dict[str, dict[str, int]]] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        table = pd.crosstab(exchange, frame[column].astype(str))
        out[column] = {
            str(ex): {
                value: int(table.loc[ex].get(value, 0))
                for value in (MASK_TRUE, MASK_FALSE, MASK_UNKNOWN)
            }
            for ex in table.index
        }
    return out


def st_coverage_from_pit_evidence(
    pit_certificate: Mapping[str, Any] | None,
    st_manifest: Mapping[str, Any] | None = None,
    st_intervals: pd.DataFrame | None = None,
) -> tuple[frozenset[str], dict[str, Any]]:
    """Which exchanges the U0 ST register can answer for, and why.

    A field the PIT certificate marks ``AVAILABLE`` covers every exchange. A
    partial register covers the exchanges its manifest names as having dated
    history -- intersected with the exchanges actually present in the interval
    table, so a manifest claim with no rows behind it measures nothing.
    """
    field_text = str(
        ((pit_certificate or {}).get("pit_field_availability") or {}).get("st_intervals", "")
    )
    basis: dict[str, Any] = {"pit_certificate_st_field": field_text}
    if field_text.startswith("AVAILABLE"):
        basis["basis"] = "pit_certificate_available"
        return frozenset(EXCHANGE_SUFFIXES), basis

    declared: set[str] = set()
    for name in (st_manifest or {}).get("exchanges_with_dated_history", []) or []:
        suffix = EXCHANGE_NAME_TO_SUFFIX.get(str(name).upper())
        if suffix:
            declared.add(suffix)
    if not declared and "PARTIAL" in field_text.upper():
        # "... dated episodes over N securities from SZSE; no dated register for BSE, SSE"
        head = field_text.split(";", 1)[0]
        for name, suffix in EXCHANGE_NAME_TO_SUFFIX.items():
            if f"from {name}" in head:
                declared.add(suffix)
    observed: set[str] = set()
    if st_intervals is not None and not st_intervals.empty:
        observed = set(exchange_suffix(st_intervals["symbol"]).unique()) & set(EXCHANGE_SUFFIXES)
    covered = frozenset(declared & observed)
    basis.update({
        "basis": "st_manifest_exchanges_with_dated_history ∩ exchanges_in_register",
        "declared": sorted(declared),
        "observed_in_register": sorted(observed),
        "covered": sorted(covered),
        "uncovered": sorted(set(EXCHANGE_SUFFIXES) - covered),
    })
    return covered, basis


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
    panel: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    allow_unmeasured_limit_up: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach delay-1 executable forward returns and drop infeasible entries.

    Entry is infeasible when the security is suspended at ``t`` or ``t+1``, is
    ST at ``t``, or is sealed at limit-up at ``t+1`` -- you cannot buy a locked
    limit-up, and pretending otherwise is where the old phantom alpha came from.

    ``mask_limit_up`` is required. Its absence used to skip the check silently
    and nothing produced the column, so the certified 10.9M-row panel kept
    every sealed limit-up entry (R3-F01: 144,840 rows, mean 5-day forward
    return 27x the rest). A caller that genuinely cannot measure it must say so
    with ``allow_unmeasured_limit_up=True``; the unmeasured rows are counted.
    """
    if "mask_limit_up" not in panel.columns and not allow_unmeasured_limit_up:
        raise GoldBridgeError(
            "build_labels needs mask_limit_up (from build_masks) to drop sealed "
            "limit-up entries at t+1; pass allow_unmeasured_limit_up=True only if "
            "the limit state genuinely cannot be measured"
        )
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
    if "mask_no_trade" in result.columns:
        # A zero-volume bar is not a session anyone traded: no signal on it and
        # no entry into it (R10-F01: 20,851 kept rows had a zero-volume t+1).
        no_trade_now = result["mask_no_trade"] == MASK_TRUE
        no_trade_next = grouped["mask_no_trade"].shift(-1) == MASK_TRUE
        reasons["no_trade_at_t"] = int(no_trade_now.sum())
        reasons["entry_zero_volume"] = int(no_trade_next.sum())
        infeasible |= no_trade_now | no_trade_next
    if "mask_is_st" in result.columns:
        is_st = result["mask_is_st"] == MASK_TRUE
        reasons["st_at_t"] = int(is_st.sum())
        infeasible |= is_st
    if "mask_limit_up" in result.columns:
        next_limit = grouped["mask_limit_up"].shift(-1)
        sealed = next_limit == MASK_TRUE
        reasons["limit_up_at_t1"] = int(sealed.sum())
        infeasible |= sealed
    else:
        reasons["limit_up_unmeasured_rows"] = int(len(result))

    reasons["entry_price_missing"] = int(entry_price.isna().sum())
    infeasible |= entry_price.isna()
    if "mask_limit_up" in result.columns:
        # Not dropped (not *known* infeasible) but counted, so the share of the
        # kept domain whose entry feasibility is undecided stays visible.
        reasons["limit_up_unknown_at_t1_kept"] = int(
            ((next_limit == MASK_UNKNOWN) & ~infeasible).sum()
        )

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
    st_available: RegisterCoverage = False,
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

    st_coverage = normalise_coverage(st_available)
    # Only exchanges the panel actually trades on can leave a hole in the mask.
    present = set(exchange_suffix(masked["symbol"]).unique())
    uncovered = sorted((set(EXCHANGE_SUFFIXES) & present) - st_coverage)
    warnings: list[str] = []
    if not st_coverage:
        warnings.append(
            "ST intervals were not available as a complete dated register, so "
            "mask_is_st is UNKNOWN and no row can be excluded on ST grounds; "
            "any dataset built this way is NOT point-in-time complete"
        )
    elif uncovered:
        warnings.append(
            f"ST register covers only {sorted(st_coverage)}; mask_is_st is UNKNOWN for "
            f"{uncovered}, so ST rows there cannot be excluded; any dataset built "
            "this way is NOT point-in-time complete"
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
        mask_rows_by_exchange=mask_rows_by_exchange(
            masked, [c for c in masked.columns if c.startswith("mask_")]
        ),
        st_coverage=sorted(st_coverage),
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

    st_rows = (manifest.mask_rows_by_exchange or {}).get("mask_is_st", {})
    if st_rows:
        evidence["st_measured_rows_by_exchange"] = {
            ex: counts.get(MASK_TRUE, 0) + counts.get(MASK_FALSE, 0)
            for ex, counts in st_rows.items()
        }
        evidence["st_unknown_rows_by_exchange"] = {
            ex: counts.get(MASK_UNKNOWN, 0) for ex, counts in st_rows.items()
        }
        unmeasured = sorted(
            ex for ex, n in evidence["st_unknown_rows_by_exchange"].items() if n
        )
        if unmeasured:
            blockers.append(
                "mask_is_st is unmeasured on exchange(s) "
                + ", ".join(f"{ex} ({evidence['st_unknown_rows_by_exchange'][ex]:,} rows)"
                            for ex in unmeasured)
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
