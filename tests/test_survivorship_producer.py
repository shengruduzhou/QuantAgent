"""A fix that cannot read the real producer is a fix that never fires.

DEF-024 taught `build_masks` to consult a `status` column so a missing delisting
date would stop reading as "confidently never delisted". The U0 master has no
`status` column, so on the real master that fix could only ever answer UNKNOWN —
for all 5,888 names, including the 5,530 sourced from live listing registers. It
could not distinguish the 358 names it existed to catch from the ones it did not
need to, which is the DEF-025 lesson in a new place: hardening one side of an
interface is worth nothing until you check what the other side actually speaks.

The master does carry the distinction, in two agreeing columns: `status_end_blocked`
(True for exactly those 358) and `source` (`sz_delist` / `sh_delist_retry`).

Note which master: U0 has two. `historical_security_master.parquet` (H-032C) has
neither `status` nor a single delisting date, and is the one described above.
`security_master.parquet` — the one `build_u0_full_universe_gold.py` actually reads
— has `status` (5,533 listed / 361 delisted) *and* all 361 delisting dates. Judging
the shipped gold against the first of those was a mistake corrected in DEF-028
below; the resolver still matters, because a rebuild pointed at the H-032C master
would be blind.
"""

from __future__ import annotations

import pandas as pd

from quantagent.data.ashare.gold_bridge import (
    LISTING_STATUS_DELISTED,
    LISTING_STATUS_LISTED,
    LISTING_STATUS_UNKNOWN,
    MASK_FALSE,
    MASK_UNKNOWN,
    build_masks,
    resolve_listing_status,
)
from quantagent.data.v7_quality_gates import GATE_UNKNOWN, evaluate_survivorship

MASK_COLUMNS = [
    "mask_is_suspended", "mask_is_st", "mask_pre_listing",
    "mask_post_delisting", "mask_seasoning",
]


def _u0_shaped_master() -> pd.DataFrame:
    """The columns U0's `historical_security_master.parquet` actually has.

    Note what is *not* here: `status`, and any delisting date at all. Both
    `delisting_date` and `status_end` exist as columns and are empty for every row.
    """
    return pd.DataFrame(
        [
            {"symbol": "600001.SH", "listing_date": "2010-01-04", "delisting_date": None,
             "status_end": None, "status_end_blocked": False, "source": "sh_main"},
            {"symbol": "600002.SH", "listing_date": "2010-01-04", "delisting_date": None,
             "status_end": None, "status_end_blocked": False, "source": "sh_main"},
            {"symbol": "000003.SZ", "listing_date": "2010-01-04", "delisting_date": None,
             "status_end": None, "status_end_blocked": True, "source": "sz_delist"},
        ]
    )


def _panel(symbols=("600001.SH", "600002.SH", "000003.SZ"), days: int = 6) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=days)
    return pd.DataFrame(
        [{"symbol": symbol, "trade_date": date, "close": 10.0}
         for symbol in symbols for date in dates]
    )


def test_status_column_is_used_when_the_master_has_one():
    master = pd.DataFrame(
        [
            {"symbol": "a", "status": "listed"},
            {"symbol": "b", "status": "Delisted"},
            {"symbol": "c", "status": "who knows"},
        ]
    )
    resolved, basis = resolve_listing_status(master)
    assert basis == "status"
    assert resolved["a"] == LISTING_STATUS_LISTED
    assert resolved["b"] == LISTING_STATUS_DELISTED
    assert resolved["c"] == LISTING_STATUS_UNKNOWN


def test_provenance_answers_when_the_master_has_no_status_column():
    """The U0 shape. Without this the answer is UNKNOWN for every name."""
    resolved, basis = resolve_listing_status(_u0_shaped_master())
    assert basis == "source+status_end_blocked"
    assert resolved["600001.SH"] == LISTING_STATUS_LISTED
    assert resolved["600002.SH"] == LISTING_STATUS_LISTED
    assert resolved["000003.SZ"] == LISTING_STATUS_DELISTED


def test_a_master_carrying_neither_signal_stays_unknown():
    master = pd.DataFrame([{"symbol": "a", "listing_date": "2010-01-04"}])
    resolved, basis = resolve_listing_status(master)
    assert basis == "none"
    assert resolved["a"] == LISTING_STATUS_UNKNOWN


def test_the_mask_separates_live_names_from_undatable_dead_ones():
    masked = build_masks(_panel(), master=_u0_shaped_master())
    by_symbol = masked.groupby("symbol")["mask_post_delisting"].agg(set)

    # Sourced from a live register with no end date: confidently not delisted.
    assert by_symbol["600001.SH"] == {MASK_FALSE}
    assert by_symbol["600002.SH"] == {MASK_FALSE}
    # Known dead, but the register never captured *when*. Its whole history is
    # unusable for a survivorship-free claim — and it must not read as FALSE.
    assert by_symbol["000003.SZ"] == {MASK_UNKNOWN}

    assert masked.attrs["listing_status_basis"] == "source+status_end_blocked"


def test_a_name_absent_from_the_master_entirely_is_unknown_not_listed():
    """Two symbols in the shipped gold panel are in no master row at all."""
    masked = build_masks(_panel(symbols=("999999.SZ",)), master=_u0_shaped_master())
    assert set(masked["mask_post_delisting"]) == {MASK_UNKNOWN}


def test_unknown_masks_matches_the_row_wise_construction_it_replaced():
    """The vectorised build is 18x faster; it must also be the same answer.

    Row-wise `iterrows()` ran at ~30k rows/s — six minutes on the full-universe
    panel, which the M5-02 producer wiring turns into a per-run cost.
    """
    masked = build_masks(_panel(days=8), master=_u0_shaped_master())
    reference = [
        ",".join(
            column.removeprefix("mask_")
            for column in MASK_COLUMNS
            if row[column] == MASK_UNKNOWN
        )
        for _, row in masked[MASK_COLUMNS].iterrows()
    ]
    assert list(masked["unknown_masks"]) == reference


def test_survivorship_names_how_many_symbols_it_cannot_date():
    """`delisted_symbols` counts only *dated* delistings, so a register with no
    dates reports 0 there and says nothing about the panel's exposure."""
    masked = build_masks(_panel(), master=_u0_shaped_master())
    report = evaluate_survivorship(masked)

    assert report.status == GATE_UNKNOWN
    # Present in the panel and known dead — counted from `listing_status`, not from
    # post-delisting rows, of which a correctly-stopped panel has none (DEF-028).
    assert report.delisted_symbols == 1
    assert report.undated_delisted_symbols == 1
    assert report.unknown_sessions == 6
    assert report.as_metrics()["survivorship_undated_delisted_symbols"] == 1
    # The blocker must be stated as bounded, not as "the universe is unusable".
    assert "1 个标的" in report.detail or "涉及 1" in report.detail


def test_the_training_path_now_computes_survivorship_instead_of_omitting_it():
    """The producer side the ledger recorded as missing.

    The gate has refused runs without survivorship evidence since DEF-024, but
    nothing computed it, so every run reached the gate with the key simply absent.
    """
    from quantagent.data.v7_quality_gates import evaluate_data_quality_gates
    from quantagent.training.v7_experiment import _dataset_audit_metrics

    masked = build_masks(_panel(days=10), master=_u0_shaped_master())
    masked["forward_return_1d"] = 0.01
    masked["available_at"] = masked["trade_date"]

    metrics = _dataset_audit_metrics(masked, evaluate_data_quality_gates(masked))
    assert metrics["survivorship_status"] == GATE_UNKNOWN
    assert metrics["survivorship_undated_delisted_symbols"] == 1
    assert metrics["survivorship_report"]["unknown_sessions"] == 10


def test_a_dataset_with_no_masks_reports_unknown_rather_than_nothing():
    from quantagent.data.v7_quality_gates import evaluate_data_quality_gates
    from quantagent.training.v7_experiment import _dataset_audit_metrics

    plain = _panel(days=10).copy()
    plain["forward_return_1d"] = 0.01
    plain["available_at"] = plain["trade_date"]

    metrics = _dataset_audit_metrics(plain, evaluate_data_quality_gates(plain))
    assert metrics["survivorship_status"] == GATE_UNKNOWN
    assert "build_masks" in metrics["survivorship_report"]["detail"]


# ---------------------------------------------------------------------------
# DEF-028: the audit was asking a question the mask cannot answer
# ---------------------------------------------------------------------------


def _master_with_dates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "600001.SH", "listing_date": "2010-01-04",
             "delisting_date": None, "status": "listed"},
            {"symbol": "600002.SH", "listing_date": "2010-01-04",
             "delisting_date": "2026-01-20", "status": "delisted"},
        ]
    )


def _panel_stopping_on_time() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=8)
    rows = [{"symbol": "600001.SH", "trade_date": d, "close": 10.0} for d in dates]
    rows += [
        {"symbol": "600002.SH", "trade_date": d, "close": 10.0}
        for d in dates if d < pd.Timestamp("2026-01-20")
    ]
    return pd.DataFrame(rows)


def test_the_mask_alone_cannot_distinguish_a_live_name_from_a_stopped_dead_one():
    """The ambiguity that made the old verdict wrong."""
    masked = build_masks(_panel_stopping_on_time(), master=_master_with_dates(),
                         st_available=True, suspension=pd.DataFrame(), st=pd.DataFrame())
    by_symbol = masked.groupby("symbol")["mask_post_delisting"].agg(set)
    # Identical masks, opposite facts: one name is alive, the other is dead.
    assert by_symbol["600001.SH"] == {MASK_FALSE}
    assert by_symbol["600002.SH"] == {MASK_FALSE}
    # `listing_status` is what separates them.
    assert set(masked.loc[masked["symbol"] == "600002.SH", "listing_status"]) == {
        LISTING_STATUS_DELISTED
    }


def test_a_correctly_stopped_panel_is_a_pass_not_an_unknown():
    masked = build_masks(_panel_stopping_on_time(), master=_master_with_dates(),
                         st_available=True, suspension=pd.DataFrame(), st=pd.DataFrame())
    report = evaluate_survivorship(masked)
    assert report.status == "pass"
    assert report.delisted_symbols == 1
    assert "既有活下来的，也有死掉的" in report.detail


def test_a_panel_without_the_column_can_still_be_judged_against_a_master():
    """The shipped gold predates `listing_status`; it is judged via the master
    it was built from."""
    masked = build_masks(_panel_stopping_on_time(), master=_master_with_dates(),
                         st_available=True, suspension=pd.DataFrame(), st=pd.DataFrame())
    legacy = masked.drop(columns=["listing_status"])

    assert evaluate_survivorship(legacy).status == GATE_UNKNOWN
    assert "listing_status" in evaluate_survivorship(legacy).detail

    report = evaluate_survivorship(legacy, master=_master_with_dates())
    assert report.status == "pass"
    assert report.delisted_symbols == 1


def test_bars_printed_after_a_delisting_are_a_failure_not_a_coverage_gap():
    dates = pd.bdate_range("2026-01-05", periods=20)
    panel = pd.DataFrame(
        [{"symbol": s, "trade_date": d, "close": 10.0}
         for s in ("600001.SH", "600002.SH") for d in dates]
    )
    masked = build_masks(panel, master=_master_with_dates(), st_available=True,
                         suspension=pd.DataFrame(), st=pd.DataFrame())
    report = evaluate_survivorship(masked)
    assert report.status == "fail"
    assert "退市日之后" in report.detail
