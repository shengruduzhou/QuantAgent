"""A certificate a survivorship-biased panel can also earn is not certifying much.

`FULL_UNIVERSE_GOLD_READY` is granted on nine structural checks: no duplicate
security-dates, no pre-listing rows, no post-delisting rows, one adjustment mode,
non-negative volume, positive close, labels present, masks present, no infeasible
entries.

Every one of them is satisfied by a panel containing only the names that survived
(DEF-029), and `no_post_delisting_rows` — the only check that mentions delisting —
is *helped* by the bias: delete the names that died and it passes trivially.
Measured by dropping all 261 delisted names from the shipped 10,917,401-row panel:
all nine returned PASS and `structurally_valid` stayed True.

Nothing asked whether the losers were in the universe, which is the single thing
the certificate's name asserts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = PROJECT_ROOT / "scripts/build_u0_full_universe_gold.py"


@pytest.fixture(scope="module")
def build_module():
    spec = importlib.util.spec_from_file_location("build_u0_gold", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _master(*, dated: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "600001.SH", "board": "SH_Main", "status": "listed",
             "listing_date": "2010-01-04", "delisting_date": None},
            {"symbol": "600002.SH", "board": "SH_Main", "status": "listed",
             "listing_date": "2010-01-04", "delisting_date": None},
            {"symbol": "000003.SZ", "board": "SZ_Main", "status": "delisted",
             "listing_date": "2010-01-04",
             "delisting_date": "2026-01-20" if dated else None},
        ]
    )


def _dataset(master: pd.DataFrame, *, include_dead: bool = True) -> pd.DataFrame:
    from quantagent.data.ashare.gold_bridge import build_masks

    dates = pd.bdate_range("2026-01-05", periods=10)
    symbols = list(master["symbol"]) if include_dead else ["600001.SH", "600002.SH"]
    rows = [
        {"symbol": symbol, "trade_date": date}
        for symbol in symbols
        for date in dates
        # A correct panel stops a dead name when it stops trading.
        if not (symbol == "000003.SZ" and date >= pd.Timestamp("2026-01-20"))
    ]
    frame = build_masks(pd.DataFrame(rows), master=master, st_available=True,
                        suspension=pd.DataFrame(), st=pd.DataFrame())
    frame["adjustment_method"] = "hfq"
    frame["volume"] = 1_000_000.0
    frame["close"] = 10.0
    frame["entry_feasible"] = True
    frame["forward_return_1d"] = 0.01
    return frame


def _verdict(quality: dict, name: str) -> str:
    return next(check["verdict"] for check in quality["checks"] if check["check"] == name)


def test_the_nine_structural_checks_cannot_see_survivorship_bias(build_module):
    """Both panels clear every check that existed before this round."""
    master = _master()
    full = build_module.run_quality_checks(_dataset(master), master)
    biased = build_module.run_quality_checks(
        _dataset(master, include_dead=False), master
    )

    legacy = [
        "no_duplicate_security_dates", "no_pre_listing_rows", "no_post_delisting_rows",
        "adjustment_mode_declared", "volume_non_negative", "close_positive",
        "labels_present", "masks_present", "no_infeasible_entries",
    ]
    for check in legacy:
        assert _verdict(full, check) == "PASS"
        assert _verdict(biased, check) == "PASS", (
            f"{check} is satisfied by a panel with no delisted names"
        )


def test_the_certificate_is_withheld_from_a_survivors_only_panel(build_module):
    master = _master()
    biased = build_module.run_quality_checks(_dataset(master, include_dead=False), master)

    assert _verdict(biased, "universe_includes_delisted_names") == "UNKNOWN"
    assert "universe_includes_delisted_names" in biased["unknown_checks"]
    assert biased["structurally_valid"] is False
    # Withheld, not failed: nothing here is broken, it just was not demonstrated.
    assert biased["failed_checks"] == []


def test_a_panel_containing_the_dead_names_is_granted(build_module):
    master = _master()
    full = build_module.run_quality_checks(_dataset(master), master)

    assert _verdict(full, "universe_includes_delisted_names") == "PASS"
    assert full["structurally_valid"] is True
    evidence = next(
        check["evidence"] for check in full["checks"]
        if check["check"] == "universe_includes_delisted_names"
    )
    assert evidence["delisted_names_present"] == 1


def test_the_certificate_records_which_master_answered_and_how(build_module):
    """The Round 18 wrong verdict came from auditing an artifact against a master
    it was not built from. `lineage.json` recorded the path; the certificate — the
    file a reader opens — said nothing about what the master could support."""
    master = _master()
    quality = build_module.run_quality_checks(_dataset(master), master)

    identity = quality["master_identity"]
    assert identity["securities"] == 3
    assert identity["listing_status_basis"] == "status"
    assert identity["resolved"] == {"listed": 2, "delisted": 1}
    assert identity["delisting_dates_available"] == 1


def test_a_mask_built_from_a_different_master_is_detected_not_believed(build_module):
    """The specific hazard that produced the wrong verdict.

    A persisted mask carries no record of which master produced it. But two masters
    cannot both be right about the same security: if this master calls a name
    delisted and has no date for it, a confident `FALSE` in the mask could not have
    come from here.
    """
    dated_master = _master(dated=True)
    dataset = _dataset(dated_master)  # mask built from the master WITH dates

    undated_master = _master(dated=False)  # audited against the one without
    quality = build_module.run_quality_checks(dataset, undated_master)

    assert _verdict(quality, "universe_includes_delisted_names") == "UNKNOWN"
    assert quality["structurally_valid"] is False
    detail = next(
        check["evidence"]["detail"] for check in quality["checks"]
        if check["check"] == "universe_includes_delisted_names"
    )
    assert "掩码与传入的 master 不一致" in detail
    assert "lineage.json" in detail


def test_bars_printed_after_a_delisting_still_fail(build_module):
    """The pre-existing check keeps its meaning; the new one does not replace it."""
    master = _master()
    dates = pd.bdate_range("2026-01-05", periods=20)
    rows = [{"symbol": s, "trade_date": d} for s in master["symbol"] for d in dates]

    from quantagent.data.ashare.gold_bridge import build_masks

    frame = build_masks(pd.DataFrame(rows), master=master, st_available=True,
                        suspension=pd.DataFrame(), st=pd.DataFrame())
    frame["adjustment_method"] = "hfq"
    frame["volume"] = 1_000_000.0
    frame["close"] = 10.0
    frame["entry_feasible"] = True
    frame["forward_return_1d"] = 0.01

    quality = build_module.run_quality_checks(frame, master)
    assert _verdict(quality, "no_post_delisting_rows") == "FAIL"
    assert quality["structurally_valid"] is False
