"""Cycle 32: the FRESH catch-up must not write a panel with an interior hole.

The vendor returns the latest ``count`` bars; with the old hard-coded ``count=40``
a 73-session window (2026-08-06..2026-11-24) would have staged only
2026-09-22..2026-11-24, and the supervisor's ``max(trade_date)`` check would have
reported the panel CURRENT.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


catchup = _load("catchup_panel_chunked")
repair = _load("repair_fresh_window_20260704")


def _calendar(start: str, end: str) -> pd.DataFrame:
    days = pd.bdate_range(start, end)
    return pd.DataFrame({"trade_date": days, "is_open": True})


def _rows(dates, symbols) -> pd.DataFrame:
    return pd.DataFrame(
        [{"symbol": s, "trade_date": d, "close": 10.0} for d in dates for s in symbols]
    )


SEEDS = [f"{i:06d}.SZ" for i in range(100)]


def test_full_window_passes():
    sessions = catchup.window_sessions(_calendar("2026-08-03", "2026-12-31"), "2026-08-06", "2026-11-24")
    counts = catchup.require_window_coverage(_rows(sessions, SEEDS), sessions, len(SEEDS))
    assert len(counts) == len(sessions)


def test_latest_40_bars_hole_is_refused():
    sessions = catchup.window_sessions(_calendar("2026-08-03", "2026-12-31"), "2026-08-06", "2026-11-24")
    assert len(sessions) > 40
    latest_40 = sessions[-40:]  # what count=40 returns
    with pytest.raises(catchup.WindowCoverageError, match="sessions below"):
        catchup.require_window_coverage(_rows(latest_40, SEEDS), sessions, len(SEEDS))


def test_thin_session_is_refused():
    sessions = catchup.window_sessions(_calendar("2026-08-03", "2026-12-31"), "2026-08-06", "2026-08-20")
    rows = _rows(sessions, SEEDS)
    rows = rows[~((rows["trade_date"] == sessions[3]) & (rows["symbol"].isin(SEEDS[:10])))]
    with pytest.raises(catchup.WindowCoverageError):
        catchup.require_window_coverage(rows, sessions, len(SEEDS))
    # 92/100 < 93% is refused; 93/100 passes
    rows93 = _rows(sessions, SEEDS)
    rows93 = rows93[~((rows93["trade_date"] == sessions[3]) & (rows93["symbol"].isin(SEEDS[:7])))]
    catchup.require_window_coverage(rows93, sessions, len(SEEDS))


def test_off_calendar_rows_are_refused():
    sessions = catchup.window_sessions(_calendar("2026-08-03", "2026-12-31"), "2026-08-06", "2026-08-20")
    rows = pd.concat([_rows(sessions, SEEDS), _rows([pd.Timestamp("2026-08-08")], SEEDS)])  # a Saturday
    with pytest.raises(catchup.WindowCoverageError, match="not calendar sessions"):
        catchup.require_window_coverage(rows, sessions, len(SEEDS))


def test_expired_calendar_is_refused():
    with pytest.raises(catchup.WindowCoverageError, match="calendar ends"):
        catchup.window_sessions(_calendar("2026-08-03", "2026-12-31"), "2026-12-28", "2027-01-05")


def test_closed_days_are_not_sessions():
    calendar = _calendar("2026-09-28", "2026-10-12")
    calendar.loc[calendar["trade_date"].between("2026-10-01", "2026-10-08"), "is_open"] = False
    sessions = catchup.window_sessions(calendar, "2026-09-28", "2026-10-12")
    assert pd.Timestamp("2026-10-05") not in sessions
    assert pd.Timestamp("2026-10-09") in sessions


class _FakeKlines:
    def __init__(self):
        self.calls = []

    def get(self, sym, **kwargs):
        self.calls.append(kwargs)
        return pd.DataFrame()


class _FakeTF:
    def __init__(self):
        self.klines = _FakeKlines()


def test_fetch_passes_requested_count():
    tf = _FakeTF()
    repair.fetch_with_retry(tf, "000001.SZ", count=83)
    assert tf.klines.calls[-1]["count"] == 83
    repair.fetch_with_retry(tf, "000001.SZ")
    assert tf.klines.calls[-1]["count"] == 40  # legacy default for other callers
    with pytest.raises(ValueError):
        repair.fetch_with_retry(tf, "000001.SZ", count=0)


def test_staging_lock_is_exclusive(tmp_path):
    first = catchup._acquire_staging_lock(tmp_path)
    assert first is not None
    assert catchup._acquire_staging_lock(tmp_path) is None
    first.close()
    again = catchup._acquire_staging_lock(tmp_path)
    assert again is not None
    again.close()


# --- main()-level wiring (review condition, Cycle 32) -----------------------

SYMS = ["000001.SZ", "000002.SZ", "600000.SH"]


def _seed_panel(path: Path, syms=None) -> None:
    syms = SYMS if syms is None else syms
    dates = pd.bdate_range("2026-07-27", "2026-08-05")
    rows = []
    for d in dates:
        for s in syms:
            rows.append(
                {
                    "symbol": s, "trade_date": d, "open": 10.0, "high": 10.0, "low": 10.0,
                    "close": 10.0, "volume": 1e6, "amount": 1e7, "is_limit_up": False,
                    "is_limit_down": False, "is_suspended": False, "is_st": False,
                    "available_at": d + pd.Timedelta(days=1), "source": "seed",
                    "source_type": "test", "source_reliability": 1.0, "point_in_time_valid": True,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


class _VendorKlines:
    """Latest ``count`` sessions up to END, like TickFlow's ``count`` semantics."""

    def __init__(self, sessions, fail=()):
        self.sessions = sessions
        self.fail = set(fail)

    def get(self, sym, period, count, adjust, as_dataframe):
        if sym in self.fail:
            raise RuntimeError("throttled")
        days = self.sessions[-int(count):]
        return pd.DataFrame(
            {"trade_date": days, "open": 10.0, "high": 10.2, "low": 9.9,
             "close": 10.1, "volume": 1e4, "amount": 1.01e7}
        )


class _Vendor:
    def __init__(self, sessions, fail=()):
        self.klines = _VendorKlines(sessions, fail)


def _wire(monkeypatch, tmp_path, vendor, syms=None):
    syms = SYMS if syms is None else syms
    panel = tmp_path / "market_panel.parquet"
    _seed_panel(panel, syms)
    cal = tmp_path / "trading_calendar.parquet"
    _calendar("2026-07-01", "2026-12-31").to_parquet(cal, index=False)
    monkeypatch.setattr(catchup, "PANEL", panel)
    monkeypatch.setattr(catchup, "STAGING", tmp_path / "_staging_catchup")
    monkeypatch.setattr(catchup, "CALENDAR", cal)
    master = tmp_path / "security_master.parquet"
    pd.DataFrame({"symbol": syms, "listing_date": pd.Timestamp("2000-01-04"),
                  "delisting_date": pd.NaT}).to_parquet(master, index=False)
    monkeypatch.setattr(catchup, "MASTER", master)
    monkeypatch.setattr(catchup, "DELISTINGS", tmp_path / "absent_delistings.parquet")
    monkeypatch.setattr(catchup, "SUSPENSION_DIRS", [])
    monkeypatch.setattr(catchup, "COHORT_START", pd.Timestamp("2026-08-01"))
    evidence = type("E", (), {"source_sha256": "x", "mode": "test",
                              "frame": pd.DataFrame({"symbol": syms})})()
    monkeypatch.setattr(catchup, "_load_required_st_evidence", lambda path: evidence)
    monkeypatch.setattr(catchup, "_require_st_coverage", lambda *a, **k: None)

    def _attach(frame, evidence):
        frame = frame.copy()
        frame["is_st"] = False
        frame["is_st_provenance"] = "test_dated"
        frame["st_evidence_sha256"] = "x"
        return frame

    monkeypatch.setattr(catchup, "attach_historical_st", _attach)
    monkeypatch.setattr(catchup.rep, "_tf_client", lambda: vendor)
    monkeypatch.setattr(sys, "argv", ["catchup", "--end", "2026-11-24", "--historical-st-path", "unused"])
    return panel


def _sessions():
    return list(pd.bdate_range("2026-08-06", "2026-11-24"))


def test_main_fills_every_session(monkeypatch, tmp_path):
    panel = _wire(monkeypatch, tmp_path, _Vendor(_sessions()))
    assert catchup.main() == 0
    out = pd.read_parquet(panel)
    window = out[out["trade_date"] > "2026-08-05"]
    assert window["trade_date"].nunique() == len(_sessions())
    assert (window.groupby("trade_date")["symbol"].nunique() == len(SYMS)).all()
    assert not list((tmp_path).glob(".market_panel.parquet.tmp-*"))


def test_main_refuses_latest_40_bars_hole(monkeypatch, tmp_path):
    panel = _wire(monkeypatch, tmp_path, _Vendor(_sessions()))
    real_fetch = catchup.rep.fetch_with_retry
    monkeypatch.setattr(catchup.rep, "fetch_with_retry", lambda tf, sym, attempts=3, count=40: real_fetch(tf, sym, attempts, 40))
    before = panel.read_bytes()
    with pytest.raises(SystemExit, match="sessions below"):
        catchup.main()
    assert panel.read_bytes() == before


def test_main_fetch_failure_retries_instead_of_dropping_symbol(monkeypatch, tmp_path):
    vendor = _Vendor(_sessions(), fail={"600000.SH"})
    panel = _wire(monkeypatch, tmp_path, vendor)
    monkeypatch.setattr(catchup.rep.time, "sleep", lambda s: None)
    before = panel.read_bytes()
    assert catchup.main() == catchup.EXIT_RETRY
    assert panel.read_bytes() == before
    vendor.klines.fail.clear()
    assert catchup.main() == 0
    out = pd.read_parquet(panel)
    assert "600000.SH" in set(out.loc[out["trade_date"] == "2026-11-24", "symbol"])


def test_seed_is_not_the_names_printed_on_the_last_date(monkeypatch, tmp_path):
    """Audit F2: a cohort name halted on panel_max used to drop out of every later seed."""
    panel = _wire(monkeypatch, tmp_path, _Vendor(_sessions()))
    frame = pd.read_parquet(panel)
    frame = frame[~((frame["symbol"] == "600000.SH") & (frame["trade_date"] == "2026-08-05"))]
    frame.to_parquet(panel, index=False)
    assert catchup.main() == 0
    out = pd.read_parquet(panel)
    assert "600000.SH" in set(out.loc[out["trade_date"] == "2026-11-24", "symbol"])


class _EmptyFor(_VendorKlines):
    """Successful empty answers for ``empty`` names (not exceptions)."""

    def __init__(self, sessions, empty=(), fail=()):
        super().__init__(sessions, fail)
        self.empty = set(empty)

    def get(self, sym, period, count, adjust, as_dataframe):
        if sym in self.empty and sym not in self.fail:
            return pd.DataFrame(columns=["trade_date", "open", "high", "low", "close",
                                         "volume", "amount"])
        return super().get(sym, period, count, adjust, as_dataframe)


BIG = [f"{i:06d}.SZ" for i in range(1, 30)] + ["600000.SH"]   # 30 seeds


def test_failure_retries_but_vendor_empty_does_not_block(monkeypatch, tmp_path):
    """Review blocker: a dark (listed, empty, no halt) name must not stall the merge;
    a failed request still must."""
    import json

    vendor = _Vendor(_sessions())
    vendor.klines = _EmptyFor(_sessions(), empty={"600000.SH"}, fail={"000002.SZ"})
    panel = _wire(monkeypatch, tmp_path, vendor, BIG)
    monkeypatch.setattr(catchup.rep.time, "sleep", lambda s: None)
    before = panel.read_bytes()
    assert catchup.main() == catchup.EXIT_RETRY          # 000002.SZ raised
    assert panel.read_bytes() == before
    vendor.klines.fail.clear()                            # only the empty name left
    assert catchup.main() == 0
    out = pd.read_parquet(panel)
    window = out[out["trade_date"] > "2026-08-05"]
    assert set(window["symbol"]) == set(BIG) - {"600000.SH"}
    record = json.loads((panel.parent / catchup.VENDOR_EMPTY_NAME).read_text())
    assert record["symbols"] == ["600000.SH"]
    assert record["classification"] == "MISSING_UNEXPLAINED"


def test_coverage_guard_denominator_is_names_that_returned_data(monkeypatch, tmp_path):
    # 29 of 30 seeds return a full window, 1 is dark: merges (29/30 >= 93%)
    vendor = _Vendor(_sessions())
    vendor.klines = _EmptyFor(_sessions(), empty={"600000.SH"})
    _wire(monkeypatch, tmp_path, vendor, BIG)
    assert catchup.main() == 0
    # ...but a latest-N hole among names that DID return data is still refused
    vendor2 = _Vendor(_sessions())
    vendor2.klines = _EmptyFor(_sessions(), empty={"600000.SH"})
    (tmp_path / "b").mkdir()
    panel = _wire(monkeypatch, tmp_path / "b", vendor2, BIG)
    real_fetch = catchup.rep.fetch_with_retry
    monkeypatch.setattr(catchup.rep, "fetch_with_retry",
                        lambda tf, sym, attempts=3, count=40: real_fetch(tf, sym, attempts, 40))
    before = panel.read_bytes()
    with pytest.raises(SystemExit, match="sessions below"):
        catchup.main()
    assert panel.read_bytes() == before


def test_empty_name_under_a_dated_halt_is_not_flagged(monkeypatch, tmp_path):
    import json

    vendor = _Vendor(_sessions())
    vendor.klines = _EmptyFor(_sessions(), empty={"600000.SH"})
    panel = _wire(monkeypatch, tmp_path, vendor, BIG)
    pit = tmp_path / "halts"
    (pit / "_suspension").mkdir(parents=True)
    pd.DataFrame({"symbol": ["600000.SH"], "effective_start": [pd.Timestamp("2026-08-01")],
                  "effective_end": [pd.Timestamp("2026-12-31")],
                  "suspension_reason": ["r"]}).to_parquet(pit / "suspension_intervals.parquet")
    monkeypatch.setattr(catchup, "SUSPENSION_DIRS", [pit])
    assert catchup.main() == 0
    assert json.loads((panel.parent / catchup.VENDOR_EMPTY_NAME).read_text())["symbols"] == []


def test_seed_universe_modes_and_dated_delistings():
    panel = pd.DataFrame({"symbol": ["000001.SZ", "600000.SH"],
                          "trade_date": pd.to_datetime(["2026-05-01", "2026-05-01"])})
    master = pd.DataFrame({
        "symbol": ["000001.SZ", "600000.SH", "600001.SH", "688001.SH", "920001.BJ"],
        "listing_date": pd.to_datetime(["2000-01-04"] * 3 + ["2027-03-01", "2020-01-02"]),
        "delisting_date": pd.to_datetime([pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT])})
    delist = pd.DataFrame({"symbol": ["600000.SH"], "effective_start": ["2026-06-01"]})
    seed, rep = catchup.seed_universe(panel, master, delist, "2026-11-25", "2026-11-25")
    assert seed == ["000001.SZ"] and rep["excluded_dated_delisting"] == 1
    seed, _ = catchup.seed_universe(panel, master, None, "2026-11-25", "2026-11-25", mode="master")
    assert seed == ["000001.SZ", "600000.SH", "600001.SH"]      # not yet listed / BSE excluded


def test_partial_publish_of_the_last_session_is_retried_not_merged(monkeypatch, tmp_path):
    """Review repro: only some seed names return the last session (INC-P1).
    Against names that returned data it looks complete; against the seed it is not."""
    class _Partial(_EmptyFor):
        def get(self, sym, period, count, adjust, as_dataframe):
            frame = super().get(sym, period, count, adjust, as_dataframe)
            if sym != "000001.SZ":                       # 1 of 3 names has the last bar
                frame = frame[frame["trade_date"] < self.sessions[-1]]
            return frame

    vendor = _Vendor(_sessions())
    vendor.klines = _Partial(_sessions())
    panel = _wire(monkeypatch, tmp_path, vendor)
    before = panel.read_bytes()
    with pytest.raises(SystemExit, match="sessions below"):   # returned-data guard
        catchup.main()
    assert panel.read_bytes() == before


def test_many_empty_answers_are_retried_not_merged(monkeypatch, tmp_path):
    """Review repro: 2 of the seed return the full window (incl. the last session),
    the rest answer empty. 2/2 passes the returned-data guard; the seed guard refuses."""
    vendor = _Vendor(_sessions())
    vendor.klines = _EmptyFor(_sessions(), empty=set(BIG[2:]))
    panel = _wire(monkeypatch, tmp_path, vendor, BIG)
    before = panel.read_bytes()
    assert catchup.main() == catchup.EXIT_RETRY
    assert panel.read_bytes() == before
