"""Cycle 32 audit F2: trailing FRESH cohort names are planned, verified and merged."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load():
    import sys
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("backfill_under_test",
                                                  SCRIPTS / "backfill_fresh_trailing_symbols.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SESS = list(pd.bdate_range("2026-04-20", "2026-06-30"))


def _row(sym, day, close=10.0):
    return {"symbol": sym, "trade_date": day, "open": close, "high": close, "low": close,
            "close": close, "volume": 1e6, "amount": close * 1e6, "is_st": False,
            "is_limit_up": False, "is_limit_down": False, "is_suspended": False,
            "is_st_provenance": "dated_pit_st_evidence", "st_evidence_sha256": "x",
            "available_at": day + pd.Timedelta(days=1), "source": "seed"}


def _fixture(monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod.cu, "COHORT_START", pd.Timestamp("2026-05-19"))
    rows = [_row("000001.SZ", d) for d in SESS]
    rows += [_row("600000.SH", d) for d in SESS if d <= pd.Timestamp("2026-06-10")]
    panel = pd.DataFrame(rows)
    master = pd.DataFrame({"symbol": ["000001.SZ", "600000.SH"],
                           "listing_date": pd.Timestamp("2000-01-04"), "delisting_date": pd.NaT})
    cal = pd.DataFrame({"trade_date": SESS, "is_open": True})
    ev = pd.DataFrame({"symbol": ["000001.SZ", "600000.SH"],
                       "start_date": pd.to_datetime(["2016-01-04", "2016-01-04"]),
                       "end_date": pd.to_datetime(["2026-12-31", "2026-12-31"])})
    return mod, panel, master, cal, ev


def test_plan_names_the_trailing_cohort_symbol_and_eta(monkeypatch) -> None:
    mod, panel, master, cal, ev = _fixture(monkeypatch)
    plan, report = mod.plan_backfill(panel, master, None, cal, ev)
    assert list(plan["symbol"]) == ["600000.SH"]
    need = [d for d in SESS if d > pd.Timestamp("2026-06-10")]
    assert int(plan.loc[0, "sessions_to_fill"]) == len(need)
    assert report["requests"] == 1 and report["eta_minutes_at_10_per_min"] == 0.1


def test_verify_accepts_only_complete_or_halt_chained_names(monkeypatch, tmp_path) -> None:
    mod, panel, master, cal, ev = _fixture(monkeypatch)
    plan, _ = mod.plan_backfill(panel, master, None, cal, ev)
    need = [d for d in SESS if d > pd.Timestamp("2026-06-10")]
    pmax = pd.Timestamp(SESS[-1])
    staged = pd.DataFrame([_row("600000.SH", d, 11.0) for d in need[2:]])[
        ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]]
    staged.to_parquet(tmp_path / "600000.SH.parquet", index=False)
    _, verdict = mod.verify(plan, tmp_path, cal, pd.DataFrame(), pmax)
    assert verdict["verdicts"] == {"UNEXPLAINED_MISSING_SESSIONS": 1}
    halts = pd.DataFrame({"symbol": ["600000.SH"], "effective_start": [need[0]],
                          "effective_end": [need[1]]})
    new, verdict = mod.verify(plan, tmp_path, cal, halts, pmax)
    assert verdict["verdicts"] == {"VERIFIED": 1} and len(new) == len(need) - 2
    # a halt that starts after an unexplained session does not excuse the run
    late = halts.assign(effective_start=[need[1]])
    _, verdict = mod.verify(plan, tmp_path, cal, late, pmax)
    assert verdict["verdicts"] == {"UNEXPLAINED_MISSING_SESSIONS": 1}


def test_build_panel_merges_with_dated_st_and_flags(monkeypatch) -> None:
    mod, panel, master, cal, ev = _fixture(monkeypatch)
    need = [d for d in SESS if d > pd.Timestamp("2026-06-10")]
    new = pd.DataFrame([_row("600000.SH", d, 11.0) for d in need])[
        ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]]

    def attach(frame, evidence):
        return frame.assign(is_st=False, is_st_provenance="dated_pit_st_evidence",
                            st_evidence_sha256="x")
    monkeypatch.setattr(mod, "attach_historical_st", attach)
    evidence = type("E", (), {"frame": ev})()
    merged = mod.build_panel(panel, new, evidence=evidence)
    added = merged[merged["source"] == "tickflow_backfill_trailing"]
    assert len(added) == len(need)
    assert not merged.duplicated(["symbol", "trade_date"]).any()
    first = added.sort_values("trade_date").iloc[0]
    assert bool(first["is_limit_up"])          # 10.0 -> 11.0 is a +10% main-board limit
    assert set(added["is_st_provenance"]) == {"dated_pit_st_evidence"}


def test_sh_rows_before_dated_evidence_are_labelled_unknown_not_aborted(monkeypatch, tmp_path) -> None:
    """601088.SH on 2026-07-20: SH dated ST starts 2026-07-27. The build used to
    abort with HistoricalSTCoverageError; it must label the row unknown instead."""
    from quantagent.data.providers.st_pit import load_historical_st_evidence

    mod = _load()
    sym = "601088.SH"
    old_days = list(pd.bdate_range("2026-07-01", "2026-07-02"))
    panel = pd.DataFrame([_row(sym, d) for d in old_days])
    panel["is_st"] = True                                   # the period's broadcast value
    panel["is_st_provenance"] = "current_snapshot_broadcast"
    panel["st_evidence_sha256"] = None
    new_days = [pd.Timestamp("2026-07-20"), pd.Timestamp("2026-07-27")]
    new = pd.DataFrame([_row(sym, d, 10.1) for d in new_days])[
        ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]]
    path = tmp_path / "ev.parquet"
    pd.DataFrame({"symbol": [sym], "start_date": ["2026-07-27"], "end_date": ["2026-11-24"],
                  "is_st": [False], "available_at": ["2026-07-27T00:00:00+08:00"]}).to_parquet(path)
    evidence = load_historical_st_evidence(path)
    merged = mod.build_panel(panel, new, evidence)
    added = merged[merged["source"] == "tickflow_backfill_trailing"].set_index("trade_date")
    assert added.loc[new_days[0], "is_st_provenance"] == mod.ST_UNKNOWN
    assert bool(added.loc[new_days[0], "is_st"]) is True      # carried, not invented
    assert pd.isna(added.loc[new_days[0], "st_evidence_sha256"])
    assert added.loc[new_days[1], "is_st_provenance"] == "dated_pit_st_evidence"
    assert bool(added.loc[new_days[1], "is_st"]) is False
    assert "current_snapshot" in mod.ST_UNKNOWN and "broadcast" in mod.ST_UNKNOWN
    summary = mod.st_coverage_summary(new, evidence.frame)
    assert summary["rows_dated_st"] == 1 and summary["rows_unknown_st"] == 1
    assert summary["per_date_unknown"] == {"2026-07-20": 1}
    # a symbol with no existing broadcast row has nothing to carry: refused
    import pytest
    with pytest.raises(SystemExit, match="no existing unknown-ST row"):
        mod.build_panel(panel.assign(is_st_provenance="dated_pit_st_evidence"), new, evidence)


def test_symbol_without_st_value_is_excluded_whole_not_aborting_the_run(tmp_path) -> None:
    """Five SH names last printed 2026-05-18: no FRESH broadcast row to carry and no
    dated SH evidence before 07-27. They must be excluded whole (no interior hole,
    no invented ST state) while the other names still backfill."""
    from quantagent.data.providers.st_pit import load_historical_st_evidence

    mod = _load()
    carried, dark = "601088.SH", "600037.SH"
    panel = pd.DataFrame([_row(s, pd.Timestamp("2026-07-01")) for s in (carried, dark)])
    panel["is_st"] = False
    panel["is_st_provenance"] = ["current_snapshot_broadcast", "tickflow:exchanges.get_instruments"]
    panel["st_evidence_sha256"] = None
    days = [pd.Timestamp("2026-07-20"), pd.Timestamp("2026-07-27")]
    new = pd.DataFrame([_row(s, d, 10.1) for s in (carried, dark) for d in days])[
        ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]]
    path = tmp_path / "ev.parquet"
    pd.DataFrame({"symbol": [carried, dark], "start_date": ["2026-07-27"] * 2,
                  "end_date": ["2026-11-24"] * 2, "is_st": [False, False],
                  "available_at": ["2026-07-27T00:00:00+08:00"] * 2}).to_parquet(path)
    evidence = load_historical_st_evidence(path)

    kept, excluded = mod.split_symbols_without_st_value(panel, new, evidence)
    assert excluded == [dark]
    assert set(kept["symbol"]) == {carried}                  # excluded whole, both days
    merged = mod.build_panel(panel, kept, evidence)           # no SystemExit
    added = merged[merged["source"] == "tickflow_backfill_trailing"]
    assert set(added["symbol"]) == {carried} and len(added) == 2
    # nothing to exclude when every uncovered symbol has a carriable row
    _, none = mod.split_symbols_without_st_value(panel, new[new["symbol"] == carried], evidence)
    assert none == []


def test_output_refuses_live_panel_dir_and_existing(tmp_path) -> None:
    import pytest

    mod = _load()
    live = tmp_path / "silver" / "market_panel" / "market_panel.parquet"
    live.parent.mkdir(parents=True)
    with pytest.raises(SystemExit, match="new version"):
        mod.check_output_path(live, live)
    with pytest.raises(SystemExit, match="live panel directory"):
        mod.check_output_path(live.parent / "market_panel.v2.parquet", live)
    existing = tmp_path / "out.parquet"
    existing.write_bytes(b"x")
    with pytest.raises(SystemExit, match="existing"):
        mod.check_output_path(existing, live)
    mod.check_output_path(tmp_path / "fresh" / "new.parquet", live)
