"""ENG_ST31 forward-anchored ST register (pure functions)."""
from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.ashare import st_forward as fwd
from quantagent.data.ashare.st_announcements import Calendar

# Mon 2026-07-20 .. Fri 2026-08-14, weekdays only
SESSIONS = pd.bdate_range("2026-07-20", "2026-08-14")
CAL = Calendar.from_dates(SESSIONS)


def _ts(day: str) -> pd.Timestamp:
    return pd.Timestamp(day)


def _event(symbol, event, effective, announced, title="", sec_name="", aid="a1"):
    return {"symbol": symbol, "event": event, "effective_date": _ts(effective),
            "announcement_date": _ts(announced), "announcement_id": aid, "title": title,
            "sec_name": sec_name, "effective_basis": "body:test"}


def _events(*rows):
    cols = ["symbol", "event", "effective_date", "announcement_date", "announcement_id", "title",
            "sec_name", "effective_basis"]
    return pd.DataFrame(list(rows), columns=cols)


def _obs(*rows):
    cols = ["symbol", "kind", "observed_at", "sec_name", "source", "ref"]
    return pd.DataFrame(list(rows), columns=cols)


def _universe(symbol="600001.SH", name="甲公司"):
    # anchor: Sunday 2026-07-26 evening -> describes Fri 07-24 or Mon 07-27
    return pd.DataFrame([{"symbol": symbol, "anchor_session": _ts("2026-07-24"),
                          "start_session": _ts("2026-07-27"), "start_name": name}])


def _states(rows):
    return dict(zip(rows["trade_date"].dt.strftime("%m-%d"), rows["state"]))


def test_snapshot_sessions_outside_and_inside_trading_hours():
    sunday = pd.Timestamp("2026-07-26 19:39", tz="Asia/Shanghai")
    assert fwd.snapshot_sessions(sunday, CAL) == (_ts("2026-07-24"), _ts("2026-07-27"))
    evening = pd.Timestamp("2026-07-28 23:14", tz="Asia/Shanghai")
    assert fwd.snapshot_sessions(evening, CAL) == (_ts("2026-07-28"), _ts("2026-07-29"))
    early = pd.Timestamp("2026-07-28 05:00", tz="Asia/Shanghai")
    assert fwd.snapshot_sessions(early, CAL) == (_ts("2026-07-27"), _ts("2026-07-28"))
    midday = pd.Timestamp("2026-07-28 10:30", tz="Asia/Shanghai")
    assert fwd.snapshot_sessions(midday, CAL) == (_ts("2026-07-28"), _ts("2026-07-29"))
    utc = pd.Timestamp("2026-07-26T11:39:51Z")          # A0 as stored in the master
    assert fwd.snapshot_sessions(utc, CAL) == (_ts("2026-07-24"), _ts("2026-07-27"))


def test_name_sessions_tolerate_one_session_each_side():
    assert fwd.name_sessions(_ts("2026-07-29"), CAL) == (
        _ts("2026-07-28"), _ts("2026-07-29"), _ts("2026-07-30"))
    # weekend disclosure: Friday and Monday
    assert fwd.name_sessions(_ts("2026-08-01"), CAL) == (_ts("2026-07-31"), _ts("2026-08-03"))


def test_constant_state_without_events_or_observations():
    rows, log = fwd.run_forward(universe=_universe(name="*ST甲"), events=_events(),
                                observations=_obs(), calendar=CAL, as_of=_ts("2026-07-31"))
    assert set(rows["state"]) == {"TRUE"} and len(rows) == 5 and log.empty


def test_implement_and_revoke_move_the_state_on_their_effective_session():
    ev = _events(
        _event("600001.SH", "IMPLEMENT", "2026-07-29", "2026-07-28", "关于公司股票被实施其他风险警示的公告",
               "甲公司", "i1"),
        _event("600001.SH", "REVOKE", "2026-08-05", "2026-08-04", "关于撤销公司股票其他风险警示的公告",
               "ST甲", "r1"))
    obs = _obs(("600001.SH", "name", _ts("2026-07-31"), "ST甲", "cninfo", "n1"),
               ("600001.SH", "name", _ts("2026-08-07"), "甲公司", "cninfo", "n2"))
    rows, log = fwd.run_forward(universe=_universe(), events=ev, observations=obs, calendar=CAL,
                                as_of=_ts("2026-08-07"))
    s = _states(rows)
    assert s["07-28"] == "FALSE" and s["07-29"] == "TRUE" and s["08-04"] == "TRUE"
    assert s["08-05"] == "FALSE" and log.empty


def test_contradicting_name_makes_the_gap_unknown_and_restarts():
    # a missed revocation: the carried state stays ST, a plain name contradicts it
    obs = _obs(("600001.SH", "name", _ts("2026-07-29"), "*ST甲", "cninfo", "n1"),
               ("600001.SH", "name", _ts("2026-08-06"), "甲公司", "cninfo", "n2"))
    rows, log = fwd.run_forward(universe=_universe(name="*ST甲"), events=_events(),
                                observations=obs, calendar=CAL, as_of=_ts("2026-08-12"))
    s = _states(rows)
    # confirmed through 07-28 (earliest tolerated session of n1), UNKNOWN to 08-07
    assert s["07-28"] == "TRUE"
    assert all(s[d] == "UNKNOWN" for d in ("07-29", "07-30", "08-05", "08-06", "08-07"))
    assert s["08-10"] == "FALSE" and s["08-12"] == "FALSE"
    assert len(log) == 1 and log.iloc[0]["observation_ref"] == "n2"


def test_misclassified_implement_is_caught_by_the_end_anchor():
    ev = _events(_event("600001.SH", "IMPLEMENT", "2026-07-30", "2026-07-29",
                        "关于公司股票被实施其他风险警示的公告", "甲公司", "bad"))
    end = pd.Timestamp("2026-08-07 23:00", tz="Asia/Shanghai")
    obs = _obs(("600001.SH", "snapshot", end, "甲公司", "tickflow", "A1"))
    rows, log = fwd.run_forward(universe=_universe(), events=ev, observations=obs, calendar=CAL,
                                as_of=_ts("2026-08-07"))
    s = _states(rows)
    # nothing confirmed the carried state after the anchor (07-24), so every
    # session up to the contradicting snapshot is UNKNOWN (fail closed, §3.5)
    assert set(rows["state"]) == {"UNKNOWN"} and len(log) == 1


def test_anchor_rolled_over_before_an_event_on_the_first_session():
    # the vendor already shows the new ST name on Sunday for Monday's implementation
    ev = _events(_event("600001.SH", "IMPLEMENT", "2026-07-27", "2026-07-24",
                        "关于公司股票被实施退市风险警示的公告", "甲公司", "i1"))
    rows, _ = fwd.run_forward(universe=_universe(name="*ST甲"), events=ev, observations=_obs(),
                              calendar=CAL, as_of=_ts("2026-07-28"))
    assert set(rows["state"]) == {"TRUE"}


def test_negated_title_is_not_an_implementation():
    ev = _events(_event("600001.SH", "IMPLEMENT", "2026-07-29", "2026-07-28",
                        "关于公司股票交易不用被实行其他风险警示的公告", "甲公司", "neg"))
    out = fwd.prefilter_events(ev)
    assert out.iloc[0]["event"] == "IGNORE" and out.iloc[0]["prefilter"] == "negated_implement"
    rows, _ = fwd.run_forward(universe=_universe(), events=ev, observations=_obs(), calendar=CAL,
                              as_of=_ts("2026-07-31"))
    assert set(rows["state"]) == {"FALSE"}


def test_full_revocation_saying_no_other_warning_stays_a_revocation():
    ev = _events(_event("600001.SH", "REVOKE", "2026-07-29", "2026-07-28",
                        "关于撤销公司股票退市风险警示且不实施其他风险警示的公告", "*ST甲", "r1"))
    assert fwd.prefilter_events(ev).iloc[0]["event"] == "REVOKE"
    rows, _ = fwd.run_forward(universe=_universe(name="*ST甲"), events=ev, observations=_obs(),
                              calendar=CAL, as_of=_ts("2026-07-31"))
    s = _states(rows)
    assert s["07-28"] == "TRUE" and s["07-29"] == "FALSE"


def test_revoke_that_continues_a_warning_keeps_the_security_marked():
    ev = _events(_event("600001.SH", "REVOKE", "2026-07-29", "2026-07-28",
                        "关于撤销公司股票其他风险警示继续实施退市风险警示的公告", "*ST甲", "t1"))
    assert fwd.prefilter_events(ev).iloc[0]["event"] == "TRANSITION"
    rows, _ = fwd.run_forward(universe=_universe(name="*ST甲"), events=ev, observations=_obs(),
                              calendar=CAL, as_of=_ts("2026-07-31"))
    assert set(rows["state"]) == {"TRUE"}


def test_partial_revocation_uses_the_name_on_the_notice():
    # *ST with an added 其他 warning: revoking 退市 leaves ST
    ev = _events(
        _event("600001.SH", "IMPLEMENT", "2026-07-28", "2026-07-27", "关于公司股票叠加实施其他风险警示的公告",
               "*ST甲", "i1"),
        _event("600001.SH", "REVOKE", "2026-07-30", "2026-07-29", "关于撤销公司股票退市风险警示的公告",
               "*ST甲", "r1"))
    rows, _ = fwd.run_forward(universe=_universe(name="*ST甲"), events=ev, observations=_obs(),
                              calendar=CAL, as_of=_ts("2026-07-31"))
    assert set(rows["state"]) == {"TRUE"}


def test_delisting_period_ends_the_warning():
    ev = _events(_event("600001.SH", "DELIST_PERIOD", "2026-07-30", "2026-07-30",
                        "关于公司股票进入退市整理期交易的公告", "*ST甲", "d1"))
    rows, _ = fwd.run_forward(universe=_universe(name="*ST甲"), events=ev, observations=_obs(),
                              calendar=CAL, as_of=_ts("2026-07-31"))
    s = _states(rows)
    assert s["07-29"] == "TRUE" and s["07-30"] == "FALSE"


def test_rows_to_intervals_covers_weekends_and_leaves_unknown_uncovered():
    rows = pd.DataFrame({"symbol": "600001.SH",
                         "trade_date": pd.to_datetime(["2026-07-30", "2026-07-31", "2026-08-03",
                                                       "2026-08-04", "2026-08-05"]),
                         "state": ["TRUE", "TRUE", "UNKNOWN", "FALSE", "FALSE"]})
    out = fwd.rows_to_intervals(rows, CAL, as_of=_ts("2026-08-06"), source="test")
    assert len(out) == 2
    first, second = out.iloc[0], out.iloc[1]
    assert first["start_date"] == _ts("2026-07-30") and first["end_date"] == _ts("2026-08-02")
    assert bool(first["is_st"]) and first["available_at"] == "2026-07-30T00:00:00+08:00"
    assert second["start_date"] == _ts("2026-08-04") and second["end_date"] == _ts("2026-08-06")
    assert not bool(second["is_st"])


def test_intervals_load_in_the_strict_contract(tmp_path):
    from quantagent.data.providers.st_pit import (
        HistoricalSTCoverageError,
        attach_historical_st,
        load_historical_st_evidence,
    )

    rows = pd.DataFrame({"symbol": "600001.SH",
                         "trade_date": pd.to_datetime(["2026-07-30", "2026-07-31", "2026-08-03"]),
                         "state": ["TRUE", "UNKNOWN", "FALSE"]})
    path = tmp_path / "ev.parquet"
    fwd.rows_to_intervals(rows, CAL, as_of=_ts("2026-08-03"), source="test").to_parquet(path)
    evidence = load_historical_st_evidence(path)
    ok = attach_historical_st(pd.DataFrame({"symbol": ["600001.SH"] * 2,
                                            "trade_date": pd.to_datetime(["2026-07-30", "2026-08-03"])}),
                              evidence)
    assert ok["is_st"].tolist() == [True, False]
    with pytest.raises(HistoricalSTCoverageError):
        attach_historical_st(pd.DataFrame({"symbol": ["600001.SH"],
                                           "trade_date": pd.to_datetime(["2026-07-31"])}), evidence)


def test_evaluate_counts_rows_and_transitions():
    sessions = CAL.sessions[(CAL.sessions >= "2026-07-24") & (CAL.sessions <= "2026-08-07")]
    register = pd.DataFrame({"symbol": ["600001.SH"], "effective_start": [_ts("2026-07-29")],
                             "effective_end": [pd.NaT]})
    truth = fwd.register_states(register, ["600001.SH"], sessions)
    # reconstruction one session late
    recon = truth.rename(columns={"truth": "state"}).copy()
    recon["state"] = ["TRUE" if d >= _ts("2026-07-30") else "FALSE" for d in recon["trade_date"]]
    m = fwd.evaluate_forward(recon, truth, CAL, window=("2026-07-27", "2026-08-07"))
    assert m["V3_n"] == 1 and m["V3"] == 1.0          # within ±1 session
    assert m["V1"] == 1.0 and m["missed_true_rows"] == 1
    # two sessions late: transition missed
    recon["state"] = ["TRUE" if d >= _ts("2026-07-31") else "FALSE" for d in recon["trade_date"]]
    m = fwd.evaluate_forward(recon, truth, CAL, window=("2026-07-27", "2026-08-07"))
    assert m["V3"] == 0.0 and m["V3_misses"][0]["date"] == "2026-07-29"
    # an UNKNOWN span over the true date counts as matched, rows are coverage loss
    recon["state"] = ["FALSE" if d < _ts("2026-07-28") else
                      ("UNKNOWN" if d <= _ts("2026-07-30") else "TRUE") for d in recon["trade_date"]]
    m = fwd.evaluate_forward(recon, truth, CAL, window=("2026-07-27", "2026-08-07"))
    assert m["V3"] == 1.0 and m["unknown_rows"] == 3 and m["V1"] == 1.0 and m["V2"] == 1.0
    assert fwd.acceptance(m)["V4"]["pass"] is False


def test_listing_after_the_anchor_starts_unmarked():
    universe = pd.DataFrame([{"symbol": "603999.SH", "anchor_session": _ts("2026-08-04"),
                              "start_session": _ts("2026-08-04"), "start_name": ""}])
    rows, _ = fwd.run_forward(universe=universe, events=_events(), observations=_obs(),
                              calendar=CAL, as_of=_ts("2026-08-06"))
    assert rows["trade_date"].min() == _ts("2026-08-04") and set(rows["state"]) == {"FALSE"}


def test_an_event_without_an_effective_session_refuses():
    ev = _events(_event("600001.SH", "IMPLEMENT", "2026-07-29", "2026-07-28",
                        "关于公司股票被实施其他风险警示的公告", "甲公司", "i1"))
    ev["effective_date"] = pd.NaT
    with pytest.raises(ValueError, match="no effective session"):
        fwd.run_forward(universe=_universe(), events=ev, observations=_obs(), calendar=CAL,
                        as_of=_ts("2026-07-31"))
