"""Announcement-based ST register (ENG_ST30): classifier, episode folding,
register evaluation and the strict PIT interval contract.

Fixtures are hand-written event sequences, not market data; nothing here is
used to make a research result look complete.
"""
from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.ashare import st_announcements as sa
from quantagent.data.providers.st_pit import (
    HistoricalSTCoverageError,
    attach_historical_st,
    load_historical_st_evidence,
)

T = pd.Timestamp


@pytest.mark.parametrize(
    "title, expected",
    [
        ("关于公司股票被实施退市风险警示暨停牌的公告", sa.IMPLEMENT),
        ("关于公司股票交易被实施其他风险警示的公告", sa.IMPLEMENT),
        ("关于撤销公司股票退市风险警示的公告", sa.REVOKE),
        ("关于撤销公司股票其他风险警示暨停牌的公告", sa.REVOKE),
        ("关于撤销退市风险警示并实施其他风险警示的公告", sa.TRANSITION),
        ("关于公司股票可能被实施退市风险警示的风险提示公告", sa.IGNORE),
        ("关于公司股票被实施其他风险警示相关事项的进展公告", sa.IGNORE),
        ("关于申请撤销公司股票其他风险警示的公告", sa.IGNORE),
        ("关于公司股票进入退市整理期交易的公告", sa.DELIST_PERIOD),
        ("2025年年度报告", sa.IGNORE),
    ],
)
def test_classify_title(title, expected):
    assert sa.classify_title(title) == expected


def test_effective_date_is_read_from_body_and_bounded_to_announcement():
    body = "本公司股票于2025年4月28日停牌1天，自2025年4月29日起实施退市风险警示。2019年1月1日曾发生"
    when, basis = sa.extract_effective_date(body, T("2025-04-28"))
    assert when == T("2025-04-29")
    assert basis == "from_date"
    # a date before the announcement (an old event quoted in the body) is never used
    assert sa.extract_effective_date("自2019年1月2日起实施退市风险警示", T("2025-04-28")) == (
        None, "not_found")


def _cal():
    return sa.Calendar.from_dates(pd.bdate_range("2020-01-01", "2026-12-31"))


def _events(rows):
    frame = pd.DataFrame(rows, columns=["symbol", "event", "announcement_date", "effective_date"])
    frame["announcement_id"] = [f"a{i}" for i in range(len(frame))]
    frame["title"] = frame["event"]
    frame["sec_name"] = None
    frame["effective_basis"] = "test"
    for col in ("announcement_date", "effective_date"):
        frame[col] = pd.to_datetime(frame[col])
    return frame


def test_implement_then_revoke_is_one_true_episode_end_exclusive():
    ev = _events([
        ("600001.SH", sa.IMPLEMENT, "2021-04-28", "2021-04-29"),
        ("600001.SH", sa.TRANSITION, "2022-04-28", "2022-04-29"),
        ("600001.SH", sa.REVOKE, "2023-05-10", "2023-05-11"),
    ])
    eps = sa.build_episodes(ev, coverage_start=T("2016-01-04"))
    assert len(eps) == 1
    ep = eps.iloc[0]
    assert ep.st_state == sa.STATE_TRUE
    assert (ep.effective_start, ep.effective_end) == (T("2021-04-29"), T("2023-05-11"))


def test_revoke_without_observed_start_is_unknown_not_false():
    ev = _events([("600002.SH", sa.REVOKE, "2018-05-10", "2018-05-11")])
    eps = sa.build_episodes(ev, coverage_start=T("2016-01-04"))
    assert list(eps.st_state) == [sa.STATE_UNKNOWN]
    assert (eps.iloc[0].effective_start, eps.iloc[0].effective_end) == (
        T("2016-01-04"), T("2018-05-11"))


def test_pit_intervals_are_complete_and_round_trip_through_the_strict_loader(tmp_path):
    episodes = pd.DataFrame([
        {"symbol": "600001.SH", "effective_start": T("2021-04-29"),
         "effective_end": T("2023-05-11"), "st_state": sa.STATE_TRUE,
         "available_at": T("2021-04-28"), "end_available_at": T("2023-05-10"),
         "source": "cninfo", "evidence_id": "a0"},
    ])
    life = pd.DataFrame([
        {"symbol": "600001.SH", "first_day": T("2000-01-04"), "last_day": pd.NaT},
        {"symbol": "600003.SH", "first_day": T("2022-01-04"), "last_day": T("2024-06-28")},
    ])
    pit = sa.pit_intervals(episodes, life, as_of=T("2026-11-24"),
                           coverage_start=T("2016-01-04"))
    one = pit[pit.symbol == "600001.SH"].reset_index(drop=True)
    assert list(one.is_st) == [False, True, False]
    assert one.loc[0, "end_date"] == T("2021-04-28")          # inclusive, day before start
    assert one.loc[1, "end_date"] == T("2023-05-10")          # exclusive end - 1 day
    assert one.loc[2, "start_date"] == T("2023-05-11")
    assert one.loc[2, "end_date"] == T("2026-11-24")
    assert pit[pit.symbol == "600003.SH"].end_date.max() == T("2024-06-28")

    path = tmp_path / "st.parquet"
    pit.to_parquet(path, index=False)
    evidence = load_historical_st_evidence(path)
    rows = pd.DataFrame({"symbol": ["600001.SH"] * 4,
                         "trade_date": ["2021-04-28", "2021-04-29", "2023-05-10", "2023-05-11"]})
    attached = attach_historical_st(rows, evidence)
    assert list(attached.is_st) == [False, True, True, False]


def test_unknown_span_has_no_row_so_the_loader_fails_closed(tmp_path):
    episodes = pd.DataFrame([
        {"symbol": "600002.SH", "effective_start": T("2016-01-04"),
         "effective_end": T("2018-05-11"), "st_state": sa.STATE_UNKNOWN,
         "available_at": pd.NaT, "end_available_at": T("2018-05-10"),
         "source": "cninfo", "evidence_id": "a1"},
    ])
    life = pd.DataFrame([{"symbol": "600002.SH", "first_day": T("2000-01-04"), "last_day": pd.NaT}])
    pit = sa.pit_intervals(episodes, life, as_of=T("2020-12-31"), coverage_start=T("2016-01-04"))
    assert list(pit.is_st) == [False]
    assert pit.iloc[0].start_date == T("2018-05-11")
    path = tmp_path / "st.parquet"
    pit.to_parquet(path, index=False)
    evidence = load_historical_st_evidence(path)
    ok = attach_historical_st(pd.DataFrame({"symbol": ["600002.SH"], "trade_date": ["2019-01-02"]}),
                              evidence)
    assert not ok.is_st.iloc[0]
    with pytest.raises(HistoricalSTCoverageError):
        attach_historical_st(pd.DataFrame({"symbol": ["600002.SH"], "trade_date": ["2017-06-01"]}),
                             evidence)


def test_evaluate_reconstruction_counts_unknown_as_a_miss_and_false_separately():
    cal = _cal()
    truth = pd.DataFrame([
        {"symbol": "000002.SZ", "effective_start": T("2021-03-01"), "effective_end": T("2021-09-01")},
        {"symbol": "000004.SZ", "effective_start": T("2022-03-01"), "effective_end": T("2022-09-01")},
    ])
    recon = pd.DataFrame([
        {"symbol": "000002.SZ", "effective_start": T("2021-03-02"),
         "effective_end": T("2021-09-01"), "st_state": sa.STATE_TRUE},
        {"symbol": "000004.SZ", "effective_start": T("2022-03-01"),
         "effective_end": T("2022-09-01"), "st_state": sa.STATE_UNKNOWN},
    ])
    days = pd.bdate_range("2021-01-01", "2022-12-30")
    rows = pd.DataFrame({"symbol": ["000002.SZ"] * len(days) + ["000004.SZ"] * len(days),
                         "trade_date": list(days) * 2})
    m = sa.evaluate_reconstruction(recon, truth, rows, cal,
                                   window=(T("2021-01-01"), T("2022-12-30")),
                                   symbols=["000002.SZ", "000004.SZ"])
    assert m["M1_episode_recall"] == 0.5            # the UNKNOWN span is not a recall
    assert m["M2_episode_precision"] == 1.0
    assert m["M3_start_within_1_session"] == 1.0    # one session late is within tolerance
    # the session the reconstruction starts late is the only silent false negative
    assert m["M5c_register_true_marked_false"] == pytest.approx(1 / m["M5_n_register_true"])
    assert m["register_true_marked_unknown"] > 0.45
    verdict = sa.acceptance(m)
    assert not verdict["all_pass"] and not verdict["M1_episode_recall"]["pass"]


def test_merge_contiguous_joins_a_rename_inside_one_warning():
    rows = pd.DataFrame([
        {"symbol": "000004.SZ", "effective_start": T("2022-05-06"), "effective_end": T("2023-06-28")},
        {"symbol": "000004.SZ", "effective_start": T("2023-06-28"), "effective_end": pd.NaT},
    ])
    merged = sa.merge_contiguous(rows)
    assert len(merged) == 1 and pd.isna(merged.iloc[0].effective_end)
