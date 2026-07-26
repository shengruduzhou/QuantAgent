"""Panel-assembly semantics: gaps, boundaries, conflicts, coverage."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pd = pytest.importorskip("pandas")


def _load_assembler():
    spec = importlib.util.spec_from_file_location(
        "u0_assemble_panel", REPO / "scripts/u0_assemble_panel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def assembler():
    return _load_assembler()


def _panel(rows):
    frame = pd.DataFrame(rows)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _master(rows):
    frame = pd.DataFrame(rows)
    for column in ("listing_date", "delisting_date"):
        frame[column] = pd.to_datetime(frame.get(column), errors="coerce")
    return frame


def test_missing_session_inside_a_halt_is_labelled_suspended(assembler, tmp_path, monkeypatch):
    calendar = pd.DataFrame({"trade_date": pd.to_datetime(
        ["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23"])})
    halts = pd.DataFrame({
        "symbol": ["600519.SH"], "effective_start": [pd.Timestamp("2026-07-21")],
        "effective_end": [pd.Timestamp("2026-07-22")],
        "suspension_reason": ["刊登重要公告"],
    })
    pit = tmp_path / "pit"
    pit.mkdir(parents=True)
    calendar.to_parquet(pit / "trading_calendar.parquet", index=False)
    halts.to_parquet(pit / "suspension_intervals.parquet", index=False)
    monkeypatch.setattr(assembler, "PIT", pit)

    panel = _panel([
        {"symbol": "600519.SH", "trade_date": "2026-07-20", "close": 10.0},
        {"symbol": "600519.SH", "trade_date": "2026-07-23", "close": 10.5},
    ])
    master = _master([{"symbol": "600519.SH", "listing_date": "2020-01-01",
                       "delisting_date": None}])
    gaps = assembler.classify_session_gaps(panel, master)
    assert len(gaps) == 2
    assert set(gaps["classification"]) == {"SUSPENDED"}
    assert "刊登重要公告" in gaps["evidence"].iloc[0]


def test_missing_session_with_no_halt_record_is_flagged_not_hidden(assembler, tmp_path, monkeypatch):
    calendar = pd.DataFrame({"trade_date": pd.to_datetime(
        ["2026-07-20", "2026-07-21", "2026-07-22"])})
    pit = tmp_path / "pit"
    pit.mkdir(parents=True)
    calendar.to_parquet(pit / "trading_calendar.parquet", index=False)
    pd.DataFrame({"symbol": [], "effective_start": [], "effective_end": [],
                  "suspension_reason": []}).to_parquet(pit / "suspension_intervals.parquet",
                                                       index=False)
    monkeypatch.setattr(assembler, "PIT", pit)

    panel = _panel([
        {"symbol": "000001.SZ", "trade_date": "2026-07-20", "close": 10.0},
        {"symbol": "000001.SZ", "trade_date": "2026-07-22", "close": 10.5},
    ])
    master = _master([{"symbol": "000001.SZ", "listing_date": "2000-01-01",
                       "delisting_date": None}])
    gaps = assembler.classify_session_gaps(panel, master)
    assert gaps["classification"].tolist() == ["MISSING_UNEXPLAINED"]


def test_sessions_after_delisting_are_not_counted_as_gaps(assembler, tmp_path, monkeypatch):
    calendar = pd.DataFrame({"trade_date": pd.to_datetime(
        ["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23"])})
    pit = tmp_path / "pit"
    pit.mkdir(parents=True)
    calendar.to_parquet(pit / "trading_calendar.parquet", index=False)
    monkeypatch.setattr(assembler, "PIT", pit)

    panel = _panel([{"symbol": "600087.SH", "trade_date": "2026-07-20", "close": 1.0}])
    master = _master([{"symbol": "600087.SH", "listing_date": "1997-01-01",
                       "delisting_date": "2026-07-20"}])
    gaps = assembler.classify_session_gaps(panel, master)
    assert gaps.empty


def test_coverage_matrix_reports_uncovered_securities_with_a_reason(assembler):
    master = _master([
        {"symbol": "600519.SH", "code": "600519", "exchange": "SSE", "board": "SH_Main",
         "security_type": "A_share", "status": "listed", "listing_date": "2001-08-27",
         "delisting_date": None, "current_st": False, "bse_legacy_code": False},
        {"symbol": "600087.SH", "code": "600087", "exchange": "SSE", "board": "SH_Main",
         "security_type": "A_share", "status": "delisted", "listing_date": "1997-01-01",
         "delisting_date": "2019-06-01", "current_st": False, "bse_legacy_code": False},
    ])
    panel = _panel([{"symbol": "600519.SH", "trade_date": "2026-07-20", "close": 10.0,
                     "amount": 1.0, "serving_provider": "tickflow"}])
    coverage = assembler.build_coverage(master, panel, pd.DataFrame())
    covered = coverage.set_index("symbol")
    assert covered.loc["600519.SH", "covered"]
    assert not covered.loc["600087.SH", "covered"]
    assert covered.loc["600087.SH", "blocked_reason"] == "NO_VENDOR_HISTORY_DELISTED"
    assert covered.loc["600519.SH", "serving_provider"] == "tickflow"


def test_provider_precedence_puts_the_entitled_source_first(assembler):
    names = [name for name, _ in assembler.SOURCE_PRECEDENCE]
    assert names[0].startswith("tickflow")
    assert names[-1] == "tencent", "the public fallback must never outrank the entitled provider"


def test_assembler_declares_one_adjustment_for_the_whole_panel():
    text = (REPO / "scripts/u0_assemble_panel.py").read_text()
    assert "raw traded prices" in text
    # the panel must not carry null bars for non-traded sessions
    assert "traded sessions only" in text
