"""Cycle 32: a current-max panel with holes / unverified ST is never reported healthy."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freshness_is_incomplete_when_integrity_fails(tmp_path, monkeypatch) -> None:
    daily = _load("fresh_blind_daily")
    panel = tmp_path / "panel.parquet"
    pd.DataFrame({"symbol": ["600000.SH"], "trade_date": [pd.Timestamp.today().normalize()]}
                 ).to_parquet(panel)
    monkeypatch.setattr(daily, "PANEL", panel)
    monkeypatch.setattr(daily, "integrity_report", lambda *a, **k: {
        "status": "INCOMPLETE", "problems": ["INTERIOR_HOLES_UNEXPLAINED 3 (1 symbols)"]})
    out = daily.step_freshness()
    assert out["status"] == "INCOMPLETE"            # old code: "OK" on max date alone
    assert out["integrity"]["problems"] == ["INTERIOR_HOLES_UNEXPLAINED 3 (1 symbols)"]

    def boom(*a, **k):
        raise OSError("calendar missing")
    monkeypatch.setattr(daily, "integrity_report", boom)
    assert daily.step_freshness()["status"] == "INCOMPLETE"


def test_healthcheck_surfaces_integrity_problems(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    health = _load("fresh_blind_healthcheck")
    fake = types.ModuleType("fresh_blind_daily")
    fake.integrity_report = lambda: {"status": "INCOMPLETE", "problems": [
        "INTERIOR_HOLES_UNEXPLAINED 3 (1 symbols)", "ST_PROVENANCE_UNVERIFIED 5 rows"]}
    monkeypatch.setitem(sys.modules, "fresh_blind_daily", fake)
    problems, _ = health.panel_integrity_problems()
    assert problems == ["PANEL_INTERIOR_HOLES_UNEXPLAINED 3 (1 symbols)",
                        "PANEL_ST_PROVENANCE_UNVERIFIED 5 rows"]
    fake.integrity_report = lambda: (_ for _ in ()).throw(OSError("unreadable"))
    problems, _ = health.panel_integrity_problems()
    assert problems and problems[0].startswith("PANEL_INTEGRITY_UNMEASURED")


def test_update_market_panel_daily_refuses_past_calendar(tmp_path) -> None:
    import pytest

    mod = _load("update_market_panel_daily")
    cal = tmp_path / "cal.parquet"
    pd.DataFrame({"trade_date": pd.to_datetime(["2026-12-30", "2026-12-31"])}).to_parquet(cal)
    mod.require_calendar_covers(cal, pd.Timestamp("2026-12-31"))
    with pytest.raises(SystemExit, match="calendar ends 2026-12-31"):
        mod.require_calendar_covers(cal, pd.Timestamp("2027-01-04"))
    with pytest.raises(SystemExit, match="missing"):
        mod.require_calendar_covers(tmp_path / "absent.parquet", pd.Timestamp("2026-12-31"))


def test_healthcheck_remediation_uses_the_guarded_writer() -> None:
    text = (SCRIPTS / "fresh_blind_healthcheck.py").read_text()
    assert "scripts/update_market_panel_daily.py" not in text
    assert "scripts/catchup_panel_chunked.py" in text


def test_calendar_refresh_may_only_extend(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS.parent / "src"))
    mod = _load("u0_pit_intervals")
    old = pd.DataFrame({"trade_date": pd.to_datetime(["2026-12-30", "2026-12-31"])})
    ext = pd.DataFrame({"trade_date": pd.to_datetime(["2026-12-30", "2026-12-31", "2027-01-04"])})
    assert mod.calendar_regression(old, ext) is None
    assert "new last date" in mod.calendar_regression(old, old.iloc[:1])
    assert "disappear" in mod.calendar_regression(
        old, pd.DataFrame({"trade_date": pd.to_datetime(["2026-12-31", "2027-01-04"])}))
