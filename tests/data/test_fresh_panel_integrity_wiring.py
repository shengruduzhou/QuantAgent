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
