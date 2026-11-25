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


def _suspension_args(tmp_path, out):
    import argparse

    cal = tmp_path / "cal.parquet"
    pd.DataFrame({"trade_date": pd.to_datetime(["2026-11-23", "2026-11-24"])}).to_parquet(cal)
    return argparse.Namespace(calendar=str(cal), out_dir=str(out), start="2026-11-23",
                              end="2026-11-24", max_minutes=1.0)


def test_suspension_refresh_exit_codes_lock_and_atomic_files(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS.parent / "src"))
    mod = _load("u0_pit_intervals")
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    fake = types.ModuleType("akshare")

    def down(date):
        raise ConnectionError("vendor down")
    fake.stock_tfp_em = down
    monkeypatch.setitem(sys.modules, "akshare", fake)
    out = tmp_path / "fresh"
    # every date missing and nothing fetched: non-zero, not a silent success
    assert mod.cmd_suspension(_suspension_args(tmp_path, out)) == mod.EXIT_SUSPENSION_NOTHING_FETCHED

    calls = []

    def flaky(date):
        calls.append(date)
        if date == "20261123":
            raise ConnectionError("throttled")
        return pd.DataFrame({"代码": ["600000"], "名称": ["x"], "停牌时间": ["2026-11-24"],
                             "停牌截止时间": ["2026-11-24"], "停牌原因": ["r"]})
    fake.stock_tfp_em = flaky
    assert mod.cmd_suspension(_suspension_args(tmp_path, out)) == mod.EXIT_SUSPENSION_PARTIAL
    fake.stock_tfp_em = lambda date: pd.DataFrame()
    assert mod.cmd_suspension(_suspension_args(tmp_path, out)) == 0
    assert not list(out.rglob(".*.tmp-*"))            # tmp + os.replace leaves nothing behind

    import fcntl
    holder = open(out / ".suspension.lock", "w")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert mod.cmd_suspension(_suspension_args(tmp_path, out)) == mod.EXIT_SUSPENSION_LOCKED
    finally:
        holder.close()


def test_partial_halt_refresh_counts_as_a_failed_job(monkeypatch) -> None:
    daily = _load("fresh_blind_daily")
    assert "PARTIAL" in daily.FAILED_STATUSES

    class R:
        returncode = 75
        stdout = stderr = ""
    monkeypatch.setattr(daily.subprocess, "run", lambda *a, **k: R())
    assert daily.step_suspension_refresh(False)["status"] == "PARTIAL"
    R.returncode = 3
    assert daily.step_suspension_refresh(False)["status"] == "FAILED"


def test_ledger_record_carries_freshness_and_integrity(tmp_path, monkeypatch) -> None:
    import json

    daily = _load("fresh_blind_daily")
    root = tmp_path / "root"
    panel = tmp_path / "panel.parquet"
    pd.DataFrame({"symbol": ["600000.SH"], "trade_date": [pd.Timestamp("2026-11-24")]}).to_parquet(panel)
    monkeypatch.setattr(daily, "REPO", tmp_path)
    monkeypatch.setattr(daily, "ROOT", root)
    monkeypatch.setattr(daily, "LEDGER", root / "ledger.jsonl")
    monkeypatch.setattr(daily, "KEY_PATH", root / ".key")
    monkeypatch.setattr(daily, "PANEL", panel)
    monkeypatch.setattr(daily, "step_update_data", lambda dry: {"status": "OK"})
    monkeypatch.setattr(daily, "step_suspension_refresh", lambda dry: {"status": "OK"})
    monkeypatch.setattr(daily, "step_freshness", lambda: {
        "status": "INCOMPLETE",
        "integrity": {"status": "INCOMPLETE", "problems": ["INTERIOR_HOLES_UNEXPLAINED 41"]}})
    monkeypatch.setattr(daily, "step_score", lambda: {"status": "OK"})
    monkeypatch.setattr(daily, "step_books_and_fills", lambda today: {"status": "OK", "books": {}})
    monkeypatch.setattr(sys, "argv", ["fresh_blind_daily.py"])
    assert daily.main() == 1
    record = json.loads((root / "ledger.jsonl").read_text().strip().splitlines()[-1])
    assert record["freshness_status"] == "INCOMPLETE"
    assert record["panel_integrity_status"] == "INCOMPLETE"
    assert record["panel_integrity_problems"] == ["INTERIOR_HOLES_UNEXPLAINED 41"]
    assert record["order_generation_status"] == "OK"     # orders exist, flagged by the above


def test_remediation_timeout_is_recorded_not_raised(monkeypatch) -> None:
    import subprocess

    monkeypatch.syspath_prepend(str(SCRIPTS))
    health = _load("fresh_blind_healthcheck")

    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd=a[0], timeout=k.get("timeout"))
    monkeypatch.setattr(health.subprocess, "run", slow)
    problems: list[str] = []
    assert health.run_remediation(["scripts/catchup_panel_chunked.py"], 7200, problems) is None
    assert problems == ["AUTO_REMEDIATION_TIMEOUT scripts/catchup_panel_chunked.py after 7200s"]


def test_ledger_counts_a_locked_halt_refresh_explicitly(tmp_path, monkeypatch) -> None:
    import json

    daily = _load("fresh_blind_daily")
    root = tmp_path / "root"
    panel = tmp_path / "panel.parquet"
    pd.DataFrame({"symbol": ["600000.SH"], "trade_date": [pd.Timestamp("2026-11-24")]}).to_parquet(panel)
    for name, value in (("REPO", tmp_path), ("ROOT", root), ("LEDGER", root / "l.jsonl"),
                        ("KEY_PATH", root / ".key"), ("PANEL", panel)):
        monkeypatch.setattr(daily, name, value)

    class R:
        returncode = 73
        stdout = stderr = ""
    monkeypatch.setattr(daily.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(daily, "step_update_data", lambda dry: {"status": "OK"})
    monkeypatch.setattr(daily, "step_freshness", lambda: {
        "status": "OK", "integrity": {"status": "COMPLETE", "problems": []}})
    monkeypatch.setattr(daily, "step_score", lambda: {"status": "OK"})
    monkeypatch.setattr(daily, "step_books_and_fills", lambda today: {"status": "OK", "books": {}})
    monkeypatch.setattr(sys, "argv", ["fresh_blind_daily.py"])
    daily.main()
    record = json.loads((root / "l.jsonl").read_text().strip().splitlines()[-1])
    assert record["suspension_evidence_status"] == "SKIPPED_LOCKED"
    assert record["skipped_job_count"] == 1
    assert record["skipped_jobs"] == ["suspension_evidence"]
