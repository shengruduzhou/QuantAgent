"""CLI exit-code contract for valuation evidence builds."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from quantagent.cli import app
from quantagent.data.bootstrap import valuation_bootstrap


runner = CliRunner()


def _invoke(tmp_path):
    return runner.invoke(
        app,
        [
            "build-valuation-v7",
            "--as-of-dates",
            "2026-05-15",
            "--symbols",
            "600519.SH",
            "--lake-root",
            str(tmp_path / "lake"),
        ],
    )


def test_build_valuation_cli_exits_nonzero_after_printing_blocked_diagnostic(monkeypatch, tmp_path):
    diagnostic = {
        "status": "blocked",
        "output_path": None,
        "blockers": ["valuation_local_snapshot_missing_explicit_pit_evidence"],
    }
    monkeypatch.setattr(
        valuation_bootstrap,
        "build_valuation_cache",
        lambda _config: diagnostic,
    )

    result = _invoke(tmp_path)

    assert result.exit_code == 1
    assert json.loads(result.stdout) == diagnostic


def test_build_valuation_cli_keeps_zero_exit_for_passed_payload(monkeypatch, tmp_path):
    diagnostic = {
        "status": "passed",
        "output_path": str(tmp_path / "lake" / "valuation.parquet"),
        "blockers": [],
    }
    monkeypatch.setattr(
        valuation_bootstrap,
        "build_valuation_cache",
        lambda _config: diagnostic,
    )

    result = _invoke(tmp_path)

    assert result.exit_code == 0
    assert json.loads(result.stdout) == diagnostic
