"""A latched kill switch has a deliberate, audited human clear path (round-29 R10-F08)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from quantagent.cli import app
from quantagent.paper import ledger as paper_ledger
from quantagent.paper.risk import SCOPE_PORTFOLIO, RiskEngine


def _latched_ledger(tmp_path):
    path = tmp_path / "operational.jsonl"
    engine = RiskEngine(state_ledger=paper_ledger.EventLedger(path))
    engine.kill_switch.trigger(SCOPE_PORTFOLIO, "drawdown 21% exceeds 20%")
    return path


def test_clear_requires_confirmation_author_and_reason(tmp_path):
    path = _latched_ledger(tmp_path)
    runner = CliRunner()
    refused = runner.invoke(app, ["paper-clear-kill-switch", "--ledger", str(path),
                                  "--author", "pm", "--reason", "reviewed drawdown"])
    assert refused.exit_code != 0
    short = runner.invoke(app, ["paper-clear-kill-switch", "--ledger", str(path), "--confirm",
                                "--author", "pm", "--reason", "ok"])
    assert short.exit_code != 0
    assert RiskEngine(state_ledger=paper_ledger.EventLedger(path)).kill_switch.is_triggered(SCOPE_PORTFOLIO)


def test_clear_is_durable_and_attributed(tmp_path):
    path = _latched_ledger(tmp_path)
    result = CliRunner().invoke(app, ["paper-clear-kill-switch", "--ledger", str(path), "--confirm",
                                      "--author", "pm-zhang", "--reason", "reviewed drawdown, resuming"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["cleared"] is True
    replayed = RiskEngine(state_ledger=paper_ledger.EventLedger(path))
    assert not replayed.kill_switch.is_triggered(SCOPE_PORTFOLIO)
    cleared = [e for e in paper_ledger.EventLedger(path).read() if e.event_type == paper_ledger.KILL_SWITCH_CLEARED]
    assert cleared and cleared[-1].payload["author"] == "pm-zhang"
