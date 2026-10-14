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


def test_clear_is_refused_while_the_paper_loop_holds_the_account(tmp_path):
    """Round-30: the loop holds the canonical-ledger account lock, not writer.lock."""
    from quantagent.paper.account_lock import paper_account_lock

    path = _latched_ledger(tmp_path)
    canonical = tmp_path / "canonical_ledger.jsonl"
    canonical.write_text("", encoding="utf-8")
    args = ["paper-clear-kill-switch", "--ledger", str(path), "--confirm",
            "--author", "pm-zhang", "--reason", "reviewed drawdown, resuming"]
    with paper_account_lock(canonical):
        refused = CliRunner().invoke(app, args)
    assert refused.exit_code != 0
    assert "in use by a running paper loop" in refused.output
    assert RiskEngine(state_ledger=paper_ledger.EventLedger(path)).kill_switch.is_triggered(SCOPE_PORTFOLIO)

    explicit = tmp_path / "elsewhere" / "canonical.jsonl"
    explicit.parent.mkdir()
    explicit.write_text("", encoding="utf-8")
    with paper_account_lock(explicit):
        refused = CliRunner().invoke(app, [*args, "--canonical-ledger", str(explicit)])
    assert refused.exit_code != 0

    cleared = CliRunner().invoke(app, args)
    assert cleared.exit_code == 0, cleared.output
