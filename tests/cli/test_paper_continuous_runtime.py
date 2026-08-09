from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantagent.cli.paper import paper_execute_session
from quantagent.paper.runtime_paths import paper_runtime_paths


class _Result:
    def to_dict(self) -> dict[str, object]:
        return {"status": "execution_observed"}


def test_execute_session_uses_the_same_canonical_runtime_as_api_and_ui(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    home = tmp_path / "quant-home"
    market = tmp_path / "market.csv"
    pd.DataFrame(
        {
            "trade_date": ["2026-08-10"],
            "symbol": ["600000.SH"],
            "close": [10.0],
            "volume": [1_000_000.0],
            "amount": [10_000_000.0],
        }
    ).to_csv(market, index=False)
    monkeypatch.setenv("QUANTAGENT_HOME", str(home))

    captured: dict[str, object] = {}

    def fake_execute(as_of_date, frame, *, config, authoritative_sessions):
        captured["date"] = as_of_date
        captured["rows"] = len(frame)
        captured["config"] = config
        captured["authoritative_sessions"] = authoritative_sessions
        return [_Result()]

    import quantagent.paper.continuous_execution as continuous_execution

    monkeypatch.setattr(
        continuous_execution,
        "execute_pending_for_session",
        fake_execute,
    )

    paper_execute_session(
        date="2026-08-10",
        market_panel=market,
        initial_cash=1_000_000.0,
        portfolio_id="v7-paper",
        execution_clock="14:59:00+08:00",
        max_participation_rate=0.05,
        min_order_value_yuan=100.0,
    )

    paths = paper_runtime_paths()
    config = captured["config"]
    assert captured["date"] == "2026-08-10"
    assert captured["rows"] == 1
    assert captured["authoritative_sessions"] is None
    assert config.pending_signal_dir == str(paths.pending_signals)
    assert config.execution_journal_path == str(paths.execution_journal)
    assert config.canonical_ledger_path == str(paths.canonical_ledger)
    assert config.operational_ledger_path == str(paths.operational_ledger)
    assert config.idempotency_path == str(paths.idempotency)

    output = json.loads(capsys.readouterr().out)
    assert output["runtime"]["execution_journal"] == str(paths.execution_journal)
    assert output["calendarAssurance"] == "observed_market_panel_only"
    assert output["shadowAcceptanceCalendarEligible"] is False
    assert output["results"] == [{"status": "execution_observed"}]
