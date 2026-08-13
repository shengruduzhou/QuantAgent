from __future__ import annotations

from contextlib import contextmanager
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

    captured: dict[str, object] = {"lock_held": False}

    @contextmanager
    def fake_account_lock(canonical_ledger_path, **_kwargs):
        captured["lock_path"] = str(canonical_ledger_path)
        captured["lock_held"] = True
        try:
            yield Path(canonical_ledger_path)
        finally:
            captured["lock_held"] = False

    def fake_execute(as_of_date, frame, *, config, authoritative_sessions):
        captured["date"] = as_of_date
        captured["rows"] = len(frame)
        captured["config"] = config
        captured["authoritative_sessions"] = authoritative_sessions
        captured["execute_lock_held"] = captured["lock_held"]
        return [_Result()]

    import quantagent.paper.account_lock as account_lock
    import quantagent.paper.continuous_execution as continuous_execution

    monkeypatch.setattr(account_lock, "paper_account_lock", fake_account_lock)
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
    assert captured["lock_path"] == str(paths.canonical_ledger)
    assert captured["execute_lock_held"] is True
    assert captured["lock_held"] is False
    assert config.pending_signal_dir == str(paths.pending_signals)
    assert config.execution_journal_path == str(paths.execution_journal)
    assert config.canonical_ledger_path == str(paths.canonical_ledger)
    assert config.operational_ledger_path == str(paths.operational_ledger)
    assert config.idempotency_path == str(paths.idempotency)
    assert config.account_identity_path == str(paths.account_identity)
    assert config.portfolio_id == "v7-paper"
    assert config.initial_cash == 1_000_000.0

    output = json.loads(capsys.readouterr().out)
    assert output["runtime"]["execution_journal"] == str(paths.execution_journal)
    assert output["runtime"]["account_identity"] == str(paths.account_identity)
    assert output["paperAccount"] == {
        "portfolioId": "v7-paper",
        "initialCash": 1_000_000.0,
        "identityPath": str(paths.account_identity),
    }
    assert output["calendarAssurance"] == "observed_market_panel_only"
    assert output["shadowAcceptanceCalendarEligible"] is False
    assert output["results"] == [{"status": "execution_observed"}]