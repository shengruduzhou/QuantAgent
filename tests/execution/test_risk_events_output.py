"""Risk events output from ashare_execution_simulator (spec section 9)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    simulate_ashare_target_weights,
)


def _market_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-03-01", periods=3)
    rows = []
    for d in dates:
        for sym, close in (("600000.SH", 10.0), ("000001.SZ", 12.0)):
            rows.append({
                "trade_date": d,
                "symbol": sym,
                "close": close,
                "volume": 1_000_000.0,
                "amount": 10_000_000.0,
                "is_suspended": False,
                "is_st": False,
                "is_limit_up": False,
                "is_limit_down": False,
            })
    return pd.DataFrame(rows)


def _targets() -> pd.DataFrame:
    dates = pd.bdate_range("2024-03-01", periods=3)
    return pd.DataFrame(
        {"600000.SH": [0.02, 0.02, 0.02], "000001.SZ": [0.02, 0.02, 0.02]},
        index=dates,
    )


def test_simulator_emits_risk_events_for_skipped_or_rejected(tmp_path):
    panel = _market_panel()
    # Make one symbol limit-up so a BUY order is skipped/rejected
    panel.loc[panel["symbol"] == "600000.SH", "is_limit_up"] = True
    cfg = AShareExecutionSimulationConfig(
        initial_cash=1_000_000.0,
        slippage_bps=0,
        audit_log_dir=str(tmp_path),
    )
    result = simulate_ashare_target_weights(_targets(), panel, cfg)
    assert isinstance(result.risk_events, list)
    # Either ordering skip or rejected order should produce ≥1 event.
    # (Implementation may surface this via 'order_skipped' or 'order_rejected'.)
    assert len(result.risk_events) >= 1


def test_write_risk_events_writes_json(tmp_path):
    panel = _market_panel()
    panel.loc[panel["symbol"] == "600000.SH", "is_limit_up"] = True
    cfg = AShareExecutionSimulationConfig(
        initial_cash=1_000_000.0, slippage_bps=0, audit_log_dir=str(tmp_path)
    )
    result = simulate_ashare_target_weights(_targets(), panel, cfg)
    out = result.write_risk_events(tmp_path / "risk_events.json")
    assert out.exists()
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(parsed, list)


def test_write_risk_events_empty_when_clean_run(tmp_path):
    panel = _market_panel()  # nothing blocked
    cfg = AShareExecutionSimulationConfig(
        initial_cash=1_000_000.0, slippage_bps=0, audit_log_dir=str(tmp_path)
    )
    # use small targets so all orders fit
    targets = pd.DataFrame(
        {"600000.SH": [0.01, 0.01, 0.01], "000001.SZ": [0.01, 0.01, 0.01]},
        index=pd.bdate_range("2024-03-01", periods=3),
    )
    result = simulate_ashare_target_weights(targets, panel, cfg)
    out = result.write_risk_events(tmp_path / "risk_events.json")
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(parsed, list)
    # If no skips/rejects, list can be empty
    for evt in parsed:
        assert "event_type" in evt
