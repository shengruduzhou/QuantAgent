from __future__ import annotations

import pandas as pd


def test_policy_specialist_builds_directional_signal():
    from quantagent.agents.ashare_specialists import policy_specialist_signal

    signal = policy_specialist_signal(
        {
            "symbol": "600001.SH",
            "policy_strength": 0.8,
            "policy_direction": 1.0,
            "source_authority": 0.9,
            "expected_lag_days": 30,
            "theme": "advanced_packaging",
        }
    )

    assert signal is not None
    assert signal.agent_name == "policy_specialist"
    assert signal.signal_strength > 0.0
    assert signal.confidence > 0.7
    assert "policy" in signal.tags


def test_hot_money_specialist_uses_flow_volume_and_theme():
    from quantagent.agents.ashare_specialists import hot_money_specialist_signal

    signal = hot_money_specialist_signal(
        {
            "symbol": "300001.SZ",
            "turnover_amount": 1_000_000_000,
            "main_net_amount": 120_000_000,
            "dragon_tiger_net_buy": 40_000_000,
            "volume_ratio": 2.5,
            "theme_hot_count": 3,
        }
    )

    assert signal is not None
    assert signal.agent_name == "hot_money_specialist"
    assert signal.signal_strength > 0.0
    assert signal.horizon_days == 3


def test_lockup_specialist_emits_negative_supply_pressure():
    from quantagent.agents.ashare_specialists import lockup_specialist_signal

    signal = lockup_specialist_signal(
        {
            "symbol": "688001.SH",
            "lockup_ratio_float": 0.25,
            "announced_reduction_ratio": 0.03,
            "days_to_lockup": 20,
        }
    )

    assert signal is not None
    assert signal.signal_strength < 0.0
    assert signal.risk_penalty > 0.0
    assert "supply_pressure" in signal.tags


def test_build_specialist_signals_and_evidence_records():
    from quantagent.agents.ashare_specialists import (
        build_ashare_specialist_signals,
        specialist_evidence_records,
    )

    frame = pd.DataFrame(
        [
            {
                "symbol": "600001.SH",
                "policy_strength": 0.6,
                "policy_direction": 1,
                "source_authority": 0.8,
                "main_net_amount": 80_000_000,
                "turnover_amount": 800_000_000,
                "lockup_ratio_float": 0.0,
            }
        ]
    )
    signals = build_ashare_specialist_signals(frame)
    records = specialist_evidence_records(signals, "2026-06-14T00:00:00Z")

    assert len(signals) >= 2
    assert len(records) == len(signals)
    assert {record.source for record in records} >= {"policy_specialist", "hot_money_specialist"}
