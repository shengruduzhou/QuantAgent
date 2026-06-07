from __future__ import annotations

import json

from quantagent.research.forward_report import (
    build_forward_research_contract,
    render_forward_research_header,
    validate_forward_research_payload,
)


def test_weekly_contract_predicts_future_window_not_current_week():
    contract = build_forward_research_contract("2026-06-07", cadence="weekly")

    assert contract.window.as_of == "2026-06-07"
    assert contract.window.prediction_start == "2026-06-08"
    assert contract.window.prediction_end == "2026-06-14"
    assert "PIT cutoff" in render_forward_research_header(contract)


def test_monthly_contract_predicts_next_month():
    contract = build_forward_research_contract("2026-06-07", cadence="monthly")

    assert contract.window.prediction_start == "2026-07-01"
    assert contract.window.prediction_end == "2026-07-31"
    assert contract.min_events >= 10
    assert contract.min_candidate_stocks >= 40


def test_forward_payload_validation_flags_thin_reports():
    contract = build_forward_research_contract("2026-06-07", cadence="monthly")
    payload = {
        "market_outlook": "risk-on but selective",
        "event_calendar": [{"event": "PMI", "benefit": "cyclicals"}],
        "themes": [{"theme": "AI hardware"}],
    }

    validation = validate_forward_research_payload(payload, contract, stock_count=5)

    assert not validation.passed
    assert any("event_calendar too thin" in warning for warning in validation.warnings)
    assert any("candidate stock pool too small" in warning for warning in validation.warnings)


def test_contract_writes_json(tmp_path):
    contract = build_forward_research_contract("2026-06-07", cadence="weekly")
    path = contract.write(tmp_path / "contract.json")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["window"]["cadence"] == "weekly"
    assert data["required_sections"]
