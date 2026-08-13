from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from quantagent.execution.acceptance_calendar import (
    ACCEPTANCE_CALENDAR_TYPE,
    REQUIRED_EXCHANGE_SCOPE,
    build_acceptance_calendar_payload,
    canonical_session_set_sha256,
    load_acceptance_calendar,
    validate_acceptance_calendar_payload,
    write_acceptance_calendar,
)


MODEL_ID = "fresh-calendar-v1"
SOURCE_COMMIT = "a" * 40
SESSIONS = [
    "2026-02-24",
    "2026-02-25",
    "2026-02-26",
    "2026-02-27",
    "2026-03-02",
]


def _payload():
    return build_acceptance_calendar_payload(
        model_id=MODEL_ID,
        source_commit=SOURCE_COMMIT,
        acceptance_window_id="fresh-2026-02-24_2026-03-02",
        window_start_date=SESSIONS[0],
        window_end_date=SESSIONS[-1],
        sessions=SESSIONS,
        source_provider="SSE_SZSE_OFFICIAL_ARCHIVE",
        source_identity="sse-szse-2026-session-calendar",
        source_version="2026.rules+holiday-notices.v1",
        source_retrieved_at="2026-08-13T08:00:00+00:00",
        source_as_of="2026-08-13T08:00:00+00:00",
        source_locators=[
            "SSE:2026-trading-rules",
            "SSE:2026-holiday-arrangement",
            "SZSE:2026-holiday-arrangement",
        ],
    )


def test_valid_calendar_is_exact_ordered_and_hashable(tmp_path) -> None:
    payload = _payload()
    evidence = validate_acceptance_calendar_payload(
        payload,
        expected_model_id=MODEL_ID,
        expected_source_commit=SOURCE_COMMIT,
    )
    assert payload["evidence_type"] == ACCEPTANCE_CALENDAR_TYPE
    assert tuple(payload["exchange_scope"]) == REQUIRED_EXCHANGE_SCOPE
    assert evidence.sessions == tuple(SESSIONS)
    assert evidence.trading_days == len(SESSIONS)
    assert evidence.session_set_sha256 == canonical_session_set_sha256(SESSIONS)

    path = write_acceptance_calendar(tmp_path / "acceptance_calendar.json", payload)
    loaded = load_acceptance_calendar(
        path,
        expected_model_id=MODEL_ID,
        expected_source_commit=SOURCE_COMMIT,
    )
    assert loaded == evidence
    disk = json.loads(path.read_text(encoding="utf-8"))
    assert disk["session_set_sha256"] == evidence.session_set_sha256


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda p: p.__setitem__("market_scope", "US_EQUITY"), "unsupported_market_scope"),
        (lambda p: p.__setitem__("exchange_scope", ["SSE"]), "exchange_scope_mismatch"),
        (lambda p: p.__setitem__("timezone", "UTC"), "timezone_mismatch"),
        (lambda p: p.__setitem__("frequency", "weekday"), "frequency_mismatch"),
        (lambda p: p["source"].__setitem__("authoritative", False), "source_not_authoritative"),
        (lambda p: p["source"].__setitem__("provider", ""), "source_provider_missing"),
        (lambda p: p["source"].__setitem__("locators", []), "source_locators_missing"),
    ],
)
def test_market_and_source_metadata_fail_closed(mutate, reason) -> None:
    payload = _payload()
    mutate(payload)
    with pytest.raises(ValueError, match=reason):
        validate_acceptance_calendar_payload(payload)


def test_duplicate_unordered_weekend_and_out_of_window_sessions_fail() -> None:
    payload = _payload()
    payload["sessions"] = [SESSIONS[0], SESSIONS[0], *SESSIONS[2:]]
    payload["session_set_sha256"] = canonical_session_set_sha256(payload["sessions"])
    with pytest.raises(ValueError, match="duplicate_session"):
        validate_acceptance_calendar_payload(payload)

    payload = _payload()
    payload["sessions"][1], payload["sessions"][2] = payload["sessions"][2], payload["sessions"][1]
    payload["session_set_sha256"] = canonical_session_set_sha256(payload["sessions"])
    with pytest.raises(ValueError, match="sessions_not_strictly_ordered"):
        validate_acceptance_calendar_payload(payload)

    payload = _payload()
    payload["sessions"] = ["2026-02-21", *payload["sessions"]]
    payload["window_start_date"] = "2026-02-21"
    payload["session_set_sha256"] = canonical_session_set_sha256(payload["sessions"])
    with pytest.raises(ValueError, match="weekend_session_invalid"):
        validate_acceptance_calendar_payload(payload)

    payload = _payload()
    payload["window_start_date"] = "2026-02-23"
    with pytest.raises(ValueError, match="window_boundary_mismatch"):
        validate_acceptance_calendar_payload(payload)


def test_repacked_or_interior_session_change_cannot_keep_original_digest() -> None:
    payload = _payload()
    original_digest = payload["session_set_sha256"]
    # Same count and same boundary dates, but a different interior date.
    payload["sessions"][2] = "2026-03-01"
    assert payload["session_set_sha256"] == original_digest
    with pytest.raises(ValueError, match="session_set_sha256_mismatch"):
        validate_acceptance_calendar_payload(payload)


def test_naive_source_timestamps_and_identity_mismatch_fail_closed() -> None:
    payload = _payload()
    payload["source"]["retrieved_at"] = "2026-08-13T08:00:00"
    with pytest.raises(ValueError, match="source_retrieved_at_timezone_missing"):
        validate_acceptance_calendar_payload(payload)

    payload = _payload()
    with pytest.raises(ValueError, match="model_id_mismatch"):
        validate_acceptance_calendar_payload(payload, expected_model_id="other-model")
    with pytest.raises(ValueError, match="source_commit_mismatch"):
        validate_acceptance_calendar_payload(payload, expected_source_commit="b" * 40)
