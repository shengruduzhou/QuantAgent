"""Authoritative, hashable A-share acceptance-calendar evidence.

This module deliberately does **not** synthesize trading sessions from generic
weekdays or market bars.  A caller must provide the exact session set obtained
from a source whose identity is preserved in the evidence.  The governed
model-trust layer binds the resulting artifact bytes and compares downstream
FRESH evidence against the exact ordered set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from uuid import uuid4


ACCEPTANCE_CALENDAR_SCHEMA_VERSION = 1
ACCEPTANCE_CALENDAR_TYPE = "quantagent.acceptance_calendar.v1"
REQUIRED_MARKET_SCOPE = "CN_A_SHARE"
REQUIRED_EXCHANGE_SCOPE = ("SSE", "SZSE")
REQUIRED_TIMEZONE = "Asia/Shanghai"
REQUIRED_FREQUENCY = "1d"
REQUIRED_SESSION_SEMANTICS = "exchange_trading_session_date_v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class AcceptanceCalendarEvidence:
    model_id: str
    source_commit: str
    acceptance_window_id: str
    window_start_date: str
    window_end_date: str
    sessions: tuple[str, ...]
    session_set_sha256: str
    source_provider: str
    source_identity: str
    source_version: str
    source_retrieved_at: str
    source_as_of: str
    source_locators: tuple[str, ...]

    @property
    def trading_days(self) -> int:
        return len(self.sessions)


def canonical_session_set_sha256(sessions: Sequence[str]) -> str:
    """Hash an ordered session set with a stable JSON representation."""
    encoded = json.dumps(
        list(sessions), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_acceptance_calendar_payload(
    *,
    model_id: str,
    source_commit: str,
    acceptance_window_id: str,
    window_start_date: str,
    window_end_date: str,
    sessions: Sequence[str],
    source_provider: str,
    source_identity: str,
    source_version: str,
    source_retrieved_at: str,
    source_as_of: str,
    source_locators: Sequence[str],
) -> dict[str, Any]:
    """Build evidence from an externally sourced exact A-share session set.

    ``sessions`` is intentionally required.  There is no helper here that turns
    weekdays into exchange sessions: exchange holidays / exceptional closures
    are source evidence, not a calendar inference problem.
    """
    payload: dict[str, Any] = {
        "schema_version": ACCEPTANCE_CALENDAR_SCHEMA_VERSION,
        "evidence_type": ACCEPTANCE_CALENDAR_TYPE,
        "model_id": str(model_id),
        "source_commit": str(source_commit).lower(),
        "acceptance_window_id": str(acceptance_window_id),
        "market_scope": REQUIRED_MARKET_SCOPE,
        "exchange_scope": list(REQUIRED_EXCHANGE_SCOPE),
        "timezone": REQUIRED_TIMEZONE,
        "frequency": REQUIRED_FREQUENCY,
        "session_semantics": REQUIRED_SESSION_SEMANTICS,
        "window_start_date": str(window_start_date),
        "window_end_date": str(window_end_date),
        "sessions": [str(value) for value in sessions],
        "source": {
            "provider": str(source_provider),
            "authoritative": True,
            "identity": str(source_identity),
            "version": str(source_version),
            "retrieved_at": str(source_retrieved_at),
            "as_of": str(source_as_of),
            "locators": [str(value) for value in source_locators],
        },
    }
    payload["session_set_sha256"] = canonical_session_set_sha256(payload["sessions"])
    validate_acceptance_calendar_payload(payload)
    return payload


def validate_acceptance_calendar_payload(
    payload: Mapping[str, Any],
    *,
    expected_model_id: str | None = None,
    expected_source_commit: str | None = None,
) -> AcceptanceCalendarEvidence:
    """Validate canonical semantics and return normalized immutable evidence."""
    if payload.get("schema_version") != ACCEPTANCE_CALENDAR_SCHEMA_VERSION:
        raise ValueError("acceptance_calendar:schema_version_mismatch")
    if payload.get("evidence_type") != ACCEPTANCE_CALENDAR_TYPE:
        raise ValueError("acceptance_calendar:evidence_type_mismatch")

    model_id = str(payload.get("model_id") or "").strip()
    source_commit = str(payload.get("source_commit") or "").strip().lower()
    window_id = str(payload.get("acceptance_window_id") or "").strip()
    if not model_id:
        raise ValueError("acceptance_calendar:model_id_missing")
    if expected_model_id is not None and model_id != str(expected_model_id):
        raise ValueError("acceptance_calendar:model_id_mismatch")
    if not _HEX40.fullmatch(source_commit):
        raise ValueError("acceptance_calendar:source_commit_invalid")
    if expected_source_commit is not None and source_commit != str(expected_source_commit).lower():
        raise ValueError("acceptance_calendar:source_commit_mismatch")
    if not window_id:
        raise ValueError("acceptance_calendar:acceptance_window_id_missing")

    if payload.get("market_scope") != REQUIRED_MARKET_SCOPE:
        raise ValueError("acceptance_calendar:unsupported_market_scope")
    exchange_scope = payload.get("exchange_scope")
    if not isinstance(exchange_scope, list) or tuple(exchange_scope) != REQUIRED_EXCHANGE_SCOPE:
        raise ValueError("acceptance_calendar:exchange_scope_mismatch")
    if payload.get("timezone") != REQUIRED_TIMEZONE:
        raise ValueError("acceptance_calendar:timezone_mismatch")
    if payload.get("frequency") != REQUIRED_FREQUENCY:
        raise ValueError("acceptance_calendar:frequency_mismatch")
    if payload.get("session_semantics") != REQUIRED_SESSION_SEMANTICS:
        raise ValueError("acceptance_calendar:session_semantics_mismatch")

    start = _strict_iso_date(payload.get("window_start_date"), "window_start_date")
    end = _strict_iso_date(payload.get("window_end_date"), "window_end_date")
    if start > end:
        raise ValueError("acceptance_calendar:window_reversed")

    raw_sessions = payload.get("sessions")
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError("acceptance_calendar:sessions_invalid")
    sessions = tuple(
        _strict_iso_date(value, "session") for value in raw_sessions
    )
    if len(set(sessions)) != len(sessions):
        raise ValueError("acceptance_calendar:duplicate_session")
    if tuple(sorted(sessions)) != sessions:
        raise ValueError("acceptance_calendar:sessions_not_strictly_ordered")
    for session in sessions:
        parsed = datetime.strptime(session, "%Y-%m-%d").date()
        if parsed.weekday() >= 5:
            raise ValueError("acceptance_calendar:weekend_session_invalid")
        if session < start or session > end:
            raise ValueError("acceptance_calendar:session_outside_window")
    if sessions[0] != start or sessions[-1] != end:
        raise ValueError("acceptance_calendar:window_boundary_mismatch")

    stated_session_sha = str(payload.get("session_set_sha256") or "").lower()
    actual_session_sha = canonical_session_set_sha256(sessions)
    if not _HEX64.fullmatch(stated_session_sha):
        raise ValueError("acceptance_calendar:session_set_sha256_invalid")
    if stated_session_sha != actual_session_sha:
        raise ValueError("acceptance_calendar:session_set_sha256_mismatch")

    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("acceptance_calendar:source_missing")
    provider = str(source.get("provider") or "").strip()
    identity = str(source.get("identity") or "").strip()
    version = str(source.get("version") or "").strip()
    retrieved_at = str(source.get("retrieved_at") or "").strip()
    source_as_of = str(source.get("as_of") or "").strip()
    locators = source.get("locators")
    if source.get("authoritative") is not True:
        raise ValueError("acceptance_calendar:source_not_authoritative")
    if not provider:
        raise ValueError("acceptance_calendar:source_provider_missing")
    if not identity:
        raise ValueError("acceptance_calendar:source_identity_missing")
    if not version:
        raise ValueError("acceptance_calendar:source_version_missing")
    _aware_timestamp(retrieved_at, "source_retrieved_at")
    _aware_timestamp(source_as_of, "source_as_of")
    if not isinstance(locators, list) or not locators:
        raise ValueError("acceptance_calendar:source_locators_missing")
    normalized_locators = tuple(str(value).strip() for value in locators)
    if not all(normalized_locators) or len(set(normalized_locators)) != len(normalized_locators):
        raise ValueError("acceptance_calendar:source_locators_invalid")

    return AcceptanceCalendarEvidence(
        model_id=model_id,
        source_commit=source_commit,
        acceptance_window_id=window_id,
        window_start_date=start,
        window_end_date=end,
        sessions=sessions,
        session_set_sha256=actual_session_sha,
        source_provider=provider,
        source_identity=identity,
        source_version=version,
        source_retrieved_at=retrieved_at,
        source_as_of=source_as_of,
        source_locators=normalized_locators,
    )


def load_acceptance_calendar(
    path: str | Path,
    *,
    expected_model_id: str | None = None,
    expected_source_commit: str | None = None,
) -> AcceptanceCalendarEvidence:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"acceptance_calendar:json_invalid:{type(exc).__name__}") from exc
    if not isinstance(raw, dict):
        raise ValueError("acceptance_calendar:json_not_object")
    return validate_acceptance_calendar_payload(
        raw,
        expected_model_id=expected_model_id,
        expected_source_commit=expected_source_commit,
    )


def write_acceptance_calendar(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Validate then atomically write canonical JSON evidence."""
    validate_acceptance_calendar_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        try:
            if tmp.exists() or tmp.is_symlink():
                tmp.unlink()
        except OSError:
            pass
    return target


def _strict_iso_date(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not _ISO_DATE.fullmatch(text):
        raise ValueError(f"acceptance_calendar:{label}_invalid")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"acceptance_calendar:{label}_invalid") from exc
    if parsed.isoformat() != text:
        raise ValueError(f"acceptance_calendar:{label}_invalid")
    return text


def _aware_timestamp(value: object, label: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"acceptance_calendar:{label}_missing")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"acceptance_calendar:{label}_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"acceptance_calendar:{label}_timezone_missing")
    return parsed


__all__ = [
    "ACCEPTANCE_CALENDAR_SCHEMA_VERSION",
    "ACCEPTANCE_CALENDAR_TYPE",
    "AcceptanceCalendarEvidence",
    "REQUIRED_EXCHANGE_SCOPE",
    "REQUIRED_FREQUENCY",
    "REQUIRED_MARKET_SCOPE",
    "REQUIRED_SESSION_SEMANTICS",
    "REQUIRED_TIMEZONE",
    "build_acceptance_calendar_payload",
    "canonical_session_set_sha256",
    "load_acceptance_calendar",
    "validate_acceptance_calendar_payload",
    "write_acceptance_calendar",
]
