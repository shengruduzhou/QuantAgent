"""Quarantined-window guard for trusted evaluations.

The burned final holdout (see HOLDOUT_CONTAMINATION_AUDIT.md) must not be
consumed by trusted evaluators by accident. This module is dependency-light
so any script can import it: windows come from configs/quarantined_windows.json;
if that file is missing the built-in default below still protects the burned
window (fail-safe, never fail-open).

Overrides are forensic-only: callers that pass a justification must stamp
their outputs ``trust_class = contaminated_holdout_forensics`` and every
access is appended to an audit log.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "quarantined_windows.json"
FORENSICS_TRUST_CLASS = "contaminated_holdout_forensics"

# Fail-safe fallback: protects the burned holdout even if the config file is
# deleted or unreadable. Keep in sync with configs/quarantined_windows.json.
_BUILTIN_WINDOWS = [
    {
        "start": "2025-09-01",
        "end": "2026-05-18",
        "reason": "burned final holdout (builtin fallback; config file unavailable)",
        "evidence": "HOLDOUT_CONTAMINATION_AUDIT.md",
    }
]
_BUILTIN_LOG_PATH = "runtime/state/holdout_access_log.jsonl"


@dataclass(frozen=True)
class QuarantineWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    reason: str
    evidence: str


class QuarantineViolation(RuntimeError):
    """Raised when a trusted evaluation window intersects a quarantined window."""

    def __init__(self, message: str, window: QuarantineWindow):
        super().__init__(message)
        self.window = window


def _parse(entries: list[dict]) -> list[QuarantineWindow]:
    return [
        QuarantineWindow(
            start=pd.Timestamp(e["start"]),
            end=pd.Timestamp(e["end"]),
            reason=str(e.get("reason", "")),
            evidence=str(e.get("evidence", "")),
        )
        for e in entries
    ]


def load_windows(config_path: str | Path | None = None) -> tuple[list[QuarantineWindow], str]:
    """Return (windows, log_path). Falls back to builtin windows if unreadable."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _parse(payload["windows"]), str(payload.get("log_path", _BUILTIN_LOG_PATH))
    except Exception:  # noqa: BLE001 — fail-safe, never fail-open
        print(
            f"[quarantine] WARNING: could not read {path}; using builtin quarantine windows",
            file=sys.stderr,
        )
        return _parse(_BUILTIN_WINDOWS), _BUILTIN_LOG_PATH


def check_window(
    start: object,
    end: object | None,
    windows: list[QuarantineWindow] | None = None,
) -> QuarantineWindow | None:
    """Return the first quarantined window intersecting [start, end] (end=None ⇒ open)."""
    if windows is None:
        windows, _ = load_windows()
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) if end is not None else None
    for w in windows:
        if s <= w.end and (e is None or e >= w.start):
            return w
    return None


def violation_message(requested_start: object, requested_end: object | None, w: QuarantineWindow) -> str:
    end_str = str(pd.Timestamp(requested_end).date()) if requested_end is not None else "open-ended"
    return (
        f"QUARANTINE VIOLATION: requested window {pd.Timestamp(requested_start).date()}..{end_str} "
        f"intersects quarantined holdout {w.start.date()}..{w.end.date()}\n"
        f"  reason  : {w.reason}\n"
        f"  evidence: {w.evidence}\n"
        "This window must not be used for evaluation or selection.\n"
        'To proceed for forensics only, pass:  --allow-quarantined "<justification>"\n'
        "(access is logged; outputs are stamped trust_class=contaminated_holdout_forensics)"
    )


def clamp_panel_window(
    panel_start: pd.Timestamp,
    panel_end: pd.Timestamp | None,
    windows: list[QuarantineWindow] | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    """Clamp a *buffered* panel read range so buffers never spill into quarantine.

    Only called when the evaluation window itself is clean; the +/-10d fill
    buffers may still cross a boundary (e.g. delay-1 fills executing on
    2025-09-01 for signals dated 2025-08-31 — empirically confirmed in the
    Phase 2.5 replay).
    """
    if windows is None:
        windows, _ = load_windows()
    ps, pe = panel_start, panel_end
    for w in windows:
        if pe is not None and ps < w.start <= pe:
            pe = w.start - pd.Timedelta(days=1)
        if w.start <= ps <= w.end:
            ps = w.end + pd.Timedelta(days=1)
    return ps, pe


def _git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT, timeout=5
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def log_access(
    window: QuarantineWindow,
    reason: str,
    requested_start: object,
    requested_end: object | None,
    log_path: str | Path | None = None,
) -> dict:
    """Append a forensic-access record; returns the record for output stamping."""
    if log_path is None:
        _, log_path = load_windows()
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": _git_hash(),
        "argv": sys.argv,
        "requested_window": [str(requested_start), str(requested_end)],
        "quarantine_hit": [str(window.start.date()), str(window.end.date())],
        "reason": reason,
        "trust_class": FORENSICS_TRUST_CLASS,
    }
    path = _REPO_ROOT / log_path if not Path(log_path).is_absolute() else Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


__all__ = [
    "QuarantineWindow",
    "QuarantineViolation",
    "FORENSICS_TRUST_CLASS",
    "load_windows",
    "check_window",
    "violation_message",
    "clamp_panel_window",
    "log_access",
]
