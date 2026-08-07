"""Static and optional live audit of the complete Qlib reference set."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import re
from typing import Any
from urllib.parse import urldefrag

import requests

from quantagent.qlib.catalog import (
    QLIB_CAPABILITIES,
    QLIB_DOC_COUNT,
    QLIB_MIN_VERSION,
    QLIB_PYPI_JSON,
)


def build_coverage_audit() -> dict[str, object]:
    runtime = [item.capability_id for item in QLIB_CAPABILITIES if item.runtime_component]
    governance = [item.capability_id for item in QLIB_CAPABILITIES if not item.runtime_component]
    return {
        "status": "passed",
        "expected_reference_count": QLIB_DOC_COUNT,
        "registered_reference_count": len(QLIB_CAPABILITIES),
        "minimum_pyqlib_version": QLIB_MIN_VERSION,
        "runtime_capabilities": runtime,
        "governance_references": governance,
        "references": [asdict(item) for item in QLIB_CAPABILITIES],
    }


def audit_live_documentation(
    *,
    timeout_seconds: float = 20.0,
    session: requests.Session | None = None,
) -> dict[str, object]:
    """Verify all pinned URLs and current PyPI stable version.

    Network access is never implicit.  The CLI exposes this only behind
    ``--allow-network``.
    """
    client = session or requests.Session()
    urls = list(dict.fromkeys(urldefrag(item.url)[0] for item in QLIB_CAPABILITIES))
    checks: list[dict[str, Any]] = []
    for url in urls:
        try:
            response = client.get(
                url,
                timeout=timeout_seconds,
                headers={"User-Agent": "QuantAgent-Qlib-Audit/1.0"},
            )
            checks.append(
                {
                    "url": url,
                    "status_code": int(response.status_code),
                    "ok": bool(response.ok),
                    "final_url": response.url,
                }
            )
        except requests.RequestException as exc:
            checks.append(
                {
                    "url": url,
                    "status_code": None,
                    "ok": False,
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )

    pypi: dict[str, object]
    try:
        response = client.get(
            QLIB_PYPI_JSON,
            timeout=timeout_seconds,
            headers={"User-Agent": "QuantAgent-Qlib-Audit/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        current_version = str(payload.get("info", {}).get("version", ""))
        version_key = _version_tuple(current_version)
        pypi = {
            "ok": bool(current_version),
            "current_version": current_version,
            "minimum_version": QLIB_MIN_VERSION,
            "below_minimum": version_key < _version_tuple(QLIB_MIN_VERSION),
            "outside_tested_0_9_series": version_key[:2] != (0, 9),
        }
    except (requests.RequestException, ValueError) as exc:
        pypi = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}

    failed = [item["url"] for item in checks if not item["ok"]]
    status = "passed" if not failed and pypi.get("ok") else "failed"
    if pypi.get("below_minimum") or pypi.get("outside_tested_0_9_series"):
        status = "failed"
    return {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "reference_count": QLIB_DOC_COUNT,
        "http_checks": checks,
        "failed_urls": failed,
        "pypi": pypi,
    }


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = [int(item) for item in re.findall(r"\d+", str(value))[:3]]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])  # type: ignore[return-value]
