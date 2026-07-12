"""Governance inspection commands with no market-data side effects."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from quantagent.cli._utils import app, json_dump
from quantagent.training.feature_contract import PRODUCTION_CONTRACT, RESEARCH_CONTRACT


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_MANIFEST = _REPO_ROOT / "configs" / "legacy_cli_manifest.json"
_QUARANTINE_CONFIG = _REPO_ROOT / "configs" / "quarantined_windows.json"
_PRODUCTION_CONFIG = _REPO_ROOT / "configs" / "production_blend.json"


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@app.command("governance-status")
def governance_status() -> None:
    """Print machine-readable governance and CLI boundary status."""
    legacy_enabled = os.getenv("QUANTAGENT_ENABLE_LEGACY_CLI", "0").lower() in {
        "1", "true", "yes", "on"
    }
    production = _read_json(_PRODUCTION_CONFIG)
    payload = {
        "legacy_cli_enabled": legacy_enabled,
        "legacy_cli_manifest": _read_json(_LEGACY_MANIFEST),
        "quarantined_windows": _read_json(_QUARANTINE_CONFIG),
        "production_trust": (production or {}).get("trust"),
        "production_feature_contract": PRODUCTION_CONTRACT.name,
        "research_feature_contract": RESEARCH_CONTRACT.name,
        "tests_executed_by_command": False,
    }
    typer.echo(json_dump(payload))
