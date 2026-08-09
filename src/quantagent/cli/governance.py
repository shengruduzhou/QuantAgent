"""Governance inspection and evidence-issuance commands with no market-data side effects."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import typer

from quantagent.cli._utils import app, json_dump
from quantagent.config.paths import quant_paths
from quantagent.execution.live_model_trust_v2_execution_policy import (
    GOVERNED_REQUIRED_ARTIFACT_ROLES,
    issue_trace_proven_live_model_trust_v2 as issue_v2_certificate,
)
from quantagent.training.feature_contract import PRODUCTION_CONTRACT, RESEARCH_CONTRACT


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_MANIFEST = _REPO_ROOT / "configs" / "legacy_cli_manifest.json"
_QUARANTINE_CONFIG = _REPO_ROOT / "configs" / "quarantined_windows.json"
_PRODUCTION_CONFIG = _REPO_ROOT / "configs" / "production_blend.json"
_DEFAULT_LIVE_TRUST_MANIFEST = _REPO_ROOT / "configs" / "live_model_trust.json"


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _git_head() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise typer.BadParameter("a git checkout is required to issue model trust v2") from exc
    head = completed.stdout.strip().lower()
    if len(head) != 40:
        raise typer.BadParameter("could not resolve a 40-character git HEAD")
    return head


def _rooted_path(value: Path, roots: tuple[Path, ...]) -> Path:
    candidate = value if value.is_absolute() else (_REPO_ROOT / value)
    resolved = candidate.resolve(strict=False)
    for root in roots:
        canonical = root.resolve(strict=False)
        if resolved == canonical or canonical in resolved.parents:
            return resolved
    raise typer.BadParameter("path must remain inside the QuantAgent repo or runtime root")


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


@app.command("issue-live-model-trust-v2")
def issue_live_model_trust_v2_command(
    evidence_map: Path = typer.Option(
        ...,
        "--evidence-map",
        help="JSON mapping every governed artifact role, including strict_execution_trace, to {root, path}.",
    ),
    model_id: str = typer.Option(..., "--model-id"),
    manifest: Path = typer.Option(_DEFAULT_LIVE_TRUST_MANIFEST, "--manifest"),
    source_commit: str | None = typer.Option(
        None,
        "--source-commit",
        help="Must equal current git HEAD; omitted means use current HEAD.",
    ),
) -> None:
    """Issue governed schema-v2 evidence with a mandatory execution trace.

    This command performs no training, data fetching, backtest, broker access or
    gate override. It binds evidence that already exists. The governed verifier
    re-derives FRESH/statistical/strict facts and validates the hash-bound
    signal-date -> next-session execution trace before publishing. Successful
    issuance still does not arm economic live trading.
    """
    runtime_root = quant_paths().home.resolve(strict=False)
    roots = {"repo": _REPO_ROOT.resolve(strict=False), "runtime": runtime_root}
    map_path = _rooted_path(evidence_map, tuple(roots.values()))
    manifest_path = _rooted_path(manifest, tuple(roots.values()))
    payload = _read_json(map_path)
    if payload is None:
        raise typer.BadParameter("evidence map must be a JSON object")

    actual_head = _git_head()
    requested_commit = (source_commit or actual_head).strip().lower()
    if requested_commit != actual_head:
        raise typer.BadParameter(
            f"source_commit must equal current git HEAD: {requested_commit} != {actual_head}"
        )

    locations: dict[str, tuple[str, str]] = {}
    for role in GOVERNED_REQUIRED_ARTIFACT_ROLES:
        descriptor = payload.get(role)
        if not isinstance(descriptor, dict):
            raise typer.BadParameter(f"evidence map missing object for role {role}")
        root_name = str(descriptor.get("root") or "").strip()
        relative = str(descriptor.get("path") or "").strip()
        if root_name not in roots:
            raise typer.BadParameter(f"{role}.root must be one of {sorted(roots)}")
        if not relative:
            raise typer.BadParameter(f"{role}.path is required")
        locations[role] = (root_name, relative)
    unexpected = sorted(set(payload).difference(GOVERNED_REQUIRED_ARTIFACT_ROLES))
    if unexpected:
        raise typer.BadParameter(f"unexpected evidence roles: {unexpected}")

    result = issue_v2_certificate(
        manifest_path,
        model_id=model_id,
        source_commit=actual_head,
        artifact_locations=locations,
        artifact_roots=roots,
    )
    typer.echo(
        json_dump(
            {
                "status": "issued",
                "manifest_path": result.manifest_path,
                "model_id": model_id,
                "source_commit": actual_head,
                "verification_ok": result.verification.ok,
                "provenance_assurance": result.verification.evidence.get("provenance_assurance"),
                "execution_timing_assurance": result.verification.evidence.get("execution_timing_assurance"),
                "artifact_roles": list(GOVERNED_REQUIRED_ARTIFACT_ROLES),
                "live_trading_armed_by_command": False,
            }
        )
    )
