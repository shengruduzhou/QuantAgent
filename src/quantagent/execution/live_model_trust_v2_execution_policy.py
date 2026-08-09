"""Trace-proven governed model-trust policy.

This layer extends the schema-v2 governed evidence certificate with one extra
artifact that the historical low-level binder did not know about:
``strict_execution_trace``.  The trace is verified twice:

* byte-level SHA-256 binding protects the exact file named by the certificate;
* canonical trace hashing + structural validation proves the signal-date to
  next-session execution semantics independently of CSV serialization details.

Successful verification still does **not** grant economic-live eligibility.
That remains a separate promotion gate.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4
import os

import pandas as pd

from quantagent.backtest.execution_timing import (
    EXECUTION_TIMING_SEMANTICS,
    execution_trace_sha256,
    validate_execution_trace,
)
from quantagent.execution.live_model_trust_v2 import (
    ArtifactBinding,
    LiveModelTrustV2IssueResult,
    REQUIRED_ARTIFACT_ROLES,
    V2VerificationResult,
    sha256_file,
)
from quantagent.execution.live_model_trust_v2_policy import (
    issue_governed_live_model_trust_v2 as _issue_base_governed_v2,
    verify_governed_live_model_trust_v2 as _verify_base_governed_v2,
)


GOVERNED_EXECUTION_TRACE_ROLE = "strict_execution_trace"
GOVERNED_REQUIRED_ARTIFACT_ROLES: tuple[str, ...] = (
    *REQUIRED_ARTIFACT_ROLES,
    GOVERNED_EXECUTION_TRACE_ROLE,
)
TRACE_PROVEN_EXECUTION_ASSURANCE = (
    "trace_proven:" + EXECUTION_TIMING_SEMANTICS
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def verify_trace_proven_live_model_trust_v2(
    payload: Mapping[str, Any],
    *,
    artifact_roots: Mapping[str, str | Path],
    min_fresh_oos_days: int = 120,
    max_pbo: float = 0.25,
    min_dsr_probability: float = 0.95,
    max_spa_p_value: float = 0.05,
) -> V2VerificationResult:
    """Verify the original governed bundle plus hash-bound execution trace."""
    artifacts_raw = payload.get("artifacts")
    artifacts = dict(artifacts_raw) if isinstance(artifacts_raw, Mapping) else {}
    missing = [role for role in GOVERNED_REQUIRED_ARTIFACT_ROLES if role not in artifacts]
    unexpected = sorted(set(artifacts).difference(GOVERNED_REQUIRED_ARTIFACT_ROLES))

    # The historical governed verifier intentionally knows only the original
    # twelve roles. Feed it a filtered copy, then independently verify the trace.
    base_payload = dict(payload)
    base_payload["artifacts"] = {
        role: artifacts[role]
        for role in REQUIRED_ARTIFACT_ROLES
        if role in artifacts
    }
    base = _verify_base_governed_v2(
        base_payload,
        artifact_roots=artifact_roots,
        min_fresh_oos_days=min_fresh_oos_days,
        max_pbo=max_pbo,
        min_dsr_probability=min_dsr_probability,
        max_spa_p_value=max_spa_p_value,
    )
    reasons = list(base.reasons)
    if missing:
        reasons.append("governed_artifacts_missing:" + ",".join(missing))
    if unexpected:
        reasons.append("governed_artifacts_unexpected:" + ",".join(unexpected))

    evidence = dict(base.evidence)
    # Never allow trace certification to arm economics. The separate economic
    # promotion process remains the only writer allowed to prove eligibility.
    evidence["economic_live_eligible"] = False
    resolved = dict(base.resolved_paths)

    descriptor = artifacts.get(GOVERNED_EXECUTION_TRACE_ROLE)
    trace_path: Path | None = None
    trace_binding_sha: str | None = None
    if isinstance(descriptor, Mapping):
        root_name = str(descriptor.get("root") or "").strip()
        relative = str(descriptor.get("path") or "").strip()
        expected_sha = str(descriptor.get("sha256") or "").strip().lower()
        if not _HEX64.fullmatch(expected_sha):
            reasons.append(f"{GOVERNED_EXECUTION_TRACE_ROLE}:sha256_invalid")
        else:
            trace_path, path_error = _resolve_regular_artifact(
                root_name,
                relative,
                artifact_roots,
            )
            if trace_path is None:
                reasons.append(
                    f"{GOVERNED_EXECUTION_TRACE_ROLE}:{path_error or 'artifact_unresolvable'}"
                )
            else:
                actual_sha = sha256_file(trace_path)
                if actual_sha != expected_sha:
                    reasons.append(
                        f"{GOVERNED_EXECUTION_TRACE_ROLE}:sha256_mismatch:{actual_sha}!={expected_sha}"
                    )
                    trace_path = None
                else:
                    trace_binding_sha = expected_sha
                    resolved[GOVERNED_EXECUTION_TRACE_ROLE] = str(trace_path)
    elif GOVERVERNED_EXECUTION_TRACE_ROLE_PRESENT := (GOVERNED_EXECUTION_TRACE_ROLE in artifacts):
        # Keep a distinct diagnostic when the role exists but is not an object.
        # The assignment expression is intentionally local and deterministic.
        if GOVERVERNED_EXECUTION_TRACE_ROLE_PRESENT:
            reasons.append(f"{GOVERNED_EXECUTION_TRACE_ROLE}:descriptor_not_object")

    canonical_trace_sha: str | None = None
    timing_ok = False
    timing_reasons: tuple[str, ...] = ()
    if trace_path is not None:
        try:
            trace = pd.read_csv(trace_path)
        except Exception as exc:  # noqa: BLE001 - evidence corruption must surface
            reasons.append(
                f"{GOVERNED_EXECUTION_TRACE_ROLE}:csv_invalid:{type(exc).__name__}"
            )
        else:
            timing = validate_execution_trace(trace)
            timing_ok = timing.ok
            timing_reasons = timing.reasons
            if not timing.ok:
                reasons.extend(
                    f"{GOVERNED_EXECUTION_TRACE_ROLE}:{reason}"
                    for reason in timing.reasons
                )
            canonical_trace_sha = execution_trace_sha256(trace)
            evidence.update(
                {
                    "execution_trace_rows": int(len(trace)),
                    "execution_trace_mapped_signal_days": timing.mapped_signal_days,
                    "execution_trace_order_records": timing.order_records,
                    "execution_trace_skip_records": timing.skip_records,
                    "execution_trace_sha256": canonical_trace_sha,
                    "execution_trace_file_sha256": trace_binding_sha,
                    "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
                }
            )

    strict_summary_path = resolved.get("strict_backtest")
    if strict_summary_path and canonical_trace_sha is not None:
        try:
            strict = json.loads(Path(strict_summary_path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"strict_backtest:json_invalid_for_trace:{type(exc).__name__}")
        else:
            if not isinstance(strict, dict):
                reasons.append("strict_backtest:not_object_for_trace")
            else:
                if str(strict.get("execution_trace_sha256") or "") != canonical_trace_sha:
                    reasons.append("strict_backtest:execution_trace_sha256_mismatch")
                if str(strict.get("execution_timing_semantics") or "") != EXECUTION_TIMING_SEMANTICS:
                    reasons.append("strict_backtest:execution_timing_semantics_mismatch")

    trace_reasons = [
        reason
        for reason in reasons
        if reason.startswith(f"{GOVERNED_EXECUTION_TRACE_ROLE}:")
        or reason.startswith("strict_backtest:execution_trace_")
        or reason == "strict_backtest:execution_timing_semantics_mismatch"
        or reason.startswith("governed_artifacts_missing:")
        or reason.startswith("governed_artifacts_unexpected:")
    ]
    if timing_ok and not timing_reasons and canonical_trace_sha and not trace_reasons:
        evidence["execution_timing_assurance"] = TRACE_PROVEN_EXECUTION_ASSURANCE
    else:
        evidence["execution_timing_assurance"] = "not_trace_proven"

    unique = tuple(dict.fromkeys(reasons))
    return V2VerificationResult(
        ok=not unique,
        reasons=unique,
        evidence=evidence,
        resolved_paths=resolved,
    )


def issue_trace_proven_live_model_trust_v2(
    manifest_path: str | Path,
    *,
    model_id: str,
    source_commit: str,
    artifact_locations: Mapping[str, tuple[str, str]],
    artifact_roots: Mapping[str, str | Path] | None = None,
    issued_at: datetime | None = None,
    min_fresh_oos_days: int = 120,
    max_pbo: float = 0.25,
    min_dsr_probability: float = 0.95,
    max_spa_p_value: float = 0.05,
) -> LiveModelTrustV2IssueResult:
    """Issue the governed v2 evidence certificate with mandatory timing trace."""
    if artifact_roots is None:
        raise ValueError("trace-proven v2 issuer requires explicit artifact_roots")
    missing = [role for role in GOVERNED_REQUIRED_ARTIFACT_ROLES if role not in artifact_locations]
    unexpected = sorted(set(artifact_locations).difference(GOVERNED_REQUIRED_ARTIFACT_ROLES))
    if missing or unexpected:
        raise ValueError(f"artifact role mismatch: missing={missing}, unexpected={unexpected}")

    trace_root, trace_relative = artifact_locations[GOVERNED_EXECUTION_TRACE_ROLE]
    trace_path, path_error = _resolve_regular_artifact(
        str(trace_root),
        str(trace_relative),
        artifact_roots,
    )
    if trace_path is None:
        raise ValueError(
            f"{GOVERNED_EXECUTION_TRACE_ROLE}: {path_error or 'artifact_unresolvable'}"
        )
    trace_file_sha = sha256_file(trace_path)

    final = Path(manifest_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = final.with_name(f".{final.name}.{uuid4().hex}.trace-stage")
    base_locations = {
        role: artifact_locations[role]
        for role in REQUIRED_ARTIFACT_ROLES
    }
    try:
        staged = _issue_base_governed_v2(
            stage,
            model_id=model_id,
            source_commit=source_commit,
            artifact_locations=base_locations,
            artifact_roots=artifact_roots,
            issued_at=issued_at,
            min_fresh_oos_days=min_fresh_oos_days,
            max_pbo=max_pbo,
            min_dsr_probability=min_dsr_probability,
            max_spa_p_value=max_spa_p_value,
        )
        payload = dict(staged.payload)
        bound_artifacts = dict(payload.get("artifacts") or {})
        bound_artifacts[GOVERNED_EXECUTION_TRACE_ROLE] = ArtifactBinding(
            root=str(trace_root),
            path=_normalise_relative_path(str(trace_relative)),
            sha256=trace_file_sha,
        ).to_dict()
        payload["artifacts"] = bound_artifacts

        verification = verify_trace_proven_live_model_trust_v2(
            payload,
            artifact_roots=artifact_roots,
            min_fresh_oos_days=min_fresh_oos_days,
            max_pbo=max_pbo,
            min_dsr_probability=min_dsr_probability,
            max_spa_p_value=max_spa_p_value,
        )
        if not verification.ok:
            raise ValueError(
                "trace-proven governed v2 trust evidence rejected: "
                + "; ".join(verification.reasons)
            )

        with stage.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, final)
        return LiveModelTrustV2IssueResult(str(final), payload, verification)
    finally:
        try:
            if stage.exists() or stage.is_symlink():
                stage.unlink()
        except OSError:
            pass


def _resolve_regular_artifact(
    root_name: str,
    relative: str,
    artifact_roots: Mapping[str, str | Path],
) -> tuple[Path | None, str | None]:
    if root_name not in artifact_roots:
        return None, "unknown_root"
    raw = Path(str(relative))
    if raw.is_absolute():
        return None, "absolute_path_not_allowed"
    if not raw.parts or ".." in raw.parts:
        return None, "path_escape_not_allowed"
    root = Path(artifact_roots[root_name]).resolve(strict=False)
    current = root
    for part in raw.parts:
        current = current / part
        if current.is_symlink():
            return None, "symlink_not_allowed"
    candidate = root / raw
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except FileNotFoundError:
        return None, "not_found"
    except (OSError, ValueError):
        return None, "path_outside_root"
    if not resolved.is_file():
        return None, "not_regular_file"
    return resolved, None


def _normalise_relative_path(relative: str) -> str:
    raw = Path(str(relative))
    if raw.is_absolute() or not raw.parts or ".." in raw.parts:
        raise ValueError("artifact path must be a non-empty relative path without '..'")
    return raw.as_posix()


__all__ = [
    "GOVERNED_EXECUTION_TRACE_ROLE",
    "GOVERNED_REQUIRED_ARTIFACT_ROLES",
    "TRACE_PROVEN_EXECUTION_ASSURANCE",
    "issue_trace_proven_live_model_trust_v2",
    "verify_trace_proven_live_model_trust_v2",
]
