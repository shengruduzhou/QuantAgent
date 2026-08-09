"""Trace-complete governed model-trust policy.

The historical schema-v2 binder knows twelve core evidence roles. Economic
execution timing additionally requires two inseparable strict artifacts:

* ``strict_target_weights`` — the exact signal-dated target schedule supplied to
  the strict simulator;
* ``strict_execution_trace`` — the simulator's signal->execution mapping and
  order/skip/error trace.

The pair prevents a trace from proving only a selectively retained subset of
signals. New governed issuance requires all fourteen roles. Historical intact
12-role certificates remain readable as evidence but never receive trace-proven
timing assurance.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

from quantagent.backtest.execution_timing import (
    EXECUTION_TIMING_SEMANTICS,
    canonical_signal_dates,
    execution_trace_sha256,
    signal_schedule_sha256,
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


GOVERNED_TARGET_WEIGHTS_ROLE = "strict_target_weights"
GOVERNED_EXECUTION_TRACE_ROLE = "strict_execution_trace"
GOVERNED_EXECUTION_ARTIFACT_ROLES: tuple[str, ...] = (
    GOVERNED_TARGET_WEIGHTS_ROLE,
    GOVERNED_EXECUTION_TRACE_ROLE,
)
GOVERNED_REQUIRED_ARTIFACT_ROLES: tuple[str, ...] = (
    *REQUIRED_ARTIFACT_ROLES,
    *GOVERNED_EXECUTION_ARTIFACT_ROLES,
)
TRACE_PROVEN_EXECUTION_ASSURANCE = "trace_proven:" + EXECUTION_TIMING_SEMANTICS
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
    """Verify base governed evidence plus the complete strict timing pair."""
    artifacts_raw = payload.get("artifacts")
    artifacts = dict(artifacts_raw) if isinstance(artifacts_raw, Mapping) else {}
    missing_base = [role for role in REQUIRED_ARTIFACT_ROLES if role not in artifacts]
    target_present = GOVERNED_TARGET_WEIGHTS_ROLE in artifacts
    trace_present = GOVERNED_EXECUTION_TRACE_ROLE in artifacts
    supplemental_present = target_present or trace_present
    unexpected = sorted(set(artifacts).difference(GOVERNED_REQUIRED_ARTIFACT_ROLES))

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
    if missing_base:
        reasons.append("governed_artifacts_missing:" + ",".join(missing_base))
    if unexpected:
        reasons.append("governed_artifacts_unexpected:" + ",".join(unexpected))
    if supplemental_present and not (target_present and trace_present):
        missing_pair = [
            role for role in GOVERNED_EXECUTION_ARTIFACT_ROLES if role not in artifacts
        ]
        reasons.append(
            "governed_execution_artifacts_incomplete:missing=" + ",".join(missing_pair)
        )

    evidence = dict(base.evidence)
    evidence["economic_live_eligible"] = False
    resolved = dict(base.resolved_paths)

    target_path, target_file_sha = _verify_descriptor(
        GOVERNED_TARGET_WEIGHTS_ROLE,
        artifacts.get(GOVERNED_TARGET_WEIGHTS_ROLE),
        artifact_roots,
        reasons,
        resolved,
    ) if target_present else (None, None)
    trace_path, trace_file_sha = _verify_descriptor(
        GOVERNED_EXECUTION_TRACE_ROLE,
        artifacts.get(GOVERNED_EXECUTION_TRACE_ROLE),
        artifact_roots,
        reasons,
        resolved,
    ) if trace_present else (None, None)

    target_schedule_sha: str | None = None
    target_signal_dates: tuple[pd.Timestamp, ...] | None = None
    if target_path is not None:
        try:
            target_frame = pd.read_csv(target_path)
        except Exception as exc:  # noqa: BLE001
            reasons.append(
                f"{GOVERNED_TARGET_WEIGHTS_ROLE}:csv_invalid:{type(exc).__name__}"
            )
        else:
            if "signal_date" not in target_frame.columns:
                reasons.append(f"{GOVERNED_TARGET_WEIGHTS_ROLE}:signal_date_missing")
            else:
                try:
                    target_signal_dates = canonical_signal_dates(target_frame["signal_date"])
                except ValueError as exc:
                    reasons.append(f"{GOVERNED_TARGET_WEIGHTS_ROLE}:{exc}")
                else:
                    target_schedule_sha = signal_schedule_sha256(target_signal_dates)
                    evidence.update(
                        {
                            "strict_target_rows": int(len(target_frame)),
                            "strict_target_signal_days": len(target_signal_dates),
                            "strict_target_signal_schedule_sha256": target_schedule_sha,
                            "strict_target_weights_file_sha256": target_file_sha,
                        }
                    )

    canonical_trace_sha: str | None = None
    trace_signal_dates: tuple[pd.Timestamp, ...] | None = None
    timing_ok = False
    timing_reasons: tuple[str, ...] = ()
    if trace_path is not None:
        try:
            trace = pd.read_csv(trace_path)
        except Exception as exc:  # noqa: BLE001
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
            schedules = trace.loc[trace["record_type"].eq("session_mapping"), "signal_date"]
            try:
                trace_signal_dates = canonical_signal_dates(schedules)
            except ValueError as exc:
                reasons.append(f"{GOVERNED_EXECUTION_TRACE_ROLE}:signal_schedule:{exc}")
            canonical_trace_sha = execution_trace_sha256(trace)
            evidence.update(
                {
                    "execution_trace_rows": int(len(trace)),
                    "execution_trace_mapped_signal_days": timing.mapped_signal_days,
                    "execution_trace_terminal_censored_signal_days": timing.terminal_censored_signal_days,
                    "execution_trace_order_records": timing.order_records,
                    "execution_trace_skip_records": timing.skip_records,
                    "execution_trace_sha256": canonical_trace_sha,
                    "execution_trace_file_sha256": trace_file_sha,
                    "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
                }
            )

    if target_signal_dates is not None and trace_signal_dates is not None:
        if target_signal_dates != trace_signal_dates:
            reasons.append("strict_execution_trace:signal_schedule_not_equal_strict_targets")
        elif target_schedule_sha != signal_schedule_sha256(trace_signal_dates):
            reasons.append("strict_execution_trace:signal_schedule_digest_mismatch")

    strict_summary_path = resolved.get("strict_backtest")
    if strict_summary_path and (canonical_trace_sha is not None or target_schedule_sha is not None):
        try:
            strict = json.loads(Path(strict_summary_path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"strict_backtest:json_invalid_for_timing:{type(exc).__name__}")
        else:
            if not isinstance(strict, dict):
                reasons.append("strict_backtest:not_object_for_timing")
            else:
                if canonical_trace_sha is not None and str(
                    strict.get("execution_trace_sha256") or ""
                ) != canonical_trace_sha:
                    reasons.append("strict_backtest:execution_trace_sha256_mismatch")
                if target_schedule_sha is not None and str(
                    strict.get("strict_target_signal_schedule_sha256") or ""
                ) != target_schedule_sha:
                    reasons.append("strict_backtest:target_signal_schedule_sha256_mismatch")
                if str(strict.get("execution_timing_semantics") or "") != EXECUTION_TIMING_SEMANTICS:
                    reasons.append("strict_backtest:execution_timing_semantics_mismatch")

    supplemental_reasons = [
        reason
        for reason in reasons
        if reason.startswith("strict_target_weights:")
        or reason.startswith("strict_execution_trace:")
        or reason.startswith("strict_backtest:execution_trace_")
        or reason.startswith("strict_backtest:target_signal_schedule_")
        or reason == "strict_backtest:execution_timing_semantics_mismatch"
        or reason.startswith("governed_execution_artifacts_incomplete:")
    ]
    if supplemental_present:
        if (
            target_present
            and trace_present
            and target_signal_dates is not None
            and trace_signal_dates is not None
            and target_signal_dates == trace_signal_dates
            and timing_ok
            and not timing_reasons
            and canonical_trace_sha
            and target_schedule_sha
            and not supplemental_reasons
        ):
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
    """Issue governed v2 evidence with complete strict target+trace timing proof."""
    if artifact_roots is None:
        raise ValueError("trace-proven v2 issuer requires explicit artifact_roots")
    missing = [role for role in GOVERNED_REQUIRED_ARTIFACT_ROLES if role not in artifact_locations]
    unexpected = sorted(set(artifact_locations).difference(GOVERNED_REQUIRED_ARTIFACT_ROLES))
    if missing or unexpected:
        raise ValueError(f"artifact role mismatch: missing={missing}, unexpected={unexpected}")

    supplemental_bindings: dict[str, ArtifactBinding] = {}
    for role in GOVERNED_EXECUTION_ARTIFACT_ROLES:
        root_name, relative = artifact_locations[role]
        path, path_error = _resolve_regular_artifact(
            str(root_name), str(relative), artifact_roots
        )
        if path is None:
            raise ValueError(f"{role}: {path_error or 'artifact_unresolvable'}")
        supplemental_bindings[role] = ArtifactBinding(
            root=str(root_name),
            path=_normalise_relative_path(str(relative)),
            sha256=sha256_file(path),
        )

    final = Path(manifest_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = final.with_name(f".{final.name}.{uuid4().hex}.trace-stage")
    base_locations = {role: artifact_locations[role] for role in REQUIRED_ARTIFACT_ROLES}
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
        for role, binding in supplemental_bindings.items():
            bound_artifacts[role] = binding.to_dict()
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
        if verification.evidence.get("execution_timing_assurance") != TRACE_PROVEN_EXECUTION_ASSURANCE:
            raise ValueError("trace-proven governed v2 issuer did not obtain timing assurance")

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


def _verify_descriptor(
    role: str,
    descriptor: object,
    artifact_roots: Mapping[str, str | Path],
    reasons: list[str],
    resolved: dict[str, str],
) -> tuple[Path | None, str | None]:
    if not isinstance(descriptor, Mapping):
        reasons.append(f"{role}:descriptor_not_object")
        return None, None
    root_name = str(descriptor.get("root") or "").strip()
    relative = str(descriptor.get("path") or "").strip()
    expected_sha = str(descriptor.get("sha256") or "").strip().lower()
    if not _HEX64.fullmatch(expected_sha):
        reasons.append(f"{role}:sha256_invalid")
        return None, None
    path, path_error = _resolve_regular_artifact(root_name, relative, artifact_roots)
    if path is None:
        reasons.append(f"{role}:{path_error or 'artifact_unresolvable'}")
        return None, None
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        reasons.append(f"{role}:sha256_mismatch:{actual_sha}!={expected_sha}")
        return None, None
    resolved[role] = str(path)
    return path, expected_sha


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
    "GOVERNED_TARGET_WEIGHTS_ROLE",
    "GOVERNED_EXECUTION_TRACE_ROLE",
    "GOVERNED_EXECUTION_ARTIFACT_ROLES",
    "GOVERNED_REQUIRED_ARTIFACT_ROLES",
    "TRACE_PROVEN_EXECUTION_ASSURANCE",
    "issue_trace_proven_live_model_trust_v2",
    "verify_trace_proven_live_model_trust_v2",
]
