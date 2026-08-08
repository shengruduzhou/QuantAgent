"""Hash-bound schema-v2 model-trust provenance.

This module deliberately borrows the *verification pattern* of provenance
systems (bind named inputs to digests, then verify both digests and expected
semantics) without claiming SLSA conformance.  A v2 QuantAgent certificate is
therefore a compact index over governed evidence, not a bag of self-attested
PASS fields.

The issuer can only write a production-accepted certificate after the same
verifier that live execution uses has re-opened every bound artifact and
validated cross-artifact consistency.  Any later byte change invalidates the
certificate through SHA-256 mismatch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from quantagent.config.paths import quant_paths
from quantagent.training.semantics import FT_TRANSFORMER_OBJECTIVE_SEMANTICS_VERSION


CERTIFICATE_SCHEMA_VERSION = 2
CERTIFICATE_TYPE = "quantagent.live_model_trust.v2"
REQUIRED_METRIC_SEMANTICS = "strict_v8_nav_v2_initial_cash"
REQUIRED_TRAINER_OBJECTIVE_SEMANTICS = FT_TRANSFORMER_OBJECTIVE_SEMANTICS_VERSION

REQUIRED_ARTIFACT_ROLES: tuple[str, ...] = (
    "model_checkpoint",
    "trainer_manifest",
    "pre_registration",
    "search_ledger",
    "data_lineage",
    "strict_backtest",
    "statistical_returns",
    "statistical_gates",
    "fresh_oos_predictions",
    "fresh_oos",
    "risk_capacity",
)
JSON_ARTIFACT_ROLES: frozenset[str] = frozenset(
    {
        "trainer_manifest",
        "pre_registration",
        "search_ledger",
        "data_lineage",
        "strict_backtest",
        "statistical_gates",
        "fresh_oos",
        "risk_capacity",
    }
)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ArtifactBinding:
    root: str
    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class V2VerificationResult:
    ok: bool
    reasons: tuple[str, ...]
    evidence: dict[str, Any]
    resolved_paths: dict[str, str]


@dataclass(frozen=True)
class LiveModelTrustV2IssueResult:
    manifest_path: str
    payload: dict[str, Any]
    verification: V2VerificationResult


def default_artifact_roots(manifest_path: str | Path) -> dict[str, Path]:
    """Resolve the canonical ``repo`` and ``runtime`` roots for a certificate.

    ``configs/live_model_trust.json`` resolves ``repo`` to its repository root.
    Tests/bundles outside a ``configs`` directory resolve ``repo`` to the
    manifest's parent.  The runtime root follows QuantAgent's canonical path
    resolver (and therefore respects ``QUANTAGENT_HOME``).
    """
    manifest = Path(manifest_path).resolve(strict=False)
    repo_root = manifest.parent.parent if manifest.parent.name == "configs" else manifest.parent
    return {
        "repo": repo_root.resolve(strict=False),
        "runtime": quant_paths().home.resolve(strict=False),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def issue_live_model_trust_v2(
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
    """Create a v2 acceptance certificate from already-existing evidence.

    The issuer does not calculate research results and does not repair evidence.
    It hashes the supplied artifacts, builds the certificate, then invokes the
    production verifier before atomically writing anything.
    """
    manifest = Path(manifest_path)
    clean_model_id = str(model_id).strip()
    clean_commit = str(source_commit).strip().lower()
    if not clean_model_id:
        raise ValueError("model_id is required")
    if not _HEX40.fullmatch(clean_commit):
        raise ValueError("source_commit must be a 40-character lowercase git SHA")
    roots = _normalise_roots(artifact_roots or default_artifact_roots(manifest))
    missing_roles = [role for role in REQUIRED_ARTIFACT_ROLES if role not in artifact_locations]
    extra_roles = sorted(set(artifact_locations).difference(REQUIRED_ARTIFACT_ROLES))
    if missing_roles or extra_roles:
        raise ValueError(
            f"artifact role mismatch: missing={missing_roles}, unexpected={extra_roles}"
        )

    bindings: dict[str, dict[str, str]] = {}
    for role in REQUIRED_ARTIFACT_ROLES:
        root_name, relative = artifact_locations[role]
        resolved, error = _resolve_artifact(root_name, relative, roots)
        if error is not None or resolved is None:
            raise ValueError(f"{role}: {error or 'artifact_unresolvable'}")
        bindings[role] = ArtifactBinding(
            root=str(root_name),
            path=_normalise_relative_path(relative),
            sha256=sha256_file(resolved),
        ).to_dict()

    now = issued_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("issued_at must be timezone-aware")
    payload: dict[str, Any] = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "certificate_type": CERTIFICATE_TYPE,
        "status": "production_accepted",
        "trust_class": "fresh_oos",
        "model_id": clean_model_id,
        "source_commit": clean_commit,
        "issued_at": now.astimezone(timezone.utc).isoformat(),
        "policy": {
            "trainer_objective_semantics": REQUIRED_TRAINER_OBJECTIVE_SEMANTICS,
            "strict_backtest_metric_semantics": REQUIRED_METRIC_SEMANTICS,
            "min_fresh_oos_days": int(min_fresh_oos_days),
            "max_pbo": float(max_pbo),
            "min_dsr_probability": float(min_dsr_probability),
            "max_spa_p_value": float(max_spa_p_value),
        },
        "artifacts": bindings,
    }
    verification = verify_live_model_trust_v2(
        payload,
        artifact_roots=roots,
        min_fresh_oos_days=min_fresh_oos_days,
        max_pbo=max_pbo,
        min_dsr_probability=min_dsr_probability,
        max_spa_p_value=max_spa_p_value,
    )
    if not verification.ok:
        raise ValueError("v2 trust evidence rejected: " + "; ".join(verification.reasons))

    manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest.with_name(manifest.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, manifest)
    return LiveModelTrustV2IssueResult(str(manifest), payload, verification)


def verify_live_model_trust_v2(
    payload: Mapping[str, Any],
    *,
    artifact_roots: Mapping[str, str | Path],
    min_fresh_oos_days: int = 120,
    max_pbo: float = 0.25,
    min_dsr_probability: float = 0.95,
    max_spa_p_value: float = 0.05,
) -> V2VerificationResult:
    """Verify all v2 digests, schemas, semantics and cross-artifact claims."""
    reasons: list[str] = []
    roots = _normalise_roots(artifact_roots)
    model_id = str(payload.get("model_id") or "").strip()
    source_commit = str(payload.get("source_commit") or "").strip().lower()

    if payload.get("schema_version") != CERTIFICATE_SCHEMA_VERSION:
        reasons.append("v2_schema_version_mismatch")
    if payload.get("certificate_type") != CERTIFICATE_TYPE:
        reasons.append("v2_certificate_type_mismatch")
    if str(payload.get("status") or "").lower() != "production_accepted":
        reasons.append("v2_status_not_production_accepted")
    if str(payload.get("trust_class") or "").lower() != "fresh_oos":
        reasons.append("v2_trust_class_not_fresh_oos")
    if not model_id:
        reasons.append("v2_model_id_missing")
    if not _HEX40.fullmatch(source_commit):
        reasons.append("v2_source_commit_invalid")
    if _parse_timestamp(payload.get("issued_at")) is None:
        reasons.append("v2_issued_at_invalid")

    policy = payload.get("policy") if isinstance(payload.get("policy"), Mapping) else {}
    expected_policy = {
        "trainer_objective_semantics": REQUIRED_TRAINER_OBJECTIVE_SEMANTICS,
        "strict_backtest_metric_semantics": REQUIRED_METRIC_SEMANTICS,
        "min_fresh_oos_days": int(min_fresh_oos_days),
        "max_pbo": float(max_pbo),
        "min_dsr_probability": float(min_dsr_probability),
        "max_spa_p_value": float(max_spa_p_value),
    }
    for key, expected in expected_policy.items():
        observed = policy.get(key)
        if observed != expected:
            reasons.append(f"v2_policy_mismatch:{key}:{observed!r}!={expected!r}")

    raw_artifacts = payload.get("artifacts")
    artifacts = raw_artifacts if isinstance(raw_artifacts, Mapping) else {}
    missing_roles = [role for role in REQUIRED_ARTIFACT_ROLES if role not in artifacts]
    unexpected_roles = sorted(set(artifacts).difference(REQUIRED_ARTIFACT_ROLES))
    if missing_roles:
        reasons.append("v2_artifacts_missing:" + ",".join(missing_roles))
    if unexpected_roles:
        reasons.append("v2_artifacts_unexpected:" + ",".join(unexpected_roles))

    resolved_paths: dict[str, Path] = {}
    bindings: dict[str, ArtifactBinding] = {}
    for role in REQUIRED_ARTIFACT_ROLES:
        descriptor = artifacts.get(role)
        if not isinstance(descriptor, Mapping):
            continue
        root_name = str(descriptor.get("root") or "")
        relative = str(descriptor.get("path") or "")
        expected_sha = str(descriptor.get("sha256") or "").lower()
        if not _HEX64.fullmatch(expected_sha):
            reasons.append(f"{role}:sha256_invalid")
            continue
        resolved, error = _resolve_artifact(root_name, relative, roots)
        if error is not None or resolved is None:
            reasons.append(f"{role}:{error or 'artifact_unresolvable'}")
            continue
        actual_sha = sha256_file(resolved)
        if actual_sha != expected_sha:
            reasons.append(f"{role}:sha256_mismatch:{actual_sha}!={expected_sha}")
            continue
        resolved_paths[role] = resolved
        bindings[role] = ArtifactBinding(root_name, _normalise_relative_path(relative), expected_sha)

    json_docs: dict[str, dict[str, Any]] = {}
    for role in JSON_ARTIFACT_ROLES:
        path = resolved_paths.get(role)
        if path is None:
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            reasons.append(f"{role}:json_invalid:{type(exc).__name__}")
            continue
        if not isinstance(doc, dict):
            reasons.append(f"{role}:json_not_object")
            continue
        json_docs[role] = doc

    # Every structured evidence artifact must be tied to the same selected
    # model and code revision.  This prevents a certificate from mixing a good
    # backtest from model A with a checkpoint from model B.
    for role, doc in json_docs.items():
        if str(doc.get("model_id") or "") != model_id:
            reasons.append(f"{role}:model_id_mismatch")
        if str(doc.get("source_commit") or "").lower() != source_commit:
            reasons.append(f"{role}:source_commit_mismatch")

    trainer = json_docs.get("trainer_manifest", {})
    if trainer:
        if trainer.get("objective_semantics") != REQUIRED_TRAINER_OBJECTIVE_SEMANTICS:
            reasons.append("trainer_manifest:objective_semantics_mismatch")
        checkpoint_sha = bindings.get("model_checkpoint")
        if checkpoint_sha is not None and trainer.get("checkpoint_sha256") != checkpoint_sha.sha256:
            reasons.append("trainer_manifest:checkpoint_sha256_mismatch")

    prereg = json_docs.get("pre_registration", {})
    if prereg:
        if prereg.get("selection_pre_registered") is not True:
            reasons.append("pre_registration:selection_not_pre_registered")
        if _parse_timestamp(prereg.get("registered_at")) is None:
            reasons.append("pre_registration:registered_at_invalid")

    search = json_docs.get("search_ledger", {})
    trial_count = _positive_int(search.get("trial_count")) if search else None
    candidate_family = _string_list(search.get("candidate_family")) if search else None
    if search:
        if search.get("final_holdout_used_for_selection") is not False:
            reasons.append("search_ledger:final_holdout_used_for_selection_not_false")
        if search.get("selection_frozen_before_fresh_oos") is not True:
            reasons.append("search_ledger:selection_not_frozen_before_fresh_oos")
        if trial_count is None:
            reasons.append("search_ledger:trial_count_invalid")
        if not candidate_family:
            reasons.append("search_ledger:candidate_family_invalid")

    lineage = json_docs.get("data_lineage", {})
    if lineage:
        if lineage.get("pit") is not True:
            reasons.append("data_lineage:pit_not_true")
        if lineage.get("universe_pit") is not True:
            reasons.append("data_lineage:universe_pit_not_true")

    strict = json_docs.get("strict_backtest", {})
    returns_binding = bindings.get("statistical_returns")
    if strict:
        if strict.get("metric_semantics") != REQUIRED_METRIC_SEMANTICS:
            reasons.append("strict_backtest:metric_semantics_mismatch")
        if strict.get("t_plus_one") is not True:
            reasons.append("strict_backtest:t_plus_one_not_true")
        if strict.get("costs_included") is not True:
            reasons.append("strict_backtest:costs_not_proven")
        if strict.get("benchmark_excess_positive") is not True:
            reasons.append("strict_backtest:benchmark_excess_not_positive")
        if strict.get("variant_c_passed") is not True:
            reasons.append("strict_backtest:variant_c_not_proven")
        if returns_binding is not None and strict.get("daily_returns_sha256") != returns_binding.sha256:
            reasons.append("strict_backtest:daily_returns_sha256_mismatch")

    stats = json_docs.get("statistical_gates", {})
    pbo = _finite_float(stats.get("pbo")) if stats else None
    dsr = _finite_float(stats.get("dsr_probability")) if stats else None
    spa = _finite_float(stats.get("spa_p_value")) if stats else None
    stats_trial_count = _positive_int(stats.get("trial_count")) if stats else None
    stats_family = _string_list(stats.get("candidate_family")) if stats else None
    if stats:
        if pbo is None or pbo > max_pbo:
            reasons.append(f"statistical_gates:pbo_above_{max_pbo}:{pbo}")
        if dsr is None or dsr < min_dsr_probability:
            reasons.append(f"statistical_gates:dsr_below_{min_dsr_probability}:{dsr}")
        if spa is None or spa > max_spa_p_value:
            reasons.append(f"statistical_gates:spa_above_{max_spa_p_value}:{spa}")
        if returns_binding is not None and stats.get("returns_sha256") != returns_binding.sha256:
            reasons.append("statistical_gates:returns_sha256_mismatch")
        if trial_count is not None and stats_trial_count != trial_count:
            reasons.append("statistical_gates:trial_count_mismatch_search_ledger")
        if candidate_family and stats_family != candidate_family:
            reasons.append("statistical_gates:candidate_family_mismatch_search_ledger")

    fresh = json_docs.get("fresh_oos", {})
    fresh_predictions = bindings.get("fresh_oos_predictions")
    fresh_days = _positive_int(fresh.get("trading_days")) if fresh else None
    fresh_reads = _positive_int(fresh.get("final_holdout_reads")) if fresh else None
    fresh_start = _parse_date(fresh.get("start_date")) if fresh else None
    fresh_end = _parse_date(fresh.get("end_date")) if fresh else None
    fresh_window = str(fresh.get("acceptance_window_id") or "") if fresh else ""
    if fresh:
        if fresh_days is None or fresh_days < min_fresh_oos_days:
            reasons.append(f"fresh_oos:trading_days_below_{min_fresh_oos_days}:{fresh_days}")
        if fresh_reads != 1:
            reasons.append(f"fresh_oos:final_holdout_reads_must_equal_1:{fresh_reads}")
        if fresh.get("contaminated_holdout") is not False:
            reasons.append("fresh_oos:contaminated_holdout_not_false")
        if fresh_start is None or fresh_end is None or fresh_start > fresh_end:
            reasons.append("fresh_oos:acceptance_dates_invalid")
        if not fresh_window:
            reasons.append("fresh_oos:acceptance_window_id_missing")
        if fresh_predictions is not None and fresh.get("predictions_sha256") != fresh_predictions.sha256:
            reasons.append("fresh_oos:predictions_sha256_mismatch")

    # Pre-registration must identify the exact untouched acceptance window.
    if prereg and fresh:
        if str(prereg.get("acceptance_window_id") or "") != fresh_window:
            reasons.append("pre_registration:acceptance_window_id_mismatch")
        if _parse_date(prereg.get("acceptance_start_date")) != fresh_start:
            reasons.append("pre_registration:acceptance_start_date_mismatch")
        if _parse_date(prereg.get("acceptance_end_date")) != fresh_end:
            reasons.append("pre_registration:acceptance_end_date_mismatch")
        registered_at = _parse_timestamp(prereg.get("registered_at"))
        if registered_at is not None and fresh_start is not None and registered_at.date() > fresh_start:
            reasons.append("pre_registration:registered_after_fresh_oos_start")

    # The final trained model and its tuning window must end strictly before the
    # final acceptance window starts.  Walk-forward validation is tuning
    # evidence; it is not the final untouched holdout.
    training_cutoff = _parse_date(lineage.get("training_cutoff")) if lineage else None
    validation_cutoff = _parse_date(lineage.get("validation_cutoff")) if lineage else None
    trainer_cutoff = _parse_date(trainer.get("training_cutoff")) if trainer else None
    trainer_valid_cutoff = _parse_date(trainer.get("validation_cutoff")) if trainer else None
    for label, cutoff in (
        ("data_lineage:training_cutoff", training_cutoff),
        ("data_lineage:validation_cutoff", validation_cutoff),
        ("trainer_manifest:training_cutoff", trainer_cutoff),
        ("trainer_manifest:validation_cutoff", trainer_valid_cutoff),
    ):
        if cutoff is None:
            reasons.append(f"{label}_invalid")
        elif fresh_start is not None and cutoff >= fresh_start:
            reasons.append(f"{label}_not_before_fresh_oos")
    if training_cutoff and trainer_cutoff and training_cutoff != trainer_cutoff:
        reasons.append("trainer_manifest:training_cutoff_mismatch_data_lineage")
    if validation_cutoff and trainer_valid_cutoff and validation_cutoff != trainer_valid_cutoff:
        reasons.append("trainer_manifest:validation_cutoff_mismatch_data_lineage")

    risk = json_docs.get("risk_capacity", {})
    if risk and risk.get("passed") is not True:
        reasons.append("risk_capacity:gate_not_passed")

    evidence = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "trainer_objective_semantics": trainer.get("objective_semantics"),
        "strict_backtest_metric_semantics": strict.get("metric_semantics"),
        "fresh_oos_days": fresh_days,
        "final_holdout_reads": fresh_reads,
        "fresh_oos_start": None if fresh_start is None else fresh_start.isoformat(),
        "fresh_oos_end": None if fresh_end is None else fresh_end.isoformat(),
        "acceptance_window_id": fresh_window or None,
        "pbo": pbo,
        "dsr_probability": dsr,
        "spa_p_value": spa,
        "trial_count": stats_trial_count,
        "candidate_family": stats_family,
        "benchmark_excess_positive": strict.get("benchmark_excess_positive"),
        "variant_c_passed": strict.get("variant_c_passed"),
        "risk_capacity_passed": risk.get("passed"),
        "selection_pre_registered": prereg.get("selection_pre_registered"),
        "contaminated_holdout": fresh.get("contaminated_holdout"),
        "artifact_sha256": {role: binding.sha256 for role, binding in bindings.items()},
    }
    return V2VerificationResult(
        ok=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        evidence=evidence,
        resolved_paths={role: str(path) for role, path in resolved_paths.items()},
    )


def _normalise_roots(roots: Mapping[str, str | Path]) -> dict[str, Path]:
    normalised: dict[str, Path] = {}
    for name, root in roots.items():
        clean_name = str(name).strip()
        if not clean_name or clean_name in normalised:
            raise ValueError("artifact root names must be unique non-empty strings")
        normalised[clean_name] = Path(root).resolve(strict=False)
    if not normalised:
        raise ValueError("at least one artifact root is required")
    return normalised


def _normalise_relative_path(value: str | Path) -> str:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("artifact path must be a non-empty relative path without '..'")
    return path.as_posix()


def _resolve_artifact(
    root_name: str,
    relative_path: str | Path,
    roots: Mapping[str, Path],
) -> tuple[Path | None, str | None]:
    root = roots.get(str(root_name))
    if root is None:
        return None, "root_unknown"
    try:
        relative = Path(_normalise_relative_path(relative_path))
    except ValueError:
        return None, "path_invalid"
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return None, "symlink_not_allowed"
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None, "path_outside_root"
    if not candidate.exists():
        return None, "artifact_missing"
    if not candidate.is_file():
        return None, "artifact_not_regular_file"
    return candidate, None


def _parse_timestamp(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _parse_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value).strip())
    except Exception:
        return None


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _string_list(value: object) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    result = [str(item).strip() for item in value]
    return result if all(result) and len(set(result)) == len(result) else None


__all__ = [
    "ArtifactBinding",
    "CERTIFICATE_SCHEMA_VERSION",
    "CERTIFICATE_TYPE",
    "LiveModelTrustV2IssueResult",
    "REQUIRED_ARTIFACT_ROLES",
    "REQUIRED_METRIC_SEMANTICS",
    "REQUIRED_TRAINER_OBJECTIVE_SEMANTICS",
    "V2VerificationResult",
    "default_artifact_roots",
    "issue_live_model_trust_v2",
    "sha256_file",
    "verify_live_model_trust_v2",
]
