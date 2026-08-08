"""Machine-enforced model-trust gate for economic live trading.

Schema v1 remains readable for forensic and explicitly BLOCKED manifests, but
it is not a production trust root: all of its acceptance fields live in one
editable JSON object. Production acceptance therefore requires schema v2,
whose claims are re-derived from SHA-256-bound governed artifacts by
``live_model_trust_v2``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from quantagent.execution.live_model_trust_v2 import (
    CERTIFICATE_SCHEMA_VERSION,
    REQUIRED_METRIC_SEMANTICS,
    default_artifact_roots,
    verify_live_model_trust_v2,
)


_ACCEPTED_STATUS = {"production_accepted", "live_accepted"}
_ACCEPTED_TRUST = {"fresh_oos", "fresh_holdout_validated", "production_accepted"}


@dataclass(frozen=True)
class LiveModelTrustReport:
    ok: bool
    status: str
    model_id: str | None
    trust_class: str | None
    reasons: tuple[str, ...]
    evidence: dict[str, Any]
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "model_id": self.model_id,
            "trust_class": self.trust_class,
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
            "manifest_path": self.manifest_path,
        }


def evaluate_live_model_trust(
    manifest_path: str | Path | None,
    *,
    min_fresh_oos_days: int = 120,
    max_pbo: float = 0.25,
    min_dsr_probability: float = 0.95,
    max_spa_p_value: float = 0.05,
    required_metric_semantics: str = REQUIRED_METRIC_SEMANTICS,
    artifact_roots: Mapping[str, str | Path] | None = None,
) -> LiveModelTrustReport:
    """Validate a model-trust certificate; missing/ambiguous evidence fails closed."""
    if manifest_path is None or not str(manifest_path).strip():
        return _report(
            manifest_path="",
            payload={},
            reasons=("model_trust_manifest_missing",),
        )
    path = Path(manifest_path)
    # The root certificate is itself part of the trust boundary.  Artifact
    # bindings already reject symlink substitution; allowing the manifest to be
    # replaced by a symlink would make the outermost trust object weaker than
    # every artifact it indexes.
    if path.is_symlink():
        return _report(
            manifest_path=str(path),
            payload={},
            reasons=("model_trust_manifest_symlink_not_allowed",),
        )
    if not path.exists() or not path.is_file():
        return _report(
            manifest_path=str(path),
            payload={},
            reasons=("model_trust_manifest_not_found",),
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _report(
            manifest_path=str(path),
            payload={},
            reasons=(f"model_trust_manifest_invalid:{type(exc).__name__}",),
        )
    if not isinstance(payload, dict):
        return _report(
            manifest_path=str(path),
            payload={},
            reasons=("model_trust_manifest_not_object",),
        )

    raw_schema = payload.get("schema_version", 1)
    try:
        schema_version = int(raw_schema)
    except (TypeError, ValueError):
        return _report(
            manifest_path=str(path),
            payload=payload,
            reasons=(f"model_trust_schema_version_invalid:{raw_schema!r}",),
        )

    if schema_version == CERTIFICATE_SCHEMA_VERSION:
        roots = artifact_roots or default_artifact_roots(path)
        verification = verify_live_model_trust_v2(
            payload,
            artifact_roots=roots,
            min_fresh_oos_days=min_fresh_oos_days,
            max_pbo=max_pbo,
            min_dsr_probability=min_dsr_probability,
            max_spa_p_value=max_spa_p_value,
        )
        reasons = list(verification.reasons)
        # The v2 policy is deliberately fixed to the canonical trusted evaluator
        # semantics. A caller requesting anything else cannot silently weaken or
        # fork the trust policy.
        if required_metric_semantics != REQUIRED_METRIC_SEMANTICS:
            reasons.append(
                "v2_required_metric_semantics_not_canonical:"
                f"{required_metric_semantics}!={REQUIRED_METRIC_SEMANTICS}"
            )
        return _report(
            manifest_path=str(path),
            payload=payload,
            reasons=tuple(dict.fromkeys(reasons)),
            evidence=verification.evidence,
        )

    if schema_version != 1:
        return _report(
            manifest_path=str(path),
            payload=payload,
            reasons=(f"model_trust_schema_version_unsupported:{schema_version}",),
        )

    # Legacy schema v1. Preserve detailed diagnostics for the current blocked
    # repository manifest, but make production acceptance impossible regardless
    # of how its scalar fields are edited.
    reasons = _legacy_v1_reasons(
        payload,
        min_fresh_oos_days=min_fresh_oos_days,
        max_pbo=max_pbo,
        min_dsr_probability=min_dsr_probability,
        max_spa_p_value=max_spa_p_value,
        required_metric_semantics=required_metric_semantics,
    )
    status = str(payload.get("status") or "").strip().lower()
    if status in _ACCEPTED_STATUS:
        reasons.insert(0, "legacy_schema_v1_not_production_eligible")
    return _report(str(path), payload, tuple(dict.fromkeys(reasons)))


def _legacy_v1_reasons(
    payload: dict[str, Any],
    *,
    min_fresh_oos_days: int,
    max_pbo: float,
    min_dsr_probability: float,
    max_spa_p_value: float,
    required_metric_semantics: str,
) -> list[str]:
    reasons: list[str] = []
    status = str(payload.get("status") or "").strip().lower()
    trust_class = str(payload.get("trust_class") or "").strip().lower()
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}

    if status not in _ACCEPTED_STATUS:
        reasons.append(f"status_not_live_accepted:{status or 'missing'}")
    if trust_class not in _ACCEPTED_TRUST:
        reasons.append(f"trust_class_not_live_eligible:{trust_class or 'missing'}")

    fresh_oos_days = _finite_float(evidence.get("fresh_oos_days"))
    if fresh_oos_days is None or fresh_oos_days < min_fresh_oos_days:
        reasons.append(
            f"fresh_oos_days_below_{min_fresh_oos_days}:"
            f"{fresh_oos_days if fresh_oos_days is not None else 'missing'}"
        )

    holdout_reads = _finite_float(evidence.get("final_holdout_reads"))
    if holdout_reads is None or holdout_reads != 1:
        reasons.append(
            "final_holdout_reads_must_equal_1:"
            f"{holdout_reads if holdout_reads is not None else 'missing'}"
        )

    pbo = _finite_float(evidence.get("pbo"))
    if pbo is None or pbo > max_pbo:
        reasons.append(f"pbo_above_{max_pbo}:{pbo if pbo is not None else 'missing'}")

    dsr = _finite_float(evidence.get("dsr_probability"))
    if dsr is None or dsr < min_dsr_probability:
        reasons.append(
            f"dsr_probability_below_{min_dsr_probability}:"
            f"{dsr if dsr is not None else 'missing'}"
        )

    spa = _finite_float(evidence.get("spa_p_value"))
    if spa is None or spa > max_spa_p_value:
        reasons.append(
            f"spa_p_value_above_{max_spa_p_value}:"
            f"{spa if spa is not None else 'missing'}"
        )

    if evidence.get("variant_c_passed") is not True:
        reasons.append("variant_c_not_proven")
    if evidence.get("benchmark_excess_positive") is not True:
        reasons.append("benchmark_excess_not_positive")
    if evidence.get("risk_capacity_passed") is not True:
        reasons.append("risk_capacity_gate_not_proven")
    if evidence.get("selection_pre_registered") is not True:
        reasons.append("selection_not_pre_registered")
    if evidence.get("contaminated_holdout") is not False:
        reasons.append("holdout_contamination_not_explicitly_false")

    metric_semantics = str(evidence.get("strict_backtest_metric_semantics") or "").strip()
    if metric_semantics != required_metric_semantics:
        reasons.append(
            "strict_backtest_metric_semantics_mismatch:"
            f"{metric_semantics or 'missing'}!={required_metric_semantics}"
        )
    return reasons


def _report(
    manifest_path: str,
    payload: dict[str, Any],
    reasons: tuple[str, ...],
    *,
    evidence: dict[str, Any] | None = None,
) -> LiveModelTrustReport:
    if evidence is None:
        raw_evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
        evidence = dict(raw_evidence)
    status = str(payload.get("status") or "missing")
    return LiveModelTrustReport(
        ok=not reasons,
        status=status,
        model_id=str(payload.get("model_id")) if payload.get("model_id") else None,
        trust_class=str(payload.get("trust_class")) if payload.get("trust_class") else None,
        reasons=reasons,
        evidence=dict(evidence),
        manifest_path=manifest_path,
    )


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


__all__ = [
    "LiveModelTrustReport",
    "REQUIRED_METRIC_SEMANTICS",
    "evaluate_live_model_trust",
]
