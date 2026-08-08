"""Machine-enforced model-trust gate for economic live trading.

A broker connection is not a model promotion decision. QuantAgent therefore
requires a separate, explicit certificate before a live gateway may become
ready. The certificate is intentionally stricter than a backtest summary:
selection hygiene, statistical gates, a genuinely fresh one-shot OOS window,
benchmark evidence, risk/capacity evidence, and the exact trusted backtest
metric semantics are all first-class fields.

This module does not *create* trust. It only verifies a certificate produced by
the governed research/promotion process. Missing or ambiguous evidence fails
closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


_ACCEPTED_STATUS = {"production_accepted", "live_accepted"}
_ACCEPTED_TRUST = {"fresh_oos", "fresh_holdout_validated", "production_accepted"}
REQUIRED_METRIC_SEMANTICS = "strict_v8_nav_v2_initial_cash"


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
) -> LiveModelTrustReport:
    """Validate a live-model certificate; every required item is AND-gated."""
    if manifest_path is None or not str(manifest_path).strip():
        return _report(
            manifest_path="",
            payload={},
            reasons=("model_trust_manifest_missing",),
        )
    path = Path(manifest_path)
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

    return _report(str(path), payload, tuple(reasons))


def _report(
    manifest_path: str,
    payload: dict[str, Any],
    reasons: tuple[str, ...],
) -> LiveModelTrustReport:
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
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
