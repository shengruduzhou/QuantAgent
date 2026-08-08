"""Production policy layer for schema-v2 model trust.

The low-level v2 module verifies path safety, SHA-256 bindings and structured
cross-artifact claims.  This layer adds the facts that must be *recomputed* from
bound data rather than trusted from summary JSON:

- FRESH OOS trading-day coverage from the prediction artifact itself;
- PBO, DSR and SPA from the complete early-OOS candidate return matrix using
  QuantAgent's existing frozen-candidate governance implementation.

Only this governed verifier/issuer is suitable for the live trust boundary.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from quantagent.execution.live_model_evidence import (
    recompute_statistical_evidence,
    validate_fresh_predictions,
)
from quantagent.execution.live_model_trust_v2 import (
    LiveModelTrustV2IssueResult,
    V2VerificationResult,
    issue_live_model_trust_v2 as _issue_digest_bound_v2,
    verify_live_model_trust_v2 as _verify_digest_bound_v2,
)


DEFAULT_MINIMUM_STATISTICAL_OOS_DAYS = 80


def verify_governed_live_model_trust_v2(
    payload: Mapping[str, Any],
    *,
    artifact_roots: Mapping[str, str | Path],
    min_fresh_oos_days: int = 120,
    max_pbo: float = 0.25,
    min_dsr_probability: float = 0.95,
    max_spa_p_value: float = 0.05,
) -> V2VerificationResult:
    """Verify digest provenance, then independently derive FRESH/statistical facts."""
    base = _verify_digest_bound_v2(
        payload,
        artifact_roots=artifact_roots,
        min_fresh_oos_days=min_fresh_oos_days,
        max_pbo=max_pbo,
        min_dsr_probability=min_dsr_probability,
        max_spa_p_value=max_spa_p_value,
    )
    reasons = list(base.reasons)
    evidence = dict(base.evidence)
    resolved = dict(base.resolved_paths)

    # If a required artifact did not even pass digest/path verification, do not
    # attempt to parse an untrusted or missing path.  The base reasons remain the
    # authoritative failure evidence.
    fresh_path = resolved.get("fresh_oos_predictions")
    fresh_summary_path = resolved.get("fresh_oos")
    if fresh_path and fresh_summary_path:
        try:
            derived_fresh = validate_fresh_predictions(fresh_path)
            fresh_summary = _read_json_object(Path(fresh_summary_path), "fresh_oos")
        except ValueError as exc:
            reasons.append(str(exc))
        else:
            claimed_days = _strict_positive_int(fresh_summary.get("trading_days"))
            claimed_start = str(fresh_summary.get("start_date") or "")
            claimed_end = str(fresh_summary.get("end_date") or "")
            if derived_fresh.trading_days < int(min_fresh_oos_days):
                reasons.append(
                    "fresh_oos_predictions:derived_trading_days_below_"
                    f"{int(min_fresh_oos_days)}:{derived_fresh.trading_days}"
                )
            if claimed_days != derived_fresh.trading_days:
                reasons.append(
                    "fresh_oos:trading_days_mismatch_predictions:"
                    f"{claimed_days}!={derived_fresh.trading_days}"
                )
            if claimed_start != derived_fresh.start_date:
                reasons.append(
                    "fresh_oos:start_date_mismatch_predictions:"
                    f"{claimed_start}!={derived_fresh.start_date}"
                )
            if claimed_end != derived_fresh.end_date:
                reasons.append(
                    "fresh_oos:end_date_mismatch_predictions:"
                    f"{claimed_end}!={derived_fresh.end_date}"
                )
            evidence.update(
                {
                    "fresh_oos_days": derived_fresh.trading_days,
                    "fresh_oos_start": derived_fresh.start_date,
                    "fresh_oos_end": derived_fresh.end_date,
                    "fresh_prediction_rows": derived_fresh.rows,
                    "fresh_prediction_symbols": derived_fresh.symbols,
                }
            )

    returns_path = resolved.get("statistical_returns")
    stats_path = resolved.get("statistical_gates")
    search_path = resolved.get("search_ledger")
    prereg_path = resolved.get("pre_registration")
    if returns_path and stats_path and search_path and prereg_path:
        try:
            stats = _read_json_object(Path(stats_path), "statistical_gates")
            search = _read_json_object(Path(search_path), "search_ledger")
            prereg = _read_json_object(Path(prereg_path), "pre_registration")
            family = _strict_string_list(search.get("candidate_family"), "search_ledger:candidate_family")
            prereg_family = _strict_string_list(
                prereg.get("candidate_family"), "pre_registration:candidate_family"
            )
            if family != prereg_family:
                raise ValueError("statistical_returns:candidate_family_not_pre_registered")
            selected = str(search.get("selected_candidate") or "").strip()
            stats_selected = str(stats.get("selected_candidate") or "").strip()
            if not selected or selected not in family:
                raise ValueError("search_ledger:selected_candidate_invalid")
            if stats_selected != selected:
                raise ValueError("statistical_gates:selected_candidate_mismatch_search_ledger")
            trial_count = _strict_positive_int(search.get("trial_count"))
            if trial_count is None:
                raise ValueError("search_ledger:trial_count_invalid")
            derived_stats = recompute_statistical_evidence(
                returns_path,
                candidate_family=family,
                selected_candidate=selected,
                cumulative_trials=trial_count,
                max_pbo=max_pbo,
                min_dsr_probability=min_dsr_probability,
                max_spa_p_value=max_spa_p_value,
                minimum_observed_days=DEFAULT_MINIMUM_STATISTICAL_OOS_DAYS,
            )
        except ValueError as exc:
            reasons.append(str(exc))
        else:
            report = derived_stats.report
            claimed = {
                "pbo": _probability(stats.get("pbo")),
                "dsr_probability": _probability(stats.get("dsr_probability")),
                "spa_p_value": _probability(stats.get("spa_p_value")),
            }
            recomputed = {
                "pbo": float(report.pbo),
                "dsr_probability": float(report.dsr_probability),
                "spa_p_value": float(report.spa_pvalue),
            }
            for key, actual in recomputed.items():
                stated = claimed[key]
                if stated is None or not math.isclose(
                    stated,
                    actual,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    reasons.append(
                        f"statistical_gates:{key}_mismatch_recomputed:{stated}!={actual}"
                    )
            if not report.accepted:
                reasons.extend(
                    f"statistical_gates:recomputed_reject:{reason}"
                    for reason in report.rejection_reasons
                )
            evidence.update(
                {
                    **recomputed,
                    "statistical_selected_candidate": report.selected_candidate,
                    "statistical_observed_days": report.observed_days,
                    "statistical_returns_start": derived_stats.start_date,
                    "statistical_returns_end": derived_stats.end_date,
                    "statistical_losing_fold_rate": report.losing_fold_rate,
                    "statistical_recomputed": True,
                }
            )

    return replace(
        base,
        ok=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        evidence=evidence,
    )


def issue_governed_live_model_trust_v2(
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
    """Stage a digest-bound cert, apply governed verification, then atomically publish."""
    final = Path(manifest_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = final.with_name(f".{final.name}.{uuid4().hex}.stage")
    try:
        staged = _issue_digest_bound_v2(
            stage,
            model_id=model_id,
            source_commit=source_commit,
            artifact_locations=artifact_locations,
            artifact_roots=artifact_roots,
            issued_at=issued_at,
            min_fresh_oos_days=min_fresh_oos_days,
            max_pbo=max_pbo,
            min_dsr_probability=min_dsr_probability,
            max_spa_p_value=max_spa_p_value,
        )
        roots = artifact_roots
        if roots is None:
            # The low-level issuer resolved its defaults from the staging path;
            # governed production callers should pass roots explicitly.  Refuse
            # ambiguity rather than deriving a potentially different runtime root.
            raise ValueError("governed v2 issuer requires explicit artifact_roots")
        verification = verify_governed_live_model_trust_v2(
            staged.payload,
            artifact_roots=roots,
            min_fresh_oos_days=min_fresh_oos_days,
            max_pbo=max_pbo,
            min_dsr_probability=min_dsr_probability,
            max_spa_p_value=max_spa_p_value,
        )
        if not verification.ok:
            raise ValueError(
                "governed v2 trust evidence rejected: " + "; ".join(verification.reasons)
            )
        os.replace(stage, final)
        return LiveModelTrustV2IssueResult(
            manifest_path=str(final),
            payload=staged.payload,
            verification=verification,
        )
    finally:
        try:
            if stage.exists() or stage.is_symlink():
                stage.unlink()
        except OSError:
            pass


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{role}:json_invalid:{type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{role}:json_not_object")
    return payload


def _strict_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}_invalid")
    items = [str(item).strip() for item in value]
    if not all(items) or len(set(items)) != len(items):
        raise ValueError(f"{label}_invalid")
    return items


def _strict_positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _probability(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


__all__ = [
    "DEFAULT_MINIMUM_STATISTICAL_OOS_DAYS",
    "issue_governed_live_model_trust_v2",
    "verify_governed_live_model_trust_v2",
]
