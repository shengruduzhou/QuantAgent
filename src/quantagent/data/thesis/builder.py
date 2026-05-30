"""CapitalFlowThesisBuilder — aggregate canonical evidence into theses.

A thesis is the abstract claim that capital, policy support, or
broker conviction is flowing toward a specific *direction* — usually
a sector or theme, sometimes a province or a single stock. The
builder is deterministic: given the same canonical evidence frame,
it produces the same list of theses.

Theses are downstream of canonical evidence and upstream of the
sector pool / decision chain. They are NEVER trading signals; the
:func:`validate_thesis` function in :mod:`.validation` re-scores
them by looking at realised forward returns, and a thesis only
graduates from ``unverified`` to ``verified`` after empirical
evidence supports it.

Design choices:

* Theses are keyed by ``(direction_kind, direction_value)``. The
  builder produces one thesis per unique direction within the
  evidence frame; multiple evidence items roll into the same thesis
  via the ``supporting_evidence_ids`` field.
* ``confidence`` is a weighted aggregate of supporting evidence
  ``confidence`` × source-type prior (policy > broker > inferred).
* ``contradiction_evidence_ids`` lists evidence whose direction score
  opposes the thesis aggregate sign; ``contradiction_score`` is the
  ratio of opposing magnitude to total magnitude.
* ``expected_lag_days`` is the median ``lag_window_candidates[0]``
  across supporting evidence — the earliest horizon at which the
  thesis can be falsified.
* ``decay_score`` and ``tradability_score`` are populated by the
  validation loop; the builder leaves them at neutral defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Schema + states
# ---------------------------------------------------------------------------

CAPITAL_FLOW_THESIS_COLUMNS: tuple[str, ...] = (
    "thesis_id",
    "direction_kind",
    "direction_value",
    "thesis_sign",
    "supporting_evidence_ids",
    "contradiction_evidence_ids",
    "n_supporting",
    "n_contradicting",
    "confidence",
    "contradiction_score",
    "expected_lag_days",
    "tradability_score",
    "decay_score",
    "validation_status",
    "created_at",
    "last_validated_at",
)


THESIS_VALIDATION_STATES: tuple[str, ...] = (
    "unverified",
    "partially_verified",
    "verified",
    "rejected",
    "expired",
)


# Source priors: how trustworthy a single evidence-source is, *before*
# its own ``confidence`` is applied. Tuned against the v8.2 builders'
# native scales so a high-credibility broker note never outweighs an
# official policy signal at parity confidence.
_SOURCE_PRIOR: dict[str, float] = {
    "policy": 1.00,
    "bond": 0.70,
    "bank": 0.65,
    "macro": 0.65,
    "fundamental": 0.60,
    "broker_view": 0.50,
    "news": 0.40,
    "state_team_inference": 0.55,
    "market_microstructure": 0.45,
    "capital_flow": 0.55,
    "sector": 0.50,
    "company": 0.50,
}


# ---------------------------------------------------------------------------
# Config + result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CapitalFlowThesisConfig:
    """Tuning knobs for thesis aggregation.

    ``min_supporting`` is the smallest evidence count for the builder
    to emit a thesis at all — a lone evidence record is too weak to
    earn a thesis-level claim. ``min_aggregate_confidence`` is the
    floor below which the resulting thesis stays in ``rejected``
    even when ``min_supporting`` passes, because the supporting
    signals are individually too weak.

    The default lag windows match the canonical
    ``lag_window_candidates``; callers can override for direction
    kinds with known different decay timescales.
    """

    min_supporting: int = 2
    min_aggregate_confidence: float = 0.30
    default_lag_windows: tuple[int, ...] = (1, 5, 20, 60, 120)
    direction_kinds: tuple[str, ...] = ("sector", "theme", "province", "symbol")


@dataclass(frozen=True)
class CapitalFlowThesis:
    """One thesis: capital is flowing in a given direction."""

    thesis_id: str
    direction_kind: str
    direction_value: str
    thesis_sign: float  # signed in [-1, 1]
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradiction_evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    contradiction_score: float = 0.0
    expected_lag_days: int = 5
    tradability_score: float = 0.50
    decay_score: float = 1.0
    validation_status: str = "unverified"
    created_at: pd.Timestamp | None = None
    last_validated_at: pd.Timestamp | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "thesis_id": self.thesis_id,
            "direction_kind": self.direction_kind,
            "direction_value": self.direction_value,
            "thesis_sign": float(self.thesis_sign),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradiction_evidence_ids": list(self.contradiction_evidence_ids),
            "n_supporting": int(len(self.supporting_evidence_ids)),
            "n_contradicting": int(len(self.contradiction_evidence_ids)),
            "confidence": float(self.confidence),
            "contradiction_score": float(self.contradiction_score),
            "expected_lag_days": int(self.expected_lag_days),
            "tradability_score": float(self.tradability_score),
            "decay_score": float(self.decay_score),
            "validation_status": str(self.validation_status),
            "created_at": self.created_at,
            "last_validated_at": self.last_validated_at,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _thesis_id(direction_kind: str, direction_value: str, created_at: pd.Timestamp) -> str:
    raw = f"{direction_kind}||{direction_value}||{created_at.isoformat()}"
    return f"thesis_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _direction_score_for_row(row: pd.Series) -> float:
    """Combine the three direction-axis fields into one signed score.

    Policy-direction and capital-flow-direction are summed because
    they target the *same* eventual question — is this direction
    bullish or bearish? Sentiment is added at half weight as
    secondary confirmation.
    """
    policy = float(row.get("policy_direction_score") or 0.0)
    capital = float(row.get("capital_flow_direction_score") or 0.0)
    sentiment = float(row.get("sentiment_score") or 0.0)
    raw = policy + capital + 0.5 * sentiment
    return float(np.clip(raw / 2.5, -1.0, 1.0))


def _explode_directions(
    canonical: pd.DataFrame,
    direction_kinds: tuple[str, ...],
) -> pd.DataFrame:
    """Long-form view: one row per (evidence, direction_kind, direction_value).

    ``entities`` is the polymorphic list — entries that look like
    ``sector:Semi`` are tagged as direction_kind=``sector``,
    entries that match a known theme list as ``theme``, raw
    symbols (``\\d{6}\\.[A-Z]{2}``) as ``symbol``, and any
    remaining strings as ``theme`` by default.
    """
    if canonical is None or canonical.empty:
        return pd.DataFrame(
            columns=[
                "evidence_id", "source_type", "available_at", "direction_kind",
                "direction_value", "direction_score", "confidence",
                "lag_window_candidates",
            ]
        )
    rows: list[dict[str, Any]] = []
    for _, ev in canonical.iterrows():
        entities = ev.get("entities") or []
        if not isinstance(entities, (list, tuple)):
            continue
        direction = _direction_score_for_row(ev)
        conf = float(ev.get("confidence") or 0.0)
        lag_cands = ev.get("lag_window_candidates") or [1, 5, 20, 60, 120]
        for entity in entities:
            entity_s = str(entity).strip()
            if not entity_s:
                continue
            kind, value = _classify_entity(entity_s)
            if kind not in direction_kinds:
                continue
            rows.append(
                {
                    "evidence_id": str(ev.get("evidence_id")),
                    "source_type": str(ev.get("source_type")),
                    "available_at": ev.get("available_at"),
                    "direction_kind": kind,
                    "direction_value": value,
                    "direction_score": direction,
                    "confidence": conf,
                    "lag_window_candidates": list(lag_cands),
                }
            )
    return pd.DataFrame(rows)


def _classify_entity(entity: str) -> tuple[str, str]:
    """Map a raw entity string to ``(direction_kind, normalised_value)``.

    Prefixed values like ``sector:Semi`` or ``index:510300.SH`` are
    interpreted verbatim. Bare alphanumeric strings are heuristically
    sorted into ``symbol`` (Chinese ticker pattern) vs ``theme``.
    """
    s = entity.strip()
    if ":" in s:
        prefix, _, value = s.partition(":")
        prefix = prefix.lower().strip()
        value = value.strip()
        if prefix in {"sector", "industry"}:
            return "sector", value
        if prefix in {"theme", "concept"}:
            return "theme", value
        if prefix == "province":
            return "province", value
        if prefix in {"index", "etf"}:
            return "theme", f"INDEX:{value}"
        return prefix, value
    # bare. Heuristics.
    if len(s) == 9 and s[6] == "." and s[:6].isdigit():
        return "symbol", s
    return "theme", s


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_capital_flow_theses(
    canonical_evidence: pd.DataFrame,
    *,
    config: CapitalFlowThesisConfig | None = None,
    as_of: pd.Timestamp | None = None,
) -> list[CapitalFlowThesis]:
    """Aggregate canonical evidence into a list of capital-flow theses.

    Parameters
    ----------
    canonical_evidence:
        Output of :func:`quantagent.data.evidence.to_canonical_evidence_frame`
        — required columns include ``evidence_id``, ``source_type``,
        ``entities``, ``policy_direction_score``,
        ``capital_flow_direction_score``, ``sentiment_score``,
        ``confidence``, ``available_at``, ``lag_window_candidates``.
    config:
        Tuning knobs (see :class:`CapitalFlowThesisConfig`).
    as_of:
        Optional builder timestamp used for ``created_at``. Defaults
        to ``max(available_at)`` across the input evidence.
    """
    cfg = config or CapitalFlowThesisConfig()
    if canonical_evidence is None or canonical_evidence.empty:
        return []
    exploded = _explode_directions(canonical_evidence, cfg.direction_kinds)
    if exploded.empty:
        return []
    exploded["available_at"] = pd.to_datetime(exploded["available_at"], errors="coerce")

    timestamp = as_of or exploded["available_at"].dropna().max()
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.now().normalize()

    theses: list[CapitalFlowThesis] = []
    grouped = exploded.groupby(["direction_kind", "direction_value"], sort=False)
    for (kind, value), group in grouped:
        if len(group) < cfg.min_supporting:
            continue
        # Weighted aggregation
        weights = group.apply(
            lambda r: _SOURCE_PRIOR.get(r["source_type"], 0.50) * max(0.05, r["confidence"]),
            axis=1,
        ).astype(float).values
        scores = group["direction_score"].astype(float).values
        if float(weights.sum()) <= 1e-9:
            continue
        weighted_score = float(np.dot(scores, weights) / weights.sum())
        thesis_sign = float(np.clip(weighted_score, -1.0, 1.0))

        supporting_ids = group.loc[
            np.sign(scores) == np.sign(thesis_sign) if thesis_sign != 0 else np.ones_like(scores, dtype=bool),
            "evidence_id",
        ].astype(str).tolist()
        contradiction_ids = group.loc[
            np.sign(scores) == -np.sign(thesis_sign) if thesis_sign != 0 else np.zeros_like(scores, dtype=bool),
            "evidence_id",
        ].astype(str).tolist()
        # Deduplicate
        supporting_ids = list(dict.fromkeys(supporting_ids))
        contradiction_ids = list(dict.fromkeys(contradiction_ids))

        if len(supporting_ids) < cfg.min_supporting:
            continue

        magnitudes = np.abs(scores) * weights
        if magnitudes.sum() <= 1e-9:
            continue
        opposing_mag = (
            magnitudes[np.sign(scores) == -np.sign(thesis_sign)].sum()
            if thesis_sign != 0
            else 0.0
        )
        contradiction_score = float(opposing_mag / magnitudes.sum())

        agg_conf = float(np.clip(weights.sum() / max(len(group), 1), 0.0, 1.0))

        # Expected lag = median first-element of lag_window_candidates
        firsts = [
            int(lst[0]) if isinstance(lst, (list, tuple)) and len(lst) > 0 else 5
            for lst in group["lag_window_candidates"]
        ]
        expected_lag = int(np.median(firsts)) if firsts else 5

        status = "unverified"
        if agg_conf < cfg.min_aggregate_confidence:
            status = "rejected"

        thesis = CapitalFlowThesis(
            thesis_id=_thesis_id(kind, value, timestamp),
            direction_kind=kind,
            direction_value=value,
            thesis_sign=thesis_sign,
            supporting_evidence_ids=supporting_ids,
            contradiction_evidence_ids=contradiction_ids,
            confidence=agg_conf,
            contradiction_score=contradiction_score,
            expected_lag_days=expected_lag,
            tradability_score=0.50,
            decay_score=1.0,
            validation_status=status,
            created_at=pd.Timestamp(timestamp),
            last_validated_at=None,
        )
        theses.append(thesis)
    # Sort by absolute conviction descending so consumers can take top-N
    theses.sort(key=lambda t: abs(t.thesis_sign) * t.confidence, reverse=True)
    return theses


def theses_to_frame(theses: Iterable[CapitalFlowThesis]) -> pd.DataFrame:
    """Materialise a list of theses to the canonical thesis DataFrame."""
    rows = [t.to_dict() for t in theses]
    if not rows:
        return pd.DataFrame(columns=list(CAPITAL_FLOW_THESIS_COLUMNS))
    frame = pd.DataFrame(rows)
    for col in CAPITAL_FLOW_THESIS_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    return frame[list(CAPITAL_FLOW_THESIS_COLUMNS)]


# ---------------------------------------------------------------------------
# Stateful wrapper
# ---------------------------------------------------------------------------

class CapitalFlowThesisBuilder:
    """Build + materialise theses with config + optional writer hook."""

    def __init__(self, config: CapitalFlowThesisConfig | None = None) -> None:
        self.config = config or CapitalFlowThesisConfig()

    def build(
        self,
        canonical_evidence: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
    ) -> list[CapitalFlowThesis]:
        return build_capital_flow_theses(
            canonical_evidence, config=self.config, as_of=as_of
        )

    def build_frame(
        self,
        canonical_evidence: pd.DataFrame,
        *,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return theses_to_frame(self.build(canonical_evidence, as_of=as_of))
