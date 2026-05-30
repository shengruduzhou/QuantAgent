"""Canonical EvidenceRecord schema + adapters from v8.2 silver layers.

The v8 spec section 1 requires every piece of evidence to land in a
unified record carrying:

    evidence_id, source_name, source_type, url_or_file_id,
    publish_time, crawl_time, available_at,
    entity_type, entities, raw_text_hash,
    extracted_claims, sentiment_score, policy_direction_score,
    capital_flow_direction_score, confidence, contradiction_score,
    lag_window_candidates, audit_trace.

The v8.2 builders (policy/bond/broker/state_team) each emit their own
native silver schema so the upstream coverage / gate logic stays
local to each source. The adapters in this module are the single seam
that produces a canonical frame from any subset of those silvers.

Adapters never reach back into the file system — they take an already
loaded DataFrame and return a canonical frame. Callers are responsible
for gating via the source-specific manifest (e.g.
``policy_events_for_features``) before adapting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

CANONICAL_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "evidence_id",
    "source_name",
    "source_type",
    "url_or_file_id",
    "publish_time",
    "crawl_time",
    "available_at",
    "entity_type",
    "entities",
    "raw_text_hash",
    "extracted_claims",
    "sentiment_score",
    "policy_direction_score",
    "capital_flow_direction_score",
    "confidence",
    "contradiction_score",
    "lag_window_candidates",
    "audit_trace",
)


CANONICAL_SOURCE_TYPES: tuple[str, ...] = (
    "policy",
    "bond",
    "bank",
    "sector",
    "company",
    "macro",
    "news",
    "broker_view",
    "market_microstructure",
    "state_team_inference",
    "fundamental",
    "capital_flow",
)


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceRecord:
    """One observation about the world, normalised across sources.

    All scoring fields use the convention ``+1 = bullish / supportive /
    inflow``, ``-1 = bearish / restrictive / outflow``, ``0 = neutral
    or unknown``. ``confidence`` and ``contradiction_score`` are
    magnitudes in ``[0, 1]``.

    ``available_at`` is the timestamp at which this record could first
    have entered a PIT join (``max(publish_time, crawl_time)`` plus any
    source-specific lag the adapter applied). It is the only timestamp
    downstream training / backtest joins are allowed to read.

    ``audit_trace`` carries provenance metadata: which adapter built
    the record, which source row keys it derived from, and which
    derivation rules ran. It is opaque to the consumer but must be
    preserved through any transformation.
    """

    evidence_id: str
    source_name: str
    source_type: str
    publish_time: pd.Timestamp
    available_at: pd.Timestamp
    entity_type: str
    entities: list[str] = field(default_factory=list)
    url_or_file_id: str | None = None
    crawl_time: pd.Timestamp | None = None
    raw_text_hash: str | None = None
    extracted_claims: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    policy_direction_score: float = 0.0
    capital_flow_direction_score: float = 0.0
    confidence: float = 0.0
    contradiction_score: float = 0.0
    lag_window_candidates: list[int] = field(default_factory=lambda: [1, 5, 20, 60, 120])
    audit_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "url_or_file_id": self.url_or_file_id,
            "publish_time": self.publish_time,
            "crawl_time": self.crawl_time,
            "available_at": self.available_at,
            "entity_type": self.entity_type,
            "entities": list(self.entities),
            "raw_text_hash": self.raw_text_hash,
            "extracted_claims": list(self.extracted_claims),
            "sentiment_score": float(self.sentiment_score),
            "policy_direction_score": float(self.policy_direction_score),
            "capital_flow_direction_score": float(self.capital_flow_direction_score),
            "confidence": float(self.confidence),
            "contradiction_score": float(self.contradiction_score),
            "lag_window_candidates": list(self.lag_window_candidates),
            "audit_trace": dict(self.audit_trace),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return [str(value)]


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "||".join([prefix, *(str(p) for p in parts)])
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _now_utc_naive() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))


def _clip01(value: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _clip_signed(value: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(-1.0, value)))


# ---------------------------------------------------------------------------
# Adapters: policy_events -> canonical
# ---------------------------------------------------------------------------

def policy_events_to_evidence(
    policy_events: pd.DataFrame,
    *,
    source_version: str | None = None,
) -> pd.DataFrame:
    """Convert ``silver/policy_events.parquet`` rows into canonical evidence.

    Maps:

    * ``event_id`` -> ``evidence_id`` (prefixed with ``policy_``)
    * ``source`` -> ``source_name`` (``source_type = "policy"``)
    * ``url`` -> ``url_or_file_id``
    * ``announced_at`` -> ``publish_time``
    * ``fetched_at`` -> ``crawl_time``
    * ``available_at`` -> ``available_at``
    * ``themes`` ∪ ``sectors_hint`` -> ``entities``
    * ``policy_strength`` -> ``policy_direction_score`` (already in
      [-1, 1] by builder convention; clipped defensively)
    * ``policy_strength`` magnitude -> ``confidence`` (|strength|)
    * ``themes / sectors_hint / title`` -> ``extracted_claims``
    """
    if policy_events is None or policy_events.empty:
        return pd.DataFrame(columns=list(CANONICAL_EVIDENCE_COLUMNS))

    out_rows: list[dict[str, Any]] = []
    for _, row in policy_events.iterrows():
        themes = _ensure_list(row.get("themes"))
        sectors_hint = _ensure_list(row.get("sectors_hint"))
        entities = sorted(set(themes) | set(sectors_hint))
        strength = float(row.get("policy_strength") or 0.0)
        publish_time = _coerce_ts(row.get("announced_at")) or _coerce_ts(row.get("available_at"))
        crawl_time = _coerce_ts(row.get("fetched_at"))
        available_at = _coerce_ts(row.get("available_at")) or publish_time or _now_utc_naive()
        evidence_id = str(row.get("event_id") or "")
        if not evidence_id:
            evidence_id = _stable_id(
                "policy",
                row.get("source", ""),
                row.get("url", ""),
                publish_time.isoformat() if publish_time is not None else "",
            )
        else:
            evidence_id = f"policy_{evidence_id}"
        claims: list[str] = []
        title = str(row.get("title") or "").strip()
        if title:
            claims.append(title)
        body_summary = str(row.get("body_summary") or "").strip()
        if body_summary and body_summary != title:
            claims.append(body_summary)
        if themes:
            claims.append("themes=" + ",".join(themes))
        if sectors_hint:
            claims.append("sectors=" + ",".join(sectors_hint))

        out_rows.append(
            {
                "evidence_id": evidence_id,
                "source_name": str(row.get("source") or ""),
                "source_type": "policy",
                "url_or_file_id": str(row.get("url") or "") or None,
                "publish_time": publish_time,
                "crawl_time": crawl_time,
                "available_at": available_at,
                "entity_type": "policy_event",
                "entities": entities,
                "raw_text_hash": None,
                "extracted_claims": claims,
                "sentiment_score": 0.0,
                "policy_direction_score": _clip_signed(strength),
                "capital_flow_direction_score": 0.0,
                "confidence": _clip01(abs(strength)),
                "contradiction_score": 0.0,
                "lag_window_candidates": [1, 5, 20, 60, 120],
                "audit_trace": {
                    "adapter": "policy_events_to_evidence",
                    "source_event_id": str(row.get("event_id") or ""),
                    "source_version": str(
                        row.get("source_version") or source_version or "unknown"
                    ),
                },
            }
        )
    return pd.DataFrame(out_rows, columns=list(CANONICAL_EVIDENCE_COLUMNS))


# ---------------------------------------------------------------------------
# Adapters: bond_flows -> canonical
# ---------------------------------------------------------------------------

def bond_flows_to_evidence(
    bond_flows: pd.DataFrame,
    *,
    source_version: str | None = None,
) -> pd.DataFrame:
    """Convert ``silver/bond_flows.parquet`` rows into canonical evidence.

    Each day becomes one ``market_microstructure`` evidence row. The
    ``capital_flow_direction_score`` is the sign of ``bond_fund_flow``
    rescaled to ``[-1, 1]``; the ``policy_direction_score`` is encoded
    via the inverted ``dr007`` and ``credit_spread_aa`` (low rates +
    tight spreads = supportive easing posture).

    Confidence is the share of bond-field non-nulls vs the canonical
    9 numeric bond columns, bounded to ``[0, 1]``.
    """
    if bond_flows is None or bond_flows.empty:
        return pd.DataFrame(columns=list(CANONICAL_EVIDENCE_COLUMNS))

    numeric_cols = (
        "yield_1y", "yield_5y", "yield_10y",
        "spread_10y_1y", "spread_10y_3m",
        "credit_spread_aa", "credit_spread_aaa_aa",
        "dr007", "bond_fund_flow",
    )

    # Symmetric tanh-style squash so unit changes around the typical
    # operating point produce meaningful score deltas.
    def _direction_capital(flow_cny_bn: float) -> float:
        if flow_cny_bn is None or not np.isfinite(flow_cny_bn):
            return 0.0
        return float(np.tanh(flow_cny_bn / 50.0))  # 50 亿 ≈ tanh(1) ≈ 0.76

    def _direction_policy(dr007: float, credit_spread_aa: float) -> float:
        if dr007 is None or not np.isfinite(dr007):
            dr = 2.5
        else:
            dr = dr007
        if credit_spread_aa is None or not np.isfinite(credit_spread_aa):
            cs = 1.0
        else:
            cs = credit_spread_aa
        # Both <1 = aggressive easing; both >3 = tight stance
        rate_signal = float(np.tanh((2.5 - dr) / 1.5))  # rates below 2.5 -> positive
        credit_signal = float(np.tanh((1.0 - cs) / 0.5))  # spread below 1 -> positive
        return _clip_signed(0.5 * (rate_signal + credit_signal))

    out_rows: list[dict[str, Any]] = []
    for _, row in bond_flows.iterrows():
        publish_time = _coerce_ts(row.get("trade_date"))
        available_at = _coerce_ts(row.get("available_at")) or publish_time
        if publish_time is None or available_at is None:
            continue
        present = sum(
            1 for c in numeric_cols if c in row.index and pd.notna(row.get(c))
        )
        confidence = _clip01(present / len(numeric_cols))
        flow_score = _direction_capital(float(row.get("bond_fund_flow") or 0.0))
        policy_score = _direction_policy(
            float(row.get("dr007") or float("nan")),
            float(row.get("credit_spread_aa") or float("nan")),
        )
        claims: list[str] = []
        for c in numeric_cols:
            if c in row.index and pd.notna(row.get(c)):
                claims.append(f"{c}={float(row[c]):.4f}")

        evidence_id = _stable_id(
            "bond",
            row.get("source", ""),
            publish_time.isoformat(),
        )
        out_rows.append(
            {
                "evidence_id": evidence_id,
                "source_name": str(row.get("source") or ""),
                "source_type": "bond",
                "url_or_file_id": None,
                "publish_time": publish_time,
                "crawl_time": _coerce_ts(row.get("fetched_at")),
                "available_at": available_at,
                "entity_type": "bond_market_snapshot",
                "entities": ["CN_TREASURY", "DR007", "AA_CREDIT"],
                "raw_text_hash": None,
                "extracted_claims": claims,
                "sentiment_score": 0.0,
                "policy_direction_score": policy_score,
                "capital_flow_direction_score": flow_score,
                "confidence": confidence,
                "contradiction_score": 0.0,
                "lag_window_candidates": [1, 5, 20, 60],
                "audit_trace": {
                    "adapter": "bond_flows_to_evidence",
                    "source_version": str(
                        row.get("source_version") or source_version or "unknown"
                    ),
                },
            }
        )
    return pd.DataFrame(out_rows, columns=list(CANONICAL_EVIDENCE_COLUMNS))


# ---------------------------------------------------------------------------
# Adapters: broker_reports -> canonical
# ---------------------------------------------------------------------------

def broker_reports_to_evidence(
    broker_reports: pd.DataFrame,
    *,
    source_version: str | None = None,
) -> pd.DataFrame:
    """Convert ``silver/broker_reports.parquet`` rows into canonical evidence.

    ``rating`` maps to ``sentiment_score``; ``broker_credibility``
    propagates as ``confidence``; ``rating_change`` drives an
    additional confidence bump for upgrades / haircut for downgrades
    so the final ``confidence`` reflects both who said it and how
    surprising the change is.
    """
    if broker_reports is None or broker_reports.empty:
        return pd.DataFrame(columns=list(CANONICAL_EVIDENCE_COLUMNS))

    rating_score_map = {
        "buy": 1.0, "overweight": 0.5, "hold": 0.0,
        "underweight": -0.5, "sell": -1.0, "n/a": 0.0,
    }
    change_bump_map = {
        "upgrade": 0.10,
        "initiate": 0.05,
        "maintain": 0.0,
        "downgrade": -0.10,
        "drop": -0.15,
        "n/a": 0.0,
    }

    out_rows: list[dict[str, Any]] = []
    for _, row in broker_reports.iterrows():
        publish_time = _coerce_ts(row.get("announced_at"))
        available_at = _coerce_ts(row.get("available_at")) or publish_time
        if publish_time is None or available_at is None:
            continue
        symbol = str(row.get("symbol") or "")
        rating = str(row.get("rating") or "n/a").lower()
        rating_change = str(row.get("rating_change") or "maintain").lower()
        sentiment = float(rating_score_map.get(rating, 0.0))
        credibility = float(row.get("broker_credibility") or 0.0)
        bump = float(change_bump_map.get(rating_change, 0.0))
        confidence = _clip01(credibility + bump)
        tp_pct = row.get("target_price_pct_change")
        try:
            tp_pct_f = float(tp_pct) if tp_pct is not None and pd.notna(tp_pct) else 0.0
        except (TypeError, ValueError):
            tp_pct_f = 0.0

        claims: list[str] = [
            f"broker={row.get('broker', '')}",
            f"rating={rating}",
            f"rating_change={rating_change}",
        ]
        summary = str(row.get("summary") or "").strip()
        if summary:
            claims.append(summary)
        if np.isfinite(tp_pct_f) and tp_pct_f != 0.0:
            claims.append(f"target_price_pct_change={tp_pct_f:+.4f}")

        evidence_id = _stable_id(
            "broker",
            row.get("broker", ""),
            symbol,
            publish_time.isoformat(),
        )
        out_rows.append(
            {
                "evidence_id": evidence_id,
                "source_name": str(row.get("source") or ""),
                "source_type": "broker_view",
                "url_or_file_id": None,
                "publish_time": publish_time,
                "crawl_time": _coerce_ts(row.get("fetched_at")),
                "available_at": available_at,
                "entity_type": "broker_research_note",
                "entities": [symbol] if symbol else [],
                "raw_text_hash": None,
                "extracted_claims": claims,
                "sentiment_score": _clip_signed(sentiment),
                "policy_direction_score": 0.0,
                "capital_flow_direction_score": 0.0,
                "confidence": confidence,
                "contradiction_score": 0.0,
                "lag_window_candidates": [5, 20, 60],
                "audit_trace": {
                    "adapter": "broker_reports_to_evidence",
                    "broker": str(row.get("broker") or ""),
                    "broker_tier": str(row.get("broker_tier") or ""),
                    "source_version": str(
                        row.get("source_version") or source_version or "unknown"
                    ),
                },
            }
        )
    return pd.DataFrame(out_rows, columns=list(CANONICAL_EVIDENCE_COLUMNS))


# ---------------------------------------------------------------------------
# Adapters: state_team_inference -> canonical
# ---------------------------------------------------------------------------

def state_team_events_to_evidence(
    state_team_events: pd.DataFrame,
    *,
    source_version: str | None = None,
) -> pd.DataFrame:
    """Convert ``silver/state_team_inference.parquet`` rows into canonical.

    All such rows are inferred (not observed), so the canonical record:

    * sets ``source_type = "state_team_inference"``
    * stores the original ``evidence_label`` ("inferred") in
      ``audit_trace`` so consumers can surface it
    * encodes ``evidence_strength`` directly as ``capital_flow_direction_score``
      (positive — these detectors only fire on inferred buying)
    * uses ``evidence_strength`` as confidence too (already in [0, 1])
    """
    if state_team_events is None or state_team_events.empty:
        return pd.DataFrame(columns=list(CANONICAL_EVIDENCE_COLUMNS))

    out_rows: list[dict[str, Any]] = []
    for _, row in state_team_events.iterrows():
        publish_time = _coerce_ts(row.get("trade_date"))
        available_at = _coerce_ts(row.get("available_at")) or publish_time
        if publish_time is None or available_at is None:
            continue
        strength = float(row.get("evidence_strength") or 0.0)
        scope = str(row.get("scope") or "")
        scope_value = str(row.get("scope_value") or "")
        entities = [scope_value] if scope_value else []
        if scope == "index_wide" and scope_value:
            entities = [f"index:{scope_value}"]
        elif scope == "sector" and scope_value:
            entities = [f"sector:{scope_value}"]
        evidence_type = str(row.get("evidence_type") or "state_team_event")
        claims: list[str] = [
            f"evidence_type={evidence_type}",
            f"scope={scope}",
            f"scope_value={scope_value}",
            f"strength={strength:.4f}",
        ]
        description = str(row.get("description") or "").strip()
        if description:
            claims.append(description)

        evidence_id = str(row.get("event_id") or "")
        if evidence_id:
            evidence_id = f"state_team_{evidence_id}"
        else:
            evidence_id = _stable_id(
                "state_team", evidence_type, scope_value, publish_time.isoformat()
            )
        out_rows.append(
            {
                "evidence_id": evidence_id,
                "source_name": str(row.get("source") or ""),
                "source_type": "state_team_inference",
                "url_or_file_id": None,
                "publish_time": publish_time,
                "crawl_time": _coerce_ts(row.get("fetched_at")),
                "available_at": available_at,
                "entity_type": evidence_type,
                "entities": entities,
                "raw_text_hash": None,
                "extracted_claims": claims,
                "sentiment_score": 0.0,
                "policy_direction_score": 0.0,
                "capital_flow_direction_score": _clip_signed(strength),
                "confidence": _clip01(strength),
                "contradiction_score": 0.0,
                "lag_window_candidates": [1, 5, 20, 60, 120],
                "audit_trace": {
                    "adapter": "state_team_events_to_evidence",
                    "evidence_label": str(row.get("evidence_label") or "inferred"),
                    "source_version": str(
                        row.get("source_version") or source_version or "unknown"
                    ),
                },
            }
        )
    return pd.DataFrame(out_rows, columns=list(CANONICAL_EVIDENCE_COLUMNS))


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def to_canonical_evidence_frame(
    *,
    policy_events: pd.DataFrame | None = None,
    bond_flows: pd.DataFrame | None = None,
    broker_reports: pd.DataFrame | None = None,
    state_team_events: pd.DataFrame | None = None,
    source_version: str | None = None,
) -> pd.DataFrame:
    """Adapt every supplied silver frame and concat into one canonical frame.

    The output is sorted by ``available_at`` and dedup'd on
    ``evidence_id`` (first-write-wins). Callers must NOT mutate the
    result in place — downstream PIT joins depend on a stable schema.
    """
    parts: list[pd.DataFrame] = []
    if policy_events is not None and not policy_events.empty:
        parts.append(policy_events_to_evidence(policy_events, source_version=source_version))
    if bond_flows is not None and not bond_flows.empty:
        parts.append(bond_flows_to_evidence(bond_flows, source_version=source_version))
    if broker_reports is not None and not broker_reports.empty:
        parts.append(broker_reports_to_evidence(broker_reports, source_version=source_version))
    if state_team_events is not None and not state_team_events.empty:
        parts.append(
            state_team_events_to_evidence(state_team_events, source_version=source_version)
        )
    if not parts:
        return pd.DataFrame(columns=list(CANONICAL_EVIDENCE_COLUMNS))
    combined = pd.concat(parts, ignore_index=True, sort=False)
    if "evidence_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["evidence_id"], keep="first")
    return combined.sort_values("available_at").reset_index(drop=True)


# ---------------------------------------------------------------------------
# PIT lint
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PITLintReport:
    n_rows: int
    n_missing_available_at: int
    n_missing_publish_time: int
    n_available_before_publish: int
    n_future_publish: int
    by_source_type: dict[str, int]
    sample_violations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.n_missing_available_at == 0
            and self.n_available_before_publish == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_missing_available_at": self.n_missing_available_at,
            "n_missing_publish_time": self.n_missing_publish_time,
            "n_available_before_publish": self.n_available_before_publish,
            "n_future_publish": self.n_future_publish,
            "by_source_type": dict(self.by_source_type),
            "sample_violations": list(self.sample_violations),
            "passed": self.passed,
        }


def validate_pit_safety(
    canonical: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
    max_samples: int = 5,
) -> PITLintReport:
    """Lint a canonical evidence frame for PIT correctness.

    Hard rules:

    * Every row must carry a non-null ``available_at``.
    * Every row's ``available_at`` must be ``>= publish_time`` whenever
      ``publish_time`` is set. The two timestamps may be equal (e.g.
      manual_local_import sources where crawl == publish).

    Soft signal (reported, not failed):

    * ``publish_time`` in the future of ``as_of`` (caller-supplied).
    """
    if canonical is None or canonical.empty:
        return PITLintReport(
            n_rows=0,
            n_missing_available_at=0,
            n_missing_publish_time=0,
            n_available_before_publish=0,
            n_future_publish=0,
            by_source_type={},
        )
    frame = canonical.copy()
    frame["available_at"] = pd.to_datetime(frame["available_at"], errors="coerce")
    frame["publish_time"] = pd.to_datetime(frame["publish_time"], errors="coerce")

    miss_avail = frame["available_at"].isna()
    miss_pub = frame["publish_time"].isna()
    inverted = (~miss_avail) & (~miss_pub) & (frame["available_at"] < frame["publish_time"])
    future_pub = pd.Series(False, index=frame.index)
    if as_of is not None:
        future_pub = (~miss_pub) & (frame["publish_time"] > pd.Timestamp(as_of))

    by_source: dict[str, int] = (
        frame.groupby("source_type", dropna=False).size().to_dict()
    )

    sample_rows: list[dict[str, Any]] = []
    bad = frame[miss_avail | inverted]
    for _, r in bad.head(max_samples).iterrows():
        sample_rows.append(
            {
                "evidence_id": str(r.get("evidence_id")),
                "source_type": str(r.get("source_type")),
                "publish_time": str(r.get("publish_time")),
                "available_at": str(r.get("available_at")),
                "reason": (
                    "missing_available_at" if pd.isna(r.get("available_at")) else "available_at_before_publish"
                ),
            }
        )

    return PITLintReport(
        n_rows=int(len(frame)),
        n_missing_available_at=int(miss_avail.sum()),
        n_missing_publish_time=int(miss_pub.sum()),
        n_available_before_publish=int(inverted.sum()),
        n_future_publish=int(future_pub.sum()),
        by_source_type={str(k): int(v) for k, v in by_source.items()},
        sample_violations=sample_rows,
    )
